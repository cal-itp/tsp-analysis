-- validation: how far dim_stop_arrivals and fct_stop_time_metrics line up
--
-- on time performance compares one realtime arrival to the scheduled arrival for
-- the same trip at the same stop, so it is only computable where a trip_id
-- appears on both sides. This counts, per service date, how many trip_ids do --
-- and then whether the two tables also agree about when one of those trips was
-- actually somewhere, which set membership alone would not tell us.
--
-- Feed wide rather than route scoped, deliberately. fct_stop_time_metrics
-- carries no route, so a realtime trip can only be attributed to a route through
-- the schedule -- which is exactly what a trip missing from the schedule has not
-- got. Filtering to one route would drop every unmatched realtime trip before it
-- could be counted, forcing n_trip_ids_stop_time_metrics_only to zero by
-- construction. Feed wide keeps the three counts a real partition of the union
-- of the two tables.
--
-- Standalone: paste into the BigQuery console and run. It is a script rather
-- than a single statement so the partition filters below are variables, which
-- BigQuery can prune on. A subquery in their place would not prune, and an
-- unpruned dim_stop_arrivals is the entire table -- dry run before running for
-- real.

-- ---------------------------------------------------------------- edit these
-- the reporting window. Same dates as headway_expected_wait.ipynb
declare target_dates array<date> default [
    date '2026-06-15',
    date '2026-06-16',
    date '2026-06-17',
    date '2026-06-18',
    date '2026-06-19'
];
declare target_schedule_feed string default 'Bay Area 511 SamTrans Schedule';
-- how far apart the scheduled and actual departure may be and still count as
-- the two tables agreeing about the same trip
declare tolerance_minutes int64 default 30;
-- how far along the trip to compare. 5 is the fifth stop the trip calls at, by
-- position rather than by stop_sequence = 5: GTFS only requires stop_sequence to
-- increase, so a feed numbering its stops 10, 20, 30 has no stop_sequence 5 at
-- all and would silently contribute nothing
declare compared_stop_position int64 default 5;
-- ---------------------------------------------------------------------------

-- the feeds active over the window, resolved up front so the scan filters are
-- constants by the time the query below runs
declare feed_keys array<string>;
declare feed_valid_froms array<timestamp>;
declare schedule_base64_urls array<string>;

set (feed_keys, feed_valid_froms, schedule_base64_urls) = (
    select as struct
        array_agg(distinct feeds.feed_key),
        array_agg(distinct schedule_feeds._valid_from),
        array_agg(distinct schedule_feeds.base64_url),
    from mart_gtfs.fct_daily_schedule_feeds as feeds
    inner join mart_gtfs.dim_schedule_feeds as schedule_feeds
        on feeds.feed_key = schedule_feeds.key
    -- fct_daily_schedule_feeds.date is a DATETIME, and the service_date columns
    -- it is matched against everywhere else are DATEs. The table is not
    -- partitioned, so casting it costs no pruning
    where date(feeds.date) in unnest(target_dates)
        and feeds.gtfs_dataset_name = target_schedule_feed
);

-- the feed in effect can change part way through a multi-date window, so pair
-- each service date with the schedule feed that was actually active on it
with feed_dates as (
    select
        date(feeds.date) as service_date,
        feeds.feed_key,
        schedule_feeds.base64_url as schedule_base64_url,
    from mart_gtfs.fct_daily_schedule_feeds as feeds
    inner join mart_gtfs.dim_schedule_feeds as schedule_feeds
        on feeds.feed_key = schedule_feeds.key
    where date(feeds.date) in unnest(target_dates)
        and feeds.gtfs_dataset_name = target_schedule_feed
),
-- the trips actually in service on each date. dim_stop_arrivals is grained by
-- feed version, so it holds every trip the feed defines whether or not it runs
-- on a given date -- without this the weekday/weekend split alone would report
-- most of the feed as schedule only, which says nothing about alignment
scheduled_trips as (
    select service_date, feed_key, trip_id
    from mart_gtfs.fct_scheduled_trips
    -- the variable filters keep the scan pruned, the join keeps each date paired
    -- with only the feed that was active on it
    inner join feed_dates
    using(service_date, feed_key)
    where service_date in unnest(target_dates)
        and feed_key in unnest(feed_keys)
),
-- the published schedule. no service date to filter on: scheduled_trips above is
-- what ties a feed version's trips to the dates they ran
stop_arrivals as (
    select
        feed_key,
        trip_id,
        stop_id,
        stop_sequence,
        departure_sec,
    from mart_gtfs.dim_stop_arrivals
    where
        -- _feed_valid_from partitions dim_stop_arrivals
        _feed_valid_from in unnest(feed_valid_froms)
        and feed_key in unnest(feed_keys)
),
-- the realtime side, unfiltered by trip: a trip the schedule never mentions is
-- the thing this query exists to count, so it has to survive to the join below
rt_stop_times as (
    select
        service_date,
        trip_id,
        stop_id,
        stop_sequence,
        actual_departure_pacific,
    from mart_gtfs.fct_stop_time_metrics
    inner join feed_dates
    using(service_date, schedule_base64_url)
    where service_date in unnest(target_dates)
        and schedule_base64_url in unnest(schedule_base64_urls)
),
-- one row per trip_id per service date on each side, so the join below counts
-- trips rather than stop times
stop_arrivals_trip_ids as (
    select distinct scheduled_trips.service_date, scheduled_trips.trip_id
    from scheduled_trips
    inner join stop_arrivals
    using(feed_key, trip_id)
),
rt_trip_ids as (
    select distinct service_date, trip_id
    from rt_stop_times
),
-- full outer, so a trip_id on either side alone is kept and labelled rather
-- than dropped
trip_coverage as (
    select
        coalesce(schedule.service_date, rt.service_date) as service_date,
        schedule.trip_id is not null as in_stop_arrivals,
        rt.trip_id is not null as in_stop_time_metrics,
    from stop_arrivals_trip_ids as schedule
    full outer join rt_trip_ids as rt
        on schedule.service_date = rt.service_date
        and schedule.trip_id = rt.trip_id
),
-- the stop to compare on, taken by position along the trip
schedule_compared_stop as (
    select feed_key, trip_id, stop_id, stop_sequence, departure_sec
    from (
        select
            *,
            row_number() over (
                partition by feed_key, trip_id
                order by stop_sequence asc
            ) as stop_position,
        from stop_arrivals
    )
    where stop_position = compared_stop_position
),
-- that stop matched to the realtime row for the same stop, so a disagreement
-- about which stop comes fifth cannot masquerade as a difference in time. Same
-- join keys the metric queries in ../templates/ use
compared_stop as (
    select
        rt_stop_times.service_date,
        rt_stop_times.trip_id,
        -- departure_sec counts from midnight on the trip's own service date and
        -- can run past 86400 for a trip crossing midnight; datetime_add carries
        -- it into the next day. Both sides are local wall clock time
        abs(datetime_diff(
            rt_stop_times.actual_departure_pacific,
            datetime_add(
                datetime(rt_stop_times.service_date),
                interval cast(schedule_compared_stop.departure_sec as int64) second
            ),
            second
        )) as departure_difference_seconds,
    from schedule_compared_stop
    -- back to a service date grain: a feed version's trips run on several dates
    inner join scheduled_trips
    using(feed_key, trip_id)
    inner join rt_stop_times
        on rt_stop_times.service_date = scheduled_trips.service_date
        and rt_stop_times.trip_id = schedule_compared_stop.trip_id
        and rt_stop_times.stop_id = schedule_compared_stop.stop_id
        and rt_stop_times.stop_sequence = schedule_compared_stop.stop_sequence
    -- a null on either side is a missing observation rather than a
    -- disagreement, and would otherwise sit in the denominator with no way of
    -- ever reaching the within-tolerance count
    where schedule_compared_stop.departure_sec is not null
        and rt_stop_times.actual_departure_pacific is not null
),
coverage_counts as (
    select
        service_date,
        countif(in_stop_time_metrics and not in_stop_arrivals)
            as n_trip_ids_stop_time_metrics_only,
        countif(in_stop_arrivals and not in_stop_time_metrics)
            as n_trip_ids_stop_arrivals_only,
        countif(in_stop_arrivals and in_stop_time_metrics)
            as n_trip_ids_both,
    from trip_coverage
    group by service_date
),
compared_stop_counts as (
    select
        service_date,
        count(distinct trip_id) as n_trip_ids_compared_stop_compared,
        count(distinct if(
            departure_difference_seconds <= tolerance_minutes * 60,
            trip_id,
            null
        )) as n_trip_ids_compared_stop_within_tolerance,
    from compared_stop
    group by service_date
)
select
    service_date,
    -- reported realtime but with no scheduled stop times on that date: either
    -- absent from the feed entirely, or running on a date it is not scheduled for
    n_trip_ids_stop_time_metrics_only,
    -- in service that date with published stop times, but never reported
    n_trip_ids_stop_arrivals_only,
    -- on both sides. These three partition the union
    n_trip_ids_both,
    -- of those, the ones the comparison stop is populated on in both tables, and
    -- then the ones the two tables put within tolerance of each other. The gap
    -- between the two is trips both tables know about but disagree about, which
    -- the counts above cannot show
    --
    -- a date nothing matched on drops out of the join entirely; that is zero
    -- compared, not an unknown
    coalesce(n_trip_ids_compared_stop_compared, 0)
        as n_trip_ids_compared_stop_compared,
    coalesce(n_trip_ids_compared_stop_within_tolerance, 0)
        as n_trip_ids_compared_stop_within_tolerance,
from coverage_counts
left join compared_stop_counts
using(service_date)
order by service_date
