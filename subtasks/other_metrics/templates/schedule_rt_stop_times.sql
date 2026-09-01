-- gets schedule + rt stop times for timepoints

-- the feed in effect can change part way through a multi-date window, so pair
-- each service date with the schedule feed that was actually active on it
with feed_dates as (
    select * from unnest([
        {%- for entry in FEED_DATES %}
        struct(
            date '{{ entry.service_date }}' as service_date,
            '{{ entry.feed_key }}' as feed_key,
            '{{ entry.base64_url }}' as schedule_base64_url
        ){{ "," if not loop.last }}
        {%- endfor %}
    ])
),
-- get entries from dim_stop_times. we need this because it's the only table that exposes timepoint
stop_times_valid_on_date as (
    select feed_key, stop_id, trip_id, timepoint, stop_sequence from mart_gtfs.dim_stop_times
    where
        _feed_valid_from in ('{{ FEED_VALID_FROMS | join("', '") }}') -- feed_valid_from used to partition dim_stop_times
        and feed_key in ('{{ FEED_KEYS | join("', '") }}')
),
-- get all trips on the correct date
valid_trips as (
    select service_date, trip_id, feed_key, shape_id, direction_id
    from mart_gtfs.fct_scheduled_trips
    -- the literal filters keep the scan pruned, the join keeps each date paired
    -- with only the feed that was active on it
    inner join feed_dates
    using(service_date, feed_key)
    where service_date in ('{{ TARGET_DATES | join("', '") }}')
        and feed_key in ('{{ FEED_KEYS | join("', '") }}')
        and shape_id in ('{{ SHAPE_IDS | join("', '") }}')
),
-- determine timepoint status
-- a stop counts as a timepoint only if it is one on every trip on every selected date
-- todo: consider a better way of determining this / get timepoint locations from agencies instead
stop_times_summary_on_route as (
    select
        stop_times_valid_on_date.stop_id,
        valid_trips.shape_id,
        sum(stop_times_valid_on_date.timepoint) as n_timepoint,
        count(*) as n_stop_times
    from stop_times_valid_on_date
    inner join valid_trips
    using(trip_id, feed_key)
    group by stop_id, shape_id
),
-- get schedule stop sequences
stop_sequence_values as (
    select distinct stop_id, stop_sequence
    from stop_times_valid_on_date
    inner join valid_trips
    using(trip_id, feed_key)
),
-- join stop time grain data to stops, to get geometry and aggregate
-- TODO: determine whether stop_sequence is guaranteed to match the rt stop sequence
--   TODO: so we can use it as a merge key (as assumed), or if we need to use stop_id instead
timepoints_on_route as (
    select
        stops.stop_id,
        stop_times_summary_on_route.n_timepoint,
        stop_times_summary_on_route.n_stop_times,
        stop_times_summary_on_route.shape_id,
        stop_sequence_values.stop_sequence,
        -- stops repeat once per selected service date, so collapse to one row per stop
        any_value(stops.pt_geom) as pt_geom,

    from mart_gtfs.fct_daily_scheduled_stops as stops
    inner join stop_times_summary_on_route
    using(stop_id)
    inner join stop_sequence_values
    using(stop_id)
    where stops.service_date in ('{{ TARGET_DATES | join("', '") }}')
        and stops.feed_key in ('{{ FEED_KEYS | join("', '") }}')
        and stop_times_summary_on_route.n_timepoint = stop_times_summary_on_route.n_stop_times
    group by 1, 2, 3, 4, 5
),
-- the published arrival at those same stops. dim_stop_arrivals is grained by
-- feed version rather than service date, so trip_id is what ties it to the
-- realtime side
--
-- the arrival, not the departure: on time performance asks when the vehicle
-- reached the stop, so a bus that pulled in on time and then stood too long is
-- on time. the wait metrics measure departure to departure instead, and the
-- dwell between the two is what separates the answers
stop_arrivals as (
    select
        feed_key,
        trip_id,
        stop_id,
        stop_sequence,
        arrival_sec,
    from mart_gtfs.dim_stop_arrivals
    where
        -- _feed_valid_from partitions dim_stop_arrivals
        _feed_valid_from in ('{{ FEED_VALID_FROMS | join("', '") }}')
        and feed_key in ('{{ FEED_KEYS | join("', '") }}')
),
-- get rt data
rt_stop_times as (
    select
        service_date,
        trip_id,
        trip_key,
        stop_id,
        stop_sequence,
        actual_arrival_pacific,
        actual_departure_pacific,
        n_predictions,
    from mart_gtfs.fct_stop_time_metrics
    inner join feed_dates
    using(service_date, schedule_base64_url)
    where service_date in ('{{ TARGET_DATES | join("', '") }}')
    and schedule_base64_url in ('{{ SCHEDULE_BASE64_URLS | join("', '") }}')
),
-- get rt stop arrivals, with the arrival the schedule published for the same
-- trip at the same stop alongside
timepoint_stop_times as (
    select
        rt_stop_times.service_date,
        rt_stop_times.trip_id,
        rt_stop_times.trip_key,
        rt_stop_times.stop_id,
        valid_trips.feed_key,
        valid_trips.shape_id,
        rt_stop_times.stop_sequence,
        rt_stop_times.actual_arrival_pacific,
        rt_stop_times.actual_departure_pacific,
        rt_stop_times.n_predictions,
        stop_arrivals.arrival_sec as scheduled_arrival_sec,
        -- GTFS counts a stop time from midnight on the trip's own service date
        -- and keeps counting past 24:00, so a trip listed under the 17th but
        -- calling at 00:15 on the 18th publishes 87300 rather than 900. Adding
        -- the seconds to that midnight puts it back on the wall clock, which is
        -- where the realtime side already is -- so the two are subtractable and
        -- the day part comes out right either side of the boundary
        datetime_add(
            datetime(rt_stop_times.service_date),
            interval cast(stop_arrivals.arrival_sec as int64) second
        ) as scheduled_arrival_pacific,
        timepoints_on_route.pt_geom
    from rt_stop_times
    -- trip_id is only unique within a service date
    inner join valid_trips
    using(service_date, trip_id)
    inner join timepoints_on_route
    using(stop_sequence, shape_id)
    -- left, so a trip the schedule does not cover keeps its realtime stop times
    -- and reports a null published arrival rather than dropping out -- the
    -- travel time metric reads this same frame and needs no schedule at all
    left join stop_arrivals
        on stop_arrivals.feed_key = valid_trips.feed_key
        and stop_arrivals.trip_id = rt_stop_times.trip_id
        and stop_arrivals.stop_id = rt_stop_times.stop_id
        and stop_arrivals.stop_sequence = rt_stop_times.stop_sequence
)
select
    *,
    -- signed so that late is positive. null wherever the schedule had nothing
    -- to compare against, which is what on time performance drops
    datetime_diff(actual_arrival_pacific, scheduled_arrival_pacific, second)
        as arrival_deviation_seconds,
from timepoint_stop_times
order by service_date, trip_key, stop_sequence asc
