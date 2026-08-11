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
stop_times_valid_on_date as (
    select feed_key, stop_id, trip_id, timepoint, stop_sequence from mart_gtfs.dim_stop_times
    where
        _feed_valid_from in ('{{ FEED_VALID_FROMS | join("', '") }}')
        and feed_key in ('{{ FEED_KEYS | join("', '") }}')
),
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
-- determine stop sequence
stop_sequence_values as (
    select distinct stop_id, stop_sequence
    from stop_times_valid_on_date
    inner join valid_trips
    using(trip_id, feed_key)
),
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
)

select
    rt_stop_times.service_date,
    rt_stop_times.trip_id,
    rt_stop_times.trip_key,
    rt_stop_times.stop_id,
    valid_trips.shape_id,
    rt_stop_times.stop_sequence,
    rt_stop_times.actual_arrival_pacific,
    rt_stop_times.actual_departure_pacific,
    rt_stop_times.n_predictions,
    timepoints_on_route.pt_geom
from rt_stop_times
-- trip_id is only unique within a service date
inner join valid_trips
using(service_date, trip_id)
inner join timepoints_on_route
using(stop_sequence, shape_id)
order by service_date, trip_key, stop_sequence asc
