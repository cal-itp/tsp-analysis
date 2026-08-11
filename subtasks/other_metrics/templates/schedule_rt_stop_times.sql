with stop_times_valid_on_date as (
    select stop_id, trip_id, timepoint, stop_sequence from mart_gtfs.dim_stop_times
    where 
        _feed_valid_from = '{{ FEED_VALID_FROM }}'
        and feed_key = '{{ FEED_KEY }}'
),
valid_trips as (
    select trip_id, shape_id, direction_id from mart_gtfs.fct_scheduled_trips
    where service_date = '{{ TARGET_DATE }}' 
        and feed_key = '{{ FEED_KEY }}'
        and shape_id in ('{{ SHAPE_IDS | join("', '") }}')
),
-- determine timepoint status
stop_times_summary_on_route as (
    select 
        stop_times_valid_on_date.stop_id, 
        valid_trips.shape_id, 
        sum(stop_times_valid_on_date.timepoint) as n_timepoint, 
        count(*) as n_stop_times
    from stop_times_valid_on_date
    inner join valid_trips
    using(trip_id)
    group by stop_id, shape_id
),
-- determine stop sequence
stop_sequence_values as (
    select distinct stop_id, stop_sequence
    from stop_times_valid_on_date
    inner join valid_trips
    using(trip_id)
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
        stops.pt_geom,

    from mart_gtfs.fct_daily_scheduled_stops as stops
    inner join stop_times_summary_on_route
    using(stop_id)
    inner join stop_sequence_values
    using(stop_id)
    where stops.service_date = '{{ TARGET_DATE }}'
        and stops.feed_key = '{{ FEED_KEY }}'
        and stop_times_summary_on_route.n_timepoint = stop_times_summary_on_route.n_stop_times
),
rt_stop_times as (
    select 
        trip_id,
        trip_key,
        stop_id,
        stop_sequence,
        actual_arrival_pacific,
        actual_departure_pacific,
        n_predictions,
    from mart_gtfs.fct_stop_time_metrics
    where service_date = '{{ TARGET_DATE }}'
    and schedule_base64_url = '{{ SCHEDULE_BASE64_URL }}'
)

select
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
inner join valid_trips
using(trip_id)
inner join timepoints_on_route
using(stop_sequence, shape_id)
order by trip_key, stop_sequence asc
