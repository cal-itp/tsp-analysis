-- gets scheduled and realtime headways between consecutive trips, at every stop

-- a headway is the gap between vehicles at a stop: the time from one pulling out
-- to the next one pulling out. Both sides measure it departure to departure --
-- the realtime one between the actual departures, the scheduled one between the
-- departures dim_stop_arrivals published for those same two trips.
--
-- measuring to the next arrival instead would fold the second vehicle's dwell
-- into the gap, so the same 15 minute service would report a different headway
-- at a stop where buses stand longer. Nothing about the service differs there.

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
-- get all trips on the correct date
valid_trips as (
    select service_date, trip_id, feed_key, shape_id
    from mart_gtfs.fct_scheduled_trips
    -- the literal filters keep the scan pruned, the join keeps each date paired
    -- with only the feed that was active on it
    inner join feed_dates
    using(service_date, feed_key)
    where service_date in ('{{ TARGET_DATES | join("', '") }}')
        and feed_key in ('{{ FEED_KEYS | join("', '") }}')
        and shape_id in ('{{ SHAPE_IDS | join("', '") }}')
),
-- get rt data. unlike the trip duration metric this is every stop, not just the
-- timepoints, so it needs nothing from dim_stop_times
rt_stop_times as (
    select
        service_date,
        trip_id,
        trip_key,
        stop_id,
        stop_sequence,
        actual_departure_pacific,
    from mart_gtfs.fct_stop_time_metrics
    inner join feed_dates
    using(service_date, schedule_base64_url)
    where service_date in ('{{ TARGET_DATES | join("', '") }}')
    and schedule_base64_url in ('{{ SCHEDULE_BASE64_URLS | join("', '") }}')
),
-- the published schedule for the same stops. dim_stop_arrivals is grained by
-- feed version rather than service date, so trip_id is what ties it to the
-- realtime side
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
        _feed_valid_from in ('{{ FEED_VALID_FROMS | join("', '") }}')
        and feed_key in ('{{ FEED_KEYS | join("', '") }}')
),
-- stop geometry, so the metric can be mapped. a stop repeats once per selected
-- service date, so collapse it to one row -- the same shape schedule_rt_stop_times
-- uses for its pt_geom
stop_geometry as (
    select
        stop_id,
        any_value(pt_geom) as pt_geom,
    from mart_gtfs.fct_daily_scheduled_stops
    where service_date in ('{{ TARGET_DATES | join("', '") }}')
        and feed_key in ('{{ FEED_KEYS | join("', '") }}')
    group by stop_id
),
-- one row per trip and stop, narrowed to the target service patterns
trip_stops as (
    select
        service_date,
        rt_stop_times.trip_key,
        rt_stop_times.stop_id,
        rt_stop_times.stop_sequence,
        rt_stop_times.actual_departure_pacific,
        valid_trips.shape_id,
        stop_arrivals.departure_sec as scheduled_departure_sec,
    from rt_stop_times
    -- trip_id is only unique within a service date
    inner join valid_trips
    using(service_date, trip_id)
    -- left, so a trip the schedule does not cover keeps its realtime headway
    -- and reports a null scheduled one rather than dropping out
    left join stop_arrivals
        on stop_arrivals.feed_key = valid_trips.feed_key
        and stop_arrivals.trip_id = rt_stop_times.trip_id
        and stop_arrivals.stop_id = rt_stop_times.stop_id
        and stop_arrivals.stop_sequence = rt_stop_times.stop_sequence
),
-- pair each trip with the next one to leave the stop. ordering on the actual
-- departure rather than the scheduled start means an overtaking vehicle is
-- paired the way it really ran -- and it is the same column the gap is measured
-- from, so a headway can never come out negative. trip_key breaks ties so the
-- pairing is deterministic
--
-- partition on service_date to handle overnight changes
headways as (
    select
        service_date,
        trip_key as first_trip_key,
        lead(trip_key) over (
            partition by service_date, shape_id, stop_sequence, stop_id
            order by actual_departure_pacific asc, trip_key asc
        ) as second_trip_key,
        stop_sequence,
        stop_id,
        shape_id,
        actual_departure_pacific as first_actual_departure,
        lead(actual_departure_pacific) over (
            partition by service_date, shape_id, stop_sequence, stop_id
            order by actual_departure_pacific asc, trip_key asc
        ) as second_actual_departure,
        scheduled_departure_sec as first_scheduled_departure_sec,
        lead(scheduled_departure_sec) over (
            partition by service_date, shape_id, stop_sequence, stop_id
            order by actual_departure_pacific asc, trip_key asc
        ) as second_scheduled_departure_sec,
    from trip_stops
)
-- the last trip to call at a stop on a service date starts no headway
select
    service_date,
    first_trip_key,
    second_trip_key,
    stop_sequence,
    stop_id,
    shape_id,
    first_actual_departure,
    second_actual_departure,
    -- the actuals are DATETIME, already in the feed's local time
    datetime_diff(second_actual_departure, first_actual_departure, second)
        as rt_headway_seconds,
    first_scheduled_departure_sec,
    second_scheduled_departure_sec,
    -- both scheduled columns are seconds since midnight on the same service
    -- date, which the partition above guarantees, so a plain subtraction
    second_scheduled_departure_sec - first_scheduled_departure_sec
        as scheduled_headway_seconds,
    stop_geometry.pt_geom,
from headways
-- left, so a stop missing from the daily stops table keeps its headway and
-- reports a null geometry rather than dropping out of the metric entirely
left join stop_geometry
using(stop_id)
where second_trip_key is not null
order by service_date, shape_id, stop_sequence, first_actual_departure asc
