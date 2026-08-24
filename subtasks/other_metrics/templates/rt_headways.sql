-- gets realtime headways between consecutive trips, at every stop

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
        actual_arrival_pacific,
        actual_departure_pacific,
    from mart_gtfs.fct_stop_time_metrics
    inner join feed_dates
    using(service_date, schedule_base64_url)
    where service_date in ('{{ TARGET_DATES | join("', '") }}')
    and schedule_base64_url in ('{{ SCHEDULE_BASE64_URLS | join("', '") }}')
),
-- one row per trip and stop, narrowed to the target service patterns
trip_stops as (
    select
        rt_stop_times.trip_key,
        rt_stop_times.stop_id,
        rt_stop_times.stop_sequence,
        rt_stop_times.actual_arrival_pacific,
        rt_stop_times.actual_departure_pacific,
        valid_trips.shape_id,
    from rt_stop_times
    -- trip_id is only unique within a service date
    inner join valid_trips
    using(service_date, trip_id)
),
-- pair each trip with the next one to reach the stop. ordering on the actual
-- arrival rather than the scheduled start means an overtaking vehicle is paired
-- the way it really ran. trip_key breaks ties so the pairing is deterministic
headways as (
    select
        trip_key as first_trip_key,
        lead(trip_key) over (
            partition by shape_id, stop_sequence, stop_id
            order by actual_arrival_pacific asc, trip_key asc
        ) as second_trip_key,
        stop_sequence,
        stop_id,
        shape_id,
        actual_departure_pacific as first_actual_departure,
        lead(actual_arrival_pacific) over (
            partition by shape_id, stop_sequence, stop_id
            order by actual_arrival_pacific asc, trip_key asc
        ) as second_actual_arrival,
    from trip_stops
)
-- the last trip to call at a stop starts no headway
select
    *,
    -- the actuals are DATETIME, already in the feed's local time
    datetime_diff(second_actual_arrival, first_actual_departure, second)
        as headway_seconds,
from headways
where second_trip_key is not null
order by shape_id, stop_sequence, first_actual_departure asc
