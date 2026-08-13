with feeds as (
    select date as service_date, feed_key, gtfs_dataset_name
    from mart_gtfs.fct_daily_schedule_feeds
    where date in ('{{ TARGET_DATES | join("', '") }}')
        and gtfs_dataset_name = '{{ TARGET_SCHEDULE_FEED }}'
),
shapes_routes as (
    select service_date, shape_id, route_id, route_short_name, feed_key, count(*) as ct
    from mart_gtfs.fct_scheduled_trips
    where service_date in ('{{ TARGET_DATES | join("', '") }}')
        and route_short_name = '{{ TARGET_ROUTE_SHORT_NAME }}'
    group by service_date, shape_id, route_id, route_short_name, feed_key
)
-- one row per service date and shape - the in-use shapes can change part way through a multi-date window
select * from feeds
left join shapes_routes using(feed_key, service_date)
order by ct desc
