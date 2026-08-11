with feeds as (
    select feed_key, gtfs_dataset_name
    from mart_gtfs.fct_daily_schedule_feeds
    where date = '{{ TARGET_DATE }}' and gtfs_dataset_name = '{{ TARGET_SCHEDULE_FEED }}'
),
shapes_routes as (
    select shape_id, route_id, route_short_name, feed_key, count(*) as ct
    from mart_gtfs.fct_scheduled_trips
    where service_date = '{{ TARGET_DATE }}' and route_short_name = '{{ TARGET_ROUTE_SHORT_NAME }}'
    group by shape_id, route_id, route_short_name, feed_key
),
shapes as (
    select shape_id, feed_key, pt_array
    from mart_gtfs.fct_daily_scheduled_shapes
    where service_date = '{{ TARGET_DATE }}'
)
select * from feeds
left join shapes_routes using(feed_key)
left join shapes using(shape_id, feed_key)
order by ct desc