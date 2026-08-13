select service_date, shape_id, feed_key, pt_array
from mart_gtfs.fct_daily_scheduled_shapes
where service_date in ('{{ GEOMETRY_DATES | join("', '") }}')
    and feed_key in ('{{ FEED_KEYS | join("', '") }}')
    and shape_id in ('{{ SHAPE_IDS | join("', '") }}')
