select key as feed_key, base64_url, _valid_from
from mart_gtfs.dim_schedule_feeds
where key in ('{{ FEED_KEYS | join("', '") }}')
