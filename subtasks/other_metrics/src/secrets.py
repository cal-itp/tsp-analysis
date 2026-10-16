"""Secrets the notebooks need, read from GCP Secret Manager.

Nothing here is cached to disk or written into the repo: the value is fetched
when asked for, with whatever credentials the BigQuery engine is already using.
"""

from __future__ import annotations

import functools

from google.cloud import secretmanager

GCP_PROJECT = "cal-itp-data-infra"

CARTO_BASEMAPS_SECRET = "CARTO_BASEMAPS_KEY"

# the same tiles geopandas' "CartoDB positron" shorthand draws, but as a URL so
# the API key can ride along on it
CARTO_POSITRON_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"

# folium wants an attribution for any tiles given as a URL rather than a name
CARTO_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
    'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
)


def get_secret(name: str, project: str = GCP_PROJECT, version: str = "latest") -> str:
    """The payload of one secret version, as text."""
    client = secretmanager.SecretManagerServiceClient()
    path = client.secret_version_path(project, name, version)
    response = client.access_secret_version(name=path)
    return response.payload.data.decode("utf-8").strip()


@functools.cache
def carto_map_parameters(project: str = GCP_PROJECT) -> dict[str, str]:
    """The basemap arguments for ``explore``, carrying the CARTO API key.

    Spread them into the call, ``gdf.explore(..., **carto_map_parameters())``.
    Cached, so a notebook drawing several maps fetches the secret once.
    """
    key = get_secret(CARTO_BASEMAPS_SECRET, project=project)
    return {"tiles": f"{CARTO_POSITRON_URL}?key={key}", "attr": CARTO_ATTRIBUTION}
