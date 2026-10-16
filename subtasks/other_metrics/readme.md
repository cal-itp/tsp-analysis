Folder for exploratory work for GTFS-realtime derived metrics

### How to run

Use `uv sync` to create a development environment. Then open `trip_duration_variability.ipynb`, select the local venv in your Jupyter Notebook viewer, and run the entire notebook. You will need to be authenticated to Caltrans GCP.

### Tests

Use `uv run --group test pytest` to run tests. Tests are WIP since this is a draft state - they will be cut down in the production version.
