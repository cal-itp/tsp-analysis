## heading_correction

#### Task Overview

This task is in response to a reported issue from the City of Culver City regarding their transit signal priority (TSP) system. This request-based system requires buses to report their headings to the TSP-enabled signal, which would determine the signal that the heading corresponded to and process the request. They found that the headings of streets at signalized intersections across the city generally did not match the headings that were assigned by their TSP system, preventing activation at those intersections. This tool attempts to simplify this process.

#### Running

This subtash uses its own dependencies, managed through Poetry. Enable them by following these instructions:

- Install the Poetry environment: `poetry install`
  - If using VS Code to run your notebook, it is easier if the environment installs to your local directory. Enable this by running `poetry config virtualenvs.in-project true`
- Add the notebook to Jupyter, and run the notebook. The exact steps will vary based on the client you are using to run the notebook.
  - In VS Code: Run the notebook, and use the newly created `.venv` environment
  - In JupyterHub: It's a little more complicated:
    - Activate the Poetry environment by running `poetry env activate` and copying the output into the shell
    - Note the name of the new environment that is activated, likely `non-package-mode-py3.11`
    - Run  `python -m ipykernel install --user --name=[env_name] --display-name "[display name]"` where `my_env` is the name of the environment and display name is a memorable name for the environment (i.e. "Culver City Headings")
- Run all cells of the notebook to create a tabular output and an embedded Folium map for validation

#### Configuration
The current phase of the project requires you to configure street names by matching street names (as defined in OpenStreetMap) and their orientation, either `NORTH_SOUTH` or `EAST_WEST`. This is currently done using the `STREET_DIRECTIONS` argument in the notebook. As this task expands, a more standard and flexible configuration method will be used.