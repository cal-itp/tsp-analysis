## heading_correction

#### Task Overview

This task is in preparation for implementing measures of intersection delay and dwell time. We received a sample of CAD/AVL data from Culver CityBus, for a small number of trips on their CC1 route. This analysis is to determine the frequency and reliability of this data in order to determine whether it can be used for implementing intersection delay and dwell time. The data is stored on a bucket in Caltrans' Google Cloud Storage.

#### Running

This subtask uses its own dependencies, managed through Poetry. Enable them by following these instructions:

- Install the Poetry environment: `poetry install`
  - If using VS Code to run your notebook, it is easier if the environment installs to your local directory. Enable this by running `poetry config virtualenvs.in-project true`
- Add the notebook to Jupyter, and run the notebook. The exact steps will vary based on the client you are using to run the notebook.
  - In VS Code: Run the notebook, and use the newly created `.venv` environment
  - In JupyterHub or JupyterLab: It's a little more complicated:
    - Activate the Poetry environment by running `poetry env activate` and copying the output into the shell
    - Note the name of the new environment that is activated, likely `non-package-mode-py3.11`
    - Run  `python -m ipykernel install --user --name=[env_name] --display-name "[display name]"` where `env_name` is the name of the environment and `display name` is a memorable name for the environment (i.e. "Culver City Headings")
    - From the Jupyterhub kernel button (top right), select the display name you set for the new environment. It may take a moment for the option to appear.
- Run all cells of the notebook to create a tabular output and an embedded Folium map for validation
