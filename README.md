# Bat Head Direction System Simulation

## Installation
Activate your virtual environment and install the required dependencies using pip:
```bash
python -m venv venv
source venv/bin/activate
```

Next, you install the package in editable mode using pip:
```bash
pip install -e .
```


When committing changes it is advised to run `nbstripout` to remove output from the notebooks. To set up `nbstripout` for this repository, run the following command in the root directory of the repository:
```bash
nbstripout --install
```

The models can be run with the provided jupyter notebooks in the `notebooks` directory. The `notebooks` directory contains a number of example notebooks that demonstrate how to use the models and visualize the results. You can run these notebooks in a Jupyter environment.