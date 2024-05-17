# WigglyRivers

Sinuous channels are ubiquitous features along river networks. Their complex patterns span scales and influence morphodynamic processes, landscape evolution, and ecosystem services. Identifying and characterizing meandering features along river transects has challenged traditional curvature-based algorithms. Here, we present _WigglyRivers_, a Python package that builds on existing work using wavelet-based methods to create an unsupervised identification tool. This tool allows the characterization of the multiscale nature of river transects and the identification of individual meandering features. The package uses any set of river coordinates and calculates the curvature and direction-angle to perform the characterization, and also leverages the use of the High-Resolution National Hydrography Dataset (NHDPlus HR) to assess river transects at a catchment scale. Additionally, the _WigglyRivers_ package contains a supervised river identification tool that allows the visual selection of individual meandering features with satellite imagery in the background. Here, we provide examples in idealized river transects and show the capabilities of the _WigglyRivers_ package at a catchment scale. We also use the supervised identification tool to validate the unsupervised identification on river transects across the US. The package presented here can provide crucial data that represents an essential step toward understanding the multiscale characteristics of river networks and the link between river geomorphology and river corridor connectivity.

## Installation

### Requirements

This package has a few requirements. I encourage using a virtual environment of [Anaconda 3](https://www.anaconda.com/products/individual) with Python 3.10 or higher. The virtual environment creation can be seen [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Below, we list the process to create a virtual environment and install the requirements for the package.

```bash
conda create -n wigglyrivers_env python=3.XX
conda activate wigglyrivers_env
conda install -c conda-forge geopandas
conda install -c conda-forge h5py
```

This package was tested in Python version 3.10 and higher. Some incompatible dependencies might arise with older versions of Python. `geopandas` is installed first because the package generates the most incompatibilities. After that package, we install the other dependencies with pip

```bash
pip install simpledbf
pip install statsmodels
pip install seaborn
pip install pyarrow
pip install plotly
pip install salem
pip install anytree
pip install meanderpy
pip install circle-fit
```

The package uses `anytree` to store the information of the meanders.

For interactive plots
```bash
pip install ipympl
```

If you are using `.env` files, remember to also install

```bash
pip install python-dotenv
```

### Install WigglyRivers

To install the package you need to clone the repository and install it using `pip`.

```bash
pip install -e .
```
Some known incompatible dependencies are addressed in the troubleshooting section. If you have any issues not discussed in the troubleshooting section, please open an issue in the repository.

### Test Installation



### Troubleshooting Package Installation

- If you have problems with `geopandas` look at [this website](https://wilcoxen.maxwell.insightworks.com/pages/6373.html#:~:text=It%20has%20complex%20links%20to,between%2010%20and%2030%20minutes.).
- `h5py` and `fiona` might have some issues when importing at the same time. Installing both of them using `conda install -c -conda-forge` solved the issue for me.

- If the interactive plot with `plotly` gives you issues with `ipywidgets`  and `jupyterlab-widgets`, install the following versions  `pip install ipywidgets==7.7.1 jupyterlab-widgets==1.1.1` 

- There is a known issue with `plotly<=5.15` where plotting MAPBOX with the interactive widget will prompt the following error message:

    ```python
    ValueError:
    Invalid property path 'mapbox._derived' for layout
    ```

  There is a temporary fix to this issue given in the following [GitHub issue webpage](https://github.com/plotly/plotly.py/issues/2570) that requires the use of the function below and restart the kernel.

    ```python
    def fix_widget_error():
        """
        Fix FigureWidget - 'mapbox._derived' Value Error.
        Adopted from: https://github.com/plotly/plotly.py/issues/2570#issuecomment-738735816
        """
        import shutil
        import pkg_resources

        pkg_dir = os.path.dirname(pkg_resources.resource_filename("plotly", "plotly.py"))

        basedatatypesPath = os.path.join(pkg_dir, "basedatatypes.py")

        backup_file = basedatatypesPath.replace(".py", "_bk.py")
        shutil.copyfile(basedatatypesPath, backup_file)

        # read basedatatypes.py
        with open(basedatatypesPath, "r") as f:
            lines = f.read()

        find = "if not BaseFigure._is_key_path_compatible(key_path_str, self.layout):"

        replace = """if not BaseFigure._is_key_path_compatible(key_path_str, self.layout):
                    if key_path_str == "mapbox._derived":
                        return"""

        # add new text
        lines = lines.replace(find, replace)

        # overwrite old 'basedatatypes.py'
        with open(basedatatypesPath, "w") as f:
            f.write(lines)

    # Run fix
    fix_widget_error()
    ```

# Workflows

The workflows can be found in the [examples/new_user_workflow/](https://github.com/gomezvelezlab/WigglyRivers/tree/stable/examples/new_user_workflow) folder. You can see examples of synthetic river transects and natural river transects using the NHDPlus High-Resolution dataset.

# License
[License](https://github.com/gomezvelezlab/WigglyRivers/blob/stable/LICENSE)

# How to cite

If you use this package, please cite the following paper:

Gonzalez-Duque, D., & Gomez-Velez, J. D. (2024). WigglyRivers: Characterizing the Multiscale Nature of Meandering Channels [Submitted]. Environmental Modelling & Software.



