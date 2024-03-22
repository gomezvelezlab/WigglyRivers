# WigglyRivers

_WigglyRivers_ is a Python-based tool that allows the characterization of the multiscale of 

## Installation

### Requirements

This package has a few requirements. I encourage using a virtual environment of [Anaconda 3](https://www.anaconda.com/products/individual) with Python 3.6 or higher. The virtual environment creation can be seen [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Below, we list the process to create a virtual environment and install the requirements for the package.

```bash
conda create -n wigglyrivers_env10 python=3.10
conda activate wigglyrivers_env10
conda install -c conda-forge geopandas
conda install -c conda-forge h5py
```
`geopandas` is installed first because it is the package that generates the most incompatibilities. After that package, we install the other dependencies with pip
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

The package uses `anytree` to store the information of the meanders. To plot the tree please install [graphviz](https://graphviz.org/download/).

For interactive plots
```bash
pip install ipympl
```
If you are using `.env` files remember to also install

```bash
pip install python-dotenv
```

### Install WigglyRivers

To install the package you need to clone the repository and install it using `pip`.

```bash
pip install -e .
```

### Troubleshooting Package Installation

- If you have problems with `geopandas` look at [this website](https://wilcoxen.maxwell.insightworks.com/pages/6373.html#:~:text=It%20has%20complex%20links%20to,between%2010%20and%2030%20minutes.) as some of the troubleshooting might help.
- `h5py` and `fiona` might have some issues when importing at the same time. installing both of them using `conda install -c -conda-forge` solved the issue for me.
- If the interactive plot with `plotly` gives you issues with `ipywidgets`  and `jupyterlab-widgets`, install the following versions  `pip install ipywidgets==7.7.1 jupyterlab-widgets==1.1.1` 
- If you run into any issues with newer versions of `plotly` and Jupyter Notebooks, try installing the following versions of Jupyter lab widgets:

```bash
pip install ipywidgets==7.7.1 jupyterlab-widgets==1.1.1
```

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


# License
[License](https://github.com/gomezvelezlab/WigglyRivers/blob/stable/LICENSE)

# How to cite


# Reference