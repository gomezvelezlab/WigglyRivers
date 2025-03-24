<p align="center">
<img src="docs/repo_figures/title_fig.png" alt="drawing" width="300"/>
</p>

# _WigglyRivers_

_WigglyRivers_ is a Python package that builds on existing work using wavelet-based methods to create supervised and unsupervised meander identification tools. This tool allows the characterization of the multiscale nature of river transects and the identification of individual meandering features. The package uses any set of river coordinates and calculates the curvature and direction-angle to perform the characterization, and also leverages the use of the High-Resolution National Hydrography Dataset (NHDPlus HR) to assess river transects at a catchment scale. Additionally, the _WigglyRivers_ package contains a supervised river identification tool that allows the visual selection of individual meandering features with satellite imagery in the background.

## Example Workflows

You can find examples of how to use this package as examples workflows for synthetic and natural river transects in the [examples/new_user_workflow/](https://github.com/gomezvelezlab/WigglyRivers/tree/stable/examples/new_user_workflow) folder.

The synthetic river transect examples are generated using idealized Kinoshita-type curves, implemented in the code, and using the [`meanderpy`](https://github.com/zsylvester/meanderpy) package coded by Sylvester, Durkin, and Covault (2019). The natural river transect example uses the information from the [High-Resolution National Hydrography Dataset (NHDPlus HR)](https://www.usgs.gov/core-science-systems/ngp/national-hydrography/nhdplus-high-resolution) to extract the river transects and assess the meandering features.

A tutorial on how to use the NHD capability of the package is available in this [link](https://drive.google.com/file/d/1LxUsNX8w74yv7fj-zl2lByl01Wuytxvm/view?usp=sharing)

## Installation

### Core Requirements

This package has a few requirements. I encourage using a virtual environment of [Anaconda 3](https://www.anaconda.com/products/individual) with Python 3.10 or higher. The virtual environment creation can be seen [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Below, we list the process to create a virtual environment and install the requirements for the package.

```bash
conda create -n wigglyrivers_env python=3.XX
conda activate wigglyrivers_env
conda install -c conda-forge geopandas
conda install -c conda-forge h5py
```

### Manual Installation of Requirements

This path is not requiere if you install the package with `pip install WigglyRivers`. Some incompatible dependencies might arise with older versions of Python. `geopandas` is installed first because the package generates the most incompatibilities. After that package, we install the other dependencies with pip

```bash
pip install simpledbf
pip install statsmodels
pip install seaborn
pip install pyarrow
pip install plotly
pip install salem
pip install anytree
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

To run one of the synthetic examples, you will need to install the `meanderpy` package. You can install it using the following command:

```bash
pip install meanderpy
```

### Install WigglyRivers

To install the package you need to clone the repository and install it using `pip`.

```bash
pip install WigglyRivers
```

If you want to install the package manually, you can clone the repository and install it using the following command:

```bash
git clone https://github.com/gomezvelezlab/WigglyRivers.git
```

Then, install the package using the following command:

```bash
pip install -e .
```

Some known incompatible dependencies are addressed in the troubleshooting section. If you have any issues not discussed in the troubleshooting section, please open an issue in the repository.

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

## How to cite

If you use this package, please cite the following paper:

Gonzalez-Duque, D., & Gomez-Velez, J. D. (2025). WigglyRivers: A tool to characterize the multiscale nature of meandering channels. Environmental Modelling & Software, 106423. <https://doi.org/10.1016/j.envsoft.2025.106423>


## Credit and Acknowledgments

The Continous Wavelet Transform (CWT) computation uses an updated version of the code created by Torrence and Compo (1998) brought to Python by Evgeniya Predybaylo. Also, the unsupervised identification algorithm is based on previous work coded in Matlab by Vermeulen et al. (2016). Each code and function is adequately accredited and cited in the code.

## References

Sylvester, Z., Durkin, P., & Covault, J. A. (2019). High curvatures drive river meandering. Geology, 47(3), 263–266. <https://doi.org/10.1130/G45608.1>

Torrence, C., & Compo, G. P. (1998). A Practical Guide to Wavelet Analysis. Bulletin of the American Meteorological Society, 79(1), 61–78. <https://doi.org/10.1175/1520-0477(1998)079><0061:APGTWA>2.0.CO;2

Vermeulen, B., Hoitink, A. J. F., Zolezzi, G., Abad, J. D., & Aalto, R. (2016). Multiscale structure of meanders. Geophysical Research Letters, 2016GL068238. <https://doi.org/10.1002/2016GL068238>
