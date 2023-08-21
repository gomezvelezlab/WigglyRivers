# PyTemplate

This package serves as a template to create python packages. The package uses a function to plot a sine function given a specific data.

# Installation

This package has no requirements. I encourage using a virtual environment of [Anaconda 3](https://www.anaconda.com/products/individual) with Python 3.6 or higher. The virtual environments creation can be seen [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

To install it you need to know how to activate the anaconda instance on terminal or in the command window.

```
git clone https://github.com/gomezvelezlab/PyTemplate
```

Then, you can install the whole package as

```
python setup.py install
```

**However, If you plan to be working on the package I recommend that you install the package using**
```
pip install -e .
```

The latter command installs the package in the bin folder but creates a link to this folder, thus any modification in this folder will be available when importing the package.

# Usage

```python
from pyModel import Celsius

temp_c = Celsius(20)
print(temp_c)
temp_f = temp_c.to_fahrenheit()

```

# License
[License](https://github.com/DGD042/pycomsol/blob/master/LICENSE)
