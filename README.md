# Installation

1. Open a terminal at the root directory of this repository and create the Python environment using the provided `environment.yml` file:

```shell
mamba env create -f python/environment.yml
```

2. Activate the newly created environment:

```shell
conda activate topoqc
```

After activation, you should see `(topoqc)` at the beginning of your terminal prompt.

3. Install the project package in editable mode:

```shell
pip install -e .
```
