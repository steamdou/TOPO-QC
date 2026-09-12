# Installation

1. Open a terminal at the root directory of this repository and create the Python environment using the provided `environment.yml` file:

```shell
mamba env create -f environment.yml
```

2. Activate the newly created environment:

```shell
conda activate topoqc
```

After activation, you should see `(topoqc)` at the beginning of your terminal prompt.

3. Install the package for this project, which registers a useful command for running the code. Navigate your terminal to the directory called python in this repo and run the following command:

```shell
pip install -e.
```

4. Set a environment variable to point to this repository:

```shell
export ML_MOGA=<path to this repo>
```