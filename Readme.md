# CTLearn Optimizer [GSOC 2019]

Framework for optimizing [CTLearn v0.3.0](https://github.com/ctlearn-project/ctlearn/tree/v031) 
models developed during the Google Summer of Code 2019.

This optimization utility uses [Tune](https://ray.readthedocs.io/en/latest/tune.html), 
a scalable framework for hyperparameter search and model training, and supports:

- Random search based optimization.
- Tree Parzen Estimators based optimization.
- Gaussian Processes based optimization.
- Genetic Algorithm based optimization.
- Parallel optimization (depending on available hardware resources).

## Authors

* **[Juan Alfonso Redondo Pizarro](https://github.com/jredondopizarro)**

## Project dependencies

- CTLearn
- environment_kernels
- Hyperopt
- Matplotlib
- NumPy
- Pandas
- pip
- PyYAML 
- Ray
- Scikit-Optimize 
- SciPy
- Scikit-learn
- Seaborn
- setproctitle

## Documentation dependencies (optional)

The packages listed below are only necessary if you want to build the 
documentation from the source.

- ipython
- nbsphinx
- pip
- Sphinx
- sphinx-autoapi
- sphinx_rtd_theme

### Installation, basic usage and configuration

Take a look at the the [documentation](https://ctlearn-optimizer.readthedocs.io/en/latest/index.html).





