************
Installation
************

.. note::
    CTLearn Optimizer supports Linux and MacOS, this is subject to the 
    `Ray <https://ray.readthedocs.io/en/latest/installation.html>`_
    package compability.

1. Install `CTLearn <https://github.com/ctlearn-project/ctlearn/>`_ by following 
the Readme.md hosted on the project's Github repository.
This will create a new conda environment named *ctlearn* (by default).

.. warning::
    Currently, only `CTLearn v0.3.0 <https://github.com/ctlearn-project/ctlearn/tree/v031>`_.
    is supported. Newer versions support is planned for the future.

2. Clone this repository with Git:

.. code-block:: bash

    cd </installation/path>
    git clone https://github.com/ctlearn-project/ctlearn_optimizer.git

Now, activate the conda environment created in the first step:

.. code-block:: bash

    conda activate ctlearn

3. Install the dependencies for CTLearn Optimizer inside the environment:

.. code-block:: bash

    conda env update --file </installation/path>/environment.yml

.. note::

    Optional: install the dependencies for building the CTLearn Optimizer's
    documentation from the source.

    .. code-block:: bash

        conda env update --file </installation/path>/docs_environment.yml

4. Finally, install CTLearn Optimizer into  the conda environment with pip:

.. code-block:: bash

    cd </installation/path>/ctlearn_optimizer
    pip install --upgrade .


