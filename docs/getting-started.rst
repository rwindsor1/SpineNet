Getting Started
===============

Installation
------------

Currently, SpineNet can only be installed by cloning the git repository:

.. code-block:: console

   git clone git@github.com:rwindsor1/SpineNet.git

The easiest way to set up the environment is with `uv <https://docs.astral.sh/uv/>`_. From the root of the repository, run:

.. code-block:: console

   uv sync

This installs SpineNet and its dependencies into a ``.venv``, so no ``PYTHONPATH`` changes are needed. Use ``uv run`` to run scripts, e.g. ``uv run jupyter lab``.

Alternatively, you can install SpineNet with pip.

.. code-block:: console

   python -m venv spinenet-venv
   source spinenet-venv/bin/activate
   pip install -e .

Download Weights
----------------

To use any of the neural networks SpineNet relies on, you must first download the weights from our public server and insert
them at the location expected by the code.

This can be done automatically by starting a python environment and running the following script:


.. code-block:: python 

        import spinenet
        spinenet.download_weights(verbose=True)

.. autofunction:: spinenet.download_weights
