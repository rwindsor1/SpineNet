Getting Started
===============

Installation
------------

Currently, SpineNet can only be installed by cloning the git repository:

.. code-block:: console

   git clone git@github.com:rwindsor1/SpineNet.git

Once this is done, you can then install a minimal conda environment to run the project.

.. code-block:: console

   conda create --name spinenet_env --file requirements.txt

Finally, add spinenet to your python path so it can be imported into scripts:

.. code-block:: console

        export PYTHONPATH=$PYTHONPATH:/path/to/SpineNet

Download Weights
----------------

To use any of the neural networks SpineNet relies on, you must first download the weights from our public server and insert
them at the location expected by the code.

This can be done automatically by starting a python environment and running the following script:


.. code-block:: python 

        import python
        spinenet.download_weights(verbose=True)

