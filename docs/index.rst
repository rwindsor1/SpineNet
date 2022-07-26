.. SpineNet documentation master file, created by
   sphinx-quickstart on Mon Jul 25 06:14:09 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SpineNet
====================================

SpineNet is automated software for analysing clinical spinal MRI scans. Current functionality includes:

1. Detecting and labelling vertebral bodies in sagittally-sliced MRI scan across a range of common sequences (e.g. T1, T2, STIR etc.) and fields of view (lumbar, cervical, whole spine).
2. Performing radiological grading at each intervertebral disc level in T2 lumbar scans from T12/L1 to L5/S1 for a range of common degenerative changes.


As well as using SpineNet for detecting and labelling vertebral bodies and the existing radiological gradings, you can also use the codebase for finetuning the model to perform additional gradings. For example, we have previously adapted it to detect vertebral fractures, cauda equina, ankylosing spondylitis and other diseases.

.. note::

Please note that by using SpineNet you agree to our `terms of access <https://github.com/rwindsor1/SpineNet/blob/main/LICENCE.md>`_. Amongst other things, this prohibits the use of SpineNet for commerical purposes. If you wish to acquire a more permissive licence of SpineNet, please contact us.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting-started
   loading-scans
   running-spinenet
   api




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
