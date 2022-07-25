Running SpineNet
================

This section assumes that you have already installed SpineNet, see :doc:`getting-started` if you have not already done this.

Scan IO
-------

SpineNet interfaces with scans through the `SpinalScan` object. This is a simple wrapper class around the three variables SpineNet needs to run,namely the voxel data (volume), spacing between pixels in each slice and the space between each slice.

.. autoclass:: spinenet.io.SpinalScan
   :special-members: __init__

