Loading Scans
================

This section assumes that you have already installed SpineNet, see :doc:`getting-started` if you have not already done this.

Scan IO
-------

SpineNet interfaces with scans through the :py:class:`spinenet.io.SpinalScan` object. This is a simple wrapper class around the three variables SpineNet needs to run,namely the voxel data (volume), spacing between pixels in each slice and the space between each slice.

.. autoclass:: spinenet.io.SpinalScan
   :special-members: __init__

You can create members of this class to be run by SpineNet using the above initialisation method, e.g. by 

.. code-block:: python

        volume = np.random.rand(512,512,12) # random voxel data for scan; 12 slices of size 512x512
        pixel_spacing = [0.5, 0.5] # distance between each pixel is to 0.5mm 
        slice_thickness = 2.0 # distance between slices is 2.0mm
        # initialize scan object
        scan = spinenet.io.SpinalScan(volume, pixel_spacing, slice_thickness)


However, it is probably better to use the helper functions for loading in DICOMs as scans;
:py:func:`spinenet.io.load_dicoms` and :py:func:`spinenet.io.load_dicoms_from_folder`. Both these
scans have similar functions; they load DICOM files into a :py:class:`spinenet.io.SpinalScan` which
can be then used by the rest of SpineNet's pipeline.

To use a list of DICOM file paths, use :py:func:`spinenet.io.load_dicoms`.

.. autofunction:: spinenet.io.load_dicoms

Alternatively, for a folder containing DICOM files, use :py:func:`spinenet.io.load_dicoms_from_folder`.

.. autofunction:: spinenet.io.load_dicoms_from_folder


Example Scans
-------------

To try SpineNet on some example data, you can download the example scans from our server via :py:func:`spinenet.download_example_scan`
as follows

.. code-block:: python

   os.mkdir('example_scans')
   spinenet.download_example_scan('t2_lumbar_scan_1', './example_scans')

   # some of the example scans do not have the requisite DICOM headers, so
   # we can insert our own values using the `metadata_overwrites` argument
   overwrite_dict = {'SliceThickness': [2], 'ImageOrientationPatient': [0, 1, 0, 0, 0, -1]}

   # load scan
   spinenet.io.load_dicoms_from_folder('example_scans',require_extensions=False, metadata_overwrites=overwrite_dict)


.. autofunction:: spinenet.download_example_scan
 
