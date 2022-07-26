Running SpineNet
================

This section assumes that you have already installed SpineNet, see :doc:`getting-started` if you have not already done this.

SpineNet interacts with sagittally-sliced DICOM spinal MRIs using the :py:class:`spinenet.io.SpinalScan` class. For information on creating instances of this scan from your data, consult the :doc:`loading-scans` page.  

There are two stages in the SpineNet pipeline:

#. Detecting and labelling vertebral bodies in the raw scans. 
   This is done using the :py:func:`spinenet.SpineNet.detect_vb` class method. 

#. (T2 Lumbar Scans Only) Grading intervertebral disc volumes for a range of common degenerative 
   conditions, including:
  
  * Pfirrman Grading (5 Classes)
  * Disc Narrowing (4 Classes)
  * Central Canal Stenosis (4 Classes)
  * Upper & Lower Endplate Defects (Binary)
  * Upper & Lower Marrow Changes (Binary)
  * Left/Right Foraminal Stenosis (Binary)
  * Spondylolisthesis (Binary)
  * Disc Herniation (Binary)
  
  This is done by using the :py:func:`spinenet.SpineNet.get_ivds_from_vert_dicts` 
  and :py:func:`spinenet.SpineNet.grade_ivds` methods.

Before running any of the above steps, you must first initialize the SpineNet 
model. This can be done as follows:

.. code-block:: python

    # Run this if you have not already downloaded the model weights
    spinenet.download_weights(verbose=True, force=False)

    # load in spinenet. Replace device with 'cpu' if you are not using a CUDA-enabled machine.
    spnt = SpineNet(device='cuda:0', verbose=True)

Detecting Vertebral Bodies
--------------------------

To detect vertebral bodies in a spinal scan, pass the corresponding `SpinalScan` 
object the initialised spinenet object's :py:func:`detect_vb` method:


.. code-block:: python

    # `scan` is a `spinenet.io.SpinalScan` object
    vert_dicts = spnt.detect_vb(scan.volume, scan.pixel_spacing)

Here `vert_dicts` is a list of dictionaries. Each dictionary contains information
about the vertebra detected, including the following keys:

* `predicted_label`: The predicted label of the vertebra, e.g. S1, T12, etc.
* `slice_nos`: List of indicies of the sagittal slices in which the vertebra was detected.
* `polys`: A list of co-ordinates describing the four corners of the vertebral body quadrilateral detected.
  The co-ordinates at element `i` in the list, describe the polygon detected in `vert_dict['slice_nos'][i]`.

.. autofunction:: spinenet.SpineNet.detect_vb

You can save the vert_dicts to a CSV file using `spinenet.io.save_vert_dicts_to_csv`:

.. code-block:: python

    spinenet.io.save_vert_dicts_to_csv(vert_dicts, 'my_vert_dicts.csv')

.. autofunction:: spinenet.io.save_vert_dicts_to_csv

Grading IVDS
------------

:note: Please bear in mind that the grading models are only trained on T2 lumbar scans 
  and thus will not work for other fields of view or sequences.

Now the vertebra have been detected, you can use them to extract volumes surrounding
each IVD from L5-S1 to T12-L1. 

To do this, you can use the :py:func:`spinenet.SpineNet.get_ivds_from_vert_dicts`

.. code-block:: python

    ivd_dicts = spnt.get_ivds_from_vert_dicts(vert_dicts, scan.vol)

.. autofunction:: spinenet.SpineNet.get_ivds_from_vert_dicts

Similar to `vert_dicts`, this is also a list of dictionaries. However, 
each dictionary contains information about extracted IVD detected, 
including the following keys:

* `volume`: a np.array of the extracted IVD volume, resized for the grading network.
* `level_name`: The predicted label of the IVD, e.g. L5-S1, T12-L1, etc.

`ivd_dicts` is then passed to the grading network to grade the IVD volumes:

.. code-block:: python

    ivd_grades = spnt.grade_ivds(ivd_dicts)

    # ivd grades is a pd.DataFrame where each row contains the gradings for a
    # single IVD. You can save it as follows:
    ivd_grades.to_csv('my_ivd_grades.csv')

.. autofunction:: spinenet.SpineNet.grade_ivds

  



