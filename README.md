

# SpineNet - PyPI Package

[![PyPI version](https://badge.fury.io/py/spinenet.svg)](https://badge.fury.io/py/spinenet)
[![Python](https://img.shields.io/pypi/pyversions/spinenet.svg)](https://pypi.org/project/spinenet/)
[![Tests](https://github.com/spinetools/SpineNet/actions/workflows/publish-pkg.yml/badge.svg)](https://github.com/spinetools/SpineNet/actions/workflows/publish-pkg.yml)



SpineNet is now available as a PyPI package for easy installation and use!
Install SpineNet directly from PyPI using pip:

```bash
pip install spinenet
```

The latest version is [2.1.5](https://pypi.org/project/spinenet/2.1.5/) (see [Versions](#versions)). It needs Python 3.9 to 3.11 and PyTorch 2.0 to 2.5, and is tested with NumPy 1 and 2 and OpenCV 4.5 or newer (except 4.8.0.76).

<p align="center">
<img width='100%' src='spinenetlogo.png'> 
</p>
**DISCLAIMER: SpineNet is not a diagnostics tool nor a medical device. It should only be used for research.**


## Introduction

SpineNet is automated software for analysing clinical spinal MRI scans. Current functionality includes:

1. Detecting and labelling vertebral bodies in sagittally-sliced MRI scan across a range of common sequences (e.g. T1, T2, STIR etc.) and fields of view (lumbar, cervical, whole spine).
2. Performing radiological grading at each intervertebral disc level in T2 lumbar scans from T12/L1 to L5/S1 for a range of common degenerative changes.

As well as using SpineNet for detecting and labelling vertebral bodies and the existing radiological gradings, you can also use the codebase for finetuning the model to perform additional gradings. For example, we have previously adapted it to detect vertebral fractures, cauda equina, ankylosing spondylitis and other diseases.

Please note that by using SpineNet you agree to our [terms of access](https://github.com/spinetools/SpineNet/blob/main/LICENCE.md). Amongst other things, this prohibits the use of SpineNet for commercial purposes. If you wish to acquire a more permissive licence of SpineNet, please contact us.


<p align="center">
<img width='100%' src='pipeline-overview.png'> 
</p>

## Getting Started

SpineNet's documentation is available [here](https://spinenet.readthedocs.io/en/latest)
### Install enviroments
A minimal python environment for running SpineNet can be found in `requirements.txt` and can be installed used

```
$ python -m venv spinenet-venv
$ source spinenet-venv/bin/activate
$ pip install -r requirements.txt
```

### Running The Code

Clone the repository and add the root directory to your PYTHONPATH, e.g. via
```
$ export PYTHONPATH=$PYTHONPATH:/path/to/SpineNet
```

When using SpineNet for the first time, you first must download weights from the VGG public server, by calling the function

```
spinenet.download_weights(verbose=True)
```

or to re-download them:

```
spinenet.download_weights(verbose=True,force=True)
```

For a guide on using SpineNet, see the `01-quickstart.ipynb` tutorial notebook.

### Saving IVD Volumes as NIfTI

SpineNet needs Python 3.9 to 3.11. With it installed (`pip install spinenet`) and the weights downloaded as above, the volume around each intervertebral disc (IVD) can be saved to its own file:

```python
from spinenet import SpineNet
from spinenet.io import load_dicoms_from_folder, save_ivd_volumes

scan = load_dicoms_from_folder('path/to/dicom_folder')
spnt = SpineNet(device='cpu', verbose=True)  # 'cuda:0' on a GPU machine
vert_dicts = spnt.detect_vb(scan.volume, scan.pixel_spacing)

# (slices, height, width); only the default (9, 112, 224) can be graded
# pixel_spacing keeps the IVDs undistorted if rows and columns differ in spacing
ivd_dicts = spnt.get_ivds_from_vert_dicts(
    vert_dicts, scan.volume, output_shape=(12, 128, 256),
    pixel_spacing=scan.pixel_spacing,
)

# scan.affine: the position, orientation and voxel sizes of the scan in mm
save_ivd_volumes(ivd_dicts, 'ivd_volumes', file_format='nii.gz',
                 prefix='scan1_', scan_affine=scan.affine)
# -> ivd_volumes/scan1_L4-L5.nii.gz, ...
```

`file_format` is `'npy'` (default) or `'nii.gz'`. `scan_affine` sets the affine in the NIfTI header of each IVD volume (as sform and qform, in mm):

- Not given: the identity.
- `scan.affine`: the position, orientation and voxel sizes of each IVD in the scanner. `load_dicoms_from_folder` sets it from the positions of the DICOM slices, which it keeps in `scan.image_positions` and `scan.image_orientation`. If they have none, it only holds the voxel sizes in mm (pixel spacing and slice thickness). `SpacingBetweenSlices` is only used as slice thickness if `SliceThickness` is missing.
- For a scan stored as NIfTI, give its affine to `SpinalScan`. It has to be the affine of the volume exactly as it is passed to SpineNet (height x width x sagittal slices):

```python
import nibabel as nib
from spinenet.io import SpinalScan

img = nib.load('scan.nii.gz')
# reorient to rows head -> feet, columns front -> back, sagittal slices last
ornt = nib.orientations.ornt_transform(
    nib.io_orientation(img.affine), nib.orientations.axcodes2ornt(('I', 'P', 'L'))
)
img = img.as_reoriented(ornt)
spacing = img.header.get_zooms()
scan = SpinalScan(img.get_fdata(), spacing[:2], spacing[2], affine=img.affine)
# then detect and save as above
```

## Citation

If you use SpineNet in your research, please cite our technical report, which can be found on arxiv [here](https://arxiv.org/abs/2205.01683)

"SpineNetV2: Automated Detection, Labelling and Radiological Grading Of Clinical MR Scans" (2022), Rhydian Windsor, Amir Jamaludin, Timor Kadir, Andrew Zisserman, *Technical Report*


You may also wish to cite our other works in this area:

"A Convolutional Approach to Vertebrae Detection and Labelling in Whole Spine MRI" (2020), Rhydian Windsor, Amir Jamaludin, Timor Kadir, Andrew Zisserman, *Medical Image Computing and Computer Aided Intervention (MICCAI) 2020*. [Springer](https://link.springer.com/chapter/10.1007/978-3-030-59725-2_69), [ArXiv](https://arxiv.org/abs/2007.02606)

"Context-Aware Transformers For Spinal Cancer Detection and Radiological Grading" (2022), Rhydian Windsor, Amir Jamaludin, Timor Kadir, Andrew Zisserman,
*Medical Image Computing and Computer Aided Intervention (MICCAI) 2022*. [ArXiv](https://arxiv.org/abs/2206.13173)

"SpineNet: Automated classification and evidence visualization in spinal MRIs" (2017), Amir Jamaludin, Timor Kadir, Andrew Zisserman, *Medical Image Analysis 41, p 63-73*.

"ISSLS PRIZE IN BIOENGINEERING SCIENCE 2017: Automation of reading of radiological features from magnetic resonance images (MRIs) of the lumbar spine without human intervention is comparable with an expert radiologist", (2017) Amir Jamaludin, Meelis Lootus, Timor Kadir, Andrew Zisserman, Jill Urban, Michele C. Battié, Jeremy Fairbank, Iain McCall & The Genodisc Consortium, *European Spine Journal 26, p 1374–1383*

## Example Scan Attribution

We are grateful to the providers of the example scans used in the tutorials (originally from [radiopedia.org](radiopedia.org)).
- 't2_lumbar_scan_1': [Case Contributed by Assoc Prof Frank Gaillard](https://radiopaedia.org/cases/normal-lumbar-spine-mri?lang=gb)
- 't2_lumbar_scan_2': [Case Contributed by Dr Henry Knipe](https://radiopaedia.org/cases/schmorl-node-acute-10?lang=gb)
- 'stir_whole_spine': [Case contributed by Dr Ian Bickle](https://radiopaedia.org/cases/normal-mri-whole-spine-3-t?lang=gb)

## Versions

All releases are on [PyPI](https://pypi.org/project/spinenet/#history) and [GitHub](https://github.com/spinetools/SpineNet/releases).

| Version | Date | Changes |
| --- | --- | --- |
| [2.1.5](https://github.com/spinetools/SpineNet/releases/tag/v2.1.5) | 2026-11-23 | Package licence on PyPI set to the SpineNet [terms of access](https://github.com/spinetools/SpineNet/blob/main/LICENCE.md) instead of MIT; tests run on Ubuntu 24.04 to keep Python 3.9 ([#8](https://github.com/spinetools/SpineNet/issues/8)) |
| [2.1.4](https://github.com/spinetools/SpineNet/releases/tag/v2.1.4) | 2026-10-15 | Scan affine kept consistent with the loaded DICOM slices; numpy 2 deprecation warning fixed ([#7](https://github.com/spinetools/SpineNet/issues/7)) |
| [2.1.3](https://github.com/spinetools/SpineNet/releases/tag/v2.1.3) | 2026-10-11 | `SpacingBetweenSlices` used if `SliceThickness` is missing; GitHub releases created on publish ([#7](https://github.com/spinetools/SpineNet/issues/7)) |
| 2.1.2 | 2026-10-11 | Scan affine from the DICOM slice positions; numpy 2 and newer OpenCV supported; non-square pixel spacing in IVD and vertebra crops |
| 2.1.1 | 2026-10-05 | `voxel_to_scan` in the IVD dicts and `scan_affine` in `save_ivd_volumes` |
| 2.1.0 | 2026-10-01 | Configurable IVD volume shape and `save_ivd_volumes` |
| 2.0.5 | 2026-09-30 | Anisotropic pixel spacing fixed in detection and labelling |
| [2.0.4](https://github.com/spinetools/SpineNet/releases/tag/v2.0.4) | 2025-09-26 | Different x and y pixel spacing supported |
| [2.0.3](https://github.com/spinetools/SpineNet/releases/tag/v.2.0.3) | 2025-09-17 | Pixel spacing stored as an array; tests and automatic publishing to PyPI |
| 2.0.2 | 2025-09-09 | First release on PyPI |

## Coming Soon

- [x] Documentation
- [x] Pip installation
- [ ] In-depth tutorials on finetuning SpineNet's grading network

We are always looking to extend SpineNet to make it a more useful tool. If you have a request for features please get in touch with us.
