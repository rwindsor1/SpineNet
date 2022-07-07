# SpineNet

To do:
- [x] Detection and labelling pipeline
- [x] Grading pipeline
- [ ] Training/Finetuning Grading Model
- [ ] Pip-installable
- [ ] Note on using for non-commerical tasks only (ask AZ for draft)

## Introduction

SpineNet is automated software for analysing clinical spinal MRI scans. It's functionality is currently:

1. Detecting and labelling vertebral bodies in sagittally-sliced MRI scan across a range of common sequences (e.g. T1, T2, STIR etc.) and fields of view (lumbar, cervical, whole spine).
2. Performing radiological grading at each intervertebral disc level in T2 lumbar scans from T12/L1 to L5/S1 for a range of common degenerative changes.

## Getting Started

(To be added: pip installation)

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

## Citation

If you use SpineNet in your research, please cite our technical report, where can be found on arxiv [https://arxiv.org/abs/2205.01683](here)

`SpineNetV2: Automated Detection, Labelling and Radiological Grading Of Clinical MR Scans' (2022), Rhydian Windsor, Amir Jamaludin, Timor Kadir, Andrew Zisserman, Technical Report

You may also wish to cite our other works in this area.
