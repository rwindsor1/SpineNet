# SpineNet

Done:
- [x] Detection and labelling
- [x] Grading
- [ ] Training (TODO)
- [ ] Pip-installable
- [ ] Note on using for non-commerical tasks only (ask AZ for draft)

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

