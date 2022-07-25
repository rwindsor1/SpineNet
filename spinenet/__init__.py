import os
from typing import List, Tuple, Dict, Union, Any
import zipfile

from .io import download_weights as download_weights_io, download as download_io
from .main import SpineNet
from .utils import *


def download_weights(verbose: bool = False, force: bool = False) -> None:
    """
    Download SpineNet model weights from public server and save to a location
    where the pipeline expects them (`./spinenet/weights/`).

    Parameters
    ----------
    verbose : bool, optional
        Whether to print out extra information about the file downloads.
    force : bool, optional
        Set to force download weights even if they already have been locally downloaded
    """

    weights_dict = {
        "grading/ckpt1.pt": "https://www.robots.ox.ac.uk/~vgg/research/spinenet/weights/grading/ckpt1.pt",
        "appearance/ckpt187.pt": "https://www.robots.ox.ac.uk/~vgg/research/spinenet/weights/appearance/ckpt187.pt",
        "detect-vfr/ckpt435.pt": "https://www.robots.ox.ac.uk/~vgg/research/spinenet/weights/detect-vfr/ckpt435.pt",
        "context/ckpt16.pt": "https://www.robots.ox.ac.uk/~vgg/research/spinenet/weights/context/ckpt16.pt"
    }
    weights_path = os.path.join(os.path.dirname(__file__),"weights")

    for path, url in weights_dict.items():
        if not os.path.exists(os.path.join(weights_path, os.path.dirname(path))):
            os.makedirs(os.path.join(weights_path, os.path.dirname(path)), exist_ok=True)

    if verbose:
        print("Downloading weights...")
    download_weights_io(weights_path, weights_dict, verbose=verbose, force=force)

def download_example_scan(example_scan_name: str, file_path: Union[os.PathLike, str]) -> None:
    """
    Download & unzip example scan DICOM folders from public server for testing and tutorials.
    The scans are mostly taken from radiopedia.org and are
    attributed in the project README.md.

    Parameters
    ----------
    example_scan_name : str
        The example scan name to download locally from the server. Current options are
        't2_lumbar_scan_1', 't2_lumbar_scan_2', 'stir_whole_spine'
    file_path
        the directory to save the scan name to. For example 't2_lumbar_scan_1' would save to
        file_path/t2_lumbar_scan_1
    """
    scans_path = {
        't2_lumbar_scan_1': 'https://www.robots.ox.ac.uk/~vgg/research/spinenet/example_scans/t2_lumbar_scan_1.zip',
        't2_lumbar_scan_2': 'https://www.robots.ox.ac.uk/~vgg/research/spinenet/example_scans/t2_lumbar_scan_2.zip',
        'stir_whole_spine': 'https://www.robots.ox.ac.uk/~vgg/research/spinenet/example_scans/stir_whole_spine.zip',
        't2_whole_spine': 'https://www.robots.ox.ac.uk/~vgg/research/spinenet/example_scans/t2_whole_spine.zip',
    }
    assert example_scan_name in scans_path.keys(), f"example_scan_name must be one of {scans_path.keys()}"



    scan_url = scans_path[example_scan_name]

    download_io(scan_url, os.path.join(file_path, 'temp_file.zip'))

    # Unzip the file
    with zipfile.ZipFile(os.path.join(file_path,'temp_file.zip'), 'r') as zip_ref:
        zip_ref.extractall(file_path)

    # Remove the temp file
    os.remove(os.path.join(file_path,'temp_file.zip'))


