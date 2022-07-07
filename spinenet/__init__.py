import os
from typing import List, Tuple, Dict, Union, Any
import zipfile

from .io import download_weights as download_weights_io, download as download_io    
from .main import SpineNet
from .utils import *


def download_weights(verbose: bool = False, force: bool = False) -> None:
    """
    Download the weights for the SpineNet models.

    Parameters
    ----------
    verbose : bool, optional
        Whether to print out debug information. The default is False.
    force: bool, optional
        Whether to force download weights even if the weights already exist. The default is False.
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
    assert example_scan_name in ['t2_lumbar_scan_1', 't2_lumbar_scan_2'], "example_scan_name must be one of ['t2_lumbar_scan_1']"

    scans_path = {
        't2_lumbar_scan_1': 'https://www.robots.ox.ac.uk/~vgg/research/spinenet/example_scans/t2_lumbar_scan_1.zip',
        't2_lumbar_scan_2': 'https://www.robots.ox.ac.uk/~vgg/research/spinenet/example_scans/t2_lumbar_scan_2.zip',
    }


    scan_url = scans_path[example_scan_name]

    download_io(scan_url, os.path.join(file_path, 'temp_file.zip'))

    # Unzip the file
    with zipfile.ZipFile(os.path.join(file_path,'temp_file.zip'), 'r') as zip_ref:
        zip_ref.extractall(file_path)

    # Remove the temp file
    os.remove(os.path.join(file_path,'temp_file.zip'))


