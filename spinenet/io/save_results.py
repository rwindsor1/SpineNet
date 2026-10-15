import os
from typing import Dict, List, Optional, Tuple
from csv import DictWriter
import numpy as np
import pandas as pd
VertDicts = List[Dict]

def save_vert_dicts_to_csv(vert_dicts: VertDicts, filename: str) -> None:
    '''
    Saves a list of vert_dicts to a file in csv format.

    Parameters
    ----------
    vert_dicts: List[Dict]
        A list of dictionaries with the following keys: 
        'polys', 'average_polygon', 'slice_nos', 'predicted_label'
        Each entry corresponds to a detected vertebral body
    filename: str
        The name of the file to save the vert_dicts to.
    '''

    required_keys = ['polys', 'average_polygon', 'slice_nos', 'predicted_label']
    check_no_keys_missing(vert_dicts, required_keys)

    with open(filename, 'w') as f:
        w = DictWriter(f, fieldnames=required_keys)
        w.writeheader()
        for vert_dict in vert_dicts:
            w.writerow(vert_dict)
    return


def check_no_keys_missing(vert_dicts, required_keys):
    for vert_dict in vert_dicts:
        for key in required_keys:
            if key not in vert_dict:
                raise KeyError(f'{key} is missing from vert_dict')


def get_ivd_nifti_affine(ivd_dict, scan_affine=None):
    '''
    The NIfTI affine of an IVD volume: the identity, overridden with the IVD
    voxel -> scan voxel -> world affine if the 4x4 voxel-to-world
    `scan_affine` of the scan volume is given.
    '''
    if scan_affine is None:
        return np.eye(4)
    return scan_affine @ ivd_dict['voxel_to_scan']


def save_ivd_volumes(
    ivd_dicts: List[Dict],
    out_dir: str,
    file_format: str = 'npy',
    prefix: str = '',
    scan_affine: Optional[np.ndarray] = None,
) -> List[str]:
    '''
    Saves each IVD volume to its own float32 file, without resampling.

    Files are named `<out_dir>/<prefix><level_name>.<file_format>`,
    e.g. `results/scan1_L4-L5.npy`.

    Parameters
    ----------
    ivd_dicts: List[Dict]
        A list of dictionaries with the keys 'volume' and 'level_name', as
        returned by `SpineNet.get_ivds_from_vert_dicts`. Each volume is a
        3D array of shape (slices, height, width), set there with
        `output_shape`, e.g. (12, 128, 256).
    out_dir: str
        The folder to save the volumes to. Created if it does not exist.
    file_format: str, optional
        'npy' (numpy) or 'nii.gz' (NIfTI). NIfTI files store the array in
        the same (slices, height, width) order with an identity affine, so
        the orientation and 1 mm spacing in the header are not anatomical.
        The default is 'npy'.
    prefix: str, optional
        Added to the start of each file name, e.g. the scan name. The
        default is ''.
    scan_affine: np.ndarray, optional
        The 4x4 voxel-to-world affine of the scan volume, e.g. the `affine`
        of a `SpinalScan`. NIfTI files then get the affine
        `scan_affine @ ivd_dict['voxel_to_scan']` instead of the identity
        (see `get_ivd_nifti_affine`), as sform and qform, with mm as units.

    Returns
    -------
    List[str]
        The paths of the saved files, in the order of `ivd_dicts`.
    '''
    check_no_keys_missing(ivd_dicts, ['volume', 'level_name'])
    if file_format not in ('npy', 'nii.gz'):
        raise ValueError(
            f"file_format must be 'npy' or 'nii.gz', got {file_format!r}"
        )
    if file_format == 'nii.gz':
        import nibabel as nib
    names = [f"{prefix}{d['level_name']}" for d in ivd_dicts]
    if len(set(names)) != len(names):
        raise ValueError(f'Duplicate level names would overwrite: {names}')

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ivd_dict, name in zip(ivd_dicts, names):
        volume = np.asarray(ivd_dict['volume'], dtype=np.float32)
        path = os.path.join(out_dir, f'{name}.{file_format}')
        if file_format == 'npy':
            np.save(path, volume)
        else:
            affine = get_ivd_nifti_affine(ivd_dict, scan_affine)
            img = nib.Nifti1Image(volume, affine)
            if scan_affine is not None:
                # the affine is in mm: also store it as qform and set the units
                img.set_qform(affine, code='aligned')
                img.header.set_xyzt_units('mm')
            nib.save(img, path)
        paths.append(path)
    return paths
