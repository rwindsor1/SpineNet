import sys, os, glob
import torch
import shutil
import json
from scipy.io import loadmat
import numpy as np
import zipfile
import pydicom


def load_dicoms(temp_path: str, verbose: bool):
    """loads in dicom file saved in temp dir"""
    existing_temp_files = glob.glob(os.path.join(temp_path, "**"), recursive=True)
    dicoms = []
    dicom_names = []
    for existing_temp_file in existing_temp_files:
        try:
            dicom = pydicom.dcmread(existing_temp_file)
            dicoms.append(dicom)
            dicom_names.append(existing_temp_file)
            if verbose:
                print(f"Found DICOM at {existing_temp_file}!")
        except Exception as identifier:
            if verbose:
                print(identifier)
            continue

    if len(dicoms) == 0:
        print("Could not find any dicoms in zipped folder")

    if len(dicoms) == 1:
        scan = dicoms[0].pixel_array
        assert scan.ndim == 3, "Please use 3d scans rather than 2d slices"

    if len(dicoms) > 0:
        assert all(
            [dicom.StudyInstanceUID == dicoms[0].StudyInstanceUID for dicom in dicoms]
        ), "All slices should come from the same study; study UID differs between these DICOMS"
        assert all(
            [dicom.pixel_array.ndim == 2 for dicom in dicoms]
        ), "Each slice should be 2 dimensional"

        if dicoms[0].__contains__([0x20, 0x0032]):
            # check if dicom header has image position patient, left to right, and if so use to order slices
            dicoms.sort(key=lambda s: float(s[0x20, 0x0032].value[0]), reverse=True)
        elif dicoms[0].__contains__([0x20, 0x1041]):
            # else, check if dicom header has slice locations and if so use to order slices
            dicoms.sort(key=lambda s: float(s[0x20, 0x1041].value))
        elif dicoms[0].__contains__([0x20, 0x13]):
            # else, check if dicom header has slice instance numbers as if so use to order slices
            dicoms.sort(key=lambda s: int(s[0x20, 0x13].value))
        else:
            if verbose:
                print("Warning, could not find key to sort slices")
        scan = np.stack([dicom.pixel_array for dicom in dicoms], axis=-1)
    pixel_spacing = dicoms[0].PixelSpacing
    scan_dict = {"scan": scan, "pixel_spacing": pixel_spacing}

    return scan_dict


def extract_zipped_file(zip_path: str, temp_path: str, verbose: bool):
    """
    Extracts all files in zip file to the path specified by temp_path
    """
    assert (
        "temp" in temp_path
    ), "The temp path does not have the substring 'temp' in. Is this the right dir?"

    clear_directory(temp_path, verbose)
    # extract files to temp dir

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_path)


def clear_directory(temp_path: str, verbose: bool):
    # delete files from temp dir
    existing_temp_files = glob.glob(os.path.join(temp_path, "*"), recursive=True)
    for existing_temp_file in existing_temp_files:
        if verbose:
            print(f"Deleting {existing_temp_file} from temp dir")
        if os.path.isfile(existing_temp_file):
            os.remove(existing_temp_file)
        else:
            shutil.rmtree(existing_temp_file)


def export_vert_dicts_to_json(vert_dicts, out_path):
    """
    Exports vert_dict to json file
    """

    out_vert_dicts = []
    for vert_dict in vert_dicts:
        out_vert_dict = {}
        for key in ["polys", "average_polygon", "slice_nos", "predicted_label"]:
            if isinstance(vert_dict[key], np.ndarray) or isinstance(
                vert_dict[key], torch.Tensor
            ):
                out_vert_dict[key] = vert_dict[key].tolist()
            else:
                out_vert_dict[key] = vert_dict[key]
        out_vert_dicts.append(out_vert_dict)

    with open(out_path, "w") as f:
        json.dump(out_vert_dicts, f)
