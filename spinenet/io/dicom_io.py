import os
from typing import List, Tuple, Dict, Union, Any
import glob
import numpy as np
from pydicom import dcmread, FileDataset, DataElement
from pydicom.tag import Tag


class SpinalScan:
    def __init__(
        self,
        volume: np.array,
        pixel_spacing: Union[np.array, list],
        slice_thickness: Union[float, int],
    ) -> None:
        """
        Initialize general class for a spinal scan to be used be SpineNet.

        Parameters
        ----------
        volume : np.array
            The volume of the scan voxels (of the orientation height x width x sagittal slices).
        pixel_spacing : Union[np.array, list]
            The pixel spacing of the scan slices in mm. The order is height, width.
        slice_thickness : Union[float, int]
            The distance between consecutive slices in mm.
        """
        self.volume = volume
        self.pixel_spacing = pixel_spacing
        self.slice_thickness = slice_thickness


def load_dicoms(
        paths: List[Union[os.PathLike, str, bytes]],
        require_extensions: bool = True,
        metadata_overwrites: dict = {},
    ) -> SpinalScan:
        '''
        Generate SpinalScan from paths to each DICOM slice,
        with checks such as ensure that the correct tags are present
        and that scan is sagittal

        Parameters
        ----------
        paths : List[Union[os.PathLike, str, bytes]]
            list of paths to DICOM files
        require_extensions : bool
            flag to require that all DICOM files in the `paths` list have the same `.dcm` extension
        metadata_overwrites : dict
            dictionary of metadata to overwrite in the scan. This can be PixelSpacing, SliceThickness and ImageOrientationPatient (which should be sagittal)

        Returns
        -------
        SpinalScan
            Object representing scan from the DICOM files
        '''

        if require_extensions:
            assert all(
                [".dcm" == path[-4:] for path in paths]
            ), "All paths must have .dcm extension. To ignore extension, set require_extensions=False"

        dicom_files = [dcmread(path) for path in paths]

        for idx, dicom_file in enumerate(dicom_files):
            dicom_files[idx] = overwrite_tags(dicom_file, metadata_overwrites)

        # check relevant tags are present and slices are all sagittal
        for dicom_idx, dicom_file in enumerate(dicom_files):
            missing_tags = check_missing_tags(dicom_file)
            if len(missing_tags) > 0:
                raise ValueError(
                    f"Missing tags in file {paths[dicom_idx]}: {missing_tags}"
                )
            is_sagittal = is_sagittal_dicom_slice(dicom_file)
            if not is_sagittal:
                raise ValueError(
                    f"File at {paths[dicom_idx]} is not a sagittal dicom slice"
                )
        # sort slices by sagittal position
        dicom_files = sorted(
            dicom_files, key=lambda dicom_file: dicom_file.InstanceNumber
        )

        pixel_spacing = np.mean(
            [np.array(dicom_file.PixelSpacing) for dicom_file in dicom_files]
        )
        slice_thickness = np.mean(
            [np.array(dicom_file.SliceThickness) for dicom_file in dicom_files]
        )
        volume = np.stack(
            [np.array(dicom_file.pixel_array) for dicom_file in dicom_files], axis=-1
        )

        return SpinalScan(
            volume=volume, pixel_spacing=pixel_spacing, slice_thickness=slice_thickness
        )


def is_sagittal_dicom_slice(dicom_file: FileDataset) -> bool:
    '''
    Check if a dicom slice is sagittal.

    Parameters
    ----------
    dicom_file : FileDataset
        dicom file to check

    Returns
    -------
    bool
        True if sagittal, False otherwise
    '''
    if Tag("ImageOrientationPatient") in dicom_file:
        image_orientation = np.array(dicom_file.ImageOrientationPatient).round()
        if (image_orientation[[0, 3]] == [0, 0]).all():
            return True
        else:
            return False
    else:
        raise ValueError("ImageOrientationPatient metadata not found in dicom file")


def overwrite_tags(dicom_file: FileDataset, metadata_overwrites: dict) -> FileDataset:
    '''
    Overwrite tags in a dicom file. Currently only overwrites PixelSpacing, SliceThickness and ImageOrientationPatient.

    Parameters
    ----------
    dicom_file : FileDataset
        dicom file to overwrite values in
    metadata_overwrites : dict
        dictionary of metadata to overwrite in the scan. This can be PixelSpacing, SliceThickness and ImageOrientationPatient (which should be sagittal)

    Returns
    -------
    FileDataset
        dicom file with overwritten metadata
    '''

    possible_overwrites = {
        "PixelSpacing": "DS",
        "SliceThickness": "DS",
        "ImageOrientationPatient": "DS",
    }

    for tag, value in metadata_overwrites.items():
        if tag not in possible_overwrites:
            raise NotImplementedError(f"Overwriting tag {tag} is not supported")
        else:
            if Tag(tag) in dicom_file:
                dicom_file[Tag(tag)] = DataElement(
                    Tag(tag), possible_overwrites[tag], value
                )
            else:
                dicom_file.add_new(Tag(tag), possible_overwrites[tag], value)
    return dicom_file


def check_missing_tags(dicom_file: FileDataset) -> List[str]:
    '''
    Find which tags are missing in a dicom file from PixelData, PixelSpacing, SliceThickness and ImageOrientationPatient (all are required by SpineNet).

    Parameters
    ----------
    dicom_file : FileDataset
        dicom file to check

    Returns
    -------
    List[str]
        list of missing tags
    '''

    required_tags = ["PixelData", "PixelSpacing", "SliceThickness", "InstanceNumber"]
    missing_tags = [
        tag_name for tag_name in required_tags if Tag(tag_name) not in dicom_file
    ]
    return missing_tags


def is_dicom_file(path: Union[os.PathLike, str, bytes]) -> bool:
    '''
    Check if a file is a dicom file.
    Parameters
    ----------
    path : Union[os.PathLike, str, bytes]
        path to file to check
    Returns
    -------
    bool
        True if dicom file, False otherwise
    '''
    try:
        dcmread(path)
        return True
    except:
        return False


def load_dicoms_from_folder(
    path: Union[os.PathLike, str, bytes],
    require_extensions: bool = True,
    metadata_overwrites: dict = {},
) -> SpinalScan:
    '''
    Load a DICOM scan from a folder containing each of the slices.

    Parameters
    ----------
    path : Union[os.PathLike, str, bytes]
        path to folder containing dicom slices
    require_extensions : bool
        if True, requires all dicom files in the folder must have the extension .dcm.

    Returns
    -------
    SpinalScan
        Object representing scan from the DICOM files
    '''
    slices = [f for f in glob.glob(os.path.join(path, "*")) if is_dicom_file(f)]

    return load_dicoms(slices, require_extensions, metadata_overwrites)
