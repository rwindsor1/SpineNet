import logging
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
        affine: Union[np.array, None] = None,
        image_positions: Union[np.array, None] = None,
        image_orientation: Union[np.array, None] = None,
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
        affine : np.array, optional
            The 4x4 affine from a (height, width, slice) index of the volume to
            world coordinates in mm (RAS+, as in NIfTI). If not given, it only
            holds the pixel spacing and slice thickness, without the position
            or orientation of the scan.
        image_positions : np.array, optional
            The DICOM ImagePositionPatient of each slice, of shape (slices, 3).
        image_orientation : np.array, optional
            The DICOM ImageOrientationPatient of the slices (6 values). With
            the image positions, this gives the affine (see `get_dicom_affine`).
        """
        self.volume = volume
        self.pixel_spacing = pixel_spacing
        self.slice_thickness = slice_thickness
        self.affine = affine
        self.image_positions = image_positions
        self.image_orientation = image_orientation

    @property
    def affine(self) -> np.array:
        if self._affine is None:
            return np.diag([*self.pixel_spacing, self.slice_thickness, 1.0])
        return self._affine

    @affine.setter
    def affine(self, affine: Union[np.array, None]) -> None:
        self._affine = affine


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
            # SliceThickness is preferred: SpacingBetweenSlices only stands in
            # for it if it is missing or empty
            if not dicom_file.get("SliceThickness") and dicom_file.get(
                "SpacingBetweenSlices"
            ):
                dicom_file.SliceThickness = abs(float(dicom_file.SpacingBetweenSlices))

        # check relevant tags are present and slices are all sagittal
        sagittal_files = []
        for dicom_idx, dicom_file in enumerate(dicom_files):
            missing_tags = check_missing_tags(dicom_file)
            if len(missing_tags) > 0:
                raise ValueError(
                    f"Missing tags in file {paths[dicom_idx]}: {missing_tags}"
                )
            is_sagittal = is_sagittal_dicom_slice(dicom_file)
            if is_sagittal:
                sagittal_files.append(dicom_file)
            else:
                #raise ValueError(
                logging.warning( f"File at {paths[dicom_idx]} is not a sagittal dicom slice")
        dicom_files = sagittal_files
        # sort slices by sagittal position
        dicom_files = sorted(
            dicom_files, key=lambda dicom_file: dicom_file.InstanceNumber
        )


        pixel_spacing = np.mean(
            [np.array(dicom_file.PixelSpacing) for dicom_file in dicom_files]
        , axis=0)
        slice_thickness = np.mean(
            [np.array(dicom_file.SliceThickness) for dicom_file in dicom_files]
        )
        volume = np.stack(
            [np.array(dicom_file.pixel_array) for dicom_file in dicom_files], axis=-1
        )

        # where the slices are in the scanner, kept to give the affine
        image_positions = [f.get("ImagePositionPatient") for f in dicom_files]
        if all(np.size(position) == 3 for position in image_positions):
            image_positions = np.array(image_positions, dtype=float)
        else:
            image_positions = None
        image_orientation = np.array(
            dicom_files[0].ImageOrientationPatient, dtype=float
        )
        orientations = [f.ImageOrientationPatient for f in dicom_files]
        if not np.allclose(np.array(orientations, dtype=float), image_orientation, atol=1e-3):
            logging.warning(
                "The DICOM slices do not all have the same ImageOrientationPatient: "
                "the affine of the scan uses the one of the first slice"
            )

        affine = get_dicom_affine(
            image_positions, image_orientation, pixel_spacing, slice_thickness
        )
        if affine is None:
            logging.warning(
                "ImagePositionPatient or ImageOrientationPatient is missing or "
                "malformed in the DICOM slices: the affine of the scan only holds "
                "the pixel spacing and slice thickness, not its position or orientation"
            )

        return SpinalScan(
            volume=volume, pixel_spacing=pixel_spacing, slice_thickness=slice_thickness,
            affine=affine,
            image_positions=image_positions,
            image_orientation=image_orientation,
        )


def get_dicom_affine(
    image_positions: Union[np.array, None],
    image_orientation: Union[np.array, list],
    pixel_spacing: Union[np.array, list],
    slice_thickness: Union[float, int] = 1.0,
) -> Union[np.array, None]:
    '''
    The affine from a (row, column, slice) index of the stacked slices to
    world coordinates in mm (RAS+, as in NIfTI), from the position and
    orientation of the slices in the scanner.

    Parameters
    ----------
    image_positions : Union[np.array, None]
        the ImagePositionPatient of each slice, in the order they are stacked
        in, of shape (slices, 3)
    image_orientation : Union[np.array, list]
        the ImageOrientationPatient of the slices (6 values)
    pixel_spacing : Union[np.array, list]
        the pixel spacing of the slices in mm, in order rows, columns
    slice_thickness : Union[float, int]
        the distance between slices in mm, only used if the positions of the
        slices do not give it

    Returns
    -------
    Union[np.array, None]
        the 4x4 affine, or None if the image positions or the 6 values of
        the image orientation are missing
    '''
    if image_positions is None or np.size(image_orientation) != 6:
        return None
    positions = np.array(image_positions, dtype=float)
    # directions of increasing column and row index (DICOM stores them so)
    col_direction, row_direction = np.array(
        image_orientation, dtype=float
    ).reshape(2, 3)
    slice_step = (positions[-1] - positions[0]) / max(len(positions) - 1, 1)
    if not slice_step.any():
        slice_step = np.cross(col_direction, row_direction) * slice_thickness
    # an affine only holds slices evenly spaced along a line
    expected = positions[0] + np.outer(np.arange(len(positions)), slice_step)
    if not np.allclose(positions, expected, atol=0.1):
        logging.warning(
            "The DICOM slices are not evenly spaced in the order of their "
            "InstanceNumber: the affine uses the mean step from the first to "
            "the last slice and is off by up to "
            f"{np.abs(positions - expected).max():.2f} mm for the others"
        )
    affine = np.eye(4)
    affine[:3, 0] = row_direction * pixel_spacing[0]
    affine[:3, 1] = col_direction * pixel_spacing[1]
    affine[:3, 2] = slice_step
    affine[:3, 3] = positions[0]
    # DICOM coordinates are left, posterior, superior (LPS+)
    affine[:2] *= -1
    return affine


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
