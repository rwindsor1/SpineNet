"""Tests for extracting IVD volumes at a configurable shape and saving them
(no weights needed)."""

import os

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian

from spinenet import SpineNet
from spinenet.io import (
    SpinalScan,
    get_dicom_affine,
    load_dicoms,
    load_dicoms_from_folder,
    save_ivd_volumes,
)
from spinenet.utils import extract_volumes, gen_utils


@pytest.fixture
def ivd_dicts():
    """Synthetic IVDs shaped like SpineNet.get_ivds_from_vert_dicts output."""
    rng = np.random.default_rng(0)
    return [
        {"volume": rng.random((12, 128, 256)), "level_name": name}
        for name in ["L5-S1", "L4-L5", "L3-L4"]
    ]


def test_save_keeps_volume(ivd_dicts, tmp_path):
    paths = save_ivd_volumes(ivd_dicts, str(tmp_path / "ivds"))
    assert [os.path.basename(p) for p in paths] == [
        "L5-S1.npy",
        "L4-L5.npy",
        "L3-L4.npy",
    ]
    for path, ivd_dict in zip(paths, ivd_dicts):
        vol = np.load(path)
        assert vol.dtype == np.float32
        np.testing.assert_array_equal(
            vol, ivd_dict["volume"].astype(np.float32)
        )


def test_nifti_and_prefix(ivd_dicts, tmp_path):
    paths = save_ivd_volumes(
        ivd_dicts, str(tmp_path), file_format="nii.gz", prefix="scan1_"
    )
    assert paths[0].endswith("scan1_L5-S1.nii.gz")
    assert nib.load(paths[0]).get_fdata().shape == (12, 128, 256)


def test_invalid_inputs(ivd_dicts, tmp_path):
    with pytest.raises(KeyError):
        save_ivd_volumes([{"volume": np.zeros((9, 112, 224))}], str(tmp_path))
    with pytest.raises(ValueError):
        save_ivd_volumes(ivd_dicts, str(tmp_path), file_format="png")
    with pytest.raises(ValueError):
        save_ivd_volumes(ivd_dicts * 2, str(tmp_path))
    # a scan affine needs the 'voxel_to_scan' of each IVD
    with pytest.raises(KeyError):
        save_ivd_volumes(
            ivd_dicts, str(tmp_path), "nii.gz", scan_affine=np.eye(4)
        )


def synthetic_vert_dicts(cx, bottom_cy, n_slices=12, tilt=0):
    """Four 80x60 px vertebrae stacked up from bottom_cy, axis-aligned or
    rotated by tilt degrees about the lowest one."""
    cos, sin = np.cos(np.deg2rad(tilt)), np.sin(np.deg2rad(tilt))
    vert_dicts = []
    for i, label in enumerate(["S1", "L5", "L4", "L3"]):
        cy = bottom_cy - i * 76
        poly = [
            [cx + 40, cy - 30],
            [cx + 40, cy + 30],
            [cx - 40, cy + 30],
            [cx - 40, cy - 30],
        ]
        if tilt:
            poly = [
                [
                    cx + (x - cx) * cos - (y - bottom_cy) * sin,
                    bottom_cy + (x - cx) * sin + (y - bottom_cy) * cos,
                ]
                for x, y in poly
            ]
        vert_dicts.append(
            {
                "predicted_label": label,
                "average_polygon": poly,
                "polys": [poly] * n_slices,
                "slice_nos": list(range(n_slices)),
            }
        )
    return vert_dicts


def extract(cx=256, bottom_cy=300, n_slices=12, **kwargs):
    scan = np.random.default_rng(0).uniform(100, 200, (512, 512, n_slices))
    spnt = SpineNet.__new__(SpineNet)  # IVD extraction needs no weights
    return spnt.get_ivds_from_vert_dicts(
        synthetic_vert_dicts(cx, bottom_cy, n_slices), scan, **kwargs
    )


# centre of scan, lowest disc near the bottom edge, discs near the right edge
@pytest.mark.parametrize("cx, bottom_cy", [(256, 300), (256, 490), (440, 300)])
def test_bigger_crop_contains_default(tmp_path, cx, bottom_cy):
    default = extract(cx, bottom_cy)
    big = extract(cx, bottom_cy, output_shape=(12, 128, 256))
    assert [d["level_name"] for d in big] == ["L5-S1", "L4-L5", "L3-L4"]
    for small_dict, big_dict in zip(default, big):
        assert small_dict["volume"].shape == (9, 112, 224)
        assert big_dict["volume"].shape == (12, 128, 256)
        # same scale and slices, just more context around the default crop
        np.testing.assert_array_equal(
            big_dict["volume"][2:11, 8:120, 16:240], small_dict["volume"]
        )
    paths = save_ivd_volumes(big, str(tmp_path))
    assert all(np.load(p).shape == (12, 128, 256) for p in paths)


def test_only_real_slices():
    # 6 scan slices into 12 output slices: 6 real ones, the rest zero padded
    for ivd_dict in extract(n_slices=6, output_shape=(12, 128, 256)):
        filled = ivd_dict["volume"].reshape(12, -1).any(axis=1)
        assert filled.sum() == 6


def test_crop_bigger_than_patch():
    for ivd_dict in extract(output_shape=(15, 256, 400)):
        assert ivd_dict["volume"].shape == (15, 256, 400)
        assert np.isfinite(ivd_dict["volume"]).all()


def test_invalid_output_shape_and_grading_guard():
    with pytest.raises(ValueError):
        extract(output_shape=(128, 256))
    with pytest.raises(ValueError):
        extract(output_shape=(0, 128, 256))
    big = extract(output_shape=(12, 128, 256))
    with pytest.raises(ValueError, match="9, 112, 224"):
        SpineNet.__new__(SpineNet).grade_ivds(big)


@pytest.mark.parametrize("tilt", [0, 15, -25])
@pytest.mark.parametrize(
    "output_shape", [(9, 112, 224), (12, 128, 256), (9, 200, 330)]
)
def test_voxel_to_scan_follows_extraction(monkeypatch, tilt, output_shape):
    # fixed vertebra median, so that volume == scan / 1000 without clipping
    monkeypatch.setattr(
        gen_utils,
        "get_vbs_intensity",
        lambda volume, all_vb_x, *args: [np.full(1, 250.0)] * 4,
    )
    spnt = SpineNet.__new__(SpineNet)
    vert_dicts = synthetic_vert_dicts(256, 330, n_slices=14, tilt=tilt)
    # (slice, row, col, 1) of every voxel of an IVD volume
    index = np.stack([*np.indices(output_shape), np.ones(output_shape)], -1)
    # scans that hold their own row, column and slice index
    for axis, coordinate in enumerate(np.indices((448, 512, 14))):
        ivd_dicts = spnt.get_ivds_from_vert_dicts(
            vert_dicts, 200.0 + coordinate, output_shape=output_shape
        )
        assert len(ivd_dicts) == 3
        for ivd_dict in ivd_dicts:
            in_scan = (index @ ivd_dict["voxel_to_scan"].T)[..., axis]
            error = ivd_dict["volume"] * 1000 - 200 - in_scan
            # cv2 cubic interpolation is only exact to ~0.1 px on a ramp
            assert np.abs(error).max() < 0.2
            assert abs(error.mean()) < 0.01


@pytest.mark.parametrize("step", [(1, 2), (2, 1)])
@pytest.mark.parametrize("tilt", [0, 20, -20])
def test_non_square_pixels(monkeypatch, step, tilt):
    # fixed vertebra median, so that volume == scan / 1000 without clipping
    monkeypatch.setattr(
        gen_utils,
        "get_vbs_intensity",
        lambda volume, all_vb_x, *args: [np.full(1, 250.0)] * 4,
    )
    spnt = SpineNet.__new__(SpineNet)
    # big enough to keep the volumes around the vertebrae inside the scan
    rows, cols, _ = np.indices((640, 512, 14))
    scan = 200 + 50 * np.sin(rows / 12) * np.cos(cols / 16)
    vert_dicts = synthetic_vert_dicts(256, 420, n_slices=14, tilt=tilt)
    square = spnt.get_ivds_from_vert_dicts(vert_dicts, scan)
    square_verts = extract_volumes(scan, [dict(v) for v in vert_dicts])
    # the same scan with every second row or column, at twice the spacing
    scan = scan[:: step[0], :: step[1]]
    pixel_spacing = 0.5 * np.array(step)
    for vert_dict in vert_dicts:
        for key in ["average_polygon", "polys"]:
            vert_dict[key] = np.array(vert_dict[key]) / step[::-1]

    ivd_dicts = spnt.get_ivds_from_vert_dicts(
        vert_dicts, scan, pixel_spacing=pixel_spacing
    )
    for ivd_dict, square_dict in zip(ivd_dicts, square):
        # the same crop as on square pixels
        error = (ivd_dict["volume"] - square_dict["volume"]) * 1000
        assert np.abs(error).max() < 1
        to_scan = ivd_dict["voxel_to_scan"]
        np.testing.assert_allclose(
            np.diag([*step, 1, 1]) @ to_scan, square_dict["voxel_to_scan"]
        )
        # in mm: square voxels along orthogonal axes
        row_mm, col_mm = (np.diag([*pixel_spacing, 4, 1]) @ to_scan)[:3, 1:3].T
        assert abs(row_mm @ col_mm) < 1e-9
        np.testing.assert_allclose(
            np.linalg.norm(row_mm), np.linalg.norm(col_mm), rtol=0.01
        )
    # without the pixel spacing, the crop is another one
    distorted = spnt.get_ivds_from_vert_dicts(vert_dicts, scan)
    assert np.abs(distorted[0]["volume"] - square[0]["volume"]).max() > 0.02

    # same for the volumes around the vertebrae
    verts = extract_volumes(scan, vert_dicts, pixel_spacing=pixel_spacing)
    for vert_dict, square_dict in zip(verts, square_verts):
        error = vert_dict["volume"] - square_dict["volume"]
        assert np.abs(error).max() < 0.01


def test_nifti_affine(tmp_path):
    ivd_dicts = extract(output_shape=(12, 128, 256))
    # rows -> inferior, columns -> posterior, slices -> right, in mm
    scan_affine = np.array(
        [[0, 0, 4, -20], [0, -0.5, 0, 90], [-0.5, 0, 0, 120], [0, 0, 0, 1.0]]
    )
    paths = save_ivd_volumes(
        ivd_dicts, str(tmp_path), "nii.gz", scan_affine=scan_affine
    )
    for path, ivd_dict in zip(paths, ivd_dicts):
        img = nib.load(path)
        affine = scan_affine @ ivd_dict["voxel_to_scan"]
        np.testing.assert_allclose(img.affine, affine, atol=1e-4)
        assert nib.aff2axcodes(img.affine) == ("R", "I", "P")
        # 4 mm slices; the 80 px wide discs are scaled to 113.5 px, at 0.5 mm
        np.testing.assert_allclose(
            img.header.get_zooms(),
            (4, 0.5 * 80 / 113.5, 0.5 * 80 / 113.5),
            rtol=0.01,
        )
        # the same affine as sform and qform, in mm
        np.testing.assert_allclose(img.header.get_sform(), affine, atol=1e-4)
        np.testing.assert_allclose(img.header.get_qform(), affine, atol=1e-3)
        assert img.header["sform_code"] == img.header["qform_code"] == 2
        assert img.header.get_xyzt_units()[0] == "mm"
    # without a scan affine, the header keeps the identity
    path = save_ivd_volumes(ivd_dicts[:1], str(tmp_path / "vox"), "nii.gz")[0]
    np.testing.assert_array_equal(nib.load(path).affine, np.eye(4))
    assert nib.load(path).header.get_xyzt_units()[0] == "unknown"



def write_dicom_slices(
    folder, volume, pixel_spacing, orientation, positions, slice_spacing=None,
    slice_thickness=3,
):
    """Write a (rows, columns, slices) volume to one DICOM file per slice."""
    os.makedirs(folder)
    for i in range(volume.shape[-1]):
        path = os.path.join(folder, f"{i}.dcm")
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ImplicitVRLittleEndian
        ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.Rows, ds.Columns = volume.shape[:2]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = volume[..., i].astype(np.uint16).tobytes()
        ds.InstanceNumber = i + 1
        ds.PixelSpacing = list(pixel_spacing)
        if slice_thickness is not None:
            ds.SliceThickness = slice_thickness
        ds.ImageOrientationPatient = list(orientation)
        if positions is not None:
            ds.ImagePositionPatient = list(positions[i])
        if slice_spacing is not None:
            ds.SpacingBetweenSlices = slice_spacing
        ds.save_as(path)


def test_dicom_scan_affine(tmp_path, caplog):
    scan = np.random.default_rng(0).uniform(100, 200, (512, 512, 12))
    pixel_spacing = [0.75, 0.5]
    # sagittal slices, rotated in-plane by 16 degrees: the direction of
    # increasing column, then of increasing row index, in DICOM LPS+ mm
    along_row, along_col = np.array([[0, 0.96, 0.28], [0, 0.28, -0.96]])
    # 3 mm slices with a 0.25 mm gap, stacked from right to left
    positions = [-20.5, -90.25, 120.75] + np.outer(range(12), [3.25, 0, 0])
    orientation = [*along_row, *along_col]
    write_dicom_slices(
        tmp_path / "dicoms", scan, pixel_spacing, orientation, positions
    )
    loaded = load_dicoms_from_folder(str(tmp_path / "dicoms"))
    assert "ImagePositionPatient" not in caplog.text

    def to_world(index):
        row, col, sagittal_slice = np.moveaxis(index, -1, 0)[:3, ..., None]
        lps = (
            positions[0]
            + row * pixel_spacing[0] * along_col
            + col * pixel_spacing[1] * along_row
            + sagittal_slice * [3.25, 0, 0]
        )
        return lps * [-1, -1, 1]  # NIfTI is RAS+

    index = np.array([[0, 0, 0, 1], [511, 0, 0, 1], [0, 511, 0, 1], [7, 9, 11, 1]])
    np.testing.assert_allclose(
        (index @ loaded.affine.T)[:, :3], to_world(index), atol=1e-9
    )
    assert nib.aff2axcodes(loaded.affine) == ("I", "P", "L")
    np.testing.assert_allclose(
        nib.affines.voxel_sizes(loaded.affine), [0.75, 0.5, 3.25]
    )
    # the scan keeps the DICOM tags that give the affine
    np.testing.assert_array_equal(loaded.image_positions, positions)
    np.testing.assert_array_equal(loaded.image_orientation, orientation)
    np.testing.assert_array_equal(
        get_dicom_affine(
            loaded.image_positions,
            loaded.image_orientation,
            loaded.pixel_spacing,
            loaded.slice_thickness,
        ),
        loaded.affine,
    )
    assert "evenly spaced" not in caplog.text
    assert "same ImageOrientationPatient" not in caplog.text

    # non-sagittal slices (e.g. localisers) are all left out, even in a row
    paths = [str(tmp_path / "dicoms" / f"{i}.dcm") for i in range(12)]
    for name, instance in [("axial_a", 13), ("axial_b", 14)]:
        ds = pydicom.dcmread(paths[0])
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.InstanceNumber = instance
        ds.save_as(tmp_path / f"{name}.dcm")
    axial = [str(tmp_path / "axial_a.dcm"), str(tmp_path / "axial_b.dcm")]
    with_axial = load_dicoms(paths[:5] + axial + paths[5:])
    assert with_axial.volume.shape == (512, 512, 12)
    np.testing.assert_array_equal(with_axial.affine, loaded.affine)
    assert caplog.text.count("is not a sagittal dicom slice") == 2

    # the affine cannot hold unevenly spaced slices: warn by how much it is off
    caplog.clear()
    uneven = positions.copy()
    uneven[5] += [1, 0, 0]
    write_dicom_slices(tmp_path / "uneven", scan, pixel_spacing, orientation, uneven)
    load_dicoms_from_folder(str(tmp_path / "uneven"))
    assert "not evenly spaced" in caplog.text and "1.00 mm" in caplog.text

    # the NIfTI header puts every IVD voxel at its position in the scanner
    spnt = SpineNet.__new__(SpineNet)
    ivd_dicts = spnt.get_ivds_from_vert_dicts(
        synthetic_vert_dicts(256, 300),
        loaded.volume,
        output_shape=(12, 128, 256),
        pixel_spacing=loaded.pixel_spacing,
    )
    paths = save_ivd_volumes(
        ivd_dicts, str(tmp_path), "nii.gz", scan_affine=loaded.affine
    )
    ivd_index = np.stack([*np.indices((12, 128, 256)), np.ones((12, 128, 256))], -1)
    for path, ivd_dict in zip(paths, ivd_dicts):
        img = nib.load(path)
        np.testing.assert_allclose(
            (ivd_index @ img.affine.T)[..., :3],
            to_world(ivd_index @ ivd_dict["voxel_to_scan"].T),
            atol=1e-3,
        )
        zooms = img.header.get_zooms()
        np.testing.assert_allclose(zooms[0], 3.25, rtol=1e-5)
        np.testing.assert_allclose(zooms[1], zooms[2], rtol=0.01)

    # without slice positions: the voxel sizes only
    write_dicom_slices(
        tmp_path / "no_positions", scan, pixel_spacing, orientation, None
    )
    loaded = load_dicoms_from_folder(str(tmp_path / "no_positions"))
    assert "ImagePositionPatient or ImageOrientationPatient is missing" in caplog.text
    np.testing.assert_array_equal(loaded.affine, np.diag([0.75, 0.5, 3, 1]))
    assert loaded.image_positions is None
    np.testing.assert_array_equal(loaded.image_orientation, orientation)
    np.testing.assert_array_equal(
        SpinalScan(scan, pixel_spacing, 3).affine, loaded.affine
    )
    assert SpinalScan(scan, pixel_spacing, 3, affine=np.eye(4)).affine[0, 0] == 1
    # as before, the spacings are not checked when the scan is created
    assert SpinalScan(scan, None, None).image_positions is None
    assert get_dicom_affine(None, None, pixel_spacing) is None
    assert get_dicom_affine(positions, [0, 1, 0], pixel_spacing) is None

    # SliceThickness is preferred to SpacingBetweenSlices
    write_dicom_slices(
        tmp_path / "both", scan, pixel_spacing, orientation, None, 3.25
    )
    loaded = load_dicoms_from_folder(str(tmp_path / "both"))
    assert loaded.slice_thickness == 3
    np.testing.assert_array_equal(loaded.affine, np.diag([0.75, 0.5, 3, 1]))
    # SpacingBetweenSlices only stands in for a missing SliceThickness
    write_dicom_slices(
        tmp_path / "spacing", scan, pixel_spacing, orientation, None, 3.25, None
    )
    loaded = load_dicoms_from_folder(str(tmp_path / "spacing"))
    assert loaded.slice_thickness == 3.25
    np.testing.assert_array_equal(loaded.affine, np.diag([0.75, 0.5, 3.25, 1]))
    # with neither, the scan is refused as before
    write_dicom_slices(
        tmp_path / "neither", scan, pixel_spacing, orientation, None, None, None
    )
    with pytest.raises(ValueError, match="SliceThickness"):
        load_dicoms_from_folder(str(tmp_path / "neither"))
