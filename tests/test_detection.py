"""
SpineNet Integration Test Suite

Tests core functionality including scan loading, vertebrae detection,
and results saving. Designed for CI/CD with efficient resource management.
"""
import os
import pytest
import torch
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from pathlib import Path

import spinenet
from spinenet import SpineNet, download_example_scan
from spinenet.io import (
    load_dicoms_from_folder,
    save_vert_dicts_to_csv,
    save_ivd_volumes,
)


@pytest.fixture(scope="session")
def setup_weights():
    """Download model weights once per test session."""
    spinenet.download_weights(verbose=True, force=False)


@pytest.fixture(scope="session", params=['t2_lumbar_scan_1', 't2_lumbar_scan_2'])
def example_scan_folder(request, tmp_path_factory, setup_weights):
    """Download each example scan once per session."""
    scan_name = request.param
    folder = tmp_path_factory.mktemp('example_scans')
    download_example_scan(scan_name, file_path=str(folder))
    return folder / scan_name


@pytest.fixture(scope="session")
def loaded_scan(example_scan_folder):
    """Load scan with metadata overrides for consistent testing."""
    overwrite_dict = {
        'SliceThickness': [2],
        'ImageOrientationPatient': [0, 1, 0, 0, 0, -1]
    }
    scan = load_dicoms_from_folder(
        str(example_scan_folder),
        require_extensions=False,
        metadata_overwrites=overwrite_dict
    )
    return scan


@pytest.fixture(scope="session")
def spinenet_model(setup_weights):
    """Create SpineNet model instance once per session."""
    model = SpineNet(device='cpu', verbose=True, scan_type='lumbar')
    return model


@pytest.fixture(scope="session")
def detection_results(loaded_scan, spinenet_model):
    """Run detection once and cache results for all tests."""
    return spinenet_model.detect_vb(loaded_scan.volume, loaded_scan.pixel_spacing)


class TestScanLoading:
    """Test scan loading and data structure validation."""

    def test_scan_attributes(self, loaded_scan):
        """Test that loaded scan has required attributes."""
        scan = loaded_scan
        assert hasattr(scan, 'volume'), "Scan should have volume attribute"
        assert hasattr(scan, 'pixel_spacing'), "Scan should have pixel_spacing attribute"

    def test_volume_structure(self, loaded_scan):
        """Test volume data structure and dimensionality."""
        volume = loaded_scan.volume
        assert isinstance(volume, np.ndarray), "Volume should be numpy array"
        assert len(volume.shape) == 3, "Volume should be 3D (H x W x D)"
        assert volume.size > 0, "Volume should not be empty"

    def test_pixel_spacing_format(self, loaded_scan):
        """Test pixel spacing data format."""
        spacing = loaded_scan.pixel_spacing
        assert isinstance(spacing, (list, tuple, np.ndarray)), "Pixel spacing should be array-like"
        assert len(spacing) == 2, "Pixel spacing should have 2 elements"
        assert all(isinstance(s, (int, float)) and s > 0 for s in spacing), "Spacing values should be positive numbers"


class TestVertebraeDetection:
    """Test vertebrae detection functionality."""

    def test_detection_output_structure(self, detection_results):
        """Test detection returns properly structured results."""
        assert isinstance(detection_results, list), "Detection should return a list"
        assert len(detection_results) > 0, "Should detect at least one vertebra"

    def test_detection_dictionary_format(self, detection_results):
        """Test each detection result has required fields."""
        for vert_dict in detection_results:
            assert isinstance(vert_dict, dict), "Each detection should be a dictionary"
            assert 'predicted_label' in vert_dict, "Detection should have predicted_label"
            assert 'average_polygon' in vert_dict, "Detection should have average_polygon"

    def test_detection_labels(self, detection_results):
        """Test detection labels are valid strings."""
        labels = [vd['predicted_label'] for vd in detection_results]
        assert len(labels) > 0, "Should have at least one vertebra label"
        assert all(isinstance(l, str) for l in labels), "All labels should be strings"
        assert any(any(c.isalpha() for c in label) for label in labels), "Labels should contain letters"

    def test_polygon_coordinates(self, detection_results, loaded_scan):
        """Test polygon coordinates are within image bounds."""
        h, w, d = loaded_scan.volume.shape
        for vert_dict in detection_results:
            poly = np.array(vert_dict['average_polygon'])
            assert poly.ndim == 2, "Polygon should be 2D array"
            assert poly.shape[1] == 2, "Polygon points should be (x, y) coordinates"
            x_coords, y_coords = poly[:, 0], poly[:, 1]
            assert np.all(x_coords >= 0) and np.all(x_coords < w), "X coordinates should be within image width"
            assert np.all(y_coords >= 0) and np.all(y_coords < h), "Y coordinates should be within image height"


class TestResultsSaving:
    """Test results saving and file I/O functionality."""

    def test_csv_export(self, detection_results, tmp_path):
        """Test saving detection results to CSV file."""
        results_file = tmp_path / "test_vertebrae_detection.csv"
        save_vert_dicts_to_csv(detection_results, str(results_file))
        assert results_file.exists(), "Results file should be created"
        assert results_file.stat().st_size > 0, "Results file should not be empty"

    def test_csv_content_format(self, detection_results, tmp_path):
        """Test CSV file contains expected content structure."""
        results_file = tmp_path / "test_vertebrae_detection.csv"
        save_vert_dicts_to_csv(detection_results, str(results_file))

        # Read and verify CSV content
        content = results_file.read_text()
        lines = content.strip().split('\n')
        assert len(lines) > 1, "CSV should have header and at least one data row"
        assert ',' in lines[0], "CSV should be comma-separated"

    def test_ivd_volume_export(
        self, detection_results, loaded_scan, spinenet_model, tmp_path
    ):
        """Test extracting bigger IVD volumes and saving them."""
        default = spinenet_model.get_ivds_from_vert_dicts(
            detection_results, loaded_scan.volume
        )
        ivd_dicts = spinenet_model.get_ivds_from_vert_dicts(
            detection_results, loaded_scan.volume, output_shape=(12, 128, 256)
        )
        paths = save_ivd_volumes(ivd_dicts, str(tmp_path))
        assert len(paths) == len(ivd_dicts) == len(default) > 0
        for path, ivd_dict, small in zip(paths, ivd_dicts, default):
            assert path.endswith(f"{ivd_dict['level_name']}.npy")
            assert ivd_dict["voxel_to_scan"].shape == (4, 4)
            vol = np.load(path)
            assert vol.shape == (12, 128, 256), "Volume should match shape"
            assert np.isfinite(vol).all() and vol.max() > 0
            np.testing.assert_array_equal(
                ivd_dict["volume"][2:11, 8:120, 16:240], small["volume"]
            )

    def test_ivd_nifti_header(
        self, detection_results, loaded_scan, spinenet_model, tmp_path
    ):
        """Test that the NIfTI header puts each IVD volume on its disc."""
        scan = loaded_scan
        scan_affine = np.diag([*scan.pixel_spacing, scan.slice_thickness, 1.0])
        scan_img = nib.Nifti1Image(scan.volume.astype(np.float32), scan_affine)
        ivd_dicts = spinenet_model.get_ivds_from_vert_dicts(
            detection_results, scan.volume, output_shape=(12, 128, 256)
        )
        paths = save_ivd_volumes(
            ivd_dicts, str(tmp_path), "nii.gz", scan_affine=scan_affine
        )
        assert len(paths) > 0
        for path in paths:
            ivd_img = nib.load(path)
            vol = ivd_img.get_fdata()
            # the scan, resampled onto the grid given by the IVD header
            ref = resample_from_to(
                scan_img, ivd_img, order=1, cval=np.nan
            ).get_fdata()
            # intensities are scaled per disc and clipped; skip the padding
            mask = np.isfinite(ref) & (vol > 0) & (vol < vol.max())
            corr = np.corrcoef(ref[mask], vol[mask])[0, 1]
            assert corr > 0.995, f"{os.path.basename(path)}: {corr:.4f}"


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_workflow(self, loaded_scan, spinenet_model, tmp_path):
        """Test complete workflow from scan to saved results."""
        # Run detection
        vert_dicts = spinenet_model.detect_vb(loaded_scan.volume, loaded_scan.pixel_spacing)

        # Save results
        results_file = tmp_path / "integration_test_results.csv"
        save_vert_dicts_to_csv(vert_dicts, str(results_file))

        # Verify workflow success
        assert len(vert_dicts) > 0, "Should detect vertebrae"
        assert results_file.exists(), "Should save results successfully"
        assert results_file.stat().st_size > 0, "Results file should contain data"
