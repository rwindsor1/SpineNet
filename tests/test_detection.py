"""
SpineNet Integration Test Suite

Tests core functionality including scan loading, vertebrae detection,
and results saving. Designed for CI/CD with efficient resource management.
"""
import os
import pytest
import torch
import numpy as np
from pathlib import Path

import spinenet
from spinenet import SpineNet, download_example_scan
from spinenet.io import load_dicoms_from_folder, save_vert_dicts_to_csv


@pytest.fixture(scope="session")
def setup_weights():
    """Download model weights once per test session."""
    spinenet.download_weights(verbose=True, force=False)


@pytest.fixture(scope="session")
def example_scan_folder(tmp_path_factory, setup_weights):
    """Download example scan data once per session."""
    scan_name = 't2_lumbar_scan_1'
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
