#!/usr/bin/env python3
"""
Quick test script for SpineNet
Can be used to verify installation without running Jupyter
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import os
import spinenet
from spinenet import SpineNet, download_example_scan
from spinenet.io import load_dicoms_from_folder

def main():
    print("=" * 60)
    print("SpineNet Quick Test")
    print("=" * 60)

    # Create directories
    os.makedirs('example_scans', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Download example scan
    print("\n[1/5] Downloading example scan...")
    scan_name = 't2_lumbar_scan_2'
    download_example_scan(scan_name, file_path='example_scans')
    print(f"✓ Downloaded {scan_name}")

    # Download weights
    print("\n[2/5] Downloading model weights...")
    spinenet.download_weights(verbose=True, force=False)
    print("✓ Weights downloaded")

    # Check device
    import torch
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\n[3/5] Initializing SpineNet on device: {device}")
    if device == 'cpu':
        print("   ⚠ Warning: Running on CPU. This will be slower than GPU.")

    # Initialize SpineNet
    spnt = SpineNet(device=device, verbose=True)

    # Load scan
    print("\n[4/5] Loading DICOM scan...")
    overwrite_dict = {
        'SliceThickness': [2],
        'ImageOrientationPatient': [0, 1, 0, 0, 0, -1]
    }
    scan = load_dicoms_from_folder(
        f'example_scans/{scan_name}',
        require_extensions=False,
        metadata_overwrites=overwrite_dict
    )
    print(f"✓ Loaded scan with shape: {scan.volume.shape}")
    print(f"  - Pixel spacing: {scan.pixel_spacing} mm")
    print(f"  - Slice thickness: {scan.slice_thickness} mm")

    # Detect vertebrae
    print("\n[5/5] Detecting vertebrae...")
    vert_dicts = spnt.detect_vb(scan.volume, scan.pixel_spacing)
    detected_labels = [v["predicted_label"] for v in vert_dicts]
    print(f"✓ Detected {len(vert_dicts)} vertebrae: {detected_labels}")

    # Grade IVDs
    print("\nGrading intervertebral discs...")
    ivd_dicts = spnt.get_ivds_from_vert_dicts(vert_dicts, scan.volume)
    ivd_grades = spnt.grade_ivds(ivd_dicts)

    # Display results
    print("\n" + "=" * 60)
    print("GRADING RESULTS")
    print("=" * 60)
    print(ivd_grades)

    # Save results
    output_file = f'results/{scan_name}_test_results.csv'
    ivd_grades.to_csv(output_file)
    print(f"\n✓ Results saved to: {output_file}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()
