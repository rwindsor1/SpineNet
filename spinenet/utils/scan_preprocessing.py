import numpy as np
import torch
import cv2
from typing import List, Tuple, Dict, Union


def normalize_patch(patch : np.ndarray, 
                    upper_percentile: float = 99.5, 
                    lower_percentile: float = 0.5) -> np.ndarray:   
    """
    Takes in a patch and normalizes it in each slice robustly using the
    `upper_percentile` and `lower_percentile` percentile values. Values are linearly
    mapped such that pixels at the `upper_percentile` of intensity values are mapped to
    1 and pixels at the `lower_percentile` of intensity values are mapped to 0.

    Parameters
    ----------
    patch : np.ndarray
        The patch to be normalized
    upper_percentile : float
        The percentile value to use for the upper bound of the normalization
    lower_percentile : float
        The percentile value to use for the lower bound of the normalization

    Returns
    -------
    patch : np.ndarray
        The normalized patch
    """
    upper_percentile_val = np.percentile(patch, upper_percentile)
    lower_percentile_val = np.percentile(patch, lower_percentile)
    robust_range = np.abs(upper_percentile_val - lower_percentile_val)
    if upper_percentile_val == lower_percentile_val:
        patch = (patch - patch.min()) / (patch.ptp() + 1e-9)
    else:
        patch = (patch - patch.min()) / (robust_range + 1e-9)

    return patch

def split_into_patches_exhaustive(
    scan: np.ndarray,
    pixel_spacing: float,
    patch_edge_len: Union[int,float] = 26,
    overlap_param: float = 0.4,
    patch_size: Tuple[int,int] = (224, 224),
    using_resnet: bool = True,
) -> Tuple[torch.Tensor,List[List[Dict]]]:
    """
    Takes in a 3d scan volume and splits it into patches, resizes them then
    normalises them to be passed to the detection network.

    Parameters
    ----------
    scan : np.ndarray
        3d scan volume to be split into patches
    pixel_spacing : float
        The pixel spacing of the scan in mm. Used to calculate patch edge len
    patch_edge_len : Union[int,float]
        The edge length of the patches in cm. Default is 26cm, but can be
        adjusted to be slightly smaller or larger without performance degradation.
    overlap_param : float
        The amount of overlap to create between patches.
    remove_black_space : bool
        Whether to remove black space from the scan.
    patch_size : Tuple[int,int]
        The size of the patches to be created.  
    using_resnet : bool
        Changes patch normalization to match the method used to train the VFR ResNet.
    
    Returns
    -------
    patches : torch.Tensor
        A tensor of patches of shape (num_slices, num_patches, patch_size[0], patch_size[1])
    transform_info_dicts : List[List[Dict]]
        Lists of dictionaries containing the transform information for each patch in each slice.
        Has keys 'x1','x2','y1','y2' indication the position of the patch within the slice
    """

    h, w, d = scan.shape
    if pixel_spacing != -1:
        patch_edge_len = int(patch_edge_len * 10 / pixel_spacing)

    if patch_edge_len > min(scan.shape[0], scan.shape[1]):
        patch_edge_len = min(scan.shape[0:2]) - 1

    # effective_edge_len = how far patches should be spaced from each other
    effective_patch_edge_len = int(patch_edge_len * (1 - overlap_param))

    # work out tiling for scan
    num_patches_across = (w // effective_patch_edge_len) + 1
    num_patches_down = (h // effective_patch_edge_len) + 1
    # total number of patches in each slice
    num_patches = num_patches_down * num_patches_across

    transform_info_dicts = [[None] * num_patches for slice_no in range(d)]
    patches = [[None] * num_patches for slice_no in range(d)]

    for slice_idx in range(d):
        for i in range(num_patches_across):
            x1 = i * effective_patch_edge_len
            x2 = x1 + patch_edge_len
            if x2 >= w:
                x2 = -1
                x1 = -(patch_edge_len)
            for j in range(num_patches_down):
                y1 = j * effective_patch_edge_len
                y2 = y1 + patch_edge_len
                if y2 >= h:
                    y2 = -1
                    y1 = -(patch_edge_len)
                this_patch = np.array(scan[y1:y2, x1:x2, slice_idx])
                resized_patch = cv2.resize(
                    this_patch, patch_size, interpolation=cv2.INTER_CUBIC
                )
                resized_patch[resized_patch < this_patch.min()] = this_patch.min()
                resized_patch[resized_patch > this_patch.max()] = this_patch.max()

                if not using_resnet:
                    patches[slice_idx][i * num_patches_down + j] = 0.5 * torch.Tensor(
                        (resized_patch - np.min(resized_patch))
                        / (np.ptp(resized_patch))
                    )
                else:
                    patches[slice_idx][i * num_patches_down + j] = torch.Tensor(
                        normalize_patch(resized_patch)
                    )
                transform_info_dicts[slice_idx][i * num_patches_down + j] = {
                    "x1": x1,
                    "x2": x2,
                    "y1": y1,
                    "y2": y2,
                }

    return patches, transform_info_dicts
