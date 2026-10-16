import numpy as np
import cv2
import torch
from typing import List, Tuple, Dict, Union


def normalize_patch(
        patch: np.ndarray,
        upper_percentile: float = 99.5,
        lower_percentile: float = 0.5
) -> np.ndarray:
    """
    Normalize a single image patch using robust percentile range.

    Parameters
    ----------
    patch : np.ndarray
        2D image patch.
    upper_percentile : float, optional
        Upper percentile for normalization range.
    lower_percentile : float, optional
        Lower percentile for normalization range.

    Returns
    -------
    np.ndarray
        Patch normalized to ~[0, 1].
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
        patch_edge_len: Union[int, float] = 26,
        overlap_param: float = 0.4,
        patch_size: Tuple[int, int] = (224, 224),
        using_resnet: bool = True,
        ) -> Tuple[torch.Tensor, List[List[Dict]]]:
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

    if  isinstance(pixel_spacing, (list, tuple, np.ndarray)):
        pixel_spacing = float(pixel_spacing[0])

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


def split_into_patches_exhaustive_spacing(
        scan: np.ndarray,
        pixel_spacing: Union[List[float], float],
        patch_edge_len: Union[int, float] = 26,
        overlap_param: float = 0.4,
        patch_size: Tuple[int, int] = (224, 224),
        using_resnet: bool = True,
) -> Tuple[List[List[torch.Tensor]], List[List[Dict]]]:
    """
    Exhaustively split 3D scan volume into resized, normalized patches for detection.
    Ensures consistency with split_into_patches_exhaustive when px=py.

    Parameters
    ----------
    scan : np.ndarray
        3D scan volume (H, W, D).
    pixel_spacing : Union[List[float], float]
        List [row_spacing, col_spacing] in mm, or scalar.
    patch_edge_len : float, optional
        Patch edge in cm (converted to mm/pixels).
    overlap_param : float, optional
        Overlap fraction between patches.
    patch_size : tuple of int, optional
        Output patch size for resizing (height, width).
    using_resnet : bool, optional
        If True, use robust normalization suited for ResNet training.

    Returns
    -------
    patches : List[List[torch.Tensor]]
        List per slice, each is a list of normalized tensor patches.
    transform_info_dicts : List[List[Dict]]
        Patch spatial origin info per slice.
    """
    h, w, d = scan.shape
    # pixel_spacing is [row, col]; a scalar or 1-element value is isotropic
    px, py = np.resize(np.asarray(pixel_spacing, dtype=float), 2)

    # If spacing is sentinel -1 (as used elsewhere), fall back to pixel units
    if px != -1 and py != -1:
        # patch_edge_len is given in cm -> convert to mm then to pixels per axis
        patch_edge_len_mm = patch_edge_len * 10.0
        # cap in mm so the patch stays square if one axis FOV < 26 cm
        patch_edge_len_mm = min(patch_edge_len_mm, (h - 1) * px, (w - 1) * py)
        patch_edge_h = int(np.round(patch_edge_len_mm / px))  # height in pixels
        patch_edge_w = int(np.round(patch_edge_len_mm / py))  # width in pixels
    else:
        # treat provided patch_edge_len as pixels if spacing not provided
        patch_edge_h = patch_edge_w = int(patch_edge_len)

    # Ensure patch edge does not exceed image dims
    patch_edge_h = min(patch_edge_h, h - 1)
    patch_edge_w = min(patch_edge_w, w - 1)

    # Effective stride along each axis
    stride_h = max(1, int(patch_edge_h * (1 - overlap_param)))
    stride_w = max(1, int(patch_edge_w * (1 - overlap_param)))

    # Compute number of patches along each axis
    num_patches_across = (w // stride_w) + 1
    num_patches_down = (h // stride_h) + 1
    num_patches = num_patches_down * num_patches_across

    transform_info_dicts = [[None] * num_patches for _ in range(d)]
    patches = [[None] * num_patches for _ in range(d)]

    for slice_idx in range(d):
        for i in range(num_patches_across):
            x1 = i * stride_w
            x2 = x1 + patch_edge_w
            if x2 >= w:
                x2 = -1
                x1 = -patch_edge_w
            for j in range(num_patches_down):
                y1 = j * stride_h
                y2 = y1 + patch_edge_h
                if y2 >= h:
                    y2 = -1
                    y1 = -patch_edge_h

                this_patch = np.array(scan[y1:y2, x1:x2, slice_idx])

                # In rare cases patch may be empty due to shape issues; skip gracefully
                if this_patch.size == 0:
                    resized_patch = np.zeros(patch_size, dtype=scan.dtype)
                else:
                    resized_patch = cv2.resize(
                        this_patch, patch_size, interpolation=cv2.INTER_CUBIC
                    )
                    # Clip to original patch value range
                    resized_patch[resized_patch < this_patch.min()] = this_patch.min()
                    resized_patch[resized_patch > this_patch.max()] = this_patch.max()

                # Normalize and convert to torch tensor
                if not using_resnet:
                    # protect against zero dynamic range
                    ptp = np.ptp(resized_patch)
                    if ptp == 0:
                        normed = np.zeros_like(resized_patch, dtype=np.float32)
                    else:
                        normed = 0.5 * ((resized_patch - resized_patch.min()) / (ptp))
                    patches[slice_idx][i * num_patches_down + j] = torch.tensor(
                        normed, dtype=torch.float32
                    )
                else:
                    patches[slice_idx][i * num_patches_down + j] = torch.tensor(
                        normalize_patch(resized_patch), dtype=torch.float32
                    )

                transform_info_dicts[slice_idx][i * num_patches_down + j] = {
                    "x1": x1,
                    "x2": x2,
                    "y1": y1,
                    "y2": y2,
                    "patch_edge_h": patch_edge_h,
                    "patch_edge_w": patch_edge_w,
                    "stride_h": stride_h,
                    "stride_w": stride_w,
                }

    return patches, transform_info_dicts