import numpy as np
import torch
import cv2


def normalize_patch(patch, upper_percentile=99.5, lower_percentile=0.5):
    """
    Takes in a patch and normalizes it in each slice robustly using the
    `upper_percentile` and `lower_percentile` percentile values
    """
    upper_percentile_val = np.percentile(patch, upper_percentile)
    lower_percentile_val = np.percentile(patch, lower_percentile)
    robust_range = np.abs(upper_percentile_val - lower_percentile_val)
    if upper_percentile_val == lower_percentile_val:
        patch = (patch - patch.min()) / (patch.ptp() + 1e-9)
    else:
        patch = (patch - patch.min()) / (robust_range + 1e-9)

    # new_scan = volume.copy()
    # for i in range(volume.shape[-1]):
    #     new_scan[:,:,i] = (volume[:,:,i] - np.min(volume[:,:,i]))*np.max(volume[:,:,i])/np.percentile(volume[:,:,i],95)
    # volume = new_scan

    return patch


def split_into_patches(
    scan,
    overlap_param=0.5,
    remove_black_space=True,
    patch_size=(336, 336),
    using_resnet=True,
):
    """
    Takes in a 3d scan volume and splits it into patches, resizes them then
    normalises them to be passed to the detection network.

    Inputs
    ------
    volume (np.array or torch.tensor): The volume to be split into patches
    overlap_param, Optional (float, 0<=x<=1): A parameter controlling how
    much each patch overlaps with its neighbours, e.g. 0.4 = 40%. Defaults to
    0.4.
    remove_black_space, Optional (bool): A parameter saying whether black space
    at the sides of scans should be removed or not. Defaults to true, but should
    probably be false for lumbar only scans
    patch_size, Optional, (tuple:(int,int)): The size of the output patches

    Outputs
    -------
    patches (torch.tensor): A Nx224x224 float tensor of patches to be used
    as input to the detection network
    transform_info (list): A list of N dictionaries, with each element
    containing information on how to transform the patch into the original scans
    frame
    """

    # set min_xs and max_xs, the x values in the image at which the patches
    # should start
    if remove_black_space:
        # sum each column of the image and define a threshold to be 10% of the
        # maximum value. Also remove 10% from each edge of scan regardless of
        # threshold value
        col_sums = scan.sum(axis=0)
        thresh = col_sums.max(axis=0) / 10
        mask = np.asarray(col_sums - thresh[np.newaxis, :] > 0)
        mask[: int(mask.shape[0] * 0.1), :] = False
        mask[int(mask.shape[0] * 0.9) :, :] = False
        vals = [np.where(mask[:, i])[0] for i in range(mask.shape[-1])]
        # min_xs = np.array([np.where(mask[:,i])[0].min() for i in range(mask.shape[-1])])
        # max_xs = np.array([np.where(mask[:,i])[0].max() for i in range(mask.shape[-1])])
        min_xs = np.array(
            [vals[i].min() if len(vals[i]) else 0 for i in range(mask.shape[-1])]
        )
        max_xs = np.array(
            [
                vals[i].max() if len(vals[i]) else scan.shape[1] - 1
                for i in range(mask.shape[-1])
            ]
        )

        # to deal with mostly black slices, have minimum width of an axis of 50
        widths = np.abs(max_xs - min_xs)
        for idx in np.where(widths < 50)[0]:
            if min_xs[idx] < 25:
                max_xs[idx] += 50
            elif (scan.shape[1] - 1 - max_xs[idx]) < 25:
                min_xs[idx] -= 50
            else:
                max_xs[idx] += 25
                min_xs[idx] -= 25
    else:
        min_xs = np.array([0 for i in range(scan.shape[-1])])
        max_xs = np.array([scan.shape[1] - 1 for i in range(scan.shape[-1])])

    # calculate number of patches needed in each slice according to min_xs and
    # max_xs as well as overlap constraints. Store in list num_patches
    num_patches = np.ceil(
        scan.shape[0] * (overlap_param + 1) / (max_xs - min_xs)
    ).astype(int)

    # make list to store patches in and lists to store dictionaries containing
    # information to transform patches back into original scan frame
    patches = [[None] * num_patches[slice_no] for slice_no in range(len(num_patches))]
    transform_info_dicts = [
        [None] * num_patches[slice_no] for slice_no in range(len(num_patches))
    ]

    for slice_idx in range(len(num_patches)):
        width = np.abs(max_xs - min_xs)[slice_idx]
        x1 = int(min_xs[slice_idx])
        x2 = int(max_xs[slice_idx])
        for j in range(num_patches[slice_idx]):
            y1 = int(j / (1 + overlap_param) * width)
            y2 = int(y1 + width)
            if y2 > scan.shape[0]:
                y2 = scan.shape[0]
                y1 = int(y2 - width)

            this_patch = np.array(scan[y1:y2, x1:x2, slice_idx])
            # resize patch using cv2's default bilinear interpolation
            # resized_patch = cv2.resize(this_patch,patch_size)
            # resize patch using cv2's default bicubic interpolation
            resized_patch = cv2.resize(
                this_patch, patch_size, interpolation=cv2.INTER_CUBIC
            )
            resized_patch[resized_patch < this_patch.min()] = this_patch.min()
            resized_patch[resized_patch > this_patch.max()] = this_patch.max()
            if not using_resnet:
                patches[slice_idx][j] = 0.5 * torch.Tensor(
                    (resized_patch - np.min(resized_patch)) / (np.ptp(resized_patch))
                )
            else:
                patches[slice_idx][j] = torch.Tensor(normalize_patch(resized_patch))
            transform_info_dicts[slice_idx][j] = {
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
            }

    return patches, transform_info_dicts


def split_into_patches_exhaustive(
    scan,
    patch_edge_len=26,
    pixel_spacing=-1,
    overlap_param=0.4,
    patch_size=(224, 224),
    using_resnet=True,
):
    """
    Takes in a 3d scan volume and splits it into patches, resizes them then
    normalises them to be passed to the detection network. Unlike the
    split_into_patches function, this function does not remove black space and
    instead just exhaustively tries to detect vertebrae in every part of the
    image

    Inputs
    ------
    volume (np.array or torch.tensor): The volume to be split into patches
    overlap_param, Optional (float, 0<=x<=1): A parameter controlling how
    much each patch overlaps with its neighbours, e.g. 0.4 = 40%. Defaults to
    0.4.
    patch_edge_len (int): the edge len (in cm if ipp given, else in pixels)
    of each patch to give to the resnet
    pixel_spacing (float): the size between the pixels in mm as given by most
    DICOM files. If given the value -1, uses patch_edge_len based on pixels
    instead
    patch_size ((int,int)): the size of the patches to be given to the network
    pixel_spacing: the size of each pixel used as input to the network

    Outputs
    -------
    patches (torch.tensor): A Nxpatch_size[0]xpatch_size[1] float
    tensor of patches to be used as input to the detection network
    transform_info (list): A list of N dictionaries, with each element
    containing information on how to transform the patch into the original scans
    frame
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
