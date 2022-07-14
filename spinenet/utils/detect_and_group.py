import os
import torchvision
import torch
import cv2
from shapely.geometry import Polygon, Point
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from .scan_preprocessing import split_into_patches_exhaustive
from .detection_post_processing import make_in_slice_detections


def detect_and_group(
    detection_net,
    scan,
    remove_excess_black_space=True,
    pixel_spacing=1,
    plot_outputs=False,
    using_resnet=True,
    corner_threshold=0.5,
    centroid_threshold=0.5,
    group_across_slices_threshold=0.2,
    remove_single_slice_detections=True,
    device="cuda:0",
    debug=False,
):
    """
    Detects all vertebrae corners, groups into vertebrae and returns results

    Inputs
    ------
        detection_net : torch.nn.Module
            the VFR model used to detect vertebrae landmarks and corners
        scan : np.array
            A HxWxS array containing the scan
        remove_excess_black_space : bool [optional]
            remove any excess space at the sides of the scan. DEPRECIATED
        plot_outputs : bool [optional]
            Show response maps for detection channels of detection_net output
        using_resnet :bool [optional]
            A flag that tells the function is it is using a resnet model. If
            so this changes the normalization
        centroid_thresholding : float 0<=x<=1
            The value at which to threshold the detector output in the centroid
            channel. Higher values means more detections but risks false
            positives
        corner_thresholding : float 0<=x<=1
            Like the centroid thresholding parameter but for the corner channels
            of the detection network

    Outputs
    -------
        vert_dicts : list of dicts
            A list of dictionaries containing information about the vertebrae
            detected in the scan. Each element of the list corresponds to a
            different vertebra with the key 'average_polygon' containing the
            median location of the vertebrae through the slices, 'slice_nos'
            containing the slices in which the polygon appears and 'polys' the
            polygons detected.
    """

    # split the scan into different patches
    patches, transform_info_dicts = split_into_patches_exhaustive(
        scan, pixel_spacing=pixel_spacing, overlap_param=0.4, using_resnet=using_resnet
    )
    # group the detections made in each patch into slice level detections
    detection_dicts, patches_dicts = make_in_slice_detections(
        detection_net,
        patches,
        transform_info_dicts,
        scan.shape,
        corner_threshold,
        centroid_threshold,
        device=device,
    )

    vert_dicts = group_slice_detections(
        detection_dicts,
        iou_threshold=group_across_slices_threshold,
        remove_single_slice_detections=remove_single_slice_detections,
    )

    if not debug:
        return vert_dicts
    else:
        return vert_dicts, patches, patches_dicts, detection_dicts, transform_info_dicts


def group_slice_detections(
    detection_dicts, iou_threshold=0.1, remove_single_slice_detections=True
):
    """
    group polygons detected across slices to get 3-d bounding quadrilateral for vertebra

    Inputs
    ------
    detection_dicts: list of dicts
        list of dictionaries of length equal to the number of slices. Each dictionary describes
        the detections made in a given slice. TODO - describe structure of these dictionaries

    Outputs
    -------
    vert_dicts : list
        list of dictionaries describing each vertebra detected. Each element corresponds to a
        different vertebra. TODO - describe structure of these dictionaries
    """
    # TODO: Complete docstring
    vert_dicts = []
    # loop through slices
    for slice_no, slice_detections in enumerate(detection_dicts):
        # loop through vert bodies detected in each slice
        for polygon_in_slice in slice_detections["detection_polys"]:
            # now loop through previous detections and see if any match up
            overlaps_with_previous = False
            for vert in vert_dicts:
                try:
                    most_recent_polygon = vert["polys"][-1]
                    poly_ious = get_poly_iou(polygon_in_slice, most_recent_polygon)
                except:
                    poly_ious = 0
                if poly_ious > iou_threshold:
                    vert["polys"].append(polygon_in_slice)
                    vert["slice_nos"].append(slice_no)
                    # recalculate average poly for that vertebrae
                    vert["average_polygon"] = np.mean(vert["polys"], axis=0)
                    overlaps_with_previous = True
                    break
            # make new entry if doesn't overlap with any of the previous ones
            if overlaps_with_previous is False:
                vert_dicts.append(
                    {
                        "polys": [polygon_in_slice],
                        "average_polygon": polygon_in_slice,
                        "slice_nos": [slice_no],
                    }
                )
    remove_indices = []  # list to store indicies of verts to be deleted
    if remove_single_slice_detections:
        for vert_index, vert in enumerate(vert_dicts):
            if len(vert["polys"]) < 2:
                remove_indices.insert(
                    0, vert_index
                )  # insert at beginning to get reverse order
        [vert_dicts.pop(remove_index) for remove_index in remove_indices]

    # go through all vertebrae and check their average polygons dont overlap
    match_vert_idxs = []
    for vert_idx, vert_dict in enumerate(vert_dicts):
        for other_vert_idx, other_vert_dict in enumerate(vert_dicts):
            if vert_idx <= other_vert_idx:
                continue
            iou = get_poly_iou(
                vert_dict["average_polygon"], other_vert_dict["average_polygon"]
            )
            if iou > 0.05:
                match_vert_idxs.append([vert_idx, other_vert_idx])
                print('merging')

    # join the matching verts
    for matching_vert_pair in match_vert_idxs:
        vert_dicts[matching_vert_pair[0]]["polys"] += vert_dicts[matching_vert_pair[1]][
            "polys"
        ]
        vert_dicts[matching_vert_pair[0]]["slice_nos"] += vert_dicts[
            matching_vert_pair[1]
        ]["slice_nos"]
        vert_dicts[matching_vert_pair[0]]["average_polygon"] = np.mean(
            vert_dicts[matching_vert_pair[0]]["polys"], axis=0
        )
        print(f"merging vert {matching_vert_pair[0]} with {matching_vert_pair[1]}")

    # remove extra verts
    for matching_vert_pair in sorted(match_vert_idxs, key=lambda x: x[1], reverse=True):
        vert_dicts.pop(matching_vert_pair[1])

    # now sort list in height order
    vert_dicts.sort(key=lambda x: np.mean(np.array(x["average_polygon"])[:, 1]))
    return vert_dicts


def get_poly_iou(poly1, poly2):
    """
    calculates and returns IOU of two polygons
    Inputs
    ------
    poly1 : np.array
        4x2  numpy array containing vertices of quadrilateral
    poly2 : np.array
        4x2  numpy array containing vertices of quadrilateral

    Outputs
    -------
    iou : float
        the iou of poly1 and poly2
    """

    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    min_area = np.min([poly1.area, poly2.area])
    if union > 0:
        return intersection / union
    else:
        return 0


def red(x):
    y = np.zeros_like(x)
    return np.stack([x, y, y], axis=-1)


def blue(x):
    y = np.zeros_like(x)
    return np.stack([y, y, x], axis=-1)


def green(x):
    y = np.zeros_like(x)
    return np.stack([y, x, y], axis=-1)


def yellow(x):
    y = np.zeros_like(x)
    return np.stack([x, x, y], axis=-1)


def pink(x):
    y = np.zeros_like(x)
    return 0.5 * (yellow(x) + red(x))


def color(x):
    y = np.zeros_like(x)
    return np.stack([x, x, x], axis=-1)
