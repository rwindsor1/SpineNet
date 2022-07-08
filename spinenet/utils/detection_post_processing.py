from unittest.mock import patch
import torch
import numpy as np
import torch.nn.functional as F
import scipy.ndimage
from shapely.geometry import Polygon, Point


def make_in_slice_detections(
    detection_net,
    patches,
    transform_info_dicts,
    scan_shape,
    corner_threshold=0.3,
    centroid_threshold=0.5,
    device="cuda:0",
):
    """
    Take normalized, resized patches and pass them through the detection net,
    outputting detections in each slice

    Inputs
    ------
    detection_net : torch.nn.Module
        the VFR model used to detect vertebrae landmarks and corners
    patches: list of list of torch.Tensors
        a S-long list of P equal sized torch.FloatTensors where S indexes the
        slice number and P indexes the patch in that slice. As produced by
        scan_preprocessing.split_into_patches

    Outputs
    -------
    detection_dicts : list of dicts
        dictionaries containing information about detections made in each slice
        by the network.
    """
    # now resize patches
    detection_net.to(device)
    detection_net.eval()
    detection_dicts = []
    patches_dicts = []
    # flatten patches and pass to network
    # patches_flat_list = [patch for slice_list in patches for patch in slice_list]
    # patches_tensor = torch.stack(patches_flat_list, dim=0).unsqueeze(1)

    for slice_idx in range(len(patches)):
        patches_tensor = torch.stack(patches[slice_idx], dim=0)[:, None, :, :]
        flipped_patches_tensor = torch.flip(patches_tensor, [-1])
        net_input = torch.cat([patches_tensor, flipped_patches_tensor], axis=0)
        with torch.no_grad():
            # both_net_output = detection_net(net_input.to(device).float()).cpu()
            net_output = detection_net(patches_tensor.to(device).float()).cpu()

        patches_dicts.append({"patches": patches_tensor.numpy(), "net_output": net_output.numpy(), 
                              "landmark_points": {}, "landmark_arrows": {}})

        all_corners = {"points": {}, "arrows": {}}
        for corner_type in ["rt", "rb", "lb", "lt"]:
            all_corners["points"][corner_type] = []
            all_corners["arrows"][corner_type] = []
            patches_dicts[-1]["landmark_points"][corner_type] = []
            patches_dicts[-1]["landmark_arrows"][corner_type] = []


        scan_centroid_channel = np.zeros(scan_shape[0:2]).astype(float)
        centroid_channel_contributions = np.zeros_like(scan_centroid_channel)
        max_x = transform_info_dicts[slice_idx][0]["x2"]
        min_x = transform_info_dicts[slice_idx][0]["x1"]

        patch_edge_len = np.abs(max_x - min_x)

        # resized the centroid channel to transform it into original frame

        # loop through each patch processed in this slice
        for j in range(len(patches[slice_idx])):
            transform_info = transform_info_dicts[slice_idx][j]
            try:
                # resized the centroid channel to transform it into original frame

                resized_centroid_channels = F.interpolate(
                    net_output[:, 4, :, :].unsqueeze(1),
                    (
                        transform_info["y2"] - transform_info["y1"],
                        transform_info["x2"] - transform_info["x1"],
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
                scan_centroid_channel[
                    transform_info["y1"] : transform_info["y2"],
                    transform_info["x1"] : transform_info["x2"],
                ] = (
                    scan_centroid_channel[
                        transform_info["y1"] : transform_info["y2"],
                        transform_info["x1"] : transform_info["x2"],
                    ]
                    + resized_centroid_channels[j, 0, :, :].numpy()
                )
                centroid_channel_contributions[
                    transform_info["y1"] : transform_info["y2"],
                    transform_info["x1"] : transform_info["x2"],
                ] += 1
            except Exception as E:
                print(
                    resized_centroid_channels.shape,
                    scan_centroid_channel[
                        transform_info["y1"] : transform_info["y2"],
                        transform_info["x1"] : transform_info["x2"],
                    ].shape,
                )
                print(str(E))

            # convert corners into full scan frame
            for corner_idx, corner_type in enumerate(["rt", "rb", "lb", "lt"]):
                # get corners
                points = get_points(
                    net_output[j, corner_idx, :, :], threshold=corner_threshold
                )
                if len(points) == 0:
                    patches_dicts[-1]["landmark_points"][corner_type].append([])
                    patches_dicts[-1]["landmark_arrows"][corner_type].append([])
                    continue
                else:
                    if transform_info["y1"] < 0:
                        transform_info["y1"] = (
                            scan_centroid_channel.shape[0] + transform_info["y1"]
                        )
                    if transform_info["x1"] < 0:
                        transform_info["x1"] = (
                            scan_centroid_channel.shape[1] + transform_info["x1"]
                        )
                    transformed_patch_corners = (
                        points * patch_edge_len / 224
                        + np.array([transform_info["y1"], transform_info["x1"]])
                    )
                    all_corners["points"][corner_type].append(transformed_patch_corners)
                    arrows = np.zeros_like(points)
                    for idx, point in enumerate(points):
                        arrows[idx, 0] = net_output[
                            j, corner_idx + 9, point[0], point[1]
                        ]
                        arrows[idx, 1] = net_output[
                            j, corner_idx + 5, point[0], point[1]
                        ]
                    # scale arrows to original frame
                    all_corners["arrows"][corner_type].append(
                        arrows * patch_edge_len / 224
                    )
                    patches_dicts[-1]["landmark_points"][corner_type].append(points)
                    patches_dicts[-1]["landmark_arrows"][corner_type].append(arrows)


        # disp_corners contained the displaced corners, i.e. the vector sum
        # of the corner position and its arrow. These displaced corners are used
        # to see which centroid each corner points closest to
        disp_corners = {}
        for i in ["rt", "rb", "lb", "lt"]:
            if len(all_corners["points"][i]) > 0:
                all_corners["points"][i] = np.concatenate(
                    all_corners["points"][i], axis=0
                )
                all_corners["arrows"][i] = np.concatenate(
                    all_corners["arrows"][i], axis=0
                )
                disp_corners[i] = all_corners["points"][i] + all_corners["arrows"][i]
            else:
                disp_corners[i] = []

        # get mean centroid channel from all contributons
        centroid_channel_contributions[centroid_channel_contributions == 0] = 1
        scan_centroid_channel /= centroid_channel_contributions
        # detect the centroids
        centroids = get_points(scan_centroid_channel, threshold=centroid_threshold)

        detection_polys = []
        arrows = all_corners["arrows"]
        corners = all_corners["points"]
        # now loop through each detected centroid and find the closest displaced
        # corner of each type
        for centroid in centroids:
            if (
                len(disp_corners["lb"])
                and len(disp_corners["lt"])
                and len(disp_corners["rb"])
                and len(disp_corners["rt"])
            ):
                lt_dist = np.min(np.linalg.norm(centroid - disp_corners["lt"], axis=1))
                lt_arrow = arrows["lt"][
                    np.argmin(np.linalg.norm(centroid - disp_corners["lt"], axis=1))
                ]
                closest_lt_corner = corners["lt"][
                    np.argmin(np.linalg.norm(centroid - disp_corners["lt"], axis=1))
                ]
                lb_dist = np.min(np.linalg.norm(centroid - disp_corners["lb"], axis=1))
                lb_arrow = arrows["lb"][
                    np.argmin(np.linalg.norm(centroid - disp_corners["lb"], axis=1))
                ]
                closest_lb_corner = corners["lb"][
                    np.argmin(np.linalg.norm(centroid - disp_corners["lb"], axis=1))
                ]
                rt_dist = np.min(np.linalg.norm(centroid - disp_corners["rt"], axis=1))
                rt_arrow = arrows["rt"][
                    np.argmin(np.linalg.norm(centroid - disp_corners["rt"], axis=1))
                ]
                closest_rt_corner = corners["rt"][
                    np.argmin(np.linalg.norm(centroid - disp_corners["rt"], axis=1))
                ]
                rb_dist = np.min(np.linalg.norm(centroid - disp_corners["rb"], axis=1))
                rb_arrow = arrows["rb"][
                    np.argmin(np.linalg.norm(centroid - disp_corners["rb"], axis=1))
                ]
                closest_rb_corner = corners["rb"][
                    np.argmin(np.linalg.norm(centroid - disp_corners["rb"], axis=1))
                ]
                indiv_arrows = [rt_arrow, rb_arrow, lb_arrow, lt_arrow]
                poly = Polygon(
                    [
                        closest_rt_corner,
                        closest_rb_corner,
                        closest_lb_corner,
                        closest_lt_corner,
                    ]
                )
                # threshold arrows, do not allow if distance from centroid is over 0.5* length of arrow
                missing_arrows = arrows_threshold_check(
                    [
                        [rt_dist, rt_arrow],
                        [rb_dist, rb_arrow],
                        [lb_dist, lb_arrow],
                        [lt_dist, lt_arrow],
                    ]
                )
                detection_poly = np.array(
                    [
                        closest_rt_corner,
                        closest_rb_corner,
                        closest_lb_corner,
                        closest_lt_corner,
                    ]
                )
                if sum(missing_arrows) == 0:
                    if poly.is_valid and poly.contains(Point(centroid)):
                        # flip around poly to match form needed for matplotlib plotting
                        detection_polys.append([[i[1], i[0]] for i in detection_poly])
                # # if only one arrow is missing simply assume vertebra is a diamond shape
                elif sum(missing_arrows) == 1:
                    for i, el in enumerate(missing_arrows):
                        if el:
                            detection_poly[i] = centroid + indiv_arrows[(i + 2) % 4]
                    # # flip around poly to match form needed for matplotlib plotting
                    detection_polys.append([[i[1], i[0]] for i in detection_poly])

        # loop through each detection poly and remove those where the same corner
        # is shared between two polygons
        # else: print(slice_idx, len(patches)//2)
        detection_polys = remove_polys_sharing_corners(detection_polys, all_corners)

        all_corners["centroid_heatmap"] = scan_centroid_channel
        all_corners["centroids"] = centroids
        all_corners["detection_polys"] = detection_polys
        all_corners["vert_index"] = [None] * len(detection_polys)
        detection_dicts.append(all_corners)

    return detection_dicts, patches_dicts


def remove_polys_sharing_corners(detection_polys, all_corners):
    """
    Takes as input all the polygons detected in the slice and where two polygons
    share the same corner, remove one of them. Of the two overlapping polygons,
    the one with the biggest difference between its largest internal angle and
    its smallest is removed.

    Inputs
    ------
    detection_polys : list of 4x2 numpy float arrays
        all the polygons detected in the slice so far
    all_corners : dictionary
        information about the arrows for each polygon vertex.

    Outputs
    -------
    detection_polys : list of 4x2 lists.
        A new polygon list with spurious detections removed.
    """
    # stores indicies of the polys to be removed
    polys_to_be_removed = []
    for detection_poly_idx, detection_poly in enumerate(detection_polys):
        for other_detection_poly_idx, other_detection_poly in enumerate(
            detection_polys
        ):
            # skip over overlaps with its self
            if detection_poly != other_detection_poly:
                # vectors between all the corners
                diff_arr = np.array(detection_poly) - np.array(other_detection_poly)
                if np.min(np.linalg.norm(diff_arr, axis=-1)) < 2:
                    # should go here if the detection polys have the same corner
                    detection_poly_internal_angles = get_internal_angles(detection_poly)
                    other_detection_poly_internal_angles = get_internal_angles(
                        other_detection_poly
                    )

                    if np.ptp(detection_poly_internal_angles) > np.ptp(
                        other_detection_poly_internal_angles
                    ):
                        polys_to_be_removed.append(detection_poly_idx)
                    else:
                        polys_to_be_removed.append(other_detection_poly_idx)

    polys_to_be_removed = sorted(list(set(polys_to_be_removed)), reverse=True)

    for poly_to_be_removed in polys_to_be_removed:
        detection_polys.pop(poly_to_be_removed)

    return detection_polys


def get_internal_angles(poly):
    """
    Calculates the internal angles for a quadrilateral shaped as a 4x2 numpy list

    Inputs
    ------
    poly : 4x2 numpy array
        the polygon to calculate the internal angles of
    Outputs
    -------
    poly_internal_angles : list
        a list of internal angles for the polygon
    """
    poly_internal_angles = []
    poly = np.array(poly)
    for i in range(4):
        v1 = poly[i - 1, :] - poly[i, :]
        v2 = poly[(i + 1) % 4, :] - poly[i, :]
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        poly_internal_angles.append(np.arccos(np.dot(v1, v2)))

    return poly_internal_angles


def arrows_threshold_check(arrows_arr):
    """
    find corners without an arrow pointing to the centroid within a certain threshold distance
    """
    # TODO: Complete docstring
    # arrows arr should be a list of 2-entry lists containing the distance form the centroid and the arrow in cartesian coords
    missing_arrows = [0, 0, 0, 0]  # binary mask to show missing corner
    for idx, el in enumerate(arrows_arr):
        if el[0] > 0.5 * np.linalg.norm(el[1]) or el[0] > 20:
            missing_arrows[idx] = 1
    return missing_arrows


def get_points(image, threshold=0.5):
    """
    finds points from gaussian response maps

    this is done by thresholding the response map to get a binary map and then
    treating each interconnected volume as a detection. The point of maximum
    response in each interconnected volume becomes the location of the landmark

    Inputs
    ------
    image : np.array-like
        HxW image containing gaussian response map for landmarks
    threshold: float
        the value at which to threshold the response map.

    Outputs
    -------
    points : a Nx2 array containing the locations of each point detected
    """

    image = np.array(image)
    mask = image > threshold
    segmentation, no_labels = scipy.ndimage.label(mask)
    points = []
    for label in range(no_labels):
        point = np.unravel_index(
            np.argmax(image * (segmentation == (label + 1))), image.shape
        )
        points.append(point)

    return np.array(points)
