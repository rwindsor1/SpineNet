import torch
from shapely.geometry import Polygon as shapelyPolygon
from shapely.geometry import Point as shapelyPoint
import numpy as np
import cv2
import torch.nn.functional as F


def extract_volumes(
    scan,
    vert_dicts,
    extent=1,
    rescale_sagittal=True,
    output_shape=(224, 224, 16),
    resampling_mode="bicubic",
):
    """
    Extracts volumes surrounding each vertebra detected in the scan.

    Inputs
    -----
    scan :  np.array
        A HxWxS array containing the MRI scan. H is height of scan, W is width and S is number of sagittal slices
    vert_dicts :  list of dictionaries
        Each dictionary contains information about the vertebra detections in the scan, as given by the output
        from `utils.group_slice_detections` and `utils.discard_outliers'
    extent   : float
        parameter controlling the amount of area around each detection bounding box which should be used to generate the volume.
        An extent of x means extent by the square on each side by x*edge_length, where edge_length is the bigger of the width and height of the box
    resampling_mode: str
        mode to use for resampling the scan. Some options are 'nearest', 'linear', 'cubic' and 'bicubic'

    Outputs
    -------
    vert_dicts : list of dictionaries
        A modified version of the input `vert_dicts', except with a new key `volume' added to each dictionary containing the volume
        surrounding the vertebrae as a 224x224x16 np.array
    """

    for vert_idx, vert in enumerate(vert_dicts):
        detection_poly = vert["average_polygon"]
        detection_poly = np.array(detection_poly)

        rotated_scan, new_bb = straighten_bb(scan, detection_poly)

        new_bb_width = new_bb[:, 1].ptp()
        new_bb_height = new_bb[:, 0].ptp()
        edge_len = np.max([new_bb_height, new_bb_width])
        y_min = int(new_bb[-1, 1] - extent * edge_len)
        x_min = int(new_bb[-1, 0] - extent * edge_len)
        y_max = int(y_min + (1 + 2 * extent) * edge_len)
        x_max = int(x_min + (1 + 2 * extent) * edge_len)
        try:
            if y_min < 0:
                diff = -y_min
                diff_patch = np.zeros((diff, x_max - x_min, rotated_scan.shape[2]))
                vol = rotated_scan[0:y_max, x_min:x_max]
                volume = np.concatenate((diff_patch, vol), axis=0)
            else:
                volume = rotated_scan[y_min:y_max, x_min:x_max]

            resized_bb = resize_bb(
                volume,
                output_shape=output_shape,
                only_2d_interpolation=(not rescale_sagittal),
                resampling_mode=resampling_mode,
            ).numpy()

            vert["volume"] = resized_bb
        except Exception as error:
            print(error)
            print("Warning: Could not resize vertebral volume!")
            print(y_min, x_min, x_max, y_max)
            vert["volume"] = np.zeros(output_shape)
    return vert_dicts


def straighten_bb(scan, bounding_box):
    """
    Given a 3d scan and 2d bounding boxes, rotate the boxes and the scan so that the top edge of the bounding box is horizontal and give the coordinates of the bounding box in the new scan

    Inputs
    ------
    scan: np.array
        a 3d numpy tensor of dimensions height, width and depth (sagittal plane)
    bounding_box: np.array
        a 4x2 tensor with with [:,0] containing the x coordinates of the box corners and [:,1] containing the y coordinates
    Outputs
    -------
    rotated_scan: np.array
        a 3d numpy tensor of dimensions height, width and depth (sagittal plane). Rotated version of scan
    new_bounding_box: np.array
        a 4x2 tensor with with [:,0] containing the x coordinates of the box corners and [:,1] containing the y coordinates. Same co-ords as bounding_box, transformed to the frame of rotated_scan
    """
    x_lt = bounding_box[-1, 0]
    y_lt = bounding_box[-1, 1]
    x_rt = bounding_box[0, 0]
    y_rt = bounding_box[0, 1]
    delta_y = y_rt - y_lt
    delta_x = x_rt - x_lt
    theta = np.tanh(delta_y / delta_x) * 180 / np.pi

    rotation_matrix = cv2.getRotationMatrix2D(
        (np.mean([x_lt, x_rt]), np.mean([y_lt, y_rt])), theta, scale=1
    )
    # get new_bounding_box
    new_bounding_box = (
        rotation_matrix @ [bounding_box[:, 0], bounding_box[:, 1], np.ones(4)]
    ).T

    rotated_scan = np.zeros_like(scan).astype(np.float32)
    for i in range(scan.shape[-1]):
        try:
            rotated_scan[:, :, i] = cv2.warpAffine(
                scan[:, :, i],
                rotation_matrix,
                (scan.shape[1], scan.shape[0]),
                flags=cv2.INTER_CUBIC,
            )
        except Exception as e:
            print(e)
            import pdb

            pdb.set_trace()

    return rotated_scan, new_bounding_box


def resize_bb(
    vol,
    output_shape=(224, 224, 16),
    only_2d_interpolation=False,
    resampling_mode="bicubic",
):
    """
    Resizes a 3d tensor to be of size `output_shape` and transforms a bounding box into the new shape

    Inputs
    ------
    vol: np.array
        the 3d tensor to be resized
    output_shape:tuple
        the 3d shape to make interpolate vol to
    Outputs
    -------
    new_vol: np.array
        the new, interpolated volume of size output shape
    """
    orig_vol_shape = vol.shape
    vol = torch.einsum("ijk->kij", torch.tensor(vol)).unsqueeze(1).type(torch.double)
    if resampling_mode in ["bicubic", "linear", "bicubic"]:
        new_vol = F.interpolate(
            vol,
            (output_shape[0], output_shape[1]),
            mode=resampling_mode,
            align_corners=False,
        ).squeeze(1)
    else:
        new_vol = F.interpolate(
            vol,
            (output_shape[0], output_shape[1]),
            mode=resampling_mode,
        ).squeeze(1)

    new_vol = torch.einsum("kij->ijk", new_vol)

    new_new_vol = torch.zeros(output_shape)

    ratio = float(new_vol.shape[-1]) / float(output_shape[-1])
    for i in range(output_shape[-1]):
        interp_val = i * ratio
        high_slice = int(np.ceil(interp_val))
        low_slice = int(np.floor(interp_val))
        if high_slice >= new_vol.shape[-1]:
            high_slice = new_vol.shape[-1] - 1
        if low_slice >= new_vol.shape[-1]:
            low_slice = new_vol.shape[-1] - 1
        if high_slice == low_slice:
            new_new_vol[:, :, i] = new_vol[:, :, high_slice]
        else:
            new_new_vol[:, :, i] = new_vol[:, :, high_slice] * np.abs(
                low_slice - interp_val
            ) + new_vol[:, :, low_slice] * np.abs(high_slice - interp_val)

    arr_percentile = np.percentile(new_new_vol[:, :], 95)
    min_ = new_new_vol.min()
    for slice_ in range(new_new_vol.shape[-1]):
        new_new_vol[:, :, slice_] = (new_new_vol[:, :, slice_] - min_) / (
            arr_percentile + 0.000001
        )

    if only_2d_interpolation:
        new_new_vol = torch.tensor(new_vol)
    # change this to new_new_vol to get 16  slice volumes back
    return new_new_vol
