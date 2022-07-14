from posixpath import normcase
import cv2
import os
import numpy as np
import pandas as pd
import scipy.io as spio
from scipy import ndimage as nd
from skimage import draw
from os import listdir
from os.path import isfile, join
import pydicom
import pydicom.pixel_data_handlers.gdcm_handler as gdcm_handler

pydicom.config.image_handlers = [None, gdcm_handler]


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last) : int(last + avg)])
        last += avg

    return out


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_keys(d):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    del data["__header__"]
    del data["__version__"]
    del data["__globals__"]
    return _check_keys(data)


def get_scan(scan_path, geno, idx):

    curr_scan_dir = geno["Scan"][idx]
    curr_meta = geno["ScanMetadata"][idx]
    curr_info = geno["IVDInfo"][idx]
    curr_quad = geno["BoundingQ"][idx]

    if isinstance(curr_meta, (list,)):
        pixel_spacing = curr_meta[0]["PixelSpacing"][0]
    else:
        pixel_spacing = curr_meta["PixelSpacing"][0]

    mid_sag = np.round(np.mean(np.array(curr_info["midSagittal"]) - 1.0)).astype(
        np.uint8
    )

    # Slice index
    slice_idx = mid_sag

    # Get slice from volume
    if isinstance(curr_scan_dir, (list,)):
        s_temp = pydicom.dcmread(
            curr_scan_dir[slice_idx].replace("../../Dataset/", scan_path), force=True
        )
        if len(s_temp.file_meta) == 0:
            s_temp.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        try:
            scan = np.array(s_temp.pixel_array)
        except:
            curr_scan_dir = curr_scan_dir[0]
            slash_idx = [m.start() for m in re.finditer("/", curr_scan_dir)]
            scan = np.array(
                sio.loadmat(
                    "genomat/"
                    + curr_scan_dir[slash_idx[3] + 1 : slash_idx[-1]]
                    + "/scan.mat",
                    verify_compressed_data_integrity=False,
                )["scan"]
            )
            scan = scan[:, :, slice_idx]
    else:
        s_temp = pydicom.dcmread(
            curr_scan_dir.replace("../../Dataset/", scan_path), force=True
        )
        if len(s_temp.file_meta) == 0:
            s_temp.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        scan = np.array(s_temp.pixel_array)
        if np.size(np.shape(scan)) == 3:
            scan = scan[slice_idx, :, :]

    # Normalize
    scan_shape = np.shape(scan)
    scan = scan.astype(float)
    scan = (scan - np.min(scan)) / (np.ptp(scan) + 1e-9)

    # Get bounding quadrilaterals
    if pixel_spacing == 0:
        resize_factor = 1
    else:
        resize_factor = pixel_spacing / 0.5859
    resize_factor = 1 / resize_factor

    try:
        x = np.array(curr_quad[slice_idx]["x"])
        y = np.array(curr_quad[slice_idx]["y"])
    except:
        x = np.array(curr_quad["x"])
        y = np.array(curr_quad["y"])
    x = x * resize_factor
    y = y * resize_factor
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    return scan, x, y


def get_patch_ex(scan, x, y):
    # Initial param
    height = scan.shape[0]
    width = scan.shape[1]
    theta = np.degrees(np.arctan2(y[0] - y[3], x[0] - x[3]))

    # Rotate
    scanrot, qx, qy, offset_w, offset_h = rotate_bb_and_scan(
        scan, x, y, width, height, theta
    )

    # Patch
    pad_add = 5
    w_st = np.round(min(qx))
    w_en = np.round(max(qx))
    h_st = np.round(min(qy))
    h_en = np.round(max(qy))
    w_pad = np.round((w_en - w_st))
    h_pad = np.round((h_en - h_st))
    w_st -= pad_add
    w_en += pad_add
    h_st -= pad_add
    h_en += pad_add
    if h_st < 0:
        h_st = 0
    if w_st < 0:
        w_st = 0
    scan_patch = scanrot[int(h_st) : int(h_en), int(w_st) : int(w_en)]

    return scan_patch


def get_patch(scan, x, y):
    # Initial param
    height = scan.shape[0]
    width = scan.shape[1]
    theta = np.degrees(np.arctan2(y[0] - y[3], x[0] - x[3]))

    # Rotate
    scanrot, qx, qy, offset_w, offset_h = rotate_bb_and_scan(
        scan, x, y, width, height, theta
    )

    # Patch
    pad_mult = 0.75
    w_st = np.round(min(qx))
    w_en = np.round(max(qx))
    h_st = np.round(min(qy))
    h_en = np.round(max(qy))
    w_pad = np.round((w_en - w_st))
    h_pad = np.round((h_en - h_st))
    w_st -= w_pad * pad_mult
    w_en += w_pad * pad_mult
    h_st -= h_pad * pad_mult
    h_en += h_pad * pad_mult
    if h_st < 0:
        h_st = 0
    if w_st < 0:
        w_st = 0
    scan_patch = scanrot[int(h_st) : int(h_en), int(w_st) : int(w_en)]

    new_h = scan_patch.shape[0]
    new_w = scan_patch.shape[1]
    qx_offset = -np.round(min(qx)) + w_pad * pad_mult
    qy_offset = -np.round(min(qy)) + h_pad * pad_mult
    qx = qx + qx_offset
    qy = qy + qy_offset
    qy = qy / new_h * 224
    qx = qx / new_w * 224
    scan_patch = cv2.resize(scan_patch, (224, 224), interpolation=cv2.INTER_CUBIC)

    return scan_patch, new_w, new_h, qx_offset, qy_offset, offset_w, offset_h, theta


def rotate_bb_and_scan(scan, x, y, width, height, theta):
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), theta, scale=1)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((height * sin) + (width * cos))
    new_h = int((height * cos) + (width * sin))
    offset_w = (new_w - width) / 2
    offset_h = (new_h - height) / 2
    rotation_matrix[0, 2] += offset_w
    rotation_matrix[1, 2] += offset_h
    v = np.vstack((x.ravel(), y.ravel(), np.ones(np.size(y.ravel()))))
    calculated = np.dot(rotation_matrix, v)
    scanrot = cv2.warpAffine(
        scan, rotation_matrix, (int(new_w), int(new_h)), flags=cv2.INTER_CUBIC
    )
    qx = calculated[0]
    qy = calculated[1]

    return scanrot, qx, qy, offset_w, offset_h


def rotate_bb(x, y, width, height, theta):
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), theta, scale=1)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((height * sin) + (width * cos))
    new_h = int((height * cos) + (width * sin))
    offset_w = (new_w - width) / 2
    offset_h = (new_h - height) / 2
    rotation_matrix[0, 2] += offset_w
    rotation_matrix[1, 2] += offset_h
    v = np.vstack((x.ravel(), y.ravel(), np.ones(np.size(y.ravel()))))
    calculated = np.dot(rotation_matrix, v)
    qx = calculated[0]
    qy = calculated[1]

    return qx, qy


def retransform_bb(
    height, width, qx, qy, new_w, new_h, qx_offset, qy_offset, offset_w, offset_h, theta
):
    rotx = qx * new_w / 224
    roty = qy * new_h / 224
    rotx = rotx - qx_offset
    roty = roty - qy_offset
    rotx = rotx - offset_w
    roty = roty - offset_h
    rotx, roty = rotate_bb(rotx, roty, width, height, -theta)
    rotx = rotx - offset_w
    roty = roty - offset_h

    return rotx, roty


def get_scan_in_list(curr_scan_path, data_path, mat_path):
    # Check if mat files exist
    c_ = curr_scan_path[0]
    c_ = mat_path + "/" + c_[: c_.rfind("/")]
    if os.path.exists(c_):
        # Get volume - mat
        volume = np.array(
            spio.loadmat(c_ + "/scan.mat", verify_compressed_data_integrity=False)[
                "scan"
            ]
        )
    else:
        # Get volume - dicom
        if len(curr_scan_path) == 1:
            # 3D scans
            s_temp = pydicom.dcmread(data_path + "/" + curr_scan_path[0], force=True)
            if len(s_temp.file_meta) == 0:
                s_temp.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            volume = np.array(s_temp.pixel_array)
            if len(volume.shape) < 3:
                volume = volume[:, :, None]
            elif len(volume.shape) == 3:
                volume = np.transpose(np.array(volume), (1, 2, 0)).astype(float)
            else:
                print("Unknown volume shape")
        else:
            try:
                # 2D scans
                volume = []
                for temp_input in curr_scan_path:
                    # print(temp_input)
                    s_temp = pydicom.dcmread(data_path + "/" + temp_input, force=True)
                    if len(s_temp.file_meta) == 0:
                        s_temp.file_meta.TransferSyntaxUID = (
                            pydicom.uid.ImplicitVRLittleEndian
                        )
                    scan = np.array(s_temp.pixel_array)
                    volume.append(scan)
                volume = np.transpose(np.array(volume), (1, 2, 0)).astype(float)
            except:
                print(temp_input)
    return volume


def get_scan_in_folder(input_path):
    input_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    if len(input_files) == 1:
        # 3D scans
        s_temp = pydicom.dcmread(input_path + "/" + input_files[0], force=True)
        if len(s_temp.file_meta) == 0:
            s_temp.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        volume = np.array(s_temp.pixel_array)
        pixel_spacing = s_temp.PixelSpacing
    else:
        lis = []
        order = []
        for temp_input in input_files:
            s_temp = pydicom.dcmread(input_path + "/" + temp_input, force=True)
            # series_number = int(s_temp.SliceLocation)
            # order.append(series_number)
            series_number = float(s_temp.ImagePositionPatient[0])
            order.append(series_number)
            lis.append(temp_input)
        input_files = [x for _, x in sorted(zip(order, lis))]
        input_files.reverse()
        volume = []
        for temp_input in input_files:
            s_temp = pydicom.dcmread(input_path + "/" + temp_input, force=True)
            if len(s_temp.file_meta) == 0:
                s_temp.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            scan = np.array(s_temp.pixel_array)
            volume.append(scan)
            pixel_spacing = np.array(s_temp.PixelSpacing)
    volume = np.transpose(np.array(volume), (1, 2, 0)).astype(float)
    return {"volume": volume, "pixel_spacing": pixel_spacing}


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, shape
    )
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def get_vbs_intensity(volume, all_vb_x, all_vb_y, all_vb_mid, all_vb_label):
    vbs_intensity = []
    height, width, depth = volume.shape
    for vb_idx in range(all_vb_x.shape[1]):
        curr_label = all_vb_label[vb_idx, :]
        curr_mid = all_vb_mid[vb_idx]
        curr_mask = poly2mask(
            all_vb_y[:, vb_idx, int(curr_mid)],
            all_vb_x[:, vb_idx, int(curr_mid)],
            volume.shape[:2],
        )
        mask = np.tile(curr_mask, (depth, 1, 1))
        mask = np.transpose(mask, (1, 2, 0)).astype(float)
        mask[:, :, ~curr_label.astype(bool)] = 0
        vbs_intensity.append(volume[mask.astype(bool)])
    return vbs_intensity


def rotate_bb_and_volume(volume, x, y, normalise=True):
    height = volume.shape[0]
    width = volume.shape[1]
    depth = volume.shape[2]
    theta = np.degrees(np.arctan2(y[0] - y[3], x[0] - x[3]))

    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), theta, scale=1)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((height * sin) + (width * cos))
    new_h = int((height * cos) + (width * sin))
    offset_w = (new_w - width) / 2
    offset_h = (new_h - height) / 2
    rotation_matrix[0, 2] += offset_w
    rotation_matrix[1, 2] += offset_h
    v = np.vstack((x.ravel(), y.ravel(), np.ones(np.size(y.ravel()))))
    calculated = np.dot(rotation_matrix, v)
    qx = calculated[0]
    qy = calculated[1]

    volume_rot = []
    for idx in range(depth):
        if normalise:
            volume_rot.append(
                cv2.warpAffine(
                    volume[:, :, idx],
                    rotation_matrix,
                    (int(new_w), int(new_h)),
                    flags=cv2.INTER_CUBIC,
                )
            )
        else:
            volume_rot.append(
                cv2.warpAffine(
                    volume[:, :, idx],
                    rotation_matrix,
                    (int(new_w), int(new_h)),
                    flags=cv2.INTER_NEAREST,
                )
            )
    volume_rot = np.transpose(np.array(volume_rot), (1, 2, 0)).astype(float)

    min_x = min(qx)
    max_x = max(qx)
    qx[0] = max_x
    qx[1] = max_x
    qx[2] = min_x
    qx[3] = min_x

    min_y = min(qy)
    max_y = max(qy)
    qy[0] = min_y
    qy[1] = max_y
    qy[2] = max_y
    qy[3] = min_y

    return volume_rot, qx, qy


def get_ivd_vol(
    volume, ivd_curr_x, ivd_curr_y, vb_pair_median, curr_ivd_mid, norm_med, patch_size
):
    x = np.array(ivd_curr_x)
    y = np.array(ivd_curr_y)

    # Rotate
    volume_rot, qx, qy = rotate_bb_and_volume(volume, x, y)

    # Add 50% width
    w = max(qx) - min(qx)
    qx[0] += w * 0.5
    qx[1] += w * 0.5
    qx[2] -= w * 0.5
    qx[3] -= w * 0.5
    curr_w = max(qx) - min(qx)
    curr_h = max(qy) - min(qy)
    extra_vert = (curr_w / 2 - curr_h) * 0.5
    qy[0] -= extra_vert
    qy[1] += extra_vert
    qy[2] += extra_vert
    qy[3] -= extra_vert
    curr_w = max(qx) - min(qx)
    curr_h = max(qy) - min(qy)

    #  Jitter Crop: Border = [39.25 46.5]
    extra_w = ((patch_size[1] - 227.0) / 2.0) * (curr_w / 227.0)
    extra_h = ((patch_size[0] - (227.0 / 2.0)) / 2.0) * (curr_h / (227.0 / 2.0))
    min_x = np.round(min(qx) - extra_w)
    max_x = np.round(max(qx) + extra_w)
    min_y = np.round(min(qy) - extra_h)
    max_y = np.round(max(qy) + extra_h)
    curr_w = max_x - min_x
    curr_h = max_y - min_y

    if min_y < 0:
        y_offset = np.round(abs(min_y))
        volume_rot = np.concatenate(
            (
                np.zeros(
                    (y_offset.astype(int), volume_rot.shape[1], volume_rot.shape[2])
                ),
                volume_rot,
            ),
            axis=0,
        )
        min_y += y_offset
        max_y += y_offset
    if max_y >= volume_rot.shape[0]:
        y_offset = (abs(max_y) + 1) - volume_rot.shape[0]
        volume_rot = np.concatenate(
            (
                volume_rot,
                np.zeros((y_offset, volume_rot.shape[1], volume_rot.shape[2])),
            ),
            axis=0,
        )
    if min_x < 0:
        x_offset = np.round(abs(min_x))
        volume_rot = np.concatenate(
            (
                np.zeros(
                    (volume_rot.shape[0], x_offset.astype(int), volume_rot.shape[2])
                ),
                volume_rot,
            ),
            axis=1,
        )
        min_x += x_offset
        max_x += x_offset
    if max_x >= volume_rot.shape[1]:
        x_offset = (abs(max_x) + 1) - volume_rot.shape[1]
        volume_rot = np.concatenate(
            (
                volume_rot,
                np.zeros((volume_rot.shape[0], x_offset, volume_rot.shape[2])),
            ),
            axis=1,
        )

    min_x = np.round(min_x).astype(int)
    max_x = np.round(max_x).astype(int)
    min_y = np.round(min_y).astype(int)
    max_y = np.round(max_y).astype(int)
    sub_vol = volume_rot[min_y:max_y, min_x:max_x, :]

    # Resize
    ivd_vol = []
    for idx in range(sub_vol.shape[2]):
        ivd_vol.append(
            cv2.resize(
                sub_vol[:, :, idx], patch_size[::-1], interpolation=cv2.INTER_CUBIC
            )
        )
    ivd_vol = np.transpose(np.array(ivd_vol), (1, 2, 0)).astype(float)

    # Normalize
    ivd_vol = ivd_vol / vb_pair_median * norm_med
    ivd_vol[ivd_vol < 0] = 0
    ivd_vol[ivd_vol > 2.0] = 2.0
    ivd_vol /= 2.0
    return ivd_vol


def get_all_ivd_vol(volume, all_vb_x, all_vb_y, all_vb_mid, all_vb_label):
    # VBs intensity values
    vbs_intensity = get_vbs_intensity(
        volume, all_vb_x, all_vb_y, all_vb_mid, all_vb_label
    )

    # Get volumes
    ivds = []
    norm_med = 0.5
    patch_size = (192, 320)
    no_of_ivd = len(vbs_intensity) - 1
    for ivd_idx in range(no_of_ivd):
        curr_ivd_mid = np.round(
            np.mean([all_vb_mid[ivd_idx], all_vb_mid[ivd_idx + 1]])
        ).astype(int)
        vb_pair_intensity = np.concatenate(
            (vbs_intensity[ivd_idx], vbs_intensity[ivd_idx + 1])
        )
        vb_pair_median = np.median(vb_pair_intensity)

        vb_curr_x = all_vb_x[:, ivd_idx, curr_ivd_mid]
        vb_curr_y = all_vb_y[:, ivd_idx, curr_ivd_mid]
        vb_next_x = all_vb_x[:, ivd_idx + 1, curr_ivd_mid]
        vb_next_y = all_vb_y[:, ivd_idx + 1, curr_ivd_mid]
        ivd_curr_x = [vb_next_x[1], vb_curr_x[0], vb_curr_x[3], vb_next_x[2]]
        ivd_curr_y = [vb_next_y[1], vb_curr_y[0], vb_curr_y[3], vb_next_y[2]]

        ivd_vol = get_ivd_vol(
            volume,
            ivd_curr_x,
            ivd_curr_y,
            vb_pair_median,
            curr_ivd_mid,
            norm_med,
            patch_size,
        )

        # Centered & choose 15 slices; TO-DO: resize 3D
        ivd_vol_centered = []
        for j in range(-7, 7 + 1, 1):
            curr_slice = curr_ivd_mid + j
            if (curr_slice < 0) | (curr_slice >= ivd_vol.shape[2]):
                temp_IVD = np.zeros(patch_size)
            else:
                temp_IVD = ivd_vol[:, :, curr_slice]
            ivd_vol_centered.append(temp_IVD)
        ivd_vol_centered = np.transpose(np.array(ivd_vol_centered), (1, 2, 0)).astype(
            float
        )
        num_rows, num_cols, num_slices = ivd_vol_centered.shape
        max_cols = num_cols - 48
        min_cols = 48
        max_rows = num_rows - 40
        min_rows = 40
        ivd_vol_centered = ivd_vol_centered[min_rows:max_rows, min_cols:max_cols, 3:12]
        ivd_vol_centered = np.transpose(ivd_vol_centered, (2, 0, 1))
        ivds.append(ivd_vol_centered)
    ivds = np.array(ivds)
    return ivds


def get_vu_vol(
    volume,
    ivd_curr_x,
    ivd_curr_y,
    vb_pair_median,
    curr_ivd_mid,
    norm_med,
    patch_size,
    normalise=True,
):
    x = np.array(ivd_curr_x)
    y = np.array(ivd_curr_y)

    # Rotate
    volume_rot, qx, qy = rotate_bb_and_volume(volume, x, y, normalise=normalise)

    min_x = np.round(min(qx))
    max_x = np.round(max(qx))
    min_y = np.round(min(qy))
    max_y = np.round(max(qy))
    w = max_x - min_x
    h = max_y - min_y

    if w > h:
        max_y += (w - h) / 2.0
        min_y -= (w - h) / 2.0
        h = max_y - min_y
    else:
        max_x += (h - w) / 2.0
        min_x -= (h - w) / 2.0
        w = max_x - min_x

    # Add 50% width/height
    max_x += w * 0.5
    min_x -= w * 0.5
    max_y += h * 0.5
    min_y -= h * 0.5

    if min_y < 0:
        y_offset = abs(min_y).astype(int)
        volume_rot = np.concatenate(
            (
                np.zeros((y_offset, volume_rot.shape[1], volume_rot.shape[2])),
                volume_rot,
            ),
            axis=0,
        )
        min_y += y_offset
        max_y += y_offset
    if max_y >= volume_rot.shape[0]:
        y_offset = ((abs(max_y) + 1) - volume_rot.shape[0]).astype(int)
        volume_rot = np.concatenate(
            (
                volume_rot,
                np.zeros((y_offset, volume_rot.shape[1], volume_rot.shape[2])),
            ),
            axis=0,
        )
    if min_x < 0:
        x_offset = abs(min_x).astype(int)
        volume_rot = np.concatenate(
            (
                np.zeros((volume_rot.shape[0], x_offset, volume_rot.shape[2])),
                volume_rot,
            ),
            axis=1,
        )
        min_x += x_offset
        max_x += x_offset
    if max_x >= volume_rot.shape[1]:
        x_offset = ((abs(max_x) + 1) - volume_rot.shape[1]).astype(int)
        volume_rot = np.concatenate(
            (
                volume_rot,
                np.zeros((volume_rot.shape[0], x_offset, volume_rot.shape[2])),
            ),
            axis=1,
        )

    min_x = np.round(min_x).astype(int)
    max_x = np.round(max_x).astype(int)
    min_y = np.round(min_y).astype(int)
    max_y = np.round(max_y).astype(int)
    sub_vol = volume_rot[min_y:max_y, min_x:max_x, :]

    resize_factors = (
        15 / sub_vol.shape[2],
        patch_size[0] / sub_vol.shape[0],
        patch_size[1] / sub_vol.shape[1],
    )
    # Resize
    ivd_vol = []
    for idx in range(sub_vol.shape[2]):
        if normalise:
            ivd_vol.append(
                cv2.resize(
                    sub_vol[:, :, idx], patch_size[::-1], interpolation=cv2.INTER_CUBIC
                )
            )
        else:
            ivd_vol.append(
                cv2.resize(
                    sub_vol[:, :, idx],
                    patch_size[::-1],
                    interpolation=cv2.INTER_NEAREST,
                )
            )
    ivd_vol = np.transpose(np.array(ivd_vol), (1, 2, 0)).astype(float)

    # Normalize
    if normalise:
        ivd_vol = ivd_vol / vb_pair_median * norm_med
        ivd_vol[ivd_vol < 0] = 0
        ivd_vol[ivd_vol > 2.0] = 2.0
        ivd_vol /= 2.0
    return ivd_vol, resize_factors


def get_all_vus_vol(
    volume, all_vb_x, all_vb_y, all_vb_mid, all_vb_label, normalise=True
):
    # VBs intensity values
    vbs_intensity = get_vbs_intensity(
        volume, all_vb_x, all_vb_y, all_vb_mid, all_vb_label
    )

    # Get IVD volumes
    vus = []
    all_resize_factors = []
    norm_med = 0.5
    patch_size = (320, 320)
    no_of_ivd = len(vbs_intensity)
    for ivd_idx in range(no_of_ivd):
        if ivd_idx != (no_of_ivd - 1):
            curr_ivd_mid = np.round(
                np.mean([all_vb_mid[ivd_idx], all_vb_mid[ivd_idx + 1]])
            ).astype(int)
            vb_pair_intensity = np.concatenate(
                (vbs_intensity[ivd_idx], vbs_intensity[ivd_idx + 1])
            )
            vb_pair_median = np.median(vb_pair_intensity)

            vb_curr_x = all_vb_x[:, ivd_idx, curr_ivd_mid]
            vb_curr_y = all_vb_y[:, ivd_idx, curr_ivd_mid]
            vb_next_x = all_vb_x[:, ivd_idx + 1, curr_ivd_mid]
            vb_next_y = all_vb_y[:, ivd_idx + 1, curr_ivd_mid]

            vu_curr_x = [
                np.mean([vb_next_x[0], vb_next_x[1]]),
                np.mean([vb_curr_x[0], vb_curr_x[1]]),
                np.mean([vb_curr_x[2], vb_curr_x[3]]),
                np.mean([vb_next_x[2], vb_next_x[3]]),
            ]
            vu_curr_y = [
                np.mean([vb_next_y[0], vb_next_y[1]]),
                np.mean([vb_curr_y[0], vb_curr_y[1]]),
                np.mean([vb_curr_y[2], vb_curr_y[3]]),
                np.mean([vb_next_y[2], vb_next_y[3]]),
            ]

        else:
            curr_ivd_mid = np.round(all_vb_mid[ivd_idx]).astype(int)
            vb_pair_median = np.median(vbs_intensity[ivd_idx])

            vb_curr_x = all_vb_x[:, ivd_idx, curr_ivd_mid]
            vb_curr_y = all_vb_y[:, ivd_idx, curr_ivd_mid]

            x0 = vb_curr_x[0]
            y0 = vb_curr_y[0]
            x1 = np.mean([vb_curr_x[0], vb_curr_x[1]])
            y1 = np.mean([vb_curr_y[0], vb_curr_y[1]])
            x2 = np.mean([vb_curr_x[2], vb_curr_x[3]])
            y2 = np.mean([vb_curr_y[2], vb_curr_y[3]])
            x3 = vb_curr_x[3]
            y3 = vb_curr_y[3]
            x0 = x0 + 2 * (x0 - x1)
            y0 = y0 + 2 * (y0 - y1)
            x3 = x3 + 2 * (x3 - x2)
            y3 = y3 + 2 * (y3 - y2)

            vu_curr_x = [x0, x1, x2, x3]
            vu_curr_y = [y0, y1, y2, y3]

        vu_vol, resize_factors = get_vu_vol(
            volume,
            vu_curr_x,
            vu_curr_y,
            vb_pair_median,
            curr_ivd_mid,
            norm_med,
            patch_size,
            normalise=normalise,
        )
        all_resize_factors.append(resize_factors)

        # Resize to 15 slices
        ori_depth = vu_vol.shape[2]
        dsfactor = [
            w / float(f)
            for w, f in zip([vu_vol.shape[0], vu_vol.shape[1], 15], vu_vol.shape)
        ]
        vu_vol = nd.interpolation.zoom(vu_vol, dsfactor)
        curr_ivd_mid = np.round((curr_ivd_mid / ori_depth) * 15).astype(int)

        # Centered & choose 15 slices
        vu_vol_centered = []
        for j in range(-7, 7 + 1, 1):
            curr_slice = curr_ivd_mid + j
            if (curr_slice < 0) | (curr_slice >= vu_vol.shape[2]):
                temp_vu = np.zeros(patch_size)
            else:
                temp_vu = vu_vol[:, :, curr_slice]
            vu_vol_centered.append(temp_vu)
        vu_vol_centered = np.transpose(np.array(vu_vol_centered), (1, 2, 0)).astype(
            float
        )
        vus.append(vu_vol_centered)
    vus = np.array(vus)
    return vus, all_resize_factors


def vert_dicts_to_classification_format(vert_dicts, no_slices):
    """Converts detection output of VFR detector to format that can be read by the classification pipeline"""
    vert_dicts = sorted(
        vert_dicts, key=lambda x: np.array(x["average_polygon"])[:, 1].mean(), reverse=True
    )
    all_vb_x = np.zeros((4, len(vert_dicts), no_slices))
    all_vb_y = np.zeros((4, len(vert_dicts), no_slices))
    all_vb_label = np.zeros((len(vert_dicts), no_slices))
    all_vb_mid = np.zeros(len(vert_dicts))
    for vert_dict_idx, vert_dict in enumerate(vert_dicts):
        # begin by filling all entries with the average polygon
        all_vb_x[:, vert_dict_idx, :] = np.stack(
            [np.array(vert_dict["average_polygon"])[:, 0]] * no_slices, axis=-1
        )
        all_vb_y[:, vert_dict_idx, :] = np.stack(
            [np.array(vert_dict["average_polygon"])[:, 1]] * no_slices, axis=-1
        )
        for poly_idx, poly in enumerate(vert_dict["polys"]):
            slice_no = vert_dict["slice_nos"][poly_idx]
            poly = np.array(poly)
            all_vb_x[:, vert_dict_idx, slice_no] = poly[:, 0]
            all_vb_y[:, vert_dict_idx, slice_no] = poly[:, 1]
            all_vb_label[vert_dict_idx, slice_no] = 1

        all_vb_mid[vert_dict_idx] = np.round(np.median(vert_dict["slice_nos"]))
    vb_level_names = [vert_dict["predicted_label"] for vert_dict in vert_dicts]

    return all_vb_x, all_vb_y, all_vb_mid, all_vb_label, vb_level_names


def get_ivd_level_names(vu_level_names):
    ivd_level_names = []
    for idx in range(len(vu_level_names)):
        if idx == 0:
            continue
        ivd_level_names.append(f"{vu_level_names[idx]}-{vu_level_names[idx-1]}")
    return ivd_level_names


def format_gradings(gradings, ivd_level_names):
    ivd_level_names.reverse()
    col = []
    zl = []
    for key in gradings:
        gradings[key] = np.flip(gradings[key])
        if key in ["Pfirrmann", "Narrowing", "CentralCanalStenosis"]:
            gradings[key] += 1
        col.append(key)
        zl.append(gradings[key])
    gradings = pd.DataFrame(list(zip(*zl)), index=ivd_level_names, columns=col)
    return gradings
