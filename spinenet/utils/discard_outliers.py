import numpy as np
from numpy import polyfit, polyval


def discard_outliers(vert_dicts, polyfit_degree=5):
    """
    Fits curve through detected vertebra and discards outliers that are far from curve

    This is done by using a quintic curve in the sagittal plane, fit to the vertebra
    centroids. If no part of the curve passes through a detection box, it is discarded

    Inputs
    -------
    vert_dicts : list of dicts
        output of utils/detect_and group function. List of dictionaries where each element
        describes a different vertebra detected

    Outputs
    -------
    vert_dicts : list of dicts
        same as input but with outlier vertebrae removed
    """

    top_points = np.zeros((len(vert_dicts), 2))
    bottom_points = np.zeros((len(vert_dicts), 2))
    for vert_idx, vert in enumerate(vert_dicts):
        top_points[vert_idx, :] = np.mean(
            np.take(vert["average_polygon"], [0, 3], axis=0), axis=0
        )
        bottom_points[vert_idx, :] = np.mean(
            np.take(vert["average_polygon"], [1, 2], axis=0), axis=0
        )
        # top_points[vert_idx, :] = np.mean(vert['average_polygon'], axis=0)

    points = np.concatenate((top_points, bottom_points), axis=0)
    xs = points[:, 1]
    ys = points[:, 0]
    if len(xs) == 0:
        # if there are no vertebrae, quit function here
        return vert_dicts, None
    cooefs = polyfit(xs, ys, deg=polyfit_degree)
    distances_from_curve = np.array(
        [np.abs(polyval(cooefs, xs[i]) - ys[i]) for i in range(len(vert_dicts))]
    )

    dropped_indices = []
    # for each vert check the curve passes through them. If not, discard them
    for vert_idx, vert in enumerate(vert_dicts):
        average_polygon = vert["average_polygon"]
        passes_top = False
        passes_bottom = False
        # check top
        if average_polygon[0, 0] > polyval(
            cooefs, average_polygon[0, 1]
        ) and average_polygon[-1, 0] < polyval(cooefs, average_polygon[-1, 1]):
            passes_top = True
        # check bottom
        if average_polygon[1, 0] > polyval(
            cooefs, average_polygon[1, 1]
        ) and average_polygon[2, 0] < polyval(cooefs, average_polygon[2, 1]):
            passes_bottom = True
        if passes_top or passes_bottom:
            continue
        else:
            dropped_indices.append(vert_idx)

    dropped_indices.sort(reverse=True)
    for dropped_index in dropped_indices:
        vert_dicts.pop(dropped_index)
    # fit curve to the points
    return vert_dicts
