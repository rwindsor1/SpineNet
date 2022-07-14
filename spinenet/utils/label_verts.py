import torch
import numpy as np
import matplotlib.pyplot as plt

vert_names = [
    "S1",
    "L5",
    "L4",
    "L3",
    "L2",
    "L1",
    "T12",
    "T11",
    "T10",
    "T9",
    "T8",
    "T7",
    "T6",
    "T5",
    "T4",
    "T3",
    "T2",
    "T1",
    "C7",
    "C6",
    "C5",
    "C4",
    "C3",
    "C2",
]


def label_verts(
    vert_dicts,
    scan,
    pixel_spacing,
    appearance_net,
    context_net,
    plot_outputs=False,
    penalise_skips=False,
    debug=False,
):
    """
    Run the labelling pipeline to extract labels from each detection

    Parameters:
    -----------
    vert_dicts : list of dictionaries
        The output from the detection stage. Each dictionary contains information about each vertebra detected.
    scan : np.array
        HxWxS image containing the scan
    pixel_spacing : float
        the mm between each pixel in each sagittal slice (assumes isotropic in slice)
    appearance_net : torch.nn.Module
        the 3D apperaance network to guess a vert level from its surrounding area alone
    context_net : torch.nn.Module
        the context network used to refine predictions of the appearance_network
    plot_outputs : bool
        set to true to show guesses of appearance and refine ment network in a plot
    debug: whether to return information about the network outputs to see what
        went wrong


    Outputs
    -------
    vert_dicts : list of dictionaries
        same as the input to function but with a new key, predicted label added
    """

    # firstly get the labels from the apperance net
    vert_dicts = conv_appearance_labelling(vert_dicts, appearance_net, softmax_temp=10)
    # now construct input to the context model
    height_scaled_appearance_features, y_centroids = construct_input_to_context_model(
        vert_dicts, scan, pixel_spacing
    )
    # pad image to make it a multiple of 256 in height so it can be scaled down by U-Net
    diff = height_scaled_appearance_features.shape[0] % 256
    if diff != 0:
        height_scaled_appearance_features = np.concatenate(
            (height_scaled_appearance_features, np.zeros((256 - diff, 24))), axis=0
        )

    # run context network
    context_net_device = next(context_net.parameters()).device
    with torch.no_grad():
        context_output = context_net(
            torch.tensor(height_scaled_appearance_features)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(context_net_device)
        )

    context_features = [
        context_output[0, 0, int(y_centroid), :].tolist() for y_centroid in y_centroids
    ]
    for idx, vert_dict in enumerate(vert_dicts):
        vert_dict["context_features"] = context_features[idx]
        vert_dict["appearance_guess"] = vert_names[
            torch.argmax(vert_dict["visual_features"]).item()
        ]
        vert_dict["context_guess"] = vert_names[
            np.argmax(vert_dict["context_features"])
        ]

    # apply softmax with temp 10 to the output
    temperature = 10
    vert_centre_features = [
        torch.softmax(
            context_output[0, 0, int(y_centroid), :] / temperature, dim=0
        ).tolist()
        for y_centroid in y_centroids
    ]

    # perform beam search
    (
        sequence_predictions,
        same_both_directions,
        best_up_sequence,
        best_down_sequence,
    ) = two_way_beam_search(vert_centre_features, 100, penalise_skips=penalise_skips)
    # sequence_predictions = sequence_predictions[0]
    vert_centre_predictions = sequence_predictions[0][0]
    for idx, vert_dict in enumerate(vert_dicts):
        vert_dict["predicted_label"] = vert_names[vert_centre_predictions[idx]]

    # label S2 if we have 2 S1s
    if len(vert_dicts) > 1:
        if (
            vert_dicts[-1]["predicted_label"] == "S1"
            and vert_dicts[-2]["predicted_label"] == "S1"
        ):
            vert_dicts[-1]["predicted_label"] = "S2"

        if (
            vert_dicts[0]["predicted_label"] == "S1"
            and vert_dicts[1]["predicted_label"] == "S1"
        ):
            vert_dicts[0]["predicted_label"] = "S2"

    if plot_outputs:
        plt.figure(3, figsize=(15, 15))
        plt.subplot(121)
        plt.imshow(height_scaled_appearance_features)
        plt.subplot(122)
        plt.imshow(context_output[0, 0].detach().cpu().numpy())

    if best_up_sequence[0] == best_down_sequence[0]:
        same_both_directions = True
    else:
        same_both_directions = False

    if debug:
        return (
            vert_dicts,
            height_scaled_appearance_features,
            context_output[0, 0].detach().cpu().numpy(),
            vert_centre_features,
            sequence_predictions,
        )
    else:
        # remove keys that are not needed; 'appearance_guess', 'context_guess', 'context_features','appearance_features', 'visual_features'
        for vert_dict_idx, vert_dict in enumerate(vert_dicts):
            for key in list(vert_dict.keys()):
                if key not in ["predicted_label", "slice_nos", "polys","average_polygon"]:
                    del vert_dict[key]
            vert_dicts[vert_dict_idx] = vert_dict
        return vert_dicts


def conv_appearance_labelling(
    vert_dicts, appearance_net, softmax_temp=20, big_features=False
):
    appearance_net_device = next(appearance_net.parameters()).device
    volumes = [vert_dict["volume"] for vert_dict in vert_dicts]
    empty_vol_idxs = [
        idx for idx in range(len(vert_dicts)) if (vert_dicts[idx]["volume"].sum() == 0)
    ]
    appearance_net.eval()
    with torch.no_grad():
        appearance_predictions = []
        for volume in volumes:
            if big_features:
                appearance_predictions.append(
                    appearance_net.get_appearance_features(
                        torch.tensor(volume)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .float()
                        .to(appearance_net_device)
                    )
                )
            else:
                appearance_predictions.append(
                    appearance_net(
                        torch.tensor(volume)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .float()
                        .to(appearance_net_device)
                    )
                )
        visual_features = torch.cat(appearance_predictions, dim=0)
        appearance_predictions = torch.softmax(
            torch.cat(appearance_predictions, dim=0) / softmax_temp, dim=-1
        )

    for vert_dict_idx, vert_dict in enumerate(vert_dicts):
        vert_dict["appearance_features"] = appearance_predictions[vert_dict_idx]
        vert_dict["visual_features"] = visual_features[vert_dict_idx].cpu()

    for idx in empty_vol_idxs:
        vert_dicts[idx]["appearance_features"] = torch.zeros_like(
            vert_dicts[idx]["appearance_features"]
        ).float()
        vert_dicts[idx]["visual_features"] = (
            torch.zeros_like(vert_dicts[idx]["visual_features"]).float().cpu()
        )

    return vert_dicts


def construct_input_to_context_model(vert_dicts, scan, pixel_spacing):
    """
    Makes the image to be input into the context model.

    Inputs
    ------
    vert_dicts: list of dictionaries
        dictionaries containing information on each detected vertebra. Data structure output by detection pipeline.
    scan: np.array
        HxWxS size array containing the scan. H is height, W is width, S is sagittal slices
    pixel_spacing: float
        The distance between each pixel in mm

    Outputs
    -------
    height_scaled_appearance_features : np.array
        a HxW array containing the image used as input to the context array
    y_centroids: list
        a list of the y-coords of the centroids of each vertebra
    """

    appearance_features = [vert_dict["appearance_features"] for vert_dict in vert_dicts]
    box_coords = np.asarray([vert_dict["average_polygon"] for vert_dict in vert_dicts])
    y_centroids = (
        np.asarray([np.mean(box_coord[:, 1]) for box_coord in box_coords])
        * pixel_spacing
    )
    y_maxes = (
        np.asarray([np.max(box_coord[:, 1]) for box_coord in box_coords])
        * pixel_spacing
    )
    y_mins = (
        np.asarray([np.min(box_coord[:, 1]) for box_coord in box_coords])
        * pixel_spacing
    )
    widths = (y_maxes - y_mins) / 2
    image_height = int(scan.shape[0] * pixel_spacing)
    height_scaled_appearance_features = np.zeros((int(image_height), 24))

    for i in range(len(vert_dicts)):
        height_scaled_appearance_features[
            int(y_centroids[i])
            - int(widths[i] / 2) : int(y_centroids[i])
            + int(widths[i] / 2),
            :,
        ] = (
            appearance_features[i].cpu().numpy()
        )

    return height_scaled_appearance_features, y_centroids


def beam_search_decoder(data, k, search_type="descending", penalise_skips=True):
    assert search_type in [
        "ascending",
        "descending",
    ], "Search type should be either 'ascending' or 'descending'"
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score - np.log(row[j])]
                if len(seq):
                    if search_type == "descending":
                        if j >= seq[-1] and j != 0:
                            candidate[1] += 1000
                        if j >= seq[-1] and j == 0:
                            candidate[1] += 100
                        if penalise_skips:
                            if np.abs(j - seq[-1]) > 1:
                                candidate[1] += 100

                    else:
                        if j <= seq[-1]:
                            candidate[1] += 1000
                        if penalise_skips:
                            if np.abs(j - seq[-1]) > 1:
                                candidate[1] += 100

                all_candidates.append(candidate)
        #  order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]

    return sequences


def two_way_beam_search(data, k, down_only=True, penalise_skips=False):
    down_sequences = beam_search_decoder(
        data, k, search_type="descending", penalise_skips=penalise_skips
    )
    if down_only:
        return down_sequences, True, down_sequences[0], down_sequences[0]
    data = list(reversed(data))
    reversed_up_sequences = beam_search_decoder(
        data, k, search_type="ascending", penalise_skips=penalise_skips
    )
    up_sequences = []
    for sequence in reversed_up_sequences:
        up_sequences.append([list(reversed(sequence[0])), sequence[1]])

    best_up_sequence = sorted(up_sequences, key=lambda tup: tup[1])[0]
    best_down_sequence = sorted(down_sequences, key=lambda tup: tup[1])[0]

    if best_up_sequence == best_down_sequence:
        same_both_directions = True
    else:
        same_both_directions = False

    all_sequences = up_sequences + down_sequences
    all_sequences = sorted(all_sequences, key=lambda tup: tup[1])
