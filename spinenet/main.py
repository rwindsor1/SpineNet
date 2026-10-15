import os
import sys
import glob
import torch
import numpy as np
from scipy import io as sio
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any
from typing_extensions import TypedDict


IVDDicts = TypedDict('IVDDicts', {'volume': np.array, 'level_name': str})
VertDicts = TypedDict('VertDicts', {})

sys.path.append(str(Path(__file__).parent))

# Ladder
from .utils import *


class SpineNet:

    def __init__(
        self, device: bool = "cuda:0", verbose: bool = True, scan_type: str = "lumbar"
    ) -> None:
        """
        Initialize instance of spinenet for (1) detecting and labelling vertebrae 
        (2) Performing radiological grading for common spinal degenerative changes in T2 sagittal lumbar scans.

        Parameters
        ----------
        device : str, optional
            The pytorch-style device to use for the model. The default is "cuda:0". If you not using CUDA-enabled machine, you can use "cpu" (although this will slow performance).
        verbose : bool, optional
            Whether to print out information regarding the pipeline. The default is True.
        scan_type : str, optional
            The type of scan to use. The default is "lumbar". Can also be "whole"

        """

        assert scan_type in [
            "lumbar",
            "whole",
        ], "scan_type should be either 'lumbar' or 'whole'"
        self.device = device
        if "cuda" in device:
            torch.cuda.set_device(device)
        # the value of the threshold of corner channels in the detector network
        self.corner_threshold = 0.6
        # the value of the threshold of centroid channels in the detector network
        self.centroid_threshold = 0.6
        # the amount of overlap two polygons need to have to be grouped together
        self.group_across_slices_threshold = 0.2
        self.temp_path = "./temp"  # the path to save temporary files to

        # sets whether to split scan into pieces
        if scan_type == "lumbar":
            self.remove_black_space = False
        else:
            self.remove_black_space = True

        # Setup - Detector Model
        from .models.vfr import VFRResNetDetector

        detection_weights_path = os.path.join(
            os.path.dirname(__file__), "weights/detect-vfr"
        )
        self.detection_model = VFRResNetDetector()
        if verbose:
            print("Loading Detection Model...")
        self.detection_model.load_weights(detection_weights_path, verbose=verbose)
        self.detection_model.eval()
        self.detection_model.to(device)

        # Setup - Appearance Model
        from .models.appearance import AppearanceModel

        appearance_weights_path = os.path.join(
            os.path.dirname(__file__), "weights/appearance"
        )
        self.appearance_model = AppearanceModel()
        if verbose:
            print("Loading Appearance Model...")
        self.appearance_model.load_weights(appearance_weights_path, verbose=verbose)
        self.appearance_model.eval()
        self.appearance_model.to(device)

        # Setup - Context Model
        from .models.context import ContextModel

        context_weights_path = os.path.join(
            os.path.dirname(__file__), "weights/context"
        )
        self.context_model = ContextModel()
        if verbose:
            print("Loading Context Model...")
        self.context_model.load_weights(context_weights_path, verbose=verbose)
        self.context_model.eval()
        self.context_model.to(device)


        # Setup - Grading Model
        from .models.grading import GradingModel

        grading_weights_path = os.path.join(
            os.path.dirname(__file__), "weights/grading"
        )
        self.grading_model = GradingModel()
        if verbose:
            print("Loading Grading Model...")
        self.grading_model.load_weights(grading_weights_path, verbose=verbose)
        self.grading_model.eval()
        self.grading_model.to(device)

    def detect_vb(
        self,
        volume : np.ndarray,
        pixel_spacing : Union[np.ndarray, List[float], torch.Tensor],
        debug: bool = False,
        penalise_skips: bool = True,
        remove_single_slice_detections: bool = True,
    ) -> VertDicts:
        """
        Use SpineNet to detect and label vertebral bodies in a volume.

        Parameters
        ----------
        volume : np.ndarray
            The volume to detect vertebrae in. Should have shape (height,width, number of sagittal slices).
        pixel_spacing : Union[np.ndarray, List[float], torch.Tensor]
            The pixel spacing of the volume, specifically the distance between adjacent pixels in the sagittal direction.
            This has order height, width
        debug : bool, optional
            Whether to print out debug information. The default is False.
        penalise_skips : bool, optional
            Whether to penalise skipping a vertebrae in the detection sequence. Essentially, enabling this ensures the vertebrae labels will be in order (i.e. T12, L5, L4, etc).
            However, this may not be desired if you expect to miss detections (perhaps due to image artifacts)

        Returns
        -------
        VertDicts
            A list of dictionaries containing the vertebrae labels, their corresponding polygons and the slices in which they appear.
        """

        assert volume.ndim == 3, "scan should be a 3-dimensional array of shape HxWxS."
        detect_ans = detect_and_group(
            self.detection_model,
            volume,
            remove_excess_black_space=self.remove_black_space,
            plot_outputs=False,
            using_resnet=True,
            corner_threshold=self.corner_threshold,
            centroid_threshold=self.centroid_threshold,
            group_across_slices_threshold=self.group_across_slices_threshold,
            remove_single_slice_detections=remove_single_slice_detections,
            pixel_spacing=pixel_spacing,
            device=self.device,
            debug=debug,
        )
        if debug:
            vert_dicts, patches, patches_dicts, detection_dicts, transform_info_dicts = detect_ans
        else:
            vert_dicts = detect_ans

        vert_dicts = extract_volumes(volume, vert_dicts, pixel_spacing=pixel_spacing)
        labels_ans = label_verts(
            vert_dicts,
            volume,
            pixel_spacing,
            self.appearance_model,
            self.context_model,
            plot_outputs=False,
            penalise_skips=penalise_skips,
            debug=debug,
        )
        if debug:
            (
                vert_dicts,
                height_scaled_appearance_features,
                context_output,
                context_features,
                seq_pred,
            ) = labels_ans
        else:
            vert_dicts = labels_ans

        if debug:
            return (
                vert_dicts,
                patches,
                patches_dicts,
                detection_dicts,
                transform_info_dicts,
                height_scaled_appearance_features,
                context_output,
                context_features,
                seq_pred,
            )
        else:
            return vert_dicts

    def get_ivds_from_vert_dicts(
        self,
        vert_dicts: VertDicts,
        scan_volume: np.ndarray,
        output_shape: Tuple[int, int, int] = (9, 112, 224),
        pixel_spacing: Union[np.ndarray, List[float], torch.Tensor, None] = None,
    ) -> IVDDicts:
        '''
        Use detected vertebrae from SpineNet's detect_vb function to generate volumes surrounding each IVD.

        Parameters
        ----------
        vert_dicts : VertDicts  
            The vertebrae labels, their corresponding polygons and the slices in which they appear. Generated by SpineNet.detect_vb function
        scan_volume : np.ndarray
            The volume to extract the IVDs from. Should have shape (height,width, number of sagittal slices).
        output_shape : Tuple[int, int, int], optional
            The (slices, height, width) shape of each IVD volume. Other shapes
            crop more or less of the scan around the IVD at the same scale,
            using native sagittal slices (no resampling across slices). Only
            the default (9, 112, 224) can be graded with `grade_ivds`.
        pixel_spacing : Union[np.ndarray, List[float], torch.Tensor], optional
            The pixel spacing of the volume, in order height, width, as given
            to `detect_vb`. Needed if the two differ: the IVDs are then
            extracted on square pixels of the finer of the two. If not given,
            the pixels are taken to be square.
        
        Returns
        -------
        IVDDicts
            A list of dictionaries containing resampled IVD volumes and the corresponding names
            and 'voxel_to_scan', the 4x4 affine from IVD (slice, row, col) to scan (row, col, slice)
        '''
        if len(output_shape) != 3 or min(output_shape) < 1:
            raise ValueError(
                f"output_shape must be 3 positive ints, got {output_shape}"
            )
        (
            all_vb_x,
            all_vb_y,
            all_vb_mid,
            all_vb_label,
            vb_level_names,
        ) = vert_dicts_to_classification_format(vert_dicts, scan_volume.shape[-1])
        ivds, ivd_affines = get_all_ivd_vol(
            scan_volume,
            all_vb_x,
            all_vb_y,
            all_vb_mid,
            all_vb_label,
            output_shape,
            return_affines=True,
            pixel_spacing=pixel_spacing,
        )
        ivd_level_names = get_ivd_level_names(vb_level_names)
        ivd_dicts = [{"volume": ivd, "level_name": level_name} for ivd, level_name in zip(ivds, ivd_level_names)]
        for ivd_dict, affine in zip(ivd_dicts, ivd_affines):
            ivd_dict["voxel_to_scan"] = affine
        return ivd_dicts

    def grade_ivds(self, ivd_dicts : IVDDicts) -> pd.DataFrame:
        """
        Grade all intervertebral discs extracted from a volume. See the technical paper for details on the specific grading schemes: https://arxiv.org/abs/2205.01683.

        Parameters
        ----------
        ivd_dicts : IVDDicts
            A list of dictionaries containing resampled IVD volumes and the corresponding names. Generated by SpineNet.get_ivds_from_vert_dicts function

        Returns
        -------
        gradings: pd.DataFrame
            A pandas dataframe containing the IVD names and their corresponding grades.
        """
        for ivd_dict in ivd_dicts:
            shape = np.shape(ivd_dict['volume'])
            if shape != (9, 112, 224):
                raise ValueError(
                    f"grade_ivds needs (9, 112, 224) IVD volumes, got {shape}"
                )
        gradings = classify_ivd_v2_resnet(self.grading_model, [ivd_dict['volume'] for ivd_dict in ivd_dicts], self.device)
        gradings = format_gradings(gradings, [ivd_dict['level_name'] for ivd_dict in ivd_dicts])
        return gradings
