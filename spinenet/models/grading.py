import os
import glob
import torch
import torch.nn as nn
from collections.abc import Iterable, Iterator
from typing import Tuple, List


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        (3, 3, 3),
        stride=(1, stride, stride),
        padding=(1, dilation, dilation),
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        (1, 1, 1),
        stride=(1, stride, stride),
        padding=(0, 0, 0),
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class GradingModel(nn.Module):
    def __init__(
        self,
        block=BasicBlock,
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 2,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group=64,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(
            1, self.inplanes, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
        #                                dilate=replace_stride_with_dilation[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=1, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_pf = nn.Linear(512 * block.expansion, 5)
        self.fc_nar = nn.Linear(512 * block.expansion, 4)
        self.fc_ccs = nn.Linear(512 * block.expansion, 4)
        self.fc_spn = nn.Linear(512 * block.expansion, 3)
        self.fc_ued = nn.Linear(512 * block.expansion, 2)
        self.fc_led = nn.Linear(512 * block.expansion, 2)
        self.fc_umc = nn.Linear(512 * block.expansion, 2)
        self.fc_lmc = nn.Linear(512 * block.expansion, 2)
        self.fc_fsl = nn.Linear(512 * block.expansion, 2)
        self.fc_fsr = nn.Linear(512 * block.expansion, 2)
        self.fc_hrn = nn.Linear(512 * block.expansion, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_pf = self.fc_pf(x)
        x_nar = self.fc_nar(x)
        x_ccs = self.fc_ccs(x)
        x_spn = self.fc_spn(x)
        x_ued = self.fc_ued(x)
        x_led = self.fc_led(x)
        x_umc = self.fc_umc(x)
        x_lmc = self.fc_lmc(x)
        x_fsl = self.fc_fsl(x)
        x_fsr = self.fc_fsr(x)
        x_hrn = self.fc_hrn(x)

        return (
            x_pf,
            x_nar,
            x_ccs,
            x_spn,
            x_ued,
            x_led,
            x_umc,
            x_lmc,
            x_fsl,
            x_fsr,
            x_hrn,
        )

    def _get_classification_layers(self):
        disease_categories = [
            "pf",
            "nar",
            "ccs",
            "spn",
            "ued",
            "led",
            "umc",
            "lmc",
            "fsl",
            "fsr",
            "hrn",
        ]
        is_classification_layer = (
            lambda x: torch.Tensor(
                [disease_category in x[0] for disease_category in disease_categories]
            )
            .bool()
            .any()
        )
        for child in self.named_children():
            if is_classification_layer(child):
                yield child
            else:
                continue

    def finetune(self, reset_weights: bool = True) -> None:
        """Freeze all layers except classification, and reset weights
        for classification_layer"""
        for parameter in self.parameters():
            parameter.requires_grad = False
        for classification_layer in self._get_classification_layers():
            for parameter in classification_layer[1].parameters():
                parameter.requires_grad = True
            if reset_weights:
                classification_layer[1].reset_parameters()

    def load_weights(self, save_path: str, verbose: bool = True) -> None:
        if os.path.isdir(save_path):
            list_of_pt = glob.glob(save_path + "/*.pt")
            latest_pt = max(list_of_pt, key=os.path.getctime)
            checkpoint = torch.load(latest_pt, map_location="cpu")
            self.load_state_dict(checkpoint["model_weights"])
            start_epoch = checkpoint["epoch_no"] + 1
            if verbose:
                print(f"==> Loading model trained for {start_epoch} epochs...")
        else:
            raise NameError(f"save path {save_path} could not be found")
        return
