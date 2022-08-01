import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F


class IVDSegmentationModel(nn.Module):
    def __init__(self, num_classes=2):
        self.num_classes = num_classes

        super().__init__()
        # Down
        self.conv1 = nn.Conv2d(1, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bnrm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bnrm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bnrm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bnrm4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bnrm5 = nn.BatchNorm2d(512)

        # Up sample
        self.conv6 = nn.Conv2d(512, 256, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bnrm6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(512, 128, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bnrm7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(256, 128, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bnrm8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(256, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bnrm9 = nn.BatchNorm2d(64)
        self.conv10 = nn.Conv2d(64, 32, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bnrm10 = nn.BatchNorm2d(32)
        self.conv11 = nn.Conv2d(
            32, num_classes, (3, 3), stride=(1, 1), padding=(1, 1)
        )

    def forward(self, x):
        # Down
        input = x
        x = F.max_pool2d(
            self.bnrm1(F.relu(self.conv1(x))), kernel_size=2, stride=2
        )  # Conv1
        x2 = F.max_pool2d(
            self.bnrm2(F.relu(self.conv2(x))), kernel_size=2, stride=2
        )  # Conv2
        x3 = F.max_pool2d(
            self.bnrm3(F.relu(self.conv3(x2))), kernel_size=2, stride=2
        )  # Conv3
        x4 = F.max_pool2d(
            self.bnrm4(F.relu(self.conv4(x3))), kernel_size=2, stride=2
        )  # Conv4
        x = F.max_pool2d(
            self.bnrm5(F.relu(self.conv5(x4))), kernel_size=2, stride=2
        )  # Conv5

        # Up
        x = F.interpolate(
            x, size=x4.size()[2:], mode="bilinear", align_corners=True
        )
        x = self.bnrm6(F.relu(self.conv6(x)))
        x = torch.cat((x, x4), 1)
        x = F.interpolate(
            x, size=x3.size()[2:], mode="bilinear", align_corners=True
        )
        x = self.bnrm7(F.relu(self.conv7(x)))
        x = torch.cat((x, x3), 1)
        x = F.interpolate(
            x, size=x2.size()[2:], mode="bilinear", align_corners=True
        )
        x = self.bnrm8(F.relu(self.conv8(x)))
        x = torch.cat((x, x2), 1)
        x = F.interpolate(
            x, size=x.size()[2:], mode="bilinear", align_corners=True
        )
        x = self.bnrm9(F.relu(self.conv9(x)))
        x = F.interpolate(
            x, size=input.size()[2:], mode="bilinear", align_corners=True
        )
        x = self.bnrm10(F.relu(self.conv10(x)))
        x = self.conv11(x)
        return x

    def load_weights(self, save_path, verbose=True):
        if os.path.isdir(save_path):
            list_of_pt = glob.glob(save_path + "/*.pt")
            latest_pt = max(list_of_pt, key=os.path.getctime)
            checkpoint = torch.load(latest_pt, map_location="cpu")
            self.load_state_dict(checkpoint["net"])
            start_epoch = checkpoint["epoch"] + 1
            if verbose:
                print(f"==> Loading model trained for {start_epoch} epochs...")
        else:
            raise NameError(f"save path {save_path} could not be found")

    pass
