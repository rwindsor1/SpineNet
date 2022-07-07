import torch
import torch.nn as nn
import torch.nn.functional as F
import os, glob


class ContextModel(nn.Module):
    def __init__(self):
        super(ContextModel, self).__init__()
        # down
        self.conv1 = nn.Conv2d(1, 16, (5, 3), dilation=(2, 1), padding=(4, 1))
        self.bnrm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (5, 3), dilation=(2, 1), padding=(4, 1))
        self.bnrm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 3), dilation=(2, 1), padding=(4, 1))
        self.bnrm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (5, 3), dilation=(2, 1), padding=(4, 1))
        self.bnrm4 = nn.BatchNorm2d(128)

        # mid conv
        self.conv_mid = nn.Conv2d(128, 128, (3, 3), dilation=(1, 1), padding=(1, 1))
        self.bnrm_mid = nn.BatchNorm2d(128)
        # up
        self.conv5 = nn.Conv2d(128, 64, (5, 3), dilation=(2, 1), padding=(4, 1))
        self.bnrm5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(128, 32, (5, 3), dilation=(2, 1), padding=(4, 1))
        self.bnrm6 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(64, 16, (5, 3), dilation=(2, 1), padding=(4, 1))
        self.bnrm7 = nn.BatchNorm2d(16)
        self.conv8 = nn.Conv2d(16, 1, (5, 3), dilation=(2, 1), padding=(4, 1))

    def forward(self, x):
        # Down
        x = F.max_pool2d(self.bnrm1(F.relu(self.conv1(x))), kernel_size=(2, 1))  # Conv1
        x2 = F.max_pool2d(
            self.bnrm2(F.relu(self.conv2(x))), kernel_size=(2, 1)
        )  # Conv2
        x3 = F.max_pool2d(
            self.bnrm3(F.relu(self.conv3(x2))), kernel_size=(2, 1)
        )  # Conv3
        x = F.max_pool2d(
            self.bnrm4(F.relu(self.conv4(x3))), kernel_size=(2, 1)
        )  # Conv4
        # Mid
        x = self.bnrm_mid(F.relu(self.conv_mid(x)))
        # Up
        x = F.interpolate(x, scale_factor=(2, 1), mode="bilinear", align_corners=False)
        x = self.bnrm5(F.relu(self.conv5(x)))
        x = torch.cat((x, x3), 1)
        x = F.interpolate(x, scale_factor=(2, 1), mode="bilinear", align_corners=False)
        x = self.bnrm6(F.relu(self.conv6(x)))
        x = torch.cat((x, x2), 1)
        x = F.interpolate(x, scale_factor=(2, 1), mode="bilinear", align_corners=False)
        x = self.bnrm7(F.relu(self.conv7(x)))
        x = F.interpolate(x, scale_factor=(2, 1), mode="bilinear", align_corners=False)
        x = self.conv8(x)
        return x

    def load_weights(self, save_path, verbose=True):
        if os.path.isdir(save_path):
            list_of_pt = glob.glob(save_path + "/*.pt")
            latest_pt = max(list_of_pt, key=os.path.getctime)
            checkpoint = torch.load(latest_pt, map_location="cpu")
            self.load_state_dict(checkpoint["net"])
            best_loss = checkpoint["loss"]
            start_epoch = checkpoint["epoch"] + 1
            if verbose:
                print(f"==> Loading model trained for {start_epoch} epochs...")
        else:
            raise NameError(f"save path {save_path} could not be found")
        return
