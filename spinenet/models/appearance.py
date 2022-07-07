import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob


class AppearanceModel(nn.Module):
    def __init__(self):
        """
        conv dropout chooses if there is dropout between the conv layers
        """
        super(AppearanceModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, (3, 3, 1), padding=(3, 3, 0), stride=(2, 2, 1))
        self.bnrm1 = nn.BatchNorm3d(8)
        self.conv2 = nn.Conv3d(8, 16, (3, 3, 1), padding=(3, 3, 0))
        self.bnrm2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 32, (3, 3, 3))
        self.bnrm3 = nn.BatchNorm3d(32)
        self.conv4 = nn.Conv3d(32, 64, (3, 3, 3))
        self.bnrm4 = nn.BatchNorm3d(64)
        self.conv5 = nn.Conv3d(64, 128, (3, 3, 3))
        self.bnrm5 = nn.BatchNorm3d(128)
        self.fc1 = nn.Linear(6400, 256)
        self.fc2 = nn.Linear(256, 24)
        self.fc_dropout_layer = nn.Dropout()

    def forward(self, x):
        x = self.get_appearance_features(x)
        x = self.fc_dropout_layer(x)
        x = self.fc2(x)
        return x

    def get_appearance_features(self, x):
        x = self.bnrm1(F.relu(self.conv1(x)))
        x = F.max_pool3d(self.bnrm2(F.relu(self.conv2(x))), kernel_size=(2, 2, 1))
        x = F.max_pool3d(self.bnrm3(F.relu(self.conv3(x))), kernel_size=(2, 2, 1))
        x = F.max_pool3d(self.bnrm4(F.relu(self.conv4(x))), kernel_size=2)
        x = F.max_pool3d(self.bnrm5(F.relu(self.conv5(x))), kernel_size=2)

        # flatten layers
        x = x.flatten(start_dim=1)
        x = self.fc_dropout_layer(x)
        x = F.relu(self.fc1(x))
        return x

    def load_weights(self, save_path, verbose=True):
        if os.path.isdir(save_path):
            list_of_pt = [
                x for x in glob.glob(save_path + "/*.pt") if "encrypted" not in x
            ]
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
