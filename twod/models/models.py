from __future__ import print_function, division
import sys
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
# from dl_mapse.Code.Models import resnet3d


# file that contains all the models used in our study
# Particularly, the Resnets, resnext Unet and swinunetr


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Model(nn.Module):

    def __init__(self, seq_len=1, pretrained=True):
        super(Model, self).__init__()

        print("---- Initializing model ----")

        network = EncoderDecoder(seq_len, pretrained)

        self.network = network

    def forward(self, x):
        x = self.network(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, seq_len=1, pretrained=True):
        super(EncoderDecoder, self).__init__()

        self.three_dim = False if seq_len == 1 else True

        resnet = models.resnet50(pretrained=pretrained)
        resnet.avgpool = Identity()
        resnet.fc = Identity()

        if self.three_dim:
            resnet.conv1 = nn.Conv3d(
                1,
                64,
                kernel_size=(seq_len, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False,
            )
            self.conv1 = list(resnet.children())[0]
            self.resnet = nn.Sequential(*list(resnet.children()))[1:]
        else:
            resnet.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            self.resnet = nn.Sequential(*list(resnet.children()))[:]

        self.bn5 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn1 = nn.BatchNorm2d(1024)

        self.upconv1 = nn.ConvTranspose2d(
            2048, 1024, kernel_size=(3, 3), stride=2, padding=1, output_padding=1
        )
        self.upconv2 = nn.ConvTranspose2d(
            1024, 512, kernel_size=(3, 3), stride=2, padding=1, output_padding=1
        )
        self.upconv3 = nn.ConvTranspose2d(
            512, 256, kernel_size=(3, 3), stride=2, padding=1, output_padding=1
        )
        self.upconv4 = nn.ConvTranspose2d(
            256, 128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1
        )
        self.upconv5 = nn.ConvTranspose2d(
            128, 64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1
        )

        self.outconv = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        if self.three_dim:
            x = self.conv1(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[3], x.shape[4])
        x = self.resnet(x)

        x = F.relu(self.bn1(self.upconv1(x)))
        x = F.relu(self.bn2(self.upconv2(x)))
        x = F.relu(self.bn3(self.upconv3(x)))
        x = F.relu(self.bn4(self.upconv4(x)))
        x = F.relu(self.bn5(self.upconv5(x)))

        x = self.outconv(x)

        return x


class ResNet50Regression(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet50Regression, self).__init__()

        # Load a ResNet-50 backbone (optionally with pretrained ImageNet weights)
        self.resnet = models.resnet50(pretrained=pretrained)

        # Modify the first convolutional layer to accept grayscale (1-channel) input
        # Original expects 3 channels (RGB), so we change input channels from 3 → 1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Replace the final fully connected layer to predict 6 values:
        # (x1, y1, x2, y2, x3, y3) = 3 landmark coordinates
        self.resnet.fc = nn.Linear(
            in_features=self.resnet.fc.in_features, out_features=6
        )

    def forward(self, x):
        """
        Forward pass:
        Input:  tensor of shape (B, 1, 256, 256) = grayscale image batch
        Output: tensor of shape (B, 3, 2) = 3 (x, y) landmarks per image
        """
        x = x.view(
            x.shape[0], x.shape[2], x.shape[3], x.shape[4]
        )  # Reshape to (B, 1, 256, 256)
        x = self.resnet(x)  # Shape: (B, 6)
        x = x.view(-1, 3, 2)  # Reshape to (B, 3, 2)
        x = x * 255  # Scale to original image size
        return x


class ResNeXt50Regression(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNeXt50Regression, self).__init__()

        # Load a ResNeXt-50 backbone (optionally with pretrained ImageNet weights)
        self.resnext = models.resnext50_32x4d(pretrained=pretrained)

        # Modify the first convolutional layer to accept grayscale input
        self.resnext.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Replace the final fully connected layer to predict 6 values
        self.resnext.fc = nn.Linear(
            in_features=self.resnext.fc.in_features, out_features=6
        )

    def forward(self, x):
        """
        Forward pass:
        Input:  tensor of shape (B, 1, 256, 256)
        Output: tensor of shape (B, 3, 2)
        """
        x = x.view(
            x.shape[0], x.shape[2], x.shape[3], x.shape[4]
        )  # Reshape to (B, 1, 256, 256)
        x = self.resnext(x)  # Shape: (B, 6)
        x = x.view(-1, 3, 2)  # Reshape to (B, 3, 2)
        x = x * 255  # Scale to original image size
        return x


class ResNet34Regression(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet34Regression, self).__init__()

        # Load ResNet-34 backbone (optionally pretrained on ImageNet)
        self.resnet = models.resnet34(pretrained=pretrained)

        # Modify the first conv layer to accept grayscale images (1 channel)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Replace the fully connected layer to regress 3 (x, y) coordinates = 6 values
        self.resnet.fc = nn.Linear(
            in_features=self.resnet.fc.in_features, out_features=6
        )

    def forward(self, x):
        """
        Forward pass:
        Args:
            x: Tensor of shape (B, 1, 256, 256), where B is the batch size
        Returns:
            Tensor of shape (B, 3, 2): predicted (x, y) coordinates for 3 landmarks
        """
        x = x.view(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        x = self.resnet(x)  # Output shape: (B, 6)
        x = x.view(-1, 3, 2)  # Reshape to (B, 3, 2)
        x = x * 255  # Optionally scale to image resolution
        return x


class ResNet18Regression(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet18Regression, self).__init__()

        # Load a ResNet-18 backbone (optionally pretrained on ImageNet)
        self.resnet = models.resnet18(pretrained=pretrained)

        # Modify the first conv layer to accept 1-channel input (grayscale)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,  # from 3 to 1
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Replace the final fully connected layer to regress 6 values: (x1, y1, x2, y2, x3, y3)
        self.resnet.fc = nn.Linear(
            in_features=self.resnet.fc.in_features, out_features=6
        )

    def forward(self, x):
        """
        Forward pass for landmark regression.
        Args:
            x (Tensor): Input tensor of shape (B, 1, 256, 256)
        Returns:
            Tensor of shape (B, 3, 2) representing 3 landmark coordinates
        """
        x = x.view(
            x.shape[0], x.shape[2], x.shape[3], x.shape[4]
        )  # Reshape to (B, 1, 256, 256)
        x = self.resnet(x)  # Output shape: (B, 6)
        x = x.view(-1, 3, 2)  # Reshape to (B, 3, 2)
        x = (
            x * 255
        )  # Optional: scale to image size (if coordinates were not normalized)
        return x


class Block(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class CNNLSTM(nn.Module):
    def __init__(
        self,
        seq_type,
        seq_len,
        batch_size,
        cnn_output_size=16,
        lstm_hidden_size=16,
        lstm_num_layers=2,
    ):
        super(CNNLSTM, self).__init__()
        resnet = models.resnet50(pretrained=True)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seq_type = seq_type

        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        resnet.fc = nn.Linear(2048, cnn_output_size)

        self.cnn = resnet
        self.lstm = nn.LSTM(
            cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        self.visibility_fc = nn.Linear(cnn_output_size, 4)
        self.landmark_fc = nn.Linear(lstm_hidden_size, 4)

    def forward(self, x):
        x = F.relu(self.cnn(x))

        visibility = self.visibility_fc(x)

        x = x.view(self.batch_size, self.seq_len, -1)
        x, _ = self.lstm(x)

        x = x[:, -1, :] if self.seq_type == "short" else x

        coordinates = 224 * self.landmark_fc(x)

        return coordinates, visibility


class SwinUNETR(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, start_filts=8, depth=6):
        super(SwinUNETR, self).__init__()
        from monai.networks.nets import SwinUNETR as SwinUNETR_

        self.net = SwinUNETR_(
            img_size=(256, 256),
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=start_filts,
            spatial_dims=2,
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        return self.net(x)


class UNETR(nn.Module):
    def __init__(self):
        super(UNETR, self).__init__()
        from monai.networks.nets import UNETR as UNETR_

        self.net = UNETR_(
            img_size=(64, 256, 256),
            in_channels=1,
            out_channels=3,
            feature_size=8,
            spatial_dims=3,
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        return self.net(x)


class Unet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=3,
        start_filts=8,
        depth=6,
        dropout=0.0,
        num_residuals=0,
    ):
        channels = [start_filts * (2**i) for i in range(depth)]
        strides = [2] * (depth)
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        from monai.networks.nets import UNet as Unet_

        self.net = Unet_(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            spatial_dims=2,
            channels=channels,
            strides=strides,
            norm="batch",
            dropout=dropout,
            num_res_units=num_residuals,
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        return self.net(x)


class ViTAutoencoder(nn.Module):
    def __init__(self):
        super(ViTAutoencoder, self).__init__()
        from monai.networks.nets import ViTAutoEnc

        self.net = ViTAutoEnc(
            in_channels=3,
            out_channels=2,
            spatial_dims=2,
            patch_size=(16, 16),
            img_size=(256, 256),
            pos_embed="conv",
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        x = self.net(x)
        return x[0]


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            bias=True,
            output_padding=1,
        ),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class EncoderDecoderMixFormer(nn.Module):
    def __init__(self):
        super(EncoderDecoderMixFormer, self).__init__()

        resnet = models.resnet50(pretrained=True)

        resnet.avgpool = Identity()
        resnet.fc = Identity()

        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet = nn.Sequential(*list(resnet.children()))[:]
        self.conv = conv(2048, 384)

        # MixFormer head
        self.mixformer_head = nn.Sequential(
            conv(384, 384),
            conv(384, 192),
            conv(192, 96),
            conv(96, 48),
            nn.Conv2d(48, 1, kernel_size=1),
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[3], x.shape[4])
        x = self.resnet(x)
        x = self.conv(x)
        x = self.mixformer_head(x)
        return x


class Unet_3d(nn.Module):
    def __init__(self):
        super(Unet_3d, self).__init__()
        from monai.networks.nets import UNet as Unet_

        self.net = Unet_(
            in_channels=1,
            out_channels=2,
            spatial_dims=3,
            channels=(8, 16, 32),
            strides=(1, 1, 1, 1, 1),
            norm="batch",
        )

    def forward(self, x):
        return self.net(x)


class EncoderDecoder_3d(nn.Module):
    def __init__(self, num_classes=3):
        super(EncoderDecoder_3d, self).__init__()

        self.resnet = resnet3d.resnet34()
        self.num_classes = num_classes

        self.bn5 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn1 = nn.BatchNorm3d(256)

        self.upconv1 = nn.ConvTranspose3d(
            512, 256, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1
        )
        self.upconv2 = nn.ConvTranspose3d(
            256, 128, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1
        )
        self.upconv3 = nn.ConvTranspose3d(
            128, 64, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1
        )
        self.upconv4 = nn.ConvTranspose3d(
            64, 64, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1
        )

        self.outconv = nn.ConvTranspose3d(
            64,
            self.num_classes,
            3,
            stride=(1, 2, 2),
            padding=(1, 1, 1),
            output_padding=(0, 1, 1),
        )

    def forward(self, x):
        x = self.resnet(x)

        x = F.relu(self.bn1(self.upconv1(x)))
        x = F.relu(self.bn2(self.upconv2(x)))
        x = F.relu(self.bn3(self.upconv3(x)))
        x = F.relu(self.bn4(self.upconv4(x)))

        x = self.outconv(x)

        return x
