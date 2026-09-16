from __future__ import print_function, division
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from monai.networks.nets import UNet

from temporal_pipeline.models import resnet3d


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


class UNet3D(nn.Module):
    def __init__(
        self,
        device,
        initial_channels=16,
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ):
        super(UNet3D, self).__init__()

        self.device = device

        print("---- Initializing 3D_UNet ----")

        network = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(
                initial_channels,
                2 * initial_channels,
                4 * initial_channels,
                8 * initial_channels,
                16 * initial_channels,
            ),
            strides=strides,
            num_res_units=num_res_units,
        ).to(self.device)

        self.network = network

    def forward(self, x):
        x = self.network(x)
        return x


class EncoderDecoder_3d(nn.Module):
    def __init__(self):
        super(EncoderDecoder_3d, self).__init__()

        self.resnet = resnet3d.resnet34()

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
            64, 3, 3, stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)
        )

    def forward(self, x):
        x = self.resnet(x)

        x = F.relu(self.bn1(self.upconv1(x)))
        x = F.relu(self.bn2(self.upconv2(x)))
        x = F.relu(self.bn3(self.upconv3(x)))
        x = F.relu(self.bn4(self.upconv4(x)))

        x = self.outconv(x)

        return x


if __name__ == "__main__":

    # Test 3D mode (sequence of images)
    print("\nTesting 3D mode (seq_len=32)...")
    model_3d = UNet3D(device="cpu")
    x_3d = torch.randn(1, 1, 32, 224, 224)
    output_3d = model_3d(x_3d)
    print(f"Input shape: {x_3d.shape} -> Output shape: {output_3d.shape}")

    print("\nTest completed successfully!")
