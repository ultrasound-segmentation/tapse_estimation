import torch.nn as nn
import torch.nn.functional as F
import torch

"""imporved unet based on https://www.sciencedirect.com/science/article/pii/S0169260721000201
THIS IS NOT THE UNET MODEL USED IN THE MANUSCRIPT, THAT ONE IS THE MONAI UNET IN /MODELS/MODELS"""


class FirstResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        identity = self.bn2(self.conv2(out))
        out = out + identity
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        identity = self.bn3(self.conv3(identity))
        out = out + identity
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super(EncoderBlock, self).__init__()
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_layer = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.res_block(x)
        down = self.pool(x)
        down = self.dropout_layer(down)
        return down, x


class Bridge(nn.Module):
    def __init__(self, channels, dilation_rates=[1, 2, 4, 8]):
        super(Bridge, self).__init__()
        self.dilated_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=dilation_rate,
                    dilation=dilation_rate,
                )
                for dilation_rate in dilation_rates
            ]
        )

    def forward(self, x):
        out = x
        out = sum(conv(x) for conv in self.dilated_convs)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        self.dropout_layer = nn.Dropout2d(dropout)
        self.res_block = ResidualBlock(2 * in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        # x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.dropout_layer(x)
        x = self.res_block(x)
        return x


class ImprovedUnet(nn.Module):
    def __init__(self, in_channels, out_channels, max_channels=512, dropout=0.5):
        super(ImprovedUnet, self).__init__()
        self.first_res_block = FirstResidualBlock(in_channels, max_channels // 16)
        self.encoder1 = EncoderBlock(
            max_channels // 16, max_channels // 8, dropout=dropout
        )
        self.encoder2 = EncoderBlock(
            max_channels // 8, max_channels // 4, dropout=dropout
        )
        self.encoder3 = EncoderBlock(
            max_channels // 4, max_channels // 2, dropout=dropout
        )
        self.encoder4 = EncoderBlock(max_channels // 2, max_channels, dropout=dropout)

        self.bridge = Bridge(max_channels)

        self.decoder4 = DecoderBlock(max_channels, max_channels // 2, dropout=dropout)
        self.decoder3 = DecoderBlock(
            max_channels // 2, max_channels // 4, dropout=dropout
        )
        self.decoder2 = DecoderBlock(
            max_channels // 4, max_channels // 8, dropout=dropout
        )
        self.decoder1 = DecoderBlock(
            max_channels // 8, max_channels // 16, dropout=dropout
        )
        self.final_conv = nn.Conv2d(
            max_channels // 16, out_channels, kernel_size=1, padding=0
        )

    def forward(self, x):
        x = self.first_res_block(x)
        enc1, enc1_out = self.encoder1(x)
        enc2, enc2_out = self.encoder2(enc1)
        enc3, enc3_out = self.encoder3(enc2)
        enc4, enc4_out = self.encoder4(enc3)

        bridge_out = self.bridge(enc4)

        dec4 = self.decoder4(bridge_out, enc4_out)
        dec3 = self.decoder3(dec4, enc3_out)
        dec2 = self.decoder2(dec3, enc2_out)
        dec1 = self.decoder1(dec2, enc1_out)

        out = self.final_conv(dec1)

        return out


if __name__ == "__main__":
    model = ImprovedUnet(3, 3, max_channels=512)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(output.shape)
