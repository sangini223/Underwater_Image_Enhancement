import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)  # Using LeakyReLU for better performance
        )

    def forward(self, x):
        return self.conv(x)

class PhysicalNN(nn.Module):
    def __init__(self):
        super(PhysicalNN, self).__init__()

        # Encoder layers
        self.encoder1 = ConvBlock(3, 64)   # Input: 3 channels (RGB)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        
        # Bottleneck layer with residual learning
        self.bottleneck = ConvBlock(256, 256)

        # Decoder layers
        self.decoder1 = ConvBlock(256, 128)
        self.decoder2 = ConvBlock(128, 64)
        
        # Output layer
        self.output_layer = nn.Conv2d(64, 3, kernel_size=1)  # Output: 3 channels (RGB)

        # Skip connection layers
        self.skip_conv1 = nn.Conv2d(128, 128, kernel_size=1)  # For enc2
        self.skip_conv2 = nn.Conv2d(256, 256, kernel_size=1)  # For enc3

    def forward(self, x):
        # Encoder: Feature extraction
        enc1 = self.encoder1(x)  # (batch, 64, H, W)
        enc2 = self.encoder2(enc1)  # (batch, 128, H, W)
        enc3 = self.encoder3(enc2)  # (batch, 256, H, W)

        # Bottleneck with residual learning
        bottleneck = self.bottleneck(enc3) + enc3  # (batch, 256, H, W)

        # Decoder: Feature reconstruction with skip connections
        dec1 = self.decoder1(bottleneck + self.skip_conv2(enc3))  # (batch, 128, H, W)
        dec2 = self.decoder2(dec1 + self.skip_conv1(enc2))  # (batch, 64, H, W)

        output = self.output_layer(dec2 + enc1)  # Final output (batch, 3, H, W)

        return output
