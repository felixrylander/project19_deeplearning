import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two convolutional layers used inside the CNN denoiser.

    Each convolution keeps the image size the same because padding=1 is used.
    Batch Normalization and ReLU are added after each convolution to help the
    model learn useful image features.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: input feature map
        Returns:
            output feature map after two convolutional layers
        """
        return self.block(x)


class CNNDenoiser(nn.Module):
    """
    CNN denoiser for grayscale Gaussian image denoising.

    The model takes a noisy image as input and directly predicts the denoised
    image. The output has the same shape as the input, so it can be compared
    directly with the clean target image during training.

    Args:
        num_channels: number of input image channels. Use 1 for grayscale.
        num_filters: number of filters in the first encoder block.
    """

    def __init__(self, num_channels: int = 1, num_filters: int = 32):
        super().__init__()

        # -- Encoder: going down --
        # The image gets smaller, but the number of feature channels grows.
        self.enc1 = DoubleConv(num_channels, num_filters)
        self.enc2 = DoubleConv(num_filters, num_filters * 2)
        self.enc3 = DoubleConv(num_filters * 2, num_filters * 4)

        # MaxPool halves the height and width of the feature map.
        self.pool = nn.MaxPool2d(kernel_size=2)

        # -- Bottleneck: deepest part of the network --
        self.bottleneck = DoubleConv(num_filters * 4, num_filters * 8)

        # -- Decoder: going up --
        # ConvTranspose2d doubles the height and width of the feature map.
        self.up3 = nn.ConvTranspose2d(num_filters * 8, num_filters * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(num_filters * 8, num_filters * 4)

        self.up2 = nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(num_filters * 4, num_filters * 2)

        self.up1 = nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(num_filters * 2, num_filters)

        # Last convolution: turns feature maps back into a denoised image.
        self.out = nn.Conv2d(num_filters, num_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """
        Kaiming-normal for conv layers, ones/zeros for Batch Normalization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # Kaiming-normal is designed for ReLU networks.
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    # Bias starts at 0.
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                # Weight starts at 1, bias starts at 0.
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, noisy):
        """
        Args:
            noisy: noisy image tensor normalised to [0, 1]
        Returns:
            denoised: predicted clean image
        """

        # Encoder. We save e1, e2 and e3 to reuse as skip connections later.
        e1 = self.enc1(noisy)              # original size
        e2 = self.enc2(self.pool(e1))      # 1/2 size
        e3 = self.enc3(self.pool(e2))      # 1/4 size

        # Bottleneck. This is the smallest feature map, but with most channels.
        b = self.bottleneck(self.pool(e3)) # 1/8 size

        # Decoder. Each skip connection gives image details back to the decoder.
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Direct denoising: return the predicted clean image.
        denoised = self.out(d1)
        return denoised


def denoise(model, noisy):
    """
    Denoise an image using the CNN model.

    Args:
        model: trained CNN denoiser
        noisy: noisy image tensor normalised to [0, 1]
    Returns:
        denoised image tensor clipped to [0, 1]
    """
    denoised = model(noisy)
    denoised = torch.clamp(denoised, 0, 1)
    return denoised


if __name__ == "__main__":
    # Quick shape check.
    model = CNNDenoiser()
    dummy = torch.randn(4, 1, 64, 64)
    denoised = denoise(model, dummy)

    print("input   :", tuple(dummy.shape))
    print("denoised:", tuple(denoised.shape))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")