import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """
    Layer Normalization for 4D image tensors of shape [N, C, H, W].

    Normalization is applied across the channel dimension at each spatial
    location, which is the variant used in NAFNet. Standard nn.LayerNorm
    expects the normalized dimension to be last, so this small wrapper keeps
    the channel-first image layout used everywhere else in the project.

    Args:
        num_channels: number of feature channels to normalize over.
        eps: small constant added to the variance for numerical stability.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()

        # Learnable scale and shift, one value per channel.
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: feature map of shape [N, C, H, W]
        Returns:
            normalized feature map of the same shape
        """
        # Mean and variance are taken over the channel axis (dim=1).
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)

        # Reshape the per-channel parameters so they broadcast over N, H and W.
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """
    SimpleGate from NAFNet.

    This replaces the usual nonlinear activation (ReLU/GELU). The input is split
    into two halves along the channel dimension, and the two halves are
    multiplied together element-wise. The multiplication is what gives the
    network its nonlinearity, so no separate activation function is needed.
    """

    def forward(self, x):
        """
        Args:
            x: feature map with an even number of channels
        Returns:
            feature map with half as many channels
        """
        # Split the channels into two equal halves, then multiply them.
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free Block (Chen et al. 2022).

    Each block has two stages, and both stages are wrapped in a residual
    connection with its own learnable scale (beta and gamma):

        1. A spatial stage: LayerNorm, a 1x1 convolution to expand channels, a
           3x3 depthwise convolution, a SimpleGate, simplified channel
           attention, and a 1x1 convolution back to the original width.
        2. A feed-forward stage: LayerNorm, a 1x1 convolution to expand
           channels, a SimpleGate, and a 1x1 convolution back.

    There is no ReLU or GELU anywhere. Nonlinearity comes only from the
    SimpleGate multiplications and the channel attention.

    Args:
        num_channels: number of input and output channels for the block.
        dw_expand: channel expansion factor for the spatial (depthwise) stage.
        ffn_expand: channel expansion factor for the feed-forward stage.
        dropout_rate: dropout applied at the end of each stage. 0 disables it.
    """

    def __init__(
        self,
        num_channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        # -- Spatial stage --
        dw_channels = num_channels * dw_expand

        # 1x1 conv expands the channels, then a 3x3 depthwise conv mixes space.
        self.conv1 = nn.Conv2d(num_channels, dw_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channels, dw_channels, kernel_size=3, padding=1,
            groups=dw_channels, bias=True,  # groups=channels makes it depthwise
        )
        # SimpleGate halves the channels, so conv3 starts from dw_channels // 2.
        self.conv3 = nn.Conv2d(dw_channels // 2, num_channels, kernel_size=1, bias=True)

        # Simplified Channel Attention: squeeze each channel to a single number
        # with global average pooling, then rescale the channels with a 1x1 conv.
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1, bias=True),
        )

        # -- Feed-forward stage --
        ffn_channels = num_channels * ffn_expand
        self.conv4 = nn.Conv2d(num_channels, ffn_channels, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channels // 2, num_channels, kernel_size=1, bias=True)

        self.gate = SimpleGate()

        self.norm1 = LayerNorm2d(num_channels)
        self.norm2 = LayerNorm2d(num_channels)

        # Dropout is optional and turned off by default.
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Learnable residual scales. They start at 0, so at the beginning of
        # training each block behaves like an identity mapping and only slowly
        # learns to change its input. This makes the deep network easy to train.
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, inp):
        """
        Args:
            inp: feature map of shape [N, C, H, W]
        Returns:
            feature map of the same shape
        """
        # -- Spatial stage --
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gate(x)
        x = x * self.sca(x)        # reweight channels by their attention scores
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta    # scaled residual connection

        # -- Feed-forward stage --
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.gate(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma  # scaled residual connection


class NAFNet(nn.Module):
    """
    NAFNet: Nonlinear Activation Free Network for image restoration
    (Chen et al. 2022, "Simple Baselines for Image Restoration").

    The model is a U-Net built from NAFBlocks. The image is encoded down through
    several resolution levels, processed at the lowest resolution by the middle
    blocks, and decoded back up, with skip connections carrying detail from each
    encoder level to the matching decoder level.

    A global residual connection adds the input image back at the very end, so
    the network only has to predict the correction needed to clean the image
    rather than rebuild it from scratch. The output is the denoised image
    directly, which means it can be compared straight to the clean target during
    training (the same convention as the plain CNN denoiser, and unlike DnCNN
    which predicts the noise residual).

    Args:
        img_channels: number of image channels. Use 1 for grayscale.
        width: number of feature channels after the first convolution.
        middle_blk_num: number of NAFBlocks at the lowest resolution.
        enc_blk_nums: number of NAFBlocks at each encoder level (top to bottom).
        dec_blk_nums: number of NAFBlocks at each decoder level (bottom to top).
    """

    def __init__(
        self,
        img_channels: int = 1,
        width: int = 32,
        middle_blk_num: int = 2,
        enc_blk_nums: tuple[int, ...] = (1, 1, 1),
        dec_blk_nums: tuple[int, ...] = (1, 1, 1),
    ):
        super().__init__()

        # First and last convolutions map between image channels and features.
        self.intro = nn.Conv2d(img_channels, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channels, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width

        # -- Encoder: going down --
        # Each level runs some NAFBlocks, then a strided conv halves the spatial
        # size while doubling the number of channels.
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan = chan * 2

        # -- Middle: deepest part of the network --
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        # -- Decoder: going up --
        # PixelShuffle doubles the spatial size. The 1x1 conv first grows the
        # channels by 4 so that PixelShuffle(2) leaves us with chan // 2.
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        # The image is padded so its height and width are divisible by this
        # number, since every encoder level halves the resolution.
        self.padder_size = 2 ** len(self.encoders)

        self._init_weights()

    def _init_weights(self):
        """
        Truncated-normal for convolutions, zeros for their biases.

        Kaiming initialization is designed for ReLU networks, but NAFNet has no
        ReLU or GELU, so its gain assumption does not apply here. A small-variance
        truncated normal keeps the early activations well-behaved. The key
        initialization for stable training, the zero-initialized residual scales
        beta and gamma, is handled inside each NAFBlock.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def check_image_size(self, x):
        """
        Pad the image so its height and width are divisible by padder_size.

        Each encoder level halves the resolution, so an input that is not
        divisible by 2 ** num_levels would not line up with its skip connection.
        Patches of size 64 are already divisible, but full test images of
        arbitrary size are not, so they are padded here and cropped back at the
        end of forward.
        """
        _, _, h, w = x.size()
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size

        # Pad on the bottom and right only, so existing pixels keep their indices.
        return F.pad(x, (0, pad_w, 0, pad_h))

    def forward(self, noisy):
        """
        Args:
            noisy: noisy image tensor normalised to [0, 1], shape [N, C, H, W]
        Returns:
            denoised: predicted clean image of the same shape as the input
        """
        _, _, h, w = noisy.shape
        x = self.check_image_size(noisy)
        inp = x  # padded input, kept for the global residual connection

        x = self.intro(x)

        # Encoder. We save the output of each level to reuse as a skip connection.
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle_blks(x)

        # Decoder. Each level upsamples, adds its skip connection, then runs
        # NAFBlocks. The skips are consumed in reverse order (bottom to top).
        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp  # global residual: predict the correction, not the whole image

        # Crop away the padding added by check_image_size.
        return x[:, :, :h, :w]


def denoise(model, noisy):
    """
    Denoise an image using the NAFNet model.

    Args:
        model: trained NAFNet denoiser
        noisy: noisy image tensor normalised to [0, 1]
    Returns:
        denoised image tensor clipped to [0, 1]
    """
    denoised = model(noisy)
    denoised = torch.clamp(denoised, 0, 1)
    return denoised


if __name__ == "__main__":
    # Quick shape check on a square patch and an odd-sized full image.
    model = NAFNet()

    for shape in [(4, 1, 64, 64), (1, 1, 180, 255)]:
        dummy = torch.randn(*shape)
        out = denoise(model, dummy)
        print(f"input {tuple(dummy.shape)} -> denoised {tuple(out.shape)}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")