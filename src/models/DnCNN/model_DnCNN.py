import torch
import torch.nn as nn
import torch.nn.functional as F

class DnCNN(nn.Module):
    """
    DnCNN: Denoising CNN via Residual Learning (Zhang et al. 2017).

    The network predicts the noise residual:
        residual = forward(noisy)
        clean = noisy - residual
    """
    def __init__(self, depth: int = 17, num_filters: int = 64, num_channels: int = 1):

        super().__init__()
        layers: list[nn.Module] = []

        # -- Layer 1: Convolutional + ReLU activation -- 
        # (no Batch Normalization since input images are already noemalized)
        layers += [ nn.Conv2d(num_channels, num_filters, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True) ]

        # -- Layer 2 (to depth - 1): Convolutional + ReLU activation + Batch Normalization-- 
        for _ in range(depth - 2): # - 2 since 1st layer and last is handeled sperately
            layers += [ nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(num_filters) , nn.ReLU(inplace=True)]

        # -- Last layer: Convolutional only
        layers += [ nn.Conv2d(num_filters, num_channels, kernel_size=3, padding=1, bias=True) ]


        self.net = nn.Sequential(*layers) # unpacks layers - output from one layer is the input to the next
        self._init_weights()


    def _init_weights(self):
        """
        Kaiming-normal for conv, ones/zeros for BN (standard practice).
        """
        for m in self.modules(): # Iterate through every layer in the network
            if isinstance(m, nn.Conv2d):
            # Kaiming-normal is designed for ReLU networks
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:   # Bias starts at 0 
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                # weight – starts at 1, no scaling at the beginning
                # bias – starts at 0, no offset at the beginning
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


    def forward(self, noisy):
        """
        Args:
            noisy: normalised to [0, 1]
        Returns:
            residual: estimated noise
        """
        return self.net(noisy)
    