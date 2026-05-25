import torch
import torch.nn as nn
from torchvision import transforms

# =========================
# DnCNN Model
# =========================
class DnCNN(nn.Module):
    def __init__(self, depth=17, channels=64):
        super(DnCNN, self).__init__()

        layers = []

        # First Layer
        layers.append(
            nn.Conv2d(
                3,
                channels,
                kernel_size=3,
                padding=1
            )
        )
        layers.append(nn.ReLU(inplace=True))

        # Middle Layers
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1
                )
            )
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))

        # Last Layer
        layers.append(
            nn.Conv2d(
                channels,
                3,
                kernel_size=3,
                padding=1
            )
        )

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise