import torch
import torch.nn as nn
from torchvision import transforms


# =========================
# Autoencoder Model
# =========================
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)   # 32 -> 16
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=2,
                stride=2
            ),  # 16 -> 32

            nn.ReLU(),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=2,
                stride=2
            ),  # 32 -> 64

            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 3,
                kernel_size=2,
                stride=2
            ),  # 64 -> 128

            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x