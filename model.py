import torch
import torch.nn as nn
LATENT_DIM = 100


class Generator(nn.Module):
    """DCGAN-style generator that outputs 1x28x28 MNIST images."""

    def __init__(self, g_output_dim: int = 28 * 28, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 256 * 7 * 7)
        self.net = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.size(1) != self.latent_dim:
            raise ValueError(f"Expected latent input of shape (batch, {self.latent_dim}).")
        x = self.fc(z)
        x = x.view(z.size(0), 256, 7, 7)
        return self.net(x)


class Discriminator(nn.Module):
    """Convolutional discriminator producing Wasserstein critic scores."""

    def __init__(self, d_input_dim: int = 28 * 28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(256 * 7 * 7, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            # Accept flattened input for backwards compatibility.
            x = x.view(x.size(0), 1, 28, 28)
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError("Discriminator expects input of shape (batch, 1, 28, 28).")
        features = self.net(x)
        features = features.view(x.size(0), -1)
        return self.fc(features)
