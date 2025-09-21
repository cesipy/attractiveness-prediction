from torch import nn; import torch; from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=1024, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim


        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)     # 64 x 128 x 128
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)   # 128 x 64 x 64
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)  # 256 x 32 x 32
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)  # 512 x 16 x 16
        self.conv5 = nn.Conv2d(512, 512, 4, 2, 1)  # 512 x 8 x 8
        self.conv6 = nn.Conv2d(512, 512, 4, 2, 1)  # 512 x 4 x 4

        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(512 * 16, latent_dim)
        )

    def forward(self, x):
        # Store skip connections
        skip1 = F.leaky_relu(self.conv1(x), 0.2)           # 64 x 128
        skip2 = F.leaky_relu(self.conv2(skip1), 0.2)       # 128 x 64
        skip3 = F.leaky_relu(self.conv3(skip2), 0.2)       # 256 x 32
        skip4 = F.leaky_relu(self.conv4(skip3), 0.2)       # 512 x 16
        skip5 = F.leaky_relu(self.conv5(skip4), 0.2)       # 512 x 8
        skip6 = F.leaky_relu(self.conv6(skip5), 0.2)       # 512 x 4

        latent = self.final(skip6)

        # Return latent + skip connections for generator
        return latent, [skip1, skip2, skip3, skip4, skip5]