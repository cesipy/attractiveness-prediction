import torch; from torch import nn; from torch.nn import functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim=1024, image_size=256):
        super().__init__()
        self.initial = nn.Linear(latent_dim, 512 * 4 * 4)

        # Decoder with skip connection inputs
        self.up1 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(512 + 512, 512, 4, 2, 1)  # +skip5
        self.up3 = nn.ConvTranspose2d(512 + 512, 256, 4, 2, 1)  # +skip4
        self.up4 = nn.ConvTranspose2d(256 + 256, 128, 4, 2, 1)  # +skip3
        self.up5 = nn.ConvTranspose2d(128 + 128, 64, 4, 2, 1)   # +skip2
        self.up6 = nn.ConvTranspose2d(64 + 64, 3, 4, 2, 1)      # +skip1

    def forward(self, z, skip_connections=None):
        x = F.relu(self.initial(z)).view(-1, 512, 4, 4)

        x = F.relu(self.up1(x))  # 512 x 8
        if skip_connections:
            x = torch.cat([x, skip_connections[4]], dim=1)  # Concat skip5
        x = F.relu(self.up2(x))  # 512 x 16
        if skip_connections:
            x = torch.cat([x, skip_connections[3]], dim=1)  # Concat skip4
        x = F.relu(self.up3(x))  # 256 x 32
        if skip_connections:
            x = torch.cat([x, skip_connections[2]], dim=1)  # Concat skip3
        x = F.relu(self.up4(x))  # 128 x 64
        if skip_connections:
            x = torch.cat([x, skip_connections[1]], dim=1)  # Concat skip2
        x = F.relu(self.up5(x))  # 64 x 128
        if skip_connections:
            x = torch.cat([x, skip_connections[0]], dim=1)  # Concat skip1
        x = torch.tanh(self.up6(x))  # 3 x 256

        return x