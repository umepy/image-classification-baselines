import torch
import torch.nn as nn


class VitInputLayer(nn.Module):
    def __init__(self, in_channels: int = 3, emb_dim: int = 384, num_patch_row: int = 2, image_size: int = 32) -> None:
        super(VitInputLayer, self).__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.num_patch_row = num_patch_row
        self.image_size = image_size

        # TODO: flexible patching
        self.num_patch = self.num_patch_row**2
        self.patch_size = self.image_size // self.num_patch_row

        self.patch_emb_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patch + 1, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward process

        Args:
            x (torch.Tensor): (B,C,H,W)

        Returns:
            torch.Tensor: (B, N, D) N is number of tokens. D is size of embedding dim.
        """
        x = self.patch_emb_layer(x)  # (B, D, H/P, W/P) P is size of patch
        x = x.flatten(2)  # (B, D, Np)
        x = x.transpose(1, 2)  # (B, Np, D)
        cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_emb
        return x


if __name__ == "__main__":
    x = torch.randn((2, 3, 32, 32))
    model = VitInputLayer()
    model(x)
