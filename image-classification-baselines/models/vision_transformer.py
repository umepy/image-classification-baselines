import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = x.flatten(2)  # (B, D, Np) Np is number of patches
        x = x.transpose(1, 2)  # (B, Np, D)
        cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_emb  # (B, Np+1, D)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim: int = 384, head: int = 8, dropout: float = 0.0) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        assert emb_dim % head == 0
        self.emb_dim = emb_dim
        self.head = head
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim**0.5

        self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.w_o = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.size()) == 3  # (B, Np+1, D)
        batch_size, num_patch, emb_dim = x.size()
        assert emb_dim == self.emb_dim

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # multi-head split
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        # self attention was done to the matrix of num_patch and head_dim
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weight = q @ k.transpose(2, 3) / self.sqrt_dh  # (B,H,N,N)
        attn = F.softmax(attn_weight, dim=-1)
        attn = self.attn_dropout(attn_weight)
        x = attn @ v
        x = x.transpose(1, 2)

        # concat multi-head
        x = x.view(batch_size, num_patch, -1)

        x = self.w_o(x)

        return x

class VitEncoderBlock(nn.Module):
    def __init__

if __name__ == "__main__":
    x = torch.randn((2, 3, 32, 32))
    model = VitInputLayer()
    model(x)
