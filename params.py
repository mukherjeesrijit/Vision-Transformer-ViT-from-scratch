import torch

# Training parameters
B = 32  # batch size

# Image parameters
C = 3
H = 128
W = 128
x = torch.rand(B, C, H, W)

# Model parameters
D = 64  # hidden size
P = 4  # patch size
N = int(H * W / P**2)  # number of tokens
k = 4  # number of attention heads
Dh = int(D / k)  # attention head size
p = 0.1  # dropout rate
mlp_size = D * 4  # mlp size
L = 4  # number of transformer blocks
n_classes = 3  # number of classes
