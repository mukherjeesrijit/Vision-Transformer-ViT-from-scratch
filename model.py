import torch
import torch.nn as nn
import math
from params import B, C, H, W, D, P, N, k, Dh, p, mlp_size, L, n_classes

# Image Embeddings
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()

        self.unfold = nn.Unfold(kernel_size=P, stride=P)
        self.project = nn.Linear(P**2 * C, D)
        self.cls_token = nn.Parameter(torch.randn((1, 1, D)))
        self.pos_embedding = nn.Parameter(torch.randn(1, N+1, D))
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.unfold(x).transpose(1, 2)
        x = self.project(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        pos_embedding = self.pos_embedding.expand(B, -1, -1)
        z0 = x + pos_embedding
        z0 = self.dropout(z0)
        return z0

# Single Head Attention
class Single_Head_Attention(nn.Module):
    def __init__(self):
        super(Single_Head_Attention, self).__init__()

        self.U_qkv = nn.Linear(D, 3 * Dh)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z):
        qkv = self.U_qkv(z)
        q = qkv[:, :, :Dh]
        k = qkv[:, :, Dh:2*Dh]
        v = qkv[:, :, 2*Dh:]
        qkTbysqrtDh = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        A = self.softmax(qkTbysqrtDh)
        SAz = torch.matmul(A, v)
        return SAz

# Multi Head Self Attention
class Multi_Head_Self_Attention(nn.Module):
    def __init__(self):
        super(Multi_Head_Self_Attention, self).__init__()

        self.heads = nn.ModuleList([Single_Head_Attention() for _ in range(k)])
        self.U_msa = nn.Linear(D, D)
        self.dropout = nn.Dropout(p)

    def forward(self, z):
        ConSAz = torch.cat([head(z) for head in self.heads], dim=-1)
        msaz = self.U_msa(ConSAz)
        msaz = self.dropout(msaz)
        return msaz

# MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.U_mlp = nn.Linear(D, mlp_size)
        self.gelu = nn.GELU()
        self.U_mlp2 = nn.Linear(mlp_size, D)
        self.dropout = nn.Dropout(p)

    def forward(self, z):
        z = self.U_mlp(z)
        z = self.gelu(z)
        z = self.dropout(z)
        z = self.U_mlp2(z)
        z = self.gelu(z)
        z = self.dropout(z)
        return z

# Transformer Block
class Transformer_Block(nn.Module):
    def __init__(self):
        super(Transformer_Block, self).__init__()

        self.layernorm_1 = nn.LayerNorm(D)
        self.msa = Multi_Head_Self_Attention()
        self.layernorm_2 = nn.LayerNorm(D)
        self.mlp = MLP()

    def forward(self, z):
        z1 = self.layernorm_1(z)
        z1 = self.msa(z1)
        z2 = z + z1
        z3 = self.layernorm_2(z2)
        z3 = self.mlp(z3)
        z4 = z2 + z3
        return z4

# Vision Transformer (ViT)
class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()

        self.embedding = Embedding()
        self.transformer_encoder = nn.ModuleList([Transformer_Block() for _ in range(L)])
        self.layernorm = nn.LayerNorm(D)
        self.U_mlp = nn.Linear(D, n_classes)

    def forward(self, x):
        z = self.embedding(x)
        for block in self.transformer_encoder:
            z = block(z)
        z = self.layernorm(z)
        z = z[:, 0, :]
        z = self.U_mlp(z)
        return z
