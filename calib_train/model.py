#!/usr/bin/env python3
"""Hafif U-Net: çizgi-maskesi (1xHxW) -> 15 keypoint ısı-haritası (15xHxW)."""
import torch
import torch.nn as nn


def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class CalibUNet(nn.Module):
    def __init__(self, n_kp=15, base=32):
        super().__init__()
        self.d1 = cbr(1, base); self.d2 = cbr(base, base * 2)
        self.d3 = cbr(base * 2, base * 4); self.d4 = cbr(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2); self.u3 = cbr(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2); self.u2 = cbr(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2); self.u1 = cbr(base * 2, base)
        self.head = nn.Conv2d(base, n_kp, 1)

    def forward(self, x):
        c1 = self.d1(x); c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2)); c4 = self.d4(self.pool(c3))
        x = self.u3(torch.cat([self.up3(c4), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.head(x)
