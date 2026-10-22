"""LaneDG-inspired semantic reimplementation, not the cited instance model.

The module layout implements DGF downsampling, true C2f concatenation,
SPPF, a top-down and bottom-up FPN-PAN, and EUCB decoding. No dummy
parameters or artificial parameter-count targets are used. Author code and
exact hyperparameters were not published in the cited paper; equivalence
to its instance segmentation network is NOT claimed.
"""
import torch
from torch import nn
import torch.nn.functional as F

MODEL_NAME = "LaneDG-Semantic-Reimplementation-v1"
ARCHITECTURE = {
    "name": MODEL_NAME,
    "status": "documented semantic adaptation; not verified equivalent to cited YOLOv8n-LaneDG",
    "backbone": "Conv stem; DGF at P2/P3/P5; C2f; SPPF",
    "neck": "C2f FPN-PAN with both top-down and bottom-up paths",
    "head": "EUCB multi-scale semantic decoder; one full-resolution binary logit map",
    "channel_widths": [16, 32, 64, 96, 128],
    "differences_from_cited_model": [
        "semantic union mask replaces instance/prototype/detection outputs",
        "no CIoU box or object-classification loss, detection confidences, or NMS",
        "channel widths, C2f repeats, gates, shuffle groups, and decoder are explicit reconstruction choices",
        "no claim to reproduce the cited paper's parameter count or published scores",
    ],
    "reference": "https://doi.org/10.3390/computers15090561",
    "initialization": "random; no pretrained weights",
}


class Conv(nn.Sequential):
    def __init__(self, a, b, kernel=3, stride=1, groups=1):
        super().__init__(nn.Conv2d(a, b, kernel, stride, kernel//2, groups=groups, bias=False),
                         nn.BatchNorm2d(b), nn.SiLU())


class Bottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(Conv(channels, channels), Conv(channels, channels))

    def forward(self, x):
        return x + self.layers(x)


class C2f(nn.Module):
    def __init__(self, a, b, repeats=1):
        super().__init__()
        self.input = Conv(a, b, 1)
        self.blocks = nn.ModuleList(Bottleneck(b//2) for _ in range(repeats))
        self.output = Conv((2+repeats)*(b//2), b, 1)

    def forward(self, x):
        parts = list(self.input(x).chunk(2, 1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.output(torch.cat(parts, 1))


class DGFBlock(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.down = Conv(a, b, 3, 2)
        half = b//2
        self.enhance = nn.Sequential(Conv(half, half, 1), Conv(half, half, 3, groups=half))
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(half, half, 1), nn.Sigmoid())
        self.fuse = Conv(b, b, 1)

    def forward(self, x):
        preserved, enhanced = self.down(x).chunk(2, 1)
        return self.fuse(torch.cat([preserved, self.enhance(enhanced)*self.gate(enhanced)], 1))


class SPPF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.input = Conv(channels, channels//2, 1)
        self.pool = nn.MaxPool2d(5, 1, 2)
        self.output = Conv(2*channels, channels, 1)

    def forward(self, x):
        a = self.input(x)
        b = self.pool(a)
        c = self.pool(b)
        return self.output(torch.cat([a, b, c, self.pool(c)], 1))


class EUCB(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.depthwise = Conv(a, a, 3, groups=a)
        self.pointwise = Conv(a, b, 1)

    def forward(self, x, size):
        x = self.depthwise(F.interpolate(x, size=size, mode="bilinear", align_corners=False))
        n, c, h, w = x.shape
        x = x.reshape(n, 2, c//2, h, w).transpose(1, 2).contiguous().reshape(n, c, h, w)
        return self.pointwise(x)


class LaneDGSemantic(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Conv(3, 16, 3, 2)
        self.p2 = nn.Sequential(DGFBlock(16, 32), C2f(32, 32))
        self.p3 = nn.Sequential(DGFBlock(32, 64), C2f(64, 64, 2))
        self.p4 = nn.Sequential(Conv(64, 96, 3, 2), C2f(96, 96, 2))
        self.p5 = nn.Sequential(DGFBlock(96, 128), C2f(128, 128), SPPF(128))
        self.up4, self.top4 = EUCB(128, 96), C2f(192, 96)
        self.up3, self.top3 = EUCB(96, 64), C2f(128, 64)
        self.down4, self.bottom4 = Conv(64, 64, 3, 2), C2f(160, 96)
        self.down5, self.bottom5 = Conv(96, 96, 3, 2), C2f(224, 128)
        self.decode5, self.merge4 = EUCB(128, 96), C2f(192, 96)
        self.decode4, self.merge3 = EUCB(96, 64), C2f(128, 64)
        self.decode3, self.merge2 = EUCB(64, 32), C2f(64, 32)
        self.head = nn.Sequential(Conv(32, 16), nn.Conv2d(16, 1, 1))

    def forward(self, image):
        a = self.p2(self.stem(image))
        b = self.p3(a)
        c = self.p4(b)
        d = self.p5(c)
        t4 = self.top4(torch.cat([self.up4(d, c.shape[-2:]), c], 1))
        t3 = self.top3(torch.cat([self.up3(t4, b.shape[-2:]), b], 1))
        n4 = self.bottom4(torch.cat([self.down4(t3), t4], 1))
        n5 = self.bottom5(torch.cat([self.down5(n4), d], 1))
        h4 = self.merge4(torch.cat([self.decode5(n5, n4.shape[-2:]), n4], 1))
        h3 = self.merge3(torch.cat([self.decode4(h4, t3.shape[-2:]), t3], 1))
        h2 = self.merge2(torch.cat([self.decode3(h3, a.shape[-2:]), a], 1))
        return F.interpolate(self.head(h2), image.shape[-2:], mode="bilinear", align_corners=False)


def segmentation_loss(logits, truth, valid):
    probability = logits.sigmoid()
    bce = (F.binary_cross_entropy_with_logits(logits, truth, reduction="none")*valid).sum()/valid.sum().clamp_min(1)
    dims = (1, 2, 3)
    dice = (1 - (2*(probability*truth*valid).sum(dims)+1) /
            ((probability*valid).sum(dims)+(truth*valid).sum(dims)+1)).mean()
    vertical = valid[:, :, 1:, :]*valid[:, :, :-1, :]*truth[:, :, 1:, :]*truth[:, :, :-1, :]
    horizontal = valid[:, :, :, 1:]*valid[:, :, :, :-1]*truth[:, :, :, 1:]*truth[:, :, :, :-1]
    continuity = ((probability[:, :, 1:, :]-probability[:, :, :-1, :]).abs()*vertical).sum()
    continuity += ((probability[:, :, :, 1:]-probability[:, :, :, :-1]).abs()*horizontal).sum()
    continuity /= (vertical.sum()+horizontal.sum()).clamp_min(1)
    return bce + dice + 0.5*continuity
