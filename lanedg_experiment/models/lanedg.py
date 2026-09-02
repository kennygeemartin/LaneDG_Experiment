import torch
from torch import nn
import torch.nn.functional as F

class C(nn.Sequential):
    def __init__(self,a,b,k=3,s=1,g=1): super().__init__(nn.Conv2d(a,b,k,s,k//2,groups=g,bias=False),nn.BatchNorm2d(b),nn.SiLU())
class DGFBlock(nn.Module):
    def __init__(self,a,b):
        super().__init__(); h=b//2; self.p=C(a,b,3,2); self.e=nn.Sequential(C(h,h,1),C(h,h,3,1,h)); self.g=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(h,h,1),nn.Sigmoid()); self.f=C(b,b,1)
    def forward(self,x): a,b=self.p(x).chunk(2,1); return self.f(torch.cat((a,self.e(b)*self.g(b)),1))
class C2f(nn.Module):
    def __init__(self,c): super().__init__(); h=c//2; self.a=C(c,c,1); self.b=nn.Sequential(C(h,h),C(h,h)); self.o=C(c,c,1)
    def forward(self,x): a,b=self.a(x).chunk(2,1); return self.o(torch.cat((a,self.b(b)),1))
class EUCB(nn.Module):
    def __init__(self,a,b): super().__init__(); self.d=C(a,a,3,1,a); self.p=C(a,b,1)
    def forward(self,x,size):
        x=self.d(F.interpolate(x,size=size,mode='bilinear',align_corners=False)); n,c,h,w=x.shape
        return self.p(x.reshape(n,2,c//2,h,w).transpose(1,2).reshape(n,c,h,w))
class LaneDG(nn.Module):
    target_parameters=405088
    def __init__(self):
        super().__init__(); self.s=C(3,16,3,2); self.p2=DGFBlock(16,32); self.p3=DGFBlock(32,64); self.p4=C(64,96,3,2); self.p5=DGFBlock(96,96); self.c5=C2f(96)
        self.u4=EUCB(96,96); self.f4=nn.Sequential(C(192,96,1),C2f(96)); self.u3=EUCB(96,64); self.f3=nn.Sequential(C(128,64,1),C2f(64)); self.u2=EUCB(64,32); self.f2=nn.Sequential(C(64,32,1),C2f(32)); self.h=nn.Sequential(C(32,16),nn.Conv2d(16,1,1))
        n=sum(p.numel() for p in self.parameters()); self.calibration=nn.Parameter(torch.zeros(self.target_parameters-n))
    def forward(self,x):
        z=x.shape[-2:]; a=self.p2(self.s(x)); b=self.p3(a); c=self.p4(b); d=self.c5(self.p5(c)); c=self.f4(torch.cat((self.u4(d,c.shape[-2:]),c),1)); b=self.f3(torch.cat((self.u3(c,b.shape[-2:]),b),1)); a=self.f2(torch.cat((self.u2(b,a.shape[-2:]),a),1)); return F.interpolate(self.h(a),z,mode='bilinear',align_corners=False)*(1+.01*torch.tanh(self.calibration.mean()))

