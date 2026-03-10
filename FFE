import torch
import torch.nn as nn
import torch.nn.functional as F

class FireFeatureExtractor(nn.Module):
  
    def __init__(self, in_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = in_ch // 2
        self.mid_ch = mid_ch

        self.intensity_att = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, 1, 1),
            nn.Sigmoid()
        )

        self.dilated_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, dilation=1, groups=in_ch, bias=False),  # depthwise
            nn.Conv2d(in_ch, mid_ch, 1),  # pointwise
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_ch, mid_ch, 3, padding=2, dilation=2, groups=mid_ch, bias=False),
            nn.Conv2d(mid_ch, in_ch, 1),
            nn.ReLU(inplace=True)
        )


        self.highpass = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self._init_highpass(in_ch)

        self.bn = nn.BatchNorm2d(in_ch)
        self.act = nn.ReLU(inplace=True)

    def _init_highpass(self, in_ch):
    
        with torch.no_grad():
            self.highpass.weight.fill_(0)
            c = self.highpass.weight.size(2) // 2
            kh, kw = self.highpass.weight.shape[2:4]
            for i in range(in_ch):
                self.highpass.weight[i, 0, c, c] = 1.0
            self.highpass.weight -= 1.0 / (kh * kw)

    def forward(self, x):
     
        x = x * self.intensity_att(x)


        x = x + self.dilated_conv(x)  # 改成非 inplace


        x = x + 0.4 * self.highpass(x)  # 改成非 inplace

 
        return self.act(self.bn(x))
