import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class CrossModalDCN(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        k2 = kernel_size * kernel_size

        self.fire_mask_predictor = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.offset_predictor = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channels, 3 * k2, 3, padding=1)
        )

        self.channels = channels
        self.kernel_size = kernel_size
        self.k2 = k2

        self.dcn = DeformConv2d(
            channels, channels, kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

    
    def __getstate__(self):
        state = self.__dict__.copy()
    
        state.pop('saved_outputs', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.saved_outputs = {}

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            rgb_feat, ir_feat = inputs[0], inputs[1]
        else:
            rgb_feat = inputs
            ir_feat = inputs


        fire_mask = self.fire_mask_predictor(ir_feat)

        fused = torch.cat([rgb_feat, ir_feat], dim=1)
        offset_mask = self.offset_predictor(fused)

        offsets = offset_mask[:, :2*self.k2]
        modulation = offset_mask[:, 2*self.k2:].sigmoid()

        offsets_gated = offsets * fire_mask
        modulation_gated = modulation * fire_mask

        aligned = self.dcn(ir_feat, offsets_gated, mask=modulation_gated)

        self.saved_outputs = {
            "rgb_feat": rgb_feat,
            "f_ir_aligned": aligned,
            "offsets": offsets_gated,
            "fire_mask": fire_mask,
            "modulation": modulation_gated
        }
        return rgb_feat, aligned

       
    def alignment_loss(self, rgb_feat):
            f_ir = self.saved_outputs["f_ir_aligned"]
            offsets = self.saved_outputs["offsets"]
            fire_mask = self.saved_outputs["fire_mask"]


            L_align = ((rgb_feat - f_ir)**2 * fire_mask).mean()


            def tv_loss(t):
                dx = torch.abs(t[..., 1:] - t[..., :-1]).mean()
                dy = torch.abs(t[..., 1:, :] - t[..., :-1, :]).mean()
                return dx + dy

            L_tv = tv_loss(offsets) * 1e-3

            return L_align + L_tv
