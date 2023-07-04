import torch
import torch.nn as nn
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F


class GroundingDownsampler(nn.Module):
    def __init__(self, resize_input=256, in_dim=152,out_dim=8):
        super().__init__()
        self.resize_input = resize_input
        self.out_dim = out_dim

        self.depth_layers = nn.Sequential(
            nn.Conv2d(1, 4, 4, 2, 1),  # in1 out4 k4x4 s2 p1
            nn.SiLU(),
            nn.Conv2d(4, self.out_dim, 4, 2, 1)
        )

        self.sem_layers = nn.Sequential(
            nn.Conv2d(in_dim,16,4,2,1),
            nn.SiLU(),
            nn.Conv2d(16,self.out_dim,4,2,1)
        )
    def forward(self, grounding_extra_input):
        # this is actually gary scale, but converted to rgb in dataset, information redudant
        sem_out = torch.nn.functional.interpolate(grounding_extra_input[0], (self.resize_input,self.resize_input), mode='nearest')
        sem_out = self.sem_layers(sem_out)

        grounding_extra_input = grounding_extra_input[1][:, 0].unsqueeze(1)  # [0]depth 第0列数据转成列向量

        dep_out = torch.nn.functional.interpolate(grounding_extra_input, (self.resize_input, self.resize_input), mode='bicubic')
        dep_out = self.depth_layers(dep_out)



        assert dep_out.shape[1] == self.out_dim
        assert sem_out.shape[1] == self.out_dim
        return dep_out,sem_out


