import torch.nn as nn
import torch
import torch.nn.functional as F
from model.utils import utils

def normal(feat, eps=1e-5):
    feat_mean, feat_std= utils.calc_mean_std(feat, eps)
    normalized=(feat-feat_mean)/feat_std
    return normalized  

class SANet(nn.Module):
    def __init__(self, in_dim):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_dim , in_dim , (1,1))
        self.g = nn.Conv2d(in_dim , in_dim , (1,1))
        self.h = nn.Conv2d(in_dim , in_dim , (1,1))
        self.softmax  = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))
    
    def forward(self,content_feat,style_feat):
        B,C,H,W = content_feat.size()
        F_Fc_norm  = self.f(normal(content_feat)).view(B,-1,H*W).permute(0,2,1)
        B,C,H,W = style_feat.size()
        G_Fs_norm =  self.g(normal(style_feat)).view(B,-1,H*W) 
        energy =  torch.bmm(F_Fc_norm,G_Fs_norm)
        attention = self.softmax(energy)
        H_Fs = self.h(style_feat).view(B,-1,H*W)
        out = torch.bmm(H_Fs,attention.permute(0,2,1) )
        B,C,H,W = content_feat.size()
        out = out.view(B,C,H,W)
        out = self.out_conv(out)
        out += content_feat
        return out

class Self_Attention_Module(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attention_Module, self).__init__()
        self.SAN1=SANet(in_dim)
        self.SAN2=SANet(in_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_dim, in_dim, (3, 3))
    def forward(self, content_feats, style_feats):
        Fcsc_5 = self.SAN1(content_feats[-1], style_feats[-1])
        Fcsc_5_up=self.upsample(Fcsc_5)
        Fcsc_4 = self.SAN2(content_feats[-2], style_feats[-2])
        Fcsc_4_plus_5=Fcsc_4+Fcsc_5_up
        Fcsc_4_plus_5=self.merge_conv_pad(Fcsc_4_plus_5)
        Fcsc_m=self.merge_conv(Fcsc_4_plus_5)
        return Fcsc_m

def decoder():
	return nn.Sequential(
nn.ReflectionPad2d((1, 1, 1, 1)),
nn.Conv2d(512, 256, (3, 3)),
nn.ReLU(),
nn.Upsample(scale_factor=2, mode='nearest'),
nn.ReflectionPad2d((1, 1, 1, 1)),
nn.Conv2d(256, 256, (3, 3)),
nn.ReLU(),
nn.ReflectionPad2d((1, 1, 1, 1)),
nn.Conv2d(256, 256, (3, 3)),
nn.ReLU(),
nn.ReflectionPad2d((1, 1, 1, 1)),
nn.Conv2d(256, 256, (3, 3)),
nn.ReLU(),
nn.ReflectionPad2d((1, 1, 1, 1)),
nn.Conv2d(256, 128, (3, 3)),
nn.ReLU(),
nn.Upsample(scale_factor=2, mode='nearest'),
nn.ReflectionPad2d((1, 1, 1, 1)),
nn.Conv2d(128, 128, (3, 3)),
nn.ReLU(),
nn.ReflectionPad2d((1, 1, 1, 1)),
nn.Conv2d(128, 64, (3, 3)),
nn.ReLU(),
nn.Upsample(scale_factor=2, mode='nearest'),
nn.ReflectionPad2d((1, 1, 1, 1)),
nn.Conv2d(64, 64, (3, 3)),
nn.ReLU(),
nn.ReflectionPad2d((1, 1, 1, 1)),
nn.Conv2d(64, 3, (3, 3)),
)
