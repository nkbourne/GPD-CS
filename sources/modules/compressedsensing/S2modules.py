# pytorch_diffusion + derived encoder decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from sources.modules.compressedsensing.S1modules import ResBlk, Down, nonlinearity

class ENCODES2(nn.Module):
    def __init__(self, prior_dim, ch_in = 1, n_feats = 32, norm = 'gn'):
        super(ENCODES2, self).__init__()
        self.n_feats=n_feats

        self.en_in = nn.Conv2d(ch_in, n_feats, kernel_size=3, padding=1)

        EN = [
            ResBlk(in_channels=n_feats, out_channels=n_feats, norm = norm),
            ResBlk(in_channels=n_feats, out_channels=n_feats*2, norm = norm),
            Down(),#64*32*32
            ResBlk(in_channels=n_feats*2, out_channels=n_feats*2, norm = norm),
            ResBlk(in_channels=n_feats*2, out_channels=n_feats*4, norm = norm),
            Down(),#128*16*16

            ResBlk(in_channels=n_feats*4, out_channels=n_feats*4, norm = norm)
        ]
        self.encoder = nn.Sequential(
            *EN
        )

        self.en_out = torch.nn.Conv2d(n_feats*4,prior_dim,kernel_size=3,padding=1)

    def encode(self,input):
        out = self.en_in(input)
        out = self.encoder(out)
        out = nonlinearity(out)
        out = self.en_out(out)
        return out

    def forward(self, gt):
        latent = self.encode(gt)
        return latent

class denoise_cnn(nn.Module):
    def __init__(self, prior_dim, n_feats = 64, n_denoise_res = 5,timesteps=5, norm = 'gn'):
        super(denoise_cnn, self).__init__()
        self.max_period=timesteps*10
        mlp = [
            nn.Conv2d(prior_dim*2+1, n_feats, kernel_size=3, stride=1, padding=1)
        ]
        for _ in range(n_denoise_res):
            mlp.append(ResBlk(in_channels = n_feats, out_channels= n_feats, norm=norm))
        self.resmlp = nn.Sequential(*mlp)
        self.cout = nn.Conv2d(n_feats, prior_dim, kernel_size=3, stride=1, padding=1)

    def forward(self,x, t, c):
        t=t.float()
        t =t/self.max_period
        t=t.view(-1,1,1,1).repeat(1,1,16,16)
        out = torch.cat([x,c,t],dim=1)
        out = self.resmlp(out)
        out = self.cout(out)
        return out