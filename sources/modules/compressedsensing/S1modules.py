import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def nonlinearity(x, type = 'swish'):
    if type == 'swish':
        return x*torch.sigmoid(x)
    elif type == 'lrelu':
        return F.leaky_relu(x, negative_slope=0.01)
    else:
        return x

def Normalize(in_channels, num_groups=32, type = 'gn'):
    if type == 'gn':
        return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif type == 'bn':
        return nn.BatchNorm2d(in_channels)
    else:
        return None


def vectorize(inputs, block_size, channles):
    xdim = int(block_size*block_size*channles)
    inputs = torch.cat(torch.split(inputs, split_size_or_sections=block_size, dim=3), dim=0)
    inputs = torch.cat(torch.split(inputs, split_size_or_sections=block_size, dim=2), dim=0)
    inputs = torch.reshape(inputs, [-1, xdim])
    inputs = torch.transpose(inputs, 0, 1)
    return inputs

def devectorize(inputs, batch_size, block_size, image_size, channles):
    rows = image_size//block_size
    recon = torch.reshape(torch.transpose(inputs, 0, 1), [-1, channles, block_size, block_size])
    recon = torch.cat(torch.split(recon, split_size_or_sections=rows * batch_size, dim=0), dim=2)
    recon = torch.cat(torch.split(recon, split_size_or_sections=batch_size, dim=0), dim=3)
    return recon

class ResBlk(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, norm = 'gn'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, type = norm)
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)

        self.norm2 = Normalize(out_channels, type = norm)
        self.conv2 = torch.nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                out_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x+h

class Down(nn.Module):
    def __init__(self):
        super(Down,self).__init__()

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class Up(nn.Module):
    def __init__(self):
        super(Up,self).__init__()
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return x
    
class CSCNNBlock(nn.Module):
    def __init__(self, feature_dim, prior_dim, blocks = 4, norm = 'gn'):
        super(CSCNNBlock, self).__init__()

        self.blocks = blocks

        self.conv_in = nn.Conv2d(feature_dim+prior_dim, feature_dim, kernel_size=3, stride=1, padding=1)
        self.CAblock = nn.ModuleList()
        for i in range(blocks):
            self.CAblock.append(ResBlk(in_channels=feature_dim, norm = norm))
        # self.norm2 = Normalize(feature_dim, type = norm)
        self.conv_out = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1)
    def forward(self,x, hidden_x, prior_f):
        hx = torch.cat([x, hidden_x], 1)
        out = torch.cat([hx, prior_f], 1)
        out = self.conv_in(out)
        for i in range(self.blocks):
            out = self.CAblock[i](out)
        # out = self.norm2(out)
        # out = nonlinearity(out)
        out = self.conv_out(out)
        out = out + hx
        return out

class GPLM(nn.Module):
    def __init__(self, prior_dim, ch_in = 1, n_feats = 32, norm = 'gn'):
        super(GPLM, self).__init__()
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
        self.en_out = torch.nn.Conv2d(n_feats*4,prior_dim*2,kernel_size=3,padding=1)

        

        self.de_in = nn.Conv2d(prior_dim, n_feats*4, kernel_size=3, padding=1)

        DE = [
            ResBlk(in_channels=n_feats*4, out_channels=n_feats*4, norm = norm),#256*4*4

            Up(),
            ResBlk(in_channels=n_feats*4, out_channels=n_feats*2, norm = norm),#32*32*32
            ResBlk(in_channels=n_feats*2, out_channels=n_feats*2, norm = norm),
            Up(),
            ResBlk(in_channels=n_feats*2, out_channels=n_feats, norm = norm),#16*64*64
            ResBlk(in_channels=n_feats, out_channels=n_feats, norm = norm)
        ]

        self.decoder = nn.Sequential(
            *DE
        )
        self.de_out = torch.nn.Conv2d(n_feats,ch_in,kernel_size=3,stride=1,padding=1)

    def encode(self,input):
        out = self.en_in(input)
        out = self.encoder(out)
        out = nonlinearity(out)
        out = self.en_out(out)
        return out
    
    def decode(self,input):
        out = self.de_in(input)
        out = self.decoder(out)
        out = nonlinearity(out)
        out = self.de_out(out)
        return out

    def forward(self, gt, ir = None):
        if ir is not None:
            fea = torch.cat([gt, ir], 1)
        else:
            fea = gt
        latent = self.encode(fea)
        out = self.decode(latent)
        return out, latent
    
class CSINIT(nn.Module):
    def __init__(self,
                 sr, 
                 block_size = 32,
                 image_size = 64,
                 in_channels = 1):
        super(CSINIT,self).__init__()
        self.sr = sr
        self.block_size = block_size
        self.image_size = image_size
        self.channels = in_channels
        xdim = int(block_size*block_size*in_channels)
        ydim = int(sr*xdim)
        Phi_init = np.random.normal(0.0, (1 / xdim) ** 0.5, size=(ydim, xdim))
        self.Phi = nn.Parameter(torch.from_numpy(Phi_init).float(), requires_grad=True)
        self.Phi_T = nn.Parameter(torch.from_numpy(np.transpose(Phi_init)).float(), requires_grad=True)

    def sampling(self, inputs):
        x = vectorize(inputs,self.block_size,self.channels)
        y = torch.matmul(self.Phi, x)
        return y

    def initial(self, y, batch_size):
        x = torch.matmul(self.Phi_T, y)
        out = devectorize(x, batch_size,self.block_size,self.image_size,self.channels)
        return out
    
    def forward(self, x):
        batch_size = x.size()[0]
        y = self.sampling(x)
        output = self.initial(y, batch_size)
        return output, y, self.Phi
    
class DeepRecon(nn.Module):
    def __init__(self,
                 prior_dim,
                 dim = 32,
                 block_size = 32,
                 image_size = 64,
                 in_channels = 1,
                 stages = 6,
                 norm = 'gn'):
        super(DeepRecon,self).__init__()
        self.weights = []
        self.block_size = block_size
        self.image_size = image_size
        self.channles = in_channels
        self.stages = stages

        self.fe = nn.Conv2d(in_channels, dim-in_channels, kernel_size=3, stride=1, padding=1)

        self.transblock = nn.ModuleList()
        for i in range(stages):
            self.weights.append(nn.Parameter(torch.tensor(1.), requires_grad=True))
            self.transblock.append(CSCNNBlock(dim, prior_dim, norm = norm))
            
    def forward(self, ir, pri_f, y, Phi):
        batch_size = ir.size()[0]
        x = ir
        hidden_x = self.fe(x)
        for i in range(self.stages):
            x = vectorize(x,self.block_size,self.channles)
            x = x - self.weights[i] * torch.mm(torch.transpose(Phi, 0, 1), (torch.mm(Phi, x) - y))
            x = devectorize(x,batch_size,self.block_size,self.image_size,self.channles)
            output = self.transblock[i](x,hidden_x,pri_f)
            x = output[:, :1, :, :]
            hidden_x = output[:, 1:, :, :]
        final_x = x
        return final_x