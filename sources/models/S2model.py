import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange

from sources.util import instantiate_from_config
from sources.modules.ldm.ddpm import DDPM
from sources.models.S1model import DCSNet as dcs_s1
from sources.modules.losses.kd_loss import KDLoss
from sources.modules.compressedsensing.S2modules import denoise_cnn, ENCODES2

class DCSNet(pl.LightningModule):
    """main class"""
    def __init__(self,
                 s1_config,
                 n_denoise_res = 5,
                 timesteps = 5,
                 in_channels = 1,
                 prior_dim = 1,
                 loss_type="l2",
                 norm_type = 'gn',
                 block_size = 32,
                 patch_size = 64,
                 ckpt_path=None,
                 monitor="val/loss",
                 device = 'cuda',
                 linear_start= 0.1,
                 linear_end= 0.99,
                 warm_steps = 50000,
                 scheduler_config=None):
        super().__init__()
        self.setdevice = device
        self.loss_type = loss_type
        self.warm_steps = warm_steps
        self.block_size = block_size
        self.patch_size = patch_size
        condition = ENCODES2(prior_dim,ch_in=in_channels,norm=norm_type)
        denoise= denoise_cnn(prior_dim=prior_dim, n_feats=64, n_denoise_res=n_denoise_res,timesteps=timesteps, norm=norm_type)
        self.diffusion = DDPM(denoise=denoise, condition=condition ,n_feats=prior_dim,linear_start= linear_start,
                            linear_end= linear_end, timesteps = timesteps)
        if s1_config is not None:
            self.instantiate_S1_stage(s1_config)
        else:
            self.s1_model = dcs_s1(0.1)
        if monitor is not None:
            self.monitor = monitor

        self.kdloss = KDLoss()

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
            self.restarted_from_ckpt = True

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def instantiate_S1_stage(self, config):
        model = instantiate_from_config(config)
        self.s1_model = model
        for param in self.s1_model.dpem.parameters():
            param.requires_grad = False
    
    def get_ir(self, input):
        return self.s1_model.csinit(input)
    
    def apply_model(self, img):
        encoder_posterior = self.s1_model.get_encode(img)
        IPRS1 = encoder_posterior.mode().detach()
        ir, y, phi = self.get_ir(img)
        IPRS2,_ = self.diffusion(ir.detach(), IPRS1)
        IPRS2_F = self.s1_model.get_decode(IPRS2)
        
        if self.global_step <= self.warm_steps:
            with torch.no_grad():
                out = self.s1_model.deeprecon(ir, IPRS2_F, y, phi)
        else:
            out = self.s1_model.deeprecon(ir, IPRS2_F, y, phi)
        return out, IPRS2, IPRS1

    @torch.no_grad()
    def sample(self, img):
        ir, y, phi = self.s1_model.csinit(img)
        IPRS2 = self.diffusion(ir)
        IPRS2_F = self.s1_model.get_decode(IPRS2)
        out = self.s1_model.deeprecon(ir, IPRS2_F, y, phi)
        return out, IPRS2_F
    
    @torch.no_grad()
    def sampling(self, img):
        b, c, h, w = img.shape
        y = self.s1_model.csinit.sampling(img)
        y_reshape = torch.transpose(y, 0, 1).unsqueeze(-1).unsqueeze(-1)
        y_reshape = torch.cat(torch.split(y_reshape, split_size_or_sections=w//self.block_size, dim=0), dim=2)
        y_reshape = torch.cat(torch.split(y_reshape, split_size_or_sections=1, dim=0), dim=3)
        return y_reshape
    
    @torch.no_grad()  
    def implement(self, img, retrun_p = False, stride=1):
        bs = img.size()[0]
        kernel_size = self.patch_size//self.block_size
        y = self.sampling(img)
        fold, unfold, normalization, weighting = self.get_fold_unfold(y,stride=stride)
        y = unfold(y)
        y = y.view((y.shape[0], -1, kernel_size, kernel_size, y.shape[-1]))
        y = torch.cat(torch.split(y, split_size_or_sections=1, dim=3), dim=0)
        y = torch.cat(torch.split(y, split_size_or_sections=1, dim=2), dim=0)
        y = torch.reshape(y, [kernel_size*kernel_size, y.size()[1],-1])
        y = torch.transpose(y, 0, 1)
        ir_list = [self.s1_model.csinit.initial(y[:, :, i],bs) for i in range(y.shape[-1])]
        phi = self.s1_model.csinit.Phi
        IPRS2_list = [self.diffusion(ir_list[i]) for i in range(len(ir_list))]
        IPRS2_F_list = [self.s1_model.get_decode(IPRS2_list[i]) for i in range(len(ir_list))]
        recon_list = [self.s1_model.deeprecon(ir_list[i], IPRS2_F_list[i], y[:, :, i], phi) for i in range(len(ir_list))]

        out = torch.stack(recon_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
        out = out * weighting
        out = out.view((out.shape[0], -1, out.shape[-1]))
        out = fold(out)
        out = out / normalization 

        if retrun_p:
            pri = torch.stack(IPRS2_F_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
            pri = pri * weighting
            pri = pri.view((pri.shape[0], -1, pri.shape[-1]))
            pri = fold(pri)
            pri = pri / normalization 
            return out, pri
        else:
            return out


    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, 0.01, 0.5)
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)
        return weighting

    def get_fold_unfold(self, y, stride=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = y.shape
        kernel_size = self.patch_size//self.block_size

        Ly = (h - kernel_size) // stride + 1
        Lx = (w - kernel_size) // stride + 1

        fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
        unfold = torch.nn.Unfold(**fold_params)

        fold_params2 = dict(kernel_size=kernel_size * self.block_size, dilation=1, padding=0, stride=stride * self.block_size)
        fold = torch.nn.Fold(output_size=(y.shape[2] * self.block_size, y.shape[3] * self.block_size), **fold_params2)

        weighting = self.get_weighting(kernel_size * self.block_size, kernel_size * self.block_size, Ly, Lx, y.device).to(y.dtype)
        
        normalization = fold(weighting).view(1, 1, h * self.block_size, w * self.block_size)  # normalizes the overlap
        weighting = weighting.view((1, 1, kernel_size * self.block_size, kernel_size * self.block_size, Ly * Lx))
        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def get_input(self, batch, k='image', bs=None):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        return x

    def shared_step(self, batch, **kwargs):
        x = self.get_input(batch)
        loss = self(x)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start):
        target = x_start
        recon_x, pris2, pirs1 = self.apply_model(x_start)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        l_kd, loss_l = self.kdloss([pirs1],[pris2])
        loss_dict.update({f'{prefix}/loss_l': loss_l.detach()})
        loss_dict.update({f'{prefix}/loss_kd': l_kd.detach()})
        loss = loss_l
        loss_mse = self.get_loss(target,recon_x)
        loss_dict.update({f'{prefix}/loss_mse': loss_mse.detach()})
        if self.global_step > self.warm_steps:
            loss = loss + 10.0 * loss_mse
        
        loss_dict.update({f'{prefix}/loss': loss.detach()})
        return loss, loss_dict
    
    @torch.no_grad()
    def test(self, inputs, test = True):
        if test:
            return self.sample(inputs)[0]
        else:
            return self.sample(inputs)

    @torch.no_grad()
    def log_images(self, batch, N=8, **kwargs):
        log = dict()
        x = self.get_input(batch)
        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        
        recon_x, pir_f = self.test(x, test=False)

        log["inputs"] = x.detach()
        log["pir_f"] = pir_f.detach()
        log["reconstructon"] = recon_x.detach()
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.diffusion.parameters())
        params = params + list(self.s1_model.csinit.parameters())
        params = params + list(self.s1_model.deeprecon.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt