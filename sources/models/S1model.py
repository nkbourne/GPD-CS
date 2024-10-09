import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange

from sources.util import instantiate_from_config
from sources.modules.compressedsensing.S1modules import CSINIT,DeepRecon,GPLM
from sources.modules.distributions.distributions import DiagonalGaussianDistribution

class DCSNet(pl.LightningModule):
    """main class"""
    def __init__(self,
                 sr, 
                 hidden_dim = 64,
                 block_size = 32,
                 image_size = 64,
                 in_channels = 1,
                 stages = 12,
                 prior_dim = 1,
                 loss_type = "l1",
                 norm_type = "gn",
                 ckpt_path=None,
                 monitor="val/loss",
                 device = 'cuda',
                 scheduler_config=None):
        super().__init__()
        self.setdevice = device
        self.loss_type = loss_type
        self.kl_weight = 0.000001
        self.csinit = CSINIT(sr,block_size,image_size,in_channels)
        self.dpem = GPLM(prior_dim,in_channels,norm=norm_type)
        self.deeprecon = DeepRecon(prior_dim, hidden_dim, block_size, image_size, in_channels,stages,norm=norm_type)

        if monitor is not None:
            self.monitor = monitor

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

    @torch.no_grad()
    def cs_sample(self, input):
        return self.csinit.sampling(input)
    
    @torch.no_grad()
    def get_ir(self, input):
        batch_size = input.size()[0]
        y = self.csinit.sampling(input)
        output = self.csinit.initial(y, batch_size)
        return output
    
    @torch.no_grad()
    def get_encode(self,inputs):
        moments = self.dpem.encode(inputs)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    @torch.no_grad()
    def get_decode(self,inputs):
        return self.dpem.decode(inputs)
    
    def apply_model(self, img, train = True):
        ir, y, phi = self.csinit(img)
        moments = self.dpem.encode(img)
        posterior = DiagonalGaussianDistribution(moments)
        if train:
            pri = posterior.sample()
        else:
            pri = posterior.mode()
        pri_f = self.dpem.decode(pri)
        out = self.deeprecon(ir, pri_f, y, phi)
        return out, pri_f, posterior

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

    def get_loss(self, pred, target, loss_ty, mean=True):
        if loss_ty == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif loss_ty == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    @torch.no_grad()
    def psnr(self, x,target):
        out = torch.nn.functional.mse_loss(target, x, reduction='none').mean(dim=[1, 2, 3])
        out = 20*torch.log10(2./torch.sqrt(out))
        out = torch.clamp(out,0,100)
        return out


    def p_losses(self, x_start):
        target = x_start
        recon_x, pri_f, posterior = self.apply_model(x_start,train=True)

        # loss = self.get_loss(target, pir_f, self.loss_type)

        loss_mse = self.get_loss(target, recon_x, self.loss_type)
        loss_kl = posterior.kl()
        loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
        loss = loss_mse + self.kl_weight * loss_kl

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{prefix}/loss_mse': loss_mse.detach().mean()})
        loss_dict.update({f'{prefix}/loss_kl': loss_kl.detach().mean()})
        loss_dict.update({f'{prefix}/loss': loss.detach().mean()})
        # psnr = self.psnr(recon_x,target)
        # loss_dict.update({f'{prefix}/psnr': psnr.detach().mean()})
        return loss, loss_dict
    
    @torch.no_grad()
    def test(self, inputs):
        recon_x, pri_f, _ = self.apply_model(inputs,train=False)
        return recon_x, pri_f

    @torch.no_grad()
    def log_images(self, batch, N=8, **kwargs):
        log = dict()
        x = self.get_input(batch)
        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        
        recon_x, pri_f = self.test(x)
        
        log["inputs"] = x.detach()
        log["reconstructon"] = recon_x.detach()
        log["prior"] = pri_f.detach()
        return log
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.csinit.parameters())
        params = params + list(self.dpem.parameters())
        params = params + list(self.deeprecon.parameters())
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
