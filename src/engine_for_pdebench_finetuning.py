import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class NMSELoss(nn.MSELoss):
    """ Normalized MSE Loss: L = MSE(input/target_norm, target/target_norm) where target_norm is the l2 norm on spatial dimensions
        This normalization is important when the channels represent different physical quantities (e.g. velocity, pressure, etc.).
    """
    def __init__(self, reduction='mean', p0=2, p1=16, p2=16, p0_sel=0):
        super().__init__(reduction=reduction)
        self.eps = 1e-7
        self.p0 = p0 # Number of frames per cube/token
        self.p1 = p1 # Height of the cube/token
        self.p2 = p2 # Width of the cube/token
        self.p0_sel = p0_sel # So that normalization only uses a specific frame of the cube (to match Mike's normalization where only one frame is predicted instead of p0)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # input, target: (Batch, Nb_masked_tokens, Patch) where Patch = p0*p1*p2*c
        input = rearrange(input, 'b n (p0 p1 p2 c) -> b n p0 (p1 p2) c', p0=self.p0, p1=self.p1, p2=self.p2)
        target = rearrange(target, 'b n (p0 p1 p2 c) -> b n p0 (p1 p2) c', p0=self.p0, p1=self.p1, p2=self.p2)
        target_norm = torch.sqrt((target**2).mean(dim=-2, keepdim=True) + self.eps)
        if self.p0_sel is not None:
            target_norm = target_norm[:, :, self.p0_sel:self.p0_sel+1]
        return super().forward(input/target_norm, target/target_norm)
    
def norm_batch(x, norm_mode='videomae', patch_size=(2, 16, 16)):
    """ Normalize a batch of videos x, where x is of shape (Batch, Channel, Time, Height, Width)"""
    p0, p1, p2 = patch_size
    if norm_mode == 'videomae':
        x_squeeze = rearrange(x, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=p0, p1=p1, p2=p2)
        x_norm = (x_squeeze - x_squeeze.mean(dim=-2, keepdim=True)
            ) / (x_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        # we find that the mean is about 0.48 and standard deviation is about 0.08.
        x_patch = rearrange(x_norm, 'b n p c -> b n (p c)')
    elif norm_mode == 'last_frame':
        x_squeeze = rearrange(x, 'b c (t p0) (h p1) (w p2) -> b t (h w) (p0 p1 p2) c', p0=p0, p1=p1, p2=p2)
        x_squeeze_mean = x_squeeze[:, :-1].mean(dim=(1, 2, 3), keepdim=True)
        x_squeeze_std = x_squeeze[:, :-1].var(dim=(1, 2, 3), keepdim=True, unbiased=True).sqrt() + 1e-6
        x_norm = (x_squeeze - x_squeeze_mean) / x_squeeze_std
        x_patch = rearrange(x_norm, 'b t n p c -> b (t n) (p c)')
    elif norm_mode == 'none':
        x_patch = rearrange(x, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=p0, p1=p1, p2=p2)
    else:
        raise NotImplementedError
    return x_patch

def unnorm_batch(x, norm_mode='videomae', patch_size=(2, 16, 16), context=None, bool_masked_pos=None):
    """ Inverse function for norm_batch. (WARNING: this function needs to be finished)
        Input x is of shape (Batch, Nb_masked_tokens, Patch_size*Channel)
        context when provided should be of shape (Batch, Channel, Time, Height, Width)
        bool_masked_pos when provided should be of shape (Batch, Nb_tokens)
        Output is of shape (Batch, Nb_masked_tokens, Patch_size*Channel)
    """
    p0, p1, p2 = patch_size
    if norm_mode == 'videomae':
        # context must be here masked x
        B, C = context.shape[0], context.shape[1]
        context_squeeze = rearrange(context, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=p0, p1=p1, p2=p2)
        context_squeeze_mean = context_squeeze.mean(dim=-2, keepdim=True)
        context_squeeze_std = context_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
        x_squeeze_mean_loc = context_squeeze_mean[bool_masked_pos].reshape(B, -1, 1, C)
        x_squeeze_std_loc = context_squeeze_std[bool_masked_pos].reshape(B, -1, 1, C)
        x = rearrange(x, 'b n (p c) -> b n p c', c=C)
        x_unnorm = x * x_squeeze_std_loc + x_squeeze_mean_loc
        x_unnorm = rearrange(x_unnorm, 'b n p c -> b n (p c)')
    elif norm_mode == 'last_frame':
        B, C = context.shape[0], context.shape[1]
        context_squeeze = rearrange(context, 'b c (t p0) (h p1) (w p2) -> b t (h w) (p0 p1 p2) c', p0=p0, p1=p1, p2=p2)
        context_squeeze_mean = context_squeeze[:, :-1].mean(dim=(1, 2, 3), keepdim=True)
        context_squeeze_std = context_squeeze[:, :-1].var(dim=(1, 2, 3), keepdim=True, unbiased=True).sqrt() + 1e-6
        context_squeeze_mean = rearrange(context_squeeze_mean, 'b 1 1 1 c -> b 1 1 c')
        context_squeeze_std = rearrange(context_squeeze_std, 'b 1 1 1 c -> b 1 1 c')
        x = rearrange(x, 'b n (p c) -> b n p c', c=C)
        x_unnorm = x * context_squeeze_std + context_squeeze_mean
        x_unnorm = rearrange(x_unnorm, 'b n p c -> b n (p c)')
    elif norm_mode == 'none':
        x_unnorm = x
    else:
        raise NotImplementedError
    return x_unnorm

def get_targets(videos, bool_masked_pos, norm_target_mode, p0=2, p1=16, p2=16):
     with torch.no_grad():
        videos_patch = norm_batch(videos, norm_mode=norm_target_mode, patch_size=(p0, p1, p2))
        B, _, C = videos_patch.shape
        labels = videos_patch[bool_masked_pos].reshape(B, -1, C)
        return labels # Labels are a misnomer here, they are the targets (but name is kept for consistency with the rest of the code)

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    norm_target_mode: str = 'videomae', log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}] (train)'.format(epoch)
    print_freq = 10

    # Patch size in time, height, width
    p0, p1, p2 = 2, patch_size, patch_size
    
    #loss_func = nn.MSELoss()
    loss_func = NMSELoss(p0=p0, p1=p1, p2=p2, p0_sel=0)

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # Shapes:
        # videos: (Batch, Channel, Time, Height, Width)
        # bool_masked_pos: (Batch, Nb_tokens)

        labels = get_targets(videos, bool_masked_pos, norm_target_mode, p0=p0, p1=p1, p2=p2)

        with torch.cuda.amp.autocast():
            outputs = model(videos, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def test_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                   device: torch.device, epoch: int, patch_size: int = 16, 
                   norm_target_mode: str = 'videomae', log_writer=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}] (test)'.format(epoch)
    print_freq = 10

    # Patch size in time, height, width
    p0, p1, p2 = 2, patch_size, patch_size

    #loss_func = nn.MSELoss()
    loss_func = NMSELoss(p0=p0, p1=p1, p2=p2, p0_sel=0)

    for _, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        videos, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        labels = get_targets(videos, bool_masked_pos, norm_target_mode, p0=p0, p1=p1, p2=p2)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(videos, bool_masked_pos)
                loss = loss_func(input=outputs, target=labels)

            loss_value = loss.item()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
