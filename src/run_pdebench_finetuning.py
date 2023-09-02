import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import json
import os
from pathlib import Path
from timm.models import create_model
from optim_factory import create_optimizer
from datasets import build_pdebench_dataset
from engine_for_pdebench_finetuning import train_one_epoch, test_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import wandb
import modeling_pretrain # for VideoMAE models


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args(interactive=False):
    parser = argparse.ArgumentParser('VideoMAE finetuning PDEBench script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)

    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--checkpoint', default=None, type=str)

    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')

    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube', 'last_frame'],
                        type=str, help='masked strategy of video tokens/patches')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    parser.add_argument('--norm_target_mode', default='videomae')
    # parser.add_argument('--normlize_target', default=True, type=bool,
    #                     help='normalized the target patch pixels')

    # Wandb parameters
    parser.add_argument('--wb_project', default='videomae_finetunint', type=str)
    parser.add_argument('--wb_group', default=None, type=str)
    parser.add_argument('--wb_name', default=None, type=str)
    parser.add_argument('--wb_sweep_id', default=None, type=str)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Dataset parameters
    parser.add_argument('--data_set', default='compNS_turb', type=str,
                        help='dataset')
    parser.add_argument('--fields', default=['Vx', 'Vy', 'density'], type=lambda x: x.split(','))
    parser.add_argument('--data_tmp_copy', default=False, type=str2bool)
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    if interactive:
        args = parser.parse_args('')
    else:
        args = parser.parse_args()

    if args.wb_sweep_id is None:
        args.wb_sweep_id = ''

    return args


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth,
        use_checkpoint=args.use_checkpoint
    )
    if args.checkpoint is not None:
        base_dir = '/mnt/home/bregaldosaintblancard/Projects/Foundation Models/VideoMAE_comparison/ceph/'
        if args.checkpoint == 'k400_vit-s':
            path = os.path.join(base_dir, 'pretrained_models/k400_small_1600_epochs/checkpoint.pth')
        elif args.checkpoint == 'k400_vit-b':
            path = os.path.join(base_dir, 'pretrained_models/k400_base_1600_epochs/checkpoint.pth')
        elif args.checkpoint == 'ssv2_vit-s':
            path = os.path.join(base_dir, 'pretrained_models/ssv2_small_2400_epochs/checkpoint.pth')
        elif args.checkpoint == 'ssv2_vit-b':
            path = os.path.join(base_dir, 'pretrained_models/ssv2_base_2400_epochs/checkpoint.pth')
        else:
            path = os.path.join(base_dir, args.checkpoint + '.pth')
        checkpoint = torch.load(path, map_location='cpu')
        utils.load_state_dict(model, checkpoint['model'])
    return model

def main(args):
    # Wandb initialization
    if len(args.wb_sweep_id) > 0: # if we are running a sweep
        wandb.init(group=args.wb_group,
                   job_type='finetuning',
                   mode=None if args.rank == 0  else "disabled")
        
        # We broadcast the sweep config to all processes
        config_updates_list = [None]
        if args.rank == 0:
            config_updates = []
            for key, value in wandb.config.items():
                config_updates.append((key, value))
            config_updates_list[0] = config_updates
        dist.broadcast_object_list(config_updates_list, src=0)
        dist.barrier()

        # We update the args with the sweep config
        for key, value in config_updates_list[0]:
            print(f"Updating {key} to {value}")
            setattr(args, key, value)

        # We update the wandb run name
        run_name = args.wb_name \
                 + f'_lr_{args.lr:.6f}'
        wandb.run.name = run_name
        if args.output_dir:
            args.output_dir = os.path.join(args.output_dir, run_name)
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    else:
        wandb.init(project=args.wb_project,
                   group=args.wb_group,
                   job_type='finetuning',
                   entity='flatiron-scipt',
                   name=args.wb_name,
                   config=args,
                   mode=None if args.rank == 0  else "disabled")

    device = torch.device(args.device)

    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)
    print("Model loaded")
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    print("output_dir=", args.output_dir)
    print("log_dir", args.log_dir)

    # Save args to output_dir
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(dict(args._get_kwargs()), f, indent=4)

    # get datasets
    dataset_train = build_pdebench_dataset(args, 'train')
    dataset_test = build_pdebench_dataset(args, 'test')

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank

    total_batch_size = args.batch_size * num_tasks
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
    )
    sampler_test = torch.utils.data.DistributedSampler(
        dataset_test, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    print("Sampler_test = %s" % str(sampler_test))


    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker
    )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    torch.cuda.empty_cache()
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            patch_size=patch_size[0],
            norm_target_mode=args.norm_target_mode,
        )
        test_stats = test_one_epoch(
            model, data_loader_test,
            device, epoch, log_writer=log_writer,
            patch_size=patch_size[0],
            norm_target_mode=args.norm_target_mode,
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                print("Saving model at epoch %d in %s" % (epoch + 1, args.output_dir))
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}
        
        wandb.log(log_stats)

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args()

    # Initialize distributed training
    utils.init_distributed_mode(args)
    print("Distributed mode initialized")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if len(args.wb_sweep_id) > 0:
        if args.output_dir:
            args.output_dir = os.path.join(args.output_dir, args.wb_sweep_id)
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        if args.rank == 0:
            wandb.agent(args.wb_sweep_id,
                        function=lambda: main(args),
                        entity='flatiron-scipt',
                        project=args.wb_project)
        else:
            main(args)
    else:
        main(args)
