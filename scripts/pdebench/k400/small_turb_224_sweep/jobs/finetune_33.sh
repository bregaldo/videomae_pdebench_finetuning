#!/bin/bash
#SBATCH --job-name=ft_k400_s_turb_224_33
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH -o jobs/finetune_33.log

module --force purge
source ~/.bashrc
module load modules/2.2
module load python/3.10.10
module load cuda/11.8 cudnn nccl
module load slurm
pysource videomae

# Wandb
wb_project=videomae_finetuning
wb_group=Calibration
wb_name=k400_s_turb_224_sweeps

# Directories
BASE_DIR=/mnt/home/bregaldosaintblancard/Projects/Foundation\ Models/VideoMAE_comparison
OUTPUT_DIR="$BASE_DIR/ceph/pdebench_finetuning/k400_s/$wb_name/"

# Data
data_set=compNS_turb # Among: compNS_turb, compNS_rand
fields=Vx,Vy,density
input_size=224
num_frames=16
sampling_rate=1
data_tmp_copy=False # Set to True if you want to first copy the dataset to /tmp

# Model
model_size=small # Among: small, base
checkpoint=k400_vit-s # Typically among: k400_vit-s, k400_vit-b, ssv2_vit-s, ssv2_vit-b
model=pretrain_videomae_${model_size}_patch16_${input_size} # Must be consistent with checkpoint

# Masking
mask_type=last_frame # Among: tube, last_frame
mask_ratio=0.9 # Only applicable for tube masking

# Normalization
norm_target_mode=last_frame

# Optimization
epoch=50
warmup_epochs=5
batch_size=4 # Batch size per GPU
num_workers=4
opt=adamw
lr=1e-3 # Base learning rate, effective one is determined through: lr * total_batch_size / 256
beta1=0.9
beta2=0.999
weight_decay=0.05

# Saving
save_ckpt_freq=25

export OMP_NUM_THREADS=1

cd "$BASE_DIR"

echo "Finetuning VideoMAE on PDEBENCH dataset"
echo "Python path: $(which python3)"
echo "Output dir: $OUTPUT_DIR"
echo "Finetuning model $checkpoint on $data_set dataset"

master_node=$SLURMD_NODENAME

srun python `which torchrun` \
        --nnodes $SLURM_JOB_NUM_NODES \
        --nproc_per_node $SLURM_GPUS_PER_NODE \
        --rdzv_id $SLURM_JOB_ID \
        --rdzv_backend c10d \
        --rdzv_endpoint $master_node:29500 \
        src/run_pdebench_finetuning.py \
        --wb_project $wb_project \
        --wb_group $wb_group \
        --wb_name $wb_name \
        --wb_sweep_id 7v4j0yxz \
        --data_set $data_set \
        --fields $fields \
        --input_size $input_size \
        --data_tmp_copy $data_tmp_copy \
        --mask_type $mask_type \
        --mask_ratio $mask_ratio \
        --model $model \
        --checkpoint $checkpoint \
        --lr $lr \
        --batch_size $batch_size \
        --num_workers $num_workers \
        --num_frames $num_frames \
        --sampling_rate $sampling_rate \
        --opt $opt \
        --opt_betas $beta1 $beta2 \
        --weight_decay $weight_decay \
        --warmup_epochs $warmup_epochs \
        --epochs $epoch \
        --save_ckpt_freq $save_ckpt_freq \
        --norm_target_mode $norm_target_mode \
        --log_dir "$OUTPUT_DIR" \
        --output_dir "$OUTPUT_DIR"
