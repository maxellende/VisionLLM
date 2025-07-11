#!/bin/bash
#SBATCH --job-name=eval_det_seg_coco # nom du job
#SBATCH --output=logs/date/%x-%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=logs/date/%x-%j.err # fichier d’erreur (%j = job ID)
#SBATCH --partition=gpu_p2  # 8 GPU V100
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:8 
#SBATCH --cpus-per-task=24 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=01:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --account=mao@v100 # comptabilite V100
#SBATCH --hint=nomultithread

set -x

date=$(date +%F)

mkdir -p logs/${date}
touch logs/${date}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out
touch logs/${date}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err
ln -f logs/all/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out logs/${date}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out
ln -f logs/all/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err logs/${date}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err

nvidia-smi -L

source ~/.bashrc
module load pytorch-gpu/py3/2.0.1
export PYTHONUSERBASE=$WORK/envs/vllmv2

# e.g.
bash scripts/vllmv2_7b/eval/dist_eval_det.sh work_dirs/visionllmv2-7b visionllmv2/datasets/configs/det/coco_val.py