#!/bin/bash
#SBATCH --job-name=datagen
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=2:00:00
#SBATCH --mem=32Gb

module --quiet load anaconda/3
conda activate con2model
python audiovisual_datagen_amb_audio.py