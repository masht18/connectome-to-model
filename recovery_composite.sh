#!/bin/bash
#SBATCH --job-name=ambimg_composite_random
#SBATCH --output=job_output.txt
#SBATCH --error=job_error_oscar.txt
#SBATCH --time=47:00:00
#SBATCH --mem=32Gb
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=unkillable

module load anaconda/3
conda activate con2model
for i in 5 6 7 8 9 10
do
    #python scripts/amb_audio_training.py --seed $i --epochs 50 --align_to 'image' --graph_loc 'graphs/ambaudio_composite/multimodal_brainlike_32.csv' --results_save 'results/ambimg_composite_mfcc/thickness_'${i}'.npy' --model_save 'saved_models/ambimg_composite_mfcc/thickness_'${i}'.pt'
    #python scripts/amb_audio_training.py --seed $i --epochs 50 --align_to 'image' --graph_loc 'graphs/ambaudio_composite/multimodal_brainlike_MPC_32.csv' --results_save 'results/ambimg_composite_mfcc/MPC_'${i}'.npy' --model_save 'saved_models/ambimg_composite_mfcc/brainlike_MPC_'${i}'.pt'
    #python scripts/amb_audio_training.py --seed $i --epochs 50 --align_to 'image' --graph_loc 'graphs/ambaudio_composite/multimodal_random_32.csv' --results_save 'results/ambimg_composite_mfcc/random1_'${i}'.npy' --model_save 'saved_models/ambimg_composite_mfcc/random1_'${i}'.pt'
    #python scripts/amb_audio_training.py --seed $i --epochs 50 --align_to 'image' --graph_loc 'graphs/ambaudio_composite/multimodal_random2_32.csv' --results_save 'results/ambimg_composite_mfcc/random2_'${i}'.npy' --model_save 'saved_models/ambimg_composite_mfcc/random2_'${i}'.pt'
    python scripts/amb_audio_training.py --seed $i --epochs 50 --align_to 'image' --graph_loc 'graphs/ambaudio_composite/multimodal_random3_32.csv' --results_save 'results/ambimg_composite_mfcc/random3_'${i}'.npy' --model_save 'saved_models/ambimg_composite_mfcc/random3_'${i}'.pt' --hstates_save 'saved_models/hstates_vs/big_rnn_hstate_'${i}'.pt'
done


