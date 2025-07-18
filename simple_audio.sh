#!/bin/bash
#SBATCH --job-name=simple_image
#SBATCH --output=job_output.txt
#SBATCH --error=job_error_oscar.txt
#SBATCH --time=47:00:00
#SBATCH --mem=32Gb
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=unkillable

module load anaconda/3
conda activate con2model
for i in 6 7 8 9 10
do
    #python scripts/simple_training.py --seed $i --epochs 30 --graph_loc 'graphs/ambaudio/multimodal_brainlike.csv' --results_save 'results/image_benchmark/thickness_'${i}'.npy' --model_save 'saved_models/image_benchmark/brainlike_thickness_'${i}'.pt'
    #python scripts/simple_training.py --seed $i --epochs 30 --graph_loc 'graphs/ambaudio/multimodal_brainlike_MPC.csv' --results_save 'results/image_benchmark/MPC_'${i}'.npy' --model_save 'saved_models/image_benchmark/brainlike_MPC_'${i}'.pt'
    #python scripts/simple_training.py --seed $i --epochs 30 --graph_loc 'graphs/ambaudio/multimodal_random.csv' --results_save 'results/image_benchmark/random1_'${i}'.npy' --model_save 'saved_models/image_benchmark/random1_'${i}'.pt'
    #python scripts/simple_training.py --seed $i --epochs 30 --graph_loc 'graphs/ambaudio/multimodal_random2.csv' --results_save 'results/image_benchmark/random2_'${i}'.npy' --model_save 'saved_models/image_benchmark/random2_'${i}'.pt'
    python scripts/simple_audio.py --seed $i --align_to audio --epochs 30 --graph_loc 'graphs/ambaudio/multimodal_big_rnn.csv' --results_save 'results/audio_benchmark/big_rnn_'${i}'.npy' --model_save 'saved_models/audio_benchmark/big_rnn_'${i}'.pt'
done


