#!/bin/bash
#SBATCH --job-name=big_rnn_2
#SBATCH --output=job_output_latent
#SBATCH --error=job_error_latent.txt
#SBATCH --time=3:00:00
#SBATCH --mem=16Gb
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=main

module load anaconda/3
conda activate con2model

for i in 1
do
    python scripts/latent_analysis.py --graph_loc 'graphs/4task_models/multimodal_brainlike.csv' --save_hstates 'dim_red/saved_hstates/brainlike.npy' --model_root 'saved_models/4task_composite_hstates' --model_type 'brainlike_thickness' --align_to 'image' &
    python scripts/latent_analysis.py --graph_loc 'graphs/4task_models/multimodal_brainlike_MPC.csv' --save_hstates 'dim_red/saved_hstates/MPC.npy' --model_root 'saved_models/4task_composite_hstates' --model_type 'brainlike_MPC'
    #python scripts/latent_analysis.py --graph_loc 'graphs/4task_models/multimodal_random2.csv' --save_hstates 'dim_red/saved_hstates/random2.npy' --model_root 'saved_models/4task_composite' --model_type 'random2'
    #python scripts/latent_analysis.py --graph_loc 'graphs/4task_models/multimodal_big_rnn.csv' --save_hstates 'dim_red/saved_hstates/big_rnn.npy' --model_root 'saved_models/4task' --model_type 'big_rnn' --reciprocal False
    #python scripts/latent_analysis.py --graph_loc 'graphs/multimodal_big_rnn.csv' --save_hstates 'dim_red/saved_hstates/big_rnn_vs.npy' --model_root 'saved_models/ambimg_brainlike' --model_type 'big_rnn' --reciprocal False --align_to 'image' &
    #python scripts/latent_analysis.py --graph_loc 'graphs/ambaudio/multimodal_big_rnn.csv' --save_hstates 'dim_red/saved_hstates/big_rnn_as.npy' --model_root 'saved_models/ambaudio_brainlike/big_rnn' --model_type 'big_rnn' --reciprocal False --align_to 'audio'
done
