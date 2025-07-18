#!/bin/bash
#SBATCH --job-name=big_rnn
#SBATCH --output=amg_fig5_train.txt
#SBATCH --error=job_error_ambimg.txt
#SBATCH --time=47:00:00
#SBATCH --mem=32Gb
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=main

module load anaconda/3
conda activate con2model

# Define seeds and graphs
seeds=(1 2 3)
graphs=("big_rnn")

# Base directories for saving models and results
model_dir="saved_models/ambaudio_brainlike"
hstate_dir="saved_models/hstates_as"
graph_dir="graphs/ambaudio/multimodal"

# Iterate over each graph
for graph in "${graphs[@]}"; do
  for seed in "${seeds[@]}"; do
  
    model_save="${model_dir}/${graph}_${seed}.pt"
    #readout_save="${model_dir}/${graph}_${seed}.pt"
    hstates_save="${hstate_dir}/${graph}_hstate_${seed}.pt"
    
    # Run the Python script
    python scripts/amb_audio_training.py \
      --seed "$seed" \
      --epochs 1 \
      --graph_loc "graphs/ambaudio/multimodal_${graph}.csv" \
      --results_save "dud.npy" \
      --model_save "$model_save" \
      --hstates_save "$hstates_save" \
      --reciprocal False
      
    # Check if the script ran successfully
    if [[ $? -ne 0 ]]; then
      echo "Script failed for graph $graph with seed $seed."
      exit 1
    else
      echo "Completed for graph $graph with seed $seed."
    fi
  done
done

echo "All runs completed successfully."