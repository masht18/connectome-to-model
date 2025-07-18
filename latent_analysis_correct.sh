#!/bin/bash
#SBATCH --job-name=brainlike_latent
#SBATCH --output=amg_fig6_train.txt
#SBATCH --error=job_error_ambimg_comp.txt
#SBATCH --time=47:00:00
#SBATCH --mem=32Gb
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=long

module load anaconda/3
conda activate con2model

# Define seeds and graphs
seeds=(1 2 3)
graphs=("brainlike_thickness")

# Base directories for saving models and results
model_dir="saved_models/ambimg_composite"
hstate_dir="saved_models/hstates_vs"
graph_dir="graphs/multimodal"

# Iterate over each graph
for graph in "${graphs[@]}"; do
  for seed in "${seeds[@]}"; do
  
    model_save="${model_dir}/${graph}_${seed}.pt"
    #readout_save="${model_dir}/${graph}_${seed}.pt"
    hstates_save="${hstate_dir}/${graph}_hstate_composite${seed}.pt"
    
    # Run the Python script
    python scripts/amb_digit_training.py \
      --seed "$seed" \
      --epochs 1 \
      --graph_loc "graphs/multimodal_${graph}.csv" \
      --results_save "dud.npy" \
      --model_save "$model_save" \
      --hstates_save "$hstates_save"
      
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