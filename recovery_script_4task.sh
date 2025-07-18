#!/bin/bash
#SBATCH --job-name=big_rnn
#SBATCH --output=job_output_4task
#SBATCH --error=job_error_4task.txt
#SBATCH --time=47:00:00
#SBATCH --mem=32Gb
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=unkillable

module load anaconda/3
conda activate con2model
for i in 1 2 3
do
    python scripts/multimodal_all_scenarios.py --seed $i --epochs 1 --graph_loc 'graphs/4task_models/multimodal_brainlike.csv' --results_save 'results/4task_composite_hstates/thickness_'${i}'.npy' --model_save 'saved_models/4task_composite_hstates/brainlike_thickness_'${i}'.pt' --hstates_save 'saved_models/4task_composite_hstates/brainlike_thickness_hstate_'${i}'.pt' &
    python scripts/multimodal_all_scenarios.py --seed $i --epochs 1 --graph_loc 'graphs/4task_models/multimodal_brainlike_MPC.csv' --results_save 'results/4task_composite_hstates/MPC_'${i}'.npy' --model_save 'saved_models/4task_composite_hstates/brainlike_MPC_'${i}'.pt' --hstates_save 'saved_models/4task_composite_hstates/MPC_hstate_'${i}'.pt'&
    python scripts/multimodal_all_scenarios.py --seed $i --epochs 1 --graph_loc 'graphs/4task_models/multimodal_random.csv' --results_save 'results/4task_composite_hstates/random1_'${i}'.npy' --model_save 'saved_models/4task_composite_hstates/random1_'${i}'.pt' --hstates_save 'saved_models/4task_composite_hstates/random_hstate_'${i}'.pt' &
    #python scripts/multimodal_all_scenarios.py --seed $i --epochs 50 --graph_loc 'graphs/4task_models/multimodal_random2.csv' --results_save 'results/4task_composite/random2_'${i}'.npy' --model_save 'saved_models/4task_composite/random2_'${i}'.pt' &
    #python scripts/multimodal_all_scenarios.py --seed $i --epochs 50 --graph_loc 'graphs/4task_models/multimodal_random3.csv' --results_save 'results/4task_composite/random3_'${i}'.npy' --model_save 'saved_models/4task_composite/random3_'${i}'.pt'
    python scripts/multimodal_all_scenarios.py --seed $i --epochs 1 --graph_loc 'graphs/4task_models/multimodal_big_rnn.csv' --results_save 'results/4task_composite_hstates/big_rnn_'${i}'.npy' --model_save 'saved_models/4task_composite_hstates/big_rnn_'${i}'.pt' --reciprocal False --hstates_save 'saved_models/4task_composite_hstates/big_rnn_hstate_'${i}'.pt'
done