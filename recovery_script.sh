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
#for i in 2 3 4 5 6 7 8 9 10
#do
    #python scripts/amb_audio_training.py --seed $i --epochs 50 --align_to 'image' --graph_loc 'graphs/ambaudio/multimodal_brainlike.csv' --results_save 'results/ambimg_mel/thickness_'${i}'.npy' --model_save 'saved_models/ambimg_mel/brainlike_thickness_'${i}'.pt'
    #python scripts/amb_audio_training.py --seed $i --epochs 50 --align_to 'image' --graph_loc 'graphs/ambaudio/multimodal_brainlike_MPC.csv' --results_save 'results/ambimg_mel/MPC_'${i}'.npy' --model_save 'saved_models/ambimg_mel/brainlike_MPC_'${i}'.pt'
    #python scripts/amb_digit_training.py --seed $i --epochs 50 --graph_loc 'graphs/multimodal_brainlike.csv' --results_save 'results/audio_recovery_brainlike/thickness/brainlike_extra.npy' --model_save 'saved_models/ambimg_brainlike/brainlike_thickness_extra'${i}'.pt' --readout_save 'saved_models/ambimg_brainlike/brainlike_thickness_extra_classifier.pt'
    #python scripts/amb_audio_training.py --seed $i --epochs 50 --align_to 'image' --graph_loc 'graphs/ambaudio/multimodal_random.csv' --results_save 'results/ambimg_mel/random1_'${i}'.npy' --model_save 'saved_models/ambimg_mel/random1_'${i}'.pt'
    #python scripts/amb_audio_training.py --seed $i --epochs 50 --align_to 'image' --graph_loc 'graphs/ambaudio/multimodal_random2.csv' --results_save 'results/ambimg_mel/random2_'${i}'.npy' --model_save 'saved_models/ambimg_mel/random2_'${i}'.pt'
    #python scripts/amb_digit_training.py --seed $i --epochs 50 --graph_loc 'graphs/multimodal_big_rnn.csv' --results_save 'results/audio_recovery_brainlike/big_rnn_'${i}'.npy' --model_save 'saved_models/ambimg_brainlike/big_rnn_'${i}'.pt'
#done

#python scripts/amb_digit_training.py --seed 2 --epochs 50 --graph_loc 'graphs/multimodal_big_rnn.csv' --results_save 'dud.npy' --model_save 'saved_models/ambimg_brainlike/big_rnn_2.pt' --readout_save 'saved_models/ambimg_brainlike/big_rnn2_classifier.pt'
#python scripts/amb_digit_training.py --seed 1 --epochs 50 --graph_loc 'graphs/multimodal_random2.csv' --results_save 'dud.npy' --model_save 'saved_models/ambimg_brainlike/random2_extra.pt' --readout_save 'saved_models/ambimg_brainlike/brainlike_random2_extra_classifier.pt'
#python scripts/amb_audio_training.py --seed 1 --epochs 50 --align_to 'audio' --graph_loc 'graphs/ambaudio/multimodal_brainlike.csv' --results_save 'dud.npy' --model_save 'saved_models/ambaudio_brainlike/brainlike_extra_1.pt'
#python scripts/amb_audio_training.py --seed 2 --epochs 50 --align_to 'audio' --graph_loc 'graphs/ambaudio/multimodal_brainlike.csv' --results_save 'dud.npy' --model_save 'saved_models/ambaudio_brainlike/brainlike_extra_2.pt'
#python scripts/amb_audio_training.py --seed 3 --epochs 50 --align_to 'audio' --graph_loc 'graphs/ambaudio/multimodal_brainlike.csv' --results_save 'dud.npy' --model_save 'saved_models/ambaudio_brainlike/brainlike_extra_3.pt'
python scripts/amb_digit_training.py --seed 1 --epochs 1 --graph_loc 'graphs/multimodal_brainlike.csv' --results_save 'dud.npy' --model_save 'saved_models/ambimg_brainlike/brainlike_thickness_extra_1.pt' --readout_save 'saved_models/ambimg_brainlike/brainlike_1_classifier.pt' --hstates_save 'saved_models/hstates_vs/brainlike_hstate_1.pt'
python scripts/amb_digit_training.py --seed 2 --epochs 1 --graph_loc 'graphs/multimodal_brainlike.csv' --results_save 'dud.npy' --model_save 'saved_models/ambimg_brainlike/brainlike_thickness_extra_2.pt' --readout_save 'saved_models/ambimg_brainlike/brainlike_2_classifier.pt' --hstates_save 'saved_models/hstates_vs/brainlike_hstate_2.pt'
python scripts/amb_digit_training.py --seed 3 --epochs 1 --graph_loc 'graphs/multimodal_brainlike.csv' --results_save 'dud.npy' --model_save 'saved_models/ambimg_brainlike/brainlike_thickness_extra_3.pt' --readout_save 'saved_models/ambimg_brainlike/brainlike_3_classifier.pt' --hstates_save 'saved_models/hstates_vs/brainlike_hstate_3.pt'

