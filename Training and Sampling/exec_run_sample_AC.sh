#!/bin/bash

dataname=newLentiMPRAHepG2_labels-seed0_random0_25000
chosen_model=NewResNet
MODEL_PATH=./exp_local/NewResNet_K562_25K_random/2024.06.24/114726
STEPS=128

cp run_sample_XXX.py run_sample.py
sed -i 's/XXX/'$dataname'/g' run_sample.py
sed -i 's/YYY/'$chosen_model'/g' run_sample.py

visdevs=$1 #0,1,2,3
if [ ${#visdevs} = 0 ] ; then visdevs=0 ; fi
echo "Visible devices: $visdevs"
date
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$visdevs python run_sample.py --model_path $MODEL_PATH --steps $STEPS


echo "SCRIPT END."