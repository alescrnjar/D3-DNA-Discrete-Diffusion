#!/bin/bash

batch_size=128 #128
#name=NewResNet_K562 #deepstarr
#name=NewResNet_K562_25K_random #deepstarr
#name=LegNetPK_K562_20K_random 
name=LegNetPK_K562_130K_random 
#name=NewResNet_K562_20K_random 
seqlength=230 #249
ckpt_freq=50000
n_iters=500000
# Anirban: checkpoint_meta is the latest.
##dataname=newLentiMPRAK562_processed_for_dal
#dataname=newLentiMPRAK562_labels-seed0_random0_25000
#dataname=newLentiMPRAK562_labels-seed0_random0_20000
dataname=newLentiMPRAK562_labels-seed0_random0_130000
#chosen_model=NewResNet
chosen_model=LegNetPK

echo \
'name: small
type: ddit
hidden_size: 768
cond_dim: 128
length: '$seqlength'
n_blocks: 12
n_heads: 12
scale_by_sigma: True
dropout: 0.1
' > ./configs/model/small.yaml

echo \
'defaults:
  - _self_
  - model: small
  - override hydra/launcher: submitit_slurm

ngpus: 1
tokens: 4

training:
  batch_size: '$batch_size'
  accum: 1
  n_iters: '$n_iters'
  snapshot_freq: '$ckpt_freq'
  log_freq: 5000
  eval_freq: 5000
  snapshot_freq_for_preemption: 10000
  weight: standard
  snapshot_sampling: True
  ema: 0.9999

data:
  train: '$name'
  valid: '$name'
  cache_dir: data

graph:
  type: absorb
  file: data
  report_all: False

noise:
  type: loglinear
  sigma_min: 1e-4
  sigma_max: 20

sampling:
  predictor: euler
  steps: 128
  noise_removal: True

eval:
  batch_size: 256
  perplexity: False
  perplexity_batch_size: 32

optim:
  weight_decay: 0
  optimizer: AdamW
  lr: 3e-4
  beta1: 0.9
  beta2: 0.999
  eps: 1e-8
  warmup: 2500
  grad_clip: 1.


hydra:
  run:
    dir: exp_local/${data.train}/${now:%Y.%m.%d}/${now:%H%M%S}
  sweep:
    dir: exp/${data.train}/${now:%Y.%m.%d}/${now:%H%M%S}
    subdir: ${hydra.job.num}
  launcher:
    max_num_timeout: 100000
    # timeout_min: 10079
    partition: g40x
    account: stanford
    mem_gb: 96
    cpus_per_task: 40
    gpus_per_node: ${ngpus}
    constraint: null

' > ./configs/config.yaml


cp data_XXX.py data.py
sed -i 's/XXX/'$dataname'/g' data.py
sed -i 's/YYY/'$chosen_model'/g' data.py
cp run_sample_XXX.py run_sample.py
sed -i 's/XXX/'$dataname'/g' run_sample.py
sed -i 's/YYY/'$chosen_model'/g' run_sample.py

#echo "conda activate d3"
visdevs=$1 #0,1,2,3
if [ ${#visdevs} = 0 ] ; then visdevs=0 ; fi
echo "Visible devices: $visdevs"
date
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$visdevs python train.py noise.type=geometric graph.type=uniform model=small model.scale_by_sigma=False

# pip install torch==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117

#pip install hydra can work but if it doesnt:
#pip install hydra-core==1.3.2
#pip install hydra-submitit-launcher==1.2.0

echo "SCRIPT END."