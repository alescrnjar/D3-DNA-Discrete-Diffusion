#!/bin/bash

batch_size=128 #128
name=NewResNet_K562 #deepstarr
seqlength=230 #249
ckpt_freq=50000
# Anirban: checkpoint_meta is the latest.

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
  n_iters: 500000
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



#echo "conda activate d3"
visdevs=$1 #0,1,2,3
if [ ${#visdevs} = 0 ] ; then visdevs=0 ; fi
echo "Visible devices: $visdevs"
date
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$visdevs python train.py noise.type=geometric graph.type=uniform model=small model.scale_by_sigma=False

#pip install hydra can work but if it doesnt:
#pip install hydra-core==1.3.2
#pip install hydra-submitit-launcher==1.2.0
