#!/bin/bash

#direc="../../../D3-DNA-Discrete-Diffusion/Training\ and\ Sampling"
#direc='../../../D3-DNA-Discrete-Diffusion/Training and Sampling'
#cd $direc

echo "conda activate d3"
# conda remove --name d3 --all

#cd "../../../D3-DNA-Discrete-Diffusion/Training and Sampling"
#python train.py noise.type=geometric graph.type=uniform model=small model.scale_by_sigma=False

visdevs=$1 #0,1,2,3
if [ ${#visdevs} = 0 ] ; then visdevs=0 ; fi
echo "Visible devices: $visdevs"
date
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$visdevs python train.py noise.type=geometric graph.type=uniform model=small model.scale_by_sigma=False




# PROSS: esegui senza il .h5 di deepstarr: cosi' capirai dove manca. Probabilmente data.py get_dataloaders

#pip install hydra can work but if it doesnt:
#pip install hydra-core==1.3.2
#pip install hydra-submitit-launcher==1.2.0
