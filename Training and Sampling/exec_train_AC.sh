#!/bin/bash

#direc="../../../D3-DNA-Discrete-Diffusion/Training\ and\ Sampling"
#direc='../../../D3-DNA-Discrete-Diffusion/Training and Sampling'
#cd $direc

echo "conda activate d3"
# conda remove --name d3 --all

#cd "../../../D3-DNA-Discrete-Diffusion/Training and Sampling"
python train.py noise.type=geometric graph.type=uniform model=small model.scale_by_sigma=False

# PROSS: esegui senza il .h5 di deepstarr: cosi' capirai dove manca. Probabilmente data.py get_dataloaders