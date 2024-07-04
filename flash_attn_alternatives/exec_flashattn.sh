#!/bin/bash

visdevs=$1 #0,1,2,3
if [ ${#visdevs} = 0 ] ; then visdevs=0 ; fi
echo "Visible devices: $visdevs"
date
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$visdevs python test1_flash_attn.py