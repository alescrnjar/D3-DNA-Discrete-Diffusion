#$ -S /bin/bash
#$ -cwd
#$ -N D3_train
#$ -l m_mem_free=80G
#$ -l gpu=1 
#$ -o nohup.D3_train.out
#$ -e stderr.D3_train.out

##cudapremise="CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$visdev "
##cudapremise=""
## $cudapremise CUBLAS_WORKSPACE_CONFIG=:4096:8 python Merger_iAL_proposed.py
python train.py noise.type=geometric graph.type=uniform model=small model.scale_by_sigma=False

