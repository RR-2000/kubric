#!/bin/bash
#PBS -N movi_a
#PBS -l select=1:ncpus=16:ngpus=0:mem=32gb:host=cvml06

# Activate the Conda environment
# source /apps/miniconda3/etc/profile.d/conda.sh
source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
conda activate kubric

cd /home/ramanathan/VLM/kubric

# python /home/ramanathan/VLM/kubric/challenges/movi/export_movi_a.py \
#   --output-dir /home/ramanathan/data/movi_a_export \
#   --split validation \
#   --save-modalities

python /home/ramanathan/VLM/kubric/challenges/movi/build_movi_a_3dsr_dataset.py \
  --input-dir /home/ramanathan/data/movi_a_export \
  --output-jsonl /home/ramanathan/data/movi_a_3dsr_better_sample/movi_a_validation.jsonl \
  --output-parquet /home/ramanathan/data/movi_a_3dsr_better_sample/movi_a_validation.parquet \
  --sample-frame-start 20 \
  --seed 0