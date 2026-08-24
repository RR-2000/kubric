#!/bin/bash
#PBS -N movi_e_dir_obj
#PBS -l select=1:ncpus=16:ngpus=0:mem=48gb:host=cvml06

set -euo pipefail

source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
conda activate kubric

cd /home/ramanathan/VLM/kubric

python challenges/movi/export_movi_e.py \
  --dataset movi_e/256x256 \
  --split validation \
  --data-dir gs://kubric-public/tfds \
  --output-dir /home/ramanathan/data/movi_e_export \
  --save-modalities

python challenges/movi/build_movi_e_direction_object_dataset.py \
  --input-dir /home/ramanathan/data/movi_e_export \
  --output-dir /home/ramanathan/data/movi_e_better_sample \
  --split validation \
  --sample-frame-start 20 \
  --max-frames-per-sequence 4 \
  --max-pairs-per-sequence 24 \
  --seed 0
