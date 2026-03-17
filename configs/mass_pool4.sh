#!/bin/bash

gpu_id=${1}
if [ -z "$gpu_id" ]; then
  echo "No GPU ID provided. Using default GPU (0)."
  gpu_id=0
fi

python main.py \
  --train \
  --label "Mass" \
  --output_dir results \
  --data_dir '' \
  --clip_chk_pt_path /home/walsh/.cache/huggingface/hub/models--shawn24--Mammo-CLIP/snapshots/2f356926a00fc3f0d9fdad1193c1464fd9adf564/Pre-trained-checkpoints/b2-model-best-epoch-10.tar \
  --arch upmc_breast_clip_det_b2_period_n_ft \
  --csv_file vindrmammo_grouped_df.csv \
  --feat_dir /data/walsh/datasets/vindr-mammo-mil-b2 \
  --img_dir preprocessed_mammoclip \
  --dataset 'ViNDr' \
  --feature_extraction "offline" \
  --epochs 30 \
  --batch-size 8 \
  --eval_scheme 'kruns_train+val+test' \
  --n_runs 3 \
  --lr 5.0e-5 \
  --weighted-BCE 'y' \
  --mil_type 'pyramidal_mil' \
  --multi_scale_model 'fpn' \
  --fpn_dim 256 \
  --fcl_encoder_dim 256 \
  --fcl_dropout 0.25 \
  --pooling_type 'gated-attention' \
  --drop_attention_pool 0.25 \
  --type_scale_aggregator 'gated-attention' \
  --deep_supervision \
  --scales 16 32 128 \
  --device cuda:$gpu_id \
  --num-workers 0 \
  --final_pooling 'pool4' \
  --weight-decay 1e-1