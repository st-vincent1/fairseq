#!/bin/bash
set -e
CKPT=checkpoints/cue
SEED=1
mkdir -p logs

mkdir -p ${CKPT}
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/cue.en.de.bpe8k/ \
    --max-update 20000 \
    --ddp-backend=legacy_ddp \
    --task cue_translation \
    --arch cue_transformer_base \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    --save-interval-updates 1500 \
    --no-epoch-checkpoints \
    --save-dir ${CKPT} \
    --max-tokens 1000 \
    --update-freq 4 --fp16 \
    --tensorboard-logdir logs \
    --seed ${SEED} \
    --context-inclusion add-encoder-outputs
