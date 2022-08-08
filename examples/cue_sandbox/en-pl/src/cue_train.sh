#!/bin/bash
set -e
SEED=1
ARCH=$1
ARGS=${@:2}
mkdir -p logs
CKPT=checkpoints/en-pl.${ARCH}
echo $ARGS
mkdir -p ${CKPT}
# currently lr must be in ARGS
CUDA_VISIBLE_DEVICES=0,1 fairseq-train data-bin/cue.en.pl/ \
    --max-update 150000 \
    --ddp-backend=legacy_ddp \
    --task cue_translation \
    --arch ${ARCH} \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    --no-epoch-checkpoints \
    --save-dir ${CKPT} \
    --max-tokens 20000 \
    --memory-efficient-fp16 \
    --tensorboard-logdir logs \
    --seed ${SEED} \
    ${ARGS} 

# removed save interval checkpoints
