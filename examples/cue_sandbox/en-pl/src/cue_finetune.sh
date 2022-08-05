#!/bin/bash
set -e
SEED=1
BLOCK_CXT_ENCODER=${1:-false}
SUFFIX=$2
mkdir -p logs
CKPT=checkpoints/cue.en.pl${SUFFIX}

if [ ${BLOCK_CXT_ENCODER} = true ]; then
  EXTRA_ARGS=(--context-just-embed)
fi

mkdir -p ${CKPT}
CUDA_VISIBLE_DEVICES=0,1 fairseq-train data-bin/cue.en.pl/ \
    --finetune-from-model ${CKPT}_nocxt/checkpoint_best.pt \
    --max-update 500000 \
    --ddp-backend=legacy_ddp \
    --task cue_translation \
    --arch cue_transformer_base \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0002 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    --no-epoch-checkpoints \
    --save-dir ${CKPT} \
    --max-tokens 100000 \
    --memory-efficient-fp16 \
    --tensorboard-logdir logs \
    --seed ${SEED} \
    --context-inclusion cxt-src-concat ${EXTRA_ARGS[@]} \

# removed save interval checkpoints
