#!/bin/bash

set -e
SRC=en
TGT=de

ROOT="examples/cue_sandbox"
SCRIPTS=scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

DATA=$ROOT/data
DEST=data-bin/cue.${SRC}.${TGT}.bpe8k
BPESIZE=8192
# Assume data is already downloaded

# This is optional as data will already have been filtered; keep for now
TRAIN_MINLEN=1  # remove sentences with <1 BPE token
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens

# learn BPE with sentencepiece
if [ ! -f $DATA/spm.bpe.vocab ]; then
  # shellcheck disable=SC2116
  TRAIN_FILES=$(echo $DATA/train.${SRC},$DATA/train.${TGT})
  echo "learning joint BPE over ${TRAIN_FILES}..."
  python "$SPM_TRAIN" \
      --input=$TRAIN_FILES \
      --model_prefix=$DATA/spm.bpe \
      --vocab_size=$BPESIZE \
      --character_coverage=1.0 \
      --model_type=bpe
fi

# ENCODE FILES WITH SENTENCEPIECE
if [ ! -f $DATA/train.bpe.${SRC} ]; then
        echo "encoding train with learned BPE..."
        python "$SPM_ENCODE" \
            --model "$DATA/spm.bpe.model" \
            --output_format=piece \
            --inputs $DATA/train.${SRC} $DATA/train.${TGT} \
            --outputs $DATA/train.bpe.${SRC} $DATA/train.bpe.${TGT} \
            --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN


        echo "encoding dev/test with learned BPE..."
        for SPLIT in dev tst-COMMON; do
          python "$SPM_ENCODE" \
              --model "$DATA/spm.bpe.model" \
              --output_format=piece \
              --inputs $DATA/${SPLIT}.${SRC} $DATA/${SPLIT}.${TGT} \
              --outputs $DATA/${SPLIT}.bpe.${SRC} $DATA/${SPLIT}.bpe.${TGT}
        done
fi

# BINARIZE DATA
fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT} \
    --trainpref $DATA/train.bpe \
    --validpref $DATA/dev.bpe \
    --testpref $DATA/tst-COMMON.bpe \
    --destdir ${DEST} \
    --workers 10

# Move cls embeddings to data-bin/
cp examples/cue_sandbox/data/dev.pkl ${DEST}/valid.${SRC}-${TGT}.pkl
cp examples/cue_sandbox/data/tst-COMMON.pkl ${DEST}/test.${SRC}-${TGT}.pkl
cp examples/cue_sandbox/data/train.pkl ${DEST}/train.${SRC}-${TGT}.pkl
