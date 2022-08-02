#!/bin/bash

set -e
SRC=en
TGT=pl

ROOT="examples/cue_sandbox/en-pl"
SCRIPTS=scripts

DATA=${ROOT}/data/${SRC}-${TGT}
DEST=data-bin/cue.${SRC}.${TGT}

# BINARIZE DATA
fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT} \
    --trainpref ${DATA}.train.bpe \
    --validpref ${DATA}.dev.bpe \
    --testpref ${DATA}.test.bpe \
    --joined-dictionary \
    --destdir ${DEST} \
    --workers 10

# Move cls embeddings to data-bin/
cp ${ROOT}/data/dev.bin ${DEST}/valid.${SRC}-${TGT}.bin
cp ${ROOT}/data/test.bin ${DEST}/test.${SRC}-${TGT}.bin
cp ${ROOT}/data/train.bin ${DEST}/train.${SRC}-${TGT}.bin
