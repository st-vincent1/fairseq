#!/bin/bash

set -e
ROOT=examples/cue_sandbox/en-pl
DATA=cue.en.pl

SPM_MODEL=$ROOT/spm.bpe.model

BLOCK_CXT_ENCODER=$1
SUFFIX=$2
MODEL=cue.en.pl${SUFFIX}
CKPT=${3:-'checkpoint_best'}
mkdir -p logs
TMP=examples/cue_sandbox/en-pl/tmp${SUFFIX}
mkdir -p $TMP

if [ ${BLOCK_CXT_ENCODER} == true ]; then
  EXTRA_ARGS=(--context-just-embed)
fi


python ${ROOT}/annotation_tool/annotate.py --src ${ROOT}/data/en-pl.test.en --mark ${ROOT}/data/en-pl.test.bpe.marking \
	--ref ${ROOT}/data/en-pl.test.pl --hyp ${TMP}/test.hyp --cxt ${ROOT}/data/en-pl.train.bpe.cxt 

#rm -r ${TMP}
