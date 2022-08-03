#!/bin/bash

set -e
ROOT=examples/cue_sandbox/en-pl
DATA=cue.en.pl

SPM_MODEL=$ROOT/spm.bpe.model
TMP=examples/cue_sandbox/en-pl/tmp${SUFFIX}
mkdir -p $TMP

BLOCK_CXT_ENCODER=$1
SUFFIX=$2
MODEL=cue.en.pl${SUFFIX}
CKPT=${3:-'checkpoint_best'}
mkdir -p logs

if [ ${BLOCK_CXT_ENCODER} == true ]; then
  EXTRA_ARGS=(--context-just-embed)
fi

# SPM encode
#cat test.en-de.en | python scripts/spm_encode.py \
#  --model ${SPM_MODEL} > ${TMP}/test.en-de.en.bpe

#cat ${TMP}/test.en-de.en.bpe | fairseq-interactive data-bin/${DATA}/ \
#  --task cue_translation --source-lang en --target-lang de \
#  --path checkpoints/cue/checkpoint_best.pt \
#  --buffer-size 2000 --batch-size 128 \
#  --beam 5 --remove-bpe=sentencepiece \
#  ${TMP}/mustc.${SET}.en-${TGT}.${TGT}.sys 2> /dev/null
#
#
#grep ^H ${TMP}/mustc.${SET}.en-${TGT}.${TGT}.sys | cut -f3 > ${TMP}/mustc.${SET}.en-${TGT}.${TGT}.hyp
#sacrebleu $ROOT/${TASK}/en-${TGT}/${SET}.${TGT} -i ${TMP}/mustc.${SET}.en-${TGT}.${TGT}.hyp
#
#rm -r ${TMP}

#fairseq-generate data-bin/${DATA} \
#    --task cue_translation --source-lang en --target-lang pl \
#    --path checkpoints/${MODEL}/${CKPT}.pt \
#    --batch-size 128 \
#    --remove-bpe=sentencepiece \
#    --context-inclusion cxt-src-concat ${EXTRA_ARGS} > ${TMP}/test.sys 

#grep ^H ${TMP}/test.sys | LC_ALL=C sort -V | cut -f3- > ${TMP}/test.hyp

#sacrebleu $ROOT/data/en-pl.test.pl -i ${TMP}/test.hyp


python ${ROOT}/annotation_tool/annotate.py --src ${ROOT}/data/en-pl.test.en --mark ${ROOT}/data/en-pl.test.bpe.marking \
	--ref ${ROOT}/data/en-pl.test.pl --hyp ${TMP}/test.hyp --cxt ${ROOT}/data/en-pl.train.bpe.cxt 

#rm -r ${TMP}
