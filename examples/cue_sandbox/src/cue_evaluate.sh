#!/bin/bash

set -e
ROOT=examples/cue_sandbox/data
DATA=cue.en.de.bpe8k
MODEL=cue

SPM_MODEL=$ROOT/${TASK}/${TASK}.spm.bpe.model
TMP=examples/cue_sandbox/tmp
mkdir -p $TMP

CKPT=$1
#
## SPM encode
#cat test.en-de.en | python scripts/spm_encode.py \
#  --model ${SPM_MODEL} > ${TMP}/test.en-de.en.bpe
#
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

fairseq-generate data-bin/${DATA} \
    --task cue_translation --source-lang en --target-lang de \
    --path checkpoints/${MODEL}/${CKPT}.pt \
    --batch-size 128 \
    --remove-bpe=sentencepiece > ${TMP}/test.en-de.out
grep ^H ${TMP}/test.en-de.out | cut -f3 > ${TMP}/test.en-de.hyp

sacrebleu $ROOT/tst-COMMON.de -i ${TMP}/test.en-de.hyp
#rm -r ${TMP}
