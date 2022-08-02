#!/bin/bash

set -e
ROOT=examples/cue_sandbox/en-pl
DATA=cue.en.pl
MODEL=cue.en.pl

SPM_MODEL=$ROOT/spm.bpe.model
TMP=examples/cue_sandbox/en-pl/tmp
mkdir -p $TMP

CKPT=${1:-'checkpoint_best'}
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

#fairseq-generate data-bin/${DATA} \
#    --task cue_translation --source-lang en --target-lang pl \
#    --path checkpoints/${MODEL}/${CKPT}.pt \
#    --batch-size 64 \
#    --remove-bpe=sentencepiece \
#    --context-inclusion cxt-src-concat > ${TMP}/test.sys #2> /dev/null 
#
#grep ^H ${TMP}/test.sys | LC_ALL=C sort -V | cut -f3- > ${TMP}/test.hyp

sacrebleu $ROOT/data/en-pl.test.pl -i ${TMP}/test.hyp


#rm -r ${TMP}
