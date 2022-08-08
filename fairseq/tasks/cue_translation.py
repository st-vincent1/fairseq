from dataclasses import dataclass, field

import logging
import numpy as np
import os
import glob
from functools import partial

from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    CueDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    indexed_dataset,
)

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask

from torch.utils.data import TensorDataset
from torch import FloatStorage, FloatTensor

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


def load_cue_dataset(
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        append_source_id=False,
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
        prepend_bos_src=None,
        # num_contexts=2
):
    """Load dataset.

    Args:
        src, tgt: lang codes
        """

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    # infer langcode
    if split_exists(split, src, tgt, src, data_path):
        prefix = os.path.join(data_path, "{}.{}-{}.".format(split, src, tgt))
    elif split_exists(split, tgt, src, src, data_path):
        prefix = os.path.join(data_path, "{}.{}-{}.".format(split, tgt, src))
    else:
        raise FileNotFoundError(
            "Dataset not found: {} ({})".format(split, data_path)
        )

    src_dataset = data_utils.load_indexed_dataset(
        prefix + src, src_dict, dataset_impl
    )
    if truncate_source:
        src_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDataset(src_dataset, src_dict.eos()),
                max_source_positions - 1,
            ),
            src_dict.eos(),
        )

    tgt_dataset = data_utils.load_indexed_dataset(
        prefix + tgt, tgt_dict, dataset_impl
    )

    logging.info(f"--- Loading context from {prefix}cxt.bin")
    cxt_vectors = FloatTensor(
        FloatStorage.from_file(
            f"{prefix}cxt.bin", shared=False, size=len(src_dataset) * 3 * 768)) \
        .reshape(len(src_dataset), 3, 768)

    cxt = TensorDataset(cxt_vectors)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return CueDataset(
        cxt,
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@dataclass
class CueConfig(TranslationConfig):
    """Copied config from Translation task. New arguments can be added below."""

    """These arguments are identical as in CUE Config. This is to eliminate discrepancies between train and generate"""
    context_just_embed: bool = field(
        default=False, metadata={"help": "if true, just embed context, don't go through layers of encoder"}
    )
    context_inclusion: str = field(default='add-encoder-outputs',
                                   metadata={"choices": ['cxt-src-concat', 'add-encoder-outputs', 'tag-enc',
                                                         'replace-dec-bos'],
                                             "help": 'how output from context encoder should be included'})
    context_average: bool = field(default=False,
                        metadata={"help": 'average context or not'}
    )
    cls_dim: int = field(default=768, metadata={"help": 'dimension of CLS token input'})
    cls_context: bool = field(default=False, metadata={"help": 'use cls token for context'})
    # cls_context: bool = field(default=False, metadata={"help": 'use cls token for context'})
    pretrain_only: bool = field(default=False, metadata={"help": 'if true, freeze context encoder and only pretrain encoder/decoder on translation'})

@register_task("cue_translation", dataclass=CueConfig)
class CueTranslationTask(TranslationTask):
    cfg: CueConfig

    @classmethod
    def setup_task(cls, cfg: CueConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Overwriting since cfg is different

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Overwriting to use a different function for loading dataset

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_cue_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            # num_contexts=self.num_contexts
        )

    def build_dataset_for_inference(self, cxt_vectors, src_tokens, src_lengths, constraints=None):
        """Builds dataset without targets. Overwriting due to inclusion of cxt_vectors"""
        return CueDataset(
            cxt_vectors,
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )
