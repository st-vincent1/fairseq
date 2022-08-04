# Implementation of an experimental model: CUE Transformer
# Implemented in fairseq v0.12.2

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    BaseFairseqModel,
    register_model,
    register_model_architecture
)

from fairseq.models.cue_transformer.cue_config import (
    CUEConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)

from fairseq.models.fairseq_model import check_type
from fairseq.models.transformer import (
    TransformerEncoderBase,
    TransformerDecoderBase,
    Embedding
)
from fairseq.models.cue_transformer import (
    ContextEncoderBase,
    ContextDecoderBase
)
from fairseq.models.transformer.transformer_legacy import base_architecture

import logging


class DoubleEncoderDecoderModel(BaseFairseqModel):
    """Heavily based on FairseqEncoderDecoderModel, with the only changes being the extra encoder.
    Does not implement FairseqEncoderDecoderModel since the signature in forward needs to change"""

    @classmethod
    def build_model(cls, args, task):
        pass

    def __init__(self, cxt_encoder, src_encoder, decoder):
        super().__init__()

        self.cxt_encoder = cxt_encoder
        self.src_encoder = src_encoder
        self.decoder = decoder

        check_type(self.cxt_encoder, FairseqEncoder)
        check_type(self.src_encoder, FairseqEncoder)
        check_type(self.decoder, FairseqDecoder)

    # deleted cxt_lengths
    def forward(self, cxt_vectors, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        # cxt_encoder_out = self.encoder(cxt_vectors, cxt_lengths=cxt_lengths, **kwargs)
        cxt_encoder_out = self.encoder(cxt_vectors, **kwargs)
        src_encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, cxt_encoder_out=cxt_encoder_out, src_encoder_out=src_encoder_out, **kwargs
        )
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    # deleted lengths
    def extract_features(self, cxt_vectors, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # cxt_encoder_out = self.encoder(cxt_vectors, cxt_lengths=cxt_lengths, **kwargs)
        cxt_encoder_out = self.encoder(cxt_vectors, **kwargs)
        src_encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(
            prev_output_tokens, cxt_encoder_out=cxt_encoder_out, src_encoder_out=src_encoder_out, **kwargs
        )
        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.src_encoder.max_positions(), self.decoder.max_positions()

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


class CUETransformerBase(DoubleEncoderDecoderModel):
    """
    ST Vincent: Heavily based on fairseq/models/transformer/transformer_base.py:TransformerModelBase
    Adapted to extend DoubleEncoderDecoderModel (implemented above)

    --------------------------------------
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, cxt_encoder, src_encoder, decoder):
        super().__init__(cxt_encoder, src_encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, CUEConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing

        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.cxt_encoder.layers_to_keep:
            cfg.cxt_encoder.layers = len(cfg.cxt_encoder.layers_to_keep.split(","))
        if cfg.src_encoder.layers_to_keep:
            cfg.src_encoder.layers = len(cfg.src_encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.cxt_encoder.embed_dim != cfg.src_encoder.embed_dim \
                    and cfg.src_encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )

            if cfg.decoder.embed_path and (
                    cfg.decoder.embed_path != cfg.src_encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )

            src_encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.src_encoder.embed_dim, cfg.src_encoder.embed_path
            )
            decoder_embed_tokens = src_encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            src_encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.src_encoder.embed_dim, cfg.src_encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        cxt_encoder = cls.build_cxt_encoder(cfg)
        src_encoder = cls.build_source_encoder(cfg, src_dict, src_encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, cxt_encoder, src_encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_cxt_encoder(cls, cfg):
        return ContextEncoderBase(cfg)

    @classmethod
    def build_source_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        logging.info("Building decoder")
        return ContextDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
            self,
            cxt_vectors,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        ST Vincent: For now I am assuming that the outputs of the two encoders are added position-wise.
        ----------------------------------------------------------------
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        cxt_encoder_out = self.cxt_encoder(
            cxt_vectors=cxt_vectors,
            return_all_hiddens=return_all_hiddens
        )
        src_encoder_out = self.src_encoder(
            src_tokens=src_tokens, src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens
        )
        # if self.cfg.context_inclusion == 'add-encoder-outputs':
            # currently only supported option; adds cxt vectors to each encoder output position wise
            # src_encoder_out['encoder_out'] += cxt_encoder_out['cxt_encoder_out']

        encoder_out = src_encoder_out | cxt_encoder_out

        # Need to figure out what "prev_output_tokens" is during inference.
        # During training, I could simply replace
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


# Shadow of transformer_legacy.TransformerModel
@register_model('cue_transformer')
class CUETransformer(CUETransformerBase):
    def __init__(self, args, src_encoder, cxt_encoder, decoder):
        cfg = CUEConfig.from_namespace(args)
        super().__init__(cfg, src_encoder, cxt_encoder, decoder)
        self.args = args

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        gen_parser_from_dataclass(
            parser, CUEConfig(), delete_default=True, with_prefix=""
        )
        # ST Vincent: Adding extra parameters
        parser.add_argument('--context-inclusion', choices=['cxt-src-concat', 'add-encoder-outputs', 'tag-enc', 'replace-dec-bos'],
                            default='add-encoder-outputs', help='how output from context encoder should be included')
        parser.add_argument('--context-just-embed', default=False, action='store_true',
                            help='if True, context vectors get embedded and skip past ContextEncoder')
        parser.add_argument('--cls-dim', default=768, help='dimension of CLS token input')
        parser.add_argument('--context-average', default=False, action='store_true', help='average context or not')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.src_encoder_layers = len(args.encoder_layers_to_keep.split(","))
            args.cxt_encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        cfg = CUEConfig.from_namespace(args)
        return super().build_model(cfg, task)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
            CUEConfig.from_namespace(args), dictionary, embed_dim, path
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return super().build_encoder(
            CUEConfig.from_namespace(args), src_dict, embed_tokens
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return super().build_decoder(
            CUEConfig.from_namespace(args), tgt_dict, embed_tokens
        )


@register_model_architecture('cue_transformer', 'cue_transformer_base')
def cue_transformer_base(args):
    args.context_inclusion = getattr(args, 'context_inclusion', 'add-encoder-outputs')
    args.context_just_embed = getattr(args, 'context_just_embed', False)
    args.context_just_embed = getattr(args, 'context_average', False)
    args.cxt_encoder_layers = getattr(args, 'cxt_encoder_layers', 4)
    base_architecture(args)
