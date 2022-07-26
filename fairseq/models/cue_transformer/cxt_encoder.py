"""The Lin. Proj. layer could be used as the embedding
So our "raw input" is BERT vectors (bsz, #contexts, 768)
This gets embedded into (bsz, #contexts, 768)
So the "embeddign" is really 768 x 768 forward layer
"""

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoder
from fairseq.models.cue_transformer import CUEConfig
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    transformer_layer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerEncoderBase":
        return "TransformerEncoder"
    else:
        return module_name


class ContextEncoderBase(FairseqEncoder):
    """
    Encoder for context.

    Transformer cxt_encoder consisting of *cfg.cxt_encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, return_fc=False):
        self.cfg = cfg
        super().__init__(dictionary=None)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.cxt_encoder_layerdrop = cfg.cxt_encoder.layerdrop
        self.return_fc = return_fc

        embed_dim = 512
        self.padding_idx = 0
        self.max_source_positions = cfg.max_source_positions

        self.lin_proj = nn.Linear(cfg.cls_dim, embed_dim, bias=False)
        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.cxt_encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.cxt_encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_cxt_encoder_layer(cfg) for i in range(cfg.cxt_encoder.layers)]
        )
        self.num_layers = len(self.layers)

        if cfg.cxt_encoder.normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

    def build_cxt_encoder_layer(self, cfg):
        layer = transformer_layer.TransformerEncoderLayerBase(
            cfg, return_fc=self.return_fc
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(self, cls_embeddings):
        # pass embeddings through a linear projection layer
        context_embedding = self.lin_proj(cls_embeddings)
        x = embed = self.embed_scale * context_embedding
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed


    def forward(
        self,
        cxt_vectors,
        return_all_hiddens: bool = False
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            cxt_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            dict:
                - **cxt_encoder_out** (Tensor): the last cxt_encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **cxt_encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **cxt_encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            cxt_vectors, return_all_hiddens
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        cxt_vectors,
        return_all_hiddens: bool = False
    ):
        """
        Args:
            cxt_vectors (LongTensor): context vectors in the shape of
                `(batch, cxt_len, cls_dim)`
            cxt_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            dict:
                - **cxt_encoder_out** (Tensor): the last cxt_encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **cxt_encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **cxt_encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        # cxt_encoder_padding_mask = torch.sum(cxt_vectors, dim=-1).eq(self.padding_idx)

        # has_pads = cxt_vectors.device.type == "xla" or cxt_encoder_padding_mask.any()

        x, cxt_encoder_embedding = self.forward_embedding(cxt_vectors)
        # account for padding while computing the representation
        # if has_pads:
        #     x = x * (1 - cxt_encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        cxt_encoder_states = []
        fc_results = []

        if return_all_hiddens:
            cxt_encoder_states.append(x)

        # nested tensor and BT enable
        layer = self.layers[0]
        BT_flag = False
        NT_flag = False
        # torch version check, BT>=1.12.0 and NT>=1.13.0.dev20220613
        # internal format is '1.13.0a0+fb'
        # external format is '1.13.0.dev20220613'(cpu&gpu) for nightly or "1.11.0"(cpu) or '1.11.0+cu102'(gpu) for stable
        BT_version = False
        NT_version = False
        if "fb" in torch.__version__:
            BT_version = True
            NT_version = True
        else:
            if "+" in torch.__version__:
                torch_version = torch.__version__.split("+")[0]
            else:
                torch_version = torch.__version__

            torch_version = torch_version.split(".")
            int_version = (
                int(torch_version[0]) * 1000
                + int(torch_version[1]) * 10
                + int(torch_version[2])
            )
            if len(torch_version) == 3:
                if int_version >= 1120:
                    BT_version = True
                if int_version >= 1131:
                    NT_version = True
            elif len(torch_version) == 4:
                if int_version >= 1130:
                    BT_version = True
                # Consider _nested_tensor_from_mask_left_aligned is landed after "20220613"
                if int_version >= 1131 or (
                    int_version == 1130 and torch_version[3][3:] >= "20220613"
                ):
                    NT_version = True

        if (
            BT_version
            and x.dim() == 3
            and layer.load_to_BT
            and not layer.return_fc
            and layer.can_use_fastpath
            and not layer.training
            and not layer.ever_training
            and not layer.cfg_checkpoint_activations
        ):
            # Batch first can not be justified but needs user to make sure
            x = x.transpose(0, 1)
            # Check mask conditions for nested tensor
            # if NT_version:
            #     if (
                    # cxt_encoder_padding_mask is not None
                    # and torch._nested_tensor_from_mask_left_aligned(
                    #     x, cxt_encoder_padding_mask.logical_not()
                    # )
                # ):
                #     if not torch.is_grad_enabled() or not x.requires_grad:
                #         x = torch._nested_tensor_from_mask(
                #             x, cxt_encoder_padding_mask.logical_not()
                #         )
                #         NT_flag = True
            # BT_flag = True

        # cxt_encoder layers
        # if NT_flag:
        #     processing_mask = None
        # else:
        #     processing_mask = cxt_encoder_padding_mask
        # cxt_encoder_padding_mask_out = processing_mask if has_pads else None
        for layer in self.layers:
            # lr = layer(x, encoder_padding_mask=cxt_encoder_padding_mask_out)
            lr = layer(x, encoder_padding_mask=None)

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert cxt_encoder_states is not None
                cxt_encoder_states.append(x)
                fc_results.append(fc_result)

        # change back to non-nested and Batch second
        if NT_flag:
            x = x.to_padded_tensor(0.0)

        if NT_flag or BT_flag:
            x = x.transpose(0, 1)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # average outputs
        x = torch.mean(x, dim=0)

        return {
            "cxt_encoder_out": [x],  # T x B x C
            # "cxt_encoder_padding_mask": [cxt_encoder_padding_mask],  # B x T
            "cxt_encoder_embedding": [cxt_encoder_embedding],  # B x T x C
            "cxt_encoder_states": cxt_encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "cxt_vectors": [],  # B x T x E
        }

    @torch.jit.export
    def reorder_encoder_out(self, cxt_encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder cxt_encoder output according to *new_order*.

        Args:
            cxt_encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *cxt_encoder_out* rearranged according to *new_order*
        """
        if len(cxt_encoder_out["cxt_encoder_out"]) == 0:
            new_cxt_encoder_out = []
        else:
            new_cxt_encoder_out = [cxt_encoder_out["cxt_encoder_out"][0].index_select(0, new_order)]
        # if len(cxt_encoder_out["cxt_encoder_padding_mask"]) == 0:
        #     new_cxt_encoder_padding_mask = []
        # else:
        #     new_cxt_encoder_padding_mask = [
        #         cxt_encoder_out["cxt_encoder_padding_mask"][0].index_select(0, new_order)
        #     ]
        if len(cxt_encoder_out["cxt_encoder_embedding"]) == 0:
            new_cxt_encoder_embedding = []
        else:
            new_cxt_encoder_embedding = [
                cxt_encoder_out["cxt_encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(cxt_encoder_out["cxt_vectors"]) == 0:
            cxt_vectors = []
        else:
            cxt_vectors = [(cxt_encoder_out["cxt_vectors"][0]).index_select(0, new_order)]

        # if len(cxt_encoder_out["cxt_lengths"]) == 0:
        #     cxt_lengths = []
        # else:
        #     cxt_lengths = [(cxt_encoder_out["cxt_lengths"][0]).index_select(0, new_order)]

        cxt_encoder_states = cxt_encoder_out["cxt_encoder_states"]
        if len(cxt_encoder_states) > 0:
            for idx, state in enumerate(cxt_encoder_states):
                cxt_encoder_states[idx] = state.index_select(1, new_order)

        return {
            "cxt_encoder_out": new_cxt_encoder_out,  # T x B x C
            # "cxt_encoder_padding_mask": new_cxt_encoder_padding_mask,  # B x T
            "cxt_encoder_embedding": new_cxt_encoder_embedding,  # B x T x C
            "cxt_encoder_states": cxt_encoder_states,  # List[T x B x C]
            "cxt_vectors": cxt_vectors,  # B x T
            # "cxt_lengths": cxt_lengths,  # B x 1
        }

    @torch.jit.export
    def _reorder_encoder_out(self, cxt_encoder_out: Dict[str, List[Tensor]], new_order):
        """Dummy re-order function for beamable enc-dec attention"""
        return cxt_encoder_out

    def max_positions(self):
        """Maximum input length supported by the cxt_encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class ContextEncoder(ContextEncoderBase):
    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        self.args = args
        super().__init__(
            CUEConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )

    def build_cxt_encoder_layer(self, args):
        return super().build_cxt_encoder_layer(
            CUEConfig.from_namespace(args),
        )
