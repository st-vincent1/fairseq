# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


class ContextEncoderLayerBase(nn.Module):

    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.encoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, cfg, return_fc=False):
        super().__init__()
        self.cfg = cfg
        self.return_fc = return_fc
        self.embed_dim = cfg.encoder.embed_dim
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size
        self.self_attn = self.build_self_attention(self.embed_dim, cfg)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.encoder.normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.encoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.encoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        self.num_heads = cfg.encoder.attention_heads
        self.load_to_BT = False
        self.ever_training = False
        # For BT, we need continuous mem
        self.in_proj_weight = torch.nn.Parameter(
            torch.zeros(
                self.self_attn.q_proj.weight.shape[0] * 3,
                self.self_attn.q_proj.weight.shape[1],
            )
        )
        self.in_proj_bias = torch.nn.Parameter(
            torch.zeros(self.self_attn.q_proj.bias.shape[0] * 3)
        )
        self.out_proj_weight = torch.nn.Parameter(
            torch.zeros(self.self_attn.out_proj.weight.shape)
        )
        self.out_proj_bias = torch.nn.Parameter(
            torch.zeros(self.self_attn.out_proj.bias.shape)
        )
        self.fc1_weight = torch.nn.Parameter(torch.zeros(self.fc1.weight.shape))
        self.fc1_bias = torch.nn.Parameter(torch.zeros(self.fc1.bias.shape))
        self.fc2_weight = torch.nn.Parameter(torch.zeros(self.fc2.weight.shape))
        self.fc2_bias = torch.nn.Parameter(torch.zeros(self.fc2.bias.shape))

        if (
            self.activation_fn is torch.nn.functional.relu
            or isinstance(self.activation_fn, torch.nn.ReLU)
            or self.activation_fn == "relu"
        ):
            self.activation_relu_or_gelu = 1
        elif (
            self.activation_fn is torch.nn.functional.gelu
            or isinstance(self.activation_fn, torch.nn.GELU)
            or self.activation_fn == "gelu"
        ):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        # Batch first can not be justified but needs user to make sure
        self.can_use_fastpath = (
            not self.normalize_before
            and self.activation_relu_or_gelu
            and (self.self_attn_layer_norm.eps == self.final_layer_norm.eps)
        )
        self.cfg_checkpoint_activations = self.cfg.checkpoint_activations
        # torch version check
        # make sure BT version is >=1.12.0
        self.BT_version = False
        if "fb" in torch.__version__:
            self.BT_version = True
        else:
            if "+" in torch.__version__:
                self.torch_version = torch.__version__.split("+")[0]
            else:
                self.torch_version = torch.__version__

            self.torch_version = self.torch_version.split(".")
            self.int_version = (
                int(self.torch_version[0]) * 1000
                + int(self.torch_version[1]) * 10
                + int(self.torch_version[2])
            )
            if len(self.torch_version) == 3:
                if self.int_version >= 1120:
                    self.BT_version = True
            elif len(self.torch_version) == 4:
                if self.int_version >= 1130:
                    self.BT_version = True

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.load_to_BT = True

        old_name = prefix + "self_attn."
        q_proj_weight = state_dict[old_name + "q_proj.weight"]
        k_proj_weight = state_dict[old_name + "k_proj.weight"]
        v_proj_weight = state_dict[old_name + "v_proj.weight"]
        q_proj_bias = state_dict[old_name + "q_proj.bias"]
        k_proj_bias = state_dict[old_name + "k_proj.bias"]
        v_proj_bias = state_dict[old_name + "v_proj.bias"]

        new_name = prefix
        state_dict[new_name + "in_proj_weight"] = torch.cat(
            (q_proj_weight, k_proj_weight, v_proj_weight), dim=0
        )
        state_dict[new_name + "in_proj_bias"] = torch.cat(
            (q_proj_bias, k_proj_bias, v_proj_bias), dim=0
        )
        state_dict[new_name + "out_proj_weight"] = state_dict[
            old_name + "out_proj.weight"
        ]
        state_dict[new_name + "out_proj_bias"] = state_dict[old_name + "out_proj.bias"]
        state_dict[new_name + "fc1_weight"] = state_dict[prefix + "fc1.weight"]
        state_dict[new_name + "fc1_bias"] = state_dict[prefix + "fc1.bias"]
        state_dict[new_name + "fc2_weight"] = state_dict[prefix + "fc2.weight"]
        state_dict[new_name + "fc2_bias"] = state_dict[prefix + "fc2.bias"]
        super(ContextEncoderLayerBase, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def _get_fc_rank(self, remove_num: int) -> List[int]:
        f1_filter_param = []
        for i in range(self.fc1.out_features):
            f1_filter_param.append(
                torch.sum(torch.abs(self.fc1.weight[i]))
                + torch.sum(torch.abs(self.fc2.weight[:, i]))
                + torch.abs(self.fc1.bias[i])
            )
        return sorted(
            range(len(f1_filter_param)), key=lambda k: f1_filter_param[k], reverse=False
        )[0:remove_num]

    def _prune_fc_layer(self, remove_index: List[int]):
        new_fc1_weight = []
        new_fc1_bias = []
        for i in range(self.fc1.out_features):
            if i not in remove_index:
                new_fc1_weight.append(self.fc1.weight[i])
                new_fc1_bias.append(self.fc1.bias[i])

        new_fc1_weight = torch.stack(new_fc1_weight).detach()
        new_fc1_weight.requires_grad = True

        new_fc1_bias = torch.stack(new_fc1_bias).detach()
        new_fc1_bias.requires_grad = True

        self.fc1 = quant_noise(
            nn.Linear(self.fc1.in_features, self.fc1.out_features - len(remove_index)),
            p=self.quant_noise,
            block_size=self.quant_noise_block_size,
        )
        self.fc1.weight = torch.nn.Parameter(new_fc1_weight)
        self.fc1.bias = torch.nn.Parameter(new_fc1_bias)

        new_fc2_weight = []
        new_fc2_bias = []
        for i in range(self.fc2.in_features):
            if i not in remove_index:
                new_fc2_weight.append(self.fc2.weight[:, i])
        new_fc2_bias = self.fc2.bias.detach()

        new_fc2_weight = torch.stack(new_fc2_weight, dim=-1).detach()
        new_fc2_weight.requires_grad = True

        new_fc2_bias = self.fc2.bias.detach()
        new_fc2_bias.requires_grad = True

        self.fc2 = quant_noise(
            nn.Linear(self.fc2.in_features - len(remove_index), self.fc2.out_features),
            p=self.quant_noise,
            block_size=self.quant_noise_block_size,
        )
        self.fc2.weight = torch.nn.Parameter(new_fc2_weight)
        self.fc2.bias = torch.nn.Parameter(new_fc2_bias)

    def build_self_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.encoder.xformers_att_config,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters

        if self.training:
            self.ever_training = True

        if (
            self.BT_version
            and x.dim() == 3
            and self.load_to_BT
            and not self.return_fc
            and self.can_use_fastpath
            and not self.training
            and not self.ever_training
            and not self.cfg_checkpoint_activations
        ):
            # assume is Batch first and nested tensor
            output = torch._transformer_encoder_layer_fwd(
                x,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj_weight,
                self.out_proj_bias,
                self.activation_relu_or_gelu == 2,
                False,  # norm_first, currently not supported
                self.self_attn_layer_norm.eps,
                self.self_attn_layer_norm.weight,
                self.self_attn_layer_norm.bias,
                self.final_layer_norm.weight,
                self.final_layer_norm.bias,
                self.fc1_weight,
                self.fc1_bias,
                self.fc2_weight,
                self.fc2_bias,
                encoder_padding_mask if encoder_padding_mask is not None else attn_mask,
            )
            return output

        else:
            if attn_mask is not None:
                attn_mask = attn_mask.masked_fill(
                    attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4
                )

            residual = x
            if self.normalize_before:
                x = self.self_attn_layer_norm(x)
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.self_attn_layer_norm(x)

            residual = x
            return residual

            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)

            fc_result = x

            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)

            if self.return_fc and not torch.jit.is_scripting():
                return x, fc_result
            return x

# backward compatible with the legacy argparse format
class ContextEncoderLayer(ContextEncoderLayerBase):
    def __init__(self, args):
        super().__init__(TransformerConfig.from_namespace(args))
        self.args = args

    def build_self_attention(self, embed_dim, args):
        return super().build_self_attention(
            embed_dim, TransformerConfig.from_namespace(args)
        )