import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional, Union, Tuple
from transformers import CLIPVisionConfig, PretrainedConfig

from experiments.multi_test import load_safe_tensors
from experiments.multimodal.vision_transformer import ConfigVision


class MPNetAttention:
    """
        Implements the attention mechanism for MPNet.

        This class is responsible for computing the attention scores and applying
        them to the input hidden states, following the architecture of MPNet. It
        involves multiple linear layers for query (q), key (k), value (v), and output
        (o), along with layer normalization and intermediate dense layers.

        Parameters
        ----------
        num_hidden_layers : int
            The number of hidden layers in the attention mechanism.
        all_head_size : int
            Total size of all attention heads.
        hidden_size : int
            The size of the hidden layers.
        intermediate_size : int
            The size of the intermediate layers.
        weights : Dict
            A dictionary containing the weights for various layers.
        attention_head_size : int
            The size of each attention head.
    """

    def __init__(self, config: ConfigVision,
                 weights: dict):

        self.num_hidden_layers = config.num_hidden_layers

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout

        self.weights = weights

        self.attention_head_size = self.hidden_size//self.num_heads
        if self.attention_head_size * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.attention_head_size ** -.5

        self.q_linears = [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(12)]
        self.k_linears = [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(12)]
        self.v_linears = [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(12)]
        self.o_linears = [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(12)]

        self.layer_norm = [
            nn.LayerNorm(self.hidden_size, gamma=torch.ones(self.hidden_size), beta=torch.zeros(self.hidden_size))
            for _ in range(12)]

        self.output_dense = [nn.Linear(self.intermediate_size, self.hidden_size) for _ in range(12)]

        self.intermediate_dense = [nn.Linear(self.hidden_size, self.intermediate_size) for _ in range(12)]

        self.output_layernorm = [nn.LayerNorm(self.hidden_size, gamma=np.ones(self.hidden_size),
                                              beta=np.zeros(self.hidden_size)) for _ in range(12)]

    def forward(self, hidden_states: torch.Tensor, relative_position_bias: np.ndarray,
                extended_attention_mask: np.ndarray) ->  torch.Tensor:
        """

        This method applies the attention mechanism to the input hidden states using
        the provided relative position bias and extended attention mask. It sequentially
        processes the data through multiple layers of the network.

        Parameters
        ----------
        hidden_states : np.ndarray
            The hidden states input tensor of shape (batch_size, seq_length, hidden_size).
        relative_position_bias : np.ndarray
            Tensor containing relative position bias.
        extended_attention_mask : np.ndarray
            Tensor representing the extended attention mask.

        Returns
        -------
        np.ndarray
            The output tensor after applying the attention mechanism, of the same shape as `hidden_states`.
        """
        bsz, tgt_len, embed_dim = hidden_states.size()

        for layer_idx in range(self.num_hidden_layers):
            q_linear = self.q_linears[layer_idx]
            q_linear.weights = self.weights[f'vision_model.encoder.layer.{layer_idx}.self_attn.q_proj.weight']
            q_linear.bias = self.weights[f'vision_model.encoder.layer.{layer_idx}.self_attn.q_proj.bias']

            k_linear = self.k_linears[layer_idx]
            k_linear.weights = self.weights[f'vision_model.encoder.layer.{layer_idx}.self_attn.k_proj.weight']
            k_linear.bias = self.weights[f'vision_model.encoder.layer.{layer_idx}.self_attn.k_proj.bias']

            v_linear = self.v_linears[layer_idx]
            v_linear.weights = self.weights[f'vision_model.encoder.layer.{layer_idx}.self_attn.v_proj.weight']
            v_linear.bias = self.weights[f'vision_model.encoder.layer.{layer_idx}.self_attn.v_proj.bias']

            o_linear = self.o_linears[layer_idx]
            o_linear.weights = self.weights[f'vision_model.encoder.layer.{layer_idx}.self_attn.out_proj.weight']
            o_linear.bias = self.weights[f'vision_model.encoder.layer.{layer_idx}.self_attn.out_proj.bias']

            layer_norm = self.layer_norm[layer_idx]
            layer_norm.gamma = self.weights[f'vision_model.encoder.layer.{layer_idx}.attention.LayerNorm.weight']
            layer_norm.beta = self.weights[f'vision_model.encoder.layer.{layer_idx}.attention.LayerNorm.bias']

            intermediate_dense = self.intermediate_dense[layer_idx]
            intermediate_dense.weights = self.weights[f'vision_model.encoder.layer.{layer_idx}.intermediate.dense.weight']
            intermediate_dense.bias = self.weights[f'vision_model.encoder.layer.{layer_idx}.intermediate.dense.bias']

            output_dense = self.output_dense[layer_idx]
            output_dense.weights = self.weights[f'vision_model.encoder.layer.{layer_idx}.output.dense.weight']
            output_dense.bias = self.weights[f'vision_model.encoder.layer.{layer_idx}.output.dense.bias']

            output_layernorm = self.output_layernorm[layer_idx]
            output_layernorm.gamma = self.weights[f'vision_model.encoder.layer.{layer_idx}.output.LayerNorm.weight']
            output_layernorm.beta = self.weights[f'vision_model.encoder.layer.{layer_idx}.output.LayerNorm.bias']

            q_ = q_linear.forward(hidden_states) * self.scale
            k_ = k_linear.forward(hidden_states)
            v_ = v_linear.forward(hidden_states)

            q = transpose_for_scores(q_)  # (Batch, num_attention_heads, seq_len, attention_head_size)
            k = transpose_for_scores(k_)
            v = transpose_for_scores(v_)

            attention_scores = np.matmul(q, np.transpose(k, (0, 1, -1, -2)))

            attention_scores = attention_scores / np.sqrt(
                self.attention_head_size)  # (Batch, num_attention_heads, seq_len, seq_len )

            if relative_position_bias is not None:
                attention_scores += relative_position_bias

            if extended_attention_mask is not None:
                attention_scores = attention_scores + extended_attention_mask

            attention_probs = Activation.softmax(attention_scores)

            context_layer = np.matmul(attention_probs, v)  # (Batch, num_attention_heads,seq_len,attention_head_size)

            context_layer = np.transpose(context_layer,
                                         (0, 2, 1, 3))  # (Batch, seq_len, num_attention_heads, attention_head_size)
            new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
            context_layer = np.reshape(context_layer, new_context_layer_shape)  # (1, seq_len, emb_dim)
            o = o_linear.forward(context_layer)  # (Batch, seq_len, emb_dim)

            attention_output = layer_norm.forward(o + hidden_states)

            intermediate_output = intermediate_dense.forward(attention_output)  # (Batch, seq_len, intermediate_size)
            intermediate_output = Activation.gelu(intermediate_output, approximate=None)

            layer_output = output_dense.forward(intermediate_output)  # (Batch, seq_len, emb_dim)

            current_layer_output = output_layernorm.forward(
                layer_output + attention_output)

            hidden_states = current_layer_output  # (Batch, seq_len, emb_dim)

        return hidden_states
