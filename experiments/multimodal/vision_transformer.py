from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import PretrainedConfig
from helpers.helper import load_safe_tensors


def gelu(x: torch.Tensor, approximate: bool = False) -> torch.Tensor:
    """
    GELU activation function. Uses an approximation if `approximate=True`.
    """
    if approximate:
        # Approximation version of GELU using tanh
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    else:
        # Exact version using error function
        return F.gelu(x)


class ConfigVision(PretrainedConfig):
    model_type = "clip_vision_model"

    def __init__(
            self,
            hidden_size=768,
            intermediate_size=3072,
            projection_dim=512,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=32,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            **kwargs,

    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, weights: Optional[Dict] = None, indx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        if weights:
            self.k_proj.weight = nn.Parameter(weights[f'vision_model.encoder.layers.{indx}.self_attn.k_proj.weight'])
            self.k_proj.bias = nn.Parameter(weights[f'vision_model.encoder.layers.{indx}.self_attn.k_proj.bias'])
            self.v_proj.weight = nn.Parameter(weights[f'vision_model.encoder.layers.{indx}.self_attn.v_proj.weight'])
            self.v_proj.bias = nn.Parameter(weights[f'vision_model.encoder.layers.{indx}.self_attn.v_proj.bias'])
            self.q_proj.weight = nn.Parameter(weights[f'vision_model.encoder.layers.{indx}.self_attn.q_proj.weight'])
            self.q_proj.bias = nn.Parameter(weights[f'vision_model.encoder.layers.{indx}.self_attn.q_proj.bias'])
            self.out_proj.weight = nn.Parameter(
                weights[f'vision_model.encoder.layers.{indx}.self_attn.out_proj.weight'])
            self.out_proj.bias = nn.Parameter(weights[f'vision_model.encoder.layers.{indx}.self_attn.out_proj.bias'])

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            causal_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class CLIPMLP(nn.Module):
    def __init__(self, config, weight: Optional[dict] = None, idx: Optional[int] = None):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        if weight:
            self.fc1.weight = nn.Parameter(weight[f'vision_model.encoder.layers.{idx}.mlp.fc1.weight'])
            self.fc1.bias = nn.Parameter(weight[f'vision_model.encoder.layers.{idx}.mlp.fc1.bias'])
            self.fc2.weight = nn.Parameter(weight[f'vision_model.encoder.layers.{idx}.mlp.fc2.weight'])
            self.fc2.bias = nn.Parameter(weight[f'vision_model.encoder.layers.{idx}.mlp.fc2.bias'])

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = gelu(hidden_states, approximate=True)
        return self.fc2(hidden_states)


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config, weights: Optional[Dict], indx: Optional[int]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.weights = weights
        self.self_attn = CLIPAttention(config, self.weights, indx)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm1.weight = nn.Parameter(self.weights[f'vision_model.encoder.layers.{indx}.layer_norm1.weight'])

        self.layer_norm1.bias = nn.Parameter(self.weights[f'vision_model.encoder.layers.{indx}.layer_norm1.bias'])

        self.mlp = CLIPMLP(config, self.weights, indx)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2.weight = nn.Parameter(self.weights[f'vision_model.encoder.layers.{indx}.layer_norm2.weight'])

        self.layer_norm2.bias = nn.Parameter(self.weights[f'vision_model.encoder.layers.{indx}.layer_norm2.bias'])

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            causal_attention_mask: torch.Tensor,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.

        Parameters
        ----------
        causal_attention_mask
        """

        # if torch.isnan(hidden_states).all():
        #     print("all Nan in hidden states")
        # if torch.isnan(hidden_states).any():
        #     print("Nan in hidden states")
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        if torch.isnan(hidden_states).any():
            print("Nan in hidden states after layer norm2")
        if torch.isnan(hidden_states).all():
            print("all Nan in hidden states after layer norm2")
        hidden_states = self.mlp(hidden_states)
        if torch.isnan(hidden_states).any():
            print("Nan in hidden states after mlp")
        if torch.isnan(hidden_states).all():
            print("all Nan in hidden states after mlp")
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: ConfigVision, weights: Optional[dict] = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.weights = weights

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        if weights:
            self.class_embedding = nn.Parameter(self.weights['vision_model.embeddings.class_embedding'])
            self.patch_embedding.weight = nn.Parameter(self.weights['vision_model.embeddings.patch_embedding.weight'])
            self.position_embedding.weight = nn.Parameter(
                self.weights['vision_model.embeddings.position_embedding.weight'])

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)

        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
    """

    def __init__(self, config, weight: Optional[dict] = None):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPEncoderLayer(config, weight, indx=_) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
            self,
            inputs_embeds,
            attention_mask: Optional[torch.Tensor] = None,
            causal_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = None,
    ) -> Tuple:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)


class CLIPVisionTransformerModel(nn.Module):
    def __init__(self, config: ConfigVision = ConfigVision(), weights: Dict = None):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config, weights=weights)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config, weights)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.vision_proj =  nn.Linear(embed_dim, config.projection_dim) # [512, 768]

        if weights:
            self.pre_layrnorm.weight = nn.Parameter(weights['vision_model.pre_layrnorm.weight'])
            self.pre_layrnorm.bias = nn.Parameter(weights['vision_model.pre_layrnorm.bias'])

            self.post_layernorm.weight = nn.Parameter(weights['vision_model.post_layernorm.weight'])
            self.post_layernorm.bias = nn.Parameter(weights['vision_model.post_layernorm.bias'])
            self.vision_proj.weight = nn.Parameter(weights['visual_projection.weight'])

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Tuple:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        if torch.isnan(hidden_states).any():
            print("Nan in hidden states emb")
        if torch.isnan(hidden_states).all():
            print("all Nan in hidden states after emb")
        hidden_states = self.pre_layrnorm(hidden_states)
        if torch.isnan(hidden_states).any():
            print("Nan in hidden states after pre layer norm")
        if torch.isnan(hidden_states).all():
            print("all Nan in hidden states after pre layer norm")

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output =last_hidden_state.mean(dim=1)
        pooled_output = self.post_layernorm(pooled_output)
        # projected_output = pooled_output @ self.vision_proj.t()
        projected_output =  self.vision_proj(pooled_output)

        return (last_hidden_state, projected_output) + encoder_outputs[1:]


if __name__ == "__main__":
    from transformers import CLIPImageProcessor

    img_processor = CLIPImageProcessor(do_rescale=True,
                                       do_resize=True,
                                       do_center_crop=True,
                                       do_convert_rgb=True,
                                       do_normalize=True,
                                       resample=3,
                                       crop_size={"height": 224, "width": 224},
                                       rescale_factor=0.00392156862745098,
                                       size={"shortest_edge": 224},
                                       image_std=[
                                           0.26862954,
                                           0.26130258,
                                           0.27577711
                                       ],
                                       image_mean=[
                                           0.48145466,
                                           0.4578275,
                                           0.40821073
                                       ],

                                       )
    # image_processor = CLIPImageProcessor()

    # img = image_processor.preprocess(
    #     Image.open("/Users/anapaunovic/PycharmProjects/Master_rad/data/images/pantalone.jpg"), do_normalize=True,
    #     do_resize=True, size={"height": 224, "width": 224}, return_tensors='pt', do_rescale=True)
    img = img_processor.preprocess(Image.open("/Users/anapaunovic/PycharmProjects/Master_rad/data/images/valentina2.png"),
                                   return_tensors='pt')
    img2 = img_processor.preprocess(
        images=Image.open("/Users/anapaunovic/PycharmProjects/Master_rad/data/images/valentina_ja.png"),
        return_tensors='pt',
        )
    # img2 = image_processor.preprocess(
    #     Image.open("/Users/anapaunovic/PycharmProjects/Master_rad/data/images/elephant.jpg"), do_normalize=True,
    #     do_resize=True, size={"height": 224, "width": 224}, return_tensors='pt')
    vision_weights = load_safe_tensors(
        '/Users/anapaunovic/PycharmProjects/Master_rad/models/fashion_clip/model.safetensors')
    vision_model = CLIPVisionTransformerModel(weights=vision_weights, config=ConfigVision())
    res = vision_model.forward(img['pixel_values'])
    res2 = vision_model.forward(img2['pixel_values'])
    cosi = torch.nn.CosineSimilarity(dim=-1)
    sim = torch.mean(cosi(res[1], res2[1]))
    print('Sim valetnina and lion', sim)

    print('Sentence transformer')
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    sent_transf = SentenceTransformer('/Users/anapaunovic/PycharmProjects/Master_rad/models/fashion_clip/')
    res_ = sent_transf.encode(Image.open("/Users/anapaunovic/PycharmProjects/Master_rad/data/images/valentina1.png"),
                              convert_to_tensor=True, device=device)

    # # img_model = SentenceTransformer('clip-ViT-B-32')
    res_2 = sent_transf.encode(Image.open("/Users/anapaunovic/PycharmProjects/Master_rad/data/images/valentina_ja.png"),
                               convert_to_tensor=True, device=device)
    #
    if res_.dim() == 1:
        res_ = res_.unsqueeze(0)
    if res_2.dim() == 1:
        res_2 = res_2.unsqueeze(0)

    # Compute cosine similarity
    #
    cs = util.cos_sim(res_2, res_)
    print('Similarity sentence trans',cs)

    # cs = torch.mean(cosi(res_2, res_))
    #
    # print('Similarity:', cos_sent)

    # res = vision_model.get_image_features(img['pixel_values'])
    # res2 = vision_model.get_image_features(img2['pixel_values'])
    # cosi = torch.nn.CosineSimilarity(dim=-1)
    # sim = torch.mean(cosi(res[1], res2[1]))
    # print(sim)
