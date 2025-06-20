import torch
import torch.nn as nn


# polygama
class SiglipVisionConfig:
    def __init__(self,
                 hidden_size=768,
                 intermediate_size=3072,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 num_channels=3,
                 image_size=224,
                 patch_size=16,
                 layer_norm_eps=1e-6,
                 attention_droput=0.0,
                 num_image_tokens: int = None,
                 **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads,
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_droput
        self.num_image_tokens = num_image_tokens


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.projection = Projection
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # pixel_values : [Batch_size, Channels, Height, Width] -> [Batch_size, Num_Patches, Embed_Dim
        hidden_states = self.embeddings(pixels)
        last_hidden_state = self.encoder(hidden_states)
        return self.layer_norm(last_hidden_state)


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"
        )

        self.num_patches = (self.image_size) // (self.patch_size) ** 2  # img is 2 dim
        self.num_positions - self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "positions_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, weight = pixel_values.shape  # [Batch size, Channel, H, W]
        # num_patches_H = height// patch_size, num_patches_W = weight// patch_size
        patch_embeds = self.patch_embedding(pixel_values)
        # [batch size, embed_dim, num_patches_H, num_patches_W] ->[batch size, embed dim, num_patches]
        # num_patches = num_patches_H * num_patches_W
        embeddings = patch_embeds.flatten(2)
        # [batch size, embed dim, num_patches] ->[batch_size, num_patches, embed_dim]
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings  # [batch_size, num_patches, embed_dim]


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, interm_size]
        hidden_states = self.fc1(hidden_states)
        # [batch_size, num_patches, interm_size]
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')

        return self.fc2(hidden_states)


class SiglipAttentin(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(dself.head_dim)
        self.droupout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, _ = hidden_states.size()
        # [Batch_size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # [Batch_size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        # [ Batch_Size, Num_HeSA, Num_Patches, Num_Patches] Q*K^T/ sqrt(d_k)
        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) * self.scale # how are states related to each other
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if attn_weights.size !=(batch_size, self.num_attention_heads, seq_len, seq_len):
            raise  ValueError("Wrong attention size")
        attn_weights = nn.functional.dropout(attn_weights, p=self.droupout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.o(attn_output)
        return attn_output, attn_weights



class SiglipEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attention(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
