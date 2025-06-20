import torch
from transformers import CLIPTextModel, CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer, AutoTokenizer
import torch.nn as nn
# DistilBERT configuration
distilbert_config = {
    "activation": "gelu",
    "architectures": ["DistilBertModel"],
    "attention_dropout": 0.1,
    "dim": 768,
    "dropout": 0.1,
    "hidden_dim": 3072,
    "initializer_range": 0.02,
    "max_position_embeddings": 512,
    "model_type": "distilbert",
    "n_heads": 12,
    "n_layers": 6,
    "output_past": True,
    "pad_token_id": 0,
    "qa_dropout": 0.1,
    "seq_classif_dropout": 0.2,
    "sinusoidal_pos_embds": False,
    "tie_weights_": True,
    "torch_dtype": "float32",
    "transformers_version": "4.44.2",
    "vocab_size": 119547,
    "bos_token_id": 101,
    "eos_token_id": 102,
}

# Map to CLIPTextConfig
clip_text_config = CLIPTextConfig(
    vocab_size=distilbert_config["vocab_size"],
    hidden_size=distilbert_config["dim"],
    intermediate_size=distilbert_config["hidden_dim"],
    projection_dim=512,  # Default for CLIP
    num_hidden_layers=distilbert_config["n_layers"],
    num_attention_heads=distilbert_config["n_heads"],
    max_position_embeddings=distilbert_config["max_position_embeddings"],
    hidden_act=distilbert_config["activation"],  # Assuming 'gelu' is compatible
    layer_norm_eps=1e-5,  # Default for CLIPTextConfig
    attention_dropout=distilbert_config["attention_dropout"],
    initializer_range=distilbert_config["initializer_range"],
    initializer_factor=1.0,  # Default for CLIPTextConfig
    pad_token_id=distilbert_config["pad_token_id"],
    bos_token_id=distilbert_config['bos_token_id'],  # Default for CLIPTextConfig
    eos_token_id=distilbert_config['eos_token_id']  # Default for CLIPTextConfig
)

# Print the configuration
print(clip_text_config)

# Path to saved weights
saved_weights_path = "/Users/anapaunovic/Desktop/Master/models/distilation/50_epochs_2024-12-03-17-42.pt"

# Load the weights
state_dict = torch.load(saved_weights_path, map_location="cpu")  # Adjust device as needed
new_state_dict = {}
# Initialize a new state dictionary with renamed keys
for key, value in state_dict.items():
    # Map embeddings
    if key.startswith("text_encoder.embeddings.word_embeddings"):
        new_key = key.replace("text_encoder.embeddings.word_embeddings", "text_model.embeddings.token_embedding")
    elif key.startswith("text_encoder.embeddings.position_embeddings"):
        new_key = key.replace("text_encoder.embeddings.position_embeddings", "text_model.embeddings.position_embedding")
    elif key.startswith("text_encoder.embeddings.LayerNorm"):
        new_key = key.replace("text_encoder.embeddings.LayerNorm", "text_model.final_layer_norm")

    # Map transformer layers
    elif key.startswith("text_encoder.transformer.layer"):
        new_key = key.replace("text_encoder.transformer.layer", "text_model.encoder.layers")
        new_key = new_key.replace("attention.q_lin", "self_attn.q_proj")
        new_key = new_key.replace("attention.k_lin", "self_attn.k_proj")
        new_key = new_key.replace("attention.v_lin", "self_attn.v_proj")
        new_key = new_key.replace("attention.out_lin", "self_attn.out_proj")
        new_key = new_key.replace("sa_layer_norm", "layer_norm1")
        new_key = new_key.replace("ffn.lin1", "mlp.fc1")
        new_key = new_key.replace("ffn.lin2", "mlp.fc2")
        new_key = new_key.replace("output_layer_norm", "layer_norm2")

    # Map projection layer
    elif key.startswith("projection"):
        if key.startswith('projection.bias'):
            continue
        new_key = key.replace("projection", "text_projection")

    # Use the key as-is if no mapping is needed
    else:
        new_key = key

    new_state_dict[new_key] = value

tokenizers = AutoTokenizer.from_pretrained("/Users/anapaunovic/Desktop/Master/models/clip_multilingual_text")
clip_text_model = CLIPTextModelWithProjection(clip_text_config)
# missing_keys, unexpected_keys = clip_text_model.load_state_dict(new_state_dict, strict=False)
# print("Missing keys:", missing_keys)
# print("Unexpected keys:", unexpected_keys)
clip_text_model.load_state_dict(new_state_dict, strict=True)
clip_text_model.text_projection.weight = nn.Parameter(new_state_dict['text_projection.weight'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_text_model.to(device)
clip_text_model.eval()
input_tokenized = tokenizers('haljina sa leopard printom',
    return_tensors='pt',
    padding='max_length',
    truncation=True,
    max_length=128)
res = clip_text_model(input_tokenized['input_ids'])

input_tokenized_eng = tokenizers('Leopard print dress',
    return_tensors='pt',
    padding='max_length',
    truncation=True,
    max_length=128)
res_eng = clip_text_model(input_tokenized['input_ids'])


input_tokenized_diff = tokenizers('Pejzazi su krajoliki i veoma lepi na brdima Beograda',
    return_tensors='pt',
    padding='max_length',
    truncation=True,
    max_length=512)
res_diff = clip_text_model(input_tokenized['input_ids'])


from sentence_transformers import util

cos_sim = util.cos_sim(res['text_embeds'], res_diff['text_embeds'])
print(cos_sim)

# Model is now ready for use
print("CLIPTextModel loaded successfully!")
