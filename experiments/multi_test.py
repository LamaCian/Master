import json
import logging
import os
from typing import Dict, Literal, List, Union
import torch
import numpy as np
from safetensors import safe_open
from scipy.special import erf
from PIL import Image

from experiments.sent_transf_clip import img_emb2, similarity_scores

logging.basicConfig(level=logging.INFO)


def load_safe_tensors(safetensors_path: str, mode: Literal['sentence_transformer', 'numpy'] = 'sentence_transformer') -> \
        Dict[str, np.ndarray]:
    tensors = {}
    f = safe_open(safetensors_path, framework='pt')
    for key in f.keys():
        if mode == 'numpy':
            tensors[key] = np.array(f.get_tensor(key))
        else:
            if key.startswith('text_model') or key.startswith('text_projection'):
                continue
            tensors[key] = f.get_tensor(key)
    return tensors


if __name__ == "__main__":
    vision_tensors = load_safe_tensors(
        safetensors_path='/Users/anapaunovic/PycharmProjects/Master_rad/models/fashion_clip/model.safetensors')
    embedic_tensors = load_safe_tensors(
        safetensors_path='/Users/anapaunovic/PycharmProjects/Master_rad/models/djovak/model.safetensors')
    print(embedic_tensors)

    from transformers import CLIPModel, CLIPVisionModel, CLIPImageProcessor
    import torch.nn.functional as F

    model_name = "/Users/anapaunovic/PycharmProjects/Master_rad/models/fashion_clip"
    image_processor = CLIPImageProcessor()
    # Load the Fashion-CLIP model
    # model = CLIPVisionModel.from_pretrained(model_name)
    # vision_encoder = model.vision_model
    # vit_weights = vision_encoder.state_dict()
    # img = image_processor.preprocess(
    #     Image.open("/Users/anapaunovic/PycharmProjects/Master_rad/data/images/valentina1.png"), do_normalize=True,
    #     do_resize=True, size={"height": 224, "width": 224}, return_tensors='pt')
    # res0 = model.forward(img['pixel_values'])
    # clip_model = CLIPModel.from_pretrained(model_name)
    # res = clip_model.get_image_features(img['pixel_values'])
    # img2 =  image_processor.preprocess(
    #     Image.open("/Users/anapaunovic/PycharmProjects/Master_rad/data/images/valentina2.png"), do_normalize=True,
    #     do_resize=True, size={"height": 224, "width": 224}, return_tensors='pt')
    # res1 = clip_model.forward(img2['pixel_values'])
    # similarity_scores = F.cosine_similarity(res1['last_hidden_state'], res['last_hidden_state'])
    # cosi = torch.nn.CosineSimilarity(dim=-1)
    # output = torch.mean(cosi(res1['last_hidden_state'], res['last_hidden_state']))

    # print(res)
