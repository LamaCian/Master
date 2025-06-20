from mpnet import MPNetModel
from modeling_siglip import SiglipVisionTransformer

import torch.nn as nn

mpnet_output_dim = 768
vision_output_dim=768
common_dim = 512
class MultimodalModel(nn.Module):
    def __init__(self, mpnet_model, vision_model, common_dim):
        super(MultimodalModel, self).__init__()
        self.mpnet = mpnet_model
        self.vision_transformer = vision_model
        self.proj_mpnet = nn.Linear(mpnet_output_dim, common_dim)
        self.proj_vision = nn.Linear(vision_output_dim, common_dim)

    def forward(self, text_input, image_input):
        text_emb = self.mpnet(text_input)
        image_emb = self.vision_transformer(image_input)
        text_proj = self.proj_mpnet(text_emb)
        image_proj = self.proj_vision(image_emb)
        text_proj = nn.functional.normalize(text_proj, p=2, dim=1)
        image_proj = nn.functional.normalize(image_proj, p=2, dim=1)
        return text_proj, image_proj

