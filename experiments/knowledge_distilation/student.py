from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class StudentModel(nn.Module):
    def __init__(self, student_model_name: str, projection_dim=512):
        super(StudentModel, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(student_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        self.projection = nn.Linear(self.text_encoder.config.hidden_size, projection_dim)

    def forward(self, text: Optional[str] = None,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None):
        if text:
            res = self.tokenizer(text,
                                 return_tensors="pt",
                                 padding='max_length',
                                 truncation=True,
                                 max_length=512)
            input_ids = res['input_ids']
            attention_mask = res['attention_mask']

        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = outputs.last_hidden_state[:, 0, :]
        proj = self.projection(cls_vec)
        embeddings = nn.functional.normalize(proj, p=2, dim=1)
        return embeddings

    # def mean_pooling(self,
    #                  model_output,
    #                  attention_mask):
    #     token_embeddings = model_output.last_hidden_state  # (batch_size, seq_length, hidden_size)
    #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    #
    #     sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    #
    #     sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    #
    #     return sum_embeddings / sum_mask
