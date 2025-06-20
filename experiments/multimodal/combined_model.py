import logging
from dataclasses import dataclass
from typing import List, Tuple
from helpers.helper import get_root_directory
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoTokenizer
from experiments.dataset_skeleton import TripletDataset, Collate
from experiments.multimodal.projection_layer import Projection
from experiments.loss import TripletLoss
from experiments.multimodal.vision_transformer import CLIPVisionTransformerModel
from helpers.helper import load_safe_tensors
from sentence_transformers import SentenceTransformer, util
import numpy as np

logging.basicConfig(level=logging.INFO)
import arrow
from typing import Optional


@dataclass
class TextConfig:
    """
    Configuration class for the CLIP training script.
    """

    embed_dim: int = 512  # Embedding dimension
    transformer_embed_dim: int = 1024  # Transformer embedding dimension
    max_seq_len: int = 32  # Maximum text length
    text_model: str = "djovak/embedic-large"  # Text model name
    epochs: int = 5  # Number of training epochs
    batch_size: int = 128  # Batch size


def CLIP_loss(logits: torch.Tensor) -> torch.Tensor:
    # Assuming n is the number of classes
    n = logits.shape[1]

    # Create labels tensor
    labels = torch.arange(n)

    # Calculate cross entropy losses along axis 0 and 1
    loss_i = F.cross_entropy(logits.transpose(0, 1), labels, reduction="mean")
    loss_t = F.cross_entropy(logits, labels, reduction="mean")

    # Calculate the final loss
    loss = (loss_i + loss_t) / 2

    return loss


def collate_fn(batch):
    pixel_values = torch.stack([sample['anchor_imag'] for sample in batch])

    positives = [sample['positive_description'] for sample in batch]
    negatives = [sample['negative_description'] for sample in batch]

    positive_example = pad_sequence(positives, batch_first=True, padding_value=0)
    negative_example = pad_sequence(negatives, batch_first=True, padding_value=0)

    batch = {
        "image": pixel_values,
        "positive": positive_example,
        "negative": negative_example
    }

    return batch


def metrics(similarity: torch.Tensor):
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc


class CustomModel(nn.Module):
    def __init__(self, text_encoder, vision_encoder, tokenizer, embed_dim=512):
        super().__init__()
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.tokenizer = tokenizer

        # Projection layers
        self.text_proj = Projection(
            in_features=text_encoder.config.hidden_size, out_features=embed_dim)

        self.vision_proj = self.vision_encoder.vision_proj

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        for param in self.vision_proj.parameters():
            param.requires_grad = True

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        for param in self.text_proj.parameters():
            param.requires_grad = True

    def forward(self, images, positive_input_ids: Optional = None, positive_attention_mask: Optional = None,
                negative_input_ids: Optional = None, negative_attention_mask: Optional = None):
        # Encode images
        image_embeds = self.get_image_features(images)
        logging.info('Encoding positive prompt')

        # Encode positive texts
        s = arrow.now()
        positive_outputs = self.text_encoder(
            input_ids=positive_input_ids, attention_mask=positive_attention_mask)
        logging.info(f'Encoded input in {arrow.now() - s}')

        positive_embeds = positive_outputs.last_hidden_state[:, 0, :]  # CLS token
        positive_embeds = self.text_proj(positive_embeds)
        logging.info('Encoding negative prompt')
        # Encode negative texts
        negative_outputs = self.text_encoder(
            input_ids=negative_input_ids, attention_mask=negative_attention_mask)
        negative_embeds = negative_outputs.last_hidden_state[:, 0, :]
        negative_embeds = self.text_proj(negative_embeds)
        logging.info(f'Text proj finished in {arrow.now() - s}')

        # Normalize embeddings
        logging.info('Normalizing embeddings')
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        positive_embeds = F.normalize(positive_embeds, p=2, dim=-1)
        negative_embeds = F.normalize(negative_embeds, p=2, dim=-1)

        return image_embeds, positive_embeds, negative_embeds

    def get_image_features(self, images, include_projection: bool = True):
        logging.info('Encoding vision input')
        s = arrow.now()
        vision_outputs = self.vision_encoder(images)[1]

        logging.info(f'Encoded text input in {arrow.now() - s}')
        if include_projection:
            vision_outputs = self.vision_proj(vision_outputs)
        logging.info(f'Encoded vision input in {arrow.now() - s}')

        return vision_outputs

    def get_similarity(self, image, text):

        text_embeddings = self.get_text_features(text)
        image_embeddings = self.get_image_features(image)

    def zero_shot_classification(self, image, text_labels: List[str], debug: bool = True):
        """
        Perform zero-shot image classification
        :return:
        """
        # encode text
        text_vectors = torch.stack([self.get_text_features(label) for label in text_labels])
        image_vector = self.get_image_features(image).unsqueeze(0)

        cosine_sim = F.cosine_similarity(image_vector, text_vectors, dim=-1)
        similarity_dict = {label: sim.item() for label, sim in zip(text_labels, cosine_sim)}
        if debug:
            for label, sim in similarity_dict.items():
                print(f"Label: {label}, Similarity: {sim:.4f}")

        predicted_label = max(similarity_dict, key=similarity_dict.get)
        return predicted_label, similarity_dict

    def get_text_features(self, text, include_projection: bool = True, batch_size: int = 1):
        logging.info('Encoding text input')
        s = arrow.now()
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        text_outputs = self.text_encoder(input_ids=input_ids,
                                         attention_mask=attention_mask)
        text_outputs = text_outputs.last_hidden_state[:, 0, :]
        logging.info(f'Encoded text input in {arrow.now() - s}')
        if include_projection:
            projected_output = self.text_proj(text_outputs)
            return projected_output
        return text_outputs


if __name__ == "__main__":
    import os

    dataset = TripletDataset(f"{get_root_directory()}/triplets_fashion.json")
    dataset_subset = Subset(dataset, indices=[i for i in range(0, 2000)])
    text_model_name = 'djovak/embedic-base'
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    vision_weights = load_safe_tensors(
        f'{get_root_directory()}/models/fashion_clip/model.safetensors')
    vision_model = CLIPVisionTransformerModel(weights=vision_weights)
    text_model = AutoModel.from_pretrained(text_model_name)
    # Initialize the custom model
    model = CustomModel(
        text_encoder=text_model,
        vision_encoder=vision_model,
        tokenizer=tokenizer,
        embed_dim=512
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dataloader = DataLoader(dataset_subset, batch_size=8, shuffle=True, num_workers=1, collate_fn=Collate(tokenizer))
    criterion = TripletLoss(margin=1.0)
    num_epochs = 100
    # Initialize optimizer with only the parameters that require gradients
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_parameters, lr=1e-3)

    # Training loop remains largely the same
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            # Move inputs to device
            images = batch['image'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)

            optimizer.zero_grad()

            # Forward pass
            image_embeds, positive_embeds, negative_embeds = model(
                images,
                positive_input_ids,
                positive_attention_mask,
                negative_input_ids,
                negative_attention_mask
            )

            loss = criterion(anchor=image_embeds, positive=positive_embeds, negative=negative_embeds)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    save_dir = f"{get_root_directory()}/models/trained"

    os.makedirs(save_dir, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(),
               f"{save_dir}/trained_{num_epochs}_epochs_{arrow.now().format('YYYY-MM-DD-HH-mm')}.pt")
