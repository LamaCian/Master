from glob import iglob
from typing import Optional

from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
import torch
from torchvision import transforms, models
from PIL import Image
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from transformers import AutoModel, AutoTokenizer, BertTokenizer, CLIPModel, CLIPImageProcessor

import torch.nn as nn
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"


# Define a custom dataset class for Flickr30k
class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset = load_dataset("/Users/anapaunovic/PycharmProjects/Master_rad/experiments/huggingface_data")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.cap_per_image = 2

    def __len__(self):
        return self.dataset.num_rows["test"] * self.cap_per_image

    def __getitem__(self, idx):
        original_idx = idx // self.cap_per_image
        # image_path = self.dataset[idx]["image_path"]
        image = self.dataset["test"][original_idx]["image"].convert("RGB")
        image = self.transform(image)

        # You might need to adjust the labels based on your task
        caption = self.dataset["test"][original_idx]["caption"][idx % self.cap_per_image]

        return {"image": image, "caption": caption}


# Create an instance of the custom dataset
flickr30k_custom_dataset = Flickr30kDataset()

from dataclasses import dataclass


@dataclass
class TextConfig:
    """
    Configuration class for the CLIP training script.
    """

    embed_dim: int = 512  # Embedding dimension
    transformer_embed_dim: int = 1024  # Transformer embedding dimension
    max_len: int = 32  # Maximum text length
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


def metrics(similarity: torch.Tensor):
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class VisionEncoder(nn.Module):
    def __init__(self, d_out: int, base: Optional = None) -> None:
        super().__init__()
        if not base:
            base = models.vit_l_32(weights=True)
            d_in = base.fc.in_features
            base.fc = nn.Identity()
        self.base = base
        self.projection = Projection(d_in, d_out)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = x.permute(3, 2, 1, 0)
        projected_vec = self.projection(self.base(x))
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


class TextEncoder(nn.Module):
    def __init__(self, d_out: int, base: Optional = None) -> None:
        super().__init__()
        if not base:
            base = AutoModel.from_pretrained(TextConfig.text_model)
        self.base = base
        self.projection = Projection(TextConfig.transformer_embed_dim, d_out)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base(x)[0]
        out = out[:, 0, :]  # get CLS token output
        projected_vec = self.projection(out)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


class Tokenizer:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, x: str) -> AutoTokenizer:
        return self.tokenizer(
            x, max_length=TextConfig.max_len, truncation=True, padding=True, return_tensors='pt'
        )


class VisionEncoderWithProjection(nn.Module):
    def __init__(self, vit_model, projection_dim=256):
        super(VisionEncoderWithProjection, self).__init__()
        self.vit_model = vit_model
        self.preprocessor = CLIPImageProcessor()
        self.projection = nn.Linear(vit_model.config.hidden_size, projection_dim)

    def forward(self, pixel_values):
        # pixel_values = self.preprocessor.preprocess(images=pixel_values, do_normalize=True, do_rescale=False,
        #                                             return_tensors='pt')['pixel_values']
        processor = self.preprocessor.preprocess(images=pixel_values, do_normalize=True, do_rescale=True,
                                                 return_tensors='pt')
        pixel_values = processor['pixel_values']
        outputs = self.vit_model(pixel_values=pixel_values)
        # Get the [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Apply the projection
        projected = self.projection(cls_output)
        # Normalize the embeddings
        normalized = F.normalize(projected, p=2, dim=1)
        return normalized


class TextEncoderWithProjection(nn.Module):
    def __init__(self, text_model, projection_dim=256):
        super(TextEncoderWithProjection, self).__init__()
        self.text_model = text_model
        self.text_projection = nn.Linear(text_model.config.hidden_size, projection_dim)

    def forward(self, text, attention_mask):
        outputs = self.text_model(text, attention_mask)
        # Get the [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Apply the projection
        projected = self.text_projection(cls_output)
        # Normalize the embeddings
        normalized = F.normalize(projected, p=2, dim=1)
        return normalized


class CustomModels(nn.Module):
    def __init__(self, text_encoder, vision_encoder: VisionEncoder, tokenizer, lr=1e-3):
        super().__init__()
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.tokenizer = tokenizer
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, images, text):
        text = self.tokenizer(text)

        image_embed = self.vision_encoder(images)

        text_embded = self.text_encoder(text['input_ids'], attention_mask=text['attention_mask'])
        if torch.isnan(image_embed).all() or torch.isnan(text_embded).all():
            print("All nans in embeddings")
            return None,None,None

        similarity = text_embded @ image_embed.T

        loss = CLIP_loss(similarity)
        img_acc, cap_acc = metrics(similarity)
        return loss, img_acc, cap_acc


if __name__ == "__main__":
    clip_dataloader = DataLoader(flickr30k_custom_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)

    text_model_name = 'djovak/embedic-large'
    tokenizer = Tokenizer(AutoTokenizer.from_pretrained(text_model_name))
    text_model = TextEncoderWithProjection(text_model=AutoModel.from_pretrained(text_model_name),
                                           projection_dim=Config.embed_dim)



    # vision_model_name = "patrickjohncyh/fashion-clip"
    # clip_fashion = CLIPModel.from_pretrained(vision_model_name)
    # clip_fashion.save_pretrained('/Users/anapaunovic/PycharmProjects/Master_rad/models/fashion_clip')
    vision_encoder = VisionEncoderWithProjection(vit_model=clip_fashion,
                                                 projection_dim=Config.embed_dim)

    multimodel = CustomModels(tokenizer=tokenizer, text_encoder=text_model, vision_encoder=vision_encoder)

    optimizer = torch.optim.Adam([
        {'params': multimodel.vision_encoder.parameters()},
        {'params': multimodel.text_encoder.parameters()}
    ], lr=multimodel.lr)

    start_epoch = 0
    num_epochs = 3

    batch_zero = True
    for epoch in range(start_epoch, num_epochs):
        multimodel.train()
        for batch in clip_dataloader:
            image = batch["image"].to(device)
            text = batch["caption"]
            # images, text = batch
            loss, img_acc, cap_acc = multimodel(image, text)
            if loss and img_acc and cap_acc:

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_zero:
                    print(f"Epoch [{0}/{num_epochs}], Batch Loss: {loss.item()}")
                    batch_zero = False
            continue

        # Print training statistics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch Loss: {loss.item()}")

    print("Training complete.")
