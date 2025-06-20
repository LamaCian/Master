import os
from typing import Optional
from torch.utils.data import DataLoader, random_split, Subset
from student import StudentModel
from helpers.helper import get_root_directory, load_safe_tensors
import torch
import torch.nn as nn
import torch.nn.functional as F
from image_text_dataset import ImageTextCollate, ImageTextDataset
import logging
import  arrow
from experiments.multimodal.vision_transformer import \
    CLIPVisionTransformerModel

logging.basicConfig(level=logging.INFO)


def clip_loss(similarity):
    labels = torch.arange(similarity.size(0)).to(similarity.device)
    loss_i = F.cross_entropy(similarity, labels)
    loss_t = F.cross_entropy(similarity.T, labels)
    return (loss_i + loss_t) / 2


def compute_retrieval_accuracy(similarity_matrix: torch.Tensor) -> float:
    """
    Compute a simple retrieval accuracy:
    ( image->text correct + text->image correct ) / (2*batch_size).
    """
    image_to_text_pred = similarity_matrix.argmax(dim=1)
    correct_i2t = (image_to_text_pred == torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)).sum()

    text_to_image_pred = similarity_matrix.argmax(dim=0)
    correct_t2i = (text_to_image_pred == torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)).sum()

    accuracy = (correct_i2t + correct_t2i) / (2 * similarity_matrix.size(0))
    return accuracy.item()


class FashionClipich(nn.Module):
    def __init__(self,
                 text_model,
                 vision_model,
                 text_model_weights_path: Optional[str] = None,
                 vision_model_weights_path: Optional[str] = None, ):
        super(FashionClipich, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_model = text_model
        self.vision_model = vision_model
        if text_model_weights_path:
            self.text_model.load_state_dict(torch.load(text_model_weights_path))
        if vision_model_weights_path:
            self.vision_model.load_state_dict(torch.load(vision_model_weights_path))

    def get_text_features(self, text):
        return self.text_model(**text)

    def get_vision_features(self, image):
        return self.vision_model(image)

    def forward(self, text, image):
        text_embeds = self.get_text_features(text)
        image_embeds = self.get_vision_features(image)
        return text_embeds, image_embeds

    def validate(self, val_dataloader: DataLoader):
        self.text_model.eval()
        self.vision_model.eval()

        count_batches = 0
        total_val_accuracy = 0
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_dataloader:
                if 'pixel_values' not in batch:
                    logging.info('Batch {} has no pixel_values'.format(count_batches))
                    continue

                images = batch['pixel_values'].to(self.device)
                text_inputs = {k: v.to(self.device) for k, v in batch.items() if
                               k in ['input_ids', 'attention_mask']}

                image_embeds = self.get_vision_features(images)[1]
                text_embeds = self.get_text_features(text_inputs)

                image_embeds = F.normalize(image_embeds, dim=1)
                text_embeds = F.normalize(text_embeds, dim=1)

                similarity = image_embeds @ text_embeds.T

                loss = clip_loss(similarity)

                total_val_loss += loss.item()

            batch_accuracy = compute_retrieval_accuracy(similarity)
            total_val_accuracy += batch_accuracy

            count_batches += 1

        avg_val_loss = total_val_loss / count_batches
        avg_val_accuracy = total_val_accuracy / count_batches

        return avg_val_loss, avg_val_accuracy



    def train_n_epochs(self,
                       num_epochs: int,
                       train_dataset: Subset[ImageTextDataset],
                       val_dataset: Subset[ImageTextDataset],
                       model_save_path: str,
                       checkpoint_path: Optional[str] = None):

        self.text_model.to(self.device)
        self.vision_model.to(self.device)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=ImageTextCollate())
        val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=ImageTextCollate())
        optimizer = torch.optim.AdamW(list(self.text_model.parameters()),
                                      lr=1e-4)
        start_epoch = 0
        best_val_loss = float('inf')

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.text_model.load_state_dict(checkpoint["text_model_state_dict"])
            self.vision_model.load_state_dict(checkpoint["vision_model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            logging.info(f"Resuming training from epoch {start_epoch + 1}")

        for epoch in range(start_epoch, start_epoch + num_epochs):

            self.text_model.train()
            self.vision_model.train()

            total_train_loss = 0
            for batch in train_dataloader:
                if not 'pixel_values' in batch:
                    logging.info(f"Skipping batch {batch}")
                    continue
                images = batch['pixel_values'].to(self.device)
                text_inputs = {k: v.to(self.device) for k, v in batch.items() if
                               k in ['input_ids', 'attention_mask']}

                image_embeds = self.get_vision_features(images)[1]
                text_embeds = self.get_text_features(text_inputs)

                image_embeds = F.normalize(image_embeds, dim=1)
                text_embeds = F.normalize(text_embeds, dim=1)

                similarity = image_embeds @ text_embeds.T

                loss = clip_loss(similarity)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(dataset)
            val_loss, val_accuracy = self.validate(val_dataloader)

            logging.info(
                f"Epoch [{epoch + 1}/{start_epoch + num_epochs}] "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}"
            )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

                self.save_model(
                    save_path=model_save_path,
                    epoch=epoch,
                    optimizer=optimizer,
                    best_val_loss=best_val_loss,
                    is_best=is_best
                )

    def save_model(
            self,
            save_path: str,
            epoch: int,
            optimizer: torch.optim.Optimizer,
            best_val_loss: float,
            is_best: bool = False
    ):
        os.makedirs(save_path, exist_ok=True)
        timestamp = arrow.now().format("YYYY-MM-DD-HH-mm")
        checkpoint = {
            "text_model_state_dict": self.text_model.state_dict(),
            "vision_model_state_dict": self.vision_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        }

        epoch_ckpt_path = os.path.join(save_path, f"checkpoint_epoch_{epoch + 1}_{timestamp}.pt")
        torch.save(checkpoint, epoch_ckpt_path)
        logging.info(f"Checkpoint saved at {epoch_ckpt_path}")

        if is_best:
            best_ckpt_path = os.path.join(save_path, "best_model.pt")
            torch.save(checkpoint, best_ckpt_path)
            logging.info("Best model updated.")


if __name__ == '__main__':
    #     torch.load(f'{get_root_directory()}/models/trained/query_text_model_epoch_20_2024-12-11-20-33.pt'))

    student_text_model = StudentModel(f'{get_root_directory()}/models/clip_multilingual_text', projection_dim=512)
    vision_weights = load_safe_tensors(
        '/Users/anapaunovic/Desktop/Master/models/fashion_clip/model.safetensors')
    vision_model = CLIPVisionTransformerModel(weights=vision_weights)

    dataset_path = f'{get_root_directory()}/Parsing/zara/clothes_combined.csv'
    dataset = ImageTextDataset(dataset=dataset_path,
                               tokenizer=student_text_model.tokenizer,
                               image_col='Local Images',
                               caption_col=['Description', 'Product Name'])

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset,[train_size, val_size])

    model = FashionClipich(
        text_model=student_text_model,
        vision_model=vision_model,
        text_model_weights_path=f'{get_root_directory()}/models/trained/query_text_model_epoch_20_2024-12-11-20-33.pt',

    )

    model.train_n_epochs(num_epochs=20,
                         train_dataset=train_dataset,
                         val_dataset=val_dataset,
                         model_save_path=f'{get_root_directory()}/models/trained/image_text_model',
                         checkpoint_path=f'{get_root_directory()}/models/trained/image_text_model/checkpoint_epoch_100_2024-12-16-13-22.pt')

    # checkpoint_path = f'{get_root_directory()}/models/trained/image_text_model/checkpoint_epoch_8_2024-12-16-01-31.pt'
