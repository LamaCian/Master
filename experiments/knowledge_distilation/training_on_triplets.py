import torch
from torch.utils.data import DataLoader
from torch import nn
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)


class TripletTrainer:
    def __init__(self,
                 student_text_model: nn.Module,
                 vision_model: nn.Module,
                 tokenizer,
                 optimizer: torch.optim.Optimizer,
                 device: Optional[str] = None,
                 margin: float = 0.2):
        """
        Args:
            student_text_model: The text model (student) to be fine-tuned.
            vision_model: The vision model (e.g., CLIPVisionModel) for encoding images.
            tokenizer: The tokenizer corresponding to the student text model.
            optimizer: The optimizer to update parameters of the student model (and vision model if training).
            device: 'cuda' or 'cpu'.
            margin: Margin for the triplet loss.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.student_text_model = student_text_model.to(self.device)
        self.vision_model = vision_model.to(self.device)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.criterion = nn.TripletMarginLoss(margin=margin)
        self.patience = 5
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0

    def train_epoch(self, dataloader: DataLoader, num_epochs: int, save_path: str):
        self.student_text_model.train()
        self.vision_model.eval()  # often, we keep the vision model fixed
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                images = batch['image'].to(self.device)  # (B, 3, 224, 224)
                positive_input_ids = batch['positive_input_ids'].to(self.device)
                positive_attention_mask = batch['positive_attention_mask'].to(self.device)
                negative_input_ids = batch['negative_input_ids'].to(self.device)
                negative_attention_mask = batch['negative_attention_mask'].to(self.device)

                with torch.no_grad():
                    image_outputs = self.vision_model(pixel_values=images)[1]

                    image_embeds = nn.functional.normalize(image_outputs, p=2, dim=1)

                logging.info('Computed image embeddings')

                positive_embeds = self.student_text_model(
                    input_ids=positive_input_ids,
                    attention_mask=positive_attention_mask
                )

                negative_embeds = self.student_text_model(
                    input_ids=negative_input_ids,
                    attention_mask=negative_attention_mask
                )
                logging.info('Computed text embeddings')

                loss = self.criterion(image_embeds, positive_embeds, negative_embeds)
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                logging.info('Computed loss: {}, total loss: {}'.format(loss, epoch_loss))

            avg_epoch_loss = epoch_loss / len(dataloader)
            logging.info(f"Epoch {epoch + 1}, Average Loss: {avg_epoch_loss:.4f}")
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self.epochs_without_improvement = 0

            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs. Best loss: {self.best_loss:.4f}")
                break
        self.save_model(save_path=save_path, epoch=num_epochs)

    def save_model(self, save_path: str, epoch: int):
        import os
        import arrow
        os.makedirs(save_path, exist_ok=True)
        model_name = f"triplet_model_epoch_{epoch}_{arrow.now().format('YYYY-MM-DD-HH-mm')}.pt"
        torch.save(self.student_text_model.state_dict(), os.path.join(save_path, model_name))
        logging.info(f"Model saved at {os.path.join(save_path, model_name)}")


if __name__ == '__main__':
    from experiments.dataset_skeleton import TripletDataset, Collate
    from student import StudentModel
    from helpers.helper import get_root_directory, load_safe_tensors
    from experiments.multimodal.vision_transformer import CLIPVisionTransformerModel, ConfigVision

    text_model = StudentModel(f'{get_root_directory()}/models/clip_multilingual_text',
                              projection_dim=512)

    text_model.load_state_dict(torch.load(f'{get_root_directory()}/models/trained/query_text_model_epoch_20_2024-12-11-20-33.pt'))
    dataset = TripletDataset(dataset='/Users/anapaunovic/Desktop/Master/Parsing/triplet_dataset_with_less_examples.csv',
                             tokenizer=text_model.tokenizer)
    collate_fn = Collate(tokenizer=text_model.tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    vision_weights = load_safe_tensors(
        '/Users/anapaunovic/Desktop/Master/models/fashion_clip/model.safetensors')
    vision_model = CLIPVisionTransformerModel(weights=vision_weights, config=ConfigVision())
    optimizer = torch.optim.Adam(text_model.parameters(), lr=1e-5)
    num_epochs = 1
    trainer = TripletTrainer(
        student_text_model=text_model,
        vision_model=vision_model,
        tokenizer=text_model.tokenizer,
        optimizer=optimizer
    )
    trainer.train_epoch(dataloader, num_epochs=num_epochs, save_path=f'{get_root_directory()}/models/trained')

    # for epoch in range(num_epochs):
    #     loss = trainer.train_epoch(dataloader)
    #     print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
