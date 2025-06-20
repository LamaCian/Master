import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from typing import Optional, Dict, Tuple
import logging
from tqdm import tqdm
import numpy as np
from huggingface_hub import upload_file


class ImageTextTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_dataset: torch.utils.data.Dataset,
            val_dataset: Optional[torch.utils.data.Dataset] = None,
            optimizer: Optional[Optimizer] = None,
            criterion: Optional[_Loss] = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            batch_size: int = 32,
            num_epochs: int = 10,
            checkpoint_dir: str = "models/ImageTextTrainer/checkpoints",
            log_dir: str = "models/ImageTextTrainer/logs",
            collate_fn=None,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device

        self.optimizer = optimizer if optimizer else torch.optim.Adam(model.parameters(), lr=1e-5)
        self.criterion = criterion if criterion else torch.nn.CrossEntropyLoss()

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # Create directories if they don't exist
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize dataloaders
        self.collate_fn = collate_fn
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        ) if val_dataset else None

        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/training.log"),
                logging.StreamHandler()
            ]
        )

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch}")
        for batch in progress_bar:
            # Move batch to device
            images = batch['pixel_values'].to(self.device)
            text_inputs = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device)
            }

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images=images, **text_inputs)

            # Compute loss
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            ground_truth = torch.arange(len(images)).to(self.device)

            loss_i = self.criterion(logits_per_image, ground_truth)
            loss_t = self.criterion(logits_per_text, ground_truth)
            loss = (loss_i + loss_t) / 2

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        self.logger.info(f"Epoch {self.current_epoch} - Train Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self) -> float:
        if not self.val_loader:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {self.current_epoch}")
            for batch in progress_bar:
                # Move batch to device
                images = batch['pixel_values'].to(self.device)
                text_inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }

                # Forward pass
                outputs = self.model(images=images, **text_inputs)

                # Compute loss
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                ground_truth = torch.arange(len(images)).to(self.device)

                loss_i = self.criterion(logits_per_image, ground_truth)
                loss_t = self.criterion(logits_per_text, ground_truth)
                loss = (loss_i + loss_t) / 2

                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({'val_loss': loss.item()})

        avg_loss = total_loss / num_batches
        self.logger.info(f"Epoch {self.current_epoch} - Val Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, is_best: bool = False) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        checkpoint_name = f"checkpoint_epoch_{self.current_epoch}_{timestamp}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1] if hasattr(self, 'train_losses') else 0,
            'val_loss': self.val_losses[-1] if hasattr(self, 'val_losses') else 0,
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(self.model.state_dict(), best_path)
            self.logger.info(f"New best model saved at {best_path}")

        self.logger.info(f"Checkpoint saved at {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}. Resuming from epoch {self.current_epoch}")

    def upload_to_hub(self, checkpoint_path: str, repo_id: str) -> None:
        try:
            upload_file(
                path_or_fileobj=checkpoint_path,
                path_in_repo=os.path.basename(checkpoint_path),
                repo_id=repo_id,
                repo_type="model",
            )
            self.logger.info(f"Successfully uploaded {checkpoint_path} to {repo_id}")
        except Exception as e:
            self.logger.error(f"Failed to upload checkpoint: {e}")

    def train(self) -> Tuple[list, list]:
        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch + 1

            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate() if self.val_loader else 0.0

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Save checkpoint
            self.save_checkpoint()

            # Save best model
            if self.val_loader and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)

            # Log epoch results
            self.logger.info(
                f"Epoch {self.current_epoch}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f}"
            )

        return self.train_losses, self.val_losses


if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from  image_text_dataset import ImageTextDataset, ImageTextCollate

    # Initialize your model, datasets, and components
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Assuming you have train and validation datasets
    train_dataset = ImageTextDataset(
        train_image_paths,
        train_serbian_captions,
        tokenizer,
        image_transform
    )
    val_dataset = ImageTextDataset(
        val_image_paths,
        val_serbian_captions,
        tokenizer,
        image_transform
    )

    # Initialize the trainer
    trainer = ImageTextTrainer(
        model=fashion_clip_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=torch.optim.Adam(fashion_clip_model.parameters(), lr=1e-5),
        criterion=torch.nn.CrossEntropyLoss(),
        batch_size=16,
        num_epochs=10,
        collate_fn=ImageTextCollate()
    )

    # Optionally load from checkpoint
    # trainer.load_checkpoint("models/ImageTextTrainer/checkpoints/checkpoint_epoch_5_2023-01-01-12-00.pt")

    # Start training
    train_losses, val_losses = trainer.train()

    # Optionally upload best model to Hugging Face Hub
    # trainer.upload_to_hub(
    #     "models/ImageTextTrainer/checkpoints/best_model.pt",
    #     "your-username/your-model-repo"
    # )