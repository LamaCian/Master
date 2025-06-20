from typing import List, Dict, Any, Union, Optional
import arrow
import logging
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from helpers.helper import get_root_directory
from student import StudentModel
from query_pairs_dataset import QueryTextDataset, QueryTextLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryTextTrainer:
    def __init__(
            self,
            model: StudentModel,
            dataset_path: str,
            model_save_dir: str = None,
            batch_size: int = 32,
            learning_rate: float = 1e-5,
            validation_split: float = 0.2,
            checkpoint_freq: int = 5,
            device: str = None
    ):
        """
        Initialize the trainer for Query-Text pair model.

        Args:
            model: The StudentModel to train
            dataset_path: Path to the training data
            model_save_dir: Directory to save models and checkpoints
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            validation_split: Fraction of data to use for validation
            checkpoint_freq: Save checkpoint every N epochs
            device: Device to use for training (cuda/cpu)
        """
        self.model = model
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.checkpoint_freq = checkpoint_freq

        # Set device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Setup paths
        self.model_save_dir = model_save_dir or os.path.join(
            get_root_directory(), 'models', 'QueryTextPair'
        )
        self.checkpoint_dir = os.path.join(self.model_save_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Initialize dataset and dataloaders
        self._prepare_datasets()

        # Training components
        self.criterion = nn.CosineEmbeddingLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training state
        self.best_val_loss = float('inf')
        self.current_epoch = 0

    def _prepare_datasets(self):
        """Prepare train and validation datasets and dataloaders."""
        full_dataset = QueryTextDataset(
            dataset=self.dataset_path,
            tokenizer=self.model.tokenizer
        )

        # Calculate split lengths
        val_size = int(self.validation_split * len(full_dataset))
        train_size = len(full_dataset) - val_size

        # Split dataset
        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # for reproducibility
        )

        # Create dataloaders
        self.train_dataloader = QueryTextLoader(
            self.train_dataset,
            tokenizer=self.model.tokenizer
        )()
        self.val_dataloader = QueryTextLoader(
            self.val_dataset,
            tokenizer=self.model.tokenizer
        )()

    def _train_epoch(self) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0

        for batch in self.train_dataloader:
            query_inputs, text_inputs = batch['query'], batch['text']

            query_inputs = {k: v.squeeze(1).to(self.device) for k, v in query_inputs.items()}
            text_inputs = {k: v.squeeze(1).to(self.device) for k, v in text_inputs.items()}

            # Get embeddings
            query_embeddings = self.model(**query_inputs).mean(dim=1).unsqueeze(-1)
            text_embeddings = self.model(**text_inputs).mean(dim=1).unsqueeze(-1)

            # Labels: 1 for similar pairs
            labels = torch.ones(query_embeddings.size(0)).to(self.device)

            # Compute loss
            loss = self.criterion(query_embeddings, text_embeddings, labels)
            total_loss += loss.item()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return total_loss / len(self.train_dataloader)

    def _validate(self) -> float:
        """Validate model and return average loss."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_dataloader:
                query_inputs, text_inputs = batch['query'], batch['text']

                query_inputs = {k: v.squeeze(1).to(self.device) for k, v in query_inputs.items()}
                text_inputs = {k: v.squeeze(1).to(self.device) for k, v in text_inputs.items()}

                # Get embeddings
                query_embeddings = self.model(**query_inputs).mean(dim=1).unsqueeze(-1)
                text_embeddings = self.model(**text_inputs).mean(dim=1).unsqueeze(-1)

                # Labels: 1 for similar pairs
                labels = torch.ones(query_embeddings.size(0)).to(self.device)

                # Compute loss
                loss = self.criterion(query_embeddings, text_embeddings, labels)
                total_loss += loss.item()

        return total_loss / len(self.val_dataloader)

    def _save_model(self, epoch: int, is_checkpoint: bool = False):
        """Save model or checkpoint."""
        save_path = self.checkpoint_dir if is_checkpoint else self.model_save_dir

        if is_checkpoint:
            model_name = f"checkpoint_epoch_{epoch}_{arrow.now().format('YYYY-MM-DD-HH-mm')}.pt"
        else:
            model_name = f"query_text_model_epoch_{epoch}_{arrow.now().format('YYYY-MM-DD-HH-mm')}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_val_loss,
        }, os.path.join(save_path, model_name))

        logger.info(f"Model saved at {os.path.join(save_path, model_name)}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model from checkpoint and return the epoch to resume from."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['loss']
            logger.info(f"Loaded checkpoint from epoch {self.current_epoch} with loss {self.best_val_loss}")
            return self.current_epoch + 1  # Resume from next epoch
        return 0

    def train(self, num_epochs: int, resume_from_checkpoint: str = None):
        """
        Train the model for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train
            resume_from_checkpoint: Path to checkpoint to resume training from
        """
        start_epoch = self.load_checkpoint(resume_from_checkpoint) if resume_from_checkpoint else 0

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self._train_epoch()

            # Validate
            val_loss = self._validate()

            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save checkpoint periodically
            if (epoch + 1) % self.checkpoint_freq == 0:
                self._save_model(epoch + 1, is_checkpoint=True)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_model(epoch + 1)

        # Save final model
        self._save_model(num_epochs)


if __name__ == "__main__":
    # Configuration
    dataset_path = f'{get_root_directory()}/data/zara/parsed_examples/all_zara_products_deduped_clened_from_nonsib.csv'

    # Initialize model
    student_model = StudentModel(
        f'{get_root_directory()}/models/clip_multilingual_text',
        projection_dim=512
    )

    # Create trainer
    trainer = QueryTextTrainer(
        model=student_model,
        dataset_path=dataset_path,
        batch_size=32,
        learning_rate=1e-5,
        validation_split=0.2,
        checkpoint_freq=5
    )

    # Train (optionally resume from checkpoint)
    trainer.train(
        num_epochs=2,
        # resume_from_checkpoint="path/to/checkpoint.pt"  # Uncomment to resume
    )