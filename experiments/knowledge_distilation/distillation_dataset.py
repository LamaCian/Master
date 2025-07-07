import logging
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset

from helpers.helper import get_root_directory, cast_csv_to_dict, load_json
from student import StudentModel


class Distilationcollate:
    def __init__(self,
                 teacher_tokenizer,
                 student_tokenizer,
                 ):
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        # self.max_seq_teacher_length = max_seq_teacher_length
        # self.max_seq_student_length = max_seq_student_length

    def __call__(self, batch):
        english_inputs = [item[0]['input_ids'].squeeze(0) for item in batch]
        serbian_inputs = [item[1]['input_ids'].squeeze(0) for item in batch]

        english_padded = pad_sequence(english_inputs, batch_first=True,
                                      padding_value=self.teacher_tokenizer.pad_token_id)
        serbian_padded = pad_sequence(serbian_inputs, batch_first=True,
                                      padding_value=self.student_tokenizer.pad_token_id)

        english_attention_masks = (english_padded != self.teacher_tokenizer.pad_token_id).long()
        serbian_attention_masks = (serbian_padded != self.student_tokenizer.pad_token_id).long()

        return {'english': {
            'input_ids': english_padded,
            'attention_mask': english_attention_masks,
        },
            'serbian': {
                'input_ids': serbian_padded,
                'attention_mask': serbian_attention_masks,
            }}

#
# class DistillationLoader:
#     def __init__(self, dataset: Union[DistillationDataset, str],
#                  teacher_tokenizer,
#                  student_tokenizer,
#                  is_train: bool,
#                  val_split: float = 0.2,
#                  shuffle: bool = True,
#                  random_state: int = 42,
#                  stratify: bool = True,
#                  batch_size: int = 32):
#         self.teacher_tokenizer = teacher_tokenizer
#         self.student_tokenizer = student_tokenizer
#         self.is_train = is_train
#         self.val_split = val_split
#         self.shuffle = shuffle
#         self.random_state = random_state
#         self.dataset = dataset
#         self.batch_size = batch_size
#
#
#         if isinstance(dataset, str):
#             self.distillation_data = DistillationDataset(teacher_tokenizer=self.teacher_tokenizer,
#                                                          student_tokenizer=self.student_tokenizer,
#                                                          dataset=dataset)
#         else:
#             self.distillation_data = dataset
#
#     def __len__(self):
#         return len(self.distillation_data)
#
#     def __call__(self):
#         return DataLoader(dataset=self.distillation_data,
#                 batch_size=32,
#                 shuffle=True,
#                 num_workers=4,
#                 collate_fn=distilationCollate(teacher_tokenizer=self.teacher_tokenizer,
#                                               student_tokenizer=self.student_tokenizer))
#
#
# if __name__ == '__main__':
#     import json
#
#     teacher_model = SentenceTransformer(f'{get_root_directory()}/models/fashion_clip')
#     teacher_tokenizer = teacher_model._first_module().tokenizer.tokenizer
#     teacher_text_encoder = teacher_model._first_module().model.text_model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     teacher_text_encoder.to(device).eval()
#
#     student_text_model = StudentModel(f'{get_root_directory()}/models/clip_multilingual_text', projection_dim=512)
#     student_text_model.to(device)
#     student_text_model.eval()
#     student_tokenizer = student_text_model.tokenizer
#
#     distillation_data = DistillationDataset(teacher_tokenizer=teacher_tokenizer,
#                                             student_tokenizer=student_tokenizer,
#                                             dataset=f'{get_root_directory()}/data/zara/parsed_examples/all_zara_products_deduped_clened_from_nonsib.csv')
#     distillation_dataloader = DataLoader(dataset=distillation_data,
#                                          batch_size=32,
#                                          shuffle=True,
#                                          num_workers=4,
#                                          collate_fn=distilationCollate(teacher_tokenizer=teacher_tokenizer,
#                                                                        student_tokenizer=student_tokenizer))
#
#     for batch in distillation_dataloader:
#         print(batch)

from typing import List, Dict, Any, Union
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class DistillationDataset(Dataset):
    def __init__(self,
                 dataset: Union[List[Dict[str, Any]], str],
                 teacher_tokenizer,
                 student_tokenizer,
                 max_seq_student_length: int = 512,
                 max_seq_teacher_length: int = 77,
                 category_field: str = 'Category'):

        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.max_seq_student_length = max_seq_student_length
        self.max_seq_teacher_length = max_seq_teacher_length

        # Load data
        if isinstance(dataset, str):
            if dataset.endswith('.csv'):
                self.full_dataset = cast_csv_to_dict(dataset)
            elif dataset.endswith('.json'):
                self.full_dataset = load_json(dataset)
        else:
            self.full_dataset = dataset

        self.indices = list(range(len(self.full_dataset)))

        self.category_field = category_field
        self._process_categories()

    def _process_categories(self):
        """Extract and validate category information"""
        self.categories = []
        self.unique_categories = set()
        self.category_counts = defaultdict(int)

        for item in self.full_dataset:
            # Handle missing categories
            category = item.get(self.category_field, 'uncategorized')
            if not isinstance(category, str):
                category = str(category)

            self.categories.append(category)
            self.unique_categories.add(category)
            self.category_counts[category] += 1

        self.unique_categories = sorted(self.unique_categories)

        min_samples = min(self.category_counts.values())
        if min_samples < 2:
            logging.warning(f"Some categories have very few samples (minimum: {min_samples})")

    def get_category_distribution(self, indices=None):
        """Return category distribution for given indices (or all if None)"""
        if indices is None:
            indices = range(len(self.full_dataset))

        dist = defaultdict(int)
        for idx in indices:
            dist[self.categories[idx]] += 1
        return dict(dist)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        item = self.full_dataset[actual_idx]

        english_text = item['Description English']
        serbian_text = item['Description']

        english_inputs = self.teacher_tokenizer(
            english_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_teacher_length
        )

        serbian_inputs = self.student_tokenizer(
            serbian_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_student_length
        )

        return english_inputs, serbian_inputs

    def get_stratified_split(self, val_size=0.2, random_state=42):
        """Returns stratified train/validation indices"""
        return train_test_split(
            self.indices,
            test_size=val_size,
            random_state=random_state,
            stratify=self.categories
        )


class DistillationLoader:
    def __init__(self,
                 dataset_to_train: Union[DistillationDataset, str],
                 teacher_tokenizer,
                 student_tokenizer,
                 is_train: bool = True,
                 val_split: float = 0.2,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: int = 42,
                 stratify: bool = True):

        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_train = is_train
        self.random_state = random_state

        # Initialize dataset
        if isinstance(dataset_to_train, str):
            self.full_dataset = DistillationDataset(
                dataset=dataset_to_train,
                teacher_tokenizer=teacher_tokenizer,
                student_tokenizer=student_tokenizer
            )
        else:
            self.full_dataset = dataset_to_train

        # Create stratified or random split
        if stratify and hasattr(self.full_dataset, 'categories'):
            train_idx, val_idx = self.full_dataset.get_stratified_split(
                val_size=val_split,
                random_state=random_state
            )
            self.indices = train_idx if is_train else val_idx
        else:
            # Fallback to random split
            val_size = int(len(self.full_dataset) * val_split)
            indices = list(range(len(self.full_dataset)))
            if shuffle:
                np.random.shuffle(indices)
            self.indices = indices[:-val_size] if is_train else indices[-val_size:]

        # Create subset dataset
        self.dataset = DistillationDataset(
            dataset=self.full_dataset.full_dataset,
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer
        )
        self.dataset.indices = self.indices

        # Initialize collate function
        self.collate_fn = Distilationcollate(
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer
        )

    def __len__(self):
        return len(self.dataset)

    def __call__(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle if self.is_train else False,
            num_workers=4,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def get_category_distribution(self):
        """Returns category distribution for the current split"""
        if not hasattr(self.full_dataset, 'categories'):
            return None
        return {cat: sum(1 for i in self.indices if self.full_dataset.categories[i] == cat)
                for cat in self.full_dataset.unique_categories}


class DistillationTrainer:
    def __init__(self, student_model: StudentModel,
                 teacher_model,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler=None,
                 criterion=nn.MSELoss(),
                 patience: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.patience = patience
        self.device = device
        if self.optimizer is None:
            self.optimizer=torch.optim.Adam(student_text_model.parameters(), lr=1e-5)

        # Training state
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'similarity_metrics': []
        }

        # Move models to device
        self.student_model.to(self.device)
        self.teacher_model.to(self.device)

    def train(self, train_loader, val_loader, num_epochs=10, save_path=None):
        for epoch in range(num_epochs):
            # Training phase
            self.student_model.train()
            epoch_train_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                teacher_inputs = {k: v.to(self.device) for k, v in batch['english'].items()}
                student_inputs = {k: v.to(self.device) for k, v in batch['serbian'].items()}

                # Forward pass
                with torch.no_grad():
                    teacher_emb = self.teacher_model(**teacher_inputs)['pooler_output']
                student_emb = self.student_model(
                    input_ids=student_inputs['input_ids'],
                    attention_mask=student_inputs['attention_mask']
                )

                # Normalize and compute loss
                teacher_emb = nn.functional.normalize(teacher_emb, p=2, dim=1)
                student_emb = nn.functional.normalize(student_emb, p=2, dim=1)
                loss = self.criterion(student_emb, teacher_emb)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)

            # Validation phase
            val_loss, val_metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['similarity_metrics'].append(val_metrics)

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)
                self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])

            # Early stopping and model checkpointing
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.epochs_without_improvement = 0
                self.best_model_state = self.student_model.state_dict().copy()
                if save_path:
                    self.save_checkpoint(epoch, save_path)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Print epoch summary
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"CosSim: {val_metrics['avg_cosine_sim']:.4f}")

        # Final evaluation and plots
        self._generate_plots(save_path)
        return self.history

    def validate(self, val_loader):
        self.student_model.eval()
        total_loss = 0
        all_student_embs = []
        all_teacher_embs = []

        with torch.no_grad():
            for batch in val_loader:
                teacher_inputs = {k: v.to(self.device) for k, v in batch['english'].items()}
                student_inputs = {k: v.to(self.device) for k, v in batch['serbian'].items()}

                teacher_emb = self.teacher_model(**teacher_inputs)['pooler_output']
                student_emb = self.student_model(
                    input_ids=student_inputs['input_ids'],
                    attention_mask=student_inputs['attention_mask']
                )

                teacher_emb = nn.functional.normalize(teacher_emb, p=2, dim=1)
                student_emb = nn.functional.normalize(student_emb, p=2, dim=1)

                loss = self.criterion(student_emb, teacher_emb)
                total_loss += loss.item()

                all_student_embs.append(student_emb.cpu().numpy())
                all_teacher_embs.append(teacher_emb.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        student_embs = np.concatenate(all_student_embs)
        teacher_embs = np.concatenate(all_teacher_embs)
        similarities = cosine_similarity(student_embs, teacher_embs)
        diagonal = np.diag(similarities)

        metrics = {
            'avg_cosine_sim': np.mean(diagonal),
            'median_cosine_sim': np.median(diagonal),
            'min_cosine_sim': np.min(diagonal),
            'max_cosine_sim': np.max(diagonal)
        }

        return avg_loss, metrics

    def save_checkpoint(self, epoch, save_path):
        os.makedirs(save_path, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'student_model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'history': self.history
        }
        torch.save(checkpoint, os.path.join(save_path, f'checkpoint_epoch_{epoch}.pt'))

    def _generate_plots(self, save_path):
        plt.figure(figsize=(12, 4))

        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Similarity metrics
        plt.subplot(1, 2, 2)
        similarities = [m['avg_cosine_sim'] for m in self.history['similarity_metrics']]
        plt.plot(similarities, label='Avg Cosine Similarity')
        plt.xlabel('Epoch')
        plt.ylabel('Similarity')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_metrics.png'))
        plt.close()


if __name__ == '__main__':
    student_text_model = StudentModel(f'{get_root_directory()}/models/clip_multilingual_text', projection_dim=512)
    teacher = SentenceTransformer(f'{get_root_directory()}/models/fashion_clip')
    dataset = f'{get_root_directory()}/data/zara/parsed_examples/all_zara_products_deduped_clened_from_nonsib.csv'

    train_loader = DistillationLoader(
        dataset_to_train= dataset,
        teacher_tokenizer=teacher.tokenizer,
        student_tokenizer=student_text_model.tokenizer,
        is_train=True,
        stratify=True
    )

    val_loader = DistillationLoader(
        dataset_to_train=dataset,
        teacher_tokenizer=teacher.tokenizer,
        student_tokenizer=student_text_model.tokenizer,
        is_train=False,
        stratify=True,
        shuffle=False
    )

    # Initialize trainer
    trainer = DistillationTrainer(
        student_model=student_text_model,
        teacher_model=teacher._first_module().model.text_model,

    )

    # Train and validate
    history = trainer.train(
        train_loader=train_loader(),
        val_loader=val_loader(),
        num_epochs=20,
        save_path='model/trained/distilation/checkpoints/'
    )