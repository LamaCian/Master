import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from student import StudentModel
import arrow
import os
from distillation_dataset import DistillationLoader, DistillationDataset, Distilationcollate
from helpers.helper import get_root_directory
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)

# loss = 1 - F.cosine_similarity(student_embs, teacher_embs, dim=-1).mean()

class DistillationTrainer:
    def __init__(self, student_model: StudentModel,
                 teacher_model,
                 optimizer: torch.optim.Optimizer,
                 scheduler=None,
                 criterion=nn.MSELoss(),
                 patience: int = 5):

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.criterion = criterion
        self.patience = patience
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0

        self.student_model.to(self.device)
        self.teacher_model.to(self.device)

        self.train_loss_history = []
        self.val_loss_history = []
        self.best_model_state = None

    def forward(self,
                teacher_inputs: dict,
                student_inputs: dict):

        if not isinstance(teacher_inputs, dict):
            raise ValueError("teacher_inputs must be a dictionary.")
        if not all(k in teacher_inputs for k in ["input_ids", "attention_mask"]):
            raise ValueError("teacher_inputs must contain 'input_ids' and 'attention_mask'.")

        if not isinstance(student_inputs, dict):
            raise ValueError("student_inputs must be a dictionary.")
        if not all(k in student_inputs for k in ["input_ids", "attention_mask"]):
            raise ValueError("student_inputs must contain 'input_ids' and 'attention_mask'.")

        teacher_inputs = {k: v.to(self.device) for k, v in teacher_inputs.items()}
        student_inputs = {k: v.to(self.device) for k, v in student_inputs.items()}

        for p in self.teacher_model.parameters():
            p.requires_grad = False

        with torch.no_grad():
            teacher_embeddings = self.teacher_model(**teacher_inputs)['pooler_output']

        student_embeddings = self.student_model(
            input_ids=student_inputs['input_ids'],
            attention_mask=student_inputs['attention_mask']
        )

        teacher_embeddings = nn.functional.normalize(teacher_embeddings, p=2, dim=1)
        student_embeddings = nn.functional.normalize(student_embeddings, p=2, dim=1)

        return student_embeddings, teacher_embeddings

    def validate(self,
                 val_dataloader: DataLoader[DistillationLoader]):

        self.student_model.eval()

        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                teacher_inputs, student_inputs = batch['english'], batch['serbian']
                student_outputs, teacher_outputs = self.forward(teacher_inputs, student_inputs)
                loss = self.criterion(student_outputs, teacher_outputs)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_dataloader)

        return avg_loss

    def train(self,
              train_loader: DistillationLoader,
              val_loader: DistillationLoader,
              num_epochs: int = 10,
              save_model: bool = True,
              save_path: str = None,
              log_interval: int = 10,
              checkpoint_path: str = None):

        if not hasattr(train_loader, '__iter__'):
            raise ValueError("train_loader must be an iterable.")
        if not hasattr(val_loader, '__iter__'):
            raise ValueError("val_loader must be an iterable.")

        start_epoch = 0
        if checkpoint_path:
            start_epoch = self.load_model(checkpoint_path)

        self.student_model.train()

        for epoch in range(start_epoch, start_epoch + num_epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                if not isinstance(batch, dict) or 'english' not in batch or 'serbian' not in batch:
                    raise ValueError("Batch must be a dictionary with keys 'english' and 'serbian'.")
                teacher_inputs, student_inputs = batch['english'], batch['serbian']
                student_embeddings, teacher_embeddings = self.forward(student_inputs=student_inputs,
                                                                      teacher_inputs=teacher_inputs)
                loss = self.criterion(student_embeddings, teacher_embeddings)
                epoch_loss += loss.item()
                loss.backward()
                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.step()
                self.optimizer.zero_grad()

                if batch_idx % log_interval == 0:
                    logging.info(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / len(train_loader)
            logging.info(f"Epoch {epoch + 1}, Average Training Loss: {avg_epoch_loss:.4f}")

            val_loss = self.validate(val_loader)
            logging.info(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.epochs_without_improvement = 0
                self.best_model_state = self.student_model.state_dict().copy()
                if save_model and save_path:
                    self.save_model(save_path, epoch + 1)
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                logging.info(
                    f"Early stopping triggered after {epoch + 1} epochs. Best validation loss: {self.best_loss:.4f}")
                break

            self.train_loss_history.append(avg_epoch_loss)
            self.val_loss_history.append(val_loss)

        if save_model and save_path:
            self.save_model(save_path, num_epochs)
            self._plot_learning_curves(save_path)
            self._final_evaluation(val_loader, save_path)

    def load_model(self,
                   ckpoint_path: str):

        if not os.path.exists(ckpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpoint_path}")

        checkpoint = torch.load(ckpoint_path)
        self.student_model.load_state_dict(checkpoint['student_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_loss = checkpoint.get('best_loss', float('inf'))
        start_epoch = checkpoint.get('epoch', 0) + 1
        logging.info(f"Resuming training from epoch {start_epoch}")

        return start_epoch

    def save_model(self,
                   save_path: str,
                   epoch: int):
        os.makedirs(save_path, exist_ok=True)
        model_name = f"student_model_epoch_{epoch}_{arrow.now().format('YYYY-MM-DD-HH-mm')}.pt"
        checkpoint = {
            'student_model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch,
            'best_loss': self.best_loss,
        }
        torch.save(checkpoint, os.path.join(save_path, model_name))
        logging.info(f"Model saved at {os.path.join(save_path, model_name)}")

    def _plot_learning_curves(self, save_path: str):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss_history, label='Training Loss')
        plt.plot(self.val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Learning Curves')
        plt.legend()

        plot_path = os.path.join(save_path, 'learning_curves.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved learning curves to {plot_path}")

    def _final_evaluation(self, val_loader: DataLoader, save_path: str):
        if self.best_model_state:
            self.student_model.load_state_dict(self.best_model_state)

        self.student_model.eval()
        all_student_embs = []
        all_teacher_embs = []

        with torch.no_grad():
            for batch in val_loader:
                teacher_inputs, student_inputs = batch['english'], batch['serbian']
                student_emb, teacher_emb = self.forward(teacher_inputs, student_inputs)
                all_student_embs.append(student_emb.cpu().numpy())
                all_teacher_embs.append(teacher_emb.cpu().numpy())

        student_embs = np.concatenate(all_student_embs)
        teacher_embs = np.concatenate(all_teacher_embs)

        similarities = cosine_similarity(student_embs, teacher_embs)
        diagonal = np.diag(similarities)

        metrics = {
            'avg_cosine_sim': np.mean(diagonal),
            'median_cosine_sim': np.median(diagonal),
            'min_cosine_sim': np.min(diagonal),
            'max_cosine_sim': np.max(diagonal),
            'final_train_loss': self.train_loss_history[-1],
            'final_val_loss': self.val_loss_history[-1],
            'best_val_loss': self.best_loss
        }

        logging.info("\n===== Final Evaluation Metrics =====")
        for k, v in metrics.items():
            logging.info(f"{k}: {v:.4f}")

        metrics_path = os.path.join(save_path, 'final_metrics.txt')
        with open(metrics_path, 'w') as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")

        return metrics


if __name__ == '__main__':
    student_text_model = StudentModel(f'{get_root_directory()}/models/clip_multilingual_text', projection_dim=512)
    teacher = AutoModel.from_pretrained(f'{get_root_directory()}/models/fashion_clip')
    teacher_tokenizer = AutoTokenizer.from_pretrained(f'{get_root_directory()}/models/fashion_clip')
    model_save_path = f'{get_root_directory()}/models/trained/distillation/'
    optim = torch.optim.Adam(
        list(student_text_model.parameters()),
        lr=1e-5
    )
    checkpoint_path = f'{get_root_directory()}/models/trained/student_model_epoch_9_2024-12-09-18-15.pt'

    # noinspection PyProtectedMember
    trainer = DistillationTrainer(student_model=student_text_model,
                                  teacher_model=teacher.text_model,
                                  optimizer=optim,
                                  )

    datasets = DistillationDataset(
        dataset=f'{get_root_directory()}/data/zara/parsed_examples/all_zara_products_deduped_clened_from_nonsib.csv',
        student_tokenizer=student_text_model.tokenizer,
        teacher_tokenizer=teacher_tokenizer)

    train_size = int(0.8 * len(datasets))
    train_set, val_set = torch.utils.data.random_split(datasets, [train_size, len(datasets) - train_size])
    # train_load= DistillationLoader(dataset_to_train=train_set, teacher_tokenizer=teacher.tokenizer, student_tokenizer=student_text_model.tokenizer)
    #
    # val_load = DistillationLoader(dataset_to_train=val_set, teacher_tokenizer=teacher.tokenizer, student_tokenizer=student_text_model.tokenizer) # Shuffle = false for metrics stability

    # loader = DistillationLoader(student_tokenizer=student_text_model.tokenizer,
    #                             teacher_tokenizer=teacher_model._first_module().tokenizer.tokenizer,
    #                             dataset=f'{get_root_directory()}/data/zara/parsed_examples/all_zara_products_deduped_clened_from_nonsib.csv')
    train_loader = DistillationLoader(
        dataset_to_train=datasets,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_text_model.tokenizer,
        is_train=True,
        stratify=True
    )

    val_loader = DistillationLoader(
        dataset_to_train=datasets,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_text_model.tokenizer,
        is_train=False,
        stratify=True,
        shuffle=False
    )
    trainer.train(train_loader=train_loader(),
                  val_loader=val_loader(),
                  num_epochs=40,
                  save_model=True,
                  save_path=model_save_path,
                  )
