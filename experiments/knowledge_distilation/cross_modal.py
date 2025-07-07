import logging
import os
import arrow
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from experiments.multimodal.vision_transformer import CLIPVisionTransformerModel
from helpers.helper import get_root_directory, load_safe_tensors
from image_text_dataset import ImageTextDataset, ImageTextCollate
from student import StudentModel

from helpers.logging_config import setup_logging

setup_logging(new_dir_name='cross_modal_training')

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """
    InfoNCE loss: cross-entropy over image->text and text->image similarities.
    """
    B = similarity.size(0)
    labels = torch.arange(B, device=similarity.device)
    loss_i2t = F.cross_entropy(similarity, labels)
    loss_t2i = F.cross_entropy(similarity.T, labels)
    return (loss_i2t + loss_t2i) / 2


def compute_retrieval_accuracy(similarity: torch.Tensor) -> float:
    """
    (image->text + text->image correct) / (2*B)
    """
    B = similarity.size(0)
    i2t = similarity.argmax(dim=1)
    t2i = similarity.argmax(dim=0)
    correct = (i2t == torch.arange(B, device=similarity.device)).sum() + \
              (t2i == torch.arange(B, device=similarity.device)).sum()
    return (correct.float() / (2 * B)).item()


class CrossModalTrainer:
    def __init__(self,
                 student: StudentModel,
                 vision: CLIPVisionTransformerModel,
                 lr_proj: float = 3e-5,
                 lr_ft: float = 1e-6,
                 unfreeze_epoch: int = 5,
                 unfreeze_layers: int = 1,
                 patience: int = 5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

        # Models
        self.student = student.to(self.device)
        self.vision = vision.to(self.device)

        # Freeze backbones initially
        for p in self.student.text_encoder.parameters():
            p.requires_grad = False
        for p in self.vision.parameters():
            p.requires_grad = False
        # But projection head on student is trainable
        for p in self.student.projection.parameters():
            p.requires_grad = True

        self.optimizer = torch.optim.Adam(
            [{'params': self.student.projection.parameters(), 'lr': lr_proj}],
            betas=(0.9, 0.98)
        )
        self.lr_ft = lr_ft
        self.lr_proj = lr_proj
        self.unfreeze_epoch = unfreeze_epoch
        self.unfreeze_layers = unfreeze_layers

        self.criterion = clip_loss
        self.patience = patience

        # Logging / early stop
        self.best_loss = float("inf")
        self.epochs_wo_improve = 0
        self.train_history = []
        self.val_history = []
        self.best_states = None

    def _collect_layer_ids(self,named_params, prefix="encoder.layer"):
        """
        Return the set of integer layer indices that follow `prefix`
        in parameter names.  E.g. "encoder.layer.11.self_attn..." → 11
        """
        idxs = set()
        for name, _ in named_params:
            if prefix in name:
                parts = name.split(".")
                try:
                    # "... encoder layer <N> ..."  →  find the token after 'layer'
                    i = parts.index("layer") + 1
                    idxs.add(int(parts[i]))
                except (ValueError, IndexError):
                    # token after 'layer' not an int → skip
                    continue
        return sorted(idxs)

    def _unfreeze(self):
        # ---- student text ------------------------------------------------------
        text_layers = self._collect_layer_ids(
            self.student.text_encoder.named_parameters(), prefix="encoder.layer"
        )
        to_unlock = set(text_layers[-self.unfreeze_layers:])
        for name, p in self.student.text_encoder.named_parameters():
            if "encoder.layer" in name:
                parts = name.split(".")
                if parts.index("layer") + 1 < len(parts):
                    try:
                        idx = int(parts[parts.index("layer") + 1])
                        if idx in to_unlock:
                            p.requires_grad = True
                    except ValueError:
                        pass  # skip non-numeric

        # ---- vision ------------------------------------------------------------
        vis_layers = self._collect_layer_ids(
            self.vision.named_parameters(), prefix="encoder.layers"
        )
        to_unlock = set(vis_layers[-self.unfreeze_layers:])
        for name, p in self.vision.named_parameters():
            if "encoder.layers" in name:
                parts = name.split(".")
                try:
                    idx = int(parts[2])  # encoder.layers.<N>...
                    if idx in to_unlock:
                        p.requires_grad = True
                except ValueError:
                    pass

        params = [
            {'params': self.student.projection.parameters(), 'lr': self.lr_proj},
            {'params': [p for p in self.student.text_encoder.parameters() if p.requires_grad], 'lr': self.lr_ft},
            {'params': [p for p in self.vision.parameters() if p.requires_grad], 'lr': self.lr_ft},
        ]
        self.optimizer = torch.optim.Adam(params, betas=(0.9, 0.98))

    def forward(self, batch):
        imgs = batch["pixel_values"].to(self.device)
        toks = {
            "input_ids":      batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device)
        }

        # with torch.no_grad():
        #     v = self.vision(imgs)[1]           # pooled 512-d
        v = self.vision(imgs)[1]           # pooled 512-d
        v = F.normalize(v, dim=1)

        # Student text embeddings
        out = self.student.text_encoder(**toks)
        pooled = out.last_hidden_state[:, 0, :]
        s = self.student.projection(pooled)    # (B,512)
        s = F.normalize(s, dim=1)

        # sim = v @ s.T                        # (B, B)
        scale = self.logit_scale.exp()
        sim = (v @ s.T) * scale
        return sim

    def validate(self, loader: DataLoader):
        self.student.eval();
        self.vision.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                sim = self.forward(batch)
                total += self.criterion(sim).item()
                count += 1
            if count == 0:
                raise RuntimeError("No validation batches!")
            return total / count

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int,
              save_path: str,
              start_epoch: int = 0,
              ):

        for ep in range(start_epoch, epochs):

            if ep == self.unfreeze_epoch:
                logging.info(f"Unfreezing at epoch {ep}")
                self._unfreeze()

            self.student.train()
            self.vision.train()

            running = 0.0

            for i, batch in enumerate(train_loader):
                sim    = self.forward(batch)
                loss   = self.criterion(sim)
                running += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss = running / len(train_loader)
            val_loss   = self.validate(val_loader)
            self.train_history.append(train_loss)
            self.val_history.append(val_loss)
            logging.info(f"Epoch {ep+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss <= self.best_loss:
                self.best_loss  = val_loss
                self.epochs_wo_improve = 0
                self.best_states    = {
                    "student": self.student.state_dict(),
                    "vision":  self.vision.state_dict()
                }
                ckpt = {
                    "student": self.student.state_dict(),
                    "vision": self.vision.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epochs":ep,
                    "epochs_wo_improve":self.epochs_wo_improve,
                    "best_loss": self.best_loss,

                }

                logging.info(f"Saving checkpoint at epoch {ep+1}, best loss={ckpt['best_loss']:.4f}")

                os.makedirs(save_path, exist_ok=True)
                torch.save(ckpt,
                           os.path.join(save_path, f"clip_model_{arrow.now().format('YYYY-MM-DD-HH-mm')}.pt"))
            else:
                self.epochs_wo_improve += 1
                if self.epochs_wo_improve >= self.patience:
                    logging.info("Early stopping triggered")
                    break

        # plot
        plt.plot(self.train_history, label="train")
        plt.plot(self.val_history,   label="val")
        plt.legend()
        plt.savefig(os.path.join(save_path, f"loss_curves_{arrow.now()}.png"))
        plt.close()

        return self.best_states

if __name__ == "__main__":
    import argparse
    from batch_sampler import UniqueSampler

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Path to a checkpoint .pt file to resume from"
    )
    args = parser.parse_args()



    base = get_root_directory()

    # 1) load text student
    student = StudentModel(f"{base}/models/clip_multilingual_text", projection_dim=512)
    ckpt = torch.load(f"{base}/models/trained/distillation/student_model_epoch_17_2025-05-29-15-47.pt")['student_model_state_dict']
    student.load_state_dict(ckpt)

    # 2) load vision model
    vw = load_safe_tensors(f"{base}/models/fashion_clip/model.safetensors")
    vision = CLIPVisionTransformerModel(weights=vw)

    # 3) dataset + loaders
    ds = ImageTextDataset(
        dataset=f"{base}/Parsing/combined_datasets_2025-06-19-20-48.csv",
        tokenizer=student.tokenizer,
        image_col="Local files",
        caption_col=["Description"],
    )
    tr_sz = int(0.8 * len(ds))
    train_ds, val_ds = random_split(ds, [tr_sz, len(ds) - tr_sz])
    global_caption_ids = [item["caption_id"] for item in ds.items]
    val_caption_ids = [global_caption_ids[i] for i in val_ds.indices]
    train_caption_ids = [global_caption_ids[i] for i in train_ds.indices]

    train_sampler = UniqueSampler(caption_ids=train_caption_ids,
                            batch_size=128)

    val_sampler = UniqueSampler(caption_ids=val_caption_ids,
                                batch_size=128)

    tr_ld = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=ImageTextCollate())
    vl_ld = DataLoader(val_ds,  batch_sampler=val_sampler, collate_fn=ImageTextCollate())

    # 4) train
    trainer = CrossModalTrainer(
        student=student,
        vision=vision,
        lr_proj=3e-5,
        lr_ft=1e-6,
        unfreeze_epoch=65,
        unfreeze_layers=2,
        patience=5
    )

    start_epoch = 0
    ckpt_trained_path = f'{base}/models/trained/crossmodal/clip_model_2025-07-03-20-37.pt'
    if ckpt_trained_path:
        ckpt_trained = torch.load(ckpt_trained_path)
        trainer.student.load_state_dict(ckpt_trained["student"])
        trainer.vision.load_state_dict(ckpt_trained["vision"])

        trainer.optimizer = torch.optim.Adam(
            trainer.student.projection.parameters(),
            lr=ckpt_trained.get("lr_proj", 3e-5)
        )

        if "optimizer" in ckpt_trained:
            trainer.optimizer.load_state_dict(ckpt_trained["optimizer"])

            for state in trainer.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(trainer.device)

        trainer.train_history = ckpt_trained["train_history"] if "train_history" in ckpt_trained else trainer.train_history
        trainer.val_history = ckpt_trained["val_history"] if "val_history" in ckpt_trained else trainer.val_history
        trainer.best_loss = ckpt_trained["best_loss"] if "best_loss" in ckpt_trained else trainer.best_loss
        start_epoch = ckpt_trained["epochs"] + 1 if "epochs" in ckpt_trained else start_epoch
        trainer.epochs_wo_improve = ckpt_trained['epochs_wo_improve'] if "epochs_wo_improve" in ckpt_trained else 0
        logging.info(f"Resuming from epoch {start_epoch}")
        logging.info("Best loss: {}".format(trainer.best_loss))



    trainer.train(train_loader=tr_ld,
                  val_loader=vl_ld,
                  epochs=200,
                  start_epoch=start_epoch,
                  save_path=f"{base}/models/trained/crossmodal/")
