import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from typing import List, Union


class FashionClipST:
    def __init__(self,
                 text_model: str,
                 text_model_name_or_path: str,
                 vision_model_name_or_path: str,
                 device: str = None):
        """
        Initializes a CLIP-like setup using SentenceTransformer models for text and vision.

        Parameters:
        - text_model_name_or_path: Path or model name for the text SentenceTransformer.
        - vision_model_name_or_path: Path or model name for the image SentenceTransformer.
        - device: 'cuda' or 'cpu'. If None, automatically detects if CUDA is available.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Load the text model (SentenceTransformer)
        self.text_model = SentenceTransformer(text_model, device=self.device)
        state_dict = torch.load(text_model_name_or_path)

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("text_encoder."):
                new_k = '0.auto_model.' + k[len("text_encoder."):]
            elif k.startswith("projection."):
                new_k = k.replace("projection.", "2.linear.")
            else:
                new_k = k
            new_state_dict[new_k] = v
        missing, unexpected = self.text_model.load_state_dict(new_state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        self.text_model.eval()

        # Load the vision model (SentenceTransformer)
        self.vision_model = SentenceTransformer(vision_model_name_or_path, device=self.device)
        self.vision_model.eval()

    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encodes text into the embedding space using the text SentenceTransformer model.

        Parameters:
        - texts: A string or a list of strings.

        Returns:
        Torch tensor of shape (batch_size, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        # SentenceTransformer's encode returns a NumPy array, convert to torch tensor
        embeddings = self.text_model.encode(texts, convert_to_tensor=True, show_progress_bar=False, device=self.device)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Encodes images into the embedding space using the vision SentenceTransformer model.

        Parameters:
        - images: A list of PIL.Image objects.

        Returns:
        Torch tensor of shape (batch_size, embedding_dim)
        """
        if isinstance(images, Image.Image):
            images = [images]
        # encode images using the vision model
        embeddings = self.vision_model.encode(images, convert_to_tensor=True, show_progress_bar=False,
                                              device=self.device)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def similarity(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Calculates cosine similarity between text and image embeddings.

        Parameters:
        - text_embeds: torch.Tensor of shape (batch_size_text, embedding_dim)
        - image_embeds: torch.Tensor of shape (batch_size_images, embedding_dim)

        Returns:
        Torch tensor of shape (batch_size_text, batch_size_images)
        """
        return util.cos_sim(text_embeds, image_embeds)


if __name__ == "__main__":
    from helpers.helper import get_root_directory

    model = FashionClipST(
        text_model=f'{get_root_directory()}/models/clip_multilingual_text',
        text_model_name_or_path=f'{get_root_directory()}/models/distilation/50_epochs_2024-12-03-17-42.pt',
        vision_model_name_or_path="clip-ViT-B-32"
    )

    image = Image.open("/Users/anapaunovic/Desktop/Master/data/images/elephant.jpg")
    text = "lion"

    text_embeds = model.encode_text(text)
    # text2 = model.encode_text('Svilena haljina sa leopard printom.')
    image_embeds = model.encode_images([image])

    sim = model.similarity(text_embeds, image_embeds)
    print("Similarity:", sim.item())
