import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel, CLIPTextModel, CLIPConfig
from PIL import Image
from typing import List, Union


class FashionClip:
    def __init__(self,
                 base_model_name: str,
                 text_model_name_or_path: str,
                 vision_model_name_or_path: str,
                 device: str = None):
        """
        Initializes a CLIP-like model from separate text and vision model paths.

        Parameters:
        - base_model_name: The base CLIP model name to get config and processor from (e.g. "openai/clip-vit-base-patch32").
        - text_model_name_or_path: The path or model ID for the text encoder weights.
        - vision_model_name_or_path: The path or model ID for the vision encoder weights.
        - device: 'cuda' or 'cpu'. If None, automatically detects if cuda is available.
        """

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Load the base config
        config = CLIPConfig.from_pretrained(base_model_name)

        # Initialize an empty CLIPModel with the given config
        self.clip_model = CLIPModel(config).to(self.device)
        self.text_model = CLIPTextModel.from_pretrained(text_model_name_or_path)

        # Load and replace vision model weights
        # vision_model = CLIPVisionModel.from_pretrained(vision_model_name_or_path)
        # self.clip_model.vision_model = vision_model

        # Load and replace text model weights
        # text_model = CLIPTextModel.from_pretrained(text_model_name_or_path)
        state_dict = torch.load(text_model_name_or_path)

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("text_encoder."):
                new_k = k[len("text_encoder."):]
            elif k.startswith("projection."):
                new_k = k.replace("projection.", "final_layer.")
            else:
                new_k = k
            new_state_dict[new_k] = v
        missing, unexpected = self.text_model.load_state_dict(new_state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        # self.clip_model.text_model)
        self.clip_model.text_model.load_state_dict(new_state_dict, strict=False)

        # Load the processor for consistent preprocessing
        self.processor = CLIPProcessor.from_pretrained(base_model_name)

        self.clip_model.eval()

    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encodes text into the CLIP text embedding space.

        Parameters:
        - texts: A string or a list of strings to encode.

        Returns:
        A torch.Tensor of shape (batch_size, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            text_outputs = self.clip_model.get_text_features(**inputs)
            text_embeds = F.normalize(text_outputs, p=2, dim=-1)
        return text_embeds

    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Encodes images into the CLIP vision embedding space.

        Parameters:
        - images: A list of PIL images.

        Returns:
        A torch.Tensor of shape (batch_size, embedding_dim)
        """
        if isinstance(images, Image.Image):
            images = [images]

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_outputs = self.clip_model.get_image_features(**inputs)
            image_embeds = F.normalize(image_outputs, p=2, dim=-1)
        return image_embeds

    def similarity(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Calculates cosine similarity between text embeddings and image embeddings.

        Parameters:
        - text_embeds: A tensor of shape (batch_size_text, embedding_dim)
        - image_embeds: A tensor of shape (batch_size_images, embedding_dim)

        Returns:
        A torch.Tensor of shape (batch_size_text, batch_size_images) with similarity scores.
        """
        return text_embeds @ image_embeds.T
if __name__ == "__main__":
# Example usage (assuming you have appropriate weights and model names):
    model = FashionClip(
        base_model_name="patrickjohncyh/fashion-clip",
        text_model_name_or_path="/Users/anapaunovic/Desktop/Master/models/trained/student_model_epoch_2_2024-12-14-17-09.pt",
        vision_model_name_or_path="patrickjohncyh/fashion-clip"
    )
    image = Image.open("/Users/anapaunovic/Desktop/Master/data/images/valentina1.png")
    text = "Haljina od svile sa leopard printom"

    text_embeds = model.encode_text(text)
    text_embeds2 = model.encode_text('Slon na terasi pije kafu')

    image_embeds = model.encode_images([image])

    sim = model.similarity(text_embeds, text_embeds2)
    print("Similarity:", sim.item())
