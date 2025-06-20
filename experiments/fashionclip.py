from PIL import Image
from torch.nn.modules.module import T
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer, ViTModel, ViTConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import TripletLoss
from torch.utils.data import DataLoader
from dataset_skeleton import TripletDataset
from sentence_transformers import SentenceTransformer


# Define the projection dimension


class VisionEncoderWithProjection(nn.Module):
    def __init__(self, vit_model, projection_dim=256):
        super(VisionEncoderWithProjection, self).__init__()
        self.vit_model = vit_model
        self.projection = nn.Linear(vit_model.config.hidden_size, projection_dim)

    def forward(self, pixel_values):
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

    def forward(self, text: str):
        outputs = self.text_model(text)
        # Get the [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Apply the projection
        projected = self.text_projection(cls_output)
        # Normalize the embeddings
        normalized = F.normalize(projected, p=2, dim=1)


class CustomModel(nn.Module):
    def __init__(self, vision_model, text_model, lr: float = 1e-3):
        super(CustomModel, self).__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.tokenizer = text_model.tokenizer
        self.lr = lr

        self.optimizer = torch.optim.Adam(
            list(self.vision_model.parameters()) + list(self.text_model.parameters()),
            lr=1e-4  # change to self.lr?
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def encode_image(self, image):
        self.vision_model.forward(image)

    def encode_text(self, txt: str):
        return self.text_model.forward(txt)

    def forward(self, images, text):
        tokens = self.tokenizer(text)
        text_features = self.text_model.forward(tokens)
        image_features = self.vision_model.forward(images)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

    def train(self, num_epochs: int, dataloader):
        for epoch in range(num_epochs):
            for batch in dataloader:
                image = batch[0]
                positive = batch[1]
                negative = batch[2]
                self.optimizer.zero_grad()

                anchor_emb = self.vision_model(image)  # [batch_size, projection_dim]
                positive_embs = self.text_model.forward(self.tokenizer(positive))
                negative_embs = self.text_model.forward(self.tokenizer(negative))

                # Compute triplet loss
                triplet_loss = TripletLoss(margin=0.2)
                loss = triplet_loss.forward(anchor_emb, positive_embs, negative_embs)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


if __name__ == "__main__":
    clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    vision_encoder = clip_model.vision_model
    vit_state_dict = vision_encoder.state_dict()
    text_model = SentenceTransformer('djovak/embedic-large')

    multimodal = CustomModel(vision_model=vision_encoder, text_model=text_model)
    dataset = TripletDataset("/Users/anapaunovic/PycharmProjects/Master_rad/triplets_fashion.json")
    clip_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    multimodal.train(num_epochs=3, dataloader=clip_dataloader)

# image = Image.open("/Users/anapaunovic/PycharmProjects/Master_rad/data/images/valentina1.png")
#
# inputs = processor(text=["a photo of a red shoe", "a photo of a black shoe",
#                          "Valentina is a standout dress, known for its bold and untamed spirit. With its leopard-print design, this dress perfectly captures the essence of Rat & Boa's daring and rebellious aesthetic. Exclusively designed by Rat & Boa, the Valentina dress has been carefully crafted with precision by their team, with print placements selected for maximum visual impact by Stephanie & Valentina. The dress features a silk blend fabric, a cowl neckline, a back keyhole with a self-tie, adjustable straps, and a semi-sheer design, making it a delicate item that requires careful handling.",
#                          "Rat & Boa Valentina dress leopard print silk blend semi-sheer cowl neckline adjustable straps",
#                          "leopard silk long dress"],
#                    images=image, return_tensors="pt", padding=True)
#
# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1)
# print(probs)
# image.resize((224, 224))
