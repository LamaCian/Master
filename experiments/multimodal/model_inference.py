import torch
from PIL import Image

from experiments.multimodal.combined_model import CustomModel
from helpers.helper import get_root_directory, load_safe_tensors
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor
from vision_transformer import CLIPVisionTransformerModel
import logging
from sentence_transformers import SentenceTransformer, util
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set up logging if needed
logging.basicConfig(level=logging.INFO)

# Load tokenizer and models
text_model_name = 'djovak/embedic-base'
tokenizer = AutoTokenizer.from_pretrained(text_model_name)

# Load vision model weights
vision_weights = load_safe_tensors(
    f'{get_root_directory()}/models/fashion_clip/model.safetensors')
vision_model = CLIPVisionTransformerModel(weights=vision_weights)

# Load text model
text_model = AutoModel.from_pretrained(text_model_name)

# # Initialize the custom model
model = CustomModel(
    text_encoder=text_model,
    vision_encoder=vision_model,
    tokenizer=tokenizer,
    embed_dim=512
)
model.to(device)

#
# # Load the trained model weights
# models_path = f'{get_root_directory()}/models/trained/trained_100_epochs_2024-11-28-21-16.pt'
# state_dict = torch.load(models_path)
# model.load_state_dict(state_dict)
# model.eval()
#
# # Set the device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
#
# # Initialize the image processor
# image_processor = CLIPImageProcessor(
#     do_rescale=True,
#     do_resize=True,
#     do_center_crop=True,
#     do_convert_rgb=True,
#     do_normalize=True,
#     resample=3,
#     crop_size={"height": 224, "width": 224},
#     rescale_factor=0.00392156862745098,
#     size={"shortest_edge": 224},
#     image_std=[
#         0.26862954,
#         0.26130258,
#         0.27577711
#     ],
#     image_mean=[
#         0.48145466,
#         0.4578275,
#         0.40821073
#     ],
# )
# serbian_model = SentenceTransformer('djovak/embedic-base', device="cpu")
# texts = [
#     "Full of spirit and wild at heart, Valentina is the dress everyone wants. An icon since day one, this sexy, grungy, leopard-print dream embodies the mercurial spirit of Rat & Boa – we can’t get enough of her.All our prints are exclusive and unique to Rat & Boa. Our Valentina dress has been engineered by our team and placement has been selected by Stephanie & Valentina for maximum impact."
#     "\nSilk blend"
#     "\nCowl neckline"
#     "\nBack keyhole with self tie"
#     "\nAdjustable straps"
#     "\nSemi sheer"
#     "\n Delicate item",
#     "Leopard semi-sheer silk dress",
#     'haljina sa leopard printom od svile, polu-providna',
#     'slon',
#     'tigar',
#     "slon stoji na terasi",
#
# ]
#
# serbian_model_encodings = [serbian_model.tokenize([text],
#                                                   )['input_ids'] for text in texts]
# embeddings = serbian_model.encode(texts)
# text_embeddings_model = [torch.tensor(model.get_text_features(text)) for text in texts]
#
# img1 = Image.open(f'{get_root_directory()}/data/images/valentina1.png')
# img2 = Image.open(f'{get_root_directory()}/data/images/elephant.jpg')
#
# img1_preprocessed = image_processor.preprocess(
#     images=img1,
#     return_tensors='pt'
# )['pixel_values'].to(device)
#
# img2_preprocessed = image_processor.preprocess(
#     images=img2,
#     return_tensors='pt'
# )['pixel_values'].to(device)
#
# similarity = model.zero_shot_classification(img1_preprocessed, [
#                                                                 'leopard print dress',
#                                                                 'jeans',
#                                                                 'pants',
#                                                                 'elephant',
#                                                                 'dress'])
# print(similarity)

# pairs = set()
# for i in range(len(texts)):
#     j = i + 1
#     if j ==len(texts):
#         continue
#
#     pairs.add((texts[i], texts[j]))


# logging.info('Sentence Transformer')
# for i in range(len(embeddings) - 1):
#     print("text1:",texts[i],"text2:", texts[i + 1], util.cos_sim(embeddings[i], embeddings[i + 1]))
#
#
# logging.info('Text embeddings using custom model')
# for i in range(len(text_embeddings_model) - 1):
#     print("text1:",texts[i],"text2:", texts[i + 1], util.cos_sim(text_embeddings_model[i], text_embeddings_model[i + 1]))

#
# Initialize the image processor
image_processor = CLIPImageProcessor(
    do_rescale=True,
    do_resize=True,
    do_center_crop=True,
    do_convert_rgb=True,
    do_normalize=True,
    resample=3,
    crop_size={"height": 224, "width": 224},
    rescale_factor=0.00392156862745098,
    size={"shortest_edge": 224},
    image_std=[
        0.26862954,
        0.26130258,
        0.27577711
    ],
    image_mean=[
        0.48145466,
        0.4578275,
        0.40821073
    ],
)

# Load and preprocess the images
img1 = Image.open(f'{get_root_directory()}/data/images/valentina_ja.png')
img2 = Image.open(f'{get_root_directory()}/data/images/elephant.jpg')

img1_preprocessed = image_processor.preprocess(
    images=img1,
    return_tensors='pt'
)['pixel_values'].to(device)

img2_preprocessed = image_processor.preprocess(
    images=img2,
    return_tensors='pt'
)['pixel_values'].to(device)



custom_vision_end = model.vision_encoder
custom_vision_end.eval()
custom_vision_end.to(device)

# Load the vision model weights as before
vision_weights = load_safe_tensors(f'{get_root_directory()}/models/fashion_clip/model.safetensors')
vision_model = CLIPVisionTransformerModel(weights=vision_weights)
vision_model.eval()
vision_model.to(device)

# Get the embeddings directly from the vision model
res1 = vision_model(img1_preprocessed)
res2 = vision_model(img2_preprocessed)

embeddings1_vis = res1[1]
embeddings2_vis = res2[1]

# Normalize embeddings
embeddings1_vis = torch.nn.functional.normalize(embeddings1_vis, p=2, dim=-1)
embeddings2_vis = torch.nn.functional.normalize(embeddings2_vis, p=2, dim=-1)

# Compute cosine similarity
cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
sim = cosine_similarity(embeddings1_vis, embeddings2_vis)
print("Cosine similarity using vision_model:", sim.item())

# Get image features
embeddings1 = model.get_image_features(images=img1_preprocessed)
embeddings2 = model.get_image_features(images=img2_preprocessed)


embeddings1_custom =custom_vision_end(img1_preprocessed)[1]
embeddings2_custom = custom_vision_end(img2_preprocessed)[1]
embeddings1_custom = torch.nn.functional.normalize(embeddings1_custom, p=2, dim=-1)
embeddings2_custom = torch.nn.functional.normalize(embeddings2_custom, p=2, dim=-1)

cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
sim = cosine_similarity(embeddings1_custom, embeddings2_custom)
print("Cosine similarity using vision model from combined model:", sim.item())

# If embeddings are tuples, extract the embeddings
if isinstance(embeddings1, tuple):
    embeddings1 = embeddings1[1]
if isinstance(embeddings2, tuple):
    embeddings2 = embeddings2[1]

# Normalize the embeddings
embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=-1)
embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=-1)

# Compute cosine similarity
cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
sim = cosine_similarity(embeddings1, embeddings2)
print("Cosine similarity using combined model:", sim.item())
diff1 = torch.sum(torch.abs(embeddings1 - embeddings1_vis))
diff2 = torch.sum(torch.abs(embeddings2 - embeddings2_vis))
print("Difference between embeddings1_custom and embeddings1_direct:", diff1.item())
print("Difference between embeddings2_custom and embeddings2_direct:", diff2.item())
