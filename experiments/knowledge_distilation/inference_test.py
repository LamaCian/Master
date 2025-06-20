from student import StudentModel
from helpers.helper import get_root_directory
import torch


combined_model_weights = torch.load(f'{get_root_directory()}/models/trained/image_text_model/checkpoint_epoch_117_2025-05-15-21-00.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

student_model = StudentModel(f'{get_root_directory()}/models/clip_multilingual_text',
                             projection_dim=512)

student_model.load_state_dict(torch.load(f'{get_root_directory()}/models/QueryTextPair/query_text_model_epoch_2_2025-04-21-15-25.pt')['model_state_dict'])
# student_model.load_state_dict(torch.load(f'{get_root_directory()}/models/trained/distillation/student_model_epoch_10_2025-05-29-14-38.pt')['student_model_state_dict'])
student_model.to(device)
student_model.eval()
student_tokenizer = student_model.tokenizer

serbian_sentences = ['Crvena haljina sa cvetnim uzorkom', 'Haljina crvene boje sa cvetnim dezenom',
                     'sorts sa dubokim dzepom', 'slon na terasi',
                     'London je glavni grad engleske', 'Red dress with floral pattern', 'dress', 'red dress',
                     'elephant', 'slon', 'teksas jakna', 'jakna']
inputs = student_tokenizer(
    serbian_sentences,
    return_tensors='pt',
    padding='max_length',
    truncation=True,
    max_length=512
)

serbian_inputs = {k: v.to(device) for k, v in inputs.items()}
# english_inputs = {k: v.to(device) for k, v in english_inputs.items()}
text_model = student_model
text_model.load_state_dict(
    torch.load(f'{get_root_directory()}/models/trained/image_text_model/checkpoint_epoch_117_2025-05-15-21-00.pt')[
        'text_model_state_dict'])

with torch.no_grad():
    embeddings = text_model(
        input_ids=serbian_inputs['input_ids'],
        attention_mask=serbian_inputs['attention_mask']
    )

import torch.nn.functional as F

# Ensure embeddings are normalized (they should be if normalization was applied in the model)
text_embeddings = F.normalize(embeddings, p=2, dim=1)

# Compute cosine similarities
cosine_similarities = torch.matmul(text_embeddings, text_embeddings.T)  # Shape: (batch_size, batch_size)

import pandas as pd

# Create a DataFrame to display similarities
df_txt = pd.DataFrame(
    cosine_similarities,
    index=serbian_sentences,
    columns=serbian_sentences
)

df_txt.to_csv("multi_modal.csv")

from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageFile
import requests
import torch
from experiments.multimodal.vision_transformer import \
    CLIPVisionTransformerModel

vision_weights = torch.load(
    f'{get_root_directory()}/models/trained/image_text_model/checkpoint_epoch_117_2025-05-15-21-00.pt')['vision_model_state_dict']

for k in list(vision_weights.keys()):
    if k != 'visual_proj':
        vision_weights[f'vision_model.{k}'] = vision_weights.pop(k)

vision_model = CLIPVisionTransformerModel(weights=vision_weights)

# img_model = SentenceTransformer('clip-ViT-B-32').to(device)


# fashion_clip = SentenceTransformer('/Users/anapaunovic/Desktop/Master/models/fashion_clip')

#
#
# fashion_clip.text_encoder.to(device)
# fashion_clip._first_module().model.text_model = student_model
def load_image(url_or_path):
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return Image.open(requests.get(url_or_path, stream=True).raw)
    else:
        return Image.open(url_or_path)


from transformers import CLIPImageProcessor

img_processor = CLIPImageProcessor(do_rescale=True,
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

# We load 3 images. You can either pass URLs or
# a path on your disc
img_paths = [
    f"{get_root_directory()}/data/images/valentina1.png",
    f"{get_root_directory()}/data/images/valentina_ja.png",
    f"{get_root_directory()}/data/images/elephant.jpg",
    f"{get_root_directory()}/data/images/lion.jpg",
    f"{get_root_directory()}/data/images/zara_dress.jpg",
    f"{get_root_directory()}/data/images/zara_dress2.jpg",
    f"{get_root_directory()}/data/images/basic.jpg",
    f"{get_root_directory()}/data/images/pantalone.jpg",
    f"{get_root_directory()}/data/images/valentina2.png"]
images = [img_processor.preprocess(
    images=Image.open(image),
    return_tensors='pt',
) for image in img_paths]

# images = [load_image(img) for img in img_paths]

# Map images to the vector space
# img_embeddings = img_model.encode(images)
# img_embeddings = fashion_clip.encode(images)

# Encode images with fashion_clip
res = vision_model.forward(images[0]['pixel_values'])
img_embeddings = [vision_model.forward(image['pixel_values'])[1] for image in images]

emb_list = []
for img in images:
    pixel_values = img['pixel_values'].to(device)  # [1,3,224,224]
    with torch.no_grad():
        last_hidden, proj, *rest = vision_model(pixel_values)
        # proj has shape [1,512]
    # remove the batch‐dim:
    proj = proj.squeeze(0)  # [512]
    emb_list.append(proj)

# now stack into [3,512]:
img_embeddings = torch.stack(emb_list, dim=0)  # [3,512]

# normalize to unit length
img_embeddings = F.normalize(img_embeddings, p=2, dim=1)  # [3,512]

# compute full pairwise cosine similarity matrix
sim_matrix = img_embeddings @ img_embeddings.T  # [3,3]

# (or equivalently)
cos_sim_matrix = util.cos_sim(img_embeddings, img_embeddings)  # [3,3]
img_names = [img.split('/')[-1].split('.')[0] for img in img_paths]
# Create a DataFrame to display similarities
df_img = pd.DataFrame(
    cos_sim_matrix,
    index=img_names,
    columns=img_names
)
print(df_img)

# Now we encode our text:
texts = [
    "Full of spirit and wild at heart, Valentina is the dress everyone wants. An icon since day one, this sexy, grungy, leopard-print dream embodies the mercurial spirit of Rat & Boa – we can’t get enough of her.All our prints are exclusive and unique to Rat & Boa. Our Valentina dress has been engineered by our team and placement has been selected by Stephanie & Valentina for maximum impact."
    "\nSilk blend"
    "\nCowl neckline"
    "\nBack keyhole with self tie"
    "\nAdjustable straps"
    "\nSemi sheer"
    "\n Delicate item",
    "Haljina sa leopard printom od svile",
    "Haljina",
    "jeans",
    "dress",
    "slon stoji na terasi",
    "crna haljina",
    "slon",
    'pantalone',
    'pants'
]
inputs = student_tokenizer(
    texts,
    return_tensors='pt',
    padding='max_length',
    truncation=True,
    max_length=512
)

serbian_inputs = {k: v.to(device) for k, v in inputs.items()}
# english_inputs = {k: v.to(device) for k, v in english_inputs.items()}

with torch.no_grad():
    txt_embeddings = text_model(
        input_ids=serbian_inputs['input_ids'],
        attention_mask=serbian_inputs['attention_mask']
    )
txt_embeddings = F.normalize(txt_embeddings, p=2, dim=1)
# img_embeddings = embeddings
# Compute cosine similarities:
cos_sim = util.cos_sim(txt_embeddings, img_embeddings)

# Create a DataFrame to display similarities
df = pd.DataFrame(
    cos_sim,
    index=texts,
    columns=img_names
)
print(df)
