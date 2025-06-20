from sentence_transformers import SentenceTransformer, util
from PIL import Image
from helpers.helper import get_root_directory
import pandas as pd
# model_clip = SentenceTransformer("clip-ViT-B-32")
#
# img_emb1 = model_clip.encode(Image.open(f"{get_root_directory()}/data/images/valentina1.png"))
# img_emb2 = model_clip.encode(Image.open(f"/{get_root_directory()}/data/images/valentina_ja.png"))
#
# similarity_scores = model_clip.similarity(img_emb1, img_emb1)
# print(similarity_scores)
#
# from sentence_transformers import SentenceTransformer
#
# sentences = ["ko je Nikola Tesla?", "Nikola Tesla je poznati pronalazač", "Nikola Jokić je poznati košarkaš"]
#
# serbian_model = SentenceTransformer('djovak/embedic-large')
#
# texts = [
#     "Full of spirit and wild at heart, Valentina is the dress everyone wants. An icon since day one, this sexy, grungy, leopard-print dream embodies the mercurial spirit of Rat & Boa – we can’t get enough of her.All our prints are exclusive and unique to Rat & Boa. Our Valentina dress has been engineered by our team and placement has been selected by Stephanie & Valentina for maximum impact."
#     "\nSilk blend"
#     "\nCowl neckline"
#     "\nBack keyhole with self tie"
#     "\nAdjustable straps"
#     "\nSemi sheer"
#     "\n Delicate item",
#     "Haljina sa leopard printom od svile",
#     "Nikola Tesla je poznati pronalazač",
#     "Leopard semi-sheer silk dress",
#     "crna haljina"
# ]
#
# util.cos_sim(serbian_model.encode("Leopard semi-sheer silk dress"), serbian_model.encode(
#     "Haljina bez rukava od pređe od 100% pamuka. Okrugli izrez, kaiš sa metalnom kopčom i donji rub A-kroja."))
# embeddings = serbian_model.encode(texts)
#
# for i in range(len(embeddings) - 1):
#     print(texts[i], texts[i + 1], util.cos_sim(embeddings[i], embeddings[i + 1]))
#
# print(embeddings)

from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageFile
import requests
import torch
from knowledge_distilation.student import StudentModel


# # We use the original clip_multilingual_text-ViT-B-32 for encoding images
# img_model = SentenceTransformer('clip-ViT-B-32')
# # img_model.save('/Users/anapaunovic/PycharmProjects/Master_rad/models/clip_vision')
#
# # Our text embedding model is aligned to the img_model and maps 50+
# # languages to the same vector space
# # text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
# text_model = StudentModel(f'{get_root_directory()}/models/clip_multilingual_text',
#                              projection_dim=512)
#
# text_model.load_state_dict(torch.load(f'{get_root_directory()}/models/distilation/50_epochs_2024-12-03-17-42.pt'))
# text_model.eval()
# Now we load and encode the images
def load_image(url_or_path):
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return Image.open(requests.get(url_or_path, stream=True).raw)
    else:
        return Image.open(url_or_path)

#
# # We load 3 images. You can either pass URLs or
# # a path on your disc
# img_paths = [
#     "/Users/anapaunovic/PycharmProjects/Master_rad/data/images/valentina1.png",
#     "/Users/anapaunovic/PycharmProjects/Master_rad/data/images/valentina_ja.png",
#     "/Users/anapaunovic/PycharmProjects/Master_rad/data/images/elephant.jpg"]
#
# images = [load_image(img) for img in img_paths]
#
# # Map images to the vector space
# img_embeddings = img_model.encode(images)
#
# # Now we encode our text:
# texts = [
#     "Full of spirit and wild at heart, Valentina is the dress everyone wants. An icon since day one, this sexy, grungy, leopard-print dream embodies the mercurial spirit of Rat & Boa – we can’t get enough of her.All our prints are exclusive and unique to Rat & Boa. Our Valentina dress has been engineered by our team and placement has been selected by Stephanie & Valentina for maximum impact."
#     "\nSilk blend"
#     "\nCowl neckline"
#     "\nBack keyhole with self tie"
#     "\nAdjustable straps"
#     "\nSemi sheer"
#     "\n Delicate item",
#     "Haljina sa leopard printom od svile",
#     "Haljina",
#     "jeans",
#     "dress",
#     "slon stoji na terasi",
#     "crna haljina",
#     "slon",
#     'elephant'
# ]
# text_embeddings = [text_model(text) for text in texts]
# text_embeddings = torch.cat(text_embeddings, dim=0)
#
# # Compute cosine similarities:
# cos_sim = util.cos_sim(text_embeddings, img_embeddings)
# # df = pd.DataFrame(
# #     cos_sim,
# #     index=texts,
# #     columns=img_paths
# # )
# for text, scores in zip(texts, cos_sim):
#     max_img_idx = torch.argmax(scores)
#     print("Text:", text)
#     print("Score:", scores[max_img_idx])
#     print("Path:", img_paths[max_img_idx], "\n")
# #
# # Load model directly
# from transformers import AutoProcessor, AutoModelForZeroShotImageClassification, AutoModel
#
# processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
# fashion_model = AutoModelForZeroShotImageClassification.from_pretrained("patrickjohncyh/fashion-clip")
# fashion_model.save('models/fashion-clip_multilingual_text')
# torch.save(fashion_model, 'models/fashion_clip2')
#
import torch
from transformers import CLIPModel, CLIPVisionModel, AutoModel

model_name = "patrickjohncyh/fashion-clip"
#
# Load the Fashion-CLIP model
model = CLIPModel.from_pretrained(model_name)
vision_encoder = model.vision_model
vit_weights = vision_encoder.state_dict()
vision_encoder.load_state_dict(vit_weights)
from safetensors.torch import save_file
# vision_encoder2 = CLIPVisionModel.from_pretrained(f"{get_root_directory()}/models/fashion_vision2")
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
# image_processor = CLIPImageProcessor()

# img = image_processor.preprocess(
#     Image.open("/Users/anapaunovic/PycharmProjects/Master_rad/data/images/pantalone.jpg"), do_normalize=True,
#     do_resize=True, size={"height": 224, "width": 224}, return_tensors='pt', do_rescale=True)
img = img_processor.preprocess(Image.open("/Users/anapaunovic/PycharmProjects/Master_rad/data/images/valentina2.png"),
                               return_tensors='pt')['pixel_values']
img1 =  img_processor.preprocess(Image.open(f"{get_root_directory()}/data/images/valentina1.png"),return_tensors='pt')['pixel_values']

res = model.get_image_features(pixel_values=img1)
print(res)

img1_res=vision_encoder.forward(img1)[1]
img2_res=vision_encoder.forward(img)[1]
cos_sim = util.cos_sim(img2_res, img1_res)

# Save as .safetensors
# save_file(tensors=vit_weights, filename=f'{get_root_directory()}/models/fashion_vision/', metadata = {'format': 'pt'} )

# vit_weights2 = vision_encoder2.state_dict()

# torch.save(vit_weights, f'{get_root_directory()}/models/fashion_vision2/fashion_vision.pt')
# model.get_image_features()
# vision_encoder.encode()
#
# model = SentenceTransformer('/Users/anapaunovic/PycharmProjects/Master_rad/models/fashion_clip')
# embeddings_ = model.forward(images)
#
# cos_sim = util.cos_sim(img_embeddings[0], img_embeddings[1])
# cos_sim
# model.encode()