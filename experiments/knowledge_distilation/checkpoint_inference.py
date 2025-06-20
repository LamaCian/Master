import torch
import torch.nn as nn
from PIL.ImagePalette import negative

from experiments.multimodal.vision_transformer import \
    CLIPVisionTransformerModel
from student import StudentModel
from helpers.helper import get_root_directory, load_safe_tensors
from PIL import Image


def load_model_from_checkpoint(checkpoint_path: str, text_model, vision_model):
    """
    Load model weights and optimizer state from a checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Change to 'cuda' if using GPU
    text_model.load_state_dict(checkpoint["text_model_state_dict"])
    vision_model.load_state_dict(checkpoint["vision_model_state_dict"])
    print(f"Loaded models from checkpoint at {checkpoint_path}")
    return text_model, vision_model


def extract_embeddings_and_compute_similarity(text_model, vision_model, text_inputs, image_inputs, device):
    """
    Extract text and image embeddings and compute their cosine similarity.
    """
    # Move models and inputs to the device
    text_model = text_model.to(device)
    vision_model = vision_model.to(device)
    text_model.eval()
    vision_model.eval()

    with torch.no_grad():
        # Get text embeddings
        text_embeddings = text_model(text_inputs)  # Modify based on your text model's forward method

        # Get image embeddings
        image_features = vision_model(image_inputs)
        image_embeddings = nn.functional.normalize(image_features[1], p=2, dim=1)

        # Compute cosine similarity
        similarity = torch.matmul(image_embeddings, text_embeddings.T)

    return similarity


# Example usage
if __name__ == '__main__':
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

    student_text_model = StudentModel(f'{get_root_directory()}/models/clip_multilingual_text', projection_dim=512)
    # vision_weights = load_safe_tensors(
    #     '/Users/anapaunovic/Desktop/Master/models/fashion_clip/model.safetensors')
    vision_weights = torch.load(
        f'{get_root_directory()}/models/trained/vision_model_epoch_10_2024-12-15-20-50.pt')
    for k in list(vision_weights.keys()):
        if k != 'visual_proj':
            vision_weights[f'vision_model.{k}'] = vision_weights.pop(k)

    vision_model = CLIPVisionTransformerModel(weights=vision_weights)

    # Load checkpoint
    checkpoint_path = f'{get_root_directory()}/models/trained/image_text_model/checkpoint_epoch_117_2025-05-15-21-00.pt'
    student_text_model, vision_model = load_model_from_checkpoint(checkpoint_path, student_text_model, vision_model)

    # / Volumes / T7 / data / zara / jackets / jakna_od_mesavine_vune_drveni_ugljen_2.jpg
    false_desc = 'haljina okruglog izreza i dugih rukava preklop sa prednje strane sa kopcom zlatne boje slic na donjem rubu sa prednje strane i kopcanje skrivenim rajsferslusom pozadi,kosulja sa v izrezom visokom kragnom sa perlicama i uckurom dugi rukavi poluprozirna tkanina porubi sa karnerima i skriveno kopcanje dugmicima napred'
    text = 'pantalone visokog struka sa veoma sirokom nogavicom detalj od nabrane tkanine kopcanje skrivenim rajsferslusom sa strane'
    true_desc = 'dugacka elegantna haljina sa leopard printom, poluprovidna od svile.'
    img = img_processor.preprocess(
        Image.open("/Users/anapaunovic/Desktop/Master/data/images/valentina1.png"),
        return_tensors='pt')
    img_false = img_processor.preprocess(Image.open('/Volumes/T7/data/zara/dress/midi_haljina_od_krepa_sa_kopcom_tamnozelena_boja_1.jpg'), return_tensors='pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    false_similarity = extract_embeddings_and_compute_similarity(student_text_model, vision_model, false_desc, img['pixel_values'],
                                                           device)
    true_sim_false = extract_embeddings_and_compute_similarity(student_text_model, vision_model, false_desc, img_false['pixel_values'],
                                                           device)
    print("Similarity Matrix false:",false_similarity)
    print("Similarity Matrix:",true_sim_false)
    print('similarity silk dress', extract_embeddings_and_compute_similarity(student_text_model, vision_model,true_desc, img['pixel_values'],device))
    print("Similarity Matrix:",extract_embeddings_and_compute_similarity(student_text_model, vision_model,text, img_false['pixel_values'],device))
    print("Similarity Matrix:",extract_embeddings_and_compute_similarity(student_text_model, vision_model,text, img['pixel_values'],device))
