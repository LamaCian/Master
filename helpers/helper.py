import os
from safetensors import safe_open
from typing import Dict, List
import torch
from PIL import Image
import requests
from io import BytesIO
import csv


def get_root_directory():
    current_file_path = os.path.abspath(__file__)

    current_dir = os.path.dirname(current_file_path)

    root_dir = os.path.abspath(os.path.join(current_dir, ".."))

    return root_dir


def load_safe_tensors(safetensors_path: str) -> Dict[str, torch.Tensor]:
    tensors = {}
    f = safe_open(safetensors_path, framework='pt')
    for key in f.keys():
        if key.startswith('text_model') or key.startswith('text_projection'):
            continue
        tensors[key] = f.get_tensor(key)
    return tensors


def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img


def cast_csv_to_dict(csv_path: str) -> List[Dict]:
    with open(f'{csv_path}', 'r') as file:
        csv_reader = csv.DictReader(file)
        data = [row for row in csv_reader]
    return data


def load_json(json_path: str) -> List[Dict]:
    import json
    with open(json_path, 'r') as file:
        list_of_dicts = json.load(file)
    return list_of_dicts

from sklearn.model_selection import train_test_split

def get_stratified_split_indices(categories, val_size=0.2, random_state=42):
    """Returns indices for stratified train/validation split"""
    train_idx, val_idx = train_test_split(
        range(len(categories)),
        test_size=val_size,
        random_state=random_state,
        stratify=categories
    )
    return train_idx, val_idx