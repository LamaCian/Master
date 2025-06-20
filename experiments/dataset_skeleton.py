import json
import logging

import matplotlib.pyplot as plt
from PIL import Image
from typing import Union, Optional
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import numpy as np
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor
from helpers.helper import get_root_directory, cast_csv_to_dict, load_json
from torch.nn.utils.rnn import pad_sequence


class TripletDataset(Dataset):
    def __init__(self, dataset: Union[str], tokenizer: Optional = None, image_transform=None):
        if not dataset:
            raise ValueError("Dataset can't be None")

        if isinstance(dataset, str):
            if dataset.endswith('json'):
                self.data = load_json(dataset)
            elif dataset.endswith('.csv'):
                self.data = cast_csv_to_dict(dataset)

        self.tokenizer = tokenizer

        if image_transform:
            self.image_transform = image_transform
        else:

            self.image_transform = CLIPImageProcessor(
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]

        try:
            anchor_img = Image.open(sample['anchor_image']).convert('RGB')

        except Exception as e:
                logging.warning(f"Skipping {sample}: {e}")
                return None

        if self.image_transform:
            anchor_img = self.image_transform(images=anchor_img)['pixel_values'][0]

        positive_description = sample['positive_description']
        negative_description = sample['negative_description']

        # positive_description = self.tokenizer(positive_description,padding=True, truncation=True, return_tensors="pt")
        # negative_description = self.tokenizer(negative_description,padding=True, truncation=True, return_tensors="pt")

        return {'image': anchor_img, 'positive': positive_description, 'negative': negative_description}

    def preview(self, idx=int | None, unnormalize: bool = True):
        if not idx:
            idx = torch.randint(low=0, high=self.__len__() - 1, size=(1,)).item()
        example = self.data[idx]
        anchor_image, positive_description, negative_description = example['anchor_image'], example[
            'positive_description'], example['negative_description']
        if self.image_transform:
            anchor_image = self.image_transform(Image.open(anchor_image))
            anchor_image = anchor_image.permute(1, 2, 0).numpy()

        if unnormalize:
            image = (anchor_image * 0.229) + 0.485
            image = np.clip(image, 0, 1)
        else:
            image = anchor_image

        print("The image positive description:{}".format(positive_description))
        print("\nThe image negative description:{}".format(negative_description))

        plt.imshow(image)
        plt.axis('off')
        plt.title('Anchor Image')
        plt.show()


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):

        batch = [sample for sample in batch if sample is not None]

        if len(batch) == 0:
            return {'image': torch.empty(0), 'positive_input_ids': torch.empty(0),
                    'positive_attention_mask': torch.empty(0),
                    'negative_input_ids': torch.empty(0),
                    'negative_attention_mask': torch.empty(0)}


        pixel_values = torch.stack([torch.from_numpy(sample['image']) for sample in batch])

        positive_texts = [sample['positive'] for sample in batch]
        negative_texts = [sample['negative'] for sample in batch]

        positive_encodings = self.tokenizer(positive_texts, padding=True, truncation=True, return_tensors='pt')
        negative_encodings = self.tokenizer(negative_texts, padding=True, truncation=True, return_tensors='pt')

        batch = {
            "image": pixel_values,
            "positive_input_ids": positive_encodings['input_ids'],
            "positive_attention_mask": positive_encodings['attention_mask'],
            "negative_input_ids": negative_encodings['input_ids'],
            "negative_attention_mask": negative_encodings['attention_mask'],
        }

        return batch


if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer

    text_model_name = 'djovak/embedic-large'
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    dataset = TripletDataset("/Users/anapaunovic/Desktop/Master/Parsing/triplet_dataset.csv",
                             tokenizer=tokenizer)
    # data_idx0 = dataset.__getitem__(0)
    # dataset.preview(0)
    # print(data_idx0)
    clip_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=Collate(tokenizer))
    for batch in clip_dataloader:
        images = batch['image']
        positive_input_ids = batch['positive_input_ids']
        positive_attention_mask = batch['positive_attention_mask']
        negative_input_ids = batch['negative_input_ids']
        negative_attention_mask = batch['negative_attention_mask']

        print(f'Images shape: {images.shape}')
        print(f'Positive input IDs shape: {positive_input_ids.shape}')
        print(f'Positive attention mask shape: {positive_attention_mask.shape}')
        print(f'Negative input IDs shape: {negative_input_ids.shape}')
        print(f'Negative attention mask shape: {negative_attention_mask.shape}')
        break
