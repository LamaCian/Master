from huggingface_hub import upload_file
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPImageProcessor
from typing import Optional, List, Dict, Union
import torch
from helpers.helper import get_root_directory, cast_csv_to_dict, load_json
import logging
import ast

logging.basicConfig(level=logging.INFO)


class ImageTextDataset(Dataset):
    def __init__(self, dataset: Union[str, List[Dict]],
                 image_col: str,
                 caption_col: Union[str,List[str]],
                 tokenizer,
                 image_transform: Optional = None):

        if isinstance(dataset, str):
            if dataset.endswith('.csv'):
                self.dataset = cast_csv_to_dict(dataset)
            if dataset.endswith('.json'):
                self.dataset = load_json(dataset)
        else:
            self.dataset = dataset

        if isinstance(caption_col, str):
            caption_col = [caption_col]

        self.items = []
        for row in self.dataset:
            img_paths = self._parse_image_paths(row[image_col])
            if not img_paths:
                continue
            img_path = img_paths[0]
            for col in caption_col:
                txt = row.get(col, None)
                if txt:
                    self.items.append({"image": img_path, "text": txt})

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
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        try:
            image = Image.open(item['image']).convert('RGB')

        except Exception as e:
            logging.warning(
                f"Skipping image: with following description:{
                item['text']}:ERROR: \n{e}")
            return None
        image = self.image_transform(image, return_tensors="pt")['pixel_values'].squeeze(0)

        text_inputs = self.tokenizer(item['text'],
                                     return_tensors='pt',
                                     padding='max_length',
                                     truncation=True,
                                     max_length=77)
        logging.info("Successfully tokenized image and text inputs: " + item['text'])

        return {'image': image,
                'input_ids': text_inputs['input_ids'].squeeze(0),
                'attention_mask': text_inputs['attention_mask'].squeeze(0)}

    @staticmethod
    def _parse_image_paths(image_path_entry):
        if isinstance(image_path_entry, str):
            try:
                parsed_paths = ast.literal_eval(image_path_entry)
                if isinstance(parsed_paths, list):
                    return parsed_paths
            except (ValueError, SyntaxError):
                pass
            return [image_path_entry]

        elif isinstance(image_path_entry, list):
            return image_path_entry

        elif isinstance(image_path_entry, int):
            return [f"./images/{image_path_entry}.jpg"]

        logging.warning(f"Unrecognized format for image paths: {image_path_entry}")
        return []


class ImageTextCollate:
    def __call__(self, batch):
        batch = [sample for sample in batch if sample is not None]

        if len(batch) == 0:
            return {'image': torch.empty(0),
                    'captions': torch.empty(0)}

        pixel_values = torch.stack([(sample['image']) for sample in batch])
        input_ids = torch.cat([sample['input_ids'].unsqueeze(0) for sample in batch], dim=0)
        attention_mask = torch.cat([sample['attention_mask'].unsqueeze(0) for sample in batch], dim=0)
        return {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask}


if __name__ == '__main__':
    from student import StudentModel

    student_text_model = StudentModel(f'{get_root_directory()}/models/clip_multilingual_text', projection_dim=512)
    student_text_model.load_state_dict(
        torch.load(f'{get_root_directory()}/models/trained/student_model_epoch_9_2024-12-09-18-15.pt'))

    student_tokenizer = student_text_model.tokenizer
    dataset = ImageTextDataset(
        dataset=f'{get_root_directory()}/Parsing/zara/clothes_combined.csv',
        tokenizer=student_tokenizer,
        image_col='Local Images',
        caption_col='Description')

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=ImageTextCollate())

    for btch in dataloader:
        pixel_values, input_ids, attention_mask = btch['pixel_values'], btch['input_ids'], btch['attention_mask']

        print('caption', input_ids)
        print(f'caption shape: {input_ids.shape}')
        print(f'images shape: {pixel_values.shape}')
        print(f'attention mask shape: {attention_mask.shape}')
