from typing import List, Dict, Any, Union

import torch
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from student import StudentModel
from helpers.helper import cast_csv_to_dict, load_json, get_root_directory


class QueryTextDataset(Dataset):
    def __init__(self, dataset: Union[List[Dict[str, str]], str], tokenizer):
        if isinstance(dataset, str):
            if dataset.endswith('.csv'):
                self.dataset = cast_csv_to_dict(dataset)
            if dataset.endswith('.json'):
                self.dataset = load_json(dataset)
        else:
            self.dataset = dataset

        self.queries = [x['Product Name'] for x in self.dataset]
        self.texts = [x['Description'] for x in self.dataset]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        text = self.texts[idx]
        query_inputs = self.tokenizer(query,
                                      return_tensors='pt',
                                      padding=True,
                                      truncation=True)
        text_inputs = self.tokenizer(text,
                                     return_tensors='pt',
                                     padding=True,
                                     truncation=True)
        return query_inputs, text_inputs


class QueryTextCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        query = [item[0]['input_ids'].squeeze(0) for item in batch]
        text = [item[1]['input_ids'].squeeze(0) for item in batch]

        query_padded = pad_sequence(query, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        text_padded = pad_sequence(text, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        query_attention_masks = (query_padded != self.tokenizer.pad_token_id).long()
        text_attention_masks = (text_padded != self.tokenizer.pad_token_id).long()

        return {'query': {
            'input_ids': query_padded,
            'attention_mask': query_attention_masks,
        },
            'text': {
                'input_ids': text_padded,
                'attention_mask': text_attention_masks,
            }}


class QueryTextLoader:
    def __init__(self, dataset: Union[QueryTextDataset, str], tokenizer):
        self.tokenizer = tokenizer

        if isinstance(dataset, str):
            self.dataset = QueryTextDataset(tokenizer=self.tokenizer,
                                            dataset=dataset)
        else:
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __call__(self):
        return DataLoader(dataset=self.dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=QueryTextCollate(tokenizer=self.tokenizer))




if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    student_text_model = StudentModel(f'{get_root_directory()}/models/clip_multilingual_text', projection_dim=512)
    student_text_model.to(device)
    student_text_model.eval()
    student_tokenizer = student_text_model.tokenizer

    query_text_data = QueryTextDataset(tokenizer=student_tokenizer,
                                            dataset=f'{get_root_directory()}/data/zara/parsed_examples/all_zara_products_deduped_clened_from_nonsib.csv')
    query_text_data_dataloader = QueryTextLoader(dataset=query_text_data, tokenizer=student_tokenizer)

    for batch in query_text_data_dataloader():
        print(batch)
