from typing import List, Dict, Any, Union
import arrow
import logging
import torch
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from helpers.helper import get_root_directory, cast_csv_to_dict, load_json
from student import StudentModel
from query_pairs_dataset import QueryTextDataset, QueryTextLoader


def save_model(save_path: str, epoch: int, model: StudentModel):
    import os
    os.makedirs(save_path, exist_ok=True)
    model_name = f"query_text_model_epoch_{epoch}_{arrow.now().format('YYYY-MM-DD-HH-mm')}.pt"
    torch.save(model.state_dict(), os.path.join(save_path, model_name))
    logging.info(f"Model saved at {os.path.join(save_path, model_name)}")


if __name__ == "__main__":
    dataset = f'{get_root_directory()}/data/zara/parsed_examples/all_zara_products_deduped_clened_from_nonsib.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    student_model = StudentModel(f'{get_root_directory()}/models/clip_multilingual_text',
                                 projection_dim=512)
    student_model.load_state_dict(
        torch.load(f'{get_root_directory()}/models/trained/student_model_epoch_9_2024-12-09-18-15.pt'))
    student_model.to(device)
    student_model.train()
    student_tokenizer = student_model.tokenizer

    dataset = QueryTextDataset(dataset=dataset, tokenizer=student_tokenizer)
    dataloader = QueryTextLoader(dataset, tokenizer=student_tokenizer)

    criterion = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-5)
    num_epochs = 20

    for epoch in range(num_epochs):
        for batch in dataloader():
            query_inputs, text_inputs = batch['query'], batch['text']

            query_inputs = {k: v.squeeze(1).to(device) for k, v in query_inputs.items()}
            text_inputs = {k: v.squeeze(1).to(device) for k, v in text_inputs.items()}

            # Get embeddings
            query_embeddings = student_model(**query_inputs).mean(dim=1).unsqueeze(-1)
            text_embeddings = student_model(**text_inputs).mean(dim=1).unsqueeze(-1)

            # Labels: 1 for similar pairs
            labels = torch.ones(query_embeddings.size(0)).to(device)


            # Compute loss
            loss = criterion(query_embeddings, text_embeddings , labels)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    save_model(f'{get_root_directory()}/models/trained/',
               epoch=num_epochs,
               model=student_model)
