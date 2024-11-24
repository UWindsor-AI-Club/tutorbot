import tiktoken

import torch
from torch.utils.data import Dataset, DataLoader

class Tokenizer:
    def __init__(self, raw_text=None, batch_size=4, max_length=256,
                    stride=128, output_dim=256):
        self.batch_size = batch_size
        self.max_length = max_length
        self.stride     = stride
        self.output_dim = output_dim

        # Initialize the tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab

        if raw_text is not None:
            self.tokenize()
        
    def embeddings(self):
        return self.input_embeddings
    
    def tokenize(self, raw_text):
        self.data_loader = self._create_dataloader(raw_text, self.tokenizer, shuffle=False)
        self.input_embeddings = self._create_input_embeddings()
        return self.input_embeddings

    def _create_dataloader(self, txt, tokenizer, shuffle=True, drop_last=True, num_workers=0):
        # Create dataset
        dataset = GPTDatasetV1(txt, tokenizer, self.max_length, self.stride)

        # Create dataloader
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )
    
    def _create_input_embeddings(self):
        data_iter = iter(self.data_loader)
        inputs, targets = next(data_iter)

        token_embedding_layer = torch.nn.Embedding(self.vocab_size, self.output_dim)

        token_embeddings = token_embedding_layer(inputs)

        context_length = self.max_length
        pos_embedding_layer = torch.nn.Embedding(context_length, self.output_dim)
        pos_embeddings = pos_embedding_layer(torch.arange(self.max_length))

        return token_embeddings + pos_embeddings

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]