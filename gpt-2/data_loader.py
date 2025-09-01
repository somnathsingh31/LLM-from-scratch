import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the text
        tokens = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        #use sliding window to chunck text
        for i in range(0, len(tokens) - max_length, stride):
            input_chunks = tokens[i:i+max_length]
            target_chunks = tokens[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunks))
            self.target_ids.append(torch.tensor(target_chunks))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

def create_dataloders(text_data, batch_size, max_length, stride, num_workers=0, train_ratio=None, requires_val=False):
    # Using a pre-trained tokenizer (custom one is untrained)
    tokenizer = tiktoken.get_encoding('gpt2')

    if requires_val and train_ratio:
        split_idx = int(train_ratio * len(text_data))
        train_data = text_data[:split_idx]
        val_data= text_data[split_idx:]
        train_dataset = GPTDatasetV1(train_data, tokenizer, max_length, stride)
        val_dataset = GPTDatasetV1(val_data, tokenizer, max_length, stride)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        train_dataset = GPTDatasetV1(text_data, tokenizer, max_length, stride)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = None
    return train_loader, val_loader

