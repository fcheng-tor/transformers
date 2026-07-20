from typing import Any
import torch
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset 
from torch.utils.data import random_split

FILENAME = "tiny_shakespeare.txt"
TRAIN_SPLIT = 0.9

class TinyShakespeareDataset(Dataset):
    def __init__(self) -> None:
        with open(FILENAME, 'r', encoding='utf-8') as f:
            self.text = f.read()
            self.chars = sorted(list[str](set[str](self.text)))
        self.string_to_int = {ch:i for i, ch in enumerate(self.chars)}

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index]
    
    def vocab_size(self):
        """
        Number of characters in vocabulary
        """
        return len(self.chars)
    
    def encode(self, chars):
        """
        Go from characters to integer encodings.
        """
        return [self.string_to_int[ch] for ch in chars]

    def decode(self, code):
        """
        Go from integer encodings to characters
        """
        return "".join([self.chars[i] for i in code])


class TinyShakespeareDataLoader():
    """
    Dataloader built with the TinyShakespeareDataSet.
    """
    def __init__(self, batch_size, num_workers=1):
        dataset = TinyShakespeareDataset()
        dataset_size = len(dataset)
        train_size = int(dataset_size * TRAIN_SPLIT)
        eval_size = dataset_size - train_size
        print(dataset.vocab_size())
        self._vocab_size = dataset.vocab_size()
        self.train_dataset, self.eval_dataset = random_split(
            dataset=dataset,
            lengths=[train_size, eval_size],
            generator=torch.Generator().manual_seed(42)
        )
        self._train_dataloader = DataLoader[Any](
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self._eval_dataloader = DataLoader[Any](
            dataset=self.eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def train_dataloader(self):
        return self._train_dataloader

    def eval_dataloader(self):
        return self._eval_dataloader
    
    def vocab_size(self):
        return self._vocab_size

