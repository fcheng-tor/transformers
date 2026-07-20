# from normalization.layer_norm import LayerNorm
import torch
import torch.nn as nn
from dataloader import TinyShakespeareDataLoader
# import torch.nn.functional as F

TRAIN_STEPS = 10000
BATCH_SIZE = 8
NUM_HEADS = 4
EMBEDDING_DIM = 32
SEQ_LENGTH = 8

class AttentionBlock(nn.Module):
    pass


class MultiHeadAttention(nn.Module):
    pass


if __name__ == "__main__":
    dataloader = TinyShakespeareDataLoader(batch_size=BATCH_SIZE, num_workers=2)
    print(dataloader.vocab_size())
    print("Initializing model.")

