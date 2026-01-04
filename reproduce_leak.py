
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import argparse
from model import train_model_factory
from util import Metadata
import gc
import os
import sys

def get_memory_usage():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        ram = process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        ram = 0
        
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024 / 1024
        vram_reserved = torch.cuda.memory_reserved() / 1024 / 1024
    else:
        vram = 0
        vram_reserved = 0
    return ram, vram, vram_reserved

class RandomDataset(Dataset):
    def __init__(self, vocab_size, length, count):
        self.vocab_size = vocab_size
        self.length = length
        self.count = count
        
    def __len__(self):
        return self.count
        
    def __getitem__(self, idx):
        src = torch.randint(0, self.vocab_size, (self.length,))
        tgt = torch.randint(0, self.vocab_size, (self.length,))
        return src, tgt

class Batch:
    def __init__(self, src, tgt):
        self.question = src
        self.answer = tgt

def collate_fn(batch):
    src, tgt = zip(*batch)
    src = torch.stack(src).transpose(0, 1) # (len, batch)
    tgt = torch.stack(tgt).transpose(0, 1)
    return Batch(src, tgt)

class MockArgs:
    def __init__(self):
        self.encoder_rnn_cell = 'LSTM'
        self.encoder_hidden_size = 1000
        self.encoder_num_layers = 4
        self.encoder_rnn_dropout = 0.0
        self.encoder_bidirectional = False
        self.embedding_size = 1000
        
        self.decoder_type = 'luong'
        self.decoder_rnn_cell = 'LSTM'
        self.decoder_hidden_size = 1000
        self.decoder_num_layers = 4
        self.decoder_rnn_dropout = 0.0
        self.luong_attn_hidden_size = 1000
        self.luong_input_feed = False
        self.decoder_init_type = 'adjust_pad'
        
        self.attention_type = 'global'
        self.attention_score = 'dot'
        self.half_window_size = 10
        self.local_p_hidden_size = 1000 
        self.concat_attention_hidden_size = 1000
        
        self.teacher_forcing_ratio = 1.0
        self.embedding_type = None
        self.train_embeddings = True
        
        self.num_workers = 4 
        self.amp = True 

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    args = MockArgs()
    
    vocab_size = 50000 
    src_metadata = Metadata(vocab_size=vocab_size, padding_idx=0, vectors=None)
    tgt_metadata = Metadata(vocab_size=vocab_size, padding_idx=0, vectors=None)

    print("Building model...")
    model = train_model_factory(args, src_metadata, tgt_metadata).to(device)
    model.train()
    
    optimizer = optim.SGD(model.parameters(), lr=1.0)
    scaler = GradScaler(enabled=args.amp)
    
    dataset = RandomDataset(vocab_size, 50, 1280) # 10 batches of 128
    loader = DataLoader(dataset, batch_size=128, num_workers=args.num_workers, collate_fn=collate_fn)
    
    print("Starting training loop simulation...")
    
    ram_start, vram_start, _ = get_memory_usage()
    print(f"Start - RAM: {ram_start:.2f} MB, VRAM: {vram_start:.2f} MB")
    
    for epoch in range(5):
        print(f"Epoch {epoch}")
        for i, batch in enumerate(loader):
            question = batch.question.to(device)
            answer = batch.answer.to(device)
            
            with autocast(enabled=args.amp):
                # We need to simulate the model.forward logic
                # The model internally does pre-allocation
                logits = model(question, answer)
                
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    answer[1:].reshape(-1),
                    ignore_index=0
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad(set_to_none=True)
            
            if i % 2 == 0:
                ram, vram, _ = get_memory_usage()
                print(f"  Batch {i} - VRAM: {vram:.2f} MB (Delta: {vram - vram_start:.2f} MB)")
                
    print("Done.")

if __name__ == "__main__":
    main()
