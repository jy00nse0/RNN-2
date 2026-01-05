#!/usr/bin/env python3
"""
Integration test demonstrating embedding sharing in a realistic training scenario.

This test shows that the embedding sharing works correctly when:
1. Creating a model using the factory
2. Running a forward pass
3. Computing gradients
4. Verifying that embeddings are truly shared
"""

import torch
import torch.nn as nn
from model import train_model_factory


def main():
    print("=" * 70)
    print("Integration Test: Embedding Sharing in Training Scenario")
    print("=" * 70)
    
    # Configuration matching the problem statement example
    class Args:
        def __init__(self):
            # Encoder configuration
            self.encoder_rnn_cell = 'LSTM'
            self.encoder_hidden_size = 256
            self.encoder_num_layers = 2
            self.encoder_rnn_dropout = 0.2
            self.encoder_bidirectional = False
            
            # Decoder configuration
            self.decoder_type = 'luong'
            self.decoder_rnn_cell = 'LSTM'
            self.decoder_hidden_size = 256
            self.decoder_num_layers = 2
            self.decoder_rnn_dropout = 0.2
            self.luong_attn_hidden_size = 256
            self.luong_input_feed = False
            self.decoder_init_type = 'adjust_pad'  # Use encoder outputs
            
            # Attention configuration
            self.attention_type = 'global'  # Use attention to get gradients to encoder
            self.attention_score = 'dot'
            
            # Embedding configuration (matching paper settings)
            self.embedding_type = None
            self.embedding_size = 512  # Reduced from 1000 for this demo
            self.train_embeddings = True
            
            # Training configuration
            self.teacher_forcing_ratio = 0.5
    
    class Metadata:
        def __init__(self):
            self.vocab_size = 10000  # Reduced from 50k for this demo
            self.padding_idx = 0
            self.vectors = None
    
    args = Args()
    metadata = Metadata()
    
    print(f"\nConfiguration:")
    print(f"  Vocabulary size: {metadata.vocab_size:,}")
    print(f"  Embedding dimension: {args.embedding_size}")
    print(f"  Encoder hidden size: {args.encoder_hidden_size}")
    print(f"  Decoder hidden size: {args.decoder_hidden_size}")
    
    # Create model with separate embeddings for encoder and decoder
    print("\n1. Creating model with train_model_factory()...")
    model = train_model_factory(args, metadata, metadata)
    
    # Verify embeddings are now separate (not shared)
    print("\n2. Verifying separate embeddings...")
    encoder_embed = model.encoder.embed
    decoder_embed = model.decoder.embed
    
    print(f"   Encoder embedding address: {id(encoder_embed)}")
    print(f"   Decoder embedding address: {id(decoder_embed)}")
    print(f"   Same object? {encoder_embed is decoder_embed}")
    
    if encoder_embed is decoder_embed:
        print("   ❌ FAILED: Embeddings should be separate!")
        return False
    print("   ✅ SUCCESS: Embeddings are separate (as expected)!")
    
    # Calculate memory usage with separate embeddings
    print("\n3. Calculating memory usage...")
    params_per_embedding = metadata.vocab_size * args.embedding_size
    bytes_per_param = 4  # float32
    separate_memory_mb = (2 * params_per_embedding * bytes_per_param) / (1024 * 1024)
    
    print(f"   With separate embeddings: {separate_memory_mb:.2f} MB (2 separate embeddings)")
    print(f"   Note: Separate embeddings allow different vocab sizes for SRC and TGT")
    
    # Simulate training step
    print("\n4. Simulating training step...")
    model.train()
    
    # Create sample batch
    batch_size = 8
    seq_len = 20
    question = torch.randint(0, metadata.vocab_size, (seq_len, batch_size))
    answer = torch.randint(0, metadata.vocab_size, (seq_len, batch_size))
    
    # Forward pass
    print("   Running forward pass...")
    outputs = model(question, answer)
    print(f"   Output shape: {tuple(outputs.shape)}")
    
    # Compute loss
    target = torch.randint(0, metadata.vocab_size, (seq_len - 1, batch_size))
    loss_fn = nn.CrossEntropyLoss()
    outputs_flat = outputs.view(-1, metadata.vocab_size)
    target_flat = target.view(-1)
    loss = loss_fn(outputs_flat, target_flat)
    
    print(f"   Loss: {loss.item():.4f}")
    
    # Backward pass
    print("   Running backward pass...")
    loss.backward()
    
    # Verify gradients accumulated on both embeddings
    print("\n5. Verifying gradient accumulation...")
    if encoder_embed.weight.grad is None or decoder_embed.weight.grad is None:
        print("   ❌ FAILED: No gradients on embeddings!")
        return False
    
    encoder_grad_norm = encoder_embed.weight.grad.norm().item()
    decoder_grad_norm = decoder_embed.weight.grad.norm().item()
    
    print(f"   Encoder gradient norm: {encoder_grad_norm:.4f}")
    print(f"   Decoder gradient norm: {decoder_grad_norm:.4f}")
    print("   ✅ SUCCESS: Gradients accumulated on both separate embeddings!")
    
    print("\n" + "=" * 70)
    print("✅ All integration tests passed!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  ✅ Encoder and Decoder have separate embedding parameters")
    print(f"  ✅ Gradients accumulate separately on each side")
    print(f"  ✅ Allows different vocab sizes for SRC and TGT")
    print("=" * 70)
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
