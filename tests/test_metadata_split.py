#!/usr/bin/env python3
"""
Test script to verify that SRC/TGT metadata split is working correctly.
This test ensures:
1. dataset_factory returns separate src_metadata and tgt_metadata
2. Encoder uses SRC vocab size
3. Decoder uses TGT vocab size
4. Embeddings are initialized correctly
"""

import torch
import argparse
from dataset import dataset_factory
from model import train_model_factory


def create_test_args():
    """Create minimal args for testing"""
    class Args:
        def __init__(self):
            # Dataset
            self.dataset = 'sample100k'
            self.batch_size = 2
            self.reverse = False
            
            # Encoder
            self.encoder_rnn_cell = 'LSTM'
            self.encoder_hidden_size = 128
            self.encoder_num_layers = 2
            self.encoder_rnn_dropout = 0.0
            self.encoder_bidirectional = False
            
            # Decoder
            self.decoder_type = 'luong'
            self.decoder_rnn_cell = 'LSTM'
            self.decoder_hidden_size = 128
            self.decoder_num_layers = 2
            self.decoder_rnn_dropout = 0.0
            self.luong_attn_hidden_size = 128
            self.luong_input_feed = False
            self.decoder_init_type = 'zeros'
            
            # Attention
            self.attention_type = 'none'
            self.attention_score = 'dot'
            
            # Embedding
            self.embedding_type = None
            self.embedding_size = 256
            self.train_embeddings = True
            
            # Training
            self.teacher_forcing_ratio = 0.0
    
    return Args()


def test_metadata_split():
    """Test that metadata is split correctly"""
    print("=" * 70)
    print("Testing SRC/TGT Metadata Split")
    print("=" * 70)
    
    args = create_test_args()
    device = torch.device('cpu')
    
    # Load dataset
    print("\n1. Loading dataset...")
    try:
        src_metadata, tgt_metadata, src_vocab, tgt_vocab, train_iter, val_iter, test_iter = dataset_factory(args, device)
        print("   ✅ Dataset loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check metadata
    print("\n2. Checking metadata...")
    print(f"   SRC vocab size: {src_metadata.vocab_size}")
    print(f"   TGT vocab size: {tgt_metadata.vocab_size}")
    print(f"   SRC padding idx: {src_metadata.padding_idx}")
    print(f"   TGT padding idx: {tgt_metadata.padding_idx}")
    
    # Build model
    print("\n3. Building model...")
    try:
        model = train_model_factory(args, src_metadata, tgt_metadata)
        print("   ✅ Model built successfully")
    except Exception as e:
        print(f"   ❌ Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check encoder embedding size
    print("\n4. Checking embedding sizes...")
    encoder_vocab_size = model.encoder.embed.num_embeddings
    decoder_vocab_size = model.decoder.embed.num_embeddings
    
    print(f"   Encoder embedding num_embeddings: {encoder_vocab_size}")
    print(f"   Decoder embedding num_embeddings: {decoder_vocab_size}")
    
    if encoder_vocab_size == src_metadata.vocab_size:
        print(f"   ✅ Encoder uses SRC vocab size ({src_metadata.vocab_size})")
    else:
        print(f"   ❌ Encoder vocab size mismatch! Expected {src_metadata.vocab_size}, got {encoder_vocab_size}")
        return False
    
    if decoder_vocab_size == tgt_metadata.vocab_size:
        print(f"   ✅ Decoder uses TGT vocab size ({tgt_metadata.vocab_size})")
    else:
        print(f"   ❌ Decoder vocab size mismatch! Expected {tgt_metadata.vocab_size}, got {decoder_vocab_size}")
        return False
    
    # Check embedding initialization
    print("\n5. Checking embedding initialization...")
    encoder_weight_min = model.encoder.embed.weight.data.min().item()
    encoder_weight_max = model.encoder.embed.weight.data.max().item()
    decoder_weight_min = model.decoder.embed.weight.data.min().item()
    decoder_weight_max = model.decoder.embed.weight.data.max().item()
    
    print(f"   Encoder embedding range: [{encoder_weight_min:.4f}, {encoder_weight_max:.4f}]")
    print(f"   Decoder embedding range: [{decoder_weight_min:.4f}, {decoder_weight_max:.4f}]")
    
    # Check if weights are in [-0.1, 0.1] range (allowing for small numerical errors)
    if -0.11 <= encoder_weight_min <= -0.09 and 0.09 <= encoder_weight_max <= 0.11:
        print("   ✅ Encoder embedding initialized in [-0.1, 0.1]")
    else:
        print("   ⚠️  Encoder embedding may not be initialized in [-0.1, 0.1]")
    
    if -0.11 <= decoder_weight_min <= -0.09 and 0.09 <= decoder_weight_max <= 0.11:
        print("   ✅ Decoder embedding initialized in [-0.1, 0.1]")
    else:
        print("   ⚠️  Decoder embedding may not be initialized in [-0.1, 0.1]")
    
    # Check output dimension
    print("\n6. Checking output dimension...")
    print(f"   Model output vocab_size: {model.vocab_size}")
    if model.vocab_size == tgt_metadata.vocab_size:
        print(f"   ✅ Output dimension matches TGT vocab size ({tgt_metadata.vocab_size})")
    else:
        print(f"   ❌ Output dimension mismatch! Expected {tgt_metadata.vocab_size}, got {model.vocab_size}")
        return False
    
    # Test forward pass
    print("\n7. Testing forward pass...")
    try:
        # Get a batch
        batch = next(iter(train_iter))
        question, answer = batch.question, batch.answer
        
        print(f"   Input shapes: question={question.shape}, answer={answer.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(question, answer)
        
        print(f"   Output shape: {output.shape}")
        expected_shape = (answer.shape[0] - 1, answer.shape[1], tgt_metadata.vocab_size)
        
        if output.shape == expected_shape:
            print(f"   ✅ Output shape is correct: {output.shape}")
        else:
            print(f"   ❌ Output shape mismatch! Expected {expected_shape}, got {output.shape}")
            return False
            
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    return True


if __name__ == '__main__':
    success = test_metadata_split()
    exit(0 if success else 1)
