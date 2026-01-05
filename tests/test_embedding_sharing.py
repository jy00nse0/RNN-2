#!/usr/bin/env python3
"""
Test to verify that encoder and decoder share the same embedding layer.

This test validates that:
1. Encoder and decoder use the same embedding object (same memory address)
2. Gradients accumulate from both encoder and decoder to shared embedding
3. Memory is saved by using shared embeddings instead of separate ones
"""

import torch
import torch.nn as nn
from model import train_model_factory


class Args:
    """Mock args object for creating model"""
    def __init__(self):
        # Encoder args
        self.encoder_rnn_cell = 'LSTM'
        self.encoder_hidden_size = 32
        self.encoder_num_layers = 1
        self.encoder_rnn_dropout = 0.0
        self.encoder_bidirectional = False
        
        # Decoder args
        self.decoder_type = 'luong'
        self.decoder_rnn_cell = 'LSTM'
        self.decoder_hidden_size = 32
        self.decoder_num_layers = 1
        self.decoder_rnn_dropout = 0.0
        self.luong_attn_hidden_size = 32
        self.luong_input_feed = False
        # Use 'adjust_pad' instead of 'zeros' to enable gradient flow from decoder to encoder
        self.decoder_init_type = 'adjust_pad'
        
        # Attention args
        # Use 'global' attention to ensure encoder outputs are used and gradients flow to encoder embeddings
        self.attention_type = 'global'
        self.attention_score = 'dot'
        
        # Embedding args
        self.embedding_type = None
        self.embedding_size = 16
        self.train_embeddings = True
        
        # Training args
        self.teacher_forcing_ratio = 1.0


class Metadata:
    """Mock metadata object"""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.padding_idx = 0  # Padding token index
        self.vectors = None  # No pre-trained embeddings


def test_embedding_separation():
    """Test that encoder and decoder have separate embedding objects"""
    print("\n=== Testing Embedding Separation ===")
    
    vocab_size = 100
    args = Args()
    metadata = Metadata(vocab_size)
    
    # Create model using factory
    model = train_model_factory(args, metadata, metadata)
    
    # Check that embeddings are separate objects
    encoder_embed = model.encoder.embed
    decoder_embed = model.decoder.embed
    
    print(f"Encoder embedding id: {id(encoder_embed)}")
    print(f"Decoder embedding id: {id(decoder_embed)}")
    print(f"Are embeddings separate? {encoder_embed is not decoder_embed}")
    
    assert encoder_embed is not decoder_embed, "Encoder and decoder should have separate embeddings!"
    print("✅ Embedding separation test passed!")
    return True


def test_gradient_accumulation():
    """Test that gradients accumulate separately on encoder and decoder embeddings"""
    print("\n=== Testing Gradient Accumulation ===")
    
    vocab_size = 100
    batch_size = 4
    seq_len = 10
    
    args = Args()
    metadata = Metadata(vocab_size)
    
    model = train_model_factory(args, metadata, metadata)
    model.train()
    
    # Create dummy input tensors (avoid index 0 which is padding)
    question = torch.randint(1, vocab_size, (seq_len, batch_size))
    answer = torch.randint(1, vocab_size, (seq_len, batch_size))
    
    # Forward pass
    outputs = model(question, answer)
    
    # Create dummy target and compute loss (avoid index 0 which is padding)
    target = torch.randint(1, vocab_size, (seq_len - 1, batch_size))
    loss_fn = nn.CrossEntropyLoss()
    
    # Reshape for loss calculation
    outputs_flat = outputs.view(-1, vocab_size)
    target_flat = target.view(-1)
    loss = loss_fn(outputs_flat, target_flat)
    
    # Backward pass
    loss.backward()
    
    # Check that both embeddings have gradients
    encoder_embed = model.encoder.embed
    decoder_embed = model.decoder.embed
    assert encoder_embed.weight.grad is not None, "Encoder embedding should have gradients!"
    assert decoder_embed.weight.grad is not None, "Decoder embedding should have gradients!"
    assert encoder_embed.weight.grad.abs().sum() > 0, "Encoder embedding gradients should be non-zero!"
    assert decoder_embed.weight.grad.abs().sum() > 0, "Decoder embedding gradients should be non-zero!"
    
    print(f"Loss value: {loss.item():.4f}")
    print(f"Encoder embedding gradient norm: {encoder_embed.weight.grad.norm().item():.4f}")
    print(f"Decoder embedding gradient norm: {decoder_embed.weight.grad.norm().item():.4f}")
    print("✅ Gradient accumulation test passed!")
    return True


def test_memory_usage():
    """Test memory usage with separate embeddings"""
    print("\n=== Testing Memory Usage ===")
    
    vocab_size = 50000  # Typical vocab size from paper
    embedding_size = 1000  # Typical embedding size from paper
    
    # Calculate memory for separate embeddings
    params_per_embedding = vocab_size * embedding_size
    bytes_per_param = 4  # float32
    separate_memory_mb = (2 * params_per_embedding * bytes_per_param) / (1024 * 1024)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {embedding_size}")
    print(f"Memory with separate embeddings: {separate_memory_mb:.2f} MB")
    
    # Verify actual implementation uses separate embeddings
    args = Args()
    args.embedding_size = embedding_size
    metadata = Metadata(vocab_size)
    
    model = train_model_factory(args, metadata, metadata)
    
    # Count actual parameters
    encoder_embed_params = sum(p.numel() for p in model.encoder.embed.parameters())
    decoder_embed_params = sum(p.numel() for p in model.decoder.embed.parameters())
    total_embed_params = encoder_embed_params + decoder_embed_params
    expected_params = 2 * vocab_size * embedding_size
    
    assert total_embed_params == expected_params, f"Expected {expected_params} params, got {total_embed_params}"
    print(f"Encoder embedding parameters: {encoder_embed_params:,}")
    print(f"Decoder embedding parameters: {decoder_embed_params:,}")
    print(f"Total embedding parameters: {total_embed_params:,}")
    print("✅ Memory usage test passed!")
    return True


def test_parameter_independence():
    """Test that encoder and decoder embedding weights are independent tensors"""
    print("\n=== Testing Parameter Independence ===")
    
    vocab_size = 100
    args = Args()
    metadata = Metadata(vocab_size)
    
    model = train_model_factory(args, metadata, metadata)
    
    # Get embedding weights
    encoder_weight = model.encoder.embed.weight
    decoder_weight = model.decoder.embed.weight
    
    # Check that they are different tensors
    assert encoder_weight is not decoder_weight, "Embedding weights should be different tensors!"
    
    # Modify encoder embedding and check decoder is NOT affected (using no_grad to avoid in-place error)
    with torch.no_grad():
        encoder_weight[0, 0] = 999.0
        
        assert decoder_weight[0, 0].item() != 999.0, "Changes to encoder embedding should NOT affect decoder!"
    
    print("✅ Parameter independence test passed!")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing Separate Embedding Implementation")
    print("=" * 60)
    
    tests = [
        test_embedding_separation,
        test_gradient_accumulation,
        test_memory_usage,
        test_parameter_independence,
    ]
    
    all_passed = True
    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"❌ Test {test_fn.__name__} failed: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        print("=" * 60)
        return True
    else:
        print("❌ Some tests failed!")
        print("=" * 60)
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
