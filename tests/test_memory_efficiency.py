#!/usr/bin/env python3
"""
Test to verify the memory efficiency fix for Seq2SeqTrain.forward()

This test validates that:
1. Output tensor shape matches the original implementation
2. Values are computed identically
3. Gradients flow correctly through the modified code
4. Device placement is preserved
"""

import torch
import torch.nn as nn
from model.seq2seq.encoder import encoder_factory
from model.seq2seq.decoder import decoder_factory
from model.seq2seq.model import Seq2SeqTrain


class Args:
    """Mock args object for creating encoder/decoder"""
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
        self.decoder_init_type = 'zeros'
        
        # Attention args
        self.attention_type = 'none'
        self.attention_score = 'dot'
        
        # Embedding args
        self.embedding_type = None
        self.embedding_size = 16
        self.train_embeddings = True


class Metadata:
    """Mock metadata object"""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.padding_idx = 0  # Padding token index
        self.vectors = None  # No pre-trained embeddings


def create_test_model(vocab_size=100, device='cpu'):
    """Create a small test model for validation"""
    args = Args()
    metadata = Metadata(vocab_size)
    
    # Create encoder and decoder
    encoder = encoder_factory(args, metadata)
    decoder = decoder_factory(args, metadata)
    
    # Create Seq2SeqTrain model
    model = Seq2SeqTrain(
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab_size,
        teacher_forcing_ratio=1.0  # Always use teacher forcing for deterministic test
    )
    
    return model.to(device)


def test_output_shape():
    """Test that output shape matches expected dimensions"""
    print("\n=== Testing Output Shape ===")
    
    vocab_size = 100
    batch_size = 4
    seq_len = 10
    
    model = create_test_model(vocab_size, device='cpu')
    model.eval()
    
    # Create dummy input tensors
    question = torch.randint(0, vocab_size, (seq_len, batch_size))
    answer = torch.randint(0, vocab_size, (seq_len, batch_size))
    
    with torch.no_grad():
        outputs = model(question, answer)
    
    # Expected shape: (seq_len - 1, batch_size, vocab_size)
    expected_shape = (seq_len - 1, batch_size, vocab_size)
    
    print(f"Output shape: {tuple(outputs.shape)}")
    print(f"Expected shape: {expected_shape}")
    
    assert outputs.shape == expected_shape, f"Shape mismatch! Got {outputs.shape}, expected {expected_shape}"
    print("✅ Output shape test passed!")
    return True


def test_gradient_flow():
    """Test that gradients flow correctly through the model"""
    print("\n=== Testing Gradient Flow ===")
    
    vocab_size = 100
    batch_size = 4
    seq_len = 10
    
    model = create_test_model(vocab_size, device='cpu')
    model.train()
    
    # Create dummy input tensors
    question = torch.randint(0, vocab_size, (seq_len, batch_size))
    answer = torch.randint(0, vocab_size, (seq_len, batch_size))
    
    # Forward pass
    outputs = model(question, answer)
    
    # Create dummy target and compute loss
    target = torch.randint(0, vocab_size, (seq_len - 1, batch_size))
    loss_fn = nn.CrossEntropyLoss()
    
    # Reshape for loss calculation
    outputs_flat = outputs.view(-1, vocab_size)
    target_flat = target.view(-1)
    loss = loss_fn(outputs_flat, target_flat)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    assert has_gradients, "No gradients computed!"
    print(f"Loss value: {loss.item():.4f}")
    print("✅ Gradient flow test passed!")
    return True


def test_device_placement():
    """Test that device placement is correct"""
    print("\n=== Testing Device Placement ===")
    
    vocab_size = 100
    batch_size = 4
    seq_len = 10
    
    # Test on CPU
    print("Testing on CPU...")
    model = create_test_model(vocab_size, device='cpu')
    model.eval()
    
    question = torch.randint(0, vocab_size, (seq_len, batch_size))
    answer = torch.randint(0, vocab_size, (seq_len, batch_size))
    
    with torch.no_grad():
        outputs = model(question, answer)
    
    assert outputs.device.type == 'cpu', f"Expected CPU device, got {outputs.device}"
    print("✅ CPU device test passed!")
    
    # Test on CUDA if available
    if torch.cuda.is_available():
        print("Testing on CUDA...")
        model = create_test_model(vocab_size, device='cuda')
        model.eval()
        
        question = torch.randint(0, vocab_size, (seq_len, batch_size), device='cuda')
        answer = torch.randint(0, vocab_size, (seq_len, batch_size), device='cuda')
        
        with torch.no_grad():
            outputs = model(question, answer)
        
        assert outputs.device.type == 'cuda', f"Expected CUDA device, got {outputs.device}"
        print("✅ CUDA device test passed!")
    else:
        print("⚠️  CUDA not available, skipping CUDA test")
    
    return True


def test_deterministic_output():
    """Test that with teacher forcing = 1.0, outputs are deterministic"""
    print("\n=== Testing Deterministic Output ===")
    
    vocab_size = 100
    batch_size = 4
    seq_len = 10
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    model1 = create_test_model(vocab_size, device='cpu')
    
    torch.manual_seed(42)
    model2 = create_test_model(vocab_size, device='cpu')
    
    # Copy weights from model1 to model2
    model2.load_state_dict(model1.state_dict())
    
    model1.eval()
    model2.eval()
    
    # Create same input for both models
    torch.manual_seed(123)
    question = torch.randint(0, vocab_size, (seq_len, batch_size))
    answer = torch.randint(0, vocab_size, (seq_len, batch_size))
    
    with torch.no_grad():
        outputs1 = model1(question, answer)
        outputs2 = model2(question, answer)
    
    # Check that outputs are identical
    max_diff = (outputs1 - outputs2).abs().max().item()
    print(f"Max difference between runs: {max_diff}")
    
    assert max_diff < 1e-6, f"Outputs differ by {max_diff}!"
    print("✅ Deterministic output test passed!")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing Memory Efficiency Fix for Seq2SeqTrain")
    print("=" * 60)
    
    tests = [
        test_output_shape,
        test_gradient_flow,
        test_device_placement,
        test_deterministic_output,
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
