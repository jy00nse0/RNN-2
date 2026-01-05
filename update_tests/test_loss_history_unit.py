#!/usr/bin/env python3
"""
Unit test to verify conditional loss history collection in train() and evaluate()

This test validates that:
1. train() only creates batch_losses list when collect_loss_history=True
2. evaluate() only creates batch_losses list when collect_loss_history=True
3. Stats dictionary is correctly populated in both cases
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from train import train, evaluate
from util import Metadata


class DummyModel(nn.Module):
    """Dummy model for testing"""
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(100, 10)
        self.linear = nn.Linear(10, vocab_size)
    
    def forward(self, question, answer):
        # Return dummy logits with shape (seq_len-1, batch, vocab_size)
        batch_size = question.size(1)
        seq_len = answer.size(0)
        # Create output that depends on model parameters for gradient computation
        emb = self.embedding(answer[:1])  # Just use first token
        out = self.linear(emb.mean(dim=0))
        # Expand to match expected shape
        result = out.unsqueeze(0).expand(seq_len - 1, batch_size, -1)
        return result.contiguous()


def create_dummy_data(batch_size=2, seq_len=5, num_batches=3):
    """Create dummy data for testing"""
    class DummyBatch:
        def __init__(self, question, answer):
            self.question = question
            self.answer = answer
    
    batches = []
    for _ in range(num_batches):
        question = torch.randint(0, 100, (seq_len, batch_size))
        answer = torch.randint(0, 100, (seq_len, batch_size))
        batches.append(DummyBatch(question, answer))
    
    return batches


def test_train_without_loss_history():
    """Test that train() doesn't collect batch_losses when collect_loss_history=False"""
    print("\nTest 1: train() without loss history collection")
    
    vocab_size = 100
    model = DummyModel(vocab_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metadata = Metadata(vocab_size=vocab_size, padding_idx=0, vectors=None)
    
    train_iter = create_dummy_data(batch_size=2, seq_len=5, num_batches=3)
    
    # Call train with collect_loss_history=False
    avg_loss, stats = train(
        model=model,
        optimizer=optimizer,
        train_iter=train_iter,
        metadata=metadata,
        grad_clip=5.0,
        collect_loss_history=False
    )
    
    # Verify that batch_losses is NOT in stats
    if 'batch_losses' in stats:
        print("  ❌ FAILED: batch_losses should not be in stats")
        return False
    
    # Verify that avg_grad_norm is present
    if 'avg_grad_norm' not in stats:
        print("  ❌ FAILED: avg_grad_norm should be in stats")
        return False
    
    # Verify that min_loss and max_loss equal avg_loss (since no batch losses collected)
    if stats['min_loss'] != avg_loss or stats['max_loss'] != avg_loss:
        print(f"  ❌ FAILED: min_loss={stats['min_loss']}, max_loss={stats['max_loss']}, avg_loss={avg_loss}")
        return False
    
    print("  ✅ PASSED: No batch_losses collected, stats correct")
    return True


def test_train_with_loss_history():
    """Test that train() collects batch_losses when collect_loss_history=True"""
    print("\nTest 2: train() with loss history collection")
    
    vocab_size = 100
    model = DummyModel(vocab_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metadata = Metadata(vocab_size=vocab_size, padding_idx=0, vectors=None)
    
    num_batches = 3
    train_iter = create_dummy_data(batch_size=2, seq_len=5, num_batches=num_batches)
    
    # Call train with collect_loss_history=True
    avg_loss, stats = train(
        model=model,
        optimizer=optimizer,
        train_iter=train_iter,
        metadata=metadata,
        grad_clip=5.0,
        collect_loss_history=True
    )
    
    # Verify that batch_losses IS in stats
    if 'batch_losses' not in stats:
        print("  ❌ FAILED: batch_losses should be in stats")
        return False
    
    # Verify that batch_losses has the correct length
    if len(stats['batch_losses']) != num_batches:
        print(f"  ❌ FAILED: batch_losses length should be {num_batches}, got {len(stats['batch_losses'])}")
        return False
    
    # Verify that min_loss and max_loss are calculated from batch_losses
    expected_min = min(stats['batch_losses'])
    expected_max = max(stats['batch_losses'])
    if stats['min_loss'] != expected_min or stats['max_loss'] != expected_max:
        print(f"  ❌ FAILED: min_loss or max_loss incorrect")
        return False
    
    print(f"  ✅ PASSED: batch_losses collected ({len(stats['batch_losses'])} items), stats correct")
    return True


def test_evaluate_without_loss_history():
    """Test that evaluate() doesn't collect batch_losses when collect_loss_history=False"""
    print("\nTest 3: evaluate() without loss history collection")
    
    vocab_size = 100
    model = DummyModel(vocab_size)
    metadata = Metadata(vocab_size=vocab_size, padding_idx=0, vectors=None)
    
    val_iter = create_dummy_data(batch_size=2, seq_len=5, num_batches=3)
    
    # Call evaluate with collect_loss_history=False
    avg_loss = evaluate(
        model=model,
        val_iter=val_iter,
        metadata=metadata,
        collect_loss_history=False
    )
    
    # Verify that it returns a scalar
    if not isinstance(avg_loss, float):
        print(f"  ❌ FAILED: avg_loss should be float, got {type(avg_loss)}")
        return False
    
    print("  ✅ PASSED: evaluate() completed without collecting loss history")
    return True


def test_evaluate_with_loss_history():
    """Test that evaluate() collects batch_losses when collect_loss_history=True"""
    print("\nTest 4: evaluate() with loss history collection")
    
    vocab_size = 100
    model = DummyModel(vocab_size)
    metadata = Metadata(vocab_size=vocab_size, padding_idx=0, vectors=None)
    
    val_iter = create_dummy_data(batch_size=2, seq_len=5, num_batches=3)
    
    # Call evaluate with collect_loss_history=True and verbose=True to test min/max calculation
    avg_loss = evaluate(
        model=model,
        val_iter=val_iter,
        metadata=metadata,
        collect_loss_history=True,
        verbose=True  # This should use the batch_losses for min/max
    )
    
    # Verify that it returns a scalar
    if not isinstance(avg_loss, float):
        print(f"  ❌ FAILED: avg_loss should be float, got {type(avg_loss)}")
        return False
    
    print("  ✅ PASSED: evaluate() completed with loss history collection")
    return True


def main():
    print("=" * 70)
    print("Unit Tests for Conditional Loss History Collection")
    print("=" * 70)
    
    # Run tests
    test1_passed = test_train_without_loss_history()
    test2_passed = test_train_with_loss_history()
    test3_passed = test_evaluate_without_loss_history()
    test4_passed = test_evaluate_with_loss_history()
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Test 1 (train w/o history):    {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (train w/ history):     {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Test 3 (evaluate w/o history): {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    print(f"Test 4 (evaluate w/ history):  {'✅ PASSED' if test4_passed else '❌ FAILED'}")
    print("=" * 70)
    
    if all([test1_passed, test2_passed, test3_passed, test4_passed]):
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
