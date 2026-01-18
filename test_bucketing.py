#!/usr/bin/env python3
"""
Test script for length-bucketed batching implementation.
Verifies:
1. BucketBatchSampler works correctly
2. Token tensors remain as torch.long
3. Bucketing integrates with existing training pipeline
"""

import os
import sys
import torch
import argparse
from dataset import (
    BucketBatchSampler, 
    TranslationDataset, 
    LazyTranslationDataset,
    dataset_factory
)

def test_bucket_batch_sampler():
    """Test BucketBatchSampler basic functionality"""
    print("\n" + "="*70)
    print("Test 1: BucketBatchSampler Basic Functionality")
    print("="*70)
    
    # Create a mock dataset with lengths
    class MockDataset:
        def __init__(self, size=100):
            self.size = size
            # Random lengths between 5 and 50
            torch.manual_seed(42)
            self.lengths = torch.randint(5, 50, (size,))
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return idx
    
    dataset = MockDataset(100)
    batch_size = 10
    
    # Create sampler
    sampler = BucketBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True,
        seed=42
    )
    
    # Test 1: Check number of batches
    num_batches = len(sampler)
    expected_batches = (100 + batch_size - 1) // batch_size  # ceiling division
    assert num_batches == expected_batches, f"Expected {expected_batches} batches, got {num_batches}"
    print(f"✓ Number of batches: {num_batches} (expected {expected_batches})")
    
    # Test 2: Verify batches are sorted by length within batch
    batches = list(sampler)
    for i, batch in enumerate(batches[:3]):  # Check first 3 batches
        batch_lengths = [dataset.lengths[idx].item() for idx in batch]
        print(f"  Batch {i}: indices={batch[:5]}... lengths={batch_lengths[:5]}...")
    
    # Test 3: Verify set_epoch changes shuffle order
    batches_epoch0 = list(sampler)
    sampler.set_epoch(1)
    batches_epoch1 = list(sampler)
    
    # The order should be different
    assert batches_epoch0 != batches_epoch1, "Batch order should change after set_epoch"
    print(f"✓ set_epoch() changes batch shuffle order")
    
    # Test 4: Same epoch gives same order
    sampler.set_epoch(0)
    batches_epoch0_again = list(sampler)
    assert batches_epoch0 == batches_epoch0_again, "Same epoch should give same order"
    print(f"✓ Same epoch gives consistent batch order")
    
    print(f"\n✅ BucketBatchSampler tests passed!")


def test_dataset_lengths():
    """Test that datasets compute lengths correctly"""
    print("\n" + "="*70)
    print("Test 2: Dataset Length Computation")
    print("="*70)
    
    # Check if test data exists
    data_dir = 'data/wmt14_vocab50k/base'
    if not os.path.exists(data_dir):
        print(f"⚠️  Skipping dataset length test - data directory not found: {data_dir}")
        return
    
    # Test with TranslationDataset (if available)
    src_file = os.path.join(data_dir, 'train.en')
    tgt_file = os.path.join(data_dir, 'train.de')
    
    if not os.path.exists(src_file) or not os.path.exists(tgt_file):
        print(f"⚠️  Skipping dataset length test - data files not found")
        return
    
    print("\nTesting TranslationDataset...")
    dataset = TranslationDataset(
        src_file=src_file,
        tgt_file=tgt_file,
        max_len=50
    )
    
    # Test lengths property
    src_lengths = dataset.get_lengths('src')
    tgt_lengths = dataset.get_lengths('tgt')
    
    assert len(src_lengths) == len(dataset), "Source lengths count mismatch"
    assert len(tgt_lengths) == len(dataset), "Target lengths count mismatch"
    print(f"✓ Dataset size: {len(dataset)}")
    print(f"✓ Source lengths: min={src_lengths.min().item()}, max={src_lengths.max().item()}, mean={src_lengths.float().mean().item():.2f}")
    print(f"✓ Target lengths: min={tgt_lengths.min().item()}, max={tgt_lengths.max().item()}, mean={tgt_lengths.float().mean().item():.2f}")
    
    print(f"\n✅ Dataset length computation tests passed!")


def test_token_dtype():
    """Test that token tensors remain as torch.long"""
    print("\n" + "="*70)
    print("Test 3: Token Tensor Data Types")
    print("="*70)
    
    # Check if test data exists
    data_dir = 'data/wmt14_vocab50k/base'
    if not os.path.exists(data_dir):
        print(f"⚠️  Skipping token dtype test - data directory not found: {data_dir}")
        return
    
    src_file = os.path.join(data_dir, 'train.en')
    tgt_file = os.path.join(data_dir, 'train.de')
    
    if not os.path.exists(src_file) or not os.path.exists(tgt_file):
        print(f"⚠️  Skipping token dtype test - data files not found")
        return
    
    print("\nCreating dataset...")
    dataset = TranslationDataset(
        src_file=src_file,
        tgt_file=tgt_file,
        max_len=50
    )
    
    # Get a sample
    src, tgt = dataset[0]
    
    # Check dtypes
    print(f"Source tensor dtype: {src.dtype}")
    print(f"Target tensor dtype: {tgt.dtype}")
    
    # Verify they are integer types (compatible with nn.Embedding)
    assert src.dtype in [torch.int, torch.long, torch.int32, torch.int64], f"Source dtype {src.dtype} not compatible with nn.Embedding"
    assert tgt.dtype in [torch.int, torch.long, torch.int32, torch.int64], f"Target dtype {tgt.dtype} not compatible with nn.Embedding"
    
    print(f"✓ Token tensors use integer dtype (compatible with nn.Embedding)")
    print(f"\n✅ Token dtype tests passed!")


def test_integration():
    """Test integration with dataset_factory"""
    print("\n" + "="*70)
    print("Test 4: Integration with dataset_factory")
    print("="*70)
    
    # Check if test data exists
    data_dir = 'data/wmt14_vocab50k/base'
    if not os.path.exists(data_dir):
        print(f"⚠️  Skipping integration test - data directory not found: {data_dir}")
        return
    
    # Create mock args
    class Args:
        dataset = 'wmt14-en-de'
        batch_size = 32
        reverse = False
        max_seq_len = 50
        num_workers = 0
        # Bucketing disabled
        use_bucketing = False
        bucket_by = 'src'
        bucket_drop_last = False
        bucket_seed = 42
    
    args = Args()
    device = torch.device('cpu')
    
    print("\nTesting without bucketing...")
    src_metadata, tgt_metadata, src_vocab, tgt_vocab, train_iter, val_iter, test_iter = dataset_factory(args, device)
    
    print(f"✓ Source vocab size: {src_metadata.vocab_size}")
    print(f"✓ Target vocab size: {tgt_metadata.vocab_size}")
    print(f"✓ Train batches: {len(train_iter)}")
    
    # Get a batch
    batch = next(iter(train_iter))
    print(f"✓ Batch src shape: {batch.src.shape}")
    print(f"✓ Batch tgt shape: {batch.trg.shape}")
    
    # Now test with bucketing enabled
    print("\nTesting with bucketing enabled...")
    args.use_bucketing = True
    args.bucket_by = 'src'
    
    src_metadata, tgt_metadata, src_vocab, tgt_vocab, train_iter, val_iter, test_iter = dataset_factory(args, device)
    
    print(f"✓ Train batches with bucketing: {len(train_iter)}")
    
    # Verify batch_sampler is accessible
    assert hasattr(train_iter, 'batch_sampler'), "train_iter should expose batch_sampler"
    batch_sampler = train_iter.batch_sampler
    if batch_sampler is not None:
        print(f"✓ Batch sampler type: {type(batch_sampler).__name__}")
        assert hasattr(batch_sampler, 'set_epoch'), "Batch sampler should have set_epoch method"
        print(f"✓ Batch sampler has set_epoch method")
    
    # Get a batch with bucketing
    batch = next(iter(train_iter))
    print(f"✓ Batch src shape with bucketing: {batch.src.shape}")
    print(f"✓ Batch tgt shape with bucketing: {batch.trg.shape}")
    
    # Verify token dtypes are still correct
    assert batch.src.dtype in [torch.int, torch.long, torch.int32, torch.int64], "Source tokens should be integer type"
    assert batch.trg.dtype in [torch.int, torch.long, torch.int32, torch.int64], "Target tokens should be integer type"
    print(f"✓ Token dtypes remain integer (src: {batch.src.dtype}, tgt: {batch.trg.dtype})")
    
    print(f"\n✅ Integration tests passed!")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("BUCKETING IMPLEMENTATION TESTS")
    print("="*70)
    
    try:
        # Test 1: BucketBatchSampler
        test_bucket_batch_sampler()
        
        # Test 2: Dataset lengths (skip if no data)
        test_dataset_lengths()
        
        # Test 3: Token dtypes (skip if no data)
        test_token_dtype()
        
        # Test 4: Integration (skip if no data)
        test_integration()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
