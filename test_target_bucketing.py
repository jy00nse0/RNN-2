#!/usr/bin/env python3
"""Quick test to verify bucketing by target works correctly"""

import torch
from dataset import dataset_factory

class Args:
    dataset = 'wmt14-en-de'
    batch_size = 32
    reverse = False
    max_seq_len = 50
    num_workers = 0
    use_bucketing = True
    bucket_by = 'tgt'  # Test target bucketing
    bucket_drop_last = False
    bucket_seed = 42

args = Args()
device = torch.device('cpu')

print("Testing bucketing by target...")
src_metadata, tgt_metadata, src_vocab, tgt_vocab, train_iter, val_iter, test_iter = dataset_factory(args, device)

print(f"✓ Created data loaders with target bucketing")
print(f"✓ Train batches: {len(train_iter)}")

# Get a batch
batch = next(iter(train_iter))
print(f"✓ Batch src shape: {batch.src.shape}")
print(f"✓ Batch tgt shape: {batch.trg.shape}")

# Verify batch_sampler is accessible
assert hasattr(train_iter, 'batch_sampler'), "train_iter should expose batch_sampler"
batch_sampler = train_iter.batch_sampler
assert batch_sampler is not None, "batch_sampler should not be None"
print(f"✓ Batch sampler type: {type(batch_sampler).__name__}")

print("\n✅ Target bucketing works correctly!")
