#!/usr/bin/env python3
"""
Test individual functions from train.py
"""

import sys
import os
import numpy as np

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_calculate_perplexity():
    """Test perplexity calculation"""
    print("Testing calculate_perplexity()...")
    
    from train import calculate_perplexity
    
    # Test normal case
    loss = 2.0
    ppl = calculate_perplexity(loss)
    expected = np.exp(2.0)
    print(f"  Loss: {loss:.4f} -> PPL: {ppl:.2f} (expected: {expected:.2f})")
    assert abs(ppl - expected) < 0.01, f"Expected {expected}, got {ppl}"
    
    # Test overflow protection
    loss = 150.0
    ppl = calculate_perplexity(loss)
    max_ppl = np.exp(100)
    print(f"  Loss: {loss:.4f} -> PPL: {ppl:.2f} (capped at {max_ppl:.2f})")
    assert ppl == max_ppl, f"Should be capped at {max_ppl}"
    
    print("✅ calculate_perplexity() tests passed!\n")

def test_log_batch_statistics():
    """Test batch logging function"""
    print("Testing log_batch_statistics()...")
    
    from train import log_batch_statistics
    
    # Test batch 0 (should log)
    print("  Testing batch 0 (should log):")
    log_batch_statistics(0, 1000, 5.234, 2.456, 1.0)
    
    # Test batch 50 (should not log)
    print("  Testing batch 50 (should not log):")
    log_batch_statistics(50, 1000, 5.234, 2.456, 1.0)
    
    # Test batch 100 (should log)
    print("  Testing batch 100 (should log):")
    log_batch_statistics(100, 1000, 4.123, 1.789, 0.5)
    
    print("✅ log_batch_statistics() tests passed!\n")

def test_save_training_metrics():
    """Test metrics saving function"""
    print("Testing save_training_metrics()...")
    
    import os
    import json
    import tempfile
    import shutil
    
    from train import save_training_metrics
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save metrics for epoch 1
        metrics1 = {
            'train_loss': 5.234,
            'train_ppl': 187.45,
            'val_loss': 4.876,
            'val_ppl': 131.23
        }
        save_training_metrics(temp_dir, 1, metrics1)
        
        # Save metrics for epoch 2
        metrics2 = {
            'train_loss': 4.123,
            'train_ppl': 61.56,
            'val_loss': 3.987,
            'val_ppl': 53.78
        }
        save_training_metrics(temp_dir, 2, metrics2)
        
        # Read and verify
        metrics_file = os.path.join(temp_dir, 'training_metrics.jsonl')
        assert os.path.exists(metrics_file), "Metrics file not created"
        
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"
            
            # Check epoch 1
            data1 = json.loads(lines[0])
            assert data1['epoch'] == 1, f"Expected epoch 1, got {data1['epoch']}"
            assert data1['train_loss'] == 5.234, "Train loss mismatch"
            print(f"  Epoch 1 metrics: {data1}")
            
            # Check epoch 2
            data2 = json.loads(lines[1])
            assert data2['epoch'] == 2, f"Expected epoch 2, got {data2['epoch']}"
            assert data2['train_loss'] == 4.123, "Train loss mismatch"
            print(f"  Epoch 2 metrics: {data2}")
        
        print("✅ save_training_metrics() tests passed!\n")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

def main():
    """Run all tests"""
    print("=" * 70)
    print("Testing Training Monitoring Functions")
    print("=" * 70 + "\n")
    
    try:
        test_calculate_perplexity()
        test_log_batch_statistics()
        test_save_training_metrics()
        
        print("=" * 70)
        print("✅ All function tests passed!")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
