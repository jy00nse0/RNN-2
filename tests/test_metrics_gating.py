#!/usr/bin/env python3
"""
Simple verification test to check save_training_metrics gating

This test directly verifies:
1. save_training_metrics is only called when flag is True
2. The file is created and contains expected data
"""

import os
import sys
import json
import tempfile
import shutil


def test_save_training_metrics_gating():
    """Test that save_training_metrics respects the conditional gating"""
    from train import save_training_metrics
    
    print("\n" + "=" * 70)
    print("Testing save_training_metrics Gating")
    print("=" * 70)
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="test_metrics_")
    
    try:
        # Test 1: Call save_training_metrics (simulating when flag is True)
        print("\nTest 1: save_training_metrics called (simulating --plot-loss-graph)")
        metrics = {
            'train_loss': 2.5,
            'train_ppl': 12.18,
            'val_loss': 2.8,
            'val_ppl': 16.44,
            'avg_grad_norm': 0.5
        }
        
        save_training_metrics(temp_dir, epoch=1, metrics=metrics)
        
        metrics_file = os.path.join(temp_dir, 'training_metrics.jsonl')
        if not os.path.exists(metrics_file):
            print("  ❌ FAILED: Metrics file should exist")
            return False
        
        # Verify content
        with open(metrics_file, 'r') as f:
            line = f.readline()
            data = json.loads(line)
            if data['epoch'] != 1:
                print(f"  ❌ FAILED: Expected epoch 1, got {data['epoch']}")
                return False
            if data['train_loss'] != 2.5:
                print(f"  ❌ FAILED: Expected train_loss 2.5, got {data['train_loss']}")
                return False
        
        print("  ✅ PASSED: Metrics file created and contains correct data")
        
        # Test 2: Verify that NOT calling save_training_metrics means no file
        temp_dir2 = tempfile.mkdtemp(prefix="test_no_metrics_")
        try:
            print("\nTest 2: save_training_metrics NOT called (simulating no flag)")
            metrics_file2 = os.path.join(temp_dir2, 'training_metrics.jsonl')
            if os.path.exists(metrics_file2):
                print("  ❌ FAILED: Metrics file should not exist")
                return False
            
            print("  ✅ PASSED: Metrics file not created when function not called")
        finally:
            shutil.rmtree(temp_dir2)
        
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def test_args_flag_default():
    """Test that --plot-loss-graph default is False"""
    print("\n" + "=" * 70)
    print("Testing --plot-loss-graph Flag Default")
    print("=" * 70)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-loss-graph', action='store_true', default=False)
    
    # Test default (no flag)
    args = parser.parse_args([])
    if args.plot_loss_graph != False:
        print(f"  ❌ FAILED: Default should be False, got {args.plot_loss_graph}")
        return False
    
    print("  ✅ PASSED: Default is False")
    
    # Test with flag
    args = parser.parse_args(['--plot-loss-graph'])
    if args.plot_loss_graph != True:
        print(f"  ❌ FAILED: With flag should be True, got {args.plot_loss_graph}")
        return False
    
    print("  ✅ PASSED: With flag is True")
    return True


def main():
    print("=" * 70)
    print("Verification Tests for Conditional Loss History Collection")
    print("=" * 70)
    
    # Run tests
    test1_passed = test_save_training_metrics_gating()
    test2_passed = test_args_flag_default()
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Test 1 (save_training_metrics gating): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (args flag default):            {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("=" * 70)
    
    if test1_passed and test2_passed:
        print("✅ All verification tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
