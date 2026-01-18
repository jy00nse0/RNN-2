#!/usr/bin/env python3
"""
Integration test for bucketing with training loop.
Tests that bucketing works end-to-end in train.py.
"""

import os
import sys
import subprocess
import tempfile
import shutil

def test_training_without_bucketing():
    """Test training without bucketing (baseline)"""
    print("\n" + "="*70)
    print("Test: Training WITHOUT bucketing (1 epoch)")
    print("="*70)
    
    # Create temp directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, 'train.py',
            '--dataset', 'wmt14-en-de',
            '--save-path', tmpdir,
            '--max-epochs', '1',
            '--batch-size', '32',
            '--learning-rate', '1.0',
            '--encoder-hidden-size', '128',
            '--decoder-hidden-size', '128',
            '--encoder-num-layers', '2',
            '--decoder-num-layers', '2',
            '--embedding-size', '128',
            '--max-seq-len', '20',
            '--num-workers', '0',
            '--no-cuda'
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"❌ Training failed!")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False
        
        print(f"✓ Training completed successfully")
        
        # Check that model was saved
        saved_files = os.listdir(tmpdir)
        print(f"✓ Files saved: {saved_files}")
        
        return True


def test_training_with_bucketing():
    """Test training with bucketing enabled"""
    print("\n" + "="*70)
    print("Test: Training WITH bucketing (1 epoch)")
    print("="*70)
    
    # Create temp directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, 'train.py',
            '--dataset', 'wmt14-en-de',
            '--save-path', tmpdir,
            '--max-epochs', '1',
            '--batch-size', '32',
            '--learning-rate', '1.0',
            '--encoder-hidden-size', '128',
            '--decoder-hidden-size', '128',
            '--encoder-num-layers', '2',
            '--decoder-num-layers', '2',
            '--embedding-size', '128',
            '--max-seq-len', '20',
            '--num-workers', '0',
            '--use-bucketing',
            '--bucket-by', 'src',
            '--no-cuda'
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"❌ Training with bucketing failed!")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False
        
        print(f"✓ Training with bucketing completed successfully")
        
        # Check for "Batch sampler epoch set to" in output
        if "Batch sampler epoch set to" in result.stdout:
            print(f"✓ Batch sampler epoch setting confirmed")
        else:
            print(f"⚠️  Warning: 'Batch sampler epoch set to' not found in output")
        
        # Check that model was saved
        saved_files = os.listdir(tmpdir)
        print(f"✓ Files saved: {saved_files}")
        
        return True


def main():
    """Run integration tests"""
    print("\n" + "="*70)
    print("BUCKETING INTEGRATION TESTS")
    print("="*70)
    
    # Check if mock data exists
    data_dir = 'data/wmt14_vocab50k/base'
    if not os.path.exists(data_dir):
        print(f"\n❌ Mock data not found at {data_dir}")
        print(f"Please run: python create_mock_data.py")
        sys.exit(1)
    
    success = True
    
    try:
        # Test 1: Training without bucketing
        if not test_training_without_bucketing():
            success = False
        
        # Test 2: Training with bucketing
        if not test_training_with_bucketing():
            success = False
        
        if success:
            print("\n" + "="*70)
            print("✅ ALL INTEGRATION TESTS PASSED!")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("❌ SOME TESTS FAILED!")
            print("="*70)
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
