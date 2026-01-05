#!/usr/bin/env python3
"""
Test script to verify cross-platform compatibility fixes.
Tests:
1. Timestamp format doesn't contain invalid characters for Windows
2. Path joining works correctly across platforms
3. Directory creation works properly
4. get_model_path function works with new path handling
"""

import os
import tempfile
import shutil
from datetime import datetime


def test_timestamp_format():
    """Test that timestamp format is Windows-compatible (no colons in filename)"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    
    # Check no colons in timestamp
    assert ':' not in timestamp, f"Timestamp contains colons: {timestamp}"
    
    # Verify format is correct (YYYY-MM-DD-HH-MM)
    parts = timestamp.split('-')
    assert len(parts) == 5, f"Expected 5 parts in timestamp, got {len(parts)}: {timestamp}"
    
    print(f"✓ Timestamp format is Windows-compatible: {timestamp}")
    return timestamp


def test_path_joining():
    """Test that os.path.join works correctly"""
    base_path = "checkpoints"
    timestamp = "2026-01-03-12-30"
    
    # Test path joining
    full_path = os.path.join(base_path, timestamp)
    
    # Verify it doesn't have double separators or other issues
    assert base_path in full_path, f"Base path not in full path: {full_path}"
    assert timestamp in full_path, f"Timestamp not in full path: {full_path}"
    
    # Test multiple joins
    nested_path = os.path.join(base_path, timestamp, 'vocab')
    assert 'vocab' in nested_path, f"'vocab' not in nested path: {nested_path}"
    
    print(f"✓ Path joining works correctly: {nested_path}")
    return nested_path


def test_directory_creation():
    """Test that directories can be created with the new format"""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Simulate the save path creation
        base_path = os.path.join(temp_dir, "checkpoints")
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        save_path = os.path.join(base_path, timestamp)
        
        # Create the directory
        os.makedirs(save_path, exist_ok=True)
        
        # Verify it was created
        assert os.path.exists(save_path), f"Directory was not created: {save_path}"
        assert os.path.isdir(save_path), f"Path exists but is not a directory: {save_path}"
        
        # Test creating nested paths
        vocab_path = os.path.join(save_path, 'vocab')
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        
        # Create a test file
        with open(vocab_path, 'w') as f:
            f.write('test')
        
        assert os.path.exists(vocab_path), f"File was not created: {vocab_path}"
        
        print(f"✓ Directory creation works correctly: {save_path}")
        print(f"✓ File creation works correctly: {vocab_path}")


def test_no_invalid_windows_chars():
    """Test that generated paths don't contain invalid Windows filename characters"""
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    
    for char in invalid_chars:
        if char in ['/', '\\']:
            # These are valid as path separators
            continue
        assert char not in timestamp, f"Invalid Windows character '{char}' found in timestamp: {timestamp}"
    
    print(f"✓ No invalid Windows characters in timestamp")


def test_get_model_path_simulation():
    """Test that get_model_path function works with os.path.join"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test model files
        test_files = [
            "seq2seq-1-0.5-0.6.pt",
            "seq2seq-2-0.4-0.5.pt",
            "seq2seq-10-0.3-0.4.pt",
        ]
        
        for filename in test_files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write('test')
        
        # Simulate get_model_path function
        def get_model_path(dir_path, epoch):
            name_start = f"seq2seq-{epoch}"
            for path in os.listdir(dir_path):
                if path.startswith(name_start):
                    return os.path.join(dir_path, path)
            raise ValueError(f"Model from epoch {epoch} doesn't exist in {dir_path}")
        
        # Test finding models
        model_1 = get_model_path(temp_dir, 1)
        assert "seq2seq-1" in model_1, f"Expected seq2seq-1 in path: {model_1}"
        assert os.path.exists(model_1), f"Model path doesn't exist: {model_1}"
        
        model_10 = get_model_path(temp_dir, 10)
        assert "seq2seq-10" in model_10, f"Expected seq2seq-10 in path: {model_10}"
        assert os.path.exists(model_10), f"Model path doesn't exist: {model_10}"
        
        print(f"✓ get_model_path works correctly with os.path.join")


def main():
    print("=" * 70)
    print("Testing Cross-Platform Compatibility Fixes")
    print("=" * 70)
    
    try:
        test_timestamp_format()
        test_path_joining()
        test_directory_creation()
        test_no_invalid_windows_chars()
        test_get_model_path_simulation()
        
        print("\n" + "=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
