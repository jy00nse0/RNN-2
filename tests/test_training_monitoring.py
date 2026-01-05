#!/usr/bin/env python3
"""
Test script to verify the training monitoring and debugging features.
Creates minimal mock data and runs a very short training session.
"""

import os
import sys
import shutil
import json
import random

def create_tiny_mock_data():
    """Create very small mock data for quick testing"""
    
    # Create directory structure
    base_dir = "data/sample100k"
    os.makedirs(base_dir, exist_ok=True)
    
    # Simple word lists
    en_words = ["the", "a", "cat", "dog", "runs", "sits", "is", "on", "mat"]
    de_words = ["der", "die", "katze", "hund", "laeuft", "sitzt", "ist", "auf", "matte"]
    
    def generate_sentence(words, min_len=3, max_len=6):
        """Generate a random sentence from word list"""
        length = random.randint(min_len, max_len)
        return " ".join(random.choices(words, k=length))
    
    # Generate very small datasets for quick testing
    datasets = {
        "train": 300,  # Very small for fast testing
        "valid": 50,
        "test": 50
    }
    
    for split, num_lines in datasets.items():
        en_file = os.path.join(base_dir, f"{split}.en")
        de_file = os.path.join(base_dir, f"{split}.de")
        
        print(f"Creating {split} set with {num_lines} lines...")
        
        with open(en_file, "w", encoding="utf-8") as fen, \
             open(de_file, "w", encoding="utf-8") as fde:
            
            for _ in range(num_lines):
                en_sent = generate_sentence(en_words)
                de_sent = generate_sentence(de_words)
                
                fen.write(en_sent + "\n")
                fde.write(de_sent + "\n")
    
    print(f"Mock data created in {base_dir}/")
    return base_dir

def run_training_test():
    """Run a minimal training session to test monitoring features"""
    
    print("\n" + "=" * 70)
    print("Testing Training Monitoring Features")
    print("=" * 70)
    
    # Clean up any previous test data
    if os.path.exists("data/sample100k"):
        shutil.rmtree("data/sample100k")
    
    # Create tiny mock data
    random.seed(42)
    data_dir = create_tiny_mock_data()
    
    # Test parameters for very quick training
    test_args = [
        "--dataset", "sample100k",
        "--max-epochs", "2",
        "--batch-size", "16",
        "--learning-rate", "0.1",
        "--encoder-num-layers", "1",
        "--decoder-num-layers", "1",
        "--encoder-hidden-size", "64",
        "--decoder-hidden-size", "64",
        "--embedding-size", "64",
        "--gradient-clip", "1.0",
        "--save-path", ".test_save",
        "--lr-decay-start", "5"
    ]
    
    # Import and run training
    print("\n" + "=" * 70)
    print("Starting minimal training run...")
    print("=" * 70 + "\n")
    
    # Modify sys.argv and run training
    original_argv = sys.argv.copy()
    sys.argv = ["train.py"] + test_args
    
    try:
        import train
        # Re-import to get fresh module
        import importlib
        importlib.reload(train)
        train.main()
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.argv = original_argv
        return False
    finally:
        sys.argv = original_argv
    
    print("\n" + "=" * 70)
    print("Verifying Training Outputs")
    print("=" * 70)
    
    # Verify that training_metrics.jsonl was created
    metrics_files = []
    for root, dirs, files in os.walk(".test_save"):
        for file in files:
            if file == "training_metrics.jsonl":
                metrics_files.append(os.path.join(root, file))
    
    if not metrics_files:
        print("❌ FAILED: training_metrics.jsonl not found!")
        return False
    
    print(f"✅ Found training_metrics.jsonl: {metrics_files[0]}")
    
    # Verify metrics file content
    try:
        with open(metrics_files[0], 'r') as f:
            lines = f.readlines()
            print(f"✅ Metrics file has {len(lines)} epochs recorded")
            
            # Check first epoch metrics
            if lines:
                epoch_data = json.loads(lines[0])
                required_keys = ['epoch', 'train_loss', 'train_ppl', 'val_loss', 
                               'val_ppl', 'avg_grad_norm']
                
                for key in required_keys:
                    if key in epoch_data:
                        print(f"✅ Metric '{key}' present: {epoch_data[key]}")
                    else:
                        print(f"❌ FAILED: Missing metric '{key}'")
                        return False
    except Exception as e:
        print(f"❌ FAILED: Error reading metrics file: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ All Tests Passed!")
    print("=" * 70)
    
    # Cleanup
    print("\nCleaning up test files...")
    if os.path.exists("data/sample100k"):
        shutil.rmtree("data/sample100k")
    if os.path.exists(".test_save"):
        shutil.rmtree(".test_save")
    
    return True

if __name__ == "__main__":
    success = run_training_test()
    sys.exit(0 if success else 1)
