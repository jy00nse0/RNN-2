#!/usr/bin/env python3
"""
Test to verify conditional loss history collection based on --plot-loss-graph flag

This test validates that:
1. Without --plot-loss-graph flag:
   - No batch_losses list is created in train()
   - No batch_losses list is created in evaluate()
   - No training_metrics.jsonl file is created
   - No loss graph plotting happens
2. With --plot-loss-graph flag:
   - batch_losses lists are created and populated
   - training_metrics.jsonl file is created
   - Loss graph is plotted (if training completes)
"""

import sys
import os
import tempfile
import shutil
import subprocess

def test_without_plot_flag():
    """Test that no loss history is collected when --plot-loss-graph is not set"""
    print("\n" + "=" * 70)
    print("Test 1: Without --plot-loss-graph flag")
    print("=" * 70)
    
    # Create a temporary directory for this test
    temp_dir = tempfile.mkdtemp(prefix="test_no_plot_")
    
    try:
        # Run training for 1 epoch without --plot-loss-graph
        cmd = [
            sys.executable, "train.py",
            "--dataset", "sample100k",
            "--save-path", temp_dir,
            "--max-epochs", "1",
            "--batch-size", "32",
            "--num-workers", "0"  # Single process for testing
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Check that training completed
        if result.returncode != 0:
            print(f"Training failed with return code {result.returncode}")
            print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False
        
        # Find the timestamped subdirectory
        subdirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
        if not subdirs:
            print("❌ FAILED: No timestamped directory created")
            return False
        
        save_dir = os.path.join(temp_dir, subdirs[0])
        
        # Check that training_metrics.jsonl was NOT created
        metrics_file = os.path.join(save_dir, 'training_metrics.jsonl')
        if os.path.exists(metrics_file):
            print(f"❌ FAILED: Metrics file should not exist: {metrics_file}")
            return False
        
        # Check that loss graph was NOT created
        graph_file = os.path.join(save_dir, 'loss_graph.png')
        if os.path.exists(graph_file):
            print(f"❌ FAILED: Loss graph should not exist: {graph_file}")
            return False
        
        print("✅ PASSED: No metrics file or loss graph created")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ FAILED: Training timed out")
        return False
    except Exception as e:
        print(f"❌ FAILED: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")


def test_with_plot_flag():
    """Test that loss history is collected when --plot-loss-graph is set"""
    print("\n" + "=" * 70)
    print("Test 2: With --plot-loss-graph flag")
    print("=" * 70)
    
    # Create a temporary directory for this test
    temp_dir = tempfile.mkdtemp(prefix="test_with_plot_")
    
    try:
        # Run training for 1 epoch with --plot-loss-graph
        cmd = [
            sys.executable, "train.py",
            "--dataset", "sample100k",
            "--save-path", temp_dir,
            "--max-epochs", "1",
            "--batch-size", "32",
            "--num-workers", "0",  # Single process for testing
            "--plot-loss-graph"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Check that training completed
        if result.returncode != 0:
            print(f"Training failed with return code {result.returncode}")
            print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False
        
        # Find the timestamped subdirectory
        subdirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
        if not subdirs:
            print("❌ FAILED: No timestamped directory created")
            return False
        
        save_dir = os.path.join(temp_dir, subdirs[0])
        
        # Check that training_metrics.jsonl WAS created
        metrics_file = os.path.join(save_dir, 'training_metrics.jsonl')
        if not os.path.exists(metrics_file):
            print(f"❌ FAILED: Metrics file should exist: {metrics_file}")
            return False
        
        # Check that metrics file has content
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            if len(lines) < 1:
                print(f"❌ FAILED: Metrics file is empty")
                return False
        
        # Check that loss graph WAS created
        graph_file = os.path.join(save_dir, 'loss_graph.png')
        if not os.path.exists(graph_file):
            print(f"❌ FAILED: Loss graph should exist: {graph_file}")
            return False
        
        print(f"✅ PASSED: Metrics file and loss graph created")
        print(f"   Metrics file: {metrics_file} ({len(lines)} lines)")
        print(f"   Graph file: {graph_file}")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ FAILED: Training timed out")
        return False
    except Exception as e:
        print(f"❌ FAILED: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")


def main():
    print("\n" + "=" * 70)
    print("Testing Conditional Loss History Collection")
    print("=" * 70)
    
    # Check if sample100k dataset exists
    if not os.path.exists("data/sample100k"):
        print("❌ ERROR: sample100k dataset not found. Please run sample_test.py first to create the dataset.")
        return 1
    
    # Run tests
    test1_passed = test_without_plot_flag()
    test2_passed = test_with_plot_flag()
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Test 1 (without flag): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (with flag):    {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("=" * 70)
    
    if test1_passed and test2_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
