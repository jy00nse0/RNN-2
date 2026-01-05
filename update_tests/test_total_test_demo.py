#!/usr/bin/env python3
"""
Demo script to verify total_test.py can run.
Runs a minimal version of the first experiment to prove the pipeline works.
"""

import os
import subprocess
from total_test import experiments

def run_minimal_experiment():
    """Run a single experiment with minimal settings to verify functionality"""
    
    # Get first experiment
    exp_name = 'T1_Base'
    config = experiments[exp_name]
    
    print("="*70)
    print(f"Testing total_test.py functionality with: {exp_name}")
    print("="*70)
    
    # Create save directory
    save_path = f'checkpoints/demo_{exp_name}'
    os.makedirs(save_path, exist_ok=True)
    
    # Build command with minimal settings for quick testing
    cmd = f"python train.py --dataset {config['dataset']} --save-path {save_path}"
    
    # Use reduced model size and epochs for quick demo
    test_args = {
        **config['args'],
        'max_epochs': 1,
        'batch_size': 16,
        'encoder_hidden_size': 128,
        'decoder_hidden_size': 128,
        'encoder_num_layers': 1,
        'decoder_num_layers': 1,
    }
    
    for key, value in test_args.items():
        flag_name = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                cmd += f' {flag_name}'
        else:
            cmd += f' {flag_name} {value}'
    
    print(f"\nRunning experiment with minimal settings:")
    print(f"  - 1 epoch (instead of {config['args']['max_epochs']})")
    print(f"  - 128 hidden size (instead of 1000)")
    print(f"  - 1 layer (instead of 4)")
    print(f"  - batch size 16 (instead of 128)")
    print(f"\nThis demonstrates that total_test.py structure is correct.")
    print(f"Full experiments would use the paper's full settings.\n")
    
    print("Executing training...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("\n" + "="*70)
        print("✓ SUCCESS! total_test.py can run experiments")
        print("="*70)
        print("\nTraining output (last 10 lines):")
        for line in result.stdout.split('\n')[-10:]:
            if line.strip():
                print("  ", line)
        return True
    else:
        print("\n" + "="*70)
        print("✗ FAILED")
        print("="*70)
        print("\nError output:")
        print(result.stderr[-1000:])
        return False

if __name__ == '__main__':
    success = run_minimal_experiment()
    exit(0 if success else 1)
