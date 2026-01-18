#!/usr/bin/env python3
import os
import subprocess
import shutil
import torch
import sys

# reuse evaluation functions
def get_latest_model_dir(base_path):
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path not found: {base_path}")
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {base_path}")
    subdirs.sort(reverse=True)
    return os.path.join(base_path, subdirs[0])

def get_best_epoch(model_dir):
    files = [f for f in os.listdir(model_dir) if f.startswith("seq2seq-") and f.endswith(".pt")]
    if not files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    epochs = []
    for f in files:
        try:
            parts = f.split('-')
            epoch = int(parts[1])
            epochs.append(epoch)
        except (IndexError, ValueError):
            continue
    if not epochs:
         raise ValueError(f"Could not parse epochs from files in {model_dir}")
    return max(epochs)

def run_training():
    print("\n======================= TRAIN START =======================")
    
    # Specific command requested by user
    # Note: stdbuf and tee are shell features. We can emulate similar behavior or just run via shell.
    # The user asked for exactly: 
    # stdbuf -oL python3 train.py ... 2>&1 | tee train_auth_lr6.5_ntd.log
    
    cmd = (
        "stdbuf -oL python3 train.py "
        "--dataset author_data "
        "--save-path checkpoints/author_data "
        "--max-epochs 10 "
        "--max-seq-len 50 "
        "--batch-size 256 "
        "--learning-rate 1 "
        "--encoder-hidden-size 1000 "
        "--decoder-hidden-size 1000 "
        "--encoder-num-layers 4 "
        "--decoder-num-layers 4 "
        "--attention-type none "
        "--reverse "
        "--teacher-forcing-ratio 1.0 "
        "--sample-translations "
        "--lr-decay-start 5 "
        #"--debug "
        "--cuda 2>&1 | tee train_auth_debug19.log"
    )
    
    print(f"[Exec] {cmd}")
    # Using shell=True to support pipes and stdbuf
    subprocess.run(cmd, shell=True, check=True)
    print("======================= TRAIN DONE =======================\n")

def run_evaluation():
    print("\n======================= EVAL START =======================")

    save_path_base = "checkpoints/author_data"
    # Assuming reference path for author_data is data/author_data/train.10k.de (or similar)
    # The original sample_test used train.de. Let's check where author_data is.
    # Based on finding "data/author_data/train.10k.de" earlier:
    ref_path = "data/author_data/train.10k.de" 
    
    try:
        model_path = get_latest_model_dir(save_path_base)
        print(f"Using model from: {model_path}")
        
        best_epoch = get_best_epoch(model_path)
        print(f"Using best epoch: {best_epoch}")

        # Assuming calculate_bleu.py exists and works similarly
        cmd = (
            f"{sys.executable} calculate_bleu.py "
            f"--model-path {model_path} "
            f"--reference-path {ref_path} "
            f"--epoch {best_epoch} "
            f"--cuda"
        )

        print(f"[Exec] {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    except Exception as e:
        print(f"Evaluation failed: {e}")

    print("======================= EVAL DONE ========================\n")

if __name__ == "__main__":
    run_training()
    run_evaluation()
