#!/usr/bin/env python3
import os
import subprocess
import shutil
import torch

###############################################################################
# 1. Path 설정
###############################################################################

RAW_DATA_DIR = "data/wmt14_vocab50k/base"
SAMPLE_DATA_DIR = "data/sample100k"

TRAIN_SRC = os.path.join(RAW_DATA_DIR, "train.en")
TRAIN_TGT = os.path.join(RAW_DATA_DIR, "train.de")

SAMPLE_SRC = os.path.join(SAMPLE_DATA_DIR, "train.en")
SAMPLE_TGT = os.path.join(SAMPLE_DATA_DIR, "train.de")

VAL_SRC = os.path.join(RAW_DATA_DIR, "valid.en")
VAL_TGT = os.path.join(RAW_DATA_DIR, "valid.de")

TEST_SRC = os.path.join(RAW_DATA_DIR, "test.en")
TEST_TGT = os.path.join(RAW_DATA_DIR, "test.de")


###############################################################################
# 2. 샘플 데이터셋 생성
###############################################################################
def make_sample_dataset(num_lines=5000):
    """
    Create a small sample dataset for quick testing.
    Using 1000 lines instead of 100k for reasonable testing time.
    """
    if not os.path.exists(SAMPLE_DATA_DIR):
        os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)

    print(f"[+] Creating sample dataset using first {num_lines:,} lines...")

    # Create sample train.en
    with open(TRAIN_SRC, "r", encoding="utf-8") as fin, \
         open(SAMPLE_SRC, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= num_lines:
                break
            fout.write(line)

    # Create sample train.de
    with open(TRAIN_TGT, "r", encoding="utf-8") as fin, \
         open(SAMPLE_TGT, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= num_lines:
                break
            fout.write(line)

    # Validation / Test set: use smaller subsets for quick testing
    # Using first 100 lines instead of full sets
    with open(VAL_SRC, "r", encoding="utf-8") as fin, \
         open(os.path.join(SAMPLE_DATA_DIR, "valid.en"), "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= 20000:
                break
            fout.write(line)
    
    with open(VAL_TGT, "r", encoding="utf-8") as fin, \
         open(os.path.join(SAMPLE_DATA_DIR, "valid.de"), "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= 20000:
                break
            fout.write(line)
    
    with open(TEST_SRC, "r", encoding="utf-8") as fin, \
         open(os.path.join(SAMPLE_DATA_DIR, "test.en"), "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= 20000:
                break
            fout.write(line)
    
    with open(TEST_TGT, "r", encoding="utf-8") as fin, \
         open(os.path.join(SAMPLE_DATA_DIR, "test.de"), "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= 20000:
                break
            fout.write(line)

    print("[+] Sample dataset created successfully.")


###############################################################################
# 3. Sample Test Config
###############################################################################
def run_sample_training():
    print("\n======================= SAMPLE TRAIN START =======================")

    save_path = "checkpoints/sample_test"

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA Available: {torch.cuda.get_device_name(0)}")
        cuda_flag = " --cuda"
    else:
        print("CUDA NOT Available. Running on CPU")
        cuda_flag = ""

    cmd = (
        f"python train.py "
        f"--dataset sample100k "
        f"--save-path {save_path} "
        f"--max-epochs 4 "
        f"--batch-size 100 "
        f"--learning-rate 1.0 "
        f"--encoder-hidden-size 1000 "
        f"--decoder-hidden-size 1000 "
        f"--encoder-num-layers 4 "
        f"--decoder-num-layers 4 "
        f"--attention-type none "
        f"--reverse "
        f"--teacher-forcing-ratio 0.0 "
        f"{cuda_flag}"
    )

    print("[Exec]", cmd)
    subprocess.run(cmd, shell=True, check=True)
    print("======================= SAMPLE TRAIN DONE =======================\n")


###############################################################################
# 4. BLEU Evaluation
###############################################################################
def get_latest_model_dir(base_path):
    """
    Find the most recent timestamped subdirectory under base_path.
    train.py creates directories with format YYYY-MM-DD-HH-MM
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path not found: {base_path}")
    
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {base_path}")
    
    # Sort by name (timestamp format naturally sorts correctly)
    subdirs.sort(reverse=True)
    latest_dir = os.path.join(base_path, subdirs[0])
    
    return latest_dir


def run_sample_evaluation():
    print("\n======================= SAMPLE EVAL START =======================")

    save_path_base = "checkpoints/sample_test"
    ref_path = "data/sample100k/test.de"
    
    # Detect the most recent timestamped subdirectory
    model_path = get_latest_model_dir(save_path_base)
    print(f"Using model from: {model_path}")

    cmd = (
        f"python calculate_bleu.py "
        f"--model-path {model_path} "
        f"--reference-path {ref_path} "
        f"--epoch 4 "
        f"--cuda"
    )

    print("[Exec]", cmd)
    subprocess.run(cmd, shell=True, check=True)

    print("======================= SAMPLE EVAL DONE ========================\n")


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    make_sample_dataset()
    run_sample_training()
    run_sample_evaluation()
