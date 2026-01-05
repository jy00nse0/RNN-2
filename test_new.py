#!/usr/bin/env python3
import os
import subprocess
import torch
import sys
import argparse
from util import load_training_metrics, plot_loss_graph
'''
사용 가능한 실험 목록 출력 명령: 
python test. py --list
단일 실험 실행 (예: T1_Base_Reverse) 명령:
python test.py -e T1_Base_Reverse
여러 실험을 순차 실행 명령:
python test.py -e T1_Base_Reverse T1_Global_Location_Feed
모든 실험 실행 명령:
python test.py -e all
python test.py
CUDA 강제 비활성화 (CPU로 실행) 명령:
python test. py -e T1_Base_Reverse --no-cudatrain. py 
#resume 체크포인트를 로드하고 optimizer / epoch 상태 등을 복원하여 이어 학습

#python train.py --dataset wmt14-en-de --save-path checkpoints/T1_Base_Reverse --resume checkpoints/T1_Base_Reverse/model_epoch3.pt --encoder-rnn-cell LSTM --max-epochs 10 --reverse --cuda

'''
# ============================================================
# 1. Hyperparameters & Settings (논문 Section 4.1 기반)
# ============================================================
COMMON = {
    "encoder_rnn_cell": "LSTM", "decoder_rnn_cell": "LSTM",
    "encoder_num_layers": 4, "decoder_num_layers":  4,
    "encoder_hidden_size": 1000, "decoder_hidden_size":  1000,
    "embedding_size": 1000, "batch_size": 128,
    "learning_rate": 1.0, "gradient_clip":  5.0,
}

EPOCHS_BASE, EPOCHS_DR = 10, 12
DECAY_BASE, DECAY_DR = 5, 8

# ============================================================
# 2. Full Experiment Definitions (copied from total_test.py)
# ============================================================
experiments = {

    # ========================================================
    # TABLE 1 — WMT14 En→De (Main Results)
    # ========================================================
    
    # [Row 1] Base Model
    # 설정:  Reverse=False, Dropout=0.0
    "T1_Base":  {
        "dataset": "wmt14-en-de", 
        "args": {**COMMON, "max_epochs": EPOCHS_BASE, "lr_decay_start": DECAY_BASE, 
                 "encoder_rnn_dropout": 0.0, "decoder_rnn_dropout": 0.0, 
                 "attention_type": "none", 
                 "reverse": False}
    },

    # [Row 2] Base + Reverse
    # 설정: Reverse=True
    "T1_Base_Reverse": {
        "dataset":  "wmt14-en-de",
        "args": {**COMMON, "max_epochs":  EPOCHS_BASE, "lr_decay_start": DECAY_BASE,
                 "encoder_rnn_dropout": 0.0, "decoder_rnn_dropout":  0.0,
                 "attention_type": "none",
                 "reverse": True}
    },

    # [Row 3] Base + Reverse + Dropout
    # 설정: Dropout=0.2 (Epochs=12, Decay=8)
    "T1_Base_Reverse_Dropout": {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start":  DECAY_DR,
                 "encoder_rnn_dropout":  0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "none",
                 "reverse": True}
    },

    # [Row 4] + Global (Location)
    # 논문 Table 1 Row 4: Global Attention (location)
    "T1_Global_Location":  {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "global", "attention_score": "location",
                 "luong_input_feed": False, # Attentional models typically use input feeding
                 "reverse": True}
    },

    # [Row 5] + Global (Location) + Feed
    # (Row 4와 동일하게 Feed=True로 설정됨.  논문에서는 Feed 유무로 성능 차이를 강조함)
    "T1_Global_Location_Feed": {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start":  DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "global", "attention_score": "location",
                 "luong_input_feed": True,
                 "reverse": True}
    },

    # [Row 6] + Local-p (General) + Feed
    "T1_LocalP_General_Feed": {
        "dataset":  "wmt14-en-de",
        "args": {**COMMON, "max_epochs":  EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "local-p", "attention_score":  "general",
                 "half_window_size": 10, "local_p_hidden_size": 1000,
                 "luong_input_feed": True,
                 "reverse": True}
    },

    # [Row 7] + Local-p (General) + Feed + Unk Replace
    # 학습 설정은 Row 6와 동일하며, 평가 시 UNK Replacement 수행 (Post-process)
    # [Row 7] + Local-p (General) + Feed + Unk Replace -> 실험 수행 안하고 row 6에 unk replace 만 해서 다시 테스트
    "T1_LocalP_General_Feed_Unk": {
        "dataset":  "wmt14-en-de",
        "args": {**COMMON, "max_epochs":  EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout":  0.2,
                 "attention_type": "local-p", "attention_score": "general",
                 "half_window_size": 10, "local_p_hidden_size":  1000,
                 "luong_input_feed": True,
                 "reverse": True},
        "post_process": "unk_replace"
    },
    
    # [Row 8] Ensemble (Inference Only)
    "T1_Ensemble8_Unk": {
        "dataset":  "wmt14-en-de",
        "ensemble":  True,
        "members": ["T1_Global_Location_Feed", "T1_LocalP_General_Feed"], # 실제로는 8개 모델 필요
        "post_process": "unk_replace"
    },


    # ========================================================
    # TABLE 3 — WMT15 De→En (Direction Change)
    # ========================================================
    
    "T3_Base_Reverse": {
        "dataset": "wmt15-deen", # Triggers De->En in dataset. py
        "args": {**COMMON, "max_epochs":  EPOCHS_BASE, "lr_decay_start": DECAY_BASE,
                 "encoder_rnn_dropout": 0.0, "decoder_rnn_dropout": 0.0,
                 "attention_type": "none",
                 "reverse": True}
    },

    "T3_Global_Location": {
        "dataset": "wmt15-deen",
        "args": {**COMMON, "max_epochs":  EPOCHS_BASE, "lr_decay_start": DECAY_BASE,
                 "encoder_rnn_dropout": 0.0, "decoder_rnn_dropout": 0.0,
                 "attention_type": "global", "attention_score": "location",
                 "reverse": True}
    },

    "T3_Global_Location_Feed": {
        "dataset": "wmt15-deen",
        "args":  {**COMMON, "max_epochs": EPOCHS_BASE, "lr_decay_start": DECAY_BASE,
                 "encoder_rnn_dropout": 0.0, "decoder_rnn_dropout": 0.0,
                 "attention_type":  "global", "attention_score":  "location",
                 "luong_input_feed": True,
                 "reverse": True}
    },

    "T3_Global_Dot_Drop_Feed": {
        "dataset":  "wmt15-deen",
        "args": {**COMMON, "max_epochs":  EPOCHS_DR, "lr_decay_start":  DECAY_DR,
                 "encoder_rnn_dropout":  0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "global", "attention_score": "dot",
                 "luong_input_feed": True,
                 "reverse": True}
    },

    "T3_Global_Dot_Drop_Feed_Unk": {
        "dataset": "wmt15-deen",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "global", "attention_score": "dot",
                 "luong_input_feed": True,
                 "reverse": True},
        "post_process": "unk_replace"
    },


    # ========================================================
    # TABLE 4 — Attention Ablation Study (En→De)
    # Common:  Reverse=True, Dropout=0.2, Feed=True, Epochs=12
    # ========================================================

    "T4_Global_Location":  {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start":  DECAY_DR,
                 "encoder_rnn_dropout":  0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "global", "attention_score": "location",
                 "luong_input_feed":  True,
                 "reverse":  True}
    },

    "T4_Global_Dot":  {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "global", "attention_score": "dot",
                 "luong_input_feed": True,
                 "reverse": True}
    },

    "T4_Global_General":  {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "global", "attention_score": "general",
                 "luong_input_feed": True,
                 "reverse": True}
    },

    "T4_LocalM_Dot": {
        "dataset":  "wmt14-en-de",
        "args": {**COMMON, "max_epochs":  EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "local-m", "attention_score":  "dot",
                 "half_window_size": 10,
                 "luong_input_feed": True,
                 "reverse": True}
    },

    "T4_LocalM_General": {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "local-m", "attention_score": "general",
                 "half_window_size": 10,
                 "luong_input_feed": True,
                 "reverse": True}
    },

    "T4_LocalP_Dot": {
        "dataset": "wmt14-en-de",
        "args":  {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type":  "local-p", "attention_score": "dot",
                 "half_window_size": 10, "local_p_hidden_size": 1000,
                 "luong_input_feed":  True,
                 "reverse":  True}
    },

    "T4_LocalP_General":  {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "local-p", "attention_score": "general",
                 "half_window_size": 10, "local_p_hidden_size": 1000,
                 "luong_input_feed": True,
                 "reverse": True}
    }
}


# ============================================================
# 3. Execution Engine (copied, with CLI selection added)
# ============================================================
def run_command(cmd, log_path=None):
    """
    Execute a shell command. 
    
    Args:
        cmd:  Command string to execute
        log_path:  If provided, redirect stdout/stderr to this file. 
                  If None, output to terminal.
    """
    print(f"[Exec] {cmd}")
    
    if log_path:
        # 로그 파일로 출력 (evaluation용)
        os.makedirs(os.path. dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
    else:
        # 터미널로 직접 출력 (training용)
        subprocess.run(cmd, shell=True, check=True)


def find_latest_checkpoint(save_path):
    """
    Find the latest checkpoint file in save_path.
    
    Returns:
        (checkpoint_path, epoch) or (None, 0) if no checkpoint found
    """
    if not os. path.exists(save_path):
        return None, 0
    
    checkpoint_files = [f for f in os.listdir(save_path) if f.startswith('model_epoch') and f.endswith('.pt')]
    
    if not checkpoint_files:  
        return None, 0
    
    # Extract epoch numbers and find max
    epochs = []
    for f in checkpoint_files:
        try:
            # model_epoch10.pt -> 10
            epoch_num = int(f.replace('model_epoch', '').replace('.pt', ''))
            epochs.append((epoch_num, f))
        except ValueError:
            continue
    
    if not epochs:
        return None, 0
    
    latest_epoch, latest_file = max(epochs, key=lambda x:  x[0])
    return os.path.join(save_path, latest_file), latest_epoch


def evaluate_bleu(save_path, ref_file, epoch, common_flags):
    """
    Calculate BLEU score for a specific epoch.
    
    Args:
        save_path: Path to model checkpoint directory
        ref_file: Path to reference file
        epoch: Epoch number to evaluate
        common_flags: Common flags (e.g., --cuda)
    
    Returns:
        BLEU score as float, or None if evaluation failed
    """
    eval_cmd = f"python calculate_bleu.py --model-path {save_path} --reference-path {ref_file} --epoch {epoch}" + common_flags
    
    try:
        # Run evaluation and capture output
        result = subprocess. run(
            eval_cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Parse BLEU score from output (adjust this based on your calculate_bleu.py output format)
        output_lines = result.stdout.strip().split('\n')
        for line in output_lines:
            if 'BLEU' in line or 'bleu' in line: 
                # Try to extract numerical score
                # This assumes format like "BLEU: 25.3" or "BLEU = 25.3"
                parts = line.replace('=', ': ').split(':')
                if len(parts) >= 2:
                    try:
                        bleu_score = float(parts[-1].strip().split()[0])
                        return bleu_score
                    except (ValueError, IndexError):
                        pass
        
        # If we can't parse, return the last line
        return output_lines[-1] if output_lines else "Unknown"
        
    except subprocess.CalledProcessError as e:
        print(f"BLEU evaluation failed for epoch {epoch}:  {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during BLEU evaluation: {e}")
        return None


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Run selected experiments (subset of experiments defined in the script)."
    )
    parser.add_argument(
        "--experiments", "-e", nargs="+", metavar="EXP",
        help='Experiment names to run (space-separated). Use "all" to run all experiments.  If omitted, runs all experiments.'
    )
    parser.add_argument(
        "--list", action="store_true", help="List available experiment names and exit."
    )
    parser.add_argument(
        "--no-cuda", action="store_true", help="Force disable CUDA even if available."
    )
    parser.add_argument(
        "--plot-loss-graph", action="store_true", default=False,
        help="Generate loss graphs after each experiment training."
    )
    return parser.parse_args()


def main():
    args = parse_cli()

    print("======================================================")
    print("      RNN-NMT Reproduction:  Test Runner               ")
    print("======================================================")

    # List experiments and exit if requested
    if args.list:
        print("Available experiments:")
        for name in experiments. keys():
            print(" -", name)
        return

    # Determine selected experiments
    if args.experiments:
        if "all" in args.experiments:
            selected = set(experiments.keys())
        else:
            selected = set(args.experiments)
            # Report unknown experiment names
            unknown = [n for n in selected if n not in experiments]
            if unknown:
                print(f"Warning: Unknown experiment names: {unknown}. They will be skipped.")
            # Keep only known ones
            selected = set(n for n in selected if n in experiments)
            if not selected:
                print("No valid experiments selected.  Exiting.")
                return
    else:
        # default: run all
        selected = set(experiments.keys())

    # Check CUDA
    if torch.cuda.is_available() and not args.no_cuda:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "CUDA"
        print(f"CUDA Available: {device_name}")
        common_flags = " --cuda"
    else:
        print("CUDA NOT Available or disabled.  Running on CPU (Warning: Slow)")
        common_flags = ""

    os.makedirs("checkpoints", exist_ok=True)

    for exp_name, config in experiments.items():
        if exp_name not in selected:
            print(f"\n>>> Skipping Experiment (not selected): {exp_name}")
            continue

        print(f"\n>>> Processing Experiment: {exp_name}")
        
        # 1. Skip Ensemble Training (Requires separately trained models or custom logic)
        if config.get("ensemble", False):
            print(f"Skipping training for Ensemble '{exp_name}'.  (Inference logic required)")
            continue

        # 2. Setup Directories
        SAVE_DIR = "/workspace"
        save_path = os.path.join(SAVE_DIR, "checkpoints", exp_name)
        os.makedirs(save_path, exist_ok=True)
        
        # Setup BLEU log file
        bleu_log_file = os.path.join(save_path, "bleu_scores.log")
        
        # Determine reference file
        is_deen = 'deen' in config['dataset']
        tgt_ext = 'en' if is_deen else 'de'
        is_wmt15 = 'wmt15' in config['dataset']. lower()
        dataset_name = 'wmt15' if is_wmt15 else 'wmt14'
        ref_file = f"data/{dataset_name}_vocab50k/base/test.{tgt_ext}"
        
        # ==================== Resume 로직 ====================

        # 1. 최신 체크포인트 확인
        checkpoint_path, completed_epochs = find_latest_checkpoint(save_path)
        max_epochs = config['args']['max_epochs']
        
        # 2. 이미 학습 완료된 경우 스킵
        if completed_epochs >= max_epochs:
            print(f"Training already completed ({completed_epochs}/{max_epochs} epochs). Skipping.")
        else:
            # 3. 학습 명령 구성
            cmd = f"python train.py --dataset {config['dataset']} --save-path {save_path}" + common_flags + " --num-workers 0"
            
            # Resume 플래그 추가
            if checkpoint_path:  
                cmd += f" --resume {checkpoint_path}"
                print(f"Resuming from epoch {completed_epochs}/{max_epochs}")
            else:
                print(f"Starting fresh training (0/{max_epochs} epochs)")
            
            # 실험별 인자 추가
            for key, value in config['args'].items():
                flag_name = "--" + key.replace("_", "-")
                
                if isinstance(value, bool):
                    if value:  cmd += f" {flag_name}"
                else: 
                    cmd += f" {flag_name} {value}"

            # 4. 학습 실행 (Train directly to max_epochs)
            try:
                print(f"Training started (target: {max_epochs} epochs)...")
                
                # Run training completely
                run_command(cmd, log_path=None)
                
                # Evaluate BLEU only for the final epoch
                print(f">>> Evaluating BLEU for Final Epoch {max_epochs}...")
                bleu_score = evaluate_bleu(save_path, ref_file, max_epochs, common_flags)
                
                # Log result
                with open(bleu_log_file, 'a') as bleu_log:
                    if completed_epochs == 0:
                        bleu_log.write(f"Experiment: {exp_name}\n")
                        bleu_log.write(f"Dataset: {config['dataset']}\n")
                        bleu_log.write(f"{'='*60}\n")
                    
                    if bleu_score is not None: 
                        log_line = f"Epoch {max_epochs: 2d}:  BLEU = {bleu_score}\n"
                        print(f"    {log_line.strip()}")
                        bleu_log.write(log_line)
                    else: 
                        log_line = f"Epoch {max_epochs:2d}:  BLEU evaluation failed\n"
                        print(f"    {log_line.strip()}")
                        bleu_log.write(log_line)
                
                print("Training completed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"!!! Error during training {exp_name}.")
                print(f"    Command: {cmd}")
                continue
        # ================================================================

        # 5. Generate Loss Graph after training
        if args.plot_loss_graph:
            metrics_file = os.path.join(save_path, 'training_metrics.jsonl')
            if os.path.exists(metrics_file):
                try:
                    print(f"\n>>> Generating loss graph...")
                    train_losses, val_losses = load_training_metrics(metrics_file)
                    
                    if train_losses and val_losses:
                        loss_graph_path = os.path.join(save_path, 'loss_graph.png')
                        plot_loss_graph(train_losses, val_losses, loss_graph_path)
                        print(f"    Loss graph generated successfully!")
                    else:
                        print(f"    Warning: No loss data found in metrics file")
                except Exception as e:
                    print(f"    Error generating loss graph: {e}")
            else:
                print(f"    Warning: Metrics file not found: {metrics_file}")

        # 6. Final Evaluation Summary
        print(f"\n>>> BLEU scores logged to: {bleu_log_file}")
        if os.path.exists(bleu_log_file):
            print("\n=== BLEU Score Summary ===")
            with open(bleu_log_file, 'r') as f:
                print(f.read())


if __name__ == "__main__":
    main()
