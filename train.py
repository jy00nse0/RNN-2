#!/usr/bin/env python3

import os
os.environ['PYTHONHASHSEED'] = '0'

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

import torch

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from dataset import dataset_factory
from model import train_model_factory
from serialization import save_object, save_model, save_vocab
from datetime import datetime
from util import embedding_size_from_name, load_training_metrics, plot_loss_graph, count_nonzero_grad_vocab, summarize_delta
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json
import debug_utils
from debug_utils import set_debug_mode, debug_tensor, debug_print

"""
[Optimized] train.py for RNN Paper Reproduction

Original Changes:
1. Optimizer: Adam -> SGD (Paper Sec 4.1)
2. Scheduler: Added Halving Schedule (Paper Sec 4.1)
   - Base model: Halve after epoch 5
   - Dropout model: Halve after epoch 8
3. Defaults: Updated batch_size(128), lr(1.0) to match paper.

New Optimizations:
1. view() -> reshape() for safer tensor operations
2. AMP (Mixed Precision) support with --amp flag
3. DataLoader workers configuration
4. zero_grad(set_to_none=True) for better performance
5. TF32 and cuDNN benchmarking enabled

[New] generate_sample_translations now uses greedy decoding with early stopping (<eos>).
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Script for training seq2seq chatbot.')
    
    # ===== Paper Hyperparameters =====
    # [Paper] Total Epochs: 10 (Base) or 12 (Dropout)
    parser.add_argument('--max-epochs', type=int, default=10, 
                       help='Max number of epochs models will be trained.')
    # [Paper] Gradient Clipping: Norm > 5
    parser.add_argument('--gradient-clip', type=float, default=5.0, 
                       help='Gradient clip value.')
    # [Paper] Batch Size: 128
    parser.add_argument('--batch-size', type=int, default=128, 
                       help='Batch size.')
    # [Paper] Initial Learning Rate: 1.0 (SGD)
    parser.add_argument('--learning-rate', type=float, default=1.0, 
                       help='Initial learning rate.')
    # [Paper] LR Decay Start: 5 (Base) or 8 (Dropout)
    parser.add_argument('--lr-decay-start', type=int, default=5, 
                       help='Epoch after which to start halving learning rate. (Base: 5, Dropout: 8)')
    # [Paper] Maximum sequence length for training (Truncation)
    parser.add_argument('--max-seq-len', type=int, default=50,
                       help='Maximum sequence length for source and target sentences (default: 50).')
    
    # ===== Training Configuration =====
    parser.add_argument('--train-embeddings', action='store_true', default=True, 
                       help='Should gradients be propagated to word embeddings.')
    parser.add_argument('--embedding-type', type=str, default=None)
    parser.add_argument('--save-path', default='.save',
                       help='Folder where models (and other configs) will be saved during training.')
    parser.add_argument('--save-every-epoch', action='store_true',default=False,
                       help='Save model every epoch regardless of validation loss.')
    parser.add_argument('--dataset', 
                       choices=['twitter-applesupport', 'twitter-amazonhelp', 'twitter-delta',
                               'twitter-spotifycares', 'twitter-uber_support', 'twitter-all',
                               'twitter-small', 'wmt14-en-de', 'wmt15-deen', 'sample100k', 'author_data'],
                       help='Dataset for training model.')
    parser.add_argument('--teacher-forcing-ratio', type=float, default=1.0,
                       help='Teacher forcing ratio used in seq2seq models. [0-1]')
    parser.add_argument('--teacher-forcing-decay', action='store_true', default=False,
                       help='Enable linear decay of teacher forcing ratio (1.0 -> 0.0).')
    
    # [Paper] Experimental Setup
    parser.add_argument('--reverse', action='store_true', 
                       help='[Experiment] Reverse source sequence (excluding special tokens) for LSTM inputs.')
    
    # ===== Bucketing Configuration =====
    bucketing_args = parser.add_argument_group('Bucketing', 'Length-bucketed batching settings.')
    bucketing_args.add_argument('--use-bucketing', action='store_true', default=True,
                               help='Enable length-bucketed batching for training (reduces padding, improves throughput).')
    bucketing_args.add_argument('--bucket-by', type=str, choices=['src', 'tgt'], default='tgt',
                               help='Which sequence to bucket by: src (source) or tgt (target). Default: src.')
    bucketing_args.add_argument('--bucket-drop-last', action='store_true', default=False,
                               help='Drop last incomplete batch when bucketing.')
    bucketing_args.add_argument('--bucket-seed', type=int, default=123,
                               help='Random seed for bucketing batch shuffle. Default: 42.')
    
    # ===== GPU Settings =====
    gpu_args = parser.add_argument_group('GPU', 'GPU related settings.')
    gpu_args.add_argument('--cuda', action='store_true', default=True, 
                         help='Use cuda if available.')
    gpu_args.add_argument('--multi-gpu', action='store_true', default=False, 
                         help='Use multiple GPUs if available.')
    
    # ===== [OPTIMIZED] Performance Options =====
    perf_args = parser.add_argument_group('Performance', 'Performance optimization settings.')
    perf_args.add_argument('--num-workers', type=int, default=12,
                          help='Number of DataLoader workers (0=single process, 4-8 recommended).')
    perf_args.add_argument('--amp', action='store_true', default= False,
                          help='Use Automatic Mixed Precision (1.5-2x faster but may affect reproducibility).')
    
    # ===== Debugging and Visualization Options =====
    debug_args = parser.add_argument_group('Debug', 'Debugging and visualization settings.')
    debug_args.add_argument('--debug', action='store_true', default=False,
                           help='Enable per-batch logging and verbose evaluation.')
    debug_args.add_argument('--log-interval', type=int, default=100,
                           help='Batch interval for logging when --debug is set.')
    debug_args.add_argument('--sample-translations', action='store_true', default=True,
                           help='Print sample translations after each epoch.')
    debug_args.add_argument('--print-model-summary', action='store_true', default=False,
                           help='Print model architecture and parameter counts.')
    debug_args.add_argument('--plot-loss-graph', action='store_true', default=False,
                           help='Plot training/validation loss graph at end of training.')

    # ===== Embedding Hyperparameters =====
    parser.add_argument('--embedding-size', type=int, default=1000, 
                       help='Embedding size.')
    
    # ===== Encoder Hyperparameters =====
    encoder_args = parser.add_argument_group('Encoder', 'Encoder hyperparameters.')
    encoder_args.add_argument('--encoder-rnn-cell', choices=['LSTM', 'GRU'], default='LSTM',
                             help='Encoder RNN cell type.')
    encoder_args.add_argument('--encoder-hidden-size', type=int, default=1000, 
                             help='Encoder RNN hidden size.')
    encoder_args.add_argument('--encoder-num-layers', type=int, default=4, 
                             help='Encoder RNN number of layers.')
    encoder_args.add_argument('--encoder-rnn-dropout', type=float, default=0.0, 
                             help='Encoder RNN dropout probability.')
    encoder_args.add_argument('--encoder-bidirectional', action='store_true', 
                             help='Use bidirectional encoder.')

    # ===== Decoder Hyperparameters =====
    decoder_args = parser.add_argument_group('Decoder', 'Decoder hyperparameters.')
    decoder_args.add_argument('--decoder-type', choices=['bahdanau', 'luong'], default='luong',
                             help='Type of the decoder.')
    decoder_args.add_argument('--decoder-rnn-cell', choices=['LSTM', 'GRU'], default='LSTM',
                             help='Decoder RNN cell type.')
    decoder_args.add_argument('--decoder-hidden-size', type=int, default=1000, 
                             help='Decoder RNN hidden size.')
    decoder_args.add_argument('--decoder-num-layers', type=int, default=4, 
                             help='Decoder RNN number of layers.')
    decoder_args.add_argument('--decoder-rnn-dropout', type=float, default=0.0, 
                             help='Decoder RNN dropout probability.')
    decoder_args.add_argument('--luong-attn-hidden-size', type=int, default=1000,
                             help='Luong decoder attention hidden projection size')
    decoder_args.add_argument('--luong-input-feed', action='store_true',
                             help='Whether Luong decoder should use input feeding approach.')
    decoder_args.add_argument('--decoder-init-type', 
                             choices=['zeros', 'bahdanau', 'adjust_pad', 'adjust_all'],
                             default='adjust_pad', 
                             help='Decoder initial RNN hidden state initialization.')

    # ===== Attention Hyperparameters =====
    attention_args = parser.add_argument_group('Attention', 'Attention hyperparameters.')
    attention_args.add_argument('--attention-type', 
                               choices=['none', 'global', 'local-m', 'local-p'], 
                               default='global',
                               help='Attention type.')
    attention_args.add_argument('--attention-score', 
                               choices=['dot', 'general', 'concat', 'location'], 
                               default='dot',
                               help='Attention score function type.')
    # [Paper] Window size D=10
    attention_args.add_argument('--half-window-size', type=int, default=10,
                               help='D parameter from Luong et al. paper. Used only for local attention.')
    attention_args.add_argument('--local-p-hidden-size', type=int, default=1000,
                               help='Local-p attention hidden size (used when predicting window position).')
    attention_args.add_argument('--concat-attention-hidden-size', type=int, default=1000,
                               help='Attention layer hidden size. Used only with concat score function.')

    args = parser.parse_args()

    # [Note] Embeddings logic preserved but likely not used if training from scratch with vocab 50k
    if not args.embedding_type and not args.embedding_size:
        args.embedding_size = 1000  # Paper default

    args.save_path = os.path.join(args.save_path, datetime.now().strftime("%Y-%m-%d-%H-%M"))

    print(args)
    return args



def batch_reverse_source(src_tensor, pad_idx, batch_first=False):
    """
    Reverse the content of source sequences while preserving <sos>, <eos>, and <pad> positions.
    
    Supports both (seq_len, batch) and (batch, seq_len) tensor shapes.
    Handles padding at both beginning (left) and end (right) of sequences.
    
    Args:
        src_tensor: Input tensor of shape (seq_len, batch) or (batch, seq_len)
        pad_idx: Index of padding token
        batch_first: If True, input shape is (batch, seq_len); if False, (seq_len, batch)
    
    Input structure: <sos> w1 w2 ... wn <eos> (with optional <pad> before or after)
    Output structure: <sos> wn ... w2 w1 <eos> (with same <pad> positions preserved)
    
    Returns:
        Tensor with reversed content, same shape as input
    """
    rev_src = src_tensor.clone()
    
    if batch_first:
        batch_size, seq_len = src_tensor.size()
    else:
        seq_len, batch_size = src_tensor.size()
    
    for b in range(batch_size):
        if batch_first:
            seq = src_tensor[b, :]
        else:
            seq = src_tensor[:, b]
        
        # Find indices of non-padding tokens
        non_pad_indices = (seq != pad_idx).nonzero(as_tuple=True)[0]
        
        if len(non_pad_indices) <= 2:
            # Only <sos> and <eos> or fewer - nothing to reverse
            continue
        
        # First and last non-padding positions are <sos> and <eos>
        first_non_pad = non_pad_indices[0].item()
        last_non_pad = non_pad_indices[-1].item()
        
        # Content to reverse: tokens between <sos> (first_non_pad) and <eos> (last_non_pad)
        start_idx = first_non_pad + 1  # skip <sos>
        end_idx = last_non_pad  # up to but not including <eos>
        
        if end_idx <= start_idx:
            # No content to reverse (only special tokens)
            continue
        
        # Extract content and reverse
        if batch_first:
            content = src_tensor[b, start_idx:end_idx]
            rev_src[b, start_idx:end_idx] = torch.flip(content, dims=[0])
        else:
            content = src_tensor[start_idx:end_idx, b]
            rev_src[start_idx:end_idx, b] = torch.flip(content, dims=[0])
    
    return rev_src


# Constants for monitorig
MAX_LOSS_FOR_PERPLEXITY = 100  # Cap for preventing overflow in perplexity calculation


def calculate_perplexity(loss):
    """Calculate perplexity from cross-entropy loss"""
    return np.exp(min(loss, MAX_LOSS_FOR_PERPLEXITY))  # Cap to prevent overflow


def log_batch_statistics(batch_idx, total_batches, loss, grad_norm, lr, debug=False, log_interval=100):
    """
    Log detailed batch-level statistics.
    
    Args:
        batch_idx: Current batch index
        total_batches: Total number of batches
        loss: Current batch loss
        grad_norm: Gradient norm
        lr: Learning rate
        debug: Enable logging (default: False)
        log_interval: Batch interval for logging when debug is True (default: 100)
    """
    if debug and batch_idx % log_interval == 0:
        perplexity = calculate_perplexity(loss)
        print(f"  [Batch {batch_idx}/{total_batches}] "
              f"Loss: {loss:.4f} | PPL: {perplexity:.2f} | "
              f"Grad Norm: {grad_norm:.4f} | LR: {lr:.6f}")


def _get_special_token_indices(tgt_metadata, tgt_vocab):
    """
    Try to resolve <sos> and <eos> indices from metadata; fall back to vocab.stoi if needed.
    """
    sos_idx = getattr(tgt_metadata, 'sos_idx', None)
    eos_idx = getattr(tgt_metadata, 'eos_idx', None)

    # Fallbacks if metadata does not expose indices
    if sos_idx is None:
        sos_idx = tgt_vocab.stoi.get('<sos>', tgt_vocab.stoi.get('<go>', 1))
        print('sos_idx', sos_idx)
    if eos_idx is None:
        eos_idx = tgt_vocab.stoi.get('<eos>', tgt_vocab.stoi.get('<end>', 2))
        print('eos_idx', eos_idx)
    return sos_idx, eos_idx


def _greedy_decode_sequence(model, src_seq_1, sos_idx, eos_idx, max_len, src_length=None):
    """
    Greedy decode a single example (batch size = 1) with early stopping on <eos>.
    Args:
        model: seq2seq model with forward(question, answer, src_lengths)
        src_seq_1: Tensor of shape (src_len, 1) for a single example
        sos_idx, eos_idx: indices for special tokens
        max_len: maximum generated tokens (excluding <sos>)
        src_length: Tensor of shape (1,) for source length
    Returns:
        List[int]: generated target token ids (without <sos>, and excluding <eos>)
    """
    device = src_seq_1.device

    # Start with <sos>
    tgt_seq = torch.tensor([[sos_idx]], dtype=torch.long, device=device)  # (1, 1)

    # Iteratively decode next tokens
    for _ in range(max_len):
        # Forward pass with the current partial target (teacher forcing disabled)
        # Seq2SeqTrain.forward slices input[:-1], so we append a dummy token
        # to ensure the last token in tgt_seq is used as input
        dummy_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        model_input = torch.cat([tgt_seq, dummy_token], dim=0)
        
        logits = model(src_seq_1, model_input, src_lengths=src_length)            # (tgt_len-1, 1, vocab)
        step_logit = logits[-1, 0]                    # last step for next token
        next_token = int(step_logit.argmax(dim=-1).item())

        # Early stop on <eos> (do not append <eos>)
        if next_token == eos_idx:
            break

        # Append the predicted token and continue
        next_tok_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        tgt_seq = torch.cat([tgt_seq, next_tok_tensor], dim=0)

    # Return generated tokens (excluding <sos>)
    return tgt_seq.squeeze(1).tolist()[1:]


def generate_sample_translations(model, val_iter, tgt_metadata, src_vocab, tgt_vocab, num_samples=10, max_len=50):
    """
    Generate sample translations to visualize model progress using greedy decoding with early stopping.
    - Uses model's forward iteratively to produce tokens until <eos> or max_len.
    - No teacher-forced logits.argmax over a fixed-length target; prevents trailing tokens after <eos>.
    """
    model.eval()
    samples = []
    device = next(model.parameters()).device

    sos_idx, eos_idx = _get_special_token_indices(tgt_metadata, tgt_vocab)
    src_pad_idx = 0  # Source padding index (from Vocab construction)
    tgt_pad_idx = tgt_metadata.padding_idx
    
    count = 0

    with torch.no_grad():
        # Create a fresh iterator to avoid state conflicts
        for batch in iter(val_iter):
            if count >= num_samples:
                break
            
            question, answer = batch.question, batch.answer
            
            # Check if batch has data
            if question.size(1) == 0 or answer.size(0) == 0:
                continue
            
            batch_size = question.size(1)

            # Iterate through each sample in the batch
            for b_idx in range(batch_size):
                if count >= num_samples:
                    break

                # Decode the current item in the batch
                src_seq_1 = question[:, b_idx:b_idx+1].to(device)  # keep batch dimension = 1
                
                # Handle potential empty sequence if filtering removed everything
                if src_seq_1.size(0) == 0:
                    continue

                # Prepare lengths for greedy decode (shape (1,))
                src_length = batch.src_lengths[b_idx:b_idx+1] # Keep it as tensor (1,)

                pred_token_ids = _greedy_decode_sequence(
                    model=model,
                    src_seq_1=src_seq_1,
                    sos_idx=sos_idx,
                    eos_idx=eos_idx,
                    max_len=max_len,
                    src_length=src_length
                )

                # Prepare tokens for display
                src_tokens = question[:, b_idx].cpu().tolist()
                tgt_tokens = answer[1:, b_idx].cpu().tolist()  # ground truth without <sos>

                # Convert to words (filter padding and invalid indices)
                src_words = [src_vocab.itos[idx] for idx in src_tokens
                             if 0 <= idx < len(src_vocab.itos) and idx != src_pad_idx]
                tgt_words = [tgt_vocab.itos[idx] for idx in tgt_tokens
                             if 0 <= idx < len(tgt_vocab.itos) and idx != tgt_pad_idx]
                pred_words = [tgt_vocab.itos[idx] for idx in pred_token_ids
                              if 0 <= idx < len(tgt_vocab.itos) and idx != tgt_pad_idx]
                
                samples.append({
                    'source': ' '.join(src_words),
                    'target': ' '.join(tgt_words),
                    'prediction': ' '.join(pred_words)
                })
                count += 1
    
    return samples


def save_training_metrics(save_path, epoch, metrics):
    """Save training metrics to JSON file for later analysis"""
    metrics_file = os.path.join(save_path, 'training_metrics.jsonl')
    os.makedirs(save_path, exist_ok=True)
    
    with open(metrics_file, 'a') as f:
        json.dump({'epoch': epoch, **metrics}, f)
        f.write('\n')


def evaluate(model, val_iter, metadata, reverse_src=False, verbose=False, collect_loss_history=False):
    """
    [OPTIMIZED] Evaluate model on validation set.
    - Changed view() to reshape() for safer tensor operations.
    - Added perplexity calculation and min/max loss tracking
    - Uses torch.no_grad() to avoid graph creation during validation
    
    Args:
        collect_loss_history: Whether to collect batch-level loss history
    """
    model.eval()
    total_loss = 0
    batch_losses = [] if collect_loss_history else None
    
    with torch.no_grad():
        for batch in tqdm(val_iter, desc="Evaluating", leave=False):
            question, answer = batch.question, batch.answer
            logits = model(question, answer, batch.src_lengths)
            #print("logits shape : ",logits.shape)
            logits[..., 0] = float("-inf")   # V 차원 한 칸만

            # ===== [MODIFIED] Handle dynamic decoder output length =====
            # logits may be shorter than answer[1:] due to dynamic loop bounds
            actual_output_len = logits.size(0)
            
            # Slice target labels to match logits length
            target_label_forloss = answer[1:actual_output_len+1].clone()  # (actual_len, batch)
            target_label_forloss[target_label_forloss == metadata.padding_idx] = -100
            '''
            # [OPTIMIZED] reshape() instead of view()
            loss = F.cross_entropy(
                logits.reshape(-1, metadata.vocab_size),
                target_label_forloss.reshape(-1),
                ignore_index=-100
            )
            '''
            # log_softmax는 vocab 차원으로 수행해야 함
            log_probs = F.log_softmax(logits, dim=-1)  # (T, B, V)
        # nll_loss input: (N, C) or (N, C, d1, ...) 형태 가능
            # 여기서는 (T*B, V) 로 flatten
            nll_loss = F.nll_loss(
                log_probs.reshape(-1, metadata.vocab_size),     # (T*B, V)
                target_label_forloss.reshape(-1),               # (T*B,)
                ignore_index=-100
            )
                        # 유효 위치 마스크: target != -100
            mask = (target_label_forloss != -100).float()  # (T,B)
            loss = (nll_loss * mask).sum() / mask.sum().clamp_min(1.0)
            if collect_loss_history:
                batch_losses.append(loss.item())
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_iter)
    
    if verbose:
        perplexity = calculate_perplexity(avg_loss)
        if collect_loss_history and batch_losses:
            min_loss = min(batch_losses)
            max_loss = max(batch_losses)
        else:
            min_loss = avg_loss
            max_loss = avg_loss
        print(f"\n  📊 Validation Metrics:")
        print(f"     Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
        print(f"     Min Loss: {min_loss:.4f} | Max Loss: {max_loss:.4f}")
    
    return avg_loss


def train(model, optimizer, train_iter, metadata, grad_clip, reverse_src=False, 
          use_amp=False, scaler=None, epoch=0, save_path=None, vocab=None, src_vocab=None,
          debug=False, log_interval=100, collect_loss_history=False):
    """
    [OPTIMIZED] Train model for one epoch.
    
    Optimizations:
    - AMP (Mixed Precision) support
    - reshape() instead of view()
    - zero_grad(set_to_none=True) for better memory efficiency
    - TF32 and cuDNN benchmarking enabled
    
    Args:
        model: Neural network model
        optimizer: Optimizer instance
        train_iter: Training data iterator
        metadata: Dataset metadata
        grad_clip: Gradient clipping value
        reverse_src: Whether to reverse source sequences
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for AMP (required if use_amp=True)
        epoch: Current epoch number
        save_path: Path to save training metrics
        vocab: Target vocabulary for logging
        src_vocab: Source vocabulary for logging (optional)
        debug: Enable per-batch logging
        log_interval: Batch interval for logging
        collect_loss_history: Whether to collect batch-level loss history
    
    Returns:
        avg_loss: Average training loss for the epoch
        stats: Dictionary with detailed statistics
    """
    # [OPTIMIZED] Enable performance optimizations
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    model.train()
    total_loss = 0
    batch_losses = [] if collect_loss_history else None
    grad_norms = []
    wen = model.encoder.rnn.weight_ih_l0
    wdn = model.decoder.rnn.weight_ih_l0
    wen_before = wen.detach().clone()
    wdn_before = wdn.detach().clone()
    #print("before updating -- encoder embed :",model.encoder.embed.weight[2][:20])
    #print("before updating -- decoder embed :",model.decoder.embed.weight[2][:20])
    #print("before updating -- encoder rnn weight :",model.encoder.rnn.weight_ih_l0)
    #print("before updating -- decoder rnn weight :",model.decoder.rnn.weight_ih_l0)
    #print("before updating -- decoder Linear weight :",model.decoder.out.weight.data)

    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    total_batches = len(train_iter)
    
    # [DEBUG] Set debug mode based on args
    set_debug_mode(debug)

    for batch_idx, batch in enumerate(tqdm(train_iter, desc="Training", leave=False)):
        question, answer = batch.question, batch.answer
        #print("src_lengths",batch.src_lengths)
        #print("question", question)
        #print("question shape", question.shape)
        #print("answer", answer)
        #print("answer shape", answer.shape)
        # [DEBUG] Print Natural Language Sentences for the first batch or periodically
        if debug and batch_idx == 0:
             print("\n" + "=" * 50)
             print(f"DEBUG: Processing Batch {batch_idx}")
             # Access first sample in batch
             # Note: question is (seq_len, batch)
             src_sample = question[:, 0]
             tgt_sample = answer[:, 0]
             
             src_text = []
             for token in src_sample:
                 idx = token.item()
                 if 0 <= idx < len(src_vocab.itos):
                      word = src_vocab.itos[idx]
                      if word not in ['<pad>']:
                           src_text.append(word)
             
             tgt_text = []
             for token in tgt_sample:
                 idx = token.item()
                 if 0 <= idx < len(vocab.itos):
                      word = vocab.itos[idx]
                      if word not in ['<pad>']:
                           tgt_text.append(word)

             print(f"  Sample Source Sentence: {' '.join(src_text)}")
             print(f"  Sample Target Sentence: {' '.join(tgt_text)}")
             print("=" * 50 + "\n")
        
        # [Feature] Reverse Source if flag is set
        # if reverse_src:
        #     question = batch_reverse_source(question, metadata.padding_idx)
        
        # [OPTIMIZED] AMP context (no-op if use_amp=False)
        with autocast(enabled=use_amp):
            logits = model(question, answer, batch.src_lengths)
            #print("logits shape : ",logits.shape)
            logits[..., 0] = float("-inf")   # V 차원 한 칸만
            # ===== [MODIFIED] Handle dynamic decoder output length =====
            # logits may be shorter than answer[1:] due to dynamic loop bounds
            actual_output_len = logits.size(0)
            
            # Slice target labels to match logits length
            target_label_forloss = answer[1:actual_output_len+1].clone()  # (actual_len, batch)
            target_label_forloss[target_label_forloss == metadata.padding_idx] = -100
            '''
            # [OPTIMIZED] reshape() instead of view()
            loss = F.cross_entropy(
                logits.reshape(-1, metadata.vocab_size),
                target_label_forloss.reshape(-1),
                ignore_index=-100
            )
            '''
            # log_softmax는 vocab 차원으로 수행해야 함
            log_probs = F.log_softmax(logits, dim=-1)  # (T, B, V)
        # nll_loss input: (N, C) or (N, C, d1, ...) 형태 가능
            # 여기서는 (T*B, V) 로 flatten
            nll_loss = F.nll_loss(
                log_probs.reshape(-1, metadata.vocab_size),     # (T*B, V)
                target_label_forloss.reshape(-1),               # (T*B,)
                ignore_index=-100
            )

            # 유효 위치 마스크: target != -100
            mask = (target_label_forloss != -100).float()  # (T,B)
            loss = (nll_loss * mask).sum() / mask.sum().clamp_min(1.0)

        # Check for NaN/Inf
        if not torch.isfinite(loss):
            print(f"\n⚠️  WARNING: Non-finite loss detected at batch {batch_idx}!")
            print(f"   Loss value: {loss.item()}")
            continue
        
        # [OPTIMIZED] AMP-aware backward and optimizer step
        if use_amp:
            # Scale loss and backward
            scaler.scale(loss).backward()
            
            # Unscale before gradient clipping (CRITICAL!)
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), grad_clip)
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            # [OPTIMIZED] More efficient than zero_grad()
            optimizer.zero_grad(set_to_none=True)
        else:
            # Standard training (FP32)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            #summarize_delta(wen_before, model.encoder.rnn.weight_ih_l0, name="enc weight_ih_l0", eps=1e-12)
            #summarize_delta(wdn_before, model.decoder.rnn.weight_ih_l0, name="dec weight_ih_l0", eps=1e-12)
            #print("after updating -- encoder embed :",model.encoder.embed.weight[2][:20])
            #print("grad nonzero count:",count_nonzero_grad_vocab(model.encoder.embed.weight.grad))
            #print("after updating -- decoder embed :",model.decoder.embed.weight[2][:20])
            #print("grad nonzero count:",count_nonzero_grad_vocab(model.decoder.embed.weight.grad))
            #print("after updating -- encoder rnn weight :",model.encoder.rnn.weight_ih_l0)
            #print("after updating -- decoder rnn weight :",model.decoder.rnn.weight_ih_l0)
            #print("after updating -- decoder Linear weight :",model.decoder.out.weight.data)
            #print("after updating -- encoder rnn grad", model.encoder.rnn.weight_ih_l0.grad is None)
            #print("after updating -- decoder rnn grad", model.decoder.rnn.weight_ih_l0.grad is None)
            #print("after updating -- decoder Linear grad", model.decoder.out.weight.grad is None)
        
        # Track statistics
        batch_loss = loss.item()
        total_loss += batch_loss
        if collect_loss_history:
            batch_losses.append(batch_loss)
        grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        
        # Log batch statistics every 100 batches
        log_batch_statistics(batch_idx, total_batches, batch_loss, grad_norms[-1], current_lr, debug, log_interval)
    
    avg_loss = total_loss / len(train_iter)
    avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0


    # Build stats dictionary
    stats = {
        'avg_grad_norm': avg_grad_norm,
    }
    
    # Only include batch-level statistics if loss history was collected
    if collect_loss_history and batch_losses:
        stats['batch_losses'] = batch_losses
        stats['min_loss'] = min(batch_losses)
        stats['max_loss'] = max(batch_losses)
    else:
        stats['min_loss'] = avg_loss
        stats['max_loss'] = avg_loss
    
    return avg_loss, stats


def adjust_learning_rate(optimizer, epoch, decay_start):
    """
    [Paper Sec 4.1] Halve learning rate every epoch after decay_start epoch.
    
    Paper: "after N epochs, we begin to halve the learning rate every epoch"
    - epoch is 0-indexed here
    - If decay_start=5, decay starts from epoch 6 (after epochs 0-5 are done)
    
    Args:
        optimizer: Optimizer instance
        epoch: Current epoch number (0-indexed)
        decay_start: Epoch after which to start decaying
    """
    if epoch > decay_start:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
            print(f"Decaying learning rate to {param_group['lr']}")


def main():
    args = parse_args()
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if cuda else 'cpu')

    print("=" * 70)
    print("Using %s for training" % ('GPU' if cuda else 'CPU'))
    
    # [OPTIMIZED] AMP setup
    use_amp = bool(getattr(args, 'amp', False))
    scaler = GradScaler(enabled=use_amp) if cuda else None
    
    if use_amp:
        print("⚡ Automatic Mixed Precision (AMP) enabled")
        print("   Expected speedup: 1.5-2x")
        print("   Note: May affect bit-level reproducibility")
    
    print("=" * 70)
    
    # ===== Dataset Loading =====
    print('Loading dataset...', end='', flush=True)
    src_metadata, tgt_metadata, src_vocab, tgt_vocab, train_iter, val_iter, test_iter = dataset_factory(args, device)
    print('Done.')

    print('Saving vocab and args...', end='', flush=True)
    # Save target vocab twice (legacy 'vocab' and explicit 'tgt_vocab') for backward compatibility; both writes use the same object.
    save_vocab(tgt_vocab, os.path.join(args.save_path, 'vocab'))
    save_vocab(src_vocab, os.path.join(args.save_path, 'src_vocab'))  # Save source vocabulary
    save_vocab(tgt_vocab, os.path.join(args.save_path, 'tgt_vocab'))  # Save target vocabulary
    save_object(args, os.path.join(args.save_path, 'args'))
    print('Done')

    # ===== Model Initialization =====
    model = train_model_factory(args, src_metadata, tgt_metadata)
    model = model.to(device)
    
    if cuda and args.multi_gpu:
        model = nn.DataParallel(model, dim=1)
    
    # [OPTIMIZED] Print model info
    if args.print_model_summary:
        print("\n" + "=" * 70)
        print("Model Architecture:")
        print("=" * 70)
        print(model)
        print("=" * 70)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        print("=" * 70)
    print("requires_grad:",model.decoder.embed.weight.requires_grad)
    print("grad:",model.decoder.embed.weight.grad)
    # ===== Optimizer Setup =====
    # [Paper] Optimizer: SGD with lr=1.0
    print('model.parameters()',model.parameters())
    for params in model.parameters():
        print(params)
        print('params.shape',params.shape)
        print('params.requires_grad',params.requires_grad)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    
    print(f"\nOptimizer: SGD")
    print(f"Initial learning rate: {args.learning_rate}")
    print(f"Learning rate decay starts after epoch: {args.lr_decay_start}")
    print(f"Gradient clipping: {args.gradient_clip}")
    print("=" * 70 + "\n")

    # ===== Training Loop =====
    try:
        best_val_loss = None
        
        for epoch in range(args.max_epochs):
            print("\n" + "=" * 70)
            print(f"Epoch {epoch + 1}/{args.max_epochs}")
            print("=" * 70)
            
            start = datetime.now()
            
            # Set epoch for batch sampler (for bucketing with epoch-wise shuffle)
            if hasattr(train_iter, 'batch_sampler') and train_iter.batch_sampler is not None:
                if hasattr(train_iter.batch_sampler, 'set_epoch'):
                    train_iter.batch_sampler.set_epoch(epoch)
                    if args.debug:
                        print(f"Batch sampler epoch set to {epoch}")
            
            # [Paper] LR Scheduling: Halve after specified epoch
            if epoch > args.lr_decay_start:
                adjust_learning_rate(optimizer, epoch, args.lr_decay_start)

            # [Feature] Teacher Forcing Scheduling: Linear decay
            # [Feature] Teacher Forcing Scheduling: Linear decay
            # Decay by 0.1 per epoch starting from epoch 1 (1.0 -> 0.9 -> 0.8 ...)
            if args.teacher_forcing_decay and epoch > 0:
                decay_amount = 0.05
                # Calculate new ratio based on initial ratio from args
                new_ratio = args.teacher_forcing_ratio - (epoch * decay_amount)
                new_ratio = max(0.0, new_ratio)
                model.teacher_forcing_ratio = new_ratio
                print(f"Teacher forcing ratio decayed to: {model.teacher_forcing_ratio:.2f}")
            else:
                 # Ensure it starts at the arg value (or reset if needed)
                 # If decay is disabled, it stays constant at args.teacher_forcing_ratio
                 model.teacher_forcing_ratio = args.teacher_forcing_ratio


            # Train for one epoch
            train_loss, train_stats = train(
                model=model,
                optimizer=optimizer,
                train_iter=train_iter,
                metadata=tgt_metadata,  # Use TGT metadata for loss/output dimension
                grad_clip=args.gradient_clip,
                reverse_src=args.reverse,
                use_amp=use_amp,
                scaler=scaler,
                epoch=epoch,
                save_path=args.save_path,
                vocab=tgt_vocab,
                src_vocab=src_vocab,
                debug=args.debug,
                log_interval=args.log_interval,
                collect_loss_history=args.plot_loss_graph
            )
            
            # Evaluate on validation set
            val_loss = evaluate(
                model=model,
                val_iter=val_iter,
                metadata=tgt_metadata,  # Use TGT metadata for loss/output dimension
                reverse_src=args.reverse,
                verbose=args.debug,
                collect_loss_history=args.plot_loss_graph
            )
            
            # Generate and display sample translations
            if args.sample_translations:
                print("\n  🔍 Sample Translations:")
                samples = generate_sample_translations(model, train_iter, tgt_metadata, src_vocab, tgt_vocab, num_samples=10)
                for i, sample in enumerate(samples, 1):
                    print(f"\n  Example {i}:")
                    print(f"    SRC: {sample['source']}")
                    print(f"    TGT: {sample['target']}")
                    print(f"    PRD: {sample['prediction']}")
            
            # Calculate metrics
            elapsed = datetime.now() - start
            train_ppl = calculate_perplexity(train_loss)
            val_ppl = calculate_perplexity(val_loss)
            
            # Print epoch summary
            print("\n" + "=" * 70)
            print(f"[Epoch {epoch + 1:2d}/{args.max_epochs}] Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
            print(f"  Avg Grad Norm: {train_stats['avg_grad_norm']:.4f}")
            print(f"  Time: {elapsed}")
            print("=" * 70)

            # Save training metrics to file (only if plotting is enabled)
            if args.plot_loss_graph:
                metrics = {
                    'train_loss': train_loss,
                    'train_ppl': train_ppl,
                    'val_loss': val_loss,
                    'val_ppl': val_ppl,
                    'avg_grad_norm': train_stats['avg_grad_norm'],
                    'min_train_loss': train_stats['min_loss'],
                    'max_train_loss': train_stats['max_loss'],
                    'time_seconds': elapsed.total_seconds()
                }
                save_training_metrics(args.save_path, epoch + 1, metrics)

            # Save model
            # Save model
            if args.save_every_epoch or not best_val_loss or val_loss < best_val_loss or epoch == args.max_epochs - 1:
                if epoch == args.max_epochs - 1:
                     print(f"\n💾 Saving model (last epoch save, val_loss: {val_loss:.4f})")
                elif best_val_loss is None:
                    print(f"\n💾 Saving model (initial save, val_loss: {val_loss:.4f})")
                elif val_loss < best_val_loss:
                    print(f"\n💾 Saving model (val_loss improved: {best_val_loss:.4f} → {val_loss:.4f})")
                else:
                    print(f"\n💾 Saving model (forced save, val_loss: {val_loss:.4f})")
                
                save_model(args.save_path, model, epoch + 1, train_loss, val_loss)
                
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            print()  # New line
            
    except (KeyboardInterrupt, BrokenPipeError):
        print('\n[Ctrl-C] Training stopped by user.')

    # ===== Final Evaluation =====
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)
    
    test_loss = evaluate(
        model=model,
        val_iter=test_iter,
        metadata=tgt_metadata,  # Use TGT metadata
        reverse_src=args.reverse,
        verbose=args.debug,
        collect_loss_history=False  # No need to collect loss history for final test
    )
    
    print(f"Test loss: {test_loss:.4f}")
    print("=" * 70)
    
    # ===== Plot Loss Graph =====
    if args.plot_loss_graph:
        metrics_file = os.path.join(args.save_path, 'training_metrics.jsonl')
        if os.path.exists(metrics_file):
            try:
                print("\n" + "=" * 70)
                print("Generating Loss Graph")
                print("=" * 70)
                train_losses, val_losses = load_training_metrics(metrics_file)
                if train_losses and val_losses:
                    loss_graph_path = os.path.join(args.save_path, 'loss_graph.png')
                    plot_loss_graph(train_losses, val_losses, loss_graph_path)
                    print(f"Loss graph saved to: {loss_graph_path}")
                else:
                    print("Warning: No loss data found in metrics file")
                print("=" * 70)
            except Exception as e:
                print(f"Error generating loss graph: {e}")
                print("=" * 70)
        else:
            print(f"\nWarning: Metrics file not found: {metrics_file}")


if __name__ == '__main__':
    main()
