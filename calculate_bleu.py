#!/usr/bin/env python

import torch
import os
os.environ['PYTHONHASHSEED'] = '0'
import argparse
import subprocess
from model import predict_model_factory
from dataset import metadata_factory, LazyTranslationDataset, collate_fn  # [Modified] Import Dataset classes
from serialization import load_object
from torch.utils.data import DataLoader  # [Modified] Import DataLoader
from tqdm import tqdm
from constants import MODEL_START_FORMAT


def parse_args():
    parser = argparse.ArgumentParser(description='Script for calculating chatbot BLEU score.')
    parser.add_argument('-p', '--model-path', required=True,
                        help='Path to directory with model args, vocabulary and pre-trained pytorch models.')
    parser.add_argument('-e', '--epoch', type=int, help='Model from this epoch will be loaded.')
    parser.add_argument('-s','--sampling-strategy', choices=['greedy', 'random', 'beam_search'], default='greedy',
                        help='Strategy for sampling output sequence.')
    parser.add_argument('-r', '--reference-path', required=True, help='Path to reference file.')
    parser.add_argument('--max-seq-len', type=int, default=30, help='Maximum length for output sequence.')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda if available.')
    parser.add_argument('--lowercase', action='store_true', default=False, help='Lowercase for BLEU evaluation.')
    return parser.parse_args()


def get_model_path(dir_path, epoch):
    name_start = MODEL_START_FORMAT % epoch
    for path in os.listdir(dir_path):
        if path.startswith(name_start):
            return os.path.join(dir_path, path)
    raise ValueError("Model from epoch %d doesn't exist in %s" % (epoch, dir_path))


def get_answers(model, dataloader, args, device):
    """
    [Modified] Updated to use DataLoader
    """
    answers = []
    
    # Iterate over batches
    for batch in tqdm(dataloader, desc="Decoding Batches"):
        # Unpack batch: (src, tgt, lengths, indices) depending on collate
        # dataset.py collate_fn returns: (src_padded, tgt_padded, src_lengths)
        # But wait, collate_fn in dataset.py returns: (src_padded, tgt_padded) if no internal sort?
        # Let's verify dataset.py collate_fn signature.
        # It returns (src_padded, tgt_padded).
        
        # [Fixed] collate_fn returns 3 values: src_padded, tgt_padded, src_lengths
        src, _, _ = batch 
        src = src.to(device)
        
        # Determine batch size from the source tensor
        # src shape is (seq_len, batch_size) because batch_first=False
        current_batch_size = src.size(1)
        
        # Model forward
        # Seq2SeqPredict.forward calls decoder recursively
        batch_answers = model(src, 
                              sampling_strategy=args.sampling_strategy,
                              max_seq_len=args.max_seq_len)
        
        answers.extend(batch_answers)

    return answers


def calculate_bleu_with_perl(hypotheses, reference_path, lowercase=False):
    """
    Calculate BLEU score using multi-bleu.perl script.
    """
    script_path = os.path.join(os.path.dirname(__file__), 'multi-bleu.perl')
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(
            f"multi-bleu.perl not found at {script_path}. "
            "Please ensure the script is present in the repository root."
        )
    
    cmd = ['perl', script_path]
    if lowercase:
        cmd.append('-lc')
    cmd.append(reference_path)
    
    hypotheses_text = '\n'.join(hypotheses) + '\n'
    
    try:
        result = subprocess.run(
            cmd,
            input=hypotheses_text,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Error running multi-bleu.perl: {e}\n"
            f"stderr: {e.stderr}\n"
            f"Make sure perl is installed and multi-bleu.perl is executable.\n"
            f"You can make it executable with: chmod +x {script_path}"
        )
        print(error_msg)
        raise RuntimeError(error_msg) from e

def main():
    torch.set_grad_enabled(False)
    args = parse_args()
    model_args = load_object(os.path.join(args.model_path, 'args'))
    print(args.sampling_strategy)
    
    # Load vocabularies
    src_vocab_path = os.path.join(args.model_path, 'src_vocab')
    tgt_vocab_path = os.path.join(args.model_path, 'tgt_vocab')
    legacy_vocab_path = os.path.join(args.model_path, 'vocab')
    
    src_vocab = None
    tgt_vocab = None
    
    if os.path.exists(src_vocab_path):
        src_vocab = load_object(src_vocab_path)
    if os.path.exists(tgt_vocab_path):
        tgt_vocab = load_object(tgt_vocab_path)
    
    if src_vocab is None or tgt_vocab is None:
        if os.path.exists(legacy_vocab_path):
            legacy_vocab = load_object(legacy_vocab_path)
            if src_vocab is None: src_vocab = legacy_vocab
            if tgt_vocab is None: tgt_vocab = legacy_vocab
        else:
            raise FileNotFoundError("No vocabulary files found.")

    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if cuda else 'cpu')

    # [Modified] No more SimpleField. We use Datasets.
    
    # Determine test file paths
    # Reference path is e.g. data/test.de
    base_path, ext = args.reference_path.rsplit('.', 1)
    if ext == 'de':
        test_src_path = base_path + '.en'
        test_tgt_path = args.reference_path # .de
    elif ext == 'en':
        test_src_path = base_path + '.de'
        test_tgt_path = args.reference_path # .en
    else:
        raise ValueError(f"Unknown reference file extension: {ext}")
        
    print(f"Loading test data from:")
    print(f"  Src: {test_src_path}")
    print(f"  Tgt: {test_tgt_path}")

    # [Modified] Use LazyTranslationDataset
    # Note: We need to pass the FULL path to the dataset files
    # LazyTranslationDataset expects separate src and tgt paths if customized?
    # Actually LazyTranslationDataset takes (src_path, tgt_path, src_vocab, tgt_vocab).
    # Let's check dataset.py constructor signature to be sure.
    # __init__(self, src_path, tgt_path, src_vocab, tgt_vocab, max_len=50)
    
    # Check for reverse flag in model args
    reverse_src = getattr(model_args, 'reverse', False)
    if reverse_src:
        print("Source sentence reversal: ENABLED (from model args)")
    
    test_dataset = LazyTranslationDataset(
        src_file=test_src_path,
        tgt_file=test_tgt_path,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_len=args.max_seq_len,
        reverse_src=reverse_src
    )
    
    # [User Request] Print sample reversed sentence if enabled
    if reverse_src and len(test_dataset) > 0:
        # LazyTranslationDataset returns indices, so we decode them back to tokens to verify
        src_indices, _ = test_dataset[0]
        # Decode: indices -> tokens
        # src_vocab.itos usage
        decoded_tokens = [src_vocab.itos[idx] for idx in src_indices if idx < len(src_vocab.itos)]
        print(f"\n[DEBUG] Sample Reversed Source (Index 0):")
        print(f"  Processed Tokens (Decoded): {decoded_tokens}")
    pad_idx=0
    # DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        collate_fn=lambda b:  collate_fn(b, pad_idx,pad_idx),
        shuffle=False, # Important: Must NOT shuffle to match reference order!
        num_workers=4,
        pin_memory=cuda,
        
    )

    # Load Metadata
    tgt_metadata = metadata_factory(model_args, tgt_vocab)
    src_metadata = metadata_factory(model_args, src_vocab)

    # Note: aredict_model_factory might still expect fields if it uses them?
    # Let's verify predict_model_factory signature.
    # It passes src_field, tgt_field to Seq2SeqPredict.
    # Seq2SeqPredict uses them for .vocab attribute (sos_idx, eos_idx).
    # We can pass the vocabs wrapped in a simple Namespace or just pass the vocab objects 
    # if we modify the factory call, or we can just pass the Dataset/Vocab directly if the factory supports it.
    # Waiting... the factory code (as recalled) takes (..., src_field, tgt_field).
    # We should construct dummy fields that behave like the old ones just for the factory signature,
    # OR we can just pass objects that have a .vocab attribute.
    
    class DummyField:
        def __init__(self, vocab):
            self.vocab = vocab
            
    src_dummy = DummyField(src_vocab)
    tgt_dummy = DummyField(tgt_vocab)

    model = predict_model_factory(model_args, src_metadata, tgt_metadata, get_model_path(args.model_path, args.epoch), src_dummy, tgt_dummy)
    model = model.to(device)
    model.eval()

    # [DEBUG] Register hook to inspect h_n (Hidden State) passed to Decoder
    # User Request: first 50 sentences, print source sentence and h_n vector (first 20 dims, distribution)
    # Since we use DataLoader, we can just hook the encoder.
    
    # [DEBUG] Register hook to inspect h_n (Hidden State) passed to Decoder
    # User Request: first 50 sentences, print source sentence and h_n vector (first 20 dims, distribution)
    # Since we use DataLoader, we can just hook the encoder.
    import numpy as np # Ensure numpy is available for hook

    def debug_hn_hook(module, input, output):
        # output of encoder is (encoder_outputs, h_n)
        _, h_n = output
        
        # h_n shape: (num_layers*directions, batch, hidden)
        # We want the last layer's hidden state
        # Assume LSTM tuple (h_n, c_n)
        if isinstance(h_n, tuple):
            h = h_n[0]
        else:
            h = h_n
            
        # Take the last layer
        h_last = h[-1] # (batch, hidden_size)
        
        # We only want to print for the first few samples to avoid spam (e.g. first batch)
        # We can use a counter or just print if it's the first call
        if not hasattr(debug_hn_hook, 'called'):
            debug_hn_hook.called = True
            print("\n🔍 [DEBUG] Inspecting Encoder h_n (First Batch)")
            batch_size = h_last.size(0)
            limit = min(5, batch_size) # Print for first 5 samples
            
            collected_vecs = []
            
            for i in range(limit):
                vec = h_last[i].detach().cpu().numpy()
                collected_vecs.append(vec)
                print(f"\nSample {i}:")
                print(f"  h_n[0:20]: {vec[:20]}")
                print(f"  Stats: Mean={vec.mean():.4f}, Std={vec.std():.4f}, Min={vec.min():.4f}, Max={vec.max():.4f}")
            
            # Calculate Pairwise Distances
            print("\n📊 [DEBUG] Pairwise Euclidean Distances (Sample 0-4):")
            print("      ", end="")
            for i in range(limit):
                print(f"S{i}       ", end="")
            print()
            
            for i in range(limit):
                print(f"S{i}:   ", end="")
                for j in range(limit):
                    dist = np.linalg.norm(collected_vecs[i] - collected_vecs[j])
                    print(f"{dist:.4f}   ", end="")
                print()

    # Register the hook
    model.encoder.register_forward_hook(debug_hn_hook)
    
    # Generate Answers
    answers = get_answers(model, test_loader, args, device)
    
    # Print sample
    # We don't have easy access to raw text 'questions' list anymore unless we read file again.
    # We can read the first few lines of the file for display purposes.
    with open(test_src_path, 'r', encoding='utf-8') as f:
        sample_questions = [f.readline().strip() for _ in range(5)]
    
    print("\n--- Sample Predictions ---")
    for i in range(min(5, len(answers))):
        print(f"Q: {sample_questions[i]}")
        print(f"A: {answers[i]}\n")
    
    # Calculate BLEU
    bleu_output = calculate_bleu_with_perl(answers, args.reference_path, lowercase=args.lowercase)
    print(bleu_output)


if __name__ == '__main__':
    main()
