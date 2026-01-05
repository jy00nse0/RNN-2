#!/usr/bin/env python3
"""
Test script to verify that inference works with separate src/tgt vocabularies.
This test ensures:
1. SimpleField correctly processes source and target sequences
2. Seq2SeqPredict accepts separate src_field and tgt_field
3. Source preprocessing matches training (add <eos> only, no <sos>)
4. Backward compatibility with legacy single vocab works
"""

import torch
import argparse
import os
import sys
from dataset import Vocab, metadata_factory
from model import train_model_factory
from model.seq2seq.model import Seq2SeqPredict


class SimpleField:
    """Simple field-like object that wraps vocab for compatibility"""
    def __init__(self, vocab, add_sos=False, add_eos=True):
        """
        Args:
            vocab: Vocabulary object
            add_sos: Whether to add <sos> token when processing
            add_eos: Whether to add <eos> token when processing
        """
        self.vocab = vocab
        self.add_sos = add_sos
        self.add_eos = add_eos
    
    def preprocess(self, text):
        """Tokenize text by splitting on whitespace"""
        return text.strip().split()
    
    def process(self, batch):
        """Convert list of token lists to tensor"""
        # Process tokens to indices
        processed = []
        for tokens in batch:
            indices = []
            if self.add_sos:
                indices.append(self.vocab.stoi['<sos>'])
            indices.extend([self.vocab.stoi.get(tok, self.vocab.stoi['<unk>']) for tok in tokens])
            if self.add_eos:
                indices.append(self.vocab.stoi['<eos>'])
            processed.append(indices)
        
        # Pad to same length
        pad_idx = self.vocab.stoi['<pad>']
        max_len = max(len(seq) for seq in processed)
        padded = []
        for seq in processed:
            padded.append(seq + [pad_idx] * (max_len - len(seq)))
        
        # Convert to tensor (seq_len, batch_size)
        tensor = torch.tensor(padded, dtype=torch.long).t()
        return tensor


def create_test_args():
    """Create minimal args for testing"""
    class Args:
        def __init__(self):
            # Encoder
            self.encoder_rnn_cell = 'LSTM'
            self.encoder_hidden_size = 128
            self.encoder_num_layers = 2
            self.encoder_rnn_dropout = 0.0
            self.encoder_bidirectional = False
            
            # Decoder
            self.decoder_type = 'luong'
            self.decoder_rnn_cell = 'LSTM'
            self.decoder_hidden_size = 128
            self.decoder_num_layers = 2
            self.decoder_rnn_dropout = 0.0
            self.luong_attn_hidden_size = 128
            self.luong_input_feed = False
            self.decoder_init_type = 'zeros'
            
            # Attention
            self.attention_type = 'none'
            self.attention_score = 'dot'
            
            # Embedding
            self.embedding_type = None
            self.embedding_size = 256
            self.train_embeddings = True
            
            # Training
            self.teacher_forcing_ratio = 0.0
    
    return Args()


def test_simple_field():
    """Test SimpleField with different configurations"""
    print("\n1. Testing SimpleField preprocessing...")
    
    # Create test vocab
    tokens = ['hello', 'world', 'this', 'is', 'a', 'test']
    vocab = Vocab(tokens)
    
    # Test source field (no <sos>, add <eos>)
    src_field = SimpleField(vocab, add_sos=False, add_eos=True)
    text = "hello world"
    preprocessed = src_field.preprocess(text)
    assert preprocessed == ['hello', 'world'], f"Expected ['hello', 'world'], got {preprocessed}"
    
    processed = src_field.process([preprocessed])
    # Should be: [hello, world, <eos>] transposed
    assert processed.shape == (3, 1), f"Expected shape (3, 1), got {processed.shape}"
    
    # Verify indices
    expected_indices = [vocab.stoi['hello'], vocab.stoi['world'], vocab.stoi['<eos>']]
    actual_indices = processed.squeeze().tolist()
    assert actual_indices == expected_indices, f"Expected {expected_indices}, got {actual_indices}"
    
    print("   ✅ Source field works correctly (adds <eos> only)")
    
    # Test target field (no <sos>, no <eos> - managed by Seq2SeqPredict)
    tgt_field = SimpleField(vocab, add_sos=False, add_eos=False)
    processed_tgt = tgt_field.process([preprocessed])
    # Should be: [hello, world] transposed
    assert processed_tgt.shape == (2, 1), f"Expected shape (2, 1), got {processed_tgt.shape}"
    
    print("   ✅ Target field works correctly (no special tokens)")
    
    return True


def test_seq2seq_predict_with_separate_fields():
    """Test Seq2SeqPredict with separate src and tgt fields"""
    print("\n2. Testing Seq2SeqPredict with separate fields...")
    
    args = create_test_args()
    
    # Create different sized vocabularies to simulate the real scenario
    src_tokens = ['apple', 'banana', 'cherry', 'date', 'elderberry']
    tgt_tokens = ['apfel', 'banane', 'kirsche', 'dattel', 'holunder', 'extra1', 'extra2']
    
    src_vocab = Vocab(src_tokens)
    tgt_vocab = Vocab(tgt_tokens)
    
    print(f"   SRC vocab size: {len(src_vocab)}")
    print(f"   TGT vocab size: {len(tgt_vocab)}")
    
    # Create metadata
    src_metadata = metadata_factory(args, src_vocab)
    tgt_metadata = metadata_factory(args, tgt_vocab)
    
    # Build model
    train_model = train_model_factory(args, src_metadata, tgt_metadata)
    
    # Create fields
    src_field = SimpleField(src_vocab, add_sos=False, add_eos=True)
    tgt_field = SimpleField(tgt_vocab, add_sos=False, add_eos=False)
    
    # Create predict model
    predict_model = Seq2SeqPredict(
        train_model.encoder,
        train_model.decoder,
        src_field,
        tgt_field
    )
    
    print("   ✅ Seq2SeqPredict created with separate fields")
    
    # Test inference
    predict_model.eval()
    with torch.no_grad():
        questions = ["apple banana", "cherry"]
        try:
            outputs = predict_model(questions, 'greedy', max_seq_len=10)
            print(f"   ✅ Inference successful, got {len(outputs)} outputs")
            for i, output in enumerate(outputs):
                print(f"      Input: '{questions[i]}' -> Output: '{output}'")
        except Exception as e:
            print(f"   ❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def test_backward_compatibility():
    """Test that single vocab still works (backward compatibility)"""
    print("\n3. Testing backward compatibility with single vocab...")
    
    args = create_test_args()
    
    # Use same vocab for both
    tokens = ['hello', 'world', 'test']
    vocab = Vocab(tokens)
    
    metadata = metadata_factory(args, vocab)
    
    # Build model
    train_model = train_model_factory(args, metadata, metadata)
    
    # Create fields (both using same vocab)
    src_field = SimpleField(vocab, add_sos=False, add_eos=True)
    tgt_field = SimpleField(vocab, add_sos=False, add_eos=False)
    
    # Create predict model
    predict_model = Seq2SeqPredict(
        train_model.encoder,
        train_model.decoder,
        src_field,
        tgt_field
    )
    
    # Test inference
    predict_model.eval()
    with torch.no_grad():
        questions = ["hello world"]
        try:
            outputs = predict_model(questions, 'greedy', max_seq_len=5)
            print(f"   ✅ Backward compatibility works, got output: '{outputs[0]}'")
        except Exception as e:
            print(f"   ❌ Backward compatibility test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def main():
    print("=" * 70)
    print("Testing Separate SRC/TGT Vocabulary Inference")
    print("=" * 70)
    
    tests = [
        test_simple_field,
        test_seq2seq_predict_with_separate_fields,
        test_backward_compatibility,
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
                print(f"   ❌ Test {test.__name__} failed")
        except Exception as e:
            all_passed = False
            print(f"   ❌ Test {test.__name__} raised exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All tests passed!")
        print("=" * 70)
        return 0
    else:
        print("❌ Some tests failed")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
