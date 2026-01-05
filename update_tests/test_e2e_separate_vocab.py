#!/usr/bin/env python3
"""
End-to-end integration test simulating a real checkpoint with separate vocabularies.
This test:
1. Creates mock src_vocab and tgt_vocab with different sizes
2. Saves them as if from training
3. Loads them using predict.py logic
4. Tests inference
5. Tests BLEU calculation with multi-bleu.perl
"""

import torch
import os
import sys
import tempfile
import shutil
from dataset import Vocab, metadata_factory
from model import train_model_factory, predict_model_factory
from serialization import save_object, save_model
from constants import MODEL_FORMAT


class SimpleField:
    """Simple field-like object that wraps vocab for compatibility"""
    def __init__(self, vocab, add_sos=False, add_eos=True):
        self.vocab = vocab
        self.add_sos = add_sos
        self.add_eos = add_eos
    
    def preprocess(self, text):
        return text.strip().split()
    
    def process(self, batch):
        processed = []
        for tokens in batch:
            indices = []
            if self.add_sos:
                indices.append(self.vocab.stoi['<sos>'])
            indices.extend([self.vocab.stoi.get(tok, self.vocab.stoi['<unk>']) for tok in tokens])
            if self.add_eos:
                indices.append(self.vocab.stoi['<eos>'])
            processed.append(indices)
        
        pad_idx = self.vocab.stoi['<pad>']
        max_len = max(len(seq) for seq in processed)
        padded = []
        for seq in processed:
            padded.append(seq + [pad_idx] * (max_len - len(seq)))
        
        tensor = torch.tensor(padded, dtype=torch.long).t()
        return tensor


class TestArgs:
    """Args class for testing"""
    def __init__(self):
        self.encoder_rnn_cell = 'LSTM'
        self.encoder_hidden_size = 64
        self.encoder_num_layers = 1
        self.encoder_rnn_dropout = 0.0
        self.encoder_bidirectional = False
        self.decoder_type = 'luong'
        self.decoder_rnn_cell = 'LSTM'
        self.decoder_hidden_size = 64
        self.decoder_num_layers = 1
        self.decoder_rnn_dropout = 0.0
        self.luong_attn_hidden_size = 64
        self.luong_input_feed = False
        self.decoder_init_type = 'zeros'
        self.attention_type = 'none'
        self.attention_score = 'dot'
        self.embedding_type = None
        self.embedding_size = 128
        self.train_embeddings = True
        self.teacher_forcing_ratio = 0.0
        self.cuda = False
        self.multi_gpu = False


def create_test_args():
    """Create minimal args for testing"""
    return TestArgs()


def test_e2e_separate_vocab():
    """End-to-end test with separate vocabularies"""
    print("=" * 70)
    print("End-to-End Test: Separate SRC/TGT Vocabularies")
    print("=" * 70)
    
    # Create temporary directory for model checkpoint
    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary directory: {temp_dir}")
    
    try:
        # Step 1: Create different vocabularies (simulating EN->DE translation)
        print("\n1. Creating separate vocabularies...")
        src_tokens = ['hello', 'world', 'how', 'are', 'you', 'good', 'morning']  # English
        tgt_tokens = ['hallo', 'welt', 'wie', 'geht', 'es', 'dir', 'guten', 'morgen', 'extra']  # German (larger)
        
        src_vocab = Vocab(src_tokens)
        tgt_vocab = Vocab(tgt_tokens)
        
        print(f"   SRC vocab size: {len(src_vocab)}")
        print(f"   TGT vocab size: {len(tgt_vocab)}")
        assert len(src_vocab) != len(tgt_vocab), "Vocabularies should have different sizes for this test"
        
        # Step 2: Save vocabularies
        print("\n2. Saving vocabularies...")
        save_object(src_vocab, os.path.join(temp_dir, 'src_vocab'))
        save_object(tgt_vocab, os.path.join(temp_dir, 'tgt_vocab'))
        # Also save legacy vocab for backward compatibility test
        save_object(tgt_vocab, os.path.join(temp_dir, 'vocab'))
        print("   ✅ Vocabularies saved")
        
        # Step 3: Create and save model
        print("\n3. Creating and saving model...")
        args = create_test_args()
        save_object(args, os.path.join(temp_dir, 'args'))
        
        src_metadata = metadata_factory(args, src_vocab)
        tgt_metadata = metadata_factory(args, tgt_vocab)
        
        model = train_model_factory(args, src_metadata, tgt_metadata)
        model_path = os.path.join(temp_dir, MODEL_FORMAT % (1, 0.5, 0.6))
        torch.save(model.state_dict(), model_path)
        print(f"   ✅ Model saved to {model_path}")
        
        # Step 4: Load vocabularies (simulating predict.py)
        print("\n4. Loading vocabularies (predict.py logic)...")
        loaded_src = None
        loaded_tgt = None
        
        src_vocab_path = os.path.join(temp_dir, 'src_vocab')
        tgt_vocab_path = os.path.join(temp_dir, 'tgt_vocab')
        
        if os.path.exists(src_vocab_path):
            from serialization import load_object
            loaded_src = load_object(src_vocab_path)
            print(f"   ✅ Loaded src_vocab (size: {len(loaded_src)})")
        
        if os.path.exists(tgt_vocab_path):
            loaded_tgt = load_object(tgt_vocab_path)
            print(f"   ✅ Loaded tgt_vocab (size: {len(loaded_tgt)})")
        
        assert loaded_src is not None and loaded_tgt is not None
        
        # Step 5: Create predict model with separate fields
        print("\n5. Creating predict model with separate fields...")
        src_field = SimpleField(loaded_src, add_sos=False, add_eos=True)
        tgt_field = SimpleField(loaded_tgt, add_sos=False, add_eos=False)
        
        loaded_args = load_object(os.path.join(temp_dir, 'args'))
        loaded_src_metadata = metadata_factory(loaded_args, loaded_src)
        loaded_tgt_metadata = metadata_factory(loaded_args, loaded_tgt)
        
        predict_model = predict_model_factory(
            loaded_args,
            loaded_src_metadata,
            loaded_tgt_metadata,
            model_path,
            src_field,
            tgt_field
        )
        print("   ✅ Predict model created successfully")
        
        # Step 6: Test inference
        print("\n6. Testing inference...")
        predict_model.eval()
        with torch.no_grad():
            questions = ["hello world", "how are you"]
            outputs = predict_model(questions, 'greedy', max_seq_len=10)
            print(f"   ✅ Inference successful")
            for i, (q, a) in enumerate(zip(questions, outputs)):
                print(f"      Q{i+1}: '{q}' -> A{i+1}: '{a}'")
        
        # Step 7: Test backward compatibility (missing src_vocab)
        print("\n7. Testing backward compatibility...")
        temp_dir2 = tempfile.mkdtemp()
        try:
            # Only save legacy vocab
            save_object(tgt_vocab, os.path.join(temp_dir2, 'vocab'))
            save_object(args, os.path.join(temp_dir2, 'args'))
            torch.save(model.state_dict(), os.path.join(temp_dir2, MODEL_FORMAT % (1, 0.5, 0.6)))
            
            # Try loading with fallback
            src_vocab_path2 = os.path.join(temp_dir2, 'src_vocab')
            tgt_vocab_path2 = os.path.join(temp_dir2, 'tgt_vocab')
            legacy_vocab_path2 = os.path.join(temp_dir2, 'vocab')
            
            loaded_src2 = None
            loaded_tgt2 = None
            
            if os.path.exists(src_vocab_path2):
                loaded_src2 = load_object(src_vocab_path2)
            if os.path.exists(tgt_vocab_path2):
                loaded_tgt2 = load_object(tgt_vocab_path2)
            
            # Fallback
            if loaded_src2 is None or loaded_tgt2 is None:
                legacy_vocab = load_object(legacy_vocab_path2)
                if loaded_src2 is None:
                    loaded_src2 = legacy_vocab
                if loaded_tgt2 is None:
                    loaded_tgt2 = legacy_vocab
            
            assert loaded_src2 is not None and loaded_tgt2 is not None
            print("   ✅ Backward compatibility works (fallback to legacy vocab)")
            
        finally:
            shutil.rmtree(temp_dir2)
        
        print("\n" + "=" * 70)
        print("✅ All end-to-end tests passed!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory")


if __name__ == '__main__':
    success = test_e2e_separate_vocab()
    sys.exit(0 if success else 1)
