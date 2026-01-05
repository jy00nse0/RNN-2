#!/usr/bin/env python3
"""
Test to verify the CUDA device fix for Seq2SeqPredict model.
This test ensures that input tensors are moved to the correct device.
"""

import torch
import torch.nn as nn
from model.seq2seq.model import Seq2SeqPredict
from dataset import Vocab


class DummyEncoder(nn.Module):
    """Dummy encoder for testing"""
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        
    def forward(self, x, h_0=None):
        emb = self.embed(x)
        out, (h, c) = self.rnn(emb, h_0)
        return out, (h, c)


class DummyDecoder(nn.Module):
    """Dummy decoder for testing"""
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, t, input, encoder_outputs, h_n, **kwargs):
        # Simplified decoder forward
        emb = self.embed(input)
        if 'last_state' in kwargs:
            rnn_out, new_state = self.rnn(emb.unsqueeze(0), kwargs['last_state'])
        else:
            batch_size = input.size(0)
            h0 = torch.zeros(1, batch_size, self.rnn.hidden_size, device=input.device)
            c0 = torch.zeros(1, batch_size, self.rnn.hidden_size, device=input.device)
            rnn_out, new_state = self.rnn(emb.unsqueeze(0), (h0, c0))
        
        output = self.out(rnn_out.squeeze(0))
        return output, None, {'last_state': new_state}
    
    @property
    def has_attention(self):
        return False


class SimpleField:
    """Simple field-like object that wraps vocab for compatibility"""
    def __init__(self, vocab):
        self.vocab = vocab
    
    def preprocess(self, text):
        """Tokenize text by splitting on whitespace"""
        return text.strip().split()
    
    def process(self, batch):
        """Convert list of token lists to tensor"""
        max_len = max(len(tokens) for tokens in batch)
        # Add <sos> and <eos> tokens
        processed = []
        for tokens in batch:
            indices = (
                [self.vocab.stoi['<sos>']] +
                [self.vocab.stoi.get(tok, self.vocab.stoi['<unk>']) for tok in tokens] +
                [self.vocab.stoi['<eos>']]
            )
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


def test_device_consistency():
    """Test that Seq2SeqPredict moves input tensors to the correct device"""
    
    # Create a simple vocabulary
    tokens = ['hello', 'world', 'test', 'foo', 'bar']
    vocab = Vocab(tokens)
    
    # Create dummy encoder and decoder
    hidden_size = 64
    encoder = DummyEncoder(len(vocab), hidden_size)
    decoder = DummyDecoder(len(vocab), hidden_size)
    
    # Create field
    field = SimpleField(vocab)
    
    # Create Seq2SeqPredict model
    model = Seq2SeqPredict(encoder, decoder, field)
    
    # Test on CPU first
    print("Testing on CPU...")
    model = model.to('cpu')
    model.eval()
    
    with torch.no_grad():
        questions = ["hello world", "test foo bar"]
        try:
            answers = model(questions, sampling_strategy='greedy', max_seq_len=10)
            print(f"CPU test passed. Generated {len(answers)} answers.")
        except RuntimeError as e:
            if "device" in str(e).lower():
                print(f"FAILED on CPU: {e}")
                return False
            raise
    
    # Test on CUDA if available
    if torch.cuda.is_available():
        print("\nTesting on CUDA...")
        model = model.to('cuda')
        model.eval()
        
        with torch.no_grad():
            questions = ["hello world", "test foo bar"]
            try:
                answers = model(questions, sampling_strategy='greedy', max_seq_len=10)
                print(f"CUDA test passed. Generated {len(answers)} answers.")
            except RuntimeError as e:
                if "device" in str(e).lower():
                    print(f"FAILED on CUDA: {e}")
                    return False
                raise
    else:
        print("\nCUDA not available, skipping CUDA test.")
    
    print("\n✅ All device consistency tests passed!")
    return True


if __name__ == '__main__':
    success = test_device_consistency()
    exit(0 if success else 1)
