#!/usr/bin/env python3
"""
Unit tests for decoder_init.py module, specifically for EncoderLastStateInit class.
Tests the Sutskever/Luong approach of using encoder's last state to initialize decoder.
"""
import unittest
import torch
from model.seq2seq.decoder_init import EncoderLastStateInit, ZerosInit, BahdanauInit
from constants import LSTM, GRU


class TestEncoderLastStateInit(unittest.TestCase):
    """Test cases for EncoderLastStateInit class."""
    
    def test_gru_same_layers(self):
        """Test GRU encoder to GRU decoder with same number of layers."""
        batch_size = 4
        num_layers = 3
        hidden_size = 256
        
        # Create init module
        init_module = EncoderLastStateInit(
            decoder_num_layers=num_layers,
            decoder_hidden_size=hidden_size,
            rnn_cell_type=GRU
        )
        
        # Simulate encoder final hidden state (GRU)
        encoder_h = torch.randn(num_layers, batch_size, hidden_size)
        
        # Forward pass
        decoder_h = init_module(encoder_h)
        
        # Assertions
        self.assertIsInstance(decoder_h, torch.Tensor)
        self.assertEqual(decoder_h.shape, (num_layers, batch_size, hidden_size))
        # Should be the same as input (no transformation)
        self.assertTrue(torch.equal(decoder_h, encoder_h))
    
    def test_lstm_same_layers(self):
        """Test LSTM encoder to LSTM decoder with same number of layers."""
        batch_size = 4
        num_layers = 3
        hidden_size = 256
        
        # Create init module
        init_module = EncoderLastStateInit(
            decoder_num_layers=num_layers,
            decoder_hidden_size=hidden_size,
            rnn_cell_type=LSTM
        )
        
        # Simulate encoder final hidden state (LSTM) - tuple of (h_n, c_n)
        encoder_h = torch.randn(num_layers, batch_size, hidden_size)
        encoder_c = torch.randn(num_layers, batch_size, hidden_size)
        
        # Forward pass
        decoder_h, decoder_c = init_module((encoder_h, encoder_c))
        
        # Assertions
        self.assertIsInstance(decoder_h, torch.Tensor)
        self.assertIsInstance(decoder_c, torch.Tensor)
        self.assertEqual(decoder_h.shape, (num_layers, batch_size, hidden_size))
        self.assertEqual(decoder_c.shape, (num_layers, batch_size, hidden_size))
        # Should be the same as input
        self.assertTrue(torch.equal(decoder_h, encoder_h))
        self.assertTrue(torch.equal(decoder_c, encoder_c))
    
    def test_decoder_more_layers_padding(self):
        """Test padding when decoder has more layers than encoder."""
        batch_size = 4
        encoder_layers = 2
        decoder_layers = 4
        hidden_size = 256
        
        # Create init module
        init_module = EncoderLastStateInit(
            decoder_num_layers=decoder_layers,
            decoder_hidden_size=hidden_size,
            rnn_cell_type=GRU
        )
        
        # Simulate encoder final hidden state
        encoder_h = torch.randn(encoder_layers, batch_size, hidden_size)
        
        # Forward pass
        decoder_h = init_module(encoder_h)
        
        # Assertions
        self.assertEqual(decoder_h.shape, (decoder_layers, batch_size, hidden_size))
        # First encoder_layers should match encoder output
        self.assertTrue(torch.equal(decoder_h[:encoder_layers], encoder_h))
        # Padded layers should be zeros
        self.assertTrue(torch.equal(
            decoder_h[encoder_layers:],
            torch.zeros(decoder_layers - encoder_layers, batch_size, hidden_size)
        ))
    
    def test_encoder_more_layers_slicing(self):
        """Test slicing when encoder has more layers than decoder."""
        batch_size = 4
        encoder_layers = 5
        decoder_layers = 3
        hidden_size = 256
        
        # Create init module
        init_module = EncoderLastStateInit(
            decoder_num_layers=decoder_layers,
            decoder_hidden_size=hidden_size,
            rnn_cell_type=GRU
        )
        
        # Simulate encoder final hidden state
        encoder_h = torch.randn(encoder_layers, batch_size, hidden_size)
        
        # Forward pass
        decoder_h = init_module(encoder_h)
        
        # Assertions
        self.assertEqual(decoder_h.shape, (decoder_layers, batch_size, hidden_size))
        # Should be first decoder_layers from encoder
        self.assertTrue(torch.equal(decoder_h, encoder_h[:decoder_layers]))
    
    def test_lstm_padding_both_states(self):
        """Test that both hidden and cell states are padded for LSTM."""
        batch_size = 4
        encoder_layers = 2
        decoder_layers = 4
        hidden_size = 256
        
        # Create init module
        init_module = EncoderLastStateInit(
            decoder_num_layers=decoder_layers,
            decoder_hidden_size=hidden_size,
            rnn_cell_type=LSTM
        )
        
        # Simulate encoder final states
        encoder_h = torch.randn(encoder_layers, batch_size, hidden_size)
        encoder_c = torch.randn(encoder_layers, batch_size, hidden_size)
        
        # Forward pass
        decoder_h, decoder_c = init_module((encoder_h, encoder_c))
        
        # Assertions for hidden state
        self.assertEqual(decoder_h.shape, (decoder_layers, batch_size, hidden_size))
        self.assertTrue(torch.equal(decoder_h[:encoder_layers], encoder_h))
        self.assertTrue(torch.equal(
            decoder_h[encoder_layers:],
            torch.zeros(decoder_layers - encoder_layers, batch_size, hidden_size)
        ))
        
        # Assertions for cell state
        self.assertEqual(decoder_c.shape, (decoder_layers, batch_size, hidden_size))
        self.assertTrue(torch.equal(decoder_c[:encoder_layers], encoder_c))
        self.assertTrue(torch.equal(
            decoder_c[encoder_layers:],
            torch.zeros(decoder_layers - encoder_layers, batch_size, hidden_size)
        ))
    
    def test_lstm_slicing_both_states(self):
        """Test that both hidden and cell states are sliced for LSTM."""
        batch_size = 4
        encoder_layers = 5
        decoder_layers = 3
        hidden_size = 256
        
        # Create init module
        init_module = EncoderLastStateInit(
            decoder_num_layers=decoder_layers,
            decoder_hidden_size=hidden_size,
            rnn_cell_type=LSTM
        )
        
        # Simulate encoder final states
        encoder_h = torch.randn(encoder_layers, batch_size, hidden_size)
        encoder_c = torch.randn(encoder_layers, batch_size, hidden_size)
        
        # Forward pass
        decoder_h, decoder_c = init_module((encoder_h, encoder_c))
        
        # Assertions
        self.assertEqual(decoder_h.shape, (decoder_layers, batch_size, hidden_size))
        self.assertEqual(decoder_c.shape, (decoder_layers, batch_size, hidden_size))
        self.assertTrue(torch.equal(decoder_h, encoder_h[:decoder_layers]))
        self.assertTrue(torch.equal(decoder_c, encoder_c[:decoder_layers]))
    
    def test_gru_to_lstm_edge_case(self):
        """Test edge case: GRU encoder to LSTM decoder (cell state initialized to zeros)."""
        batch_size = 4
        num_layers = 3
        hidden_size = 256
        
        # Create init module for LSTM decoder
        init_module = EncoderLastStateInit(
            decoder_num_layers=num_layers,
            decoder_hidden_size=hidden_size,
            rnn_cell_type=LSTM
        )
        
        # Simulate GRU encoder final hidden state (no cell state)
        encoder_h = torch.randn(num_layers, batch_size, hidden_size)
        
        # Forward pass
        decoder_h, decoder_c = init_module(encoder_h)
        
        # Assertions
        self.assertEqual(decoder_h.shape, (num_layers, batch_size, hidden_size))
        self.assertEqual(decoder_c.shape, (num_layers, batch_size, hidden_size))
        self.assertTrue(torch.equal(decoder_h, encoder_h))
        # Cell state should be zeros
        self.assertTrue(torch.equal(decoder_c, torch.zeros_like(decoder_h)))
    
    def test_device_preservation(self):
        """Test that device and dtype are preserved."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        batch_size = 4
        num_layers = 3
        hidden_size = 256
        
        # Create init module
        init_module = EncoderLastStateInit(
            decoder_num_layers=num_layers,
            decoder_hidden_size=hidden_size,
            rnn_cell_type=GRU
        )
        
        # Simulate encoder final hidden state on GPU
        encoder_h = torch.randn(num_layers, batch_size, hidden_size, device='cuda')
        
        # Forward pass
        decoder_h = init_module(encoder_h)
        
        # Assertions
        self.assertEqual(decoder_h.device, encoder_h.device)
        self.assertEqual(decoder_h.dtype, encoder_h.dtype)
    
    def test_dtype_preservation(self):
        """Test that dtype is preserved (e.g., float16)."""
        batch_size = 4
        num_layers = 3
        hidden_size = 256
        
        # Create init module
        init_module = EncoderLastStateInit(
            decoder_num_layers=num_layers,
            decoder_hidden_size=hidden_size,
            rnn_cell_type=GRU
        )
        
        # Simulate encoder final hidden state with float16
        encoder_h = torch.randn(num_layers, batch_size, hidden_size, dtype=torch.float16)
        
        # Forward pass
        decoder_h = init_module(encoder_h)
        
        # Assertions
        self.assertEqual(decoder_h.dtype, torch.float16)


class TestEncoderLastStateInitComparison(unittest.TestCase):
    """Test that EncoderLastStateInit behaves differently from ZerosInit."""
    
    def test_different_from_zeros_init(self):
        """Verify that EncoderLastStateInit uses encoder state, not zeros."""
        batch_size = 4
        num_layers = 3
        hidden_size = 256
        
        # Create both init modules
        encoder_init = EncoderLastStateInit(
            decoder_num_layers=num_layers,
            decoder_hidden_size=hidden_size,
            rnn_cell_type=GRU
        )
        
        zeros_init = ZerosInit(
            decoder_num_layers=num_layers,
            decoder_hidden_size=hidden_size,
            rnn_cell_type=GRU
        )
        
        # Simulate encoder final hidden state
        encoder_h = torch.randn(num_layers, batch_size, hidden_size)
        
        # Forward pass
        decoder_h_encoder = encoder_init(encoder_h)
        decoder_h_zeros = zeros_init(encoder_h)
        
        # EncoderLastStateInit should use encoder state
        self.assertTrue(torch.equal(decoder_h_encoder, encoder_h))
        
        # ZerosInit should return zeros
        self.assertTrue(torch.equal(decoder_h_zeros, torch.zeros(num_layers, batch_size, hidden_size)))
        
        # They should be different
        self.assertFalse(torch.equal(decoder_h_encoder, decoder_h_zeros))


if __name__ == '__main__':
    unittest.main()
