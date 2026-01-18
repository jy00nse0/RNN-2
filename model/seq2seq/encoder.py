import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import RNNWrapper
from abc import ABC, abstractmethod
from .embeddings import embeddings_factory
from debug_utils import debug_tensor


def encoder_factory(args, metadata, embed=None):
    if embed is None:
        print("embed is None")
        embed = embeddings_factory(args, metadata)

    return SimpleEncoder(
        rnn_cls=getattr(nn, args.encoder_rnn_cell),
        embed=embed,
        embed_size=args.embedding_size,
        hidden_size=args.encoder_hidden_size,
        num_layers=args.encoder_num_layers,
        dropout=args.encoder_rnn_dropout,
        bidirectional=args.encoder_bidirectional
    )


class Encoder(ABC, nn.Module):
    """
    Defines encoder for seq2seq model.
    """

    @abstractmethod
    def forward(self, input, input_lengths=None, h_0=None):
        pass

    @property
    @abstractmethod
    def hidden_size(self):
        pass

    @property
    @abstractmethod
    def bidirectional(self):
        pass

    @property
    @abstractmethod
    def num_layers(self):
        pass


class SimpleEncoder(Encoder):
    """
    Encoder for seq2seq models.
    """

    def __init__(self, rnn_cls, embed, embed_size, hidden_size, num_layers=1, dropout=0.2,
                 bidirectional=False):
        super(SimpleEncoder, self).__init__()

        self._hidden_size = hidden_size
        self._bidirectional = bidirectional
        self._num_layers = num_layers
        self._dropout = dropout

        self.embed = embed

        # [Optimized] cuDNN Fused Implementation
        # PyTorch nn.LSTM's dropout argument implements Zaremba-style dropout
        # (applied to outputs of each layer except the last, NOT to recurrent connections).
        rnn_dropout = dropout if num_layers > 1 else 0.0
        
        self.rnn = rnn_cls(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout,
            bidirectional=bidirectional
        )
        
        #print(self.embed.type)
        # [DEBUG] Print 10th token embedding
        if hasattr(self.embed, 'weight'):
            print(f"Embedding for token 10: {self.embed.weight[10]}")
        else:
            print("no embedding")
        

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def bidirectional(self):
        return self._bidirectional

    @property
    def num_layers(self):
        return self._num_layers

    def forward(self, input, input_lengths=None, h_0=None):
        """
        Inputs:
            input: (seq_len, batch)
            input_lengths: (batch) - Optional, lengths of each sequence
            h_0: (num_layers * num_directions, batch, hidden_size) - Optional

        Outputs:
            outputs: (seq_len, batch, hidden_size * num_directions)
            h_n: (num_layers * num_directions, batch, hidden_size)
                 - Returns raw hidden state structure from PyTorch RNN.
                 - No concatenation/projection is done here to maintain flexibility for decoder initialization.
        """
        # [Optimization] Ensure parameter compactness for cuDNN
        # Essential for DataParallel and preventing performance degradation
        print("encoder input shape",input.shape)
        total_tokens = input.view(-1)
        print(f"[DEBUG] Encoder input total tokens (raw): {total_tokens}")
        input_lengths
        
        self.rnn.flatten_parameters()

        embedded = self.embed(input)
        
        if input_lengths is not None:
            print("Use pack_padded_sequence")
            # [Fix] Use pack_padded_sequence to ignore padding
            # enforce_sorted=False allows unsorted batch (standard in current DataLoader)
            
            # pack_padded_sequence will sort internally
            packed_embedded = pack_padded_sequence(embedded, input_lengths.cpu(), enforce_sorted=False)
            
            # [Optimized] Call cuDNN fused RNN once
            outputs, h_n = self.rnn(packed_embedded, h_0)
            
            # Unpack outputs (pad_packed_sequence restores original order for outputs)
            outputs, _ = pad_packed_sequence(outputs)
            
            # Restore h_n order (manually needed because h_n corresponds to sorted batch)
            if hasattr(packed_embedded, 'unsorted_indices'):
                 unsorted_indices = packed_embedded.unsorted_indices
                 if isinstance(h_n, tuple): # LSTM returns (h_n, c_n)
                     h_n = (h_n[0].index_select(1, unsorted_indices),
                            h_n[1].index_select(1, unsorted_indices))
                 else: # GRU returns h_n tensor
                     h_n = h_n.index_select(1, unsorted_indices)
            
            #outputs, h_n = self.rnn(embedded, h_0)
        else:
            # [Optimized] Call cuDNN fused RNN once
            outputs, h_n = self.rnn(embedded, h_0)
        
        # [DEBUG] Encoder Outputs
        debug_tensor("Encoder Outputs", outputs, context="SimpleEncoder.forward")
        
        return outputs, h_n

