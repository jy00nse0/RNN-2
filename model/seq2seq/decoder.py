import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from .embeddings import embeddings_factory
from .attention import attention_factory
from .decoder_init import decoder_init_factory
from debug_utils import debug_tensor


def bahdanau_decoder_factory(args, embed, attn, init, metadata):
    return BahdanauDecoder(
        # [Fix] Bug Corrected: args.encoder_rnn_cell -> args.decoder_rnn_cell
        rnn_cls=getattr(nn, args.decoder_rnn_cell),
        embed=embed,
        attn=attn,
        init_hidden=init,
        vocab_size=metadata.vocab_size,
        embed_size=args.embedding_size,
        rnn_hidden_size=args.decoder_hidden_size,
        encoder_hidden_size=args.encoder_hidden_size * (2 if args.encoder_bidirectional else 1),
        num_layers=args.decoder_num_layers,
        dropout=args.decoder_rnn_dropout,
        pad_idx=metadata.padding_idx
    )


def luong_decoder_factory(args, embed, attn, init, metadata):
    return LuongDecoder(
        # [Fix] Bug Corrected: args.encoder_rnn_cell -> args.decoder_rnn_cell
        rnn_cls=getattr(nn, args.decoder_rnn_cell),
        embed=embed,
        attn=attn,
        init_hidden=init,
        vocab_size=metadata.vocab_size,
        embed_size=args.embedding_size,
        rnn_hidden_size=args.decoder_hidden_size,
        attn_hidden_projection_size=args.luong_attn_hidden_size,
        encoder_hidden_size=args.encoder_hidden_size * (2 if args.encoder_bidirectional else 1),
        num_layers=args.decoder_num_layers,
        dropout=args.decoder_rnn_dropout,
        input_feed=args.luong_input_feed,
        pad_idx=metadata.padding_idx
    )


decoder_map = {
    'bahdanau': bahdanau_decoder_factory,
    'luong': luong_decoder_factory
}


def decoder_factory(args, metadata, embed=None):
    if embed is None:
        embed = embeddings_factory(args, metadata)

    attn = attention_factory(args)
    init = decoder_init_factory(args)
    return decoder_map[args.decoder_type](args, embed, attn, init, metadata)


class Decoder(ABC, nn.Module):
    """
    Base Decoder class for seq2seq models.
    """

    def __init__(self, *args):
        super(Decoder, self).__init__()
        self._args = []
        self._args_init = {}

    def forward(self, t, input, encoder_outputs, h_n, encoder_mask=None, **kwargs):
        assert (t == 0 and not kwargs) or (t > 0 and kwargs)

        extra_args = []
        for arg in self.args:
            if t > 0 and arg not in kwargs:
                raise AttributeError("Mandatory arg \"%s\" not present among method arguments" % arg)
            extra_args.append(self.args_init[arg](encoder_outputs, h_n) if t == 0 else kwargs[arg])

        output, attn_weights, *out = self._forward(t, input, encoder_outputs, *extra_args, encoder_mask=encoder_mask)
        return output, attn_weights, {k: v for k, v in zip(self.args, out)}

    @abstractmethod
    def _forward(self, *args):
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_size(self):
        raise AttributeError

    @property
    @abstractmethod
    def num_layers(self):
        raise AttributeError

    @property
    @abstractmethod
    def has_attention(self):
        raise AttributeError

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, value):
        self._args = value

    @property
    def args_init(self):
        return self._args_init

    @args_init.setter
    def args_init(self, value):
        self._args_init = value


class BahdanauDecoder(Decoder):
    """
    Bahdanau Decoder Implementation.
    
    [Note on Architecture]
    This implementation includes a Maxout layer (k=2) before the final projection.
    This reduces the hidden dimension by half (rnn_hidden_size // 2).
    Therefore, rnn_hidden_size MUST be an even number.
    """

    LAST_STATE = 'last_state'
    args = [LAST_STATE]

    def __init__(self, rnn_cls, embed, attn, init_hidden, vocab_size, embed_size, rnn_hidden_size, encoder_hidden_size,
                 num_layers=1, dropout=0.2, **kwargs):
        super(BahdanauDecoder, self).__init__()

        self.args_init = {
            self.LAST_STATE: lambda encoder_outputs, h_n: self.initial_hidden(h_n)
        }

        # [Constraint] Maxout layer (k=2) requires even hidden size
        if rnn_hidden_size % 2 != 0:
            raise ValueError(f'RNN hidden size ({rnn_hidden_size}) must be divisible by 2 because of maxout layer (k=2).')

        self._hidden_size = rnn_hidden_size
        self._num_layers = num_layers
        self._hidden_size = rnn_hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._pad_idx = kwargs.get('pad_idx')

        self.initial_hidden = init_hidden
        self.embed = embed
        
        # [Optimized] cuDNN Fused Implementation
        rnn_dropout = dropout if num_layers > 1 else 0.0
        
        # Bahdanau: Input = Embedding + Context (concatenated)
        input_size = embed_size + encoder_hidden_size
        
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout
        )
            
        self.attn = attn
        # Maxout reduces dimension by factor of 2 (k=2)
        self.out = nn.Linear(in_features=rnn_hidden_size // 2, out_features=vocab_size)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def has_attention(self):
        return True

    def _forward(self, t, input, encoder_outputs, last_state, encoder_mask=None):
        # [Optimized] Flatten parameters for cuDNN performance
        self.rnn.flatten_parameters()

        embedded = self.embed(input)

        # last_state: LSTM=(h, c), GRU=h
        # Attention uses the *last layer* hidden state of the previous timestamp
        if isinstance(last_state, tuple):
            last_hidden_top = last_state[0][-1]
        else:
            last_hidden_top = last_state[-1]

        # prepare rnn input
        attn_weights, context = self.attn(t, last_hidden_top, encoder_outputs, mask=encoder_mask)
        rnn_input = torch.cat([embedded, context], dim=1)
        rnn_input = rnn_input.unsqueeze(0)  # (1, batch, embed + enc_h)

        # [Optimized] Single Fused Call
        _, state = self.rnn(rnn_input, last_state)
        
        # Extract hidden state for output calculation
        hidden = state[0] if isinstance(state, tuple) else state

        # [Maxout Layer] k=2
        # Project: (batch, rnn_hidden) -> (batch, rnn_hidden/2, 2) -> Max -> (batch, rnn_hidden/2)
        top_layer_hidden = hidden[-1]
        batch_size = top_layer_hidden.size(0)
        maxout_input = top_layer_hidden.view(batch_size, -1, 2)
        maxout, _ = maxout_input.max(dim=2)

        output = self.out(maxout)
        
        # [Fix] Mask pad_idx in logits to ensure probability is 0 after softmax (in CrossEntropyLoss)
        if self._pad_idx is not None:
             output[:, self._pad_idx] = float('-inf')

        return output, attn_weights, state


class LuongDecoder(Decoder):
    """
    Luong Decoder Implementation.
    """
    LAST_STATE = 'last_state'
    LAST_ATTN_HIDDEN = 'last_attn_hidden'

    args = [LAST_STATE]

    def __init__(self, rnn_cls, embed, attn, init_hidden, vocab_size, embed_size, rnn_hidden_size,
                 attn_hidden_projection_size, encoder_hidden_size, num_layers=1, dropout=0.2, input_feed=False, **kwargs):
        super(LuongDecoder, self).__init__()

        if input_feed:
            self.args += [self.LAST_ATTN_HIDDEN]

        self.args_init = {
            self.LAST_STATE: lambda encoder_outputs, h_n: self.initial_hidden(h_n),
            self.LAST_ATTN_HIDDEN: lambda encoder_outputs, h_n: self.last_attn_hidden_init(h_n)
        }

        self._hidden_size = rnn_hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._hidden_size = rnn_hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._pad_idx = kwargs.get('pad_idx')
        self.initial_hidden = init_hidden

        self.input_feed = input_feed
        self.attn_hidden_projection_size = attn_hidden_projection_size

        rnn_input_size = embed_size + (attn_hidden_projection_size if input_feed else 0)
        self.embed = embed
        
        # [Optimized] cuDNN Fused Implementation
        rnn_dropout = dropout if num_layers > 1 else 0.0
        
        self.rnn = rnn_cls(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout
        )
            
        self.attn = attn
        
        if attn is not None:
            self.attn_hidden_lin = nn.Linear(in_features=rnn_hidden_size + encoder_hidden_size,
                                             out_features=attn_hidden_projection_size)
            self.out = nn.Linear(in_features=attn_hidden_projection_size, out_features=vocab_size)
            with torch.no_grad():

                self.out.weight.data.uniform_(-0.1, 0.1)
                self.out.bias.data.uniform_(-0.1, 0.1)
        else:
            self.attn_hidden_lin = None
            self.out = nn.Linear(in_features=rnn_hidden_size, out_features=vocab_size)
            with torch.no_grad():
                self.out.weight.data.uniform_(-0.1, 0.1)
                self.out.bias.data.uniform_(-0.1, 0.1)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def has_attention(self):
        return self.attn is not None

    def last_attn_hidden_init(self, h_n):
        if isinstance(h_n, tuple):
            h_n = h_n[0]
        batch_size = h_n.size(1)
        if self.input_feed:
            return torch.zeros(batch_size, self.attn_hidden_projection_size, device=h_n.device)
        return None
    # output, _, kwargs = decoder(t, input_word, encoder_outputs, h_n, **kwargs)

    def _forward(self, t, input, encoder_outputs, last_state, last_attn_hidden=None, encoder_mask=None):
        print("decoder input shape",input.shape)
        total_tokens = input.view(-1)
        print(f"[DEBUG] Decoder input total tokens (raw): {total_tokens}")
    
        assert (self.input_feed and last_attn_hidden is not None) or (not self.input_feed and last_attn_hidden is None)
       
        # [Optimized] Flatten parameters
        self.rnn.flatten_parameters()
        
        embedded = self.embed(input)
        # print embedding shape
        #print('embedded shape:', embedded.shape)
        # prepare rnn input
        rnn_input = embedded
        if self.input_feed:
            rnn_input = torch.cat([rnn_input, last_attn_hidden], dim=1)
        rnn_input = rnn_input.unsqueeze(0)  # (1, batch, rnn_input_size)

        # [Optimized] Single Fused Call
        _, state = self.rnn(rnn_input, last_state)
        debug_tensor("LSTM last_state h-1", last_state[0], context="LuongDecoder._forward")

        # [DEBUG] State
        if isinstance(state, tuple):
             debug_tensor("LSTM State h", state[0], context="LuongDecoder._forward")
             debug_tensor("LSTM State c", state[1], context="LuongDecoder._forward")
        else:
             debug_tensor("GRU State", state, context="LuongDecoder._forward")
             
        hidden = state[0] if isinstance(state, tuple) else state
        
        # [DEBUG] Hidden Layer
        debug_tensor("Hidden Layer", hidden, context="LuongDecoder._forward")

        # attention context
        if self.attn is not None:
            attn_weights, context = self.attn(t, hidden[-1], encoder_outputs, mask=encoder_mask)
            attn_hidden = torch.tanh(self.attn_hidden_lin(torch.cat([context, hidden[-1]], dim=1)))
            # [DEBUG] Attention
            debug_tensor("Attn Weights", attn_weights, context="LuongDecoder._forward")
        else:
            attn_weights = None
            attn_hidden = hidden[-1]
            
        # [DEBUG] Attn Hidden
        debug_tensor("Attn Hidden", attn_hidden, context="LuongDecoder._forward")
        print("Attn Hidden shape", attn_hidden.shape)
        output = self.out(attn_hidden)
        
        # [Fix] Mask pad_idx in logits to ensure probability is 0 after softmax (in CrossEntropyLoss)
        #if self._pad_idx is not None:
        #    output[:, self._pad_idx] = float('-inf')
            
        # [DEBUG] Final Logits
        debug_tensor("Decoder Logits", output, context="LuongDecoder._forward")
        print("Decoder Logits shape", output.shape)
        return output, attn_weights, state, attn_hidden
