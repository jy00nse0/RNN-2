import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod

"""
[Revised] Attention Mechanism for RNN-based NMT
Based on Luong et al. (2015) "Effective Approaches to Attention-based Neural Machine Translation"

Modifications for Reproduction:
1. Added Location-based score function.
2. Fixed Local-p attention to use masking instead of zero-padding.
3. Enforced Uniform[-0.1, 0.1] initialization for all parameters.
"""

def attention_factory(args):
    """
    Factory method for attention module.
    """
    # Handle "none" attention type
    if args.attention_type == 'none':
        return None
    
    # Location score requires max_seq_len to define weight matrix size.
    # Defaulting to 50 (paper limit) if not specified in args.
    max_len = getattr(args, 'max_seq_len', 50) 
    
    score_map = {
        'dot': lambda: DotAttention(),
        'general': lambda: GeneralAttention(encoder_hidden_size=args.encoder_hidden_size * (2 if args.encoder_bidirectional else 1),
                                            decoder_hidden_size=args.decoder_hidden_size),
        'concat': lambda: ConcatAttention(hidden_size=args.concat_attention_hidden_size,
                                          encoder_hidden_size=args.encoder_hidden_size * (2 if args.encoder_bidirectional else 1),
                                          decoder_hidden_size=args.decoder_hidden_size),
        'location': lambda: LocationAttentionScore(decoder_hidden_size=args.decoder_hidden_size,
                                                   max_seq_len=max_len)
    }

    if args.attention_score not in score_map:
        raise ValueError(f"Unknown attention score: {args.attention_score}")

    score_module = score_map[args.attention_score]()

    attention_map = {
        'global': lambda: GlobalAttention(score_module),
        'local-m': lambda: LocalMonotonicAttention(score_module, args.half_window_size),
        'local-p': lambda: LocalPredictiveAttention(score_module, 
                                                    hidden_size=args.local_p_hidden_size, 
                                                    decoder_hidden_size=args.decoder_hidden_size,
                                                    D=args.half_window_size)
    }
    
    if args.attention_type not in attention_map:
        raise ValueError(f"Unknown attention type: {args.attention_type}")

    return attention_map[args.attention_type]()


class Attention(ABC, nn.Module):
    def __init__(self, attn_score):
        super(Attention, self).__init__()
        self.attn_score = attn_score

    @abstractmethod
    def forward(self, t, hidden, encoder_outputs, mask=None):
        raise NotImplementedError

    def attn_weights(self, hidden, encoder_outputs, mask=None):
        """
        Generates attention weights.
        :param mask: (batch, seq_len) Boolean mask where False indicates positions to be ignored (-inf).
        """
        scores = self.attn_score(hidden, encoder_outputs)
        
        if mask is not None:
            # Apply masking: set scores of invalid positions to -inf
            scores = scores.masked_fill(~mask, -1e9)
            
        return F.softmax(scores, dim=1)

    def attn_context(self, attn_weights, encoder_outputs):
        weights = attn_weights.unsqueeze(2)  # (batch, seq_len, 1)
        enc_out = encoder_outputs.permute(1, 2, 0)  # (batch, enc_h, seq_len)
        context = torch.bmm(enc_out, weights)  # (batch, enc_h, 1)
        return context.squeeze(2)


class GlobalAttention(Attention):
    def __init__(self, attn_score):
        super(GlobalAttention, self).__init__(attn_score)

    def forward(self, t, hidden, encoder_outputs, mask=None):
        # Global attention considers all encoder outputs
        attn_weights = self.attn_weights(hidden, encoder_outputs, mask=mask)
        return attn_weights, self.attn_context(attn_weights, encoder_outputs)


class LocalMonotonicAttention(Attention):
    """
    Local-m: Fixed window centered at t.
    """
    def __init__(self, attn_score, D):
        super(LocalMonotonicAttention, self).__init__(attn_score)
        self.D = D

    def forward(self, t, hidden, encoder_outputs, mask=None):
        seq_len = encoder_outputs.size(0)
        # Slicing handles boundary checks automatically (ignores out of bounds)
        start = max(0, t - self.D)
        end = min(seq_len, t + self.D + 1)
        
        enc_out = encoder_outputs[start:end]
        
        local_mask = mask[:, start:end] if mask is not None else None
        
        attn_weights = self.attn_weights(hidden, enc_out, mask=local_mask)
        return attn_weights, self.attn_context(attn_weights, enc_out)


class LocalPredictiveAttention(Attention):
    """
    Local-p: Predicted window center pt.
    [Revised] Uses masking instead of zero-padding for boundary handling.
    """
    def __init__(self, attn_score, hidden_size, decoder_hidden_size, D):
        super(LocalPredictiveAttention, self).__init__(attn_score)
        self.D = D
        self.Wp = nn.Linear(in_features=decoder_hidden_size, out_features=hidden_size)
        self.vp = nn.Linear(in_features=hidden_size, out_features=1)
        
        # Ground Truth Initialization
        nn.init.uniform_(self.Wp.weight, -0.1, 0.1)
        nn.init.uniform_(self.Wp.bias, -0.1, 0.1)
        nn.init.uniform_(self.vp.weight, -0.1, 0.1)
        nn.init.uniform_(self.vp.bias, -0.1, 0.1)

    def forward(self, t, hidden, encoder_outputs, mask=None):
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        # 1. Predict aligned position pt
        p = self.calculate_p(hidden, seq_len)  # (batch)

        # 2. Extract window and Create Mask
        # Instead of padding encoder_outputs, we extract indices and create a validity mask
        window_indices, enc_out_local, mask = self.get_window_context(encoder_outputs, p, seq_len)

        # 3. Calculate Attention Weights (with Masking)
        attn_weights = self.attn_weights(hidden, enc_out_local, mask=mask)

        # 4. Apply Gaussian Scaling
        attn_weights_scaled = self.scale_weights(window_indices, p, attn_weights)
        
        # Re-normalize after scaling? Paper doesn't explicitly say, but usually yes. 
        # However, Eq(10) is just multiplication. We return context.
        
        return attn_weights_scaled, self.attn_context(attn_weights_scaled, enc_out_local)

    def calculate_p(self, hidden, seq_len):
        Wph = torch.tanh(self.Wp(hidden))
        p = seq_len * torch.sigmoid(self.vp(Wph))
        return p.squeeze(1)

    def get_window_context(self, encoder_outputs, p, seq_len):
        """
        Extracts windows and creates a mask for valid positions.
        """
        batch_size = encoder_outputs.size(1)
        enc_hidden_size = encoder_outputs.size(2)
        window_width = 2 * self.D + 1
        
        # Create base indices for window: [0, 1, ..., 2D]
        window_offsets = torch.arange(-self.D, self.D + 1, device=encoder_outputs.device).unsqueeze(0) # (1, window_width)
        
        # Center positions (integer)
        centers = p.round().long().unsqueeze(1) # (batch, 1)
        
        # Absolute indices: (batch, window_width)
        abs_indices = centers + window_offsets 
        
        # Create Validity Mask (True if index is within [0, seq_len-1])
        mask = (abs_indices >= 0) & (abs_indices < seq_len) # (batch, window_width)
        
        # Clamp indices to valid range for gathering (to avoid index out of bounds error)
        # The masked positions will be ignored later, so value doesn't matter.
        clamped_indices = abs_indices.clamp(min=0, max=seq_len-1)
        
        # Gather encoder outputs
        # encoder_outputs: (seq_len, batch, dim) -> permute to (batch, seq_len, dim) for gathering
        enc_out_perm = encoder_outputs.permute(1, 0, 2)
        
        # Expand indices for gather: (batch, window_width, dim)
        gather_indices = clamped_indices.unsqueeze(2).expand(-1, -1, enc_hidden_size)
        
        enc_out_local = torch.gather(enc_out_perm, 1, gather_indices) # (batch, window_width, dim)
        
        # Return format expected by attn_context: (window_width, batch, dim)
        return abs_indices.permute(1, 0).float(), enc_out_local.permute(1, 0, 2), mask

    def scale_weights(self, window_indices, p, attn_weights):
        # window_indices: (window_width, batch) - absolute positions
        # p: (batch)
        stddev = self.D / 2.0
        numerator = (window_indices - p.unsqueeze(0)) ** 2
        gauss = torch.exp(-(numerator) / (2 * stddev ** 2)) # (window_width, batch)
        
        return attn_weights * gauss.t()


class AttentionScore(ABC, nn.Module):
    def __init__(self):
        super(AttentionScore, self).__init__()
        
    @abstractmethod
    def forward(self, hidden, encoder_outputs):
        raise NotImplementedError


class DotAttention(AttentionScore):
    def forward(self, hidden, encoder_outputs):
        # e = ht^T * hs
        hidden = hidden.unsqueeze(1)  # (batch, 1, h)
        enc_out = encoder_outputs.permute(1, 2, 0)  # (batch, h, seq_len)
        scores = torch.bmm(hidden, enc_out)
        return scores.squeeze(1)


class GeneralAttention(AttentionScore):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(GeneralAttention, self).__init__()
        self.W = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=False)
        # Ground Truth Init
        nn.init.uniform_(self.W.weight, -0.1, 0.1)

    def forward(self, hidden, encoder_outputs):
        # e = ht^T * Wa * hs
        # Calculate Wa * hs
        # encoder_outputs: (seq_len, batch, enc_h)
        enc_len, batch, _ = encoder_outputs.size()
        enc_flat = encoder_outputs.view(-1, encoder_outputs.size(2))
        
        # (seq_len*batch, dec_h)
        Wa_hs = self.W(enc_flat).view(enc_len, batch, -1) 
        
        hidden = hidden.unsqueeze(1) # (batch, 1, dec_h)
        Wa_hs = Wa_hs.permute(1, 2, 0) # (batch, dec_h, seq_len)
        
        scores = torch.bmm(hidden, Wa_hs)
        return scores.squeeze(1)


class ConcatAttention(AttentionScore):
    def __init__(self, hidden_size, encoder_hidden_size, decoder_hidden_size):
        super(ConcatAttention, self).__init__()
        self.W = nn.Linear(encoder_hidden_size + decoder_hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
        # Ground Truth Init
        nn.init.uniform_(self.W.weight, -0.1, 0.1)
        nn.init.uniform_(self.v.weight, -0.1, 0.1)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        h = hidden.unsqueeze(0).expand(seq_len, -1, -1)
        
        energy = torch.tanh(self.W(torch.cat([encoder_outputs, h], dim=2)))
        scores = self.v(energy)
        return scores.squeeze(2).t()


class LocationAttentionScore(AttentionScore):
    """
    [Added] Location-based score function.
    a_t = softmax(W_a h_t)
    Note: Requires max_seq_len to define the size of Wa.
    """
    def __init__(self, decoder_hidden_size, max_seq_len=50):
        super(LocationAttentionScore, self).__init__()
        self.W = nn.Linear(decoder_hidden_size, max_seq_len, bias=False)
        
        # Ground Truth Init
        nn.init.uniform_(self.W.weight, -0.1, 0.1)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, dec_h)
        # encoder_outputs: (seq_len, batch, enc_h) -> used only for sizing
        
        # 1. Calculate scores for all possible positions up to max_len
        scores_all = self.W(hidden) # (batch, max_len)
        
        # 2. Slice to current actual sequence length
        curr_seq_len = encoder_outputs.size(0)
        
        # Handle case where current seq len > max_len (though unlikely with proper preprocessing)
        if curr_seq_len > scores_all.size(1):
             # Pad scores with -inf if necessary, or just rely on constructor max_len being sufficient.
             # For reproduction, we assume max_len=50 is respected.
             pass
             
        scores = scores_all[:, :curr_seq_len]
        return scores
