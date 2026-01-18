import torch
import torch.nn as nn
import random
import string
from constants import SOS_TOKEN, EOS_TOKEN
from .sampling import GreedySampler, RandomSampler, BeamSearch
# [New] util에서 init_weights 임포트
from util import init_weights 
from debug_utils import debug_tensor 

class Seq2SeqTrain(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, teacher_forcing_ratio=0.5, tgt_pad_idx=0):
        """
        Args:
            encoder: Encoder module
            decoder: Decoder module
            vocab_size: Target vocabulary size
            teacher_forcing_ratio: Ratio for teacher forcing (0-1)
            tgt_pad_idx: Target padding index (should be from metadata.padding_idx)
        """
        super(Seq2SeqTrain, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.tgt_pad_idx = tgt_pad_idx  # Store padding index from metadata
        
        # [New] 논문 재현을 위한 파라미터 초기화 (Uniform -0.1 ~ 0.1)
        # Encoder, Decoder를 포함한 전체 서브모듈에 적용
        self.apply(init_weights)

    def forward(self, question, answer, src_lengths=None):
        # [기존 코드 유지]
        answer_seq_len = answer.size(0)
        batch_size = answer.size(1)
        
        encoder_outputs, h_n = self.encoder(question, src_lengths)
        
        # Explicit slicing for decoder input and labels
        # answer: (tgt_len, batch)
        decoder_input = answer[:-1]   # (tgt_len-1, batch)  <sos> ... 마지막 단어
        # target label for teacher forcing
        target_label = answer[1:]     # (tgt_len-1, batch)  첫 단어 ... <eos>
        
        # Handle edge case: if decoder_input is empty (answer only contains <sos>)
        if decoder_input.size(0) == 0:
            # Return an empty logits tensor with correct shape/device/dtype
            batch_size = answer.size(1)
            return torch.empty(
                0, batch_size, self.vocab_size,
                dtype=encoder_outputs.dtype, device=answer.device
            )
        
        # ===== [NEW] Calculate valid target lengths and timestep mask (exclude padding) =====
        # Count non-pad tokens in decoder_input for each sample in batch
        # decoder_input: (tgt_len-1, batch)
        # Use stored tgt_pad_idx from metadata (set in __init__)
        tgt_lengths = (decoder_input != self.tgt_pad_idx).sum(dim=0)  # (batch,)
        max_valid_len = int(tgt_lengths.max().item())
        
        # Safety: ensure we don't exceed actual tensor size
        max_valid_len = min(max_valid_len, decoder_input.size(0))
        
        # [Solution 2] Create timestep mask: (max_valid_len, batch)
        # True if sample is active at timestep t
        timestep_mask = torch.arange(max_valid_len, device=decoder_input.device).unsqueeze(1) < tgt_lengths.unsqueeze(0)
        
        kwargs = {}
        input_word = decoder_input[0]
        
        # Create encoder mask (batch, seq_len)
        # src_lengths: (batch)
        max_src_len = question.size(0)
        encoder_mask = torch.arange(max_src_len, device=question.device).expand(batch_size, max_src_len) < src_lengths.unsqueeze(1)
        
        # [DEBUG] Mask
        debug_tensor("Encoder Mask", encoder_mask, context="Seq2SeqTrain.forward", values=True)
        logits_steps = []
        
        # ===== [MODIFIED] Loop with Solution 2: Selective Masking =====
        for t in range(max_valid_len):
            active_mask = timestep_mask[t]  # (batch,)
            
            # Forward pass: process all samples
            output, attn_weights, next_kwargs = self.decoder(t, input_word, encoder_outputs, h_n, encoder_mask=encoder_mask, **kwargs)
            
            # If some samples are inactive, mask outputs and restore hidden states
            if not active_mask.all():
                # 1. Mask inactive outputs with -1e9 (cross-entropy will ignore them)
                output = output.masked_fill(~active_mask.unsqueeze(1), -1e2)
                
                # 2. Restore previous states for inactive samples (avoid padding pollution)
                # t=0 case: kwargs is empty, but decoder already initialized states in kwargs_init
                # However, for t > 0, we can restore from previous iteration's kwargs
                if kwargs:  # For t > 0
                    for key in next_kwargs:
                        val = next_kwargs[key]
                        prev_val = kwargs[key]
                        
                        if isinstance(val, torch.Tensor):
                            if val.dim() == 2:  # e.g., Luong attn hidden (batch, dim)
                                next_kwargs[key] = torch.where(active_mask.unsqueeze(-1), val, prev_val)
                            elif val.dim() == 3:  # e.g., GRU state (layers, batch, dim)
                                next_kwargs[key] = torch.where(active_mask.view(1, -1, 1), val, prev_val)
                        elif isinstance(val, tuple):  # e.g., LSTM state (h, c)
                            h, c = val
                            ph, pc = prev_val
                            # h, c: (layers, batch, dim)
                            new_h = torch.where(active_mask.view(1, -1, 1), h, ph)
                            new_c = torch.where(active_mask.view(1, -1, 1), c, pc)
                            next_kwargs[key] = (new_h, new_c)
            
            kwargs = next_kwargs
            logits_steps.append(output)

            teacher_forcing = random.random() < self.teacher_forcing_ratio
            if teacher_forcing:
                # Ensure we don't go out of bounds
                if t < target_label.size(0):
                    input_word = target_label[t]
                # else: keep previous input_word (fallback for safety)
            else:
                input_word = output.argmax(dim=1)
        
        outputs = torch.stack(logits_steps, dim=0)  # (max_valid_len, batch, vocab_size)

                
        # [DEBUG] Final Output
        debug_tensor("Final Alloc Output", outputs, context="Seq2SeqTrain.forward", values=False)
        print("Final Output shape", outputs.shape)
        return outputs

class Seq2SeqPredict(nn.Module):
    """
    This class is wrapper around pre-trained model which can be used for testing model.
    This model takes (numericalized) input, delegates it to appropriate sequence sampler and returns

    :param encoder: Pre-trained encoder.
    :param decoder: Pre-trained decoder.
    :param src_field: Field object for source language (encoder input processing).
    :param tgt_field: Field object for target language (decoder output decoding).

    Inputs: questions, sampling_strategy, max_seq_len
        - **questions** list(str): List of raw question strings.
        - **sampling_strategy** (str): Strategy for sampling output sequences. ['greedy', 'random', 'beam_search']
        - **max_seq_len** (scalar): Maximum length of output sequence.

    Outputs: sequences
        - **sequences** list(str): List of answers sequences generated by model.
    """
    def __init__(self, encoder, decoder, src_field, tgt_field):
        super(Seq2SeqPredict, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = tgt_field.vocab.stoi[SOS_TOKEN]
        self.eos_idx = tgt_field.vocab.stoi[EOS_TOKEN]
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.samplers = {
            'greedy': GreedySampler(),
            'random': RandomSampler(),
            'beam_search': BeamSearch()
        }

    def decode_sequence(self, tokens_idx):
        """
        Decodes token indices to string using target vocabulary.

        :param tokens_idx: List of token indices.
        :return: String representing decoded sequence.
        """
        seq = ''
        for idx in tokens_idx:
            tok = self.tgt_field.vocab.itos[idx]
            if tok not in string.punctuation and tok[0] != '\'':
                seq += ' '
            seq += tok
        return seq.strip()

    def forward(self, questions, sampling_strategy, max_seq_len):
        if isinstance(questions, torch.Tensor):
            q = questions
            # Recalculate lengths if not provided?
            # Ideally lengths should be passed if we pass tensor.
            # But the signature only takes 'questions'.
            # We can deduce lengths from padding.
            pad_idx = self.src_field.vocab.stoi.get('<eos>', 0)
            lengths = (q != pad_idx).sum(dim=0)
            lengths = lengths.to('cpu')
        else:
            # raw strings to tensor using source field
            q = self.src_field.process([self.src_field.preprocess(question) for question in questions])
            lengths = (q != self.src_field.vocab.stoi.get('<eos>', 0)).sum(dim=0).to('cpu')
        
        # Move tensor to the same device as the model
        device = next(self.encoder.parameters()).device
        q = q.to(device)

        # encode question sequence
        encoder_outputs, h_n = self.encoder(q, lengths)
        #print(self.encoder.embed)
                # [DEBUG] Print 10th token embedding
        '''
        if hasattr(self.encoder.embed, 'weight'):
            print(f"Embedding for token 10: {self.encoder.embed.weight[10]}")
        else:
             # If embed is not a standard Embedding layer (e.g. customized), try forward pass
             with torch.no_grad():
                 dummy_idx = torch.LongTensor([10])
                 if next(self.encoder.embed.parameters()).is_cuda:
                     dummy_idx = dummy_idx.cuda()
                 print(f"Embedding for token 10: {self.encoder.embed(dummy_idx)}")
        '''
        # sample output sequence
        sequences, lengths = self.samplers[sampling_strategy].sample(encoder_outputs, h_n, self.decoder, self.sos_idx,
                                                                     self.eos_idx, max_seq_len)

        # torch tensors -> python lists
        batch_size = sequences.size(0)
        sequences, lengths = sequences.tolist(), lengths.tolist()

        # decode output (token idx -> token string)
        seqs = []
        for batch in range(batch_size):
            seq = sequences[batch][:lengths[batch]]
            seqs.append(self.decode_sequence(seq))

        return seqs
