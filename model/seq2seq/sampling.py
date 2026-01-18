import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class SequenceSampler(ABC):
    """
    Samples output sequence from decoder given input sequence (encoded). Sequence will be sampled until EOS token is
    sampled or sequence reaches ``max_length``.

    Inputs: encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length
        - **encoder_outputs** (seq_len, batch, encoder_hidden_size): Last encoder layer outputs for every timestamp.
        - **h_n** (num_layers * num_directions, batch, hidden_size): RNN outputs for all layers for t=seq_len (last
                    timestamp)
        - **decoder**: Seq2seq decoder.
        - **sos_idx** (scalar): Index of start of sentence token in vocabulary.
        - **eos_idx** (scalar): Index of end of sentence token in vocabulary.
        - **max_length** (scalar): Maximum length of sampled sequence.

    Outputs: sequences, lengths
        - **sequences** (batch, max_seq_len): Sampled sequences.
        - **lengths** (batch): Length of sequence for every batch.
    """

    @abstractmethod
    def sample(self, encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length):
        raise NotImplementedError


class GreedySampler(SequenceSampler):
    """
    Greedy sampler: 가장 높은 확률의 토큰을 선택.
    EOS 생성 시 해당 샘플은 조기 종료하며, 모든 샘플이 종료되면 즉시 중단.
    """
    def sample(self, encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length):
        batch_size = encoder_outputs.size(1)
        device = encoder_outputs.device

        # 출력 버퍼 미리 할당
        sequences = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        lengths = torch.full((batch_size,), max_length, dtype=torch.long, device=device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        eos_scalar = torch.tensor(eos_idx, dtype=torch.long, device=device)

        input_word = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
        kwargs = {}

        for t in range(max_length):
            output, _, kwargs = decoder(t, input_word, encoder_outputs, h_n, **kwargs)
            argmax = output.argmax(dim=1)  # (batch,)
            # argmax = output.argmax(dim=1)  # (batch,)
            # [Modified] Select 3rd highest probability token as per user request
            #_, top_indices = output.topk(4, dim=1)
            #argmax = top_indices[:, 1]  # Pick the 3rd (index 2) element

            '''
            # [DEBUG] Check if predicted token is <unk> (index 3 based on dataset.py)
            # User request: print(next_token, tgt_vocab.stoi['<unk>']) verification
            # Since we don't have vocab object here, we print the raw index.
            # <unk> is typically index 3 in this codebase (pad=0, sos=1, eos=2, unk=3)
            if t < 50 and batch_size == 128: # Only print for first few tokens of single batch to avoid spam
                # Handle batch: just look at the first sample in the batch
                first_pred = argmax[0].item()
                print(f"[DEBUG] t={t}: Batch[0] Index={first_pred} (Is UNK(3)? {first_pred == 3})")
            ''' 
            # 아직 끝나지 않은 샘플만 업데이트
            if (~finished).any():
                sequences[~finished, t] = argmax[~finished]

            # 새로 종료된 샘플 갱신(EOS 포함 길이)
            newly_finished = (argmax == eos_idx) & (~finished)
            if newly_finished.any():
                lengths[newly_finished] = t + 1
                finished = finished | newly_finished

            # 모두 종료되면 중단
            if finished.all():
                break

            # 종료된 샘플은 다음 입력으로 EOS 고정
            input_word = torch.where(finished, eos_scalar, argmax)

        return sequences, lengths


class RandomSampler(SequenceSampler):
    """
    Random sampler: 소프트맥스 확률에 따른 룰렛 휠 샘플링.
    EOS 생성 시 해당 샘플은 조기 종료하며, 모든 샘플이 종료되면 즉시 중단.
    """
    def sample(self, encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length):
        print("using random...")
        batch_size = encoder_outputs.size(1)
        device = encoder_outputs.device

        sequences = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        lengths = torch.full((batch_size,), max_length, dtype=torch.long, device=device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        eos_scalar = torch.tensor(eos_idx, dtype=torch.long, device=device)

        input_word = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
        kwargs = {}

        for t in range(max_length):
            output, _, kwargs = decoder(t, input_word, encoder_outputs, h_n, **kwargs)
            probs = F.softmax(output, dim=1)
            sampled = torch.multinomial(probs, 1).squeeze(1)  # (batch,)

            if (~finished).any():
                sequences[~finished, t] = sampled[~finished]

            newly_finished = (sampled == eos_idx) & (~finished)
            if newly_finished.any():
                lengths[newly_finished] = t + 1
                finished = finished | newly_finished

            if finished.all():
                break

            input_word = torch.where(finished, eos_scalar, sampled)

        return sequences, lengths


class Sequence:
    def __init__(self, log_prob, tokens, kwargs):
        self.log_prob = log_prob
        self.tokens = tokens
        self.kwargs = kwargs

    def new_seq(self, tok, log_prob, eos_idx):
        # EOS 이후에는 확률 누적 중단
        log_prob = log_prob if self.tokens[-1] != eos_idx else 0.0
        return Sequence(self.log_prob + log_prob, self.tokens + [tok], self.kwargs)

    def normalized_score(self, alpha=1.0):
        # Google NMT 길이 정규화: log_prob / ((5 + len)/6)^alpha
        return self.log_prob / (((5 + len(self.tokens)) / 6.0) ** alpha)


class BeamSearch(SequenceSampler):
    """
    빔 서치 디코더(조기 종료 지원):
    - EOS가 나오면 해당 빔 확장 중단
    - 모든 활성 빔이 EOS를 내면 즉시 종료
    - 변수 섀도잉 제거, kwargs 안정적 복사
    """
    def __init__(self, beam_width=12, alpha=1.0):
        self.beam_width = beam_width
        self.alpha = alpha

    def sample(self, encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length):
        batch_size = encoder_outputs.size(1)
        device = encoder_outputs.device

        all_sequences = []
        all_lengths = []

        for b in range(batch_size):
            # [Fix] Handle LSTM tuple state (hidden, cell)
            if isinstance(h_n, tuple):
                h_n_sample = (h_n[0][:, b, :].unsqueeze(1).contiguous(), h_n[1][:, b, :].unsqueeze(1).contiguous())
            else:
                h_n_sample = h_n[:, b, :].unsqueeze(1).contiguous()

            seq, length = self._sample(
                encoder_outputs[:, b, :].unsqueeze(1),
                h_n_sample,
                decoder, sos_idx, eos_idx, max_length, device
            )
            all_sequences.append(seq)
            all_lengths.append(length)

        # 동일 길이로 패딩
        max_len = max(seq.size(0) for seq in all_sequences)
        sequences = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(all_sequences):
            sequences[i, :seq.size(0)] = seq

        lengths = torch.tensor(all_lengths, dtype=torch.long, device=device)
        return sequences, lengths

    def _sample(self, encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length, device):
        active_beams = [Sequence(0.0, [sos_idx], {})]
        finished_beams = []

        for t in range(max_length):
            candidates = []

            for beam in active_beams:
                if beam.tokens[-1] == eos_idx:
                    finished_beams.append(beam)
                    continue

                input_word = torch.tensor(beam.tokens[-1], device=device, dtype=torch.long).view(1)
                output, _, kwargs = decoder(t, input_word, encoder_outputs, h_n, **beam.kwargs)
                log_probs = F.log_softmax(output.squeeze(0), dim=0)

                # [Optimized] Use topk instead of iterating over entire vocab
                # Only consider top k candidates for extension (where k = beam_width * 2 for safety)
                top_k = min(self.beam_width * 2, log_probs.size(0))
                top_log_probs, top_indices = log_probs.topk(top_k)

                for i in range(top_k):
                    tok = top_indices[i].item()
                    log_prob = top_log_probs[i].item()
                    
                    new_beam = beam.new_seq(tok, log_prob, eos_idx)
                    new_beam.kwargs = dict(kwargs) if isinstance(kwargs, dict) else kwargs
                    candidates.append(new_beam)

            if not candidates:
                break

            # 길이 정규화 점수(alpha 반영)로 상위 beam_width 선택
            candidates.sort(key=lambda b: b.normalized_score(self.alpha), reverse=True)
            active_beams = candidates[:self.beam_width]

            # 모든 빔이 EOS면 중단
            if all(b.tokens[-1] == eos_idx for b in active_beams):
                finished_beams.extend(active_beams)
                break

        all_beams = finished_beams + active_beams
        best_beam = max(all_beams, key=lambda b: b.normalized_score(self.alpha))

        # 길이: EOS 포함 첫 위치(+1), 없으면 전체 길이
        try:
            length = best_beam.tokens.index(eos_idx) + 1
        except ValueError:
            length = len(best_beam.tokens)

        return torch.tensor(best_beam.tokens, device=device, dtype=torch.long), length
