## Seq2Seq (model/seq2seq)

### `embeddings.py`
- **`embeddings_factory(args, metadata)`**
  - 토큰 임베딩 레이어(`nn.Embedding`) 생성.
  - `metadata.vectors`(pretrained vectors)가 있으면 그 가중치로 초기화.
  - 없으면 논문 재현 설정으로 `Uniform(-0.1, 0.1)` 초기화.
  - `args.train_embeddings`에 따라 임베딩 학습 여부(`requires_grad`) 설정.

---

### `encoder.py`
- **`encoder_factory(args, metadata, embed=None)`**
  - 설정(args)에 맞춰 Encoder 생성(필요 시 embedding도 factory로 생성).
- **`Encoder` (ABC)**
  - 인코더 인터페이스(추상 클래스): `forward`, `hidden_size`, `bidirectional`, `num_layers` 정의.
- **`SimpleEncoder(Encoder)`**
  - RNN 기반 인코더(`LSTM/GRU` 등, `args.encoder_rnn_cell` 사용).
  - 입력 토큰 → 임베딩 → RNN 인코딩 → `(encoder_outputs, h_n)` 반환.
  - `encoder_outputs`: (seq_len, batch, hidden * dir)
  - `h_n`: (layers * dir, batch, hidden) (LSTM이면 tuple)

---

### `attention.py`
- **`attention_factory(args)`**
  - attention 사용 여부 및 타입을 설정으로 선택:
    - score: `dot`, `general`, `concat`, `location`
    - type: `global`, `local-m`, `local-p`
    - `args.attention_type == 'none'`이면 `None` 반환.
- **`Attention` (ABC)**
  - attention 공통 베이스(가중치 계산 `attn_weights`, 컨텍스트 `attn_context` 제공).
- **`GlobalAttention`**
  - 전체 encoder timestep을 대상으로 attention 수행.
- **`LocalMonotonicAttention` (local-m)**
  - 현재 시점 `t` 중심의 고정 윈도우 범위만 attention.
- **`LocalPredictiveAttention` (local-p)**
  - 디코더 hidden으로 정렬 위치 `p_t`를 예측하고, 해당 윈도우에 attention(마스킹 기반).
  - Gaussian scaling을 적용하여 윈도우 중심에 가중치 집중.
- **Score modules (`AttentionScore` 계열)**
  - `DotAttention`, `GeneralAttention`, `ConcatAttention`, `LocationAttentionScore`
  - 논문 재현을 위한 파라미터 `Uniform(-0.1, 0.1)` 초기화 포함.

---

### `decoder_init.py`
- **`decoder_init_factory(args)`**
  - 디코더 초기 hidden state 생성 전략 선택.
- **`DecoderInit` (ABC)**
  - `forward(h_n)`로 encoder의 마지막 hidden state를 받아 decoder 초기화 상태 생성.
- **`ZerosInit`**
  - decoder hidden을 0으로 초기화.
- **`BahdanauInit`**
  - (양방향 encoder 전제) backward hidden을 linear + tanh로 변환해 decoder 초기 hidden으로 사용.
  - encoder/decoder layer 수 불일치 시 pad/slice로 보정.
- **`EncoderLastStateInit`**
  - encoder의 마지막 hidden(state)을 decoder 초기 state로 직접 사용(필요 시 pad/slice).

---

### `decoder.py`
- **`decoder_factory(args, metadata, embed=None)`**
  - embedding + attention + init 모듈을 조립하여 decoder 생성.
- **`Decoder` (ABC)**
  - 공통 디코더 래퍼: `forward(t, input, encoder_outputs, h_n, **kwargs)`
  - timestep 0에서 필요한 state를 초기화하고, 이후 timestep에서는 `kwargs`로 상태를 전달/갱신.
- **`BahdanauDecoder(Decoder)`**
  - Bahdanau attention 기반 디코더.
  - 입력: (임베딩 + context) concat → RNN → (maxout k=2) → vocab projection.
  - `rnn_hidden_size`는 maxout 때문에 **짝수여야 함**.
- **`LuongDecoder(Decoder)`**
  - Luong 방식 디코더(+ optional input feeding).
  - RNN 출력과 context를 결합해 출력 로짓 생성.

---

### `sampling.py`
- **`SequenceSampler` (ABC)**
  - 추론 시 decoder로부터 output sequence를 생성하는 인터페이스.
- **`GreedySampler`**
  - 매 timestep에서 argmax 토큰 선택(조기 종료: EOS 시).
- **`RandomSampler`**
  - softmax 확률 기반 multinomial 샘플링(조기 종료: EOS 시).
- **`BeamSearch`**
  - 빔 서치 기반 디코딩(길이 정규화 `alpha` 지원, EOS 조기 종료 지원).
  - 내부 보조 클래스 `Sequence`: 누적 log-prob 및 토큰 시퀀스 관리.

---

### `model.py`
- **`Seq2SeqTrain(nn.Module)`**
  - 학습용 seq2seq wrapper.
  - encoder로 입력 인코딩 후, decoder를 teacher forcing(비율 `teacher_forcing_ratio`)로 학습.
  - 출력 텐서를 미리 할당하여 반복 concat을 피하는 방식으로 구현.
  - `util.init_weights`로 전체 서브모듈 파라미터를 `Uniform(-0.1, 0.1)` 초기화(논문 재현 목적).
- **`Seq2SeqPredict(nn.Module)`**
  - 추론/테스트용 wrapper.
  - `src_field`로 입력 전처리→텐서화, encoder 인코딩 후 sampler(`greedy/random/beam_search`)로 문장 생성.
  - `decode_sequence()`로 토큰 인덱스를 문자열로 복원.
