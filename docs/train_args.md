# Args (CLI options) Quick Guide

## Paper hyperparameters

- `--max-epochs` *(int, default=10)*  
  학습 에폭 수. *(논문: Base 10, Dropout 12)*

- `--gradient-clip` *(float, default=5.0)*  
  그래디언트 클리핑 노름 기준값. `clip_grad_norm_`에 사용.

- `--batch-size` *(int, default=128)*  
  미니배치 크기.

- `--learning-rate` *(float, default=1.0)*  
  SGD 초기 학습률.

- `--lr-decay-start` *(int, default=5)*  
  해당 epoch 이후부터 매 epoch마다 lr을 0.5로 halving.  
  코드 기준(epoch 0-index): `epoch > lr-decay-start`일 때 감소 적용.

---

## Training configuration

- `--train-embeddings` *(flag, default=True)*  
  임베딩 파라미터를 학습할지 여부(gradient 전파).

- `--embedding-type` *(str, default=None)*  
  사전학습 임베딩 타입/이름 지정(프로젝트 구현에 따라 사용될 수 있음).

- `--save-path` *(str, default=".save")*  
  체크포인트/설정/로그 저장 기본 폴더. 실행 시점 타임스탬프 하위 폴더가 생성됨.

- `--save-every-epoch` *(flag)*  
  검증 손실 개선 여부와 무관하게 매 epoch 모델 저장.

- `--dataset` *(choice)*  
  사용할 데이터셋 선택.

- `--teacher-forcing-ratio` *(float, default=1.0)*  
  Teacher forcing 비율(0~1). *(모델 구현에서 사용)*

- `--reverse` *(flag)*  
  **[실험]** 소스 시퀀스를 뒤집어 encoder 입력으로 사용.  
  *(현재 train/evaluate에서 적용 코드 주석 처리됨)*

---

## GPU

- `--cuda` *(flag, default=True)*  
  CUDA 사용 여부(가능할 때). `torch.cuda.is_available()`와 함께 사용됨.

- `--multi-gpu` *(flag, default=False)*  
  `nn.DataParallel`로 멀티 GPU 사용.

---

## Performance

- `--num-workers` *(int, default=12)*  
  `DataLoader` worker 수(데이터 로딩 병렬화).

- `--amp` *(flag, default=False)*  
  Automatic Mixed Precision 사용(속도↑, 재현성에 영향 가능).

---

## Debug / visualization

- `--debug` *(flag, default=False)*  
  배치 단위 로그/평가 verbose 출력.

- `--log-interval` *(int, default=100)*  
  `--debug` 시 배치 로그 출력 주기.

- `--sample-translations` *(flag, default=False)*  
  epoch마다 샘플 번역(예측) 출력.

- `--print-model-summary` *(flag, default=False)*  
  모델 구조/파라미터 수 출력.

- `--plot-loss-graph` *(flag, default=False)*  
  epoch별 train/val loss를 jsonl로 저장하고 종료 시 그래프 생성.

---

## Embedding hyperparameters

- `--embedding-size` *(int, default=1000)*  
  임베딩 차원 *(논문 기본값 1000)*

---

## Encoder hyperparameters

- `--encoder-rnn-cell` *(LSTM|GRU, default=LSTM)*
- `--encoder-hidden-size` *(int, default=1000)*
- `--encoder-num-layers` *(int, default=4)*
- `--encoder-rnn-dropout` *(float, default=0.0)*
- `--encoder-bidirectional` *(flag)*

---

## Decoder hyperparameters

- `--decoder-type` *(bahdanau|luong, default=luong)*
- `--decoder-rnn-cell` *(LSTM|GRU, default=LSTM)*
- `--decoder-hidden-size` *(int, default=1000)*
- `--decoder-num-layers` *(int, default=4)*
- `--decoder-rnn-dropout` *(float, default=0.0)*
- `--luong-attn-hidden-size` *(int, default=1000)*
- `--luong-input-feed` *(flag)*
- `--decoder-init-type` *(zeros|bahdanau|adjust_pad|adjust_all, default=adjust_pad)*

---

## Attention hyperparameters

- `--attention-type` *(none|global|local-m|local-p, default=global)*
- `--attention-score` *(dot|general|concat|location, default=dot)*
- `--half-window-size` *(int, default=10)*  
  local attention에서 window `D` (Luong et al.)

- `--local-p-hidden-size` *(int, default=1000)*
- `--concat-attention-hidden-size` *(int, default=1000)*
