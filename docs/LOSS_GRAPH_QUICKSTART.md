# Loss Graph Generation - Quick Start Guide

## 개요 (Overview)
학습 중 모델의 loss를 자동으로 기록하고, 학습 종료 후 시각화된 그래프를 생성하는 기능입니다.

This feature automatically records model loss during training and generates visualization graphs after training completion.

---

## 🚀 Quick Start

### 1. 필수 라이브러리 설치 (Install Required Libraries)
```bash
pip install matplotlib
```

### 2. 자동 그래프 생성 (Automatic Graph Generation)

#### test_new.py를 사용한 학습 시
학습이 완료되면 자동으로 loss 그래프가 생성됩니다:

```bash
python test_new.py -e T1_Base_Reverse
```

**출력 파일**:
- `/workspace/checkpoints/T1_Base_Reverse/training_metrics.jsonl` - 메트릭 데이터
- `/workspace/checkpoints/T1_Base_Reverse/loss_graph.png` - Loss 그래프 (자동 생성)

### 3. 수동 그래프 생성 (Manual Graph Generation)

기존 학습 데이터로부터 그래프를 생성하려면:

```bash
python generate_loss_graph.py \
    --metrics-file /path/to/training_metrics.jsonl \
    --output my_loss_graph.png
```

**예시**:
```bash
# 기본 경로 사용
python generate_loss_graph.py

# 커스텀 경로 지정
python generate_loss_graph.py \
    --metrics-file .save/2024-12-28-12:00/training_metrics.jsonl \
    --output results/loss_graph.png
```

---

## 📊 생성되는 그래프 예시 (Graph Example)

![Loss Graph Example](https://github.com/user-attachments/assets/4082a30c-4542-4a07-89c3-916f2201a975)

**그래프 특징**:
- 파란색: 학습 Loss
- 빨간색: 검증 Loss  
- X축: Epoch 번호
- Y축: Loss 값
- 고해상도 (300 DPI)

---

## 🔧 프로그래밍 방식 사용 (Programmatic Usage)

Python 코드에서 직접 사용:

```python
from util import load_training_metrics, plot_loss_graph

# 메트릭 파일에서 데이터 로드
train_losses, val_losses = load_training_metrics('training_metrics.jsonl')

# 그래프 생성
plot_loss_graph(train_losses, val_losses, 'my_graph.png')
```

또는 직접 데이터를 전달:

```python
from util import plot_loss_graph

# 학습 중 수집한 데이터
train_losses = [5.2, 4.8, 4.3, 3.9, 3.5, 3.2, 3.0, 2.8, 2.6, 2.5]
val_losses = [5.4, 5.0, 4.5, 4.1, 3.7, 3.4, 3.1, 2.9, 2.7, 2.6]

# 그래프 생성
plot_loss_graph(train_losses, val_losses, 'custom_graph.png')
```

---

## 📁 파일 구조 (File Structure)

```
RNN/
├── util.py                      # plot_loss_graph(), load_training_metrics()
├── test_new.py                  # 자동 그래프 생성 통합됨
├── generate_loss_graph.py       # 독립 실행형 스크립트
├── requirements.txt             # matplotlib 추가됨
└── .save/
    └── 2024-12-28-12:00/
        ├── training_metrics.jsonl  # 자동 생성
        └── loss_graph.png          # 자동 생성 (test_new.py 사용 시)
```

---

## 🎯 주요 기능 (Key Features)

✅ **자동 생성**: test_new.py 실행 시 자동으로 그래프 생성  
✅ **고품질**: 300 DPI PNG 형식으로 저장  
✅ **유연성**: 수동 생성도 가능  
✅ **에러 처리**: 그래프 생성 실패 시에도 학습은 계속됨  
✅ **호환성**: 기존 train.py 코드 수정 불필요  

---

## 💡 활용 방안 (Use Cases)

1. **학습 모니터링**: 학습이 잘 진행되는지 시각적으로 확인
2. **과적합 감지**: 검증 loss가 상승하는 시점 파악
3. **모델 비교**: 여러 실험의 그래프를 비교하여 최적 설정 찾기
4. **논문/보고서**: 고품질 그래프를 논문에 직접 사용

---

## 🐛 문제 해결 (Troubleshooting)

### matplotlib을 찾을 수 없음
```bash
pip install matplotlib
```

### 그래프가 생성되지 않음
- `training_metrics.jsonl` 파일이 존재하는지 확인
- 파일에 데이터가 있는지 확인 (최소 1개 epoch 필요)

### NumPy 버전 경고
현재 환경에서 NumPy 2.x와 PyTorch 간 호환성 경고가 발생할 수 있으나, 그래프 생성에는 영향 없음.

---

## 📚 자세한 문서 (Detailed Documentation)

전체 구현 세부사항은 [LOSS_GRAPH_IMPLEMENTATION.md](LOSS_GRAPH_IMPLEMENTATION.md)를 참조하세요.

---

## ⚙️ API Reference

### `plot_loss_graph(train_losses, val_losses, save_path)`
학습 및 검증 loss 그래프를 생성하고 이미지로 저장합니다.

**Parameters:**
- `train_losses` (list): 에폭별 학습 loss 리스트
- `val_losses` (list): 에폭별 검증 loss 리스트
- `save_path` (str): 그래프 저장 경로 (예: 'loss_graph.png')

**Returns:** None

**Example:**
```python
plot_loss_graph([5.0, 4.5, 4.0], [5.2, 4.7, 4.2], 'graph.png')
```

### `load_training_metrics(metrics_file)`
training_metrics.jsonl 파일에서 loss 데이터를 로드합니다.

**Parameters:**
- `metrics_file` (str): 메트릭 파일 경로

**Returns:** 
- `train_losses` (list): 학습 loss 리스트
- `val_losses` (list): 검증 loss 리스트

**Raises:**
- `FileNotFoundError`: 파일이 존재하지 않을 때

**Example:**
```python
train_losses, val_losses = load_training_metrics('training_metrics.jsonl')
```

---

## 🙏 Credits

이 기능은 기존 train.py의 메트릭 저장 기능을 활용하여 구현되었습니다.
