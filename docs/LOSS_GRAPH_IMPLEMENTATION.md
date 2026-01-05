# Loss Graph Functionality Implementation Summary

## 개요 (Overview)
이 문서는 학습 중 모델의 loss를 실시간으로 기록하고, 학습 종료 후 loss 그래프를 생성하여 이미지로 저장하는 기능의 구현 내용을 설명합니다.

This document describes the implementation of functionality to record model loss in real-time during training and generate loss graphs as images after training completion.

## 문제 분석 (Problem Analysis)

### 요구사항 (Requirements)
1. train.py에 현재 import된 라이브러리만으로 실시간 모델 학습 loss 기록 가능 여부 추론
2. util.py에 학습 중 모델 loss 그래프 그리는 코드 기능 삽입
3. test_new.py에서 학습 종료 후 학습 중 모델 loss 그래프를 생성하여 이미지로 저장

### 기존 라이브러리 분석 (Existing Library Analysis)
train.py에서 이미 import되어 있는 라이브러리:
- `json` - 데이터 저장에 사용 가능 ✓
- `numpy` - 배열 처리에 사용 가능 ✓
- **결론**: Loss 데이터 기록은 가능하지만, 그래프 생성을 위해서는 `matplotlib` 필요

## 구현 내용 (Implementation)

### 1. requirements.txt 수정
```diff
+ matplotlib
```
그래프 생성을 위해 matplotlib 라이브러리 추가

### 2. util.py 기능 추가

#### 2.1 plot_loss_graph() 함수
**기능**: 학습 및 검증 loss 데이터를 받아 그래프를 생성하고 이미지 파일로 저장

**매개변수**:
- `train_losses`: 에폭별 학습 loss 리스트
- `val_losses`: 에폭별 검증 loss 리스트  
- `save_path`: 그래프 이미지 저장 경로

**주요 특징**:
- 비대화형 백엔드 사용 ('Agg') - 서버 환경에서도 동작
- 높은 해상도 (300 DPI)
- 깔끔한 디자인 (그리드, 범례, 레이블)
- 자동 디렉토리 생성

#### 2.2 load_training_metrics() 함수
**기능**: training_metrics.jsonl 파일에서 loss 데이터 로드

**매개변수**:
- `metrics_file`: 메트릭 파일 경로

**반환값**:
- `train_losses`: 학습 loss 리스트
- `val_losses`: 검증 loss 리스트

**특징**:
- JSONL 형식 파싱
- 에러 처리 포함

### 3. test_new.py 수정

#### 3.1 Import 추가
```python
from util import load_training_metrics, plot_loss_graph
```

#### 3.2 학습 완료 후 그래프 생성 로직 추가
```python
# 5. Generate Loss Graph after training
metrics_file = os.path.join(save_path, 'training_metrics.jsonl')
if os.path.exists(metrics_file):
    try:
        print(f"\n>>> Generating loss graph...")
        train_losses, val_losses = load_training_metrics(metrics_file)
        
        if train_losses and val_losses:
            loss_graph_path = os.path.join(save_path, 'loss_graph.png')
            plot_loss_graph(train_losses, val_losses, loss_graph_path)
            print(f"    Loss graph generated successfully!")
        else:
            print(f"    Warning: No loss data found in metrics file")
    except Exception as e:
        print(f"    Error generating loss graph: {e}")
else:
    print(f"    Warning: Metrics file not found: {metrics_file}")
```

## 사용 방법 (Usage)

### 1. 라이브러리 설치
```bash
pip install matplotlib
```

### 2. 학습 실행
test_new.py를 실행하면 자동으로:
1. 학습 진행
2. 학습 메트릭 저장 (training_metrics.jsonl)
3. 학습 완료 후 자동으로 loss 그래프 생성 (loss_graph.png)

```bash
python test_new.py -e T1_Base_Reverse
```

### 3. 출력 파일
각 실험의 checkpoint 디렉토리에 다음 파일들이 생성됨:
- `training_metrics.jsonl` - 에폭별 메트릭 데이터
- `loss_graph.png` - Loss 시각화 그래프 (300 DPI)

## 테스트 결과 (Test Results)

### 테스트 환경
- Python 3.12
- matplotlib 3.10.8
- torch 2.2.2

### 테스트 케이스
1. **직접 플로팅 테스트**: ✓ PASSED
   - 수동으로 생성한 loss 데이터로 그래프 생성
   - 파일 크기: ~143KB
   
2. **메트릭 로드 및 플로팅 테스트**: ✓ PASSED
   - JSONL 파일에서 데이터 로드
   - 그래프 생성 및 저장
   - 파일 크기: ~144KB

### 생성된 그래프 예시
![Loss Graph Example](https://github.com/user-attachments/assets/cc5806d5-4227-4d4f-adf1-e51d6419ecbc)

**그래프 특징**:
- 파란색: 학습 Loss
- 빨간색: 검증 Loss
- X축: Epoch
- Y축: Loss 값
- 그리드와 범례 포함

## 기술적 세부사항 (Technical Details)

### 1. 비대화형 백엔드
```python
matplotlib.use('Agg')
```
서버 환경에서 GUI 없이 그래프 생성 가능

### 2. 높은 품질 출력
```python
plt.savefig(save_path, dpi=300, bbox_inches='tight')
```
- 300 DPI 고해상도
- 여백 최적화 (bbox_inches='tight')

### 3. 자동 디렉토리 생성
```python
os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
```

## 호환성 (Compatibility)

### 기존 코드와의 호환성
- ✓ train.py 수정 없음 (이미 metrics 저장 중)
- ✓ 기존 학습 프로세스 영향 없음
- ✓ 선택적 기능 (그래프 생성 실패 시에도 학습 계속)

### 최소 요구사항
- Python 3.6+
- matplotlib
- numpy (이미 설치됨)
- JSON 지원 (Python 기본 라이브러리)

## 결론 (Conclusion)

### 구현된 기능
✓ 실시간 loss 기록 (기존 train.py에서 이미 구현됨)
✓ loss 그래프 생성 함수 (util.py)
✓ 자동 그래프 생성 및 저장 (test_new.py)

### 추가 이점
- 학습 진행 상황 시각화
- 과적합(overfitting) 감지 용이
- 모델 성능 비교 용이
- 논문/보고서용 고품질 그래프

### 향후 개선 가능 사항
- 여러 실험 결과 비교 그래프
- 추가 메트릭 시각화 (perplexity, gradient norm 등)
- 인터랙티브 그래프 생성 옵션
