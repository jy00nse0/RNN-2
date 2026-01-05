# Test scripts overview (`*.py`)

- **`sample_test.py`**: “코드가 실제로 돌아가는지” 확인하는 ** 테스트용** (작은 샘플 데이터 생성 → 짧게 학습 → BLEU 평가).
- **`test_new.py`**: 현재 가장 최신 러너. **정의된 실험 선택 실행(-e), resume, 최종 epoch BLEU만 평가, BLEU 로그 누적, loss graph 옵션**까지 포함.

---

## 1) `sample_test.py` — 스모크 테스트(모델 동작 확인용)

### 목적
- **“학습/추론 파이프라인이 일단 돌아가는지”** 빠르게 확인하는 용도.
- WMT 전체가 아니라, 원본 데이터에서 **일부 라인만 잘라 `data/sample100k`를 생성**한 뒤 학습/평가합니다.

### 동작 흐름
1. `make_sample_dataset()`
   - `data/wmt14_vocab50k/base/train.{en,de}`에서 앞부분 일부를 잘라서 `data/sample100k/train.{en,de}` 생성
   - valid/test도 일부를 잘라 생성(실제 숫자는 다소 불일치: docstring은 100 lines라지만 코드는 최대 20000 lines 사용)
2. `run_sample_training()`
   - `train.py --dataset sample100k ... --max-epochs 4 ...`로 짧게 학습
3. `run_sample_evaluation()`
   - `checkpoints/sample_test` 아래의 **가장 최신 timestamp 디렉토리**를 찾아 `calculate_bleu.py` 수행

### 특징/주의
- 빠르게 확인하기 위한 파일이라 **실험 테이블/여러 설정 조합/체계적 로그 관리**는 없음.
- `run_sample_evaluation()`은 항상 `--cuda`를 붙입니다(환경에 CUDA가 없으면 실패 가능).

---

## 2) `test_new.py` — 최신 테스트 러너(권장)

### 목적
- Table 기반 실험(`experiments` dict)을 **선택 실행/전체 실행**하면서,
- **resume 로직을 포함한 안정적인 학습 재개**,  
- **최종 epoch의 BLEU만 평가**,  
- **BLEU 결과를 파일로 누적 기록**,  
- (옵션) **loss graph 생성**까지 지원.

### 제공하는 CLI
- `--list` : 실험 이름 목록 출력
- `-e/--experiments` : 실행할 실험 지정 (`all` 포함)
- `--no-cuda` : CUDA 강제 비활성화
- `--plot-loss-graph` : 학습 후 loss graph 생성(= `training_metrics.jsonl` 기반)

### 동작 흐름(실험 1개 기준)
1. CUDA 사용 여부 결정 → `common_flags`에 `--cuda` 또는 빈 문자열
2. `save_path = /workspace/checkpoints/<exp_name>`
3. `find_latest_checkpoint(save_path)`로 `model_epoch*.pt` 중 가장 최신 epoch 탐색
4. 완료 epoch가 `max_epochs` 이상이면 학습 스킵
5. 아니면 `train.py` 실행
   - 체크포인트 있으면 `--resume <checkpoint_path>` 추가
   - 인자들은 `config['args']`에서 자동으로 CLI flag로 변환되어 붙음
6. **BLEU 평가는 “최종 epoch 1번만” 수행**
   - `calculate_bleu.py --epoch <max_epochs>`
   - 결과를 `bleu_scores.log`에 누적 기록
7. (옵션) `--plot-loss-graph`가 켜져있으면 `training_metrics.jsonl` 읽어서 loss graph 저장
8. 마지막에 BLEU 로그 파일 내용을 출력

### 특징/장점
- 이전 파일들에 비해:
  - **학습/재개/평가 흐름이 단일 루프로 정리됨**
  - BLEU를 `eval.log` 단일 파일로 덮어쓰지 않고, `bleu_scores.log`에 누적
  - 실험 선택 실행이 편리하고, loss graph가 생성되어 학습 중 loss 감소를 확인 가능

---

## 3) `total_test.py` — 초기 올인원(전체 실험 순회) 버전

### 목적
- `파일 내 experiments`에 정의된 실험을 **전부 순회하면서** 학습 + BLEU 평가를 수행하는 “전체 실행용” 러너.

### 동작 흐름(핵심)
- `checkpoints/<exp_name>`에 저장
- `args` 파일이 없으면 학습 실행, 있으면 스킵
- 그 다음 `find_latest_checkpoint()`로 resume 판단 후 재개 학습
- 평가: `calculate_bleu.py --epoch <max_epochs>` 실행 후 `eval.log`에 기록

### 특징/한계
- CLI가 없어서 **선택 실행이 불편** (항상 전체를 도는 구조)
- 로그가 실험별로 `train.log`, `eval.log`로 단순 분리
- BLEU 로그 “누적 요약” 같은 기능은 없음

---

## 4) `test.py` — dummy 테스트 코드 (추후 최종 테스트 파일로 업데이트 예정)
---
