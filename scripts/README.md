##WMT 14,15 en, de 데이터 구축

실행:
```bash
bash scripts/prepare_data.sh
```

---

### 전체 파이프라인 개요 (3 Steps)

`scripts/prepare_data.sh`는 다음 3단계를 순서대로 수행합니다.

1. **Raw 데이터 다운로드 (Hugging Face)**
2. **Moses tokenizer로 토큰화**
3. **Train 기준 Top-50K vocab 구축 후 `<unk>` 치환(필터링)**

각 단계가 끝날 때마다 성공/실패 로그가 출력되며, 오류가 나면 스크립트가 즉시 중단됩니다(`set -e`).

---

## Step 1/3. WMT14/WMT15 Raw 데이터 다운로드 (`scripts/download_data.py`)

스크립트는 먼저 아래 명령을 실행합니다.

```bash
python scripts/download_data.py
```

이 스크립트가 하는 일:

- Hugging Face `datasets`를 사용해 데이터셋을 로드
  - **WMT14:** `load_dataset("wmt14", "de-en")`
    - 저장은 **en → de** 방향으로 저장됨
  - **WMT15:** `load_dataset("wmt15", "de-en")`
    - 저장은 **de → en** 방향으로 저장됨
- 각 split(train/validation/test)에 대해 다음 전처리를 수행하여 파일로 저장:
  - 공백 기준 토큰화(`.split()`)
  - 빈 문장 제거
  - **문장 길이 50 토큰 초과는 제거** (`max_len=50`)
  - (주석에 명시된 것처럼) **Reverse 로직은 없음**: 정방향 그대로 저장

생성되는 디렉토리/파일:

- `data/wmt14_raw/`
  - `train.clean.en`, `train.clean.de`
  - `valid.clean.en`, `valid.clean.de`
  - `test.clean.en`, `test.clean.de`
- `data/wmt15_raw/`
  - `train.clean.de`, `train.clean.en`
  - `valid.clean.de`, `valid.clean.en`
  - `test.clean.de`, `test.clean.en`

---

## Step 2/3. Moses tokenizer로 토큰화 (`tokenizer.perl`)

다음으로 Moses tokenizer를 이용해 토큰화를 수행합니다.

- 출력 디렉토리 생성:
  - `data/wmt14_tokenized/`
  - `data/wmt15_tokenized/`
- 실행 전에 **현재 작업 디렉토리에 `tokenizer.perl` 파일이 존재해야 합니다.**
  - 없으면 즉시 종료합니다.

토큰화 결과:

- `data/wmt14_raw/*.clean.{en,de}` → `data/wmt14_tokenized/*.{en,de}`
- `data/wmt15_raw/*.clean.{en,de}` → `data/wmt15_tokenized/*.{en,de}`

---

## Step 3/3. Vocabulary(Top-50K) 구축 + `<unk>` 치환 필터링 (`scripts/process_data.py`)

마지막으로 다음 명령을 실행합니다.

```bash
python scripts/process_data.py \
  --raw_dir data/wmt14_tokenized \
  --out_dir data/wmt14_vocab50k \
  --src_lang en \
  --tgt_lang de

python scripts/process_data.py \
  --raw_dir data/wmt15_tokenized \
  --out_dir data/wmt15_vocab50k \
  --src_lang en \
  --tgt_lang de
```

`process_data.py`가 하는 일:

1. **Vocabulary 구축 (Train split 기준)**
   - 소스 언어(train) 파일에서 상위 **50,000개** 단어를 vocab으로 생성
   - 타겟 언어(train) 파일에서도 상위 **50,000개** 단어를 vocab으로 생성
2. **데이터 필터링/저장**
   - train/valid/test 각각에 대해:
     - vocab에 없는 토큰은 모두 `<unk>`로 치환
     - 결과 파일을 `out_dir/base/` 아래에 저장
   - unknown token 비율(%) 로그 출력

생성되는 디렉토리 구조:

- `data/wmt14_vocab50k/base/`
  - `train.en`, `train.de`
  - `valid.en`, `valid.de`
  - `test.en`, `test.de`
- `data/wmt15_vocab50k/base/`
  - `train.en`, `train.de`
  - `valid.en`, `valid.de`
  - `test.en`, `test.de`

---

## 실행 완료 후 출력

파이프라인 완료 후:

- `data/wmt14_vocab50k/base/train.en`의 앞 3줄을 출력합니다.
- 전체 소요 시간을 `Xm Ys` 형태로 출력합니다.
- 다음 단계 예시로 학습 실행 커맨드를 출력합니다.

```bash
python train.py --dataset wmt14-en-de --cuda
```

---
