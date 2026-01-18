# RNN-based Neural Machine Translation

This repository implements RNN-based sequence-to-sequence models for Neural Machine Translation, reproducing experiments from Luong et al. (2015) "Effective Approaches to Attention-based Neural Machine Translation".

## Features

- **Multiple Attention Mechanisms**: 
  - Global attention (dot, general, concat, location score functions)
  - Local attention (monotonic and predictive variants)
  - No attention (base seq2seq model)
- **Paper-accurate Implementation**: Follows Luong et al. (2015) specifications
- **WMT14/15 Support**: English↔German translation
- **BLEU Evaluation**: Uses sacreBLEU for consistent scoring

## Quick Start

### Prerequisites

```bash
# Python 3.12 recommended
pip install torch==2.2.2 "numpy<2" datasets sentencepiece sacrebleu tqdm
```

### Running Tests

This repository includes two test scripts:

#### 1. Sample Test (Quick Demo)

Runs a minimal training and evaluation cycle for testing the pipeline:

```bash
python sample_test.py
```

This will:
- Create a small sample dataset (1,000 training examples)
- Train a small model (2 layers, 256 hidden units) for 1 epoch
- Evaluate BLEU score on test set
- Complete in ~5-10 minutes on CPU

#### 2. Total Test (Full Reproduction)

Runs all 20 experiments from the paper (WARNING: Takes many hours/days):

```bash
python total_test.py
```

This script reproduces:
- **Table 1**: WMT14 En→De experiments (8 configurations)
- **Table 3**: WMT15 De→En experiments (6 configurations)  
- **Table 4**: Attention ablation study (6 configurations)

Each experiment trains for 10-12 epochs with full-size models (4 layers, 1000 hidden units).

### Manual Training

To train a specific model configuration:

```bash
python train.py \
  --dataset wmt14-en-de \
  --save-path checkpoints/my_model \
  --max-epochs 10 \
  --batch-size 128 \
  --learning-rate 1.0 \
  --encoder-hidden-size 1000 \
  --decoder-hidden-size 1000 \
  --encoder-num-layers 4 \
  --decoder-num-layers 4 \
  --attention-type global \
  --attention-score dot \
  --reverse \
  --cuda
```

Key arguments:
- `--dataset`: Dataset to use (`wmt14-en-de`, `wmt15-deen`, `sample100k`)
- `--attention-type`: Attention mechanism (`none`, `global`, `local-m`, `local-p`)
- `--attention-score`: Score function (`dot`, `general`, `concat`, `location`)
- `--reverse`: Reverse source sentences (recommended for better performance)
- `--luong-input-feed`: Enable input feeding (for attention models)

### Length-Bucketed Batching

To improve training throughput by reducing padding, enable length-bucketed batching:

```bash
python train.py \
  --dataset wmt14-en-de \
  --use-bucketing \
  --bucket-by src \
  --batch-size 128 \
  --cuda
```

Bucketing arguments:
- `--use-bucketing`: Enable bucketing (groups sequences by similar lengths)
- `--bucket-by`: Bucket by `src` (source) or `tgt` (target) length (default: `src`)
- `--bucket-drop-last`: Drop last incomplete batch (default: False)
- `--bucket-seed`: Random seed for batch shuffle (default: 42)

Bucketing groups examples with similar sequence lengths into batches, reducing the amount of padding needed. The batch order is shuffled differently each epoch while keeping sequences within each batch grouped by length.

### Evaluation

To calculate BLEU score on a trained model:

```bash
python calculate_bleu.py \
  --model-path checkpoints/my_model/2025-12-06-12-00 \
  --reference-path data/wmt14_vocab50k/base/test.de \
  --epoch 10 \
  --sampling-strategy greedy
```

## Data Preparation

### Quick Start (Mock Data)

For quick testing without downloading real datasets:

```bash
python create_mock_data.py
```

This creates synthetic data in `data/wmt14_vocab50k/base/` for immediate testing.

### Real Data (Requires Internet)

1. **Download WMT14/15 datasets:**
   ```bash
   python scripts/download_data.py
   ```

2. **Process data (build vocabulary, apply filtering):**
   ```bash
   # For WMT14 En→De
   python scripts/process_data.py --raw_dir data/wmt14_raw --out_dir data/wmt14_vocab50k
   
   # For WMT15 De→En  
   python scripts/process_data.py --raw_dir data/wmt15_raw --out_dir data/wmt15_vocab50k --src_lang de --tgt_lang en
   ```

Note: The download script requires internet access to Hugging Face datasets. If unavailable, use the mock data option.

## Model Architecture

Based on Luong et al. (2015):

- **Encoder**: Multi-layer LSTM (4 layers, 1000 hidden units)
- **Decoder**: Multi-layer LSTM with optional attention (4 layers, 1000 hidden units)
- **Embeddings**: 1000 dimensions
- **Vocabulary**: 50,000 tokens (BPE-based)
- **Dropout**: 0.2 between LSTM layers (Zaremba-style)
- **Optimization**: SGD with learning rate 1.0, halving schedule after epoch 5/8

## Implementation Details

### Key Modifications from Standard Implementations

1. **Dataset Loading**: Custom implementation without legacy torchtext APIs
2. **Attention Support**: All variants from paper (global/local-m/local-p with dot/general/concat/location scores)
3. **No Attention Mode**: Base seq2seq without attention mechanism
4. **Dropout**: Zaremba-style (between layers only, not recurrent)
5. **Source Reversing**: Configurable via `--reverse` flag

### Differences from Paper

- Uses mock/synthetic data by default (real WMT data requires download)
- Simplified vocabulary (uses full tokens, not BPE in default setup)
- Default tests use smaller models for speed (full models available via total_test.py)

## Project Structure

```
.
├── train.py                 # Main training script
├── calculate_bleu.py        # BLEU evaluation
├── sample_test.py          # Quick test script
├── total_test.py           # Full paper reproduction
├── create_mock_data.py     # Generate synthetic test data
├── dataset.py              # Data loading utilities
├── model/                  # Model implementations
│   ├── seq2seq/
│   │   ├── encoder.py     # Encoder implementations
│   │   ├── decoder.py     # Decoder implementations
│   │   ├── attention.py   # Attention mechanisms
│   │   └── model.py       # Seq2Seq wrapper
│   └── __init__.py
├── scripts/               # Data preparation scripts
│   ├── download_data.py  # Download WMT datasets
│   └── process_data.py   # Process and filter data
└── README.md

```

## Citation

If you use this code, please cite:

```
Luong, M. T., Pham, H., & Manning, C. D. (2015). 
Effective approaches to attention-based neural machine translation. 
arXiv preprint arXiv:1508.04025.
```

## Notes

- **Training Time**: Full experiments take days on CPU, hours on GPU
- **Memory**: Full models require ~4GB GPU memory with batch size 128
- **Quick Testing**: Use `sample_test.py` for pipeline verification
- **Production Use**: This is a research/educational implementation; for production, consider Transformer-based models