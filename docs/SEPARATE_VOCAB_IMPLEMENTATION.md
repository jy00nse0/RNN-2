# Separate Source/Target Vocabulary Support - Implementation Summary

## Problem Statement

The original inference code (`Seq2SeqPredict` in `predict.py` and `calculate_bleu.py`) used a single vocabulary (`field`) for both source (encoder input) and target (decoder output) processing. This caused failures when models were trained with different source and target vocabulary sizes, resulting in:
- GPU indexing asserts
- Out-of-range embedding indices during encoder embedding lookup
- Inability to run BLEU evaluation on models with separate vocabularies

## Solution Overview

The fix separates source and target vocabulary handling throughout the inference pipeline:

### 1. Core Model Changes (`model/seq2seq/model.py`)

**Before:**
```python
class Seq2SeqPredict(nn.Module):
    def __init__(self, encoder, decoder, field):
        self.field = field
        self.sos_idx = field.vocab.stoi[SOS_TOKEN]
        self.eos_idx = field.vocab.stoi[EOS_TOKEN]
```

**After:**
```python
class Seq2SeqPredict(nn.Module):
    def __init__(self, encoder, decoder, src_field, tgt_field):
        self.src_field = src_field  # For encoder input processing
        self.tgt_field = tgt_field  # For decoder output decoding
        self.sos_idx = tgt_field.vocab.stoi[SOS_TOKEN]
        self.eos_idx = tgt_field.vocab.stoi[EOS_TOKEN]
```

**Key Changes:**
- Accepts `src_field` and `tgt_field` instead of single `field`
- Uses `src_field` for preprocessing source text into encoder input tensors
- Uses `tgt_field` for decoding generated token IDs back to text
- SOS/EOS indices come from target vocabulary

### 2. Model Factory Update (`model/__init__.py`)

**Before:**
```python
def predict_model_factory(args, src_metadata, tgt_metadata, model_path, field):
    return Seq2SeqPredict(train_model.encoder, train_model.decoder, field)
```

**After:**
```python
def predict_model_factory(args, src_metadata, tgt_metadata, model_path, src_field, tgt_field):
    return Seq2SeqPredict(train_model.encoder, train_model.decoder, src_field, tgt_field)
```

### 3. Vocabulary Loading (`predict.py` and `calculate_bleu.py`)

**Backward-Compatible Loading:**
```python
# Try to load separate vocabularies
src_vocab_path = os.path.join(args.model_path, 'src_vocab')
tgt_vocab_path = os.path.join(args.model_path, 'tgt_vocab')
legacy_vocab_path = os.path.join(args.model_path, 'vocab')

src_vocab = load_object(src_vocab_path) if os.path.exists(src_vocab_path) else None
tgt_vocab = load_object(tgt_vocab_path) if os.path.exists(tgt_vocab_path) else None

# Fallback to legacy single vocab if either is missing
if src_vocab is None or tgt_vocab is None:
    legacy_vocab = load_object(legacy_vocab_path)
    if src_vocab is None:
        src_vocab = legacy_vocab
    if tgt_vocab is None:
        tgt_vocab = legacy_vocab
```

### 4. SimpleField Implementation

A new `SimpleField` class wraps vocabularies with configurable preprocessing:

```python
class SimpleField:
    def __init__(self, vocab, add_sos=False, add_eos=True):
        self.vocab = vocab
        self.add_sos = add_sos
        self.add_eos = add_eos
```

**Usage:**
- Source field: `SimpleField(src_vocab, add_sos=False, add_eos=True)` - matches training behavior
- Target field: `SimpleField(tgt_vocab, add_sos=False, add_eos=False)` - SOS/EOS managed by model

### 5. BLEU Evaluation Switch (`calculate_bleu.py`)

**Before:**
```python
import sacrebleu
bleu = sacrebleu.corpus_bleu(answers, [ref_answers])
```

**After:**
```python
def calculate_bleu_with_perl(hypotheses, reference_path, lowercase=False):
    script_path = os.path.join(os.path.dirname(__file__), 'multi-bleu.perl')
    cmd = ['perl', script_path]
    if lowercase:
        cmd.append('-lc')
    cmd.append(reference_path)
    result = subprocess.run(cmd, input='\n'.join(hypotheses) + '\n', ...)
    return result.stdout.strip()

bleu_output = calculate_bleu_with_perl(answers, args.reference_path, lowercase=args.lowercase)
```

**Features:**
- Uses `multi-bleu.perl` script included in repository
- Supports `--lowercase` flag for case-insensitive evaluation
- Better error messages with perl installation instructions
- Validates script existence before running

## Critical Implementation Details

### Source Preprocessing

The source sequence preprocessing **must** match training behavior:

**Training (from `dataset.py`):**
```python
def __getitem__(self, idx):
    src = self.src_sentences[idx] + ['<eos>']  # Add <eos> ONLY
    tgt = ['<sos>'] + self.tgt_sentences[idx] + ['<eos>']
```

**Inference (now matches):**
```python
src_field = SimpleField(src_vocab, add_sos=False, add_eos=True)  # <eos> only
tgt_field = SimpleField(tgt_vocab, add_sos=False, add_eos=False)  # None
```

This is crucial because:
1. Encoder doesn't need `<sos>` - it just reads the sentence
2. Adding `<sos>` to encoder input would cause vocabulary index mismatches
3. Decoder's `<sos>` and `<eos>` are handled by `Seq2SeqPredict.samplers`

## Testing

### Test Suite
1. **test_separate_vocab_inference.py** - Unit tests for SimpleField and Seq2SeqPredict
2. **test_bleu_integration.py** - Tests multi-bleu.perl integration
3. **test_e2e_separate_vocab.py** - End-to-end checkpoint loading and inference
4. **demo_separate_vocab.py** - Demonstration of the feature

All tests pass, confirming:
- Separate vocabularies work correctly
- Backward compatibility maintained
- BLEU calculation works with multi-bleu.perl
- Source preprocessing matches training

## Backward Compatibility

The implementation maintains full backward compatibility:

1. **Legacy checkpoints** with single `vocab` file work without modification
2. **Fallback logic** uses legacy vocab if `src_vocab` or `tgt_vocab` missing
3. **No changes** required to existing trained models
4. **Training code** unaffected - continues to save both `src_vocab` and `tgt_vocab`

## Files Modified

1. `model/seq2seq/model.py` - Updated `Seq2SeqPredict` to accept separate fields
2. `model/__init__.py` - Updated `predict_model_factory` signature
3. `predict.py` - Added vocab loading logic and SimpleField
4. `calculate_bleu.py` - Added vocab loading, SimpleField, and multi-bleu.perl integration

## Benefits

✅ Models with different source/target vocabulary sizes now work during inference
✅ Eliminates GPU indexing errors and embedding lookup failures
✅ BLEU evaluation uses standard multi-bleu.perl script
✅ Backward compatible with legacy single-vocab checkpoints
✅ Source preprocessing correctly matches training behavior
✅ Better error messages and validation

## Usage Examples

### Using predict.py with separate vocabularies
```bash
python predict.py -p /path/to/model -e 10 --cuda
# Automatically loads src_vocab and tgt_vocab, falls back to vocab if needed
```

### Using calculate_bleu.py
```bash
python calculate_bleu.py -p /path/to/model -e 10 -r data/test.de --cuda
# Output: BLEU = 27.45, 58.2/33.1/20.4/13.0 (BP=0.978, ratio=0.978, hyp_len=68342, ref_len=69876)

# With lowercase evaluation
python calculate_bleu.py -p /path/to/model -e 10 -r data/test.de --cuda --lowercase
```

## Conclusion

This implementation resolves the core issue of inference failures with separate vocabularies while maintaining backward compatibility and improving BLEU evaluation to use the standard multi-bleu.perl script.
