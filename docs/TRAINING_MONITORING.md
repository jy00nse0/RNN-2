# Training Monitoring & Debugging Features

This document describes the comprehensive debugging and monitoring capabilities added to `train.py`.

## New Features

### 1. Perplexity Calculation
Perplexity is calculated from cross-entropy loss to provide more interpretable metrics:
- Automatically capped at exp(100) to prevent overflow
- Displayed alongside loss metrics for both training and validation

### 2. Batch-level Statistics Logging
Detailed progress logging every 100 batches showing:
- Current loss value
- Perplexity
- Gradient norm (for detecting gradient explosion/vanishing)
- Current learning rate

Example output:
```
[Batch 0/350] Loss: 8.5234 | PPL: 5031.13 | Grad Norm: 2.3451 | LR: 1.000000
[Batch 100/350] Loss: 7.2341 | PPL: 1385.89 | Grad Norm: 1.8234 | LR: 1.000000
```

### 3. Sample Translation Generation
At the end of each epoch, 2-3 sample translations are generated and displayed to visualize model learning progress:

```
🔍 Sample Translations:

Example 1:
  SRC: <sos> how are you today ? <eos>
  TGT: <sos> wie geht es dir heute ? <eos>
  PRD: <sos> wie sind sie heute ? <eos>
```

### 4. Training Metrics Logging
All training metrics are automatically saved to `<save_path>/training_metrics.jsonl` in JSON Lines format for later analysis.

Each line contains:
- `epoch`: Epoch number
- `train_loss`: Training loss
- `train_ppl`: Training perplexity
- `val_loss`: Validation loss
- `val_ppl`: Validation perplexity
- `avg_grad_norm`: Average gradient norm
- `min_train_loss`: Minimum training loss in epoch
- `max_train_loss`: Maximum training loss in epoch
- `time_seconds`: Time taken for the epoch

### 5. NaN/Inf Detection
Early warning system that detects non-finite losses during training:
```
⚠️  WARNING: Non-finite loss detected at batch 123!
   Loss value: nan
```

When detected, the batch is skipped and training continues.

### 6. Enhanced Validation Metrics
Validation now shows additional statistics:
```
📊 Validation Metrics:
   Loss: 6.8234 | Perplexity: 919.10
   Min Loss: 6.1234 | Max Loss: 8.9012
```

### 7. Enhanced Epoch Summaries
Clear, structured epoch summaries with all key metrics:
```
======================================================================
[Epoch  1/10] Summary:
  Train Loss: 7.5234 | Train PPL: 1850.85
  Val Loss:   6.8234 | Val PPL:   919.10
  Avg Grad Norm: 1.9876
  Time: 0:45:23
======================================================================
```

### 8. Improved Model Saving Feedback
Clearer feedback when models are saved:
```
💾 Saving model (val_loss improved: 6.8234 → 6.2145)
```

## Usage

All features are enabled by default. Simply run training as usual:

```bash
python train.py --dataset wmt14-en-de --max-epochs 10
```

The new output will automatically include:
- Batch-level logging every 100 batches
- Sample translations after each epoch
- Enhanced metrics and summaries
- Metrics saved to `training_metrics.jsonl`

## Testing

### Unit Tests
Test individual functions:
```bash
python test_functions.py
```

Tests:
- Perplexity calculation
- Batch statistics logging
- Metrics file saving

### Demo
See the enhanced output format without running full training:
```bash
python demo_output.py
```

### Integration Test
Run a minimal training session to verify all features:
```bash
python test_training_monitoring.py
```

## Backward Compatibility

All changes are backward compatible:
- No changes to command-line arguments
- Existing training behavior is preserved
- All new parameters have sensible defaults
- Metrics logging is non-intrusive

## Benefits

1. **Real-time Monitoring**: See model progress without waiting for BLEU scores
2. **Early Problem Detection**: Catch NaN, gradient explosion immediately
3. **Qualitative Assessment**: View actual translations to verify learning
4. **Post-training Analysis**: Use saved metrics for plotting and debugging
5. **Better User Experience**: Clearer, more informative console output

## Files Modified

- `train.py`: Main training script with all new features

## Files Added

- `test_functions.py`: Unit tests for helper functions
- `demo_output.py`: Demo script showing enhanced output
- `test_training_monitoring.py`: Integration test
- `TRAINING_MONITORING.md`: This documentation

## Example Output

See the full expected output format in the problem statement or run `demo_output.py` to see it in action.
