# ServeBot: Tennis Match Prediction with Attention

## Project Overview

**Goal**: Predict tennis match outcomes using transformer attention mechanisms applied to sequence modeling of player match histories.

**Architecture**:
- Sequence-to-sequence model where each match is predicted based on historical context
- Player-specific embeddings for names, rankings, playing styles
- Attention over chronologically ordered match sequences
- Binary classification: predict winner between two players

**Key Innovation**: Unlike static feature approaches, we model tennis as a temporal sequence problem where recent match context and player matchup history inform predictions.

## Code Philosophy

### Tensor-First Operations
- **ALWAYS prefer vectorized tensor operations over Python loops**
- Use `torch.where()` instead of conditionals
- Leverage broadcasting instead of explicit iteration
- Batch operations whenever possible

```python
# ❌ Avoid Python loops
for i in range(batch_size):
    result[i] = process_sequence(data[i])

# ✅ Use vectorized operations
result = torch.stack([process_sequence(seq) for seq in data])  # Better
result = process_batch(data)  # Best - fully vectorized
```

### Clean & Minimal Code
- Functions should do one thing well
- Avoid deeply nested logic
- Use descriptive variable names
- Type hints are mandatory
- Keep functions under 50 lines when possible

### Memory Efficiency
- Use in-place operations where safe: `tensor.add_()`, `tensor.mul_()`
- Prefer `torch.cat()` over repeated appending
- Use appropriate dtypes: `torch.long` for indices, `torch.float32` for computations
- Clear unused tensors: `del tensor` in tight loops

## Project Structure

```
servebot/
├── data/
│   ├── dataset.py          # PyTorch Dataset classes, sequence batching
│   ├── sequences.py        # Sequence generation from match histories
│   ├── preprocess.py       # Data cleaning, binning, H2H calculations
│   ├── embeddings.py       # Categorical feature embedding creation
│   └── utils.py           # Tensor utilities, padding functions
├── model/
│   └── servebot.py        # Transformer architecture, attention layers
├── train_v2.py            # Training loop, validation, checkpointing
├── main.py                # Legacy training script
├── app.py                 # Inference API, model serving
└── paths.py               # File path constants
```

## Key Components

### PlayerIndex (`data/sequences.py`)
- Fast lookup structure: player → match indices
- Enables efficient sequence generation
- Powers historical feature retrieval

### Sequence Generation
- `create_match_specific_sequences()`: Core training data generation
- Combines both players' histories chronologically
- Handles variable-length sequences with padding

### Model Architecture (`model/servebot.py`)
- Transformer encoder for sequence modeling
- Player embeddings + positional encoding
- Multi-head attention over match history
- Binary classification head

## Common Commands

**All commands use `uv run` for dependency management:**

```bash
# Training
uv run python train.py

# Inference server
uv run python app.py

# Data scraping
uv run python data/scraper.py

# Data preprocessing (if needed)
uv run python -c "from data.preprocess import preprocess_dataframe; preprocess_dataframe('data.parquet')"

# Quick model test
uv run python -c "from model.servebot import ServeBot; model = ServeBot(config); print(model)"
```

## Development Workflow

1. **Data changes**: Update `data/preprocess.py` → regenerate embeddings
2. **Model changes**: Modify `model/servebot.py` → retrain from scratch
3. **Sequence logic**: Edit `data/sequences.py` → test with small dataset first
4. **Performance**: Profile with `torch.profiler` and `memory_profiler`

## Debugging Guidelines

### Tensor Shape Debugging
```python
# Add shape assertions liberally during development
assert tensor.shape == (batch_size, seq_len, embed_dim), f"Expected {(batch_size, seq_len, embed_dim)}, got {tensor.shape}"

# Use descriptive variable names that include shapes
player_embeddings_BxSxE = embed_layer(player_tokens_BxS)  # B=batch, S=sequence, E=embedding
```

### Memory Debugging
```python
# Track GPU memory usage
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Find memory leaks
torch.cuda.empty_cache()
```

## Performance Optimization

### Data Loading
- Use `num_workers > 0` in DataLoader for parallel loading
- Pin memory: `pin_memory=True` for GPU training
- Prefetch batches with `prefetch_factor=2`

### Training Speed
- Use `torch.compile()` for PyTorch 2.0+ models
- Mixed precision: `torch.cuda.amp.autocast()`
- Gradient accumulation for large effective batch sizes
- Profile with `torch.profiler` to identify bottlenecks

### Inference Optimization
- Use `model.eval()` and `torch.no_grad()`
- Batch multiple predictions when possible
- Consider TorchScript compilation for production: `torch.jit.script(model)`

## Data Pipeline Notes

### Embedding Strategy
- All categorical features → integer tokens → learned embeddings
- Padding token = 0 for variable-length sequences
- Unknown values → padding token (handled gracefully)

### Sequence Construction
- Chronological ordering is critical for temporal modeling
- Sequences include match context: surface, tournament, date
- Both players' perspectives included in training (data augmentation)

### Memory Considerations
- Sequences stored as `torch.LongTensor` (int64) for memory efficiency
- Embeddings computed on-the-fly during forward pass
- Large datasets may require memory mapping or chunked loading

## Testing

```bash
# Run all tests (when test suite exists)
uv run pytest tests/

# Quick smoke test
uv run python -c "from data.dataset import TennisDataset; print('Dataset import works')"

# Validate model forward pass
uv run python -c "
import torch
from model.servebot import ServeBot
model = ServeBot({'embed_dim': 128, 'num_heads': 8})
x = torch.randint(0, 1000, (2, 32, 10))  # (batch, seq, features)
y = model(x)
print(f'Output shape: {y.shape}')
"
```

## AI Assistant Context

**Domain**: Sports analytics, sequence modeling, transformer architectures
**Primary Language**: Python with PyTorch
**Code Style**: Functional programming preferred, minimal OOP
**Performance**: GPU-optimized tensor operations, memory-conscious
**Data**: Tennis match records with player statistics and match outcomes

**Current Issues to Watch**:
- Embedding vocabulary consistency between training/inference
- Memory usage with long sequences (64+ matches)
- Data freshness in inference pipeline
- Sequence padding and masking correctness

**Recent Changes**:
- Migrated from `main.py` to `train_v2.py` for training
- Attention mechanism improvements in progress
- Inference pipeline optimization ongoing

## File Patterns

- `*_token` columns contain integer embeddings
- `create_*_sequences()` functions generate training data
- `PlayerIndex` enables fast player-based lookups
- Config dicts passed throughout for hyperparameters
- Type hints expected on all function signatures

This project emphasizes clean, efficient tensor operations for scalable sports prediction modeling.
