# Servebot

A transformer-based tennis match prediction system that uses player-specific attention to model match outcomes.

## Approach

The model treats tennis match prediction as a sequence modeling problem. For each match between players A and B, it creates a sequence from both players' historical matches leading up to the current match.

Key innovation: **player-specific attention masking** ensures each historical match only attends to previous matches involving the same players, preventing the model from learning spurious correlations between unrelated matches.

## Features

- Custom attention masks for coherent player histories
- Continuous time encoding for temporal match patterns
- Relative player strength embeddings
- Head-to-head record tracking
- Multiple sequence construction strategies

## Quick Start

1. **Setup environment and data**
   ```bash
   ./setup.sh
   ```

2. **Train model**
   ```bash
   # Train on all matches (ATP + Futures + Qualifiers)
   uv run python servebot/train.py --dataset all

   # Train on ATP main tour only
   uv run python servebot/train.py --dataset atp
   ```

3. **Available options**
   - `--dataset`: `all` (default) or `atp` for main tour only
   - `--sequence-type`: `player`, `temporal`, or `match`
   - `--sample N`: Use only N matches for testing
   - `--epochs N`: Number of training epochs
   - `--batch-size N`: Batch size
