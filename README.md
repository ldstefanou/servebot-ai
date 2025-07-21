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
