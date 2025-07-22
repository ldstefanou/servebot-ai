# 🎾 Servebot-AI

Attention based tennis match outcome prediction. Models every match as a sequence of matches leading up to it from both players. Every match can attend to every previous match in which either player participated. Matches are represented as the difference between two player embeddings and some additional categoricals such as rank, tournament, round, etc..

## Performance

Training on the entire ATP history (~200k matches) takes about 30m for 5 epochs with the current model. This model, when validated on a holdout set of the US Open 2024 reaches about 86% validation accuracy, which is very significant. I haven't tested bigger/different architectures or validating in different ways so results may vary.

## How It Works

### Example Sequence

Here's what a typical batch looks like for Nadal vs Djokovic:

```
🎾 Match Sequence (Last 8 matches involving either player)
┌─────┬────────────┬──────────────┬──────────────┬─────────────────┬─────────┐
│ Pos │    Date    │   Winner     │    Loser     │   Tournament    │ Surface │
├─────┼────────────┼──────────────┼──────────────┼─────────────────┼─────────┤
│  1  │ 2023-01-15 │ Rafael Nadal │ A. Zverev    │ Australian Open │  Hard   │
│  2  │ 2023-03-10 │ N. Djokovic  │ C. Alcaraz   │ Indian Wells    │  Hard   │
│  3  │ 2023-05-28 │ Rafael Nadal │ N. Djokovic  │ Roland Garros   │  Clay   │ ⭐
│  4  │ 2023-07-02 │ N. Djokovic  │ J. Sinner    │ Wimbledon       │  Grass  │
│  5  │ 2023-09-08 │ N. Djokovic  │ D. Medvedev  │ US Open         │  Hard   │
│  6  │ 2024-01-20 │ J. Sinner    │ N. Djokovic  │ Australian Open │  Hard   │
│  7  │ 2024-06-05 │ Rafael Nadal │ C. Alcaraz   │ Roland Garros   │  Clay   │
│  8  │ 2024-07-14 │    ???       │     ???      │ 🔮 PREDICT     │  Clay   │
└─────┴────────────┴──────────────┴──────────────┴─────────────────┴─────────┘
```

### Attention Mask

The key insight: matches can only attend to previous matches involving the same players!

```
🔍 Player-Specific Attention Matrix (1=can attend, 0=blocked)

     1  2  3  4  5  6  7  8
  1  ◯  0  0  0  0  0  0  0   <- Match 1: only self (no previous matches)
  2  0  ◯  0  0  0  0  0  0   <- Match 2: only self (Djokovic, can't see Nadal match 1)
  3  1  1  ◯  0  0  0  0  0   <- Match 3: sees matches 1,2 (both players involved)
  4  0  1  1  ◯  0  0  0  0   <- Match 4: sees matches 2,3 (Djokovic involved)
  5  0  1  1  1  ◯  0  0  0   <- Match 5: sees matches 2,3,4 (Djokovic involved)
  6  0  1  1  1  1  ◯  0  0   <- Match 6: sees matches 2,3,4,5 (Djokovic involved)
  7  1  0  1  0  0  0  ◯  0   <- Match 7: sees matches 1,3 (Nadal involved)
  8  1  1  1  1  1  1  1  ◯   <- PREDICT: sees ALL previous matches with either player!

Legend: ◯ = self-attention, 1 = can attend, 0 = blocked
       Causal mask: can only look at previous positions (left & up)
       Player mask: can only see matches involving same players
```

This prevents the model from learning spurious patterns between unrelated players!

## Quick Start

```bash
# Run the setup script to download all the data and setup the environment
./setup.sh

# Train current model
uv run python train.py --epochs 5 --batch-size 16

# Launch the web app to make predictions (WIP)
streamlit run app.py
```


Built with PyTorch ❤️‍🔥 by Leandros Stefanou
