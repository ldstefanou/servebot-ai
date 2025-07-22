# 🎾 Servebot-AI

Attention based tennis match outcome prediction. Models every match as a sequence of matches leading up to it from both players. Every match can attend to every previous match in which either player participated. Matches are represented as the difference between two player embeddings and some additional categoricals such as rank, tournament, round, etc..

## Performance

Training on the entire ATP history (~200k matches) takes about 30m for 5 epochs with the current model. This model, when validated on a holdout set of the US Open 2024 reaches about 86% validation accuracy, which is very significant. I haven't tested bigger/different architectures or validating in different ways so results may vary.

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
