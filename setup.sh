#!/bin/bash
set -e  # Exit on any error

echo "🏆 Setting up Servebot repository..."

# Create data/static directory
echo "📁 Creating data directories..."
mkdir -p servebot/data/static

# Install dependencies with uv
echo "📦 Installing dependencies..."
if command -v uv &> /dev/null; then
    echo "Using uv to install dependencies..."
    uv sync --group dev
else
    echo "Error: uv not found. Please install uv first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Set up pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
uv run pre-commit install

# Download ATP data
echo "🎾 Downloading ATP tennis data..."
uv run python fetch_data.py

# Move the generated file to the correct location
if [ -f "atp_matches_all_levels.parquet" ]; then
    mv atp_matches_all_levels.parquet servebot/data/static/
    echo "✅ Data moved to servebot/data/static/"
else
    echo "❌ Failed to generate ATP data file"
    exit 1
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source .venv/bin/activate (if not using uv run)"
echo "  2. Train model: uv run python servebot/train.py"
echo "  3. Make predictions: uv run python predictor.py"
