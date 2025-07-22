from pathlib import Path

# Base directory (servebot folder)
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = DATA_DIR / "static"

# Dataset paths - single source file with filtering
DATASETS = {
    "all": STATIC_DIR / "atp_matches_all_levels.parquet",
}

# Model artifacts
MODELS_DIR = STATIC_DIR / "models"


def get_dataset_path(dataset_type="all"):
    return str(DATASETS[dataset_type])


def get_model_path(model_name="latest"):
    return MODELS_DIR / f"model_{model_name}"


def save_model_artifacts(dataset, model, model_name="latest"):
    import pickle

    import torch
    import yaml

    save_path = get_model_path(model_name)
    save_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_path / "model_weights.pt")

    # Save config dict as yaml
    with open(save_path / "model_config.yaml", "w") as f:
        yaml.dump(dataset.config, f)

    with open(save_path / "embeddings.pkl", "wb") as f:
        pickle.dump(dataset.embeddings, f)

    return save_path
