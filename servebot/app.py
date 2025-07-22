import pickle

import pandas as pd
import streamlit as st
import torch
from data.dataset import TennisDataset
from data.preprocess import load_data
from data.sequences import PlayerIndex
from model.servebot import TennisTransformer
from torch.utils.data import DataLoader, SubsetRandomSampler


def create_dataset(embeddings, bank, config):
    config["seq_length"] = len(bank) + 1
    dset = TennisDataset(embeddings, bank, config, training_mode=False)

    # Use SubsetRandomSampler to get only the last sequence (most complete context)
    last_idx = len(dset) - 1
    sampler = SubsetRandomSampler([last_idx])
    dloader = DataLoader(dset, batch_size=1, sampler=sampler)
    batch = list(dloader)[0]

    # UGLY FIX: Clip position tokens to model's trained range
    batch["position_token"] = torch.clamp(batch["position_token"], max=128)

    return batch


def create_historical_bank(df, index: PlayerIndex, player_1: str, player_2: str):
    bank_idx = index.get_matches_by_players(player_1, player_2)

    filtered = df.iloc[bank_idx]
    # Convert to records (list of dicts)
    records = filtered.to_dict("records")

    # Get last record and copy it
    last_record = records[-1].copy()

    # Update with new players
    last_record["winner_name"] = player_1
    last_record["loser_name"] = player_2

    # Append to records
    records.append(last_record)

    # Convert back to dataframe
    extended_df = pd.DataFrame(records).reset_index(drop=True)
    extended_df["is_validation"] = False

    return extended_df


@st.cache_resource
def load_model():
    """Load and cache the trained model and embeddings"""
    try:
        model_path = "servebot/data/static/models/model_epoch_0"

        # Load embeddings
        with open(f"{model_path}/embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)

        # Load model
        model = TennisTransformer.from_saved_artifacts(model_path, embeddings)
        model.eval()  # Set to evaluation mode

        return model, embeddings
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None


@st.cache_data
def load_player_index(df):
    """Load unique players from the dataset"""
    index = PlayerIndex(df)
    return index


st.title("🎾 Servebot")
st.write("Select two players to predict match outcome")

# load sequence bank
df = load_data()

# Load players
index = load_player_index(df)

# model
model, embeddings = load_model()

# Searchable dropdowns
player1 = st.selectbox(
    "Player 1",
    options=index.players,
    index=None,
    placeholder="Search and select player...",
)
player2 = st.selectbox(
    "Player 2",
    options=index.players,
    index=None,
    placeholder="Search and select player...",
)

if st.button("Predict"):
    if player1 and player2:
        if player1 == player2:
            st.error("Please select different players!")
        else:
            with st.spinner("Generating prediction..."):
                bank = create_historical_bank(df, index, player1, player2)
                batch = create_dataset(embeddings, bank, model.config)
                with torch.no_grad():
                    out = model(batch)
                    probabilities = out.softmax(-1)

                st.write(f"🎾 **{player1}** vs **{player2}**")
                st.write(f"📊 Historical matches found: {len(bank)}")

                # Show recent match history
                if len(bank) > 0:
                    st.write("### Recent Match History")
                    n_recent = min(25, len(bank))  # Show last 10 matches or fewer
                    recent_matches = bank.tail(n_recent).copy()

                    # Format the display dataframe
                    display_df = pd.DataFrame(
                        {
                            "Date": recent_matches["date"].dt.strftime("%Y-%m-%d"),
                            "Left Player": recent_matches["winner_name"],
                            "Right Player": recent_matches["loser_name"],
                            "Winner": recent_matches[
                                "winner_name"
                            ],  # Winner is always the left player in this context
                            "Tournament": recent_matches["tournament"],
                        }
                    )

                    # Reverse to show most recent first
                    display_df = display_df.iloc[::-1].iloc[1:].reset_index(drop=True)

                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Show the prediction for the last position (most recent context)
                last_prob = probabilities[0, -1]  # [batch=1, last_position, classes]
                player1_win_prob = last_prob[
                    1
                ].item()  # Assuming class 1 is "player1 wins"
                player2_win_prob = last_prob[
                    0
                ].item()  # Assuming class 0 is "player2 wins"

                st.write("### Prediction:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"{player1} Win Probability", f"{player1_win_prob:.1%}")
                with col2:
                    st.metric(f"{player2} Win Probability", f"{player2_win_prob:.1%}")

                # Show raw model output for debugging
                with st.expander("Debug: Raw model output"):
                    st.write("Output shape:", out.shape)
                    st.write("Probabilities:", probabilities[0].tolist())
    else:
        st.write("Please select both players")
