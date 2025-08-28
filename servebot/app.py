import pickle

import pandas as pd
import streamlit as st
import torch
from data.encodings import apply_encoders

from servebot.data.dataset import SimpleDSet
from servebot.data.preprocess import load_data
from servebot.data.sequences import PlayerIndex, create_match_specific_sequences
from servebot.model.servebot import TennisTransformer


@st.cache_resource
def load_model():
    """Load and cache the trained model and embeddings"""
    try:
        model_path = "servebot/data/static/models/model_epoch_4"

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
    return PlayerIndex(df)


st.title("🎾 Servebot")
st.write("Select two players to predict match outcome")

# model
model, encoders = load_model()


# load sequence bank
df = load_data(filter_matches="atp")
df = apply_encoders(df, encoders)

# Load players
index = load_player_index(df)

# match_df = index.create_match_for_players("Dennis Novak", "Novak Djokovic", encoders)
# sequences = create_match_specific_sequences(match_df, index, model.config["columns"], model.config["seq_length"])
# batch = torch.utils.data.DataLoader(SimpleDSet(sequences, training=False))
# out = model(next(iter(batch)))

# print(out.softmax(dim=-1))

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
                bank_idx = index.get_matches_by_players(player1, player2)
                bank = index.df.iloc[bank_idx]
                match_df = index.create_match_for_players(player1, player2, encoders)
                sequences = create_match_specific_sequences(
                    match_df, index, model.config["columns"], model.config["seq_length"]
                )
                batch = SimpleDSet(sequences, training=False)[0]
                batch = {k: v.unsqueeze(0) for k, v in batch.items()}

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
                player1_win_prob = last_prob[0].item()
                player2_win_prob = last_prob[1].item()

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
