import pytest

from servebot.data.preprocess import load_training_dataframe
from servebot.inference.predictor import MatchPredictor
from servebot.models import Match


@pytest.fixture
def predictor():
    return MatchPredictor.load("epoch_4")


@pytest.fixture
def bank():
    return load_training_dataframe()


def test_match_creation(predictor):
    match = Match("Dennis Shapovalov", "Alcaraz").standardize(predictor.encoders)

    assert match.winner_name == "Shapovalov D."
    assert match.loser_name == "Alcaraz C."
    assert match.surface == "Hard"


def test_basic_prediction(predictor, bank):
    p1 = "Novak. Djokovic"
    p2 = "D. Shapovalov"

    more_matches = [
        Match.from_df_record(record) for record in bank.tail(5000).to_dict("records")
    ]
    matches = [
        Match(
            winner_name=p1,
            loser_name=p2,
            winner_rank=1,
            loser_rank=50,
            winner_age=37.4,
            loser_age=25.5,
        )
    ]
    prob = predictor.predict(more_matches + matches)

    assert 0 <= prob <= 1, f"Probability should be between 0 and 1, got {prob}"
    print(f"✅ {p1} vs {p2}: {prob:.3f}")
