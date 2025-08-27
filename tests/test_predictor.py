import pytest

from servebot.inference.predictor import MatchPredictor


def test_basic_prediction():
    predictor = MatchPredictor.load("epoch_4")

    assert len(predictor.training_df) > 0
    assert len(predictor.index.players) > 0

    # Test with common players
    test_players = [
        p for p in predictor.index.players if "Djokovic" in p or "Shapovalov" in p
    ][:2]

    if len(test_players) >= 2:
        p1, p2 = test_players[0], test_players[1]
        prob = predictor.predict(p1, p2)

        assert 0 <= prob <= 1, f"Probability should be between 0 and 1, got {prob}"
        print(f"✅ {p1} vs {p2}: {prob:.3f}")
    else:
        pytest.skip("Not enough test players found")


def test_categories():
    predictor = MatchPredictor.load("epoch_4")

    assert isinstance(predictor.categories, dict)
    assert len(predictor.categories) > 0

    for key, values in predictor.categories.items():
        assert isinstance(values, list)
        print(f"{key}: {len(values)} options")
