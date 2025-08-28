from servebot.inference.predictor import MatchPredictor


def test_basic_prediction():
    predictor = MatchPredictor.load("epoch_4")

    assert len(predictor.training_df) > 0
    assert len(predictor.index.players) > 0

    p1 = "N. Djokovic"
    p2 = "R. Federer"
    prob = predictor.predict(p1, p2)

    assert 0 <= prob <= 1, f"Probability should be between 0 and 1, got {prob}"
    print(f"✅ {p1} vs {p2}: {prob:.3f}")


def test_categories():
    predictor = MatchPredictor.load("epoch_4")

    assert isinstance(predictor.categories, dict)
    assert len(predictor.categories) > 0

    for key, values in predictor.categories.items():
        assert isinstance(values, list)
        print(f"{key}: {len(values)} options")
