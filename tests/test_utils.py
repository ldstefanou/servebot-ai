import pytest

from servebot.data.utils import find_most_similar_string


@pytest.mark.parametrize(
    "target,expected",
    [
        ("Roger Federer", "Roger Federer"),
        ("roger federer", "Roger Federer"),
        ("Federer", "Roger Federer"),
        ("Rafael", "Rafael Nadal"),
        ("Djokovic", "Novak Djokovic"),
        ("Andy", "Andy Murray"),
        ("Serena", "Serena Williams"),
    ],
)
def test_find_most_similar_player(target, expected):
    players = [
        "Roger Federer",
        "Rafael Nadal",
        "Novak Djokovic",
        "Andy Murray",
        "Serena Williams",
        "Maria Sharapova",
        "Venus Williams",
        "Steffi Graf",
        "Martina Navratilova",
        "Pete Sampras",
    ]

    assert find_most_similar_string(players, target) == expected
