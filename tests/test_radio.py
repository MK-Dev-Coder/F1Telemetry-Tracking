"""game_radio.py: which topic a question to the race engineer is about."""
import pytest

from game_radio import topic_of


@pytest.mark.parametrize("question, topic", [
    ("Can I win this?", "chances"),
    ("Where will I finish?", "chances"),
    ("What are my chances?", "chances"),
    ("Chance of a podium?", "chances"),
    ("How many points am I scoring?", "chances"),
    ("What's the chance of rain?", "weather"),                 # "chance" is about the rain here
    ("What are the chances it rains?", "weather"),
    ("How many laps to the finish?", "laps"),                  # and "finish" about the laps
    ("How long to the finish?", "laps"),
    ("What's my pit chance?", "pit"),
    ("Box this lap?", "pit"),
    ("How many laps left on these tyres?", "tyres"),
    ("What's the gap behind?", "behind"),
    ("Radio check", "radio_check"),
])
def test_the_question_goes_to_the_right_topic(question, topic):
    assert topic_of(question) == topic
