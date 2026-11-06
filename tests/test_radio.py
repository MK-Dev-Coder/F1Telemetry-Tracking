"""Tests for game_radio.py: which topic a question to the engineer is about, and the engineer's own calls."""
import threading

import pytest

import game_radio
import game_telemetry as gt
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


def test_the_radio_after_a_flashback():
    """A flashback removes the events after that moment, but what happens next is still called on the radio."""
    s, radio, lock = gt.Session(1, 2026), game_radio.Radio(), threading.RLock()
    s.events = [{"t": float(k), "code": "OVTK", "text": "A passes B"} for k in range(5)]
    assert radio.calls_for(s, lock) == []
    s.on_flashback(1.5)                                                      # back to before the third pass
    s.events.append({"t": 2.0, "code": "RDFL", "text": "Red flag"})          # then a red flag
    assert [c["text"] for c in radio.calls_for(s, lock)] == ["Red flag, red flag. Into the pits."]
