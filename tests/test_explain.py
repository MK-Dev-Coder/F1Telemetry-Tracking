"""explain_pit_model.py: SHAP's reasons in % points for fans."""
import numpy as np
import pandas as pd
import pytest

from explain_pit_model import as_points, sigmoid, to_points


def test_the_reasons_add_up_to_the_chance():
    base, values = -2.0, pd.Series({"TyreLife": 1.2, "SafetyCar": 0.5, "Position": -0.4})     # log-odds
    points = to_points(base, values)
    assert points.sum() == pytest.approx(sigmoid(base + values.sum()) - sigmoid(base))
    assert (np.sign(points) == np.sign(values)).all()                          # a reason never changes side
    assert points["TyreLife"] / points["SafetyCar"] == pytest.approx(1.2 / 0.5)  # and keeps its size against the others


def test_no_change():
    points = to_points(-1.0, pd.Series({"A": 0.3, "B": -0.3}))
    assert points.sum() == pytest.approx(0)


def test_labels():
    assert as_points(0.12) == "+12%"
    assert as_points(-0.003) == "-0.3%"
