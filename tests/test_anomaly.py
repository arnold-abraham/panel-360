import numpy as np
import pandas as pd

from panel360.anomaly import classify_alarms, fit_anomaly_detectors


def test_classify_alarms_flags_extreme_outlier_as_not_green():
    rng = np.random.RandomState(0)
    train = pd.DataFrame(rng.normal(loc=0.0, scale=1.0, size=(200, 4)))

    clf_yellow, clf_red = fit_anomaly_detectors(train, yellow_contamination=0.1, red_contamination=0.01)

    normal_point = pd.DataFrame([[0.0, 0.0, 0.0, 0.0]])
    outlier_point = pd.DataFrame([[50.0, 50.0, 50.0, 50.0]])

    normal_result = classify_alarms(clf_yellow, clf_red, normal_point)
    outlier_result = classify_alarms(clf_yellow, clf_red, outlier_point)

    assert normal_result["alarm"].iloc[0] == "Green"
    assert outlier_result["alarm"].iloc[0] == "Red"


def test_classify_alarms_preserves_original_columns():
    train = pd.DataFrame(np.random.RandomState(1).normal(size=(100, 3)))
    clf_yellow, clf_red = fit_anomaly_detectors(train)

    seq = pd.DataFrame(np.zeros((2, 3)), columns=[0, 1, 2])
    result = classify_alarms(clf_yellow, clf_red, seq)

    assert list(result.columns) == [0, 1, 2, "alarm"]
