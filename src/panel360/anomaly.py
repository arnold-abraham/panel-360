"""Isolation-forest based anomaly (alarm) classification on latent vectors."""

from sklearn.ensemble import IsolationForest


def fit_anomaly_detectors(train_latent_df, yellow_contamination=0.1, red_contamination=0.01, random_state=42):
    """Fit two ``IsolationForest`` detectors (looser 'yellow', stricter 'red') on training latent vectors."""
    clf_yellow = IsolationForest(random_state=random_state, contamination=yellow_contamination)
    clf_red = IsolationForest(random_state=random_state, contamination=red_contamination)
    clf_yellow.fit(train_latent_df.values)
    clf_red.fit(train_latent_df.values)
    return clf_yellow, clf_red


def classify_alarms(clf_yellow, clf_red, latent_df):
    """Label each row 'Green'/'Yellow'/'Red' based on the two fitted detectors."""
    result = latent_df.copy()
    preds_yellow = clf_yellow.predict(latent_df.values)
    preds_red = clf_red.predict(latent_df.values)

    result["alarm"] = "Green"
    result.loc[preds_yellow == -1, "alarm"] = "Yellow"
    result.loc[preds_red == -1, "alarm"] = "Red"
    return result
