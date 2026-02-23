# models/isolation_forest.py
from sklearn.neighbors import LocalOutlierFactor

## docs here: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
from detection.detector_base import BaseDetector

class LocalOutlierFactorDetector(BaseDetector):

    def __init__(self, **kwargs):
        self.model = LocalOutlierFactor(**kwargs)

    def fit(self, X):
        self.model.fit(X)

    def score(self, X):
        return self.model.score_samples(X)
    def get_offset(self):
        return self.model.offset_

    def predict(self, X, threshold=None):
        return self.model.predict(X) == -1  # Return True for anomalies

    def save_model(self, filepath):
        import joblib
        joblib.dump(self.model, filepath)
