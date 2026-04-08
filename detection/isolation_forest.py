# models/isolation_forest.py
from sklearn.ensemble import IsolationForest
from detection.detector_base import BaseDetector

class IsolationForestDetector(BaseDetector):

    def __init__(self, **kwargs):
        self.model = IsolationForest(**kwargs)

    def fit(self, X):
        self.model.fit(X)

    def score(self, X):
        return -self.model.decision_function(X)

    def predict(self, X, threshold=None):
        return self.model.predict(X) == -1  # Return True for anomalies

    def save_model(self, file_path):
        import joblib
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        import joblib
        self.model = joblib.load(file_path)
        return self.model
