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
        scores = self.score(X)
        if threshold is None:
            threshold = scores.mean() + 3 * scores.std()
        return scores > threshold
<<<<<<< HEAD

    def save_model(self, filepath):
        import joblib
        joblib.dump(self.model, filepath)
=======
>>>>>>> e6d36e26a2611e5e76317d60e89ad5f63b686c42
