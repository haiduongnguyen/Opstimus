# evaluation/experiments.py
def run_experiment(detector, X_train, X_test):
    detector.fit(X_train)
    scores = detector.score(X_test)
    preds = detector.predict(X_test)
    return scores, preds
