import matplotlib.pyplot as plt

class HistogramDiffResult:
    def __init__(self, data, labels, feature_name, model_path):
        self.data = data
        self.labels = labels
        self.feature_name = feature_name
        self.model_path = model_path
        model_name = model_path.split('/')[-1].split('_')[0]
        self.model_name = model_name

        import joblib
        import os 
        self.model = joblib.load(self.model_path)


    def calculate_():
        pass
    