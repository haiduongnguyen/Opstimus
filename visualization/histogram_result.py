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

    def plot_histogram_data_raw(self, bins=30, imbalance=True, save_path=None):
        """
        Docstring for plot_histogram_data_raw
        
        :param self: 
        :param bins: number of bins in histogram
        :param imbalance: if True, it mean the data is imbalanced, plot anomalies and normal data separately
        :return: plots histogram of the feature for raw data
        """

        if not imbalance:
            plt.figure(figsize=(10, 6))
            plt.hist(self.data[self.feature_name], bins=bins, alpha=0.7, color='blue', label='All Data')
            plt.hist(self.data[self.feature_name][self.labels == 1], bins=bins, alpha=0.7, color='red', label='Anomalies')
            plt.title(f'Histogram of {self.feature_name} - {self.model_name}')
            plt.xlabel(self.feature_name)
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            if save_path:
                plt.savefig(save_path)
            plt.show()
        else:
            ## If imbalanced, plot separately
            plt.figure(figsize=(10, 6))
            plt.hist(self.data[self.feature_name][self.labels == 0], bins=bins, alpha=0.7, color='blue', label='Normal')
            plt.title(f'Histogram of {self.feature_name} - {self.model_name} (Imbalanced)')
            plt.xlabel(self.feature_name)
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            plt.show()
            
            plt.figure(figsize=(10, 6))
            plt.hist(self.data[self.feature_name][self.labels == 1], bins=bins, alpha=0.7, color='red', label='Anomalies')
            plt.title(f'Histogram of {self.feature_name} - {self.model_name} (Imbalanced)')
            plt.xlabel(self.feature_name)
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            if save_path:
                plt.savefig(save_path)
            plt.show()
    
    def plot_histogram_prediction(self, imbalance=True, save_path=None):

        if not imbalance:
            plt.figure(figsize=(10, 6))
            predctions = self.model.predict(self.data)
            # plt.hist(self.data[self.feature_name], bins=50, alpha=0.7, color='blue', label='All Data')
        
            plt.hist(self.data[self.feature_name][predctions == 0], bins=50, alpha=0.7, color='blue', label='Predicted Normal')
            plt.hist(self.data[self.feature_name][predctions == 1], bins=50, alpha=0.7, color='red', label='Predicted Anomalies')
            plt.xlabel(self.feature_name)
            if save_path:
                plt.savefig(save_path)
            plt.show()
        else: 
            plt.figure(figsize=(10, 6))
            predctions = self.model.predict(self.data)        
            plt.hist(self.data[self.feature_name][predctions == 0], bins=50, alpha=0.7, color='blue', label='Predicted Normal')
            plt.xlabel(self.feature_name)
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            if save_path:
                plt.savefig(save_path)
            plt.show()

            plt.figure(figsize=(10, 6))
            predctions = self.model.predict(self.data)        
            plt.hist(self.data[self.feature_name][predctions == 1], bins=50, alpha=0.7, color='red', label='Predicted Anomalies')
            plt.xlabel(self.feature_name)
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            if save_path:
                plt.savefig(save_path)
            plt.show()
        