
import pandas as pd 
import numpy as np 
import os
import sys
sys.path.append('../..')

from detection.isolation_forest import IsolationForestDetector
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from preprocessing.load_data import DataLoader


card_data = pd.read_csv('../../data/raw/creditcard.csv')

print("Card data shape: ", card_data.shape) 

print(card_data.head(2))


data_use = DataLoader(data=card_data, label_col='Class')

data_use.remove_columns(['Time'])

training, label = data_use.get_features(), data_use.get_labels()

print("Training data shape: ", training.shape) 

print(training.head(2))

print("label data shape: ", label.shape) 

print(label.head(2))

X_train, X_test, y_train, y_test = data_use.train_test_split(test_size=0.3, random_state=42)

print("X_train shape: ", X_train.shape) 
print("X_test shape: ", X_test.shape)

print("Sample X_train data: ", X_train.head(2))

print("Start training Isolation Forest model...")
isolation_forest = IsolationForestDetector(n_estimators=100, contamination=0.01, random_state=42)

isolation_forest.fit(X_train)

print("Model training completed.")

print("Scoring test data...")

isolation_scores = isolation_forest.score(X_test)

isolation_predictions = isolation_forest.predict(X_test)

print("Isolation Forest Scores: ", isolation_scores[:5])
print("Isolation Forest Predictions: ", isolation_predictions[:5])



print("Evaluating model performance...")

if y_test is not None:
    print("Number of anomalies detected: ", np.sum(isolation_predictions))
    print("Percentage of anomalies detected: ", np.mean(isolation_predictions) * 100, "%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, isolation_predictions.astype(int)))
    print("\nClassification Report:")
    print(classification_report(y_test, isolation_predictions.astype(int)))

    print("AUC-ROC of isolation forest: ", roc_auc_score(y_test, isolation_scores))

    if not os.path.exists('../checkpoint/'):
        os.makedirs('../checkpoint/')

        isolation_forest.save_model('../checkpoint/isolation_forest_model.pkl')

        classification_report_df = classification_report(y_test, isolation_predictions.astype(int), output_dict=True)

        report_df = pd.DataFrame(classification_report_df).transpose()
        report_df.to_csv('../checkpoint/isolation_forest_classification_baseline_report.csv', index=True)

else:
    print("No labels available for evaluation.")
    print("Number of detected anomalies: ", np.sum(isolation_predictions))

