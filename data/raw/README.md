This project examine data from:   
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data 

## Dataset Overview

The **Credit Card Fraud Detection dataset** is sourced from Kaggle (provided by the Machine Learning Group at ULB). This is a highly imbalanced dataset containing credit card transactions labeled as fraudulent or legitimate.

### Key Characteristics

- **Total Records**: 284,807 transactions
- **Features**: 31 columns (including transaction amount, time, and 28 PCA-transformed features)
- **Class Distribution**: Highly imbalanced
  - Legitimate transactions: ~99.83%
  - Fraudulent transactions: ~0.17%
- **Time Period**: Transactions spanning 2 days
- **Amount Range**: Transaction amounts in a normalized range

### Features

- `Time`: Seconds elapsed between each transaction and the first transaction in the dataset
- `Amount`: Transaction amount
- `V1-V28`: Principal Component Analysis (PCA) transformed features for privacy preservation
- `Class`: Target variable (0 = legitimate, 1 = fraudulent)

### Data Challenges

- **Extreme Class Imbalance**: Only 492 fraudulent cases out of 284,807 transactions (~0.17%)
- **Temporal Component**: Transactions are ordered by time
- **Privacy**: Sensitive features have been transformed using PCA

### Applications

This dataset is commonly used for:
- Anomaly detection and fraud detection modeling
- Evaluating imbalanced classification algorithms
- Testing detection methods like Isolation Forest
- Model robustness and performance evaluation