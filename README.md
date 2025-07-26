# Machine Learning Project: Regularized Logistic Regression on Customer Churn

In this machine learning project, we train multiple **logistic regression models** with **L2 regularization** using the **cell2cell churn dataset**. The goal is to predict customer churn while comparing the effects of different values of the regularization hyperparameter **C** on model performance.

## 🎯 Objectives

* Load and explore the cell2cell dataset
* Prepare the data for modeling

  * Handle missing values
  * Identify labels and features
  * Split into training and test sets
* Train logistic regression models using different values of `C` for L2 regularization
* Evaluate model performance using **accuracy** and **log loss**
* Plot and analyze performance trends

## 🧪 Sample Code

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("cell2cell.csv")  # Replace with actual path

# Drop rows with missing labels
label = 'Churn'
df = df.dropna(subset=[label])

# Optional: Fill other missing values with column means
df = df.fillna(df.mean(numeric_only=True))

# One-hot encode categorical features
df = pd.get_dummies(df)

# Split features and target
X = df.drop(columns=[label])
y = df[label]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try different regularization strengths
C_values = [0.01, 0.1, 1, 10, 100]
accuracies = []
log_losses = []

for c in C_values:
    model = LogisticRegression(C=c, penalty='l2', solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)
    
    accuracies.append(acc)
    log_losses.append(loss)

# Plot accuracy and log loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(C_values, accuracies, marker='o', color='green')
plt.xscale('log')
plt.xlabel('C (Regularization Strength)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Regularization')

plt.subplot(1, 2, 2)
plt.plot(C_values, log_losses, marker='o', color='red')
plt.xscale('log')
plt.xlabel('C (Regularization Strength)')
plt.ylabel('Log Loss')
plt.title('Log Loss vs Regularization')

plt.tight_layout()
plt.show()
```

## 📊 Analysis

* **Accuracy vs. C**: Larger values of C reduce the strength of regularization, which may lead to overfitting if too large or underfitting if too small.
* **Log Loss vs. C**: Observing log loss helps us evaluate the confidence of predictions, not just correctness.
* Regularization helps create models that generalize better by preventing extremely large weights.
