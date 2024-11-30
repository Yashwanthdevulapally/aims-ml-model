import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Generate synthetic data
X, y = make_classification(
    n_samples=1000,
    n_features=3,  # Features: Age, BMI, Glucose
    n_informative=3,
    n_redundant=0,
    random_state=42,
    class_sep=1.5
)

# Convert to DataFrame
df = pd.DataFrame(X, columns=["Age", "BMI", "Glucose"])
df['Diabetes'] = y
df.to_csv('diabetes_dataset.csv', index=False)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(df[["Age", "BMI", "Glucose"]], df["Diabetes"], test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate and save the model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model trained. Accuracy: {accuracy:.2f}")
joblib.dump(model, 'diabetes_model.pkl')
print("Trained model saved as 'diabetes_model.pkl'")
