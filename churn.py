# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 2. Load Dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# 3. Drop CustomerID column
df.drop("customerID", axis=1, inplace=True)

# 4. Convert TotalCharges to numeric (handle errors)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 5. Drop rows with missing TotalCharges
df.dropna(inplace=True)

# 6. Encode Binary Columns
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn', 'gender']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

# 7. One-Hot Encode Categorical Columns
df = pd.get_dummies(df, drop_first=True)

# 8. Define Features and Target
X = df.drop('Churn', axis=1)
y = df['Churn']

# 9. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 11. Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 12. Predict
y_pred = model.predict(X_test)

# 13. Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 14. Feature Importance Plot
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.nlargest(10)

plt.figure(figsize=(10, 6))
top_features.plot(kind='barh')
plt.title('Top 10 Important Features for Churn Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
