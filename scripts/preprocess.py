import os
import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

os.makedirs("models", exist_ok=True)

df = pd.read_csv(r"C:\Users\kcarm\bankruptcy-prediction\data\Bdataset.csv")

df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(r"[^\w]", "", regex=True)

X = df.drop(columns=["Bankrupt"])
y = df["Bankrupt"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

joblib.dump((X_train_scaled, X_test_scaled, y_train_resampled, y_test), "models/processed_data.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(sklearn.__version__, "models/sklearn_version.pkl")

scaler = joblib.load("models/scaler.pkl")

print("Expected number of features:", scaler.n_features_in_)