import joblib
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, log_loss, roc_auc_score

X_train_scaled, X_test_scaled, y_train_resampled, y_test = joblib.load("models/processed_data.pkl")

xgb_model = XGBClassifier(eval_metric="logloss", random_state=42)
xgb_model.fit(X_train_scaled, y_train_resampled)

xgb_model.save_model("models/model.json")

y_pred = xgb_model.predict(X_test_scaled)
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "F1 Score": f1_score(y_test, y_pred),
    "Log Loss": log_loss(y_test, y_pred_proba),
    "AUC-ROC": roc_auc_score(y_test, y_pred_proba),
}

print(metrics)
