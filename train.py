# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Load preprocessed data
# X_train, X_test, y_train, y_test, scaler = joblib.load('data.pkl')

# # Train the model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate the model
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# print(f"Model Accuracy: {accuracy:.4f}")

# # Save the trained model
# joblib.dump(model, 'liver_model.pkl')

# print("Model training complete. Model saved.")


import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load preprocessed data
X_train, X_test, y_train, y_test, scaler = joblib.load('data.pkl')

# Train the model
n_estimators = 100
model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
joblib.dump(model, 'tests/liver_model.pkl')

print("Model training complete. Model saved.")

print(f"Model Accuracy: {accuracy:.4f}")

# ---------------- MLflow Integration ----------------
# Point MLflow to your local tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("LiverDiseasePrediction")

with mlflow.start_run():
    # Log parameters and metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model in MLflow and register it
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="LiverModel"
    )

print("✅ Model training complete and logged to MLflow.")
