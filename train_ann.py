import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_excel("dataset.xlsx")

print("First 5 rows:")
print(df.head())
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Target distribution:\n", df['result'].value_counts())

X = df[["attendance", "assignment", "quiz", "mid", "study_hours"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
)
model.fit(X_train_scaled, y_train)

print("Training complete! Iterations:", model.n_iter_)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

joblib.dump(model,  "model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("Model and scaler saved!")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail","Pass"],
            yticklabels=["Fail","Pass"], ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

axes[1].plot(model.loss_curve_, color="#2563EB", linewidth=2)
axes[1].set_title("Training Loss Curve")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_output.png", dpi=150)
print("Plot saved as training_output.png")