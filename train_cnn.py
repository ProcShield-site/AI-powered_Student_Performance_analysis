# ==========================================================
# TRAIN CNN — using the SAME preprocessing, split and scaler
# as the Random Forest / KNN / SVM models, so the comparison
# is fair.
#
# BEFORE RUNNING: upload these 5 files into this Colab session
# (left sidebar -> Files -> upload):
#   student_performance_dataset.csv
#   scaler.pkl
#   label_encoders.pkl
#   meta.pkl
#   Model_Comparison.csv
# ==========================================================

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

sns.set_style("whitegrid")

# ------------------------------------------------------------
# 1. Load the SAME scaler / encoders / column info already
#    used for Random Forest, KNN and SVM
# ------------------------------------------------------------
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
meta = joblib.load("meta.pkl")

feature_columns = meta["feature_columns"]
class_names = np.array(meta["class_names"])

# ------------------------------------------------------------
# 2. Rebuild X, y EXACTLY the same way (same fillna, same
#    encoding, same columns, same order)
# ------------------------------------------------------------
df = pd.read_csv("student_performance_dataset.csv")
df["parental_education"] = df["parental_education"].fillna("High School")

cat_cols = ["gender", "parental_education", "internet_access",
            "extracurricular_activities", "part_time_job", "final_grade"]

for col in cat_cols:
    df[col] = label_encoders[col].transform(df[col])

X = df.drop(["student_id", "final_grade"], axis=1)[feature_columns]
y = df["final_grade"]

# ------------------------------------------------------------
# 3. Same split (same random_state -> identical train/test rows
#    as the other 3 models)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

X_train_cnn = X_train_s.reshape(X_train_s.shape[0], X_train_s.shape[1], 1)
X_test_cnn = X_test_s.reshape(X_test_s.shape[0], X_test_s.shape[1], 1)

# ------------------------------------------------------------
# 4. Build CNN — same architecture as your original code
# ------------------------------------------------------------
model = Sequential([
    Conv1D(filters=32, kernel_size=2, activation="relu",
           input_shape=(X_train_s.shape[1], 1)),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(len(class_names), activation="softmax"),
])

model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    X_train_cnn, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=16,
    verbose=1,
)

# ------------------------------------------------------------
# 5. Evaluate
# ------------------------------------------------------------
loss, cnn_accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)

cnn_prob = model.predict(X_test_cnn)
cnn_pred = np.argmax(cnn_prob, axis=1)

cnn_precision = precision_score(y_test, cnn_pred, average="weighted", zero_division=0)
cnn_recall = recall_score(y_test, cnn_pred, average="weighted", zero_division=0)
cnn_f1 = f1_score(y_test, cnn_pred, average="weighted", zero_division=0)

print("=" * 60)
print("CNN RESULTS")
print("=" * 60)
print(f"Accuracy  : {cnn_accuracy:.4f}")
print(f"Precision : {cnn_precision:.4f}")
print(f"Recall    : {cnn_recall:.4f}")
print(f"F1 Score  : {cnn_f1:.4f}")
print()
print(classification_report(y_test, cnn_pred, target_names=class_names))

# ------------------------------------------------------------
# 6. Confusion matrix (colorful)
# ------------------------------------------------------------
cm = confusion_matrix(y_test, cnn_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor="black")
plt.title("CNN Confusion Matrix", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade")
plt.tight_layout()
plt.savefig("CNN_Confusion_Matrix.png", dpi=300, facecolor="white")
plt.show()

# ------------------------------------------------------------
# 7. Training curves (colorful)
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(history.history["accuracy"], label="Training", linewidth=2, color="#2ecc71")
axes[0].plot(history.history["val_accuracy"], label="Validation", linewidth=2, color="#e74c3c")
axes[0].set_title("CNN Accuracy over Epochs", fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(history.history["loss"], label="Training", linewidth=2, color="#3498db")
axes[1].plot(history.history["val_loss"], label="Validation", linewidth=2, color="#f39c12")
axes[1].set_title("CNN Loss over Epochs", fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("CNN_Training_Curves.png", dpi=300, facecolor="white")
plt.show()

# ------------------------------------------------------------
# 8. Save the CNN model
# ------------------------------------------------------------
model.save("cnn_model.h5")
print("\nSaved cnn_model.h5")

# ------------------------------------------------------------
# 9. Merge into Model_Comparison.csv (adds/updates the CNN row)
# ------------------------------------------------------------
comp = pd.read_csv("Model_Comparison.csv")
comp = comp[comp["Model"] != "CNN"]  # remove any old CNN row
comp = pd.concat([comp, pd.DataFrame([{
    "Model": "CNN",
    "Accuracy": cnn_accuracy,
    "Precision": cnn_precision,
    "Recall": cnn_recall,
    "F1 Score": cnn_f1,
}])], ignore_index=True)

comp.to_csv("Model_Comparison.csv", index=False)
print("\nUpdated Model_Comparison.csv:")
print(comp)

# ------------------------------------------------------------
# 10. Redraw the 4-model grouped comparison chart
# ------------------------------------------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(comp))
width = 0.2
for i, metric in enumerate(metrics):
    ax.bar(x + i * width, comp[metric] * 100, width, label=metric, color=colors[i])
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(comp["Model"], fontsize=11)
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 110)
ax.set_title("Model Performance Comparison (4 Models)", fontsize=15, fontweight="bold")
ax.legend(loc="upper right", ncol=4, frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("Model_Comparison_Grouped.png", dpi=300, facecolor="white")
plt.show()

print("\nDONE. Download these files and add them to your student_app folder:")
print(" - cnn_model.h5")
print(" - Model_Comparison.csv        (overwrite the old one)")
print(" - Model_Comparison_Grouped.png (overwrite the old one)")
print(" - CNN_Confusion_Matrix.png")
print(" - CNN_Training_Curves.png")
