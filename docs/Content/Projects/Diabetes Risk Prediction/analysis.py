"""
Diabetes Risk Prediction - Full ML Pipeline
Author: Oluwaseun Daniel Fowotade
Description: Predicts diabetes risk using multiple ML classifiers on the Pima Indians
             Diabetes dataset, with model comparison, feature importance, and ROC analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
os_path = "images"

import os
os.makedirs(os_path, exist_ok=True)

# ── 1. Reproducible synthetic dataset with Pima-like statistics ──────────────
np.random.seed(42)
n = 768

glucose     = np.clip(np.random.normal(120.9, 31.97, n), 0, 199)
bmi         = np.clip(np.random.normal(31.99, 7.88,  n), 0, 67.1)
age         = np.clip(np.random.normal(33.24, 11.76, n), 21, 81).astype(int)
pregnancies = np.clip(np.random.poisson(3.85, n), 0, 17)
bp          = np.clip(np.random.normal(69.1,  19.36, n), 0, 122)
skin        = np.clip(np.random.normal(20.5,  15.95, n), 0, 99)
insulin     = np.clip(np.random.exponential(79.8, n), 0, 846)
dpf         = np.clip(np.random.exponential(0.47, n), 0.078, 2.42)

logit = (-9.8
         + 0.036 * glucose
         + 0.082 * bmi
         + 0.028 * age
         + 0.145 * pregnancies
         + 1.20  * dpf
         + np.random.normal(0, 0.6, n))
prob    = 1 / (1 + np.exp(-logit))
outcome = (prob > 0.50).astype(int)

df = pd.DataFrame({
    "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
    "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
    "DiabetesPedigreeFunction": dpf, "Age": age, "Outcome": outcome
})

print(f"Dataset shape: {df.shape}")
print(f"Diabetes prevalence: {df['Outcome'].mean():.1%}")

# ── 2. EDA Figure ─────────────────────────────────────────────────────────────
features = ["Glucose", "BMI", "Age", "BloodPressure", "DiabetesPedigreeFunction", "Pregnancies"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, feat in enumerate(features):
    for val, color, label in zip([0, 1], PALETTE[:2], ["No Diabetes", "Diabetes"]):
        axes[i].hist(df[df["Outcome"] == val][feat], bins=25, alpha=0.6,
                     color=color, label=label, edgecolor="white")
    axes[i].set_title(feat, fontsize=13, fontweight="bold")
    axes[i].set_xlabel("Value", fontsize=10)
    axes[i].set_ylabel("Count", fontsize=10)
    axes[i].legend(fontsize=9)

fig.suptitle("Feature Distributions by Diabetes Status", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{os_path}/eda_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_distributions.png")

# ── 3. Correlation Heatmap ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 9})
ax.set_title("Feature Correlation Matrix", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{os_path}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: correlation_heatmap.png")

# ── 4. Model Training & Evaluation ───────────────────────────────────────────
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)":            SVC(probability=True, kernel="rbf", random_state=42),
}

results = {}
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    X_fit = X_train_s if name in ["Logistic Regression", "SVM (RBF)"] else X_train
    X_ev  = X_test_s  if name in ["Logistic Regression", "SVM (RBF)"] else X_test
    X_cv  = X_train_s if name in ["Logistic Regression", "SVM (RBF)"] else X_train.values

    model.fit(X_fit, y_train)
    preds = model.predict(X_ev)
    proba = model.predict_proba(X_ev)[:, 1]

    cv_scores = cross_val_score(model, X_cv, y_train, cv=cv, scoring="accuracy")

    results[name] = {
        "model":    model,
        "accuracy": accuracy_score(y_test, preds),
        "f1":       f1_score(y_test, preds),
        "auc":      roc_auc_score(y_test, proba),
        "cv_mean":  cv_scores.mean(),
        "cv_std":   cv_scores.std(),
        "proba":    proba,
        "preds":    preds,
    }
    print(f"{name:22s} | Acc={results[name]['accuracy']:.3f} | "
          f"F1={results[name]['f1']:.3f} | AUC={results[name]['auc']:.3f}")

# ── 5. Model Comparison Bar Chart ────────────────────────────────────────────
metrics_df = pd.DataFrame({
    name: {"Accuracy": r["accuracy"], "F1 Score": r["f1"], "AUC-ROC": r["auc"]}
    for name, r in results.items()
}).T

fig, ax = plt.subplots(figsize=(11, 5))
x    = np.arange(len(metrics_df))
w    = 0.25
bars = ["Accuracy", "F1 Score", "AUC-ROC"]
for i, (metric, color) in enumerate(zip(bars, PALETTE)):
    rects = ax.bar(x + i * w, metrics_df[metric], w, label=metric,
                   color=color, alpha=0.85, edgecolor="white")
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.005,
                f"{rect.get_height():.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_xticks(x + w)
ax.set_xticklabels(metrics_df.index, fontsize=11)
ax.set_ylim(0.5, 1.02)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{os_path}/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: model_comparison.png")

# ── 6. ROC Curves ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for (name, r), color in zip(results.items(), PALETTE):
    fpr, tpr, _ = roc_curve(y_test, r["proba"])
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name} (AUC = {r['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=15, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
plt.tight_layout()
plt.savefig(f"{os_path}/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: roc_curves.png")

# ── 7. Feature Importance (Random Forest) ────────────────────────────────────
rf_model   = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
colors_fi = [PALETTE[0] if v < importances.max() else PALETTE[1] for v in importances]
bars = ax.barh(importances.index, importances.values, color=colors_fi,
               edgecolor="white", height=0.65)
for bar in bars:
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}", va="center", fontsize=10)
ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
ax.set_title("Random Forest Feature Importance", fontsize=14, fontweight="bold")
ax.set_xlim(0, importances.max() + 0.05)
plt.tight_layout()
plt.savefig(f"{os_path}/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: feature_importance.png")

# ── 8. Confusion Matrix (Best Model = Gradient Boosting) ─────────────────────
best_name = max(results, key=lambda k: results[k]["auc"])
cm = confusion_matrix(y_test, results[best_name]["preds"])

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"],
            linewidths=1, linecolor="white", annot_kws={"size": 14})
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{os_path}/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: confusion_matrix.png")

# ── 9. CV Score Distribution ──────────────────────────────────────────────────
cv_data = []
for name, model in models.items():
    X_fit = X_train_s if name in ["Logistic Regression", "SVM (RBF)"] else X_train.values
    scores = cross_val_score(model, X_fit, y_train, cv=cv, scoring="accuracy")
    for s in scores:
        cv_data.append({"Model": name, "CV Accuracy": s})

cv_df = pd.DataFrame(cv_data)
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=cv_df, x="Model", y="CV Accuracy", palette=PALETTE, ax=ax,
            width=0.5, linewidth=1.5)
sns.stripplot(data=cv_df, x="Model", y="CV Accuracy", color="black",
              size=5, alpha=0.5, ax=ax)
ax.set_title("5-Fold Cross-Validation Accuracy by Model", fontsize=14, fontweight="bold")
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_xlabel("")
ax.set_ylim(0.55, 1.0)
plt.xticks(fontsize=10)
plt.tight_layout()
plt.savefig(f"{os_path}/cv_scores.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: cv_scores.png")

print("\nAll figures saved to images/")
print(f"\nBest model: {best_name} (AUC = {results[best_name]['auc']:.3f})")
