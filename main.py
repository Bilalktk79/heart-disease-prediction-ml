# =========================================
# ❤️ HEART DISEASE PREDICTION (INDUSTRY LEVEL)
# =========================================

# 📦 IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

# =========================================
# 📥 LOAD DATA
# =========================================
print("📊 Loading dataset...")
df = pd.read_csv("data/heart.csv")

# =========================================
# 🧹 DATA CLEANING
# =========================================
print("\n🔍 Checking missing values...")
print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

print("✅ Data cleaned!")

# =========================================
# 📊 EDA
# =========================================

# Target distribution
sns.countplot(x='target', data=df)
plt.title("Target Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# =========================================
# 🎯 FEATURES
# =========================================
X = df.drop("target", axis=1)
y = df["target"]

# =========================================
# ✂️ SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================
# 🤖 PIPELINE (NO DATA LEAKAGE)
# =========================================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# =========================================
# 🔍 HYPERPARAMETER TUNING
# =========================================
param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l2"]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\n✅ Best Parameters:", grid.best_params_)

# =========================================
# 📏 CROSS VALIDATION SCORE
# =========================================
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
print("📊 Cross-validation ROC-AUC:", np.mean(cv_scores))

# =========================================
# 📈 PREDICTIONS
# =========================================
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# =========================================
# 📊 EVALUATION
# =========================================
print("\n📊 MODEL PERFORMANCE")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# =========================================
# 🔍 FEATURE IMPORTANCE (LOGISTIC)
# =========================================
model = best_model.named_steps["model"]

importance = pd.Series(model.coef_[0], index=X.columns)
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()

# =========================================
# 🌲 DECISION TREE (COMPARISON)
# =========================================
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\n🌲 Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

# =========================================
# 💾 SAVE MODEL
# =========================================
joblib.dump(best_model, "models/heart_model.pkl")

print("\n💾 Model saved successfully!")

# =========================================
# 🔮 SAMPLE PREDICTION
# =========================================
sample = X_test.iloc[0:1]
prediction = best_model.predict(sample)

print("\n📌 Sample Prediction:", "Disease" if prediction[0] == 1 else "No Disease")