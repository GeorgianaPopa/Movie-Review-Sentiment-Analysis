import os
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


SPLIT_DIR = "data/splits"
MODEL_DIR = "models"


def load_splits(split_dir: str = SPLIT_DIR):
    """Load train/test splits saved earlier by build_features.py"""
    X_train = sparse.load_npz(os.path.join(split_dir, "X_train.npz"))
    X_test = sparse.load_npz(os.path.join(split_dir, "X_test.npz"))
    y_train = pd.read_csv(os.path.join(split_dir, "y_train.csv"))["label"].values
    y_test = pd.read_csv(os.path.join(split_dir, "y_test.csv"))["label"].values

    return X_train, X_test, y_train, y_test


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_path: str):
    """Train a model, evaluate it, save it."""
    print(f"\nTraining {model.__class__.__name__}...")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved model → {model_path}\n")


def train_all_models():
    print("Loading data splits...")
    X_train, X_test, y_train, y_test = load_splits()

    models_to_train = {
        "logreg_model.joblib": LogisticRegression(max_iter=3000),
        "svm_model.joblib": LinearSVC(),
        "nb_model.joblib": MultinomialNB(),
    }

    for filename, model in models_to_train.items():
        model_path = os.path.join(MODEL_DIR, filename)
        train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_path)


if __name__ == "__main__":
    print("=== TRAINING MODELS ===")
    train_all_models()
