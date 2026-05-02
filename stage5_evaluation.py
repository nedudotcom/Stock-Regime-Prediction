"""
=============================================================
STAGE 5: MODEL EVALUATION
=============================================================
PURPOSE:
    Load the best saved model and evaluate it on the held-out
    test set that the model has NEVER seen during training.

    Generates the following evaluation artefacts:
        1. Classification Report (Precision, Recall, F1-score)
        2. Confusion Matrix (heatmap)
        3. ROC-AUC Curves (one-vs-rest, per class)
        4. Overall test accuracy

WHY THESE METRICS?
    Accuracy alone can be misleading if classes are imbalanced.
    Precision / Recall / F1 give a clearer picture per class.
    ROC-AUC measures how well the model separates classes at
    every possible decision threshold - ideal for risk systems
    where you may want to tune sensitivity vs specificity.

    Confusion Matrix shows exactly which regimes get confused,
    e.g. whether Medium Risk is often misclassified as High.
=============================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
SEQUENCES_DIR = "data/processed/sequences"
MODELS_DIR    = "models"
RESULTS_DIR   = "results"

MODEL_PATH    = os.path.join(MODELS_DIR, "cnn_lstm_best.keras")
CLASS_NAMES   = ["Low Risk", "Medium Risk", "High Risk"]
N_CLASSES     = 3

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# STEP 1: Load Test Data & Model
# ─────────────────────────────────────────────────────────────
def load_test_data():
    """Load X_test and y_test arrays saved by Stage 3."""
    X_test = np.load(os.path.join(SEQUENCES_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(SEQUENCES_DIR, "y_test.npy"))
    print(f"  X_test : {X_test.shape}  |  y_test : {y_test.shape}")
    return X_test, y_test


def load_model():
    """Load the best trained model from disk."""
    print(f"  Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# ─────────────────────────────────────────────────────────────
# STEP 2: Generate Predictions
# ─────────────────────────────────────────────────────────────
def predict(model, X_test):
    """
    Returns:
        y_pred      : int array of predicted class indices
        y_proba     : float array of class probabilities (N, 3)
    """
    y_proba = model.predict(X_test, verbose=0)   # shape (N, 3)
    y_pred  = np.argmax(y_proba, axis=1)          # shape (N,)
    return y_pred, y_proba


# ─────────────────────────────────────────────────────────────
# STEP 3: Classification Report
# ─────────────────────────────────────────────────────────────
def print_classification_report(y_test, y_pred) -> None:
    """
    Print per-class Precision, Recall, F1-score, and Support.

    Precision  = TP / (TP + FP)  – of all High Risk predictions,
                                    how many were actually High Risk?
    Recall     = TP / (TP + FN)  – of all true High Risk days,
                                    how many did we catch?
    F1-score   = 2 * P * R / (P + R)  – harmonic mean of P and R
    """
    report = classification_report(y_test, y_pred,
                                   target_names=CLASS_NAMES,
                                   digits=4)
    print("\n  Classification Report:")
    print("  " + "-" * 56)
    for line in report.splitlines():
        print("  " + line)
    print("  " + "-" * 56)

    # Save to file
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Classification Report\n")
        f.write("=" * 56 + "\n")
        f.write(report)
    print(f"    Saved to: {report_path}")


# ─────────────────────────────────────────────────────────────
# STEP 4: Confusion Matrix
# ─────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred) -> None:
    """
    Plot and save a normalised confusion matrix heatmap.

    Each cell (i, j) shows the fraction of true class i
    samples that were predicted as class j. Diagonal cells
    should be close to 1.0 for a good model.
    """
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Normalised)"],
        ["d", ".2f"]
    ):
        sns.heatmap(
            data,
            annot=True, fmt=fmt,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cmap="Blues",
            linewidths=0.5,
            ax=ax
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"    Confusion matrix saved to: {cm_path}")


# ─────────────────────────────────────────────────────────────
# STEP 5: ROC-AUC Curves
# ─────────────────────────────────────────────────────────────
def plot_roc_curves(y_test, y_proba) -> None:
    """
    Plot ROC curves for each class using One-vs-Rest strategy.

    ROC-AUC = Area Under the Receiver Operating Characteristic
    Curve. An AUC of 1.0 is perfect; 0.5 is random guessing.
    Values above 0.85 indicate strong discrimination ability.
    """
    # Binarise labels for one-vs-rest comparison
    y_bin = label_binarize(y_test, classes=[0, 1, 2])

    colours = ["steelblue", "darkorange", "tomato"]
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (name, colour) in enumerate(zip(CLASS_NAMES, colours)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, color=colour, lw=2,
                label=f"{name}  (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_title("ROC Curves – One vs Rest", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"    ROC curves saved to: {roc_path}")

    # Print macro-average AUC
    macro_auc = roc_auc_score(y_bin, y_proba, multi_class="ovr",
                               average="macro")
    print(f"    Macro-average ROC-AUC : {macro_auc:.4f}")


# ─────────────────────────────────────────────────────────────
# MAIN EVALUATION PIPELINE
# ─────────────────────────────────────────────────────────────
def evaluate():
    """
    Full Stage 5 pipeline:
        Load test data -> Predict -> Report -> Confusion Matrix
        -> ROC-AUC -> Print summary
    """
    print("=" * 60)
    print("  STAGE 5 : MODEL EVALUATION")
    print("=" * 60)

    X_test, y_test = load_test_data()
    model          = load_model()

    print("\n  Running predictions on test set...")
    y_pred, y_proba = predict(model, X_test)

    # Overall accuracy
    acc = np.mean(y_pred == y_test)
    print(f"\n  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")

    print_classification_report(y_test, y_pred)

    print("\n  Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)

    print("\n  Plotting ROC-AUC curves...")
    plot_roc_curves(y_test, y_proba)

    print("\n" + "=" * 60)
    print("  STAGE 5 COMPLETE")
    print(f"  Evaluation plots saved to: {RESULTS_DIR}/")
    print("  Files:")
    print("    - results/classification_report.txt")
    print("    - results/confusion_matrix.png")
    print("    - results/roc_curves.png")
    print("=" * 60)

    return y_pred, y_proba


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate()
