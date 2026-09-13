import joblib
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.data_preprocessing import prepare_data

BASE_DIR = Path(__file__).resolve().parent.parent
PLOT_DIR = BASE_DIR / "plots"
MODEL_DIR = BASE_DIR / "saved_models"
PLOT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

def load_data():
    data, scaler = prepare_data()
    (
        X_train,
        X_val,
        X_test,

        y_train,
        y_val,
        y_test,

        feature_names
    ) = data

    return (
        X_train,
        X_val,
        X_test,

        y_train,
        y_val,
        y_test,

        feature_names
    )

def find_best_parameters(
        X_train,
        y_train,
        X_val,
        y_val
):
    
    kernels = [
        "linear",
        "rbf",
        "poly"
    ]

    C_values = [
        0.01,
        0.1,
        1,
        10,
        100
    ]

    results = {}

    for kernel in kernels:
        for C in C_values:

            model = SVC(
                kernel=kernel,
                C=C,
                random_state=42
            )

            model.fit(
                X_train,
                y_train
            )

            prediction = model.predict(
                X_val
            )

            accuracy = accuracy_score(
                y_val,
                prediction
            )

            results[
                (kernel,C)
            ] = accuracy

            print(
                f"Kernel={kernel}, C={C}, Accuracy={accuracy:.4f}"
            )

    best_parameter = max(
        results,
        key=results.get
    )

    print(
        "\nBest Parameters:",
        best_parameter
    )
    return best_parameter


def train_svm(
        X_train,
        y_train,
        best_parameter
):
    kernel, C = best_parameter
    model = SVC(
        kernel=kernel,
        C=C,
        # probability=True,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )
    return model

def evaluate_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test
):
    train_prediction = model.predict(
        X_train
    )
    test_prediction = model.predict(
        X_test
    )
    train_accuracy = accuracy_score(
        y_train,
        train_prediction
    )
    test_accuracy = accuracy_score(
        y_test,
        test_prediction
    )

    print("\nTraining Accuracy:")
    print(
        train_accuracy*100,
        "%"
    )
    print("\nTesting Accuracy:")
    print(
        test_accuracy*100,
        "%"
    )
    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            test_prediction
        )
    )

    cm = confusion_matrix(
        y_test,
        test_prediction
    )

    plt.figure(
        figsize=(6,5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Obese",
            "Fit"
        ],

        yticklabels=[
            "Obese",
            "Fit"
        ]
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.title(
        "SVM Confusion Matrix"
    )

    plt.savefig(
        PLOT_DIR / "svm_confusion_matrix.png"
    )

    plt.close()

    return test_accuracy

def visualize_decision_boundary(
        model,
        X_test,
        y_test
):
    plt.figure(
        figsize=(8,6)
    )

    # Find feature range
    x_min = X_test[:,0].min() - 1
    x_max = X_test[:,0].max() + 1

    y_min = X_test[:,1].min() - 1
    y_max = X_test[:,1].max() + 1

    # Create mesh grid
    xx, yy = np.meshgrid(

        np.linspace(
            x_min,
            x_max,
            300
        ),

        np.linspace(
            y_min,
            y_max,
            300
        )
    )

    # Predict every point in grid
    grid_points = np.c_[
        xx.ravel(),
        yy.ravel()
    ]

    predictions = model.predict(
        grid_points
    )

    predictions = predictions.reshape(
        xx.shape
    )

    # Draw decision regions
    plt.contourf(
        xx,
        yy,
        predictions,
        alpha=0.3
    )

    # Plot actual test points
    plt.scatter(
        X_test[:,0],
        X_test[:,1],
        c=y_test,
        edgecolors="k"
    )

    plt.xlabel(
        "Height (cm)"
    )

    plt.ylabel(
        "Weight (kg)"
    )

    plt.title(
        "SVM Decision Boundary"
    )

    plt.savefig(
        PLOT_DIR / "svm_decision_boundary.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def save_model(
        model,
        parameters,
        accuracy
):
    metadata = {
        "algorithm":
        "SVM",
        "kernel":
        parameters[0],
        "C":
        parameters[1],

        "features":
        [
            "height_cm",
            "weight_kg"
        ],

        "accuracy":
        accuracy

    }
    joblib.dump(
        model,
        MODEL_DIR / "svm_model.pkl"
    )

    joblib.dump(
        metadata,
        MODEL_DIR / "svm_metadata.pkl"
    )

    print(
        "\nSVM model saved!"
    )


if __name__ == "__main__":
    (
        X_train,
        X_val,
        X_test,

        y_train,
        y_val,
        y_test,

        feature_names

    ) = load_data()

    best_parameter = find_best_parameters(
        X_train,
        y_train,
        X_val,
        y_val
    )

    svm_model = train_svm(
        X_train,
        y_train,
        best_parameter
    )

    accuracy = evaluate_model(
        svm_model,
        X_train,
        y_train,
        X_test,
        y_test
    )

    visualize_decision_boundary(
    svm_model,
    X_test,
    y_test
    )

    save_model(
        svm_model,
        best_parameter,
        accuracy
    )