import joblib
from pathlib import Path

from sklearn.linear_model import Perceptron
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing.fruit_preprocessing import prepare_data

BASE_DIR = Path(__file__).resolve().parent.parent
PLOT_DIR = BASE_DIR / "plots"
MODEL_DIR = BASE_DIR / "saved_models"

PLOT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def load_data():
    return prepare_data()


def find_best_parameters(X_train, y_train, X_val, y_val):
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5, 1]
    epochs = [10, 50, 100, 200]

    results = {}

    for lr in learning_rates:
        for epoch in epochs:
            model = Perceptron(
                eta0=lr,
                max_iter=epoch,
                tol=None,  # force it to actually run `epoch` passes
                shuffle=True,
                random_state=42,
                class_weight=None,
            )
            model.fit(X_train, y_train)

            prediction = model.predict(X_val)
            accuracy = accuracy_score(y_val, prediction)
            results[(lr, epoch)] = accuracy

            print(f"LR={lr}, Epoch={epoch}, Accuracy={accuracy:.4f}")

    best_parameter = max(results, key=results.get)
    print("\nBest Parameters:", best_parameter)

    return best_parameter


def train_perceptron(X_train, y_train, best_parameter):
    lr, epoch = best_parameter

    model = Perceptron(
        eta0=lr,
        max_iter=epoch,
        tol=None,  # don't stop early — with only 4 prototypes it
        # can plateau on training loss before it has
        # actually learned to use the weight feature
        shuffle=True,
        random_state=42,
    )
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, label_map):
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    print("\nTraining Accuracy:")
    print(accuracy_score(y_train, train_prediction) * 100)

    print("\nTesting Accuracy:")
    test_accuracy = accuracy_score(y_test, test_prediction)
    print(test_accuracy * 100)

    class_names = [key for key, value in sorted(label_map.items(), key=lambda x: x[1])]

    print("\nClassification Report")
    print(
        classification_report(
            y_test, test_prediction, target_names=class_names, zero_division=0
        )
    )

    cm = confusion_matrix(y_test, test_prediction, labels=list(label_map.values()))

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Fruit Perceptron Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "fruit_perceptron_confusion_matrix.png")
    plt.close()

    return test_accuracy


def save_model(model, parameters, accuracy, label_map, fruit_names):
    metadata = {
        "algorithm": "Multiclass Perceptron",
        "learning_rate": parameters[0],
        "epochs": parameters[1],
        "features": ["shape_sensor", "texture_sensor", "weight_sensor"],
        "sensor_encoding": {
            "shape": {"round": 1, "elliptical": -1},
            "texture": {"smooth": 1, "rough": -1},
            "weight": {">=1 pound": 1, "<1 pound": -1},
        },
        "label_mapping": label_map,
        "fruit_names": fruit_names,
        "accuracy": accuracy,
    }

    joblib.dump(model, MODEL_DIR / "fruit_perceptron_model.pkl")
    joblib.dump(metadata, MODEL_DIR / "fruit_perceptron_metadata.pkl")

    print("\nFruit Perceptron Model Saved!")


if __name__ == "__main__":
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_names,
        label_map,
        fruit_names,
    ) = load_data()

    best_parameter = find_best_parameters(X_train, y_train, X_val, y_val)
    model = train_perceptron(X_train, y_train, best_parameter)
    accuracy = evaluate_model(model, X_train, y_train, X_test, y_test, label_map)
    save_model(model, best_parameter, accuracy, label_map, fruit_names)
