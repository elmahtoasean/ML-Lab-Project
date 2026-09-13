import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.weather_preprocessing import OUTPUT_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
PLOT_DIR = BASE_DIR / "plots"
MODEL_DIR = BASE_DIR / "saved_models"
PLOT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def load_dataset():
    df = pd.read_csv(OUTPUT_PATH)
    return df


def prepare_data():
    df = load_dataset()

    X = df.drop("Weather_State", axis=1)
    y = df["Weather_State"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    return (X_train, X_val, X_test, y_train, y_val, y_test, X.columns)


def find_best_depth(X_train, y_train, X_val, y_val):
    depths = [2, 3, 5, 7, 10, 15, None]
    results = {}

    for depth in depths:
        model = DecisionTreeClassifier(
            criterion="entropy", max_depth=depth, random_state=42
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_val)
        accuracy = accuracy_score(y_val, prediction)
        results[depth] = accuracy

        print(f"Depth={depth}, Accuracy={accuracy:.4f}")

    best_depth = max(results, key=results.get)
    print("\nBest Depth:", best_depth)
    return best_depth


def train_model(X_train, y_train, best_depth):
    model = DecisionTreeClassifier(
        criterion="entropy", max_depth=best_depth, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_prediction)
    test_accuracy = accuracy_score(y_test, test_prediction)

    print("\nTraining Accuracy:")
    print(train_accuracy * 100, "%")
    print("\nTesting Accuracy:")
    print(test_accuracy * 100, "%")
    print("\nClassification Report")
    print(classification_report(y_test, test_prediction))

    cm = confusion_matrix(y_test, test_prediction)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Rainy", "Sunny", "Windy"],
        yticklabels=["Rainy", "Sunny", "Windy"],
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Weather Decision Tree Confusion Matrix")
    plt.savefig(PLOT_DIR / "weather_decision_tree_confusion_matrix.png")
    plt.close()

    return test_accuracy


def visualize_tree(model, feature_names):
    plt.figure(figsize=(16, 10))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Rainy", "Sunny", "Windy"],
        filled=True,
    )

    plt.savefig(
        PLOT_DIR / "weather_decision_tree_structure.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def save_model(model, accuracy, feature_names):
    metadata = {
        "algorithm": "Weather Decision Tree",
        "criterion": "Entropy",
        "formula": "Information Gain = Entropy(parent) - Entropy(after split)",
        "features": list(feature_names),
        "classes": ["Rainy", "Sunny", "Windy"],
        "accuracy": accuracy,
    }

    joblib.dump(model, MODEL_DIR / "weather_decision_tree.pkl")
    joblib.dump(metadata, MODEL_DIR / "weather_decision_tree_metadata.pkl")
    print("\nWeather Decision Tree Saved!")


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data()

    best_depth = find_best_depth(X_train, y_train, X_val, y_val)
    model = train_model(X_train, y_train, best_depth)
    accuracy = evaluate_model(model, X_train, y_train, X_test, y_test)

    visualize_tree(model, feature_names)
    save_model(model, accuracy, feature_names)
