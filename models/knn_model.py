import joblib
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.data_preprocessing import prepare_data

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
PLOT_DIR = BASE_DIR / "plots"
MODEL_DIR = BASE_DIR / "saved_models"

PLOT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# Load processed data
def load_data():
    data, scaler = prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = data

    return (X_train, X_val, X_test, y_train, y_val, y_test)


# Find best K value
def find_best_k(X_train, y_train, X_val, y_val):
    k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    results = {}
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        prediction = model.predict(X_val)
        accuracy = accuracy_score(y_val, prediction)
        results[k] = accuracy
        print(f"K={k} Validation Accuracy={accuracy:.4f}")

    best_k = max(results, key=results.get)

    print("\nBest K:", best_k)

    return best_k


# Train final KNN model
def train_knn(X_train, y_train, best_k):

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)
    return model


# Evaluate model
def evaluate_model(model, X_train, y_train, X_test, y_test):

    # Training accuracy
    train_prediction = model.predict(X_train)

    train_accuracy = accuracy_score(y_train, train_prediction)

    print("\nTraining Accuracy:")
    print(train_accuracy * 100, "%")

    # Testing accuracy
    predictions = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, predictions)
    print("\nKNN Test Accuracy:")
    print(test_accuracy * 100, "%")
    print("\nClassification Report")

    print(classification_report(y_test, predictions))

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Obese", "Fit"],
        yticklabels=["Obese", "Fit"],
    )
    plt.xlabel("Predicted")

    plt.ylabel("Actual")
    plt.title("KNN Confusion Matrix")
    plt.savefig(PLOT_DIR / "knn_confusion_matrix.png")
    plt.close()

    return test_accuracy


# Save model
def save_model(model, best_k, test_accuracy):
    metadata = {
        "algorithm": "KNN",
        "best_k": best_k,
        "features": ["height_cm", "weight_kg"],
        "accuracy": test_accuracy,
    }

    joblib.dump(metadata, MODEL_DIR / "knn_metadata.pkl")
    joblib.dump(model, MODEL_DIR / "knn_model.pkl")

    print("\nKNN model saved!")


# Main
if __name__ == "__main__":

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    best_k = find_best_k(X_train, y_train, X_val, y_val)
    knn_model = train_knn(X_train, y_train, best_k)

    test_accuracy = evaluate_model(knn_model, X_train, y_train, X_test, y_test)
    save_model(knn_model, best_k, test_accuracy)
