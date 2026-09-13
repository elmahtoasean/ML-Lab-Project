import joblib
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from preprocessing.data_preprocessing import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
PLOT_DIR = BASE_DIR / "plots"
MODEL_DIR = BASE_DIR / "saved_models"
PLOT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def load_data():
    df = load_dataset()
    X = df[["height_cm", "weight_kg"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, df


def find_best_k(X):
    k_values = [2, 3, 4, 5, 6]
    scores = {}
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score
        print(f"K={k}, Silhouette Score={score:.4f}")

    best_k = max(scores, key=scores.get)
    print("\nBest K:", best_k)

    return best_k


def train_kmeans(X, best_k):
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    model.fit(X)

    return model


def visualize_clusters(X, model):
    labels = model.labels_
    centers = model.cluster_centers_

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=200)
    plt.xlabel("Height (scaled)")
    plt.ylabel("Weight (scaled)")
    plt.title("K-Means Clustering")
    plt.savefig(PLOT_DIR / "kmeans_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_model(model, scaler, best_k):
    metadata = {
        "algorithm": "K-Means",
        "clusters": best_k,
        "features": ["height_cm", "weight_kg"],
        "cluster_centers": scaler.inverse_transform(model.cluster_centers_).tolist(),
    }
    joblib.dump(model, MODEL_DIR / "kmeans_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "kmeans_scaler.pkl")
    joblib.dump(metadata, MODEL_DIR / "kmeans_metadata.pkl")

    print("\nK-Means Model Saved!")


if __name__ == "__main__":
    X, scaler, df = load_data()

    best_k = find_best_k(X)
    model = train_kmeans(X, best_k)

    # Add cluster labels to original dataset
    df["cluster"] = model.labels_

    # Save clustered dataset
    df.to_csv(BASE_DIR / "datasets" / "kmeans_clustered.csv", index=False)

    visualize_clusters(X, model)
    save_model(model, scaler, best_k)
