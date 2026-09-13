import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import CategoricalHMM
from sklearn.preprocessing import LabelEncoder

# ==============================
# Project Paths
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "datasets" / "hmm_weather_dataset.csv"
MODEL_DIR = BASE_DIR / "saved_models"
PLOT_DIR = BASE_DIR / "plots"
MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


# ==============================
# Load Dataset
# ==============================
def load_dataset():

    df = pd.read_csv(DATASET_PATH)

    print("\nDataset Preview")
    print(df.head())

    print("\nWeather Distribution")
    print(df["Weather_State"].value_counts())

    return df


# ==============================
# Encode Weather States
# ==============================
def encode_weather(df):

    encoder = LabelEncoder()
    df = df.copy()
    df["Weather_State"] = encoder.fit_transform(df["Weather_State"])

    return df, encoder


# ==============================
# Prepare HMM Input
# ==============================
def prepare_data():

    df = load_dataset()
    df, encoder = encode_weather(df)

    sequence = df["Weather_State"].values
    X = sequence.reshape(-1, 1)

    return X, sequence, encoder


# ==============================
# Train HMM
# ==============================
def train_hmm(X):

    model = CategoricalHMM(n_components=3, n_iter=500, tol=0.0001, random_state=42)
    model.fit(X)

    return model


# ==============================
# Display Model Information
# ==============================
def show_model_information(model, encoder):

    print("\nHidden State Transition Matrix")
    print(model.transmat_)

    print("\nEmission Matrix")
    print(model.emissionprob_)

    print("\nWeather Classes")
    print(encoder.classes_)

# ==============================
# Plot Transition Matrix
# ==============================
def plot_transition_matrix(model):

    plt.figure(figsize=(7, 5))
    sns.heatmap(model.transmat_, annot=True, fmt=".3f", cmap="Blues")

    plt.xlabel("Next Hidden State")
    plt.ylabel("Current Hidden State")
    plt.title("HMM Transition Matrix")
    plt.savefig(PLOT_DIR / "hmm_transition_matrix.png", dpi=300, bbox_inches="tight")

    plt.close()

# ==============================
# Predict Tomorrow Weather
# ==============================
def predict_next_weather(model, encoder, previous_days):
    """
    Input:

    [
      Sunny,
      Sunny,
      Rainy
    ]
    """

    encoded = encoder.transform(previous_days)
    observations = encoded.reshape(-1, 1)

    # Forward algorithm
    _, hidden_probability = model.score_samples(observations)
    current_hidden = hidden_probability[-1]

    # Tomorrow hidden state probability
    next_hidden = current_hidden @ model.transmat_
    # Convert hidden state probability
    # into weather probability

    weather_probability = next_hidden @ model.emissionprob_
    prediction_index = np.argmax(weather_probability)
    prediction = encoder.inverse_transform([prediction_index])[0]
    confidence = weather_probability[prediction_index] * 100

    return prediction, confidence

# ==============================
# Evaluate Prediction
# ==============================
def evaluate_model(model, encoder, sequence):

    correct = 0
    total = 0

    for i in range(len(sequence) - 3):
        previous = encoder.inverse_transform(sequence[i : i + 3])
        actual = encoder.inverse_transform([sequence[i + 3]])[0]

        predicted, confidence = predict_next_weather(model, encoder, list(previous))

        if predicted == actual:
            correct += 1
        total += 1

    accuracy = (correct / total) * 100
    print("\nPrediction Accuracy:", round(accuracy, 2), "%")

# ==============================
# Save Model
# ==============================
def save_model(model, encoder):

    joblib.dump(model, MODEL_DIR / "weather_hmm.pkl")
    joblib.dump(encoder, MODEL_DIR / "weather_hmm_encoder.pkl")
    print("\nModel Saved Successfully!")

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    X, sequence, encoder = prepare_data()
    print("\nTraining HMM...")

    model = train_hmm(X)
    show_model_information(model, encoder)
    plot_transition_matrix(model)
    evaluate_model(model, encoder, sequence)

    # Manual Test
    print("\nManual Prediction")
    previous_days = ["Sunny", "Sunny", "Rainy"]
    prediction, confidence = predict_next_weather(model, encoder, previous_days)

    print("Input:", previous_days)
    print("Tomorrow:", prediction)
    print("Confidence:", round(confidence, 2), "%")
    save_model(model, encoder)
