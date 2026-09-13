import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define Dataset Path
DATASET_PATH = "D:/Study/8th Semester/ML Lab Project/datasets/obesity_dataset.csv"


# Load Dataset
def load_dataset():

    df = pd.read_csv(DATASET_PATH)

    # Keep only required columns
    df = df[["height_cm", "weight_kg", "label"]]

    return df


# Check Dataset Information
def dataset_info(df):

    print("\nDataset Information")
    print("-------------------")

    print("Samples:", len(df))

    print("\nColumns:")
    print(df.columns)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nClass distribution:")
    print(df["label"].value_counts())


# Separate Input and Output
def split_features(df):

    X = df.drop("label", axis=1)
    y = df["label"]

    return X, y


# Split Data into Training, Validation, and Testing Sets
def split_data(X, y):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    return (X_train, X_val, X_test, y_train, y_val, y_test)


# ==========================================
# Scaled Data Pipeline
# For KNN, SVM, K-Means, Perceptron
# ==========================================
def prepare_data():

    df = load_dataset()
    X, y = split_features(df)
    feature_names = X.columns

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Scaling only using training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, X_val, X_test, y_train, y_val, y_test, feature_names), scaler


# ==========================================
# Unscaled Data Pipeline
# For Decision Tree (ETA AR LAGBE NA)
# ==========================================
def prepare_tree_data():

    df = load_dataset()
    X, y = split_features(df)
    feature_names = X.columns
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    return (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)


# Testing
if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
    dataset_info(df)

    # Test scaled pipeline
    print("\n========== Scaled Data ==========")
    data, scaler = prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = data

    print("\nScaled Dataset Split")
    print("---------------------")
    print("Training:", len(X_train))
    print("Validation:", len(X_val))
    print("Testing:", len(X_test))
    print("\nFeatures:")
    print(list(feature_names))

    joblib.dump(scaler, "D:/Study/8th Semester/ML Lab Project/saved_models/scaler.pkl")
    print("\nScaler saved successfully!")

    # Test Decision Tree pipeline
    print("\n========== Tree Data ==========")
    tree_data = prepare_tree_data()
    (
        X_train_tree,
        X_val_tree,
        X_test_tree,
        y_train_tree,
        y_val_tree,
        y_test_tree,
        tree_features,
    ) = tree_data

    print("\nTree Dataset Split")
    print("-------------------")
    print("Training:", len(X_train_tree))
    print("Validation:", len(X_val_tree))
    print("Testing:", len(X_test_tree))
    print("\nTree Features:")
    print(list(tree_features))
