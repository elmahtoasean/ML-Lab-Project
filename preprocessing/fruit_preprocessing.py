from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "datasets" / "fruit_dataset.csv"


# ==========================================================
# Label Mapping
# ==========================================================

LABEL_MAP = {"p1": 0, "p2": 1, "p3": 2, "p4": 3}


FRUIT_NAMES = {"p1": "Watermelon", "p2": "Banana", "p3": "Orange", "p4": "Apple"}


# ==========================================================
# Valid Fruit Sensor Rules
# ==========================================================

VALID_PATTERNS = {
    # shape, texture, weight, label
    (1, 1, 1, 0),  # Watermelon
    (-1, 1, -1, 1),  # Banana
    (1, -1, -1, 2),  # Orange
    (1, 1, -1, 3),  # Apple
}


# ==========================================================
# Load Dataset
# ==========================================================


def load_dataset():

    df = pd.read_csv(DATASET_PATH)

    df.columns = df.columns.str.strip().str.lower()

    return df


# ==========================================================
# Validate Dataset
# ==========================================================


def validate_dataset(df):

    required_columns = ["shape", "texture", "weight", "label"]

    for col in required_columns:

        if col not in df.columns:

            raise ValueError(f"Missing column: {col}")

    # Sensor values

    for col in ["shape", "texture", "weight"]:

        values = set(df[col].unique())

        if not values.issubset({-1, 1}):

            raise ValueError(f"{col} contains invalid values: {values}")

    # Labels

    labels = set(df["label"].unique())

    if not labels.issubset(LABEL_MAP.keys()):

        raise ValueError(f"Invalid fruit labels: {labels}")

    return True


# ==========================================================
# Encode Dataset
# ==========================================================


def encode_dataset(df):

    encoded = pd.DataFrame(index=df.index)

    # -----------------------
    # Shape
    # -----------------------

    encoded["shape"] = df["shape"].astype(int)

    # -----------------------
    # Texture
    # -----------------------

    encoded["texture"] = df["texture"].astype(int)

    # -----------------------
    # Weight
    # -----------------------

    encoded["weight"] = df["weight"].astype(int)

    # -----------------------
    # Label
    # -----------------------

    encoded["label"] = df["label"].astype(str).str.strip().map(LABEL_MAP)

    # Check unknown labels

    if encoded["label"].isna().any():

        raise ValueError("Unknown fruit label found")

    # -----------------------
    # Validate sensor rules
    # -----------------------

    invalid_rows = encoded[
        ~encoded.apply(
            lambda row: (row["shape"], row["texture"], row["weight"], row["label"])
            in VALID_PATTERNS,
            axis=1,
        )
    ]

    if len(invalid_rows) > 0:

        raise ValueError("Invalid fruit sensor combination found:\n" f"{invalid_rows}")

    return encoded.astype(int)


# ==========================================================
# Prepare Data
# ==========================================================
def check_class_distribution(df):

    counts = df["label"].value_counts()

    for label in LABEL_MAP.keys():

        if label not in counts.index:

            raise ValueError(f"No samples found for {label}")


def prepare_data():

    df = load_dataset()

    validate_dataset(df)
    check_class_distribution(df)

    encoded = encode_dataset(df)
    # print(encoded.groupby(["shape", "texture", "weight"])["label"].unique())

    feature_names = ["shape", "texture", "weight"]

    X = encoded[feature_names].values

    y = encoded["label"].values

    # 70% train
    # 15% validation
    # 15% test

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1765,
        random_state=42,
        stratify=y_train_val,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_names,
        LABEL_MAP,
        FRUIT_NAMES,
    )
