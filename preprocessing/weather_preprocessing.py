import pandas as pd
from pathlib import Path


# ===============================
# Project Paths
# ===============================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR /
    "datasets" /
    "weather_forecast_data.csv"
)


OUTPUT_PATH = (
    BASE_DIR /
    "datasets" /
    "weather_processed.csv"
)



# ===============================
# Load Dataset
# ===============================

def load_dataset():

    df = pd.read_csv(
        DATASET_PATH
    )

    return df



# ===============================
# Dataset Information
# ===============================

def dataset_info(df):

    print("\nDataset Information")
    print("-------------------")


    print(
        "Samples:",
        len(df)
    )


    print(
        "\nColumns:"
    )

    print(
        df.columns
    )


    print(
        "\nMissing Values:"
    )

    print(
        df.isnull().sum()
    )



# ===============================
# Create Weather Labels
# ===============================

def create_weather_label(df):

    df = df.copy()


    weather = []


    for index, row in df.iterrows():


        rain = row["Rain"]

        humidity = row["Humidity"]

        wind = row["Wind_Speed"]

        cloud = row["Cloud_Cover"]



        # Rainy condition

        if (
            rain.lower() == "rain"
            and humidity >= 70
        ):

            weather.append(
                "Rainy"
            )



        # Windy condition

        elif wind >= 10:

            weather.append(
                "Windy"
            )



        # Sunny condition

        else:

            weather.append(
                "Sunny"
            )



    df["Weather_State"] = weather


    return df



# ===============================
# Keep Required Features
# ===============================

def select_features(df):


    df = df[

        [

            "Temperature",

            "Humidity",

            "Wind_Speed",

            "Cloud_Cover",

            "Pressure",

            "Weather_State"

        ]

    ]


    return df



# ===============================
# Save Dataset
# ===============================

def save_dataset(df):

    df.to_csv(

        OUTPUT_PATH,

        index=False

    )


    print(
        "\nProcessed dataset saved:"
    )

    print(
        OUTPUT_PATH
    )



# ===============================
# Complete Pipeline
# ===============================

def prepare_weather_data():


    df = load_dataset()


    dataset_info(df)


    df = create_weather_label(
        df
    )


    df = select_features(
        df
    )


    print(
        "\nWeather Class Distribution:"
    )


    print(
        df["Weather_State"]
        .value_counts()
    )


    save_dataset(
        df
    )


    return df



# ===============================
# Testing
# ===============================

if __name__ == "__main__":


    weather_df = prepare_weather_data()


    print(
        "\nFirst 5 Rows:"
    )


    print(
        weather_df.head()
    )