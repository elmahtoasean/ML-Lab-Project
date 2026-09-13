# ML Lab Project

A Machine Learning Lab project implementing multiple ML algorithms with preprocessing pipelines and GUI applications.

Implemented Models:

- K-Nearest Neighbors (KNN)
- K-Means Clustering
- Decision Tree Classification
- Hidden Markov Model (HMM)
- Support Vector Machine (SVM)
- Perceptron Neural Network

Each model includes:
- Dataset preprocessing
- Model training
- Evaluation
- Saved model files
- Interactive GUI application


# Installation Guide

## 1. Open VSCode

Open Visual Studio Code.

---

## 2. Clone Repository

Open VSCode terminal and run:

```powershell
git clone https://github.com/elmahtoasean/ML-Lab-Project.git
```

Open the project folder:

```
ML Lab Project
```

in VSCode.

---

# Python Setup

## 3. Open PowerShell Terminal

In VSCode:

```
Terminal → New Terminal
```

Select:

```
PowerShell
```

---

## 4. Check Python Installation

Run:

```powershell
python --version
```

Then:

```powershell
py --list
```

Expected output:

```
-V:3.12 Python 3.12.x
-V:3.14 Python 3.14.x *
```

If Python 3.12 exists, no installation is required.

---

## 5. Install Python (If Required)

If only Python 3.14 appears:

```
-V:3.14 Python 3.14.x
```

Install Python 3.12 again using the Python official installer.

During installation:

### IMPORTANT

Before clicking **Install**, enable:

```
☑ Add python.exe to PATH
```

Example:

```
☑ Add Python 3.12 to PATH
```

Choose:

```
Customize Installation
```

Make sure these options are selected:

1. pip
2. py launcher
3. venv

After installation:

1. Close PowerShell completely.
2. Restart VSCode.
3. Open a new PowerShell terminal.

Verify:

```powershell
py --list
```

Expected:

```
-V:3.12 Python 3.12.x
-V:3.14 Python 3.14.x *
```

---

# Virtual Environment Setup

## 6. Create Virtual Environment

Inside the project folder:

```powershell
py -3.12 -m venv venv
```

A new folder will be created:

```
ML Lab Project
 ┣ venv
 ┣ datasets
 ┣ gui
 ┣ models
 ┣ preprocessing
```

---

## 7. Activate Virtual Environment

Run:

```powershell
.\venv\Scripts\Activate.ps1
```

Successful activation will show:

```
(venv)
```

before the terminal path.

Example:

```
(venv) PS D:\ML Lab Project>
```

---

## 8. Upgrade pip

Run:

```powershell
python -m pip install --upgrade pip
```

---

## 9. Install Required Packages

Run:

```powershell
python -m pip install numpy pandas scikit-learn matplotlib seaborn joblib customtkinter pillow hmmlearn scipy
```

---

# Running the Project

## Step 1: Preprocessing

Run preprocessing before training models.

---

## Obesity Dataset

Dataset:

```
datasets/obesity_dataset.csv
```

Run:

```powershell
python -m preprocessing.data_preprocessing
```

No output is expected. This is normal.

---

## Fruit Dataset

Dataset:

```
datasets/fruit_dataset.csv
```

Run:

```powershell
python -m preprocessing.fruit_preprocessing
```

No output is expected. This is normal.

---

## Weather Dataset

Dataset:

```
datasets/weather_forecast_data.csv
```

Run:

```powershell
python -m preprocessing.weather_preprocessing
```

No output is expected. This is normal.

---

# Machine Learning Models and GUI Applications

# 1. K-Nearest Neighbors (KNN)

Dataset:

```
obesity_dataset.csv
```

### Train Model

```powershell
python -m models.knn_model
```

### Run GUI

```powershell
python -m gui.knn_app
```

---

# 2. K-Means Clustering

Dataset:

```
obesity_dataset.csv
```

### Train Model

```powershell
python -m models.kmeans_model
```

### Run GUI

```powershell
python -m gui.kmeans_app
```

---

# 3. Decision Tree Classification

Dataset:

```
weather_processed.csv
```

### Train Model

```powershell
python -m models.weather_decision_tree
```

### Run GUI

```powershell
python -m gui.dt_weather_classification_app
```

---

# 4. Hidden Markov Model (HMM)

Dataset:

```
hmm_weather_dataset.csv
```

### Train Model

```powershell
python -m models.hmm_weather
```

### Run GUI

```powershell
python -m gui.hmm_weather_app
```

---

# 5. Support Vector Machine (SVM)

Dataset:

```
obesity_dataset.csv
```

### Train Model

```powershell
python -m models.svm_model
```

### Run GUI

```powershell
python -m gui.svm_app
```

---

# 6. Perceptron Neural Network

Dataset:

```
fruit_dataset.csv
```

### Train Model

```powershell
python -m models.perceptron_model
```

### Run GUI

```powershell
python -m gui.perceptron_app
```

---

# Project Structure

```
ML Lab Project
 ┣ datasets
 ┃ ┣ fruit_dataset.csv
 ┃ ┣ hmm_weather_dataset.csv
 ┃ ┣ obesity_dataset.csv
 ┃ ┣ weather_forecast_data.csv
 ┃ ┗ weather_processed.csv
 ┣ gui
 ┃ ┣ knn_app.py
 ┃ ┣ kmeans_app.py
 ┃ ┣ dt_weather_classification_app.py
 ┃ ┣ hmm_weather_app.py
 ┃ ┣ svm_app.py
 ┃ ┗ perceptron_app.py
 ┣ models
 ┃ ┣ knn_model.py
 ┃ ┣ kmeans_model.py
 ┃ ┣ weather_decision_tree.py
 ┃ ┣ hmm_weather.py
 ┃ ┣ svm_model.py
 ┃ ┗ perceptron_model.py
 ┣ preprocessing
 ┃ ┣ data_preprocessing.py
 ┃ ┣ fruit_preprocessing.py
 ┃ ┗ weather_preprocessing.py
 ┣ plots
 ┣ saved_models
 ┗ README.md
```

---

# Notes

- Always activate the virtual environment before running models or GUI applications.
- Train the model first before opening the corresponding GUI.
- Generated models are stored inside:

```
saved_models/
```

- Generated evaluation plots are stored inside:

```
plots/
```
