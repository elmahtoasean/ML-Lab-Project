import tkinter as tk
from tkinter import ttk

import joblib
import numpy as np
from pathlib import Path


# ==============================
# Paths
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "saved_models"


# ==============================
# Load HMM Model
# ==============================

model = joblib.load(
    MODEL_DIR / "weather_hmm.pkl"
)

encoder = joblib.load(
    MODEL_DIR / "weather_hmm_encoder.pkl"
)


# ==============================
# HMM Prediction Function
# ==============================

def predict_next_weather(model, encoder, previous_days):

    encoded = encoder.transform(previous_days)
    observations = encoded.reshape(-1, 1)

    logprob, hidden_probability = model.score_samples(observations)

    # Last hidden state probability
    current_hidden_probability = hidden_probability[-1]

    # Predict next hidden state
    next_hidden_probability = current_hidden_probability @ model.transmat_
    next_hidden_state = np.argmax(next_hidden_probability)

    # Convert hidden state to weather probability
    weather_probability = model.emissionprob_[next_hidden_state]
    next_weather_state = np.argmax(weather_probability)

    prediction = encoder.inverse_transform([next_weather_state])[0]
    confidence = weather_probability[next_weather_state] * 100

    return prediction, confidence


# ==============================
# Design Tokens
# ==============================

FONT_FAMILY = "Segoe UI"

COLOR_BG = "#FFFFFF"
COLOR_NAVY = "#0F2A5C"        # headings, primary text
COLOR_BLUE = "#1D4ED8"        # primary action, accents
COLOR_BLUE_HOVER = "#1E40AF"  # button hover state
COLOR_BLUE_LIGHT = "#93C5FD"  # disabled button state
COLOR_SKY = "#3B82F6"         # icon accents
COLOR_PANEL = "#F3F8FF"       # panel / card background
COLOR_BORDER = "#BFDBFE"      # borders and dividers
COLOR_MUTED = "#5B6B82"       # helper and secondary text

WEATHER_OPTIONS = ["Sunny", "Rainy", "Windy"]
WEATHER_ICONS = {"Sunny": "\u2600", "Rainy": "\u2614", "Windy": "\u2248"}
PLACEHOLDER = "Choose weather"


# ==============================
# Root Window
# ==============================

root = tk.Tk()
root.title("HMM Weather Predictor")
root.configure(bg=COLOR_BG)
root.resizable(False, False)

WIN_W = 560


# ==============================
# ttk Styling
# ==============================

style = ttk.Style()
style.theme_use("clam")

style.configure(
    "Weather.TCombobox",
    fieldbackground=COLOR_BG,
    background=COLOR_BG,
    foreground=COLOR_NAVY,
    arrowcolor=COLOR_BLUE,
    bordercolor=COLOR_BORDER,
    lightcolor=COLOR_BG,
    darkcolor=COLOR_BG,
    padding=6,
)
style.map(
    "Weather.TCombobox",
    fieldbackground=[("readonly", COLOR_BG)],
    foreground=[("readonly", COLOR_NAVY)],
    bordercolor=[("focus", COLOR_BLUE)],
)

style.configure(
    "Blue.Horizontal.TProgressbar",
    troughcolor=COLOR_PANEL,
    bordercolor=COLOR_PANEL,
    background=COLOR_BLUE,
    lightcolor=COLOR_BLUE,
    darkcolor=COLOR_BLUE,
    thickness=10,
)

root.option_add("*TCombobox*Listbox.background", COLOR_BG)
root.option_add("*TCombobox*Listbox.foreground", COLOR_NAVY)
root.option_add("*TCombobox*Listbox.selectBackground", COLOR_PANEL)
root.option_add("*TCombobox*Listbox.selectForeground", COLOR_NAVY)
root.option_add("*TCombobox*Listbox.font", (FONT_FAMILY, 11))


# ==============================
# Header
# ==============================

header = tk.Frame(root, bg=COLOR_BG)
header.pack(pady=(36, 8), padx=40, fill="x")

title = tk.Label(
    header,
    text="Tomorrow's weather",
    font=(FONT_FAMILY, 24, "bold"),
    fg=COLOR_NAVY,
    bg=COLOR_BG,
)
title.pack(anchor="w")

subtitle = tk.Label(
    header,
    text="A three-day pattern is enough for the model to forecast the next day.",
    font=(FONT_FAMILY, 11),
    fg=COLOR_MUTED,
    bg=COLOR_BG,
    justify="left",
    wraplength=460,
)
subtitle.pack(anchor="w", pady=(4, 0))


# ==============================
# Day Selectors
# ==============================

selector_section = tk.Frame(root, bg=COLOR_BG)
selector_section.pack(pady=(20, 4), padx=40, fill="x")

day_vars = [tk.StringVar(value=PLACEHOLDER) for _ in range(3)]
icon_labels = []
comboboxes = []

for i in range(3):
    column = tk.Frame(selector_section, bg=COLOR_BG)
    column.grid(row=0, column=i, padx=(0, 14) if i < 2 else 0, sticky="ew")
    selector_section.grid_columnconfigure(i, weight=1)

    day_label = tk.Label(
        column,
        text=f"Day {i + 1}",
        font=(FONT_FAMILY, 10, "bold"),
        fg=COLOR_NAVY,
        bg=COLOR_BG,
    )
    day_label.pack(anchor="w")

    combo = ttk.Combobox(
        column,
        values=WEATHER_OPTIONS,
        state="readonly",
        textvariable=day_vars[i],
        style="Weather.TCombobox",
        font=(FONT_FAMILY, 11),
    )
    combo.pack(fill="x", pady=(4, 0))
    comboboxes.append(combo)

    icon_label = tk.Label(
        column,
        text="",
        font=(FONT_FAMILY, 16),
        fg=COLOR_SKY,
        bg=COLOR_BG,
    )
    icon_label.pack(pady=(6, 0))
    icon_labels.append(icon_label)


# ==============================
# Status Line
# ==============================

status_label = tk.Label(
    root,
    text="",
    font=(FONT_FAMILY, 10),
    fg=COLOR_MUTED,
    bg=COLOR_BG,
    wraplength=460,
    justify="left",
)
status_label.pack(padx=40, pady=(14, 0), anchor="w")


# ==============================
# Predict Button
# ==============================

button = tk.Button(
    root,
    text="Predict tomorrow",
    font=(FONT_FAMILY, 12, "bold"),
    fg="#FFFFFF",
    bg=COLOR_BLUE,
    activeforeground="#FFFFFF",
    activebackground=COLOR_BLUE_HOVER,
    bd=0,
    padx=18,
    pady=10,
    cursor="hand2",
    state="disabled",
    disabledforeground="#FFFFFF",
)
button.configure(bg=COLOR_BLUE_LIGHT)
button.pack(padx=40, pady=22, fill="x")


def on_button_enter(event):
    if str(button["state"]) == "normal":
        button.configure(bg=COLOR_BLUE_HOVER)


def on_button_leave(event):
    if str(button["state"]) == "normal":
        button.configure(bg=COLOR_BLUE)


button.bind("<Enter>", on_button_enter)
button.bind("<Leave>", on_button_leave)


# ==============================
# Result Panel
# ==============================

result_panel = tk.Frame(
    root,
    bg=COLOR_PANEL,
    highlightbackground=COLOR_BORDER,
    highlightthickness=1,
)
result_panel.pack(padx=40, pady=(0, 32), fill="both", expand=True)

result_empty_label = tk.Label(
    result_panel,
    text="A forecast for tomorrow will appear here once three days are chosen.",
    font=(FONT_FAMILY, 11),
    fg=COLOR_MUTED,
    bg=COLOR_PANEL,
    wraplength=400,
    justify="center",
)
result_empty_label.pack(expand=True, pady=40)

result_content = tk.Frame(result_panel, bg=COLOR_PANEL)

result_icon = tk.Label(
    result_content,
    text="",
    font=(FONT_FAMILY, 32),
    fg=COLOR_SKY,
    bg=COLOR_PANEL,
)
result_icon.pack(pady=(24, 0))

result_weather = tk.Label(
    result_content,
    text="",
    font=(FONT_FAMILY, 18, "bold"),
    fg=COLOR_NAVY,
    bg=COLOR_PANEL,
)
result_weather.pack(pady=(4, 14))

progress_bar = ttk.Progressbar(
    result_content,
    style="Blue.Horizontal.TProgressbar",
    orient="horizontal",
    mode="determinate",
    length=340,
    maximum=100,
)
progress_bar.pack()

result_confidence = tk.Label(
    result_content,
    text="",
    font=(FONT_FAMILY, 10),
    fg=COLOR_MUTED,
    bg=COLOR_PANEL,
)
result_confidence.pack(pady=(8, 24))


# ==============================
# State Helpers
# ==============================

def get_missing_days():
    return [i + 1 for i, var in enumerate(day_vars) if var.get() == PLACEHOLDER]


def refresh_state(*_args):

    result_content.pack_forget()
    result_empty_label.pack(expand=True, pady=40)
    progress_bar["value"] = 0
    result_confidence.config(text="")

    for i, var in enumerate(day_vars):
        value = var.get()
        icon_labels[i].config(text=WEATHER_ICONS.get(value, ""))

    missing = get_missing_days()

    if missing:
        button.configure(state="disabled", bg=COLOR_BLUE_LIGHT)
        if len(missing) == 3:
            status_label.config(text="This forecast needs a weather condition for all three days.")
        else:
            names = ", ".join(f"Day {d}" for d in missing)
            status_label.config(text=f"This input is incomplete. {names} still need a selection.")
    else:
        button.configure(state="normal", bg=COLOR_BLUE)
        status_label.config(text="This selection is ready. The forecast can run now.")


for var in day_vars:
    var.trace_add("write", refresh_state)


# ==============================
# Predict Action
# ==============================

def predict():

    missing = get_missing_days()
    if missing:
        refresh_state()
        return

    previous_days = [var.get() for var in day_vars]

    prediction, confidence = predict_next_weather(model, encoder, previous_days)

    result_empty_label.pack_forget()
    result_content.pack(fill="both", expand=True)

    result_icon.config(text=WEATHER_ICONS.get(prediction, ""))
    result_weather.config(text=f"Tomorrow: {prediction}")
    progress_bar["value"] = confidence
    result_confidence.config(text=f"{confidence:.2f}% confidence in this outcome")


button.configure(command=predict)
root.bind("<Return>", lambda event: predict())


# ==============================
# Size the window to fit the real content
# ==============================
# Font rendering height varies by OS/DPI, so instead of guessing a fixed
# window height, render the result card with sample content once, measure
# it, then size the window to comfortably fit that — and only after that
# switch back to the actual initial (empty) state.

result_empty_label.pack_forget()
result_content.pack(fill="both", expand=True)
result_icon.config(text=WEATHER_ICONS["Sunny"])
result_weather.config(text="Tomorrow: Sunny")
progress_bar["value"] = 100
result_confidence.config(text="100.00% confidence in this outcome")

root.update_idletasks()
win_h = root.winfo_reqheight() + 24  # small safety margin

screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
pos_x = (screen_w - WIN_W) // 2
pos_y = (screen_h - win_h) // 3
root.geometry(f"{WIN_W}x{win_h}+{pos_x}+{pos_y}")


# ==============================
# Initial State
# ==============================

refresh_state()


# ==============================
# Start GUI
# ==============================

root.mainloop()