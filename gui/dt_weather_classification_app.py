from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path

import customtkinter as ctk
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets"
MODEL_DIR = BASE_DIR.parent / "saved_models"
DATASET = BASE_DIR.parent / "datasets" / "weather_processed.csv"


# --------------------------------------------------------------------------
# Design tokens — same family as the cluster analyzer
# --------------------------------------------------------------------------

INK = "#FFFFFF"  # page ground and field fill
PANEL = "#F6F9FD"  # panel surfaces
PANEL_HI = "#E9F1FA"  # raised rows
RULE = "#CBDCEF"  # hairlines, tracks, grid
TEXT = "#0A3466"  # primary blue type
MUTED = "#5E87B3"  # secondary blue type
MARK = "#0A2A5E"  # emphasis

STATE_INKS = {"Rainy": "#1F6FBF", "Sunny": "#0E8C8C", "Windy": "#5B54C9"}
STATE_GIFS = {"Rainy": "rainy.gif", "Sunny": "sunny.gif", "Windy": "windy.gif"}

# (label, dataframe column, unit, slider range, default)
FEATURES = [
    ("Temperature", "Temperature", "°C", (10.0, 60.0), 22.0),
    ("Humidity", "Humidity", "%", (30.0, 100.0), 64.0),
    ("Wind speed", "Wind_Speed", "km/h", (0.0, 30.0), 10.0),
    ("Cloud cover", "Cloud_Cover", "%", (0.0, 100.0), 50.0),
    ("Pressure", "Pressure", "hPa", (950.0, 1100.0), 1014.0),
]
COLUMNS = [f[1] for f in FEATURES]


# --------------------------------------------------------------------------
# Model bundle
# --------------------------------------------------------------------------


class WeatherModel:
    """Wraps the saved decision tree and the training population."""

    def __init__(self) -> None:
        self.model = joblib.load(MODEL_DIR / "weather_decision_tree.pkl")
        try:
            self.metadata = joblib.load(
                MODEL_DIR / "weather_decision_tree_metadata.pkl"
            )
        except FileNotFoundError:
            self.metadata = {}

        self.classes = list(self.model.classes_)
        self.tree = self.model.tree_
        self.depth = self.model.get_depth()
        self.importance = dict(zip(COLUMNS, self.model.feature_importances_))

        self.df: pd.DataFrame | None = None
        self.observed: dict[str, tuple[float, float]] = {}
        self.presets: dict[str, dict[str, float]] = {}

    # -- population -------------------------------------------------------

    def load_population(self) -> None:
        df = pd.read_csv(DATASET)
        self.df = df
        self.observed = {c: (float(df[c].min()), float(df[c].max())) for c in COLUMNS}
        means = df.groupby("Weather_State")[COLUMNS].mean()
        self.presets = {state: means.loc[state].to_dict() for state in means.index}

    def percentiles(self, values: dict[str, float]) -> dict[str, float]:
        if self.df is None:
            return {}
        return {c: float((self.df[c] < values[c]).mean() * 100.0) for c in COLUMNS}

    # -- inference --------------------------------------------------------

    def frame(self, values: dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame([{c: values[c] for c in COLUMNS}])

    def predict(self, values: dict[str, float]) -> dict:
        x = self.frame(values)
        label = str(self.model.predict(x)[0])
        proba = self.model.predict_proba(x)[0]
        leaf = int(self.model.apply(x)[0])
        counts = self.tree.value[leaf][0]

        return {
            "label": label,
            "proba": dict(zip(self.classes, proba)),
            "rules": self.rule_path(x),
            "leaf_samples": int(self.tree.n_node_samples[leaf]),
            "leaf_counts": dict(zip(self.classes, counts)),
        }

    def rule_path(self, x: pd.DataFrame) -> list[tuple[str, str, bool]]:
        """The exact comparisons the tree made, top to bottom."""
        indicator = self.model.decision_path(x)
        nodes = indicator.indices[indicator.indptr[0] : indicator.indptr[1]]
        leaf = int(self.model.apply(x)[0])

        rules = []
        for node in nodes:
            if node == leaf:
                break
            col = COLUMNS[self.tree.feature[node]]
            thresh = float(self.tree.threshold[node])
            value = float(x.iloc[0][col])
            went_left = value <= thresh
            label = next(f[0] for f in FEATURES if f[1] == col)
            unit = next(f[2] for f in FEATURES if f[1] == col)
            sign = "≤" if went_left else ">"
            rules.append((label, f"{value:.1f} {sign} {thresh:.1f} {unit}", went_left))
        return rules

    def flips(self, values: dict[str, float]) -> list[dict]:
        """Smallest single-feature change that lands on another state."""
        base_label = str(self.model.predict(self.frame(values))[0])
        base_row = np.array([[values[c] for c in COLUMNS]], dtype=float)

        found = []
        for label, col, unit, (lo, hi), _ in FEATURES:
            idx = COLUMNS.index(col)
            grid = np.linspace(lo, hi, 1201)
            block = np.repeat(base_row, len(grid), axis=0)
            block[:, idx] = grid
            preds = self.model.predict(pd.DataFrame(block, columns=COLUMNS))

            changed = np.flatnonzero(preds != base_label)
            if changed.size == 0:
                found.append({"label": label, "unit": unit, "target": None})
                continue

            j = changed[np.argmin(np.abs(grid[changed] - values[col]))]
            found.append(
                {
                    "label": label,
                    "unit": unit,
                    "target": float(grid[j]),
                    "delta": float(grid[j] - values[col]),
                    "state": str(preds[j]),
                    "effort": abs(grid[j] - values[col]) / (hi - lo),
                }
            )

        found.sort(key=lambda d: d.get("effort", 9.9))
        return found


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

ctk.set_appearance_mode("light")


class WeatherApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Weather classification")
        self.geometry("1180x820")
        self.minsize(1040, 740)
        self.configure(fg_color=INK)

        self.wm = WeatherModel()
        self._redraw_job: str | None = None
        self._gif_cache: dict[str, list] = {}
        self._gif_job: str | None = None
        self._gif_state: str | None = None
        self._ready = False

        self.f_display = ctk.CTkFont("Segoe UI", 30, "bold")
        self.f_state = ctk.CTkFont("Segoe UI", 34, "bold")
        self.f_title = ctk.CTkFont("Segoe UI", 15, "bold")
        self.f_body = ctk.CTkFont("Segoe UI", 13)
        self.f_small = ctk.CTkFont("Segoe UI", 12)
        self.f_readout = ctk.CTkFont("Consolas", 15)

        self.vars: dict[str, ctk.DoubleVar] = {
            col: ctk.DoubleVar(value=default) for _, col, _, _, default in FEATURES
        }
        self.entries: dict[str, ctk.CTkEntry] = {}
        self.range_notes: dict[str, ctk.CTkLabel] = {}

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_rail()
        self._build_stage()

        self.after(60, self._start_loading)

    # -- chrome -----------------------------------------------------------

    def _build_header(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=INK, corner_radius=0, height=74)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            bar, text="Weather classification", font=self.f_display, text_color=MARK
        ).grid(row=0, column=0, padx=(28, 16), pady=(16, 4), sticky="w")

        accuracy = self.wm.metadata.get("accuracy")
        note = f"Decision tree, entropy, depth {self.wm.depth}"
        if accuracy:
            note += f", {accuracy * 100:.1f}% accurate on the held-out test set"
        self.model_note = ctk.CTkLabel(
            bar, text=note, font=self.f_small, text_color=MUTED
        )
        self.model_note.grid(row=0, column=1, sticky="w", pady=(24, 0))

        ctk.CTkFrame(self, fg_color=RULE, height=1, corner_radius=0).grid(
            row=0, column=0, columnspan=2, sticky="sew"
        )

    def _build_rail(self) -> None:
        rail = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, width=340)
        rail.grid(row=1, column=0, sticky="nsw")
        rail.grid_propagate(False)
        rail.grid_rowconfigure(8, weight=1)

        ctk.CTkLabel(rail, text="Conditions", font=self.f_title, text_color=MARK).grid(
            row=0, column=0, padx=24, pady=(22, 10), sticky="w"
        )

        for i, (label, col, unit, limits, _) in enumerate(FEATURES):
            self._measure_row(rail, i + 1, label, col, unit, limits)

        ctk.CTkFrame(rail, fg_color=RULE, height=1).grid(
            row=6, column=0, padx=24, pady=(12, 14), sticky="ew"
        )

        ctk.CTkLabel(
            rail, text="Load an average day", font=self.f_small, text_color=MUTED
        ).grid(row=7, column=0, padx=24, sticky="w")

        self.preset_bar = ctk.CTkFrame(rail, fg_color="transparent")
        self.preset_bar.grid(row=8, column=0, padx=20, pady=(8, 0), sticky="new")

        ctk.CTkLabel(
            rail,
            text="Readings are classified by a tree trained on\n"
            "2,500 records. It reports a state, not a forecast.",
            font=self.f_small,
            text_color=MUTED,
            justify="left",
        ).grid(row=9, column=0, padx=24, pady=(0, 20), sticky="sw")

    def _measure_row(self, parent, row, label, col, unit, limits) -> None:
        holder = ctk.CTkFrame(parent, fg_color="transparent")
        holder.grid(row=row, column=0, padx=24, pady=(2, 8), sticky="ew")
        holder.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(holder, text=label, font=self.f_body, text_color=TEXT).grid(
            row=0, column=0, sticky="w"
        )

        entry = ctk.CTkEntry(
            holder,
            width=88,
            height=30,
            font=self.f_readout,
            justify="right",
            fg_color=INK,
            border_color=RULE,
            text_color=MARK,
            corner_radius=6,
        )
        entry.grid(row=0, column=1, sticky="e")
        entry.insert(0, f"{self.vars[col].get():.1f}")
        entry.bind("<Return>", lambda _e, c=col, l=limits: self._commit(c, l))
        entry.bind("<FocusOut>", lambda _e, c=col, l=limits: self._commit(c, l))
        self.entries[col] = entry

        ctk.CTkLabel(
            holder, text=unit, font=self.f_small, text_color=MUTED, width=34, anchor="w"
        ).grid(row=0, column=2, padx=(6, 0), sticky="e")

        ctk.CTkSlider(
            holder,
            from_=limits[0],
            to=limits[1],
            variable=self.vars[col],
            height=16,
            button_color=MARK,
            button_hover_color=TEXT,
            progress_color=TEXT,
            fg_color=RULE,
            command=lambda _v, c=col: self._on_slide(c),
        ).grid(row=1, column=0, columnspan=3, pady=(8, 2), sticky="ew")

        note = ctk.CTkLabel(holder, text="", font=self.f_small, text_color=MUTED)
        note.grid(row=2, column=0, columnspan=3, sticky="w")
        self.range_notes[col] = note

    def _build_stage(self) -> None:
        stage = ctk.CTkFrame(self, fg_color=INK, corner_radius=0)
        stage.grid(row=1, column=1, sticky="nsew")
        stage.grid_columnconfigure(0, weight=1)
        stage.grid_rowconfigure(1, weight=1)

        # --- result band -------------------------------------------------
        band = ctk.CTkFrame(
            stage, fg_color=PANEL, corner_radius=8, border_width=1, border_color=RULE
        )
        band.grid(row=0, column=0, padx=18, pady=(18, 12), sticky="ew")
        band.grid_columnconfigure(0, weight=1)

        left = ctk.CTkFrame(band, fg_color="transparent")
        left.grid(row=0, column=0, padx=22, pady=18, sticky="w")

        self.state_label = ctk.CTkLabel(
            left, text="Reading the model…", font=self.f_state, text_color=MARK
        )
        self.state_label.pack(anchor="w")

        self.state_note = ctk.CTkLabel(
            left, text="", font=self.f_small, text_color=MUTED, justify="left"
        )
        self.state_note.pack(anchor="w", pady=(4, 12))

        self.prob_holder = ctk.CTkFrame(left, fg_color="transparent")
        self.prob_holder.pack(anchor="w", fill="x")
        self.prob_holder.grid_columnconfigure(1, weight=1)
        self.prob_rows: list[tuple] = []

        self.gif_label = tk.Label(band, bg=PANEL, bd=0, highlightthickness=0)
        self.gif_label.grid(row=0, column=1, padx=(10, 24), pady=16)

        # --- detail tabs -------------------------------------------------
        self.tabs = ctk.CTkTabview(
            stage,
            fg_color=PANEL,
            corner_radius=8,
            border_width=1,
            border_color=RULE,
            segmented_button_fg_color=PANEL_HI,
            segmented_button_selected_color=TEXT,
            segmented_button_selected_hover_color=MARK,
            segmented_button_unselected_color=PANEL_HI,
            segmented_button_unselected_hover_color=RULE,
            text_color=MARK,
            anchor="w",
        )
        self.tabs.grid(row=1, column=0, padx=18, pady=(0, 18), sticky="nsew")

        self.tab_why = self.tabs.add("Why this result")
        self.tab_flip = self.tabs.add("What would flip it")
        self.tab_data = self.tabs.add("Reading vs dataset")

        for tab in (self.tab_why, self.tab_flip, self.tab_data):
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)

        self.why_box = ctk.CTkScrollableFrame(
            self.tab_why, fg_color=PANEL, scrollbar_button_color=RULE
        )
        self.why_box.grid(row=0, column=0, sticky="nsew")

        self.flip_box = ctk.CTkScrollableFrame(
            self.tab_flip, fg_color=PANEL, scrollbar_button_color=RULE
        )
        self.flip_box.grid(row=0, column=0, sticky="nsew")

        self.data_box = ctk.CTkScrollableFrame(
            self.tab_data, fg_color=PANEL, scrollbar_button_color=RULE
        )
        self.data_box.grid(row=0, column=0, sticky="nsew")

    # -- loading ----------------------------------------------------------

    def _start_loading(self) -> None:
        self._build_prob_rows()
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self) -> None:
        try:
            self.wm.load_population()
        except Exception as exc:
            self.after(
                0,
                lambda: self.state_note.configure(
                    text=f"Could not read {DATASET.name}: {exc}"
                ),
            )
        self.after(0, self._on_ready)

    def _on_ready(self) -> None:
        self._build_presets()
        self._ready = True
        self.refresh()

    def _build_prob_rows(self) -> None:
        for row, state in enumerate(self.wm.classes):
            ink = STATE_INKS.get(state, TEXT)

            ctk.CTkLabel(
                self.prob_holder,
                text="",
                width=10,
                height=10,
                fg_color=ink,
                corner_radius=2,
            ).grid(row=row, column=0, padx=(0, 10), pady=3)

            name = ctk.CTkLabel(
                self.prob_holder,
                text=state,
                font=self.f_body,
                text_color=MUTED,
                width=64,
                anchor="w",
            )
            name.grid(row=row, column=1, sticky="w", pady=3)

            bar = ctk.CTkProgressBar(
                self.prob_holder,
                height=6,
                corner_radius=3,
                fg_color=RULE,
                progress_color=ink,
                width=260,
            )
            bar.set(0)
            bar.grid(row=row, column=2, padx=14, pady=3)

            pct = ctk.CTkLabel(
                self.prob_holder,
                text="0%",
                font=self.f_readout,
                text_color=MUTED,
                width=56,
                anchor="e",
            )
            pct.grid(row=row, column=3, pady=3)

            self.prob_rows.append((state, name, bar, pct))

    def _build_presets(self) -> None:
        if not self.wm.presets:
            return
        for state, row in self.wm.presets.items():
            ctk.CTkButton(
                self.preset_bar,
                text=state,
                width=88,
                height=30,
                font=self.f_small,
                corner_radius=6,
                fg_color=INK,
                hover_color=PANEL_HI,
                border_width=1,
                border_color=RULE,
                text_color=STATE_INKS.get(state, TEXT),
                command=lambda r=row: self.load_values(r),
            ).pack(side="left", padx=(0, 8))

    # -- input handling ---------------------------------------------------

    def _on_slide(self, col: str) -> None:
        entry = self.entries[col]
        entry.delete(0, "end")
        entry.insert(0, f"{self.vars[col].get():.1f}")
        self.schedule_refresh()

    def _commit(self, col: str, limits) -> None:
        entry = self.entries[col]
        raw = entry.get().strip().replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            entry.delete(0, "end")
            entry.insert(0, f"{self.vars[col].get():.1f}")
            self.state_note.configure(
                text=f"'{raw}' is not a number. Enter a value like 21.5."
            )
            return
        value = min(max(value, limits[0]), limits[1])
        self.vars[col].set(value)
        entry.delete(0, "end")
        entry.insert(0, f"{value:.1f}")
        self.schedule_refresh()

    def load_values(self, row: dict) -> None:
        for col in COLUMNS:
            value = round(float(row[col]), 1)
            self.vars[col].set(value)
            self.entries[col].delete(0, "end")
            self.entries[col].insert(0, f"{value:.1f}")
        self.refresh()

    def schedule_refresh(self) -> None:
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(110, self.refresh)

    # -- output -----------------------------------------------------------

    def current(self) -> dict[str, float]:
        return {col: float(var.get()) for col, var in self.vars.items()}

    def refresh(self) -> None:
        self._redraw_job = None
        if not self._ready:
            return

        values = self.current()
        result = self.wm.predict(values)
        state = result["label"]
        ink = STATE_INKS.get(state, TEXT)

        self.state_label.configure(text=state, text_color=ink)

        share = result["proba"][state] * 100
        leaf = result["leaf_samples"]
        self.state_note.configure(
            text=(
                f"The tree ends at a leaf holding {leaf:,} training records, "
                f"{share:.0f}% of them {state.lower()}."
            )
        )

        for st, name, bar, pct in self.prob_rows:
            p = result["proba"][st]
            bar.set(float(p))
            pct.configure(text=f"{p * 100:4.0f}%")
            hot = st == state
            name.configure(text_color=MARK if hot else MUTED)
            pct.configure(text_color=MARK if hot else MUTED)

        self._update_range_notes(values)
        self._render_rules(result["rules"], state)
        self._render_flips(values, state)
        self._render_dataset(values)
        self._show_gif(state)

    def _update_range_notes(self, values: dict[str, float]) -> None:
        for label, col, unit, _, _ in FEATURES:
            note = self.range_notes[col]
            if col not in self.wm.observed:
                note.configure(text="")
                continue
            lo, hi = self.wm.observed[col]
            v = values[col]
            if v < lo or v > hi:
                note.configure(
                    text=f"Outside training data ({lo:.0f}–{hi:.0f} {unit})",
                    text_color=STATE_INKS["Windy"],
                )
            else:
                note.configure(text="", text_color=MUTED)

    def _render_rules(self, rules, state: str) -> None:
        for child in self.why_box.winfo_children():
            child.destroy()

        ink = STATE_INKS.get(state, TEXT)
        ctk.CTkLabel(
            self.why_box,
            text=f"{len(rules)} comparisons were enough to reach {state.lower()}.",
            font=self.f_body,
            text_color=MUTED,
        ).pack(anchor="w", padx=12, pady=(10, 8))

        for step, (label, text, went_left) in enumerate(rules, start=1):
            row = ctk.CTkFrame(
                self.why_box,
                fg_color=INK,
                corner_radius=6,
                border_width=1,
                border_color=RULE,
            )
            row.pack(fill="x", padx=10, pady=3)

            ctk.CTkLabel(
                row, text=f"{step}", font=self.f_small, text_color=MUTED, width=22
            ).pack(side="left", padx=(12, 6), pady=8)
            ctk.CTkLabel(
                row,
                text=label,
                font=self.f_body,
                text_color=MARK,
                width=110,
                anchor="w",
            ).pack(side="left")
            ctk.CTkLabel(
                row, text=text, font=self.f_readout, text_color=TEXT, anchor="w"
            ).pack(side="left")
            ctk.CTkLabel(
                row,
                text="left branch" if went_left else "right branch",
                font=self.f_small,
                text_color=ink,
            ).pack(side="right", padx=14)

    def _render_flips(self, values: dict[str, float], state: str) -> None:
        for child in self.flip_box.winfo_children():
            child.destroy()

        ctk.CTkLabel(
            self.flip_box,
            text="Smallest single change to each reading that produces a different state.",
            font=self.f_body,
            text_color=MUTED,
        ).pack(anchor="w", padx=12, pady=(10, 8))

        for item in self.wm.flips(values):
            row = ctk.CTkFrame(
                self.flip_box,
                fg_color=INK,
                corner_radius=6,
                border_width=1,
                border_color=RULE,
            )
            row.pack(fill="x", padx=10, pady=3)

            ctk.CTkLabel(
                row,
                text=item["label"],
                font=self.f_body,
                text_color=MARK,
                width=120,
                anchor="w",
            ).pack(side="left", padx=(14, 6), pady=9)

            if item["target"] is None:
                ctk.CTkLabel(
                    row,
                    text=f"No value in range changes {state.lower()}",
                    font=self.f_small,
                    text_color=MUTED,
                ).pack(side="left")
                continue

            arrow = "up to" if item["delta"] > 0 else "down to"
            ctk.CTkLabel(
                row,
                text=f"{arrow} {item['target']:.1f} {item['unit']}"
                f"   ({item['delta']:+.1f})",
                font=self.f_readout,
                text_color=TEXT,
            ).pack(side="left")
            ctk.CTkLabel(
                row,
                text=f"→ {item['state']}",
                font=self.f_body,
                text_color=STATE_INKS.get(item["state"], TEXT),
            ).pack(side="right", padx=14)

    def _render_dataset(self, values: dict[str, float]) -> None:
        for child in self.data_box.winfo_children():
            child.destroy()

        pct = self.wm.percentiles(values)
        importance = self.wm.importance

        ctk.CTkLabel(
            self.data_box,
            text="Where each reading sits in the training data, "
            "and how much the tree relies on it.",
            font=self.f_body,
            text_color=MUTED,
        ).pack(anchor="w", padx=12, pady=(10, 8))

        grid = ctk.CTkFrame(self.data_box, fg_color="transparent")
        grid.pack(fill="x", padx=10)
        grid.grid_columnconfigure(1, weight=1)

        for row, (label, col, unit, _, _) in enumerate(FEATURES):
            ctk.CTkLabel(
                grid,
                text=label,
                font=self.f_body,
                text_color=MARK,
                width=120,
                anchor="w",
            ).grid(row=row, column=0, pady=6, sticky="w")

            bar = ctk.CTkProgressBar(
                grid,
                height=6,
                corner_radius=3,
                fg_color=RULE,
                progress_color=TEXT,
                width=240,
            )
            bar.set(pct.get(col, 0) / 100.0)
            bar.grid(row=row, column=1, padx=12, pady=6, sticky="w")

            ctk.CTkLabel(
                grid,
                text=f"{pct.get(col, 0):3.0f}th pct",
                font=self.f_readout,
                text_color=TEXT,
                width=90,
                anchor="e",
            ).grid(row=row, column=2, pady=6)

            ctk.CTkLabel(
                grid,
                text=f"weight {importance[col] * 100:4.1f}%",
                font=self.f_small,
                text_color=MUTED,
                width=100,
                anchor="e",
            ).grid(row=row, column=3, padx=(12, 0), pady=6)

    # -- animation --------------------------------------------------------

    def _show_gif(self, state: str) -> None:
        if state == self._gif_state:
            return
        self._gif_state = state

        frames = self._load_gif(STATE_GIFS.get(state, ""))
        if self._gif_job is not None:
            self.after_cancel(self._gif_job)
            self._gif_job = None

        if not frames:
            self.gif_label.configure(image="")
            self.gif_label.image = None
            return

        self._animate(frames, 0)

    def _load_gif(self, name: str) -> list:
        if not name:
            return []
        if name in self._gif_cache:
            return self._gif_cache[name]

        path = ASSET_DIR / name
        if not path.exists():
            self._gif_cache[name] = []
            return []

        frames = []
        gif = Image.open(path)
        try:
            while True:
                frame = gif.copy().convert("RGBA")
                frame.thumbnail((190, 190))
                frames.append(ImageTk.PhotoImage(frame))
                gif.seek(len(frames))
        except EOFError:
            pass

        self._gif_cache[name] = frames
        return frames

    def _animate(self, frames: list, index: int) -> None:
        frame = frames[index % len(frames)]
        self.gif_label.configure(image=frame)
        self.gif_label.image = frame
        self._gif_job = self.after(90, self._animate, frames, index + 1)


if __name__ == "__main__":
    WeatherApp().mainloop()
