from __future__ import annotations
import tkinter as tk
from itertools import product
from pathlib import Path
import customtkinter as ctk
import joblib
import numpy as np
from PIL import Image, ImageTk

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets"
MODEL_DIR = BASE_DIR.parent / "saved_models"

# One source of truth: reuse the preprocessing constants when the app is
# started as a module, and fall back to a local copy if it is run directly.
try:
    from preprocessing.fruit_preprocessing import (
        FRUIT_NAMES,
        LABEL_MAP,
        VALID_PATTERNS,
    )
except ImportError:  # only hit when the file is run outside the package
    LABEL_MAP = {"p1": 0, "p2": 1, "p3": 2, "p4": 3}
    FRUIT_NAMES = {"p1": "Watermelon", "p2": "Banana", "p3": "Orange", "p4": "Apple"}
    VALID_PATTERNS = {
        (1, 1, 1, 0),    # Watermelon
        (-1, 1, -1, 1),  # Banana
        (1, -1, -1, 2),  # Orange
        (1, 1, -1, 3),   # Apple
    }


# --------------------------------------------------------------------------
# Design tokens — same family as the other apps in the project
# --------------------------------------------------------------------------

INK = "#FFFFFF"        # page ground and field fill
PANEL = "#F6F9FD"      # panel surfaces
PANEL_HI = "#E9F1FA"   # raised rows
RULE = "#CBDCEF"       # hairlines and tracks
TEXT = "#0A3466"       # primary blue type
MUTED = "#5E87B3"      # secondary blue type
MARK = "#0A2A5E"       # emphasis

FRUIT_INKS = {
    "Watermelon": "#2E8B45",
    "Banana": "#5B54C9",
    "Orange": "#0E8C8C",
    "Apple": "#1F6FBF",
}

SHAPE_CHOICES = {"Round": 1, "Elliptical": -1}
TEXTURE_CHOICES = {"Smooth": 1, "Rough": -1}
WEIGHT_THRESHOLD = 1.0  # pounds

# Weight sensor encoding, matching the dataset:
#   under one pound      -> -1
#   one pound or heavier -> +1
def encode_weight(pounds: float) -> int:
    return -1 if pounds < WEIGHT_THRESHOLD else 1


def code_to_fruit(code: int) -> str:
    """Turn the numeric label the model emits back into a fruit name."""
    for key, value in LABEL_MAP.items():
        if value == int(code):
            return FRUIT_NAMES[key]
    return str(code)


# --------------------------------------------------------------------------
# Model bundle
# --------------------------------------------------------------------------

class SorterBundle:
    """Wraps the saved perceptron and everything the conveyer can send it."""

    def __init__(self) -> None:
        self.model = joblib.load(MODEL_DIR / "fruit_perceptron_model.pkl")
        try:
            self.metadata = joblib.load(MODEL_DIR / "fruit_perceptron_metadata.pkl")
        except FileNotFoundError:
            self.metadata = {}

        self.fruits = [code_to_fruit(c) for c in self.model.classes_]

        # The sensor vector each catalogued fruit produces.
        self.catalogue = {
            (shape, texture, weight): code_to_fruit(label)
            for shape, texture, weight, label in VALID_PATTERNS
        }

        # Three sensors, each +1 or -1, so the machine has exactly eight
        # possible readings. Work all of them out once.
        self.table = []
        for reading in product((1, -1), repeat=3):
            result = self.analyse(reading)
            result["reading"] = reading
            result["listed"] = self.catalogue.get(reading)
            self.table.append(result)

        self.max_score = max(
            max(abs(v) for v in row["scores"].values()) for row in self.table
        ) or 1.0

    def analyse(self, reading: tuple[int, int, int]) -> dict:
        # The model was fitted on a plain numpy array, so it expects one.
        sample = np.array([reading], dtype=int)
        raw = self.model.decision_function(sample)[0]
        scores = {code_to_fruit(c): float(s) for c, s in zip(self.model.classes_, raw)}

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "verdict": code_to_fruit(self.model.predict(sample)[0]),
            "scores": scores,
            "runner_up": ranked[1][0],
            "margin": float(ranked[0][1] - ranked[1][1]),
        }


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

ctk.set_appearance_mode("light")


class SorterApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Fruit sorter — perceptron")

        # A normal desktop window: minimise, maximise and close all work.
        self.resizable(True, True)
        self.minsize(1060, 700)
        self._centre(1240, 820)
        self.configure(fg_color=INK)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind("<F11>", self.toggle_maximise)
        self.bind("<Escape>", lambda _e: self.state("normal"))

        self.sb = SorterBundle()
        self._redraw_job: str | None = None
        self._gif_job: str | None = None
        self._gif_cache: dict[str, list] = {}
        self._gif_state: str | None = None

        self.f_display = ctk.CTkFont("Segoe UI", 30, "bold")
        self.f_verdict = ctk.CTkFont("Segoe UI", 34, "bold")
        self.f_title = ctk.CTkFont("Segoe UI", 15, "bold")
        self.f_body = ctk.CTkFont("Segoe UI", 13)
        self.f_small = ctk.CTkFont("Segoe UI", 12)
        self.f_tiny = ctk.CTkFont("Segoe UI", 11)
        self.f_readout = ctk.CTkFont("Consolas", 15)

        self.shape_var = ctk.StringVar(value="Round")
        self.texture_var = ctk.StringVar(value="Smooth")
        self.weight_var = ctk.StringVar(value="0.4")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_rail()
        self._build_stage()

        self.weight_var.trace_add("write", lambda *_: self.schedule_refresh())
        self.after(80, self.refresh)

    # -- window -----------------------------------------------------------

    def _centre(self, width: int, height: int) -> None:
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        width = min(width, screen_w - 80)
        height = min(height, screen_h - 120)
        x = (screen_w - width) // 2
        y = max((screen_h - height) // 3, 0)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def toggle_maximise(self, _event=None) -> None:
        self.state("normal" if self.state() == "zoomed" else "zoomed")

    def on_close(self) -> None:
        for job in (self._redraw_job, self._gif_job):
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
        self.destroy()

    # -- chrome -----------------------------------------------------------

    def _build_header(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=INK, corner_radius=0, height=74)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(bar, text="Fruit sorter", font=self.f_display,
                     text_color=MARK).grid(row=0, column=0, padx=(28, 16),
                                           pady=(16, 4), sticky="w")

        note = "Perceptron, three sensors, four storage bins"
        accuracy = self.sb.metadata.get("accuracy")
        if accuracy:
            note += f", {accuracy * 100:.1f}% accurate on the held-out test set"
        ctk.CTkLabel(bar, text=note, font=self.f_small, text_color=MUTED).grid(
            row=0, column=1, sticky="w", pady=(24, 0))

        ctk.CTkFrame(self, fg_color=RULE, height=1, corner_radius=0).grid(
            row=0, column=0, columnspan=2, sticky="sew")

    def _build_rail(self) -> None:
        rail = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, width=320)
        rail.grid(row=1, column=0, sticky="nsw")
        rail.grid_propagate(False)
        rail.grid_rowconfigure(7, weight=1)

        ctk.CTkLabel(rail, text="Sensor readings", font=self.f_title,
                     text_color=MARK).grid(row=0, column=0, padx=24,
                                           pady=(22, 12), sticky="w")

        self._dropdown(rail, 1, "Shape", self.shape_var, list(SHAPE_CHOICES))
        self._dropdown(rail, 2, "Texture", self.texture_var, list(TEXTURE_CHOICES))

        holder = ctk.CTkFrame(rail, fg_color="transparent")
        holder.grid(row=3, column=0, padx=24, pady=(4, 4), sticky="ew")
        holder.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(holder, text="Weight", font=self.f_body,
                     text_color=TEXT).grid(row=0, column=0, sticky="w")
        self.weight_entry = ctk.CTkEntry(
            holder, textvariable=self.weight_var, width=96, height=34,
            font=self.f_readout, justify="right", fg_color=INK,
            border_color=RULE, text_color=MARK, corner_radius=6)
        self.weight_entry.grid(row=0, column=1, sticky="e")
        ctk.CTkLabel(holder, text="lb", font=self.f_small, text_color=MUTED
                     ).grid(row=0, column=2, padx=(6, 0), sticky="e")

        self.weight_hint = ctk.CTkLabel(
            rail, text="", font=self.f_small, text_color=MUTED,
            justify="left", wraplength=260)
        self.weight_hint.grid(row=4, column=0, padx=24, pady=(4, 12), sticky="w")

        ctk.CTkFrame(rail, fg_color=RULE, height=1).grid(row=5, column=0,
                                                         padx=24, sticky="ew")

        vector_row = ctk.CTkFrame(rail, fg_color="transparent")
        vector_row.grid(row=6, column=0, padx=24, pady=(14, 2), sticky="ew")
        vector_row.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(vector_row, text="Sensor vector p", font=self.f_small,
                     text_color=MUTED).grid(row=0, column=0, sticky="w")
        self.vector_label = ctk.CTkLabel(vector_row, text="", font=self.f_readout,
                                         text_color=MARK)
        self.vector_label.grid(row=0, column=1, sticky="e")

        self.gif_label = tk.Label(rail, bg=PANEL, bd=0, highlightthickness=0)
        self.gif_label.grid(row=7, column=0, pady=12)

    def _dropdown(self, parent, row, label, variable, values) -> None:
        holder = ctk.CTkFrame(parent, fg_color="transparent")
        holder.grid(row=row, column=0, padx=24, pady=(4, 10), sticky="ew")
        holder.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(holder, text=label, font=self.f_body,
                     text_color=TEXT).grid(row=0, column=0, sticky="w")

        ctk.CTkOptionMenu(
            holder, variable=variable, values=values, width=150, height=34,
            font=self.f_body, corner_radius=6,
            fg_color=INK, button_color=RULE, button_hover_color=PANEL_HI,
            text_color=MARK, dropdown_fg_color=INK,
            dropdown_text_color=TEXT, dropdown_hover_color=PANEL_HI,
            command=lambda _v: self.refresh(),
        ).grid(row=0, column=1, sticky="e")

    def _build_stage(self) -> None:
        stage = ctk.CTkFrame(self, fg_color=INK, corner_radius=0)
        stage.grid(row=1, column=1, sticky="nsew")
        stage.grid_columnconfigure(0, weight=1)
        stage.grid_rowconfigure(2, weight=1)

        # --- warning strip -------------------------------------------------
        # Built once and taken out of the layout with grid_remove() until
        # there is something to say, so nothing jumps when it appears.
        self.warning = ctk.CTkFrame(stage, fg_color=MARK, corner_radius=8)
        self.warning.grid(row=0, column=0, padx=18, pady=(18, 0), sticky="ew")
        self.warning.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.warning, text="!", font=self.f_title,
                     text_color=INK, width=18).grid(row=0, column=0,
                                                    padx=(18, 10), pady=12)
        self.warning_text = ctk.CTkLabel(self.warning, text="", font=self.f_body,
                                         text_color=INK, justify="left",
                                         anchor="w")
        self.warning_text.grid(row=0, column=1, padx=(0, 18), pady=12, sticky="w")
        self.warning.grid_remove()

        # --- verdict ------------------------------------------------------
        band = ctk.CTkFrame(stage, fg_color=PANEL, corner_radius=8,
                            border_width=1, border_color=RULE)
        band.grid(row=1, column=0, padx=18, pady=(12, 12), sticky="ew")
        band.grid_columnconfigure(0, weight=1)

        inner = ctk.CTkFrame(band, fg_color="transparent")
        inner.grid(row=0, column=0, padx=22, pady=16, sticky="w")

        self.verdict = ctk.CTkLabel(inner, text="", font=self.f_verdict,
                                    text_color=MARK)
        self.verdict.pack(anchor="w")

        self.verdict_note = ctk.CTkLabel(inner, text="", font=self.f_small,
                                         text_color=MUTED, justify="left")
        self.verdict_note.pack(anchor="w", pady=(4, 0))

        # --- one bar per bin ---------------------------------------------
        scores = ctk.CTkFrame(stage, fg_color=PANEL, corner_radius=8,
                              border_width=1, border_color=RULE)
        scores.grid(row=2, column=0, padx=18, pady=(0, 12), sticky="nsew")
        scores.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(scores, text="How strongly each bin was argued for",
                     font=self.f_title, text_color=MARK).grid(
            row=0, column=0, padx=20, pady=(14, 10), sticky="w")

        grid = ctk.CTkFrame(scores, fg_color="transparent")
        grid.grid(row=1, column=0, padx=20, pady=(0, 16), sticky="new")
        grid.grid_columnconfigure(2, weight=1)

        self.score_rows = []
        for row, fruit in enumerate(self.sb.fruits):
            ink = FRUIT_INKS.get(fruit, TEXT)

            ctk.CTkLabel(grid, text="", width=10, height=10, fg_color=ink,
                         corner_radius=2).grid(row=row, column=0,
                                               padx=(0, 10), pady=5)
            name = ctk.CTkLabel(grid, text=fruit, font=self.f_body,
                                text_color=MUTED, width=96, anchor="w")
            name.grid(row=row, column=1, sticky="w", pady=5)

            bar = ctk.CTkProgressBar(grid, height=7, corner_radius=4,
                                     fg_color=RULE, progress_color=ink)
            bar.set(0.5)
            bar.grid(row=row, column=2, padx=16, pady=5, sticky="ew")

            value = ctk.CTkLabel(grid, text="", font=self.f_readout,
                                 text_color=MUTED, width=64, anchor="e")
            value.grid(row=row, column=3, pady=5)

            self.score_rows.append((fruit, name, bar, value))

        # --- every reading the sensors can produce ------------------------
        strip = ctk.CTkFrame(stage, fg_color=PANEL, corner_radius=8,
                             border_width=1, border_color=RULE)
        strip.grid(row=3, column=0, padx=18, pady=(0, 18), sticky="ew")

        ctk.CTkLabel(strip, text="Every reading the sensors can produce",
                     font=self.f_title, text_color=MARK).pack(
            anchor="w", padx=20, pady=(14, 2))
        ctk.CTkLabel(strip, text="Eight in total. Only four match a fruit; "
                                 "the rest are invalid input.",
                     font=self.f_small, text_color=MUTED).pack(
            anchor="w", padx=20, pady=(0, 10))

        tiles = ctk.CTkFrame(strip, fg_color="transparent")
        tiles.pack(fill="x", padx=16, pady=(0, 16))

        self.tiles = []
        for i, row in enumerate(self.sb.table):
            tiles.grid_columnconfigure(i, weight=1)
            fruit = row["verdict"]
            ink = FRUIT_INKS.get(fruit, TEXT)

            tile = ctk.CTkFrame(tiles, fg_color=INK, corner_radius=6,
                                border_width=1, border_color=RULE)
            tile.grid(row=0, column=i, padx=4, sticky="ew")

            shape, texture, weight = row["reading"]
            ctk.CTkLabel(tile, text=f"[{shape:+d} {texture:+d} {weight:+d}]",
                         font=self.f_readout, text_color=MARK).pack(pady=(10, 4))
            ctk.CTkLabel(tile,
                         text=row["listed"] if row["listed"] else "Invalid input",
                         font=self.f_small,
                         text_color=ink if row["listed"] else MUTED).pack(pady=(0, 10))

            self.tiles.append(tile)

    # -- input handling ---------------------------------------------------

    def schedule_refresh(self) -> None:
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(220, self.refresh)

    def read_weight(self) -> tuple[float | None, str]:
        """Return the weight in pounds, or None plus the reason it failed."""
        raw = self.weight_var.get().strip().replace(",", ".")
        if not raw:
            return None, "Enter the weight in pounds, for example 0.4."
        try:
            value = float(raw)
        except ValueError:
            return None, f"'{raw}' is not a number. Enter a weight such as 0.4."
        if value < 0:
            return None, "Weight cannot be negative. Enter a positive number of pounds."
        if value == 0:
            return None, "Weight must be greater than 0 pounds."
        return value, ""

    def show_warning(self, message: str | None) -> None:
        if message:
            self.warning_text.configure(text=message)
            self.warning.grid()
        else:
            self.warning.grid_remove()

    # -- output -----------------------------------------------------------

    def refresh(self) -> None:
        self._redraw_job = None

        pounds, problem = self.read_weight()
        if pounds is None:
            self.show_warning(problem)
            self.verdict.configure(text="Invalid input", text_color=MARK)
            self.verdict_note.configure(text="")
            self.weight_entry.configure(border_color=MARK)
            self.weight_hint.configure(text="")
            self.vector_label.configure(text="")
            self._clear_readout()
            return

        self.weight_entry.configure(border_color=RULE)

        shape = SHAPE_CHOICES[self.shape_var.get()]
        texture = TEXTURE_CHOICES[self.texture_var.get()]
        weight = encode_weight(pounds)
        reading = (shape, texture, weight)

        self.weight_hint.configure(
            text=f"The sensor reports -1 under one pound and +1 from one "
                 f"pound up, so {pounds:g} lb reads as {weight:+d}.")
        self.vector_label.configure(
            text=f"[{shape:+d} {texture:+d} {weight:+d}]")

        result = self.sb.analyse(reading)
        fruit = result["verdict"]
        ink = FRUIT_INKS.get(fruit, TEXT)

        self.verdict.configure(text=fruit, text_color=ink)

        listed = self.sb.catalogue.get(reading)
        if listed is None:
            self.show_warning("Invalid input. These sensor readings match none "
                              "of the four fruits.")
            self.verdict.configure(text="Invalid input", text_color=MARK)
            self.verdict_note.configure(text="")
            self._clear_readout(highlight=reading)
            self._show_gif(None)
            return

        self.show_warning(None)

        lines = [f"Route to the {fruit.lower()} bin."]
        if result["margin"] <= 0.001:
            lines.append(f"It is a dead tie with {result['runner_up'].lower()}, "
                         f"settled only by bin order.")
        else:
            lines.append(f"Nearest rival is {result['runner_up'].lower()}, "
                         f"{result['margin']:.2f} behind.")
        self.verdict_note.configure(text="\n".join(lines))

        for name_key, name, bar, value in self.score_rows:
            score = result["scores"][name_key]
            bar.set(float(np.clip(0.5 + score / (2 * self.sb.max_score), 0, 1)))
            value.configure(text=f"{score:+.2f}")
            hot = name_key == fruit
            name.configure(text_color=MARK if hot else MUTED)
            value.configure(text_color=MARK if hot else MUTED)

        for tile, row in zip(self.tiles, self.sb.table):
            current = row["reading"] == reading
            tile.configure(
                border_color=FRUIT_INKS.get(row["verdict"], TEXT) if current else RULE,
                border_width=2 if current else 1,
                fg_color=PANEL_HI if current else INK)

        self._show_gif(fruit)

    # -- animation --------------------------------------------------------

    def _clear_readout(self, highlight: tuple | None = None) -> None:
        """Blank the bars and tile highlight when there is no valid answer."""
        for _key, name, bar, value in self.score_rows:
            bar.set(0.5)
            value.configure(text="", text_color=MUTED)
            name.configure(text_color=MUTED)
        for tile, row in zip(self.tiles, self.sb.table):
            current = highlight is not None and row["reading"] == highlight
            tile.configure(border_color=MARK if current else RULE,
                           border_width=2 if current else 1,
                           fg_color=PANEL_HI if current else INK)

    def _show_gif(self, fruit: str | None) -> None:
        if fruit == self._gif_state:
            return
        self._gif_state = fruit

        if fruit is None:
            if self._gif_job is not None:
                self.after_cancel(self._gif_job)
                self._gif_job = None
            self.gif_label.configure(image="")
            self.gif_label.image = None
            return

        if self._gif_job is not None:
            self.after_cancel(self._gif_job)
            self._gif_job = None

        frames = self._load_gif(f"{fruit.lower()}.gif")
        if not frames:
            self.gif_label.configure(image="")
            self.gif_label.image = None
            return
        self._animate(frames, 0)

    def _load_gif(self, name: str) -> list:
        if name in self._gif_cache:
            return self._gif_cache[name]

        path = ASSET_DIR / name
        if not path.exists():
            self._gif_cache[name] = []
            return []

        frames: list = []
        gif = Image.open(path)
        try:
            while True:
                frame = gif.copy().convert("RGBA")
                frame.thumbnail((200, 170))
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
    SorterApp().mainloop()