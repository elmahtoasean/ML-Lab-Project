"""
SVM fitness classifier — GUI.

gui/svm_app.py

Window behaviour:
  * a normal resizable window with the usual minimise, maximise and close
    buttons from the system title bar
  * every panel sits on a weighted grid, so the plot grows with the window
  * F11 toggles maximised, Escape restores, and pending timers are cancelled
    on close so shutting down never leaves a stray traceback

What the screen shows:
  * the dividing line the SVM learned, with its margin either side
  * where this reading falls, and how far it sits from that line
  * the weight at which the answer would change, for this height
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path

import customtkinter as ctk
import joblib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets"
MODEL_DIR = BASE_DIR.parent / "saved_models"
DATASET = BASE_DIR.parent / "datasets" / "obesity_dataset.csv"


# --------------------------------------------------------------------------
# Design tokens — same family as the other apps in the project
# --------------------------------------------------------------------------

INK = "#FFFFFF"        # page ground and field fill
PANEL = "#F6F9FD"      # panel surfaces
RULE = "#CBDCEF"       # hairlines, tracks, grid
TEXT = "#0A3466"       # primary blue type
MUTED = "#5E87B3"      # secondary blue type
MARK = "#0A2A5E"       # the analysed point

CLASS_INKS = {"Fit": "#0E8C8C", "Obese": "#5B54C9"}
CLASS_GIFS = {"Fit": "fit.gif", "Obese": "obese.gif"}

HEIGHT_RANGE = (140.0, 205.0)
WEIGHT_RANGE = (30.0, 160.0)


def label_name(raw) -> str:
    """The saved model stores 1 and -1; the project calls them Fit and Obese."""
    return "Obese" if int(raw) == -1 else "Fit"


# --------------------------------------------------------------------------
# Model bundle
# --------------------------------------------------------------------------

class SvmBundle:
    """Wraps the saved SVM, its scaler, and the population it was trained on."""

    def __init__(self) -> None:
        self.model = joblib.load(MODEL_DIR / "svm_model.pkl")
        self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        try:
            self.metadata = joblib.load(MODEL_DIR / "svm_metadata.pkl")
        except FileNotFoundError:
            self.metadata = {}

        self.kernel = self.metadata.get("kernel", getattr(self.model, "kernel", "rbf"))
        self.c_value = self.metadata.get("C", getattr(self.model, "C", 1.0))

        self.xy: np.ndarray | None = None      # sample of the population
        self.labels: np.ndarray | None = None
        self.grid: tuple | None = None         # (hh, ww, decision values)

    # -- preparation ------------------------------------------------------

    def load_population(self, sample: int = 2200) -> None:
        df = pd.read_csv(DATASET, usecols=["height_cm", "weight_kg", "label"])
        rng = np.random.default_rng(42)
        idx = rng.choice(len(df), size=min(sample, len(df)), replace=False)
        self.xy = df[["height_cm", "weight_kg"]].to_numpy(float)[idx]
        self.labels = np.array([label_name(v) for v in df["label"].to_numpy()[idx]])

    def build_grid(self, steps: int = 180) -> None:
        hs = np.linspace(*HEIGHT_RANGE, steps)
        ws = np.linspace(*WEIGHT_RANGE, steps)
        hh, ww = np.meshgrid(hs, ws)
        flat = pd.DataFrame({"height_cm": hh.ravel(), "weight_kg": ww.ravel()})
        zz = self.model.decision_function(self.scaler.transform(flat))
        self.grid = (hh, ww, zz.reshape(hh.shape))

    # -- inference --------------------------------------------------------

    def frame(self, height: float, weight: float) -> pd.DataFrame:
        return pd.DataFrame([{"height_cm": height, "weight_kg": weight}])

    def analyse(self, height: float, weight: float) -> dict:
        scaled = self.scaler.transform(self.frame(height, weight))
        verdict = label_name(self.model.predict(scaled)[0])
        score = float(self.model.decision_function(scaled)[0])

        return {
            "verdict": verdict,
            "score": score,
            "margins": abs(score),
            "bmi": weight / (height / 100.0) ** 2,
        }

    def line_weight(self, height: float, weight: float) -> float | None:
        """The dividing line nearest this reading, at this height.

        The boundary can cross a vertical slice more than once, so the
        crossing closest to the current weight is the honest one to quote.
        """
        grid = np.linspace(*WEIGHT_RANGE, 400)
        block = pd.DataFrame({"height_cm": np.full(grid.shape, height),
                              "weight_kg": grid})
        scores = self.model.decision_function(self.scaler.transform(block))

        crossings = np.flatnonzero(np.diff(np.sign(scores)) != 0)
        if crossings.size == 0:
            return None

        i = int(crossings[np.argmin(np.abs(grid[crossings] - weight))])
        lo, hi = grid[i], grid[i + 1]
        a, b = scores[i], scores[i + 1]
        return float(lo + (hi - lo) * abs(a) / (abs(a) + abs(b)))


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

ctk.set_appearance_mode("light")


class SvmApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Fitness classifier — support vector machine")

        # A normal desktop window: minimise, maximise and close all work.
        self.resizable(True, True)
        self.minsize(1040, 700)
        self._centre(1240, 820)
        self.configure(fg_color=INK)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind("<F11>", self.toggle_maximise)
        self.bind("<Escape>", lambda _e: self.state("normal"))

        self.sv = SvmBundle()
        self._redraw_job: str | None = None
        self._gif_job: str | None = None
        self._gif_cache: dict[str, list] = {}
        self._gif_state: str | None = None
        self._overlay: list = []
        self._ready = False

        self.f_display = ctk.CTkFont("Segoe UI", 30, "bold")
        self.f_verdict = ctk.CTkFont("Segoe UI", 34, "bold")
        self.f_title = ctk.CTkFont("Segoe UI", 15, "bold")
        self.f_body = ctk.CTkFont("Segoe UI", 13)
        self.f_small = ctk.CTkFont("Segoe UI", 12)
        self.f_readout = ctk.CTkFont("Consolas", 15)

        self.height_var = ctk.DoubleVar(value=172.0)
        self.weight_var = ctk.DoubleVar(value=74.0)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_rail()
        self._build_stage()

        self.after(60, self._start_loading)

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

        ctk.CTkLabel(bar, text="Fitness classifier", font=self.f_display,
                     text_color=MARK).grid(row=0, column=0, padx=(28, 16),
                                           pady=(16, 4), sticky="w")

        note = f"Support vector machine, {self.sv.kernel} kernel"
        accuracy = self.sv.metadata.get("accuracy")
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
        rail.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(rail, text="Measurements", font=self.f_title,
                     text_color=MARK).grid(row=0, column=0, padx=24,
                                           pady=(22, 10), sticky="w")

        self.height_entry = self._measure_row(rail, 1, "Height", "cm",
                                              self.height_var, HEIGHT_RANGE)
        self.weight_entry = self._measure_row(rail, 2, "Weight", "kg",
                                              self.weight_var, WEIGHT_RANGE)

        self.bmi_line = ctk.CTkLabel(rail, text="", font=self.f_small,
                                     text_color=MUTED, justify="left")
        self.bmi_line.grid(row=3, column=0, padx=24, pady=(4, 16), sticky="w")

        ctk.CTkFrame(rail, fg_color=RULE, height=1).grid(row=4, column=0,
                                                         padx=24, sticky="ew")

        self.gif_label = tk.Label(rail, bg=PANEL, bd=0, highlightthickness=0)
        self.gif_label.grid(row=5, column=0, pady=10)

        ctk.CTkLabel(
            rail,
            text="The model draws one dividing line through\n"
                 "height and weight. This is coursework output,\n"
                 "not a health assessment.",
            font=self.f_small, text_color=MUTED, justify="left",
        ).grid(row=6, column=0, padx=24, pady=(0, 20), sticky="sw")

    def _measure_row(self, parent, row, label, unit, var, limits) -> ctk.CTkEntry:
        holder = ctk.CTkFrame(parent, fg_color="transparent")
        holder.grid(row=row, column=0, padx=24, pady=(4, 10), sticky="ew")
        holder.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(holder, text=label, font=self.f_body,
                     text_color=TEXT).grid(row=0, column=0, sticky="w")

        entry = ctk.CTkEntry(holder, width=92, height=32, font=self.f_readout,
                             justify="right", fg_color=INK, border_color=RULE,
                             text_color=MARK, corner_radius=6)
        entry.grid(row=0, column=1, sticky="e")
        entry.insert(0, f"{var.get():.1f}")
        entry.bind("<Return>", lambda _e, v=var, e=entry, l=limits: self._commit(v, e, l))
        entry.bind("<FocusOut>", lambda _e, v=var, e=entry, l=limits: self._commit(v, e, l))

        ctk.CTkLabel(holder, text=unit, font=self.f_small,
                     text_color=MUTED).grid(row=0, column=2, padx=(6, 0), sticky="e")

        ctk.CTkSlider(holder, from_=limits[0], to=limits[1], variable=var,
                      height=16, button_color=MARK, button_hover_color=TEXT,
                      progress_color=TEXT, fg_color=RULE,
                      command=lambda _v, e=entry, vv=var: self._on_slide(vv, e),
                      ).grid(row=1, column=0, columnspan=3, pady=(10, 0), sticky="ew")
        return entry

    def _build_stage(self) -> None:
        stage = ctk.CTkFrame(self, fg_color=INK, corner_radius=0)
        stage.grid(row=1, column=1, sticky="nsew")
        stage.grid_columnconfigure(0, weight=1)
        stage.grid_rowconfigure(1, weight=1)

        band = ctk.CTkFrame(stage, fg_color=PANEL, corner_radius=8,
                            border_width=1, border_color=RULE)
        band.grid(row=0, column=0, padx=18, pady=(18, 12), sticky="ew")
        band.grid_columnconfigure(1, weight=1)

        left = ctk.CTkFrame(band, fg_color="transparent")
        left.grid(row=0, column=0, padx=22, pady=16, sticky="w")

        self.verdict = ctk.CTkLabel(left, text="Loading the model…",
                                    font=self.f_verdict, text_color=MARK)
        self.verdict.pack(anchor="w")

        self.verdict_note = ctk.CTkLabel(left, text="", font=self.f_small,
                                         text_color=MUTED, justify="left")
        self.verdict_note.pack(anchor="w", pady=(4, 0))

        meter = ctk.CTkFrame(band, fg_color="transparent")
        meter.grid(row=0, column=1, padx=(10, 24), pady=16, sticky="e")

        ctk.CTkLabel(meter, text="Distance from the line", font=self.f_small,
                     text_color=MUTED).pack(anchor="e")
        self.meter_bar = ctk.CTkProgressBar(meter, height=8, corner_radius=4,
                                            fg_color=RULE, progress_color=TEXT,
                                            width=220)
        self.meter_bar.set(0)
        self.meter_bar.pack(pady=(6, 4))
        self.meter_text = ctk.CTkLabel(meter, text="", font=self.f_small,
                                       text_color=TEXT)
        self.meter_text.pack(anchor="e")

        plot_panel = ctk.CTkFrame(stage, fg_color=PANEL, corner_radius=8,
                                  border_width=1, border_color=RULE)
        plot_panel.grid(row=1, column=0, padx=18, pady=(0, 18), sticky="nsew")
        plot_panel.grid_columnconfigure(0, weight=1)
        plot_panel.grid_rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(7.0, 4.6), dpi=100, facecolor=PANEL)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, plot_panel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew",
                                         padx=8, pady=8)

    # -- loading ----------------------------------------------------------

    def _start_loading(self) -> None:
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self) -> None:
        try:
            self.sv.load_population()
            self.sv.build_grid()
        except Exception as exc:
            self.after(0, lambda: self.verdict_note.configure(
                text=f"Could not prepare the model: {exc}"))
            return
        self.after(0, self._on_ready)

    def _on_ready(self) -> None:
        # The readout must come up even if the plot fails, otherwise the
        # band is stuck on "Loading the model…" with no explanation.
        try:
            self._draw_base_plot()
        except Exception as exc:
            self.verdict_note.configure(text=f"The plot could not be drawn: {exc}")
        finally:
            self._ready = True
            self.refresh()

    # -- input handling ---------------------------------------------------

    def _on_slide(self, var, entry) -> None:
        entry.delete(0, "end")
        entry.insert(0, f"{var.get():.1f}")
        self.schedule_refresh()

    def _commit(self, var, entry, limits) -> None:
        raw = entry.get().strip().replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            entry.delete(0, "end")
            entry.insert(0, f"{var.get():.1f}")
            self.verdict_note.configure(
                text=f"'{raw}' is not a number. Enter a value like 172.5.")
            return
        value = min(max(value, limits[0]), limits[1])
        var.set(value)
        entry.delete(0, "end")
        entry.insert(0, f"{value:.1f}")
        self.schedule_refresh()

    def schedule_refresh(self) -> None:
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(100, self.refresh)

    # -- output -----------------------------------------------------------

    def refresh(self) -> None:
        self._redraw_job = None
        if not self._ready:
            return

        h, w = self.height_var.get(), self.weight_var.get()
        result = self.sv.analyse(h, w)
        verdict = result["verdict"]
        ink = CLASS_INKS.get(verdict, TEXT)
        margins = result["margins"]

        self.bmi_line.configure(text=f"This input works out to a BMI of {result['bmi']:.1f}")
        self.verdict.configure(text=verdict, text_color=ink)

        if margins < 0.5:
            wording = "right up against the dividing line, so the answer is borderline"
        elif margins < 1.0:
            wording = "inside the margin, the strip the model treats as uncertain"
        elif margins < 2.0:
            wording = "clearly on the " + verdict.lower() + " side of the line"
        else:
            wording = "well clear of the line"

        line_w = self.sv.line_weight(h, w)
        if line_w is None:
            second = f"At {h:.0f} cm every weight in range stays on one side."
        else:
            gap = line_w - w
            direction = "more" if gap > 0 else "less"
            second = (f"At {h:.0f} cm the line sits at {line_w:.1f} kg, "
                      f"{abs(gap):.1f} kg {direction} than this reading.")

        self.verdict_note.configure(text=f"This reading sits {wording}.\n{second}")

        self.meter_bar.set(min(margins / 3.0, 1.0))
        self.meter_bar.configure(progress_color=ink)
        self.meter_text.configure(text=f"{margins:.2f} margins", text_color=ink)

        self._draw_overlay(h, w, result, line_w)
        self._show_gif(verdict)

    # -- plot -------------------------------------------------------------

    def _draw_base_plot(self) -> None:
        ax = self.ax
        ax.clear()
        ax.set_facecolor(PANEL)

        hh, ww, zz = self.sv.grid

        # Two sides, then the line itself and the margin either side of it.
        ax.contourf(hh, ww, zz, levels=[zz.min() - 1, 0, zz.max() + 1],
                    colors=[CLASS_INKS["Obese"], CLASS_INKS["Fit"]], alpha=0.10)
        ax.contour(hh, ww, zz, levels=[-1, 1], colors=[MUTED],
                   linewidths=0.9, linestyles="dashed")
        ax.contour(hh, ww, zz, levels=[0], colors=[MARK], linewidths=1.6)

        for name in ("Obese", "Fit"):
            mask = self.sv.labels == name
            ax.scatter(self.sv.xy[mask, 0], self.sv.xy[mask, 1], s=7,
                       c=CLASS_INKS[name], alpha=0.30, linewidths=0, label=name)

        ax.set_xlim(*HEIGHT_RANGE)
        ax.set_ylim(*WEIGHT_RANGE)
        ax.set_xlabel("Height (cm)", color=TEXT, fontsize=10)
        ax.set_ylabel("Weight (kg)", color=TEXT, fontsize=10)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(color=RULE, linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(RULE)

        legend = ax.legend(loc="upper left", frameon=True, fontsize=9)
        legend.get_frame().set_facecolor(INK)
        legend.get_frame().set_edgecolor(RULE)
        for text in legend.get_texts():
            text.set_color(TEXT)

        self.figure.tight_layout()
        self._overlay = []
        self.canvas.draw()

    def _draw_overlay(self, h: float, w: float, result: dict,
                      line_w: float | None) -> None:
        for artist in self._overlay:
            artist.remove()
        self._overlay = []

        ax = self.ax
        ink = CLASS_INKS.get(result["verdict"], TEXT)

        # The gap between this reading and the line, straight down or up.
        if line_w is not None:
            gap, = ax.plot([h, h], [w, line_w], color=ink, lw=1.3,
                           ls=(0, (2, 3)), alpha=0.9, zorder=7)
            self._overlay.append(gap)

            foot = ax.scatter([h], [line_w], s=46, facecolors="none",
                              edgecolors=MARK, linewidths=1.2, zorder=8)
            self._overlay.append(foot)

        dot = ax.scatter([h], [w], s=160, marker="X", c=MARK,
                         edgecolors=INK, linewidths=1.4, zorder=9)
        self._overlay.append(dot)

        tag = ax.annotate(f"{h:.1f} cm, {w:.1f} kg", xy=(h, w), xytext=(10, 12),
                          textcoords="offset points", color=INK, fontsize=9.5,
                          fontweight="bold", zorder=10,
                          bbox=dict(boxstyle="round,pad=0.32", fc=MARK, ec="none"))
        self._overlay.append(tag)

        self.canvas.draw_idle()

    # -- animation --------------------------------------------------------

    def _show_gif(self, verdict: str) -> None:
        if verdict == self._gif_state:
            return
        self._gif_state = verdict

        if self._gif_job is not None:
            self.after_cancel(self._gif_job)
            self._gif_job = None

        frames = self._load_gif(CLASS_GIFS.get(verdict, ""))
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

        frames: list = []
        gif = Image.open(path)
        try:
            while True:
                frame = gif.copy().convert("RGBA")
                frame.thumbnail((220, 180))
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
    SvmApp().mainloop()