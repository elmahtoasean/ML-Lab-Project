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
from matplotlib.patches import Ellipse
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
PANEL_HI = "#E9F1FA"   # raised rows
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
class KnnBundle:
    """Wraps the saved KNN artefacts and the reference points it votes with."""

    def __init__(self) -> None:
        self.model = joblib.load(MODEL_DIR / "knn_model.pkl")
        self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        try:
            self.metadata = joblib.load(MODEL_DIR / "knn_metadata.pkl")
        except FileNotFoundError:
            self.metadata = {}

        self.k = int(getattr(self.model, "n_neighbors", self.metadata.get("best_k", 5)))
        self.classes = [label_name(c) for c in self.model.classes_]

        self.train_xy: np.ndarray | None = None    # original units
        self.train_label: np.ndarray | None = None  # "Fit" / "Obese"
        self.grid: tuple | None = None              # decision regions

    # -- reference points -------------------------------------------------

    def load_reference(self) -> None:
        """The exact rows the model votes with, back in centimetres and kilos."""
        fitted = getattr(self.model, "_fit_X", None)
        if fitted is not None:
            self.train_xy = self.scaler.inverse_transform(np.asarray(fitted))
            raw = self.model.classes_[self.model._y]
        else:  # fallback if the private attribute ever disappears
            df = pd.read_csv(DATASET, usecols=["height_cm", "weight_kg", "label"])
            self.train_xy = df[["height_cm", "weight_kg"]].to_numpy(dtype=float)
            raw = df["label"].to_numpy()
        self.train_label = np.array([label_name(v) for v in raw])

    def build_regions(self, steps: int = 180) -> None:
        """Predict across a grid once, so the boundary can be shaded."""
        hs = np.linspace(*HEIGHT_RANGE, steps)
        ws = np.linspace(*WEIGHT_RANGE, steps)
        hh, ww = np.meshgrid(hs, ws)
        flat = np.column_stack([hh.ravel(), ww.ravel()])
        frame = pd.DataFrame(flat, columns=["height_cm", "weight_kg"])
        preds = self.model.predict(self.scaler.transform(frame))
        zz = np.array([1 if label_name(p) == "Fit" else 0 for p in preds])
        self.grid = (hh, ww, zz.reshape(hh.shape))

    # -- inference --------------------------------------------------------

    def frame(self, height: float, weight: float) -> pd.DataFrame:
        return pd.DataFrame([{"height_cm": height, "weight_kg": weight}])

    def analyse(self, height: float, weight: float) -> dict:
        scaled = self.scaler.transform(self.frame(height, weight))
        distances, indices = self.model.kneighbors(scaled, n_neighbors=self.k)
        distances, indices = distances[0], indices[0]

        neigh_labels = self.train_label[indices]
        votes = {c: int((neigh_labels == c).sum()) for c in self.classes}
        verdict = label_name(self.model.predict(scaled)[0])

        return {
            "verdict": verdict,
            "votes": votes,
            "distances": distances,
            "points": self.train_xy[indices],
            "labels": neigh_labels,
            "radius": float(distances[-1]),
            "bmi": weight / (height / 100.0) ** 2,
        }

    def flip_weight(self, height: float, weight: float) -> dict | None:
        """The weight at which the verdict changes, holding height fixed."""
        grid = np.linspace(*WEIGHT_RANGE, 900)
        block = pd.DataFrame({"height_cm": np.full(grid.shape, height),
                              "weight_kg": grid})
        preds = np.array([label_name(p) for p in
                          self.model.predict(self.scaler.transform(block))])

        current = label_name(self.model.predict(
            self.scaler.transform(self.frame(height, weight)))[0])
        changed = np.flatnonzero(preds != current)
        if changed.size == 0:
            return None

        j = changed[np.argmin(np.abs(grid[changed] - weight))]
        return {"at": float(grid[j]), "delta": float(grid[j] - weight),
                "into": str(preds[j])}


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

ctk.set_appearance_mode("light")


class KnnApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Fitness classifier")

        # A normal desktop window: minimise, maximise and close all work,
        # and the layout reflows instead of clipping.
        self.resizable(True, True)
        self.minsize(1080, 720)
        self._centre(1280, 840)
        self.configure(fg_color=INK)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind("<F11>", self.toggle_maximise)
        self.bind("<Escape>", lambda _e: self.state("normal"))

        self.kb = KnnBundle()
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
        self.f_readout = ctk.CTkFont("Consolas", 14)

        self.height_var = ctk.DoubleVar(value=172.0)
        self.weight_var = ctk.DoubleVar(value=74.0)
        self.regions_var = ctk.BooleanVar(value=True)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_rail()
        self._build_stage()

        self.after(60, self._start_loading)

    def _centre(self, width: int, height: int) -> None:
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        width = min(width, screen_w - 80)
        height = min(height, screen_h - 120)
        self.geometry(f"{width}x{height}+{(screen_w - width) // 2}+{max((screen_h - height) // 3, 0)}")

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

        note = f"K-nearest neighbours, k = {self.kb.k}"
        accuracy = self.kb.metadata.get("accuracy")
        if accuracy:
            note += f", {accuracy * 100:.1f}% accurate on the held-out test set"
        self.model_note = ctk.CTkLabel(bar, text=note, font=self.f_small,
                                       text_color=MUTED)
        self.model_note.grid(row=0, column=1, sticky="w", pady=(24, 0))

        ctk.CTkFrame(self, fg_color=RULE, height=1, corner_radius=0).grid(
            row=0, column=0, columnspan=2, sticky="sew")

    def _build_rail(self) -> None:
        rail = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, width=330)
        rail.grid(row=1, column=0, sticky="nsw")
        rail.grid_propagate(False)
        rail.grid_rowconfigure(6, weight=1)

        ctk.CTkLabel(rail, text="Measurements", font=self.f_title,
                     text_color=MARK).grid(row=0, column=0, padx=24,
                                           pady=(22, 10), sticky="w")

        self.height_entry = self._measure_row(rail, 1, "Height", "cm",
                                              self.height_var, HEIGHT_RANGE)
        self.weight_entry = self._measure_row(rail, 2, "Weight", "kg",
                                              self.weight_var, WEIGHT_RANGE)

        self.bmi_line = ctk.CTkLabel(rail, text="", font=self.f_small,
                                     text_color=MUTED, justify="left")
        self.bmi_line.grid(row=3, column=0, padx=24, pady=(4, 14), sticky="w")

        ctk.CTkFrame(rail, fg_color=RULE, height=1).grid(row=4, column=0,
                                                         padx=24, sticky="ew")

        ctk.CTkCheckBox(
            rail, text="Show decision regions", variable=self.regions_var,
            font=self.f_small, text_color=TEXT, fg_color=TEXT,
            hover_color=MARK, border_color=RULE, checkbox_width=18,
            checkbox_height=18, command=self._toggle_regions,
        ).grid(row=5, column=0, padx=24, pady=16, sticky="w")

        self.gif_label = tk.Label(rail, bg=PANEL, bd=0, highlightthickness=0)
        self.gif_label.grid(row=6, column=0, pady=(4, 8))

        ctk.CTkLabel(
            rail,
            text="The model repeats the majority label of the\n"
                 f"{self.kb.k} closest records in the training data.\n"
                 "It is a coursework model, not a health assessment.",
            font=self.f_small, text_color=MUTED, justify="left",
        ).grid(row=7, column=0, padx=24, pady=(0, 20), sticky="sw")

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

        # --- verdict band ------------------------------------------------
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

        self.vote_holder = ctk.CTkFrame(band, fg_color="transparent")
        self.vote_holder.grid(row=0, column=1, padx=(10, 24), pady=16, sticky="e")
        self.vote_rows: list[tuple] = []

        # --- plot and neighbour list -------------------------------------
        content = ctk.CTkFrame(stage, fg_color=INK, corner_radius=0)
        content.grid(row=1, column=0, padx=18, pady=(0, 18), sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(0, weight=1)

        plot_panel = ctk.CTkFrame(content, fg_color=PANEL, corner_radius=8,
                                  border_width=1, border_color=RULE)
        plot_panel.grid(row=0, column=0, sticky="nsew")
        plot_panel.grid_columnconfigure(0, weight=1)
        plot_panel.grid_rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(6.6, 4.4), dpi=100, facecolor=PANEL)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, plot_panel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew",
                                         padx=8, pady=8)

    # -- loading ----------------------------------------------------------

    def _start_loading(self) -> None:
        self._build_vote_rows()
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self) -> None:
        try:
            self.kb.load_reference()
            self.kb.build_regions()
        except Exception as exc:
            self.after(0, lambda: self.verdict_note.configure(
                text=f"Could not prepare the model: {exc}"))
            return
        self.after(0, self._on_ready)

    def _on_ready(self) -> None:
        self._draw_base_plot()
        self._ready = True
        self.refresh()

    def _build_vote_rows(self) -> None:
        for row, name in enumerate(self.kb.classes):
            ink = CLASS_INKS.get(name, TEXT)

            ctk.CTkLabel(self.vote_holder, text="", width=10, height=10,
                         fg_color=ink, corner_radius=2).grid(row=row, column=0,
                                                             padx=(0, 10), pady=3)
            label = ctk.CTkLabel(self.vote_holder, text=name, font=self.f_body,
                                 text_color=MUTED, width=54, anchor="w")
            label.grid(row=row, column=1, sticky="w", pady=3)

            bar = ctk.CTkProgressBar(self.vote_holder, height=6, corner_radius=3,
                                     fg_color=RULE, progress_color=ink, width=200)
            bar.set(0)
            bar.grid(row=row, column=2, padx=12, pady=3)

            tally = ctk.CTkLabel(self.vote_holder, text="0", font=self.f_readout,
                                 text_color=MUTED, width=56, anchor="e")
            tally.grid(row=row, column=3, pady=3)

            self.vote_rows.append((name, label, bar, tally))

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

    def _toggle_regions(self) -> None:
        if self._ready:
            self._draw_base_plot()
            self.refresh()

    # -- output -----------------------------------------------------------

    def refresh(self) -> None:
        self._redraw_job = None
        if not self._ready:
            return

        h, w = self.height_var.get(), self.weight_var.get()
        result = self.kb.analyse(h, w)
        verdict = result["verdict"]
        ink = CLASS_INKS.get(verdict, TEXT)

        self.bmi_line.configure(text=f"The BMI is: {result['bmi']:.1f}")
        self.verdict.configure(text=verdict, text_color=ink)

        won = result["votes"].get(verdict, 0)
        vote_line = (f"{won} of the {self.kb.k} closest training records carry that "
                     f"label, the furthest of them {result['radius']:.2f} standard "
                     f"units away.")

        for name, label, bar, tally in self.vote_rows:
            count = result["votes"].get(name, 0)
            bar.set(count / self.kb.k)
            tally.configure(text=f"{count}/{self.kb.k}")
            hot = name == verdict
            label.configure(text_color=MARK if hot else MUTED)
            tally.configure(text_color=MARK if hot else MUTED)

        flip = self.kb.flip_weight(h, w)
        if flip is None:
            flip_line = f"At {h:.0f} cm no weight in range changes the verdict."
        else:
            direction = "above" if flip["delta"] > 0 else "below"
            flip_line = (f"At {h:.0f} cm the verdict turns {flip['into'].lower()} "
                         f"{direction} {flip['at']:.1f} kg, "
                         f"{abs(flip['delta']):.1f} kg from this reading.")

        self.verdict_note.configure(text=f"{vote_line}\n{flip_line}")

        self._draw_overlay(h, w, result)
        self._show_gif(verdict)

    # -- plot -------------------------------------------------------------

    def _draw_base_plot(self) -> None:
        ax = self.ax
        ax.clear()
        ax.set_facecolor(PANEL)

        if self.regions_var.get() and self.kb.grid is not None:
            hh, ww, zz = self.kb.grid
            ax.contourf(hh, ww, zz, levels=[-0.5, 0.5, 1.5],
                        colors=[CLASS_INKS["Obese"], CLASS_INKS["Fit"]], alpha=0.10)
            ax.contour(hh, ww, zz, levels=[0.5], colors=[MUTED],
                       linewidths=1.0, linestyles="--")

        xy, labels = self.kb.train_xy, self.kb.train_label
        rng = np.random.default_rng(42)
        idx = rng.choice(len(xy), size=min(2400, len(xy)), replace=False)
        for name in self.kb.classes:
            mask = labels[idx] == name
            ax.scatter(xy[idx][mask, 0], xy[idx][mask, 1], s=7,
                       c=CLASS_INKS.get(name, TEXT), alpha=0.30, linewidths=0,
                       label=name)

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

    def _draw_overlay(self, h: float, w: float, result: dict) -> None:
        for artist in self._overlay:
            artist.remove()
        self._overlay = []

        ax = self.ax
        ink = CLASS_INKS.get(result["verdict"], TEXT)

        # The ring that encloses the k voters. A circle in scaled space is an
        # ellipse once it is mapped back to centimetres and kilograms.
        sx, sy = self.kb.scaler.scale_[0], self.kb.scaler.scale_[1]
        ring = Ellipse((h, w), width=2 * result["radius"] * sx,
                       height=2 * result["radius"] * sy, fill=False,
                       edgecolor=ink, linewidth=1.1, linestyle=(0, (4, 3)),
                       alpha=0.8, zorder=6)
        ax.add_patch(ring)
        self._overlay.append(ring)

        for (nx, ny), label in zip(result["points"], result["labels"]):
            line, = ax.plot([h, nx], [w, ny],
                            color=CLASS_INKS.get(label, MUTED),
                            lw=0.9, alpha=0.55, zorder=7)
            self._overlay.append(line)

        ring_pts = ax.scatter(result["points"][:, 0], result["points"][:, 1],
                              s=42, facecolors="none",
                              edgecolors=[CLASS_INKS.get(l, MUTED) for l in result["labels"]],
                              linewidths=1.2, zorder=8)
        self._overlay.append(ring_pts)

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
                frame.thumbnail((230, 190))
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
    KnnApp().mainloop()