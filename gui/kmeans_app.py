from __future__ import annotations

import threading
from pathlib import Path

import customtkinter as ctk
import joblib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "saved_models"
PLOT_DIR = BASE_DIR / "plots"
DATASET = BASE_DIR / "datasets" / "obesity_dataset.csv"


INK = "#FFFFFF"        # page ground and field fill
PANEL = "#F6F9FD"      # panel surfaces
PANEL_HI = "#E9F1FA"   # raised rows
RULE = "#CBDCEF"       # hairlines, tracks, grid
TEXT = "#0A3466"       # primary blue type
MUTED = "#5E87B3"      # secondary blue type
MARK = "#0A2A5E"       # the analysed point

GROUP_INKS = ["#1F6FBF", "#0E8C8C", "#5B54C9", "#2E8B45", "#B34A6E", "#5C7FAE"]

HEIGHT_RANGE = (140.0, 205.0)
WEIGHT_RANGE = (35.0, 160.0)


# --------------------------------------------------------------------------
# Model bundle
# --------------------------------------------------------------------------

class ClusterModel:
    """Wraps the saved K-Means artefacts and the reference population."""

    def __init__(self) -> None:
        self.model = joblib.load(MODEL_DIR / "kmeans_model.pkl")
        self.scaler = joblib.load(MODEL_DIR / "kmeans_scaler.pkl")
        self.metadata = joblib.load(MODEL_DIR / "kmeans_metadata.pkl")

        self.centers = np.asarray(self.metadata["cluster_centers"], dtype=float)
        self.k = len(self.centers)

        # Draw order: lightest build first, so colours stay stable between runs.
        bmi = self.centers[:, 1] / (self.centers[:, 0] / 100.0) ** 2
        self.order = list(np.argsort(bmi))
        self.rank = {c: i for i, c in enumerate(self.order)}

        self.df: pd.DataFrame | None = None
        self.labels: np.ndarray | None = None
        self.names: dict[int, str] = {}

    # -- population -------------------------------------------------------

    def load_population(self) -> None:
        df = pd.read_csv(DATASET, usecols=["height_cm", "weight_kg"])
        scaled = self.scaler.transform(df)
        self.df = df
        self.labels = self.model.predict(scaled)
        self._name_clusters()

    def _name_clusters(self) -> None:
        assert self.df is not None
        h_mu, h_sd = self.df.height_cm.mean(), self.df.height_cm.std()
        w_mu, w_sd = self.df.weight_kg.mean(), self.df.weight_kg.std()

        for cluster, (h, w) in enumerate(self.centers):
            hz, wz = (h - h_mu) / h_sd, (w - w_mu) / w_sd
            tall = "taller" if hz > 0.35 else "shorter" if hz < -0.35 else "mid-height"
            heavy = "heavier" if wz > 0.35 else "lighter" if wz < -0.35 else "mid-weight"
            self.names[cluster] = f"{tall.capitalize()} and {heavy}"

    # -- inference --------------------------------------------------------

    def analyse(self, height: float, weight: float) -> dict:
        point = pd.DataFrame([{"height_cm": height, "weight_kg": weight}])
        scaled = self.scaler.transform(point)
        distances = self.model.transform(scaled)[0]
        cluster = int(np.argmin(distances))

        weights = 1.0 / (distances ** 2 + 1e-9)
        shares = weights / weights.sum()

        runner_up = int(np.argsort(distances)[1]) if self.k > 1 else cluster

        return {
            "cluster": cluster,
            "distances": distances,
            "shares": shares,
            "runner_up": runner_up,
            "centre": self.centers[cluster],
            "bmi": weight / (height / 100.0) ** 2,
        }

    def peers(self, height: float, weight: float, band: float = 3.0) -> tuple[int, float]:
        """How this weight sits among people of a similar height."""
        if self.df is None:
            return 0, 0.0
        near = self.df.weight_kg[(self.df.height_cm - height).abs() <= band]
        if len(near) < 20:
            near = self.df.weight_kg
        pct = float((near < weight).mean() * 100.0)
        return len(near), pct


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

ctk.set_appearance_mode("light")


class KMeansApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Body cluster analyzer")
        self.geometry("1240x800")
        self.minsize(1060, 700)
        self.configure(fg_color=INK)

        self.cm = ClusterModel()
        self.saved: list[tuple[float, float, int]] = []
        self._redraw_job: str | None = None
        self._ready = False

        self.f_display = ctk.CTkFont("Segoe UI", 30, "bold")
        self.f_title = ctk.CTkFont("Segoe UI", 16, "bold")
        self.f_body = ctk.CTkFont("Segoe UI", 13)
        self.f_small = ctk.CTkFont("Segoe UI", 12)
        # Monospace only for the readouts that change while dragging, so the
        # digits do not jitter sideways.
        self.f_readout = ctk.CTkFont("Consolas", 15)

        self.height_var = ctk.DoubleVar(value=172.0)
        self.weight_var = ctk.DoubleVar(value=74.0)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_rail()
        self._build_stage()

        self.bind("<Return>", lambda _e: self.save_reading())
        self.after(60, self._start_loading)

    # -- chrome -----------------------------------------------------------

    def _build_header(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=INK, corner_radius=0, height=74)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            bar, text="Body cluster analyzer", font=self.f_display, text_color=TEXT
        ).grid(row=0, column=0, padx=(28, 16), pady=(16, 4), sticky="w")

        self.model_note = ctk.CTkLabel(
            bar,
            text=f"K-means with {self.cm.k} groups, fitted on height and weight",
            font=self.f_small,
            text_color=MUTED,
        )
        self.model_note.grid(row=0, column=1, sticky="w", pady=(24, 0))

        ctk.CTkFrame(self, fg_color=RULE, height=1, corner_radius=0).grid(
            row=0, column=0, columnspan=2, sticky="sew"
        )

    def _build_rail(self) -> None:
        rail = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, width=330)
        rail.grid(row=1, column=0, sticky="nsw")
        rail.grid_propagate(False)
        rail.grid_rowconfigure(6, weight=1)

        ctk.CTkLabel(
            rail, text="Measurements", font=self.f_title, text_color=TEXT
        ).grid(row=0, column=0, padx=24, pady=(24, 12), sticky="w")

        self.height_readout = self._measure_row(
            rail, 1, "Height", "cm", self.height_var, HEIGHT_RANGE
        )
        self.weight_readout = self._measure_row(
            rail, 2, "Weight", "kg", self.weight_var, WEIGHT_RANGE
        )

        self.bmi_line = ctk.CTkLabel(
            rail, text="", font=self.f_small, text_color=MUTED, justify="left"
        )
        self.bmi_line.grid(row=3, column=0, padx=24, pady=(4, 16), sticky="w")

        self.save_btn = ctk.CTkButton(
            rail,
            text="Save this reading",
            font=self.f_body,
            height=40,
            corner_radius=6,
            fg_color=TEXT,
            hover_color=MARK,
            text_color="#FFFFFF",
            command=self.save_reading,
        )
        self.save_btn.grid(row=4, column=0, padx=24, pady=(0, 20), sticky="ew")

        ctk.CTkFrame(rail, fg_color=RULE, height=1).grid(
            row=5, column=0, padx=24, sticky="ew"
        )

        self.saved_box = ctk.CTkScrollableFrame(
            rail, fg_color=PANEL, scrollbar_button_color=RULE, label_text=""
        )
        self.saved_box.grid(row=6, column=0, padx=14, pady=10, sticky="nsew")
        self.saved_empty = ctk.CTkLabel(
            self.saved_box,
            text="Saved readings appear here as rings\non the plot, for comparing inputs.",
            font=self.f_small,
            text_color=MUTED,
            justify="left",
        )
        self.saved_empty.pack(anchor="w", padx=10, pady=8)

        ctk.CTkLabel(
            rail,
            text="Groups come from height and weight only.\nThey are statistical clusters, not health categories.",
            font=self.f_small,
            text_color=MUTED,
            justify="left",
        ).grid(row=7, column=0, padx=24, pady=(0, 20), sticky="w")

    def _measure_row(self, parent, row, label, unit, var, limits) -> ctk.CTkLabel:
        holder = ctk.CTkFrame(parent, fg_color="transparent")
        holder.grid(row=row, column=0, padx=24, pady=(4, 10), sticky="ew")
        holder.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(holder, text=label, font=self.f_body, text_color=TEXT).grid(
            row=0, column=0, sticky="w"
        )

        entry = ctk.CTkEntry(
            holder,
            width=92,
            height=32,
            font=self.f_readout,
            justify="right",
            fg_color=INK,
            border_color=RULE,
            text_color=TEXT,
            corner_radius=6,
        )
        entry.grid(row=0, column=1, sticky="e")
        entry.insert(0, f"{var.get():.1f}")
        entry.bind("<Return>", lambda _e, v=var, e=entry, l=limits: self._commit(v, e, l))
        entry.bind("<FocusOut>", lambda _e, v=var, e=entry, l=limits: self._commit(v, e, l))

        ctk.CTkLabel(holder, text=unit, font=self.f_small, text_color=MUTED).grid(
            row=0, column=2, padx=(6, 0), sticky="e"
        )

        slider = ctk.CTkSlider(
            holder,
            from_=limits[0],
            to=limits[1],
            variable=var,
            height=16,
            button_color=MARK,
            button_hover_color=TEXT,
            progress_color=TEXT,
            fg_color=RULE,
            command=lambda _v, e=entry, vv=var: self._on_slide(vv, e),
        )
        slider.grid(row=1, column=0, columnspan=3, pady=(10, 0), sticky="ew")
        return entry

    def _build_stage(self) -> None:
        stage = ctk.CTkFrame(self, fg_color=INK, corner_radius=0)
        stage.grid(row=1, column=1, sticky="nsew")
        stage.grid_columnconfigure(0, weight=1)
        stage.grid_rowconfigure(0, weight=1)

        plot_panel = ctk.CTkFrame(stage, fg_color=PANEL, corner_radius=8,
                                  border_width=1, border_color=RULE)
        plot_panel.grid(row=0, column=0, padx=18, pady=(18, 10), sticky="nsew")
        plot_panel.grid_columnconfigure(0, weight=1)
        plot_panel.grid_rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(7.4, 4.6), dpi=100, facecolor=PANEL)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, plot_panel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.readout = ctk.CTkFrame(stage, fg_color=PANEL, corner_radius=8,
                                    border_width=1, border_color=RULE)
        self.readout.grid(row=1, column=0, padx=18, pady=(0, 18), sticky="ew")
        self.readout.grid_columnconfigure(0, weight=1)

        self.verdict = ctk.CTkLabel(
            self.readout, text="Reading the population…", font=self.f_title, text_color=TEXT
        )
        self.verdict.grid(row=0, column=0, padx=20, pady=(16, 2), sticky="w")

        self.context = ctk.CTkLabel(
            self.readout, text="", font=self.f_small, text_color=MUTED, justify="left"
        )
        self.context.grid(row=1, column=0, padx=20, pady=(0, 12), sticky="w")

        self.bars_holder = ctk.CTkFrame(self.readout, fg_color="transparent")
        self.bars_holder.grid(row=2, column=0, padx=20, pady=(0, 16), sticky="ew")
        self.bars_holder.grid_columnconfigure(1, weight=1)
        self.bars: list[tuple] = []

    # -- loading ----------------------------------------------------------

    def _start_loading(self) -> None:
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self) -> None:
        try:
            self.cm.load_population()
        except Exception as exc:  # surfaced in the UI rather than the console
            self.after(0, lambda: self.verdict.configure(
                text=f"Could not read {DATASET.name}: {exc}"
            ))
            return
        self.after(0, self._on_population_ready)

    def _on_population_ready(self) -> None:
        self._build_bars()
        self._draw_base_plot()
        self._ready = True
        n = len(self.cm.df)
        self.model_note.configure(
            text=f"K-means with {self.cm.k} groups, fitted on height and weight of {n:,} people"
        )
        self.refresh()

    def _build_bars(self) -> None:
        for cluster in self.cm.order:
            colour = GROUP_INKS[self.cm.rank[cluster] % len(GROUP_INKS)]
            row = self.cm.rank[cluster]

            swatch = ctk.CTkLabel(self.bars_holder, text="", width=10, height=10,
                                  fg_color=colour, corner_radius=2)
            swatch.grid(row=row, column=0, padx=(0, 10), pady=4)

            name = ctk.CTkLabel(self.bars_holder, text=self.cm.names[cluster],
                                font=self.f_body, text_color=TEXT, anchor="w")
            name.grid(row=row, column=1, sticky="w", pady=4)

            bar = ctk.CTkProgressBar(self.bars_holder, height=6, corner_radius=3,
                                     fg_color=RULE, progress_color=colour, width=220)
            bar.set(0)
            bar.grid(row=row, column=2, padx=16, pady=4)

            share = ctk.CTkLabel(self.bars_holder, text="0%", font=self.f_readout,
                                 text_color=MUTED, width=56, anchor="e")
            share.grid(row=row, column=3, pady=4)

            self.bars.append((cluster, name, bar, share))

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
            self.context.configure(text=f"'{raw}' is not a number. Enter a value like 172.5.")
            return
        value = min(max(value, limits[0]), limits[1])
        var.set(value)
        entry.delete(0, "end")
        entry.insert(0, f"{value:.1f}")
        self.schedule_refresh()

    def schedule_refresh(self) -> None:
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(90, self.refresh)

    # -- output -----------------------------------------------------------

    def refresh(self) -> None:
        self._redraw_job = None
        if not self._ready:
            return

        h, w = self.height_var.get(), self.weight_var.get()
        result = self.cm.analyse(h, w)
        cluster = result["cluster"]
        colour = GROUP_INKS[self.cm.rank[cluster] % len(GROUP_INKS)]

        self.bmi_line.configure(
            text=f"This input works out to a BMI of {result['bmi']:.1f}"
        )

        self.verdict.configure(
            text=f"Closest to group {self.cm.rank[cluster] + 1}, {self.cm.names[cluster].lower()}",
            text_color=colour,
        )

        centre = result["centre"]
        peers, pct = self.cm.peers(h, w)
        runner = result["runner_up"]
        self.context.configure(
            text=(
                f"That group centres on {centre[0]:.1f} cm and {centre[1]:.1f} kg, "
                f"{result['distances'][cluster]:.2f} standard units from this input.\n"
                f"Among the {peers:,} people within 3 cm of this height, "
                f"{pct:.0f}% weigh less than this value. "
                f"Next closest group is {self.cm.names[runner].lower()}."
            )
        )

        for cid, name_lbl, bar, share_lbl in self.bars:
            share = result["shares"][cid]
            bar.set(float(share))
            share_lbl.configure(text=f"{share * 100:4.0f}%")
            is_match = cid == cluster
            name_lbl.configure(text_color=TEXT if is_match else MUTED)
            share_lbl.configure(text_color=TEXT if is_match else MUTED)

        self._draw_overlay(h, w, result)

    # -- plot -------------------------------------------------------------

    def _draw_base_plot(self) -> None:
        df, labels = self.cm.df, self.cm.labels
        rng = np.random.default_rng(42)
        idx = rng.choice(len(df), size=min(2600, len(df)), replace=False)

        ax = self.ax
        ax.clear()
        ax.set_facecolor(PANEL)

        for cluster in self.cm.order:
            colour = GROUP_INKS[self.cm.rank[cluster] % len(GROUP_INKS)]
            mask = labels[idx] == cluster
            ax.scatter(
                df.height_cm.values[idx][mask],
                df.weight_kg.values[idx][mask],
                s=7, c=colour, alpha=0.38, linewidths=0,
            )

        for cluster, (ch, cw) in enumerate(self.cm.centers):
            colour = GROUP_INKS[self.cm.rank[cluster] % len(GROUP_INKS)]
            ax.scatter([ch], [cw], s=140, marker="P", c=colour,
                       edgecolors=PANEL, linewidths=1.6, zorder=5)

        ax.set_xlabel("Height (cm)", color=TEXT, fontsize=10)
        ax.set_ylabel("Weight (kg)", color=TEXT, fontsize=10)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(color=RULE, linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(RULE)

        self.figure.tight_layout()
        self._overlay: list = []
        self.canvas.draw()

    def _draw_overlay(self, h: float, w: float, result: dict) -> None:
        """Redraw only the marker and its leader lines."""
        for artist in getattr(self, "_overlay", []):
            artist.remove()
        self._overlay = []

        ax = self.ax
        cluster = result["cluster"]
        colour = GROUP_INKS[self.cm.rank[cluster] % len(GROUP_INKS)]
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        # Solid leader line to the matched centroid, faint dashes to the rest.
        for cid, (ch, cw) in enumerate(self.cm.centers):
            matched = cid == cluster
            line, = ax.plot(
                [h, ch], [w, cw],
                color=colour if matched else MUTED,
                lw=1.4 if matched else 0.7,
                ls="-" if matched else (0, (3, 4)),
                alpha=0.9 if matched else 0.35,
                zorder=6,
            )
            self._overlay.append(line)

        self._overlay.append(ax.axhline(w, color=MARK, lw=0.6, alpha=0.35, zorder=6))
        self._overlay.append(ax.axvline(h, color=MARK, lw=0.6, alpha=0.35, zorder=6))

        for sh, sw, _ in self.saved:
            ring = ax.scatter([sh], [sw], s=70, facecolors="none",
                              edgecolors=MARK, linewidths=1.1, alpha=0.6, zorder=7)
            self._overlay.append(ring)

        dot = ax.scatter([h], [w], s=150, marker="X", c=MARK,
                         edgecolors="#FFFFFF", linewidths=1.4, zorder=9)
        self._overlay.append(dot)

        label = ax.annotate(
            f"{h:.1f} cm, {w:.1f} kg",
            xy=(h, w), xytext=(10, 12), textcoords="offset points",
            color="#FFFFFF", fontsize=9.5, fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.32", fc=MARK, ec="none"),
        )
        self._overlay.append(label)

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        self.canvas.draw_idle()

    # -- saved readings ---------------------------------------------------

    def save_reading(self) -> None:
        if not self._ready:
            return
        h, w = self.height_var.get(), self.weight_var.get()
        result = self.cm.analyse(h, w)
        self.saved.append((h, w, result["cluster"]))
        self.saved_empty.pack_forget()

        colour = GROUP_INKS[self.cm.rank[result["cluster"]] % len(GROUP_INKS)]
        row = ctk.CTkFrame(self.saved_box, fg_color=INK, corner_radius=6,
                           border_width=1, border_color=RULE)
        row.pack(fill="x", padx=4, pady=3)

        ctk.CTkLabel(row, text="", width=4, height=26, fg_color=colour,
                     corner_radius=2).pack(side="left", padx=(8, 8), pady=6)
        ctk.CTkLabel(row, text=f"{h:.1f} cm, {w:.1f} kg", font=self.f_small,
                     text_color=TEXT).pack(side="left")
        ctk.CTkButton(
            row, text="Load", width=54, height=24, font=self.f_small,
            fg_color="transparent", hover_color=PANEL_HI, text_color=TEXT,
            command=lambda hh=h, ww=w: self.load_reading(hh, ww),
        ).pack(side="right", padx=6)

        self.refresh()

    def load_reading(self, h: float, w: float) -> None:
        self.height_var.set(h)
        self.weight_var.set(w)
        self.height_readout.delete(0, "end")
        self.height_readout.insert(0, f"{h:.1f}")
        self.weight_readout.delete(0, "end")
        self.weight_readout.insert(0, f"{w:.1f}")
        self.refresh()


if __name__ == "__main__":
    PLOT_DIR.mkdir(exist_ok=True)
    KMeansApp().mainloop()