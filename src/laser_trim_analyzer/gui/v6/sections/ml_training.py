"""Spec 3d — ML Training: drift retrain (TrainingModal) + per-model ML retrain (V5 path)."""
import threading

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.ui_dispatch import post_ui
from laser_trim_analyzer.gui.v6.widgets.training_modal import TrainingModal


def build_ml_training_section(parent, theme: ThemeManager, app) -> None:
    t = theme
    ctk.CTkLabel(parent, justify="left", wraplength=640, anchor="w", font=t.font(t.SIZE_BODY),
                 text_color=t.TEXT_SECONDARY,
                 text=("Retrain ML against current data. Drift baselines feed Triage/Model; the "
                       "per-model threshold optimizer, failure predictor, and profiler are the "
                       "existing per-model ML."))\
        .pack(side="top", fill="x", anchor="w", pady=(0, t.SPACE_MD))

    status = ctk.CTkLabel(parent, text="", font=t.font(t.SIZE_CAPTION),
                          text_color=t.TEXT_SECONDARY, anchor="w")
    status.pack(side="top", fill="x")

    def retrain_drift():
        preset = getattr(app.config.ml, "drift_sensitivity", "standard")
        TrainingModal(parent, theme=t, db=app.db, preset=preset).start()

    def retrain_per_model():
        # Verified V5 entrypoint: fresh MLManager(db).train_all_models(...) + save_all(). Off-thread.
        def work():
            try:
                from laser_trim_analyzer.ml import MLManager
                mgr = MLManager(app.db)
                results = mgr.train_all_models(
                    min_samples=getattr(app.config.ml, "min_samples_for_training", 20))
                mgr.save_all()
                msg = f"Per-model ML retrained: {len(results)} models."
            except Exception as exc:
                msg = f"Per-model ML retrain failed: {exc}"
            post_ui(app, lambda: status.winfo_exists() and status.configure(text=msg))
        status.configure(text="Retraining per-model ML…")
        threading.Thread(target=work, daemon=True).start()

    ctk.CTkButton(parent, text="Retrain drift detector", command=retrain_drift, fg_color=t.ACCENT,
                  hover_color=t.ACCENT_HOVER, text_color=t.TEXT_INVERSE, corner_radius=t.RADIUS_SM)\
        .pack(side="top", anchor="w", pady=(0, t.SPACE_SM))
    # Border so it reads as a BUTTON: fg_color=CARD on a CARD section made it
    # look like a plain caption (live-walk finding, 2026-07-08).
    ctk.CTkButton(parent, text="Retrain per-model ML (thresholds + predictor + profiler)",
                  command=retrain_per_model, fg_color=t.CARD, hover_color=t.ELEVATED,
                  text_color=t.TEXT_PRIMARY, corner_radius=t.RADIUS_SM,
                  border_width=1, border_color=t.BORDER)\
        .pack(side="top", anchor="w")
