"""Spec 3c — PredictorPanel: demoted per-unit predictor. Lazy, clearly diagnostic, graceful."""
import threading
from typing import Callable, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.ui_dispatch import resolve_dispatcher


def _default_load(db):
    def _load(model: str) -> str:
        # Show the predictor's actual content — "Predictor loaded" told the
        # user nothing (work finding #13, 2026-07-10).
        try:
            from laser_trim_analyzer.ml import get_shared_ml_manager
            mgr = get_shared_ml_manager(db)
            pred = getattr(mgr, "predictors", {}).get(model)
            if pred is None or not getattr(pred, "is_trained", False):
                raise LookupError("no trained predictor for this model")
            m = getattr(pred, "metrics", None)   # PredictorMetrics dataclass (or dict)
            def _g(obj, key):
                if obj is None:
                    return None
                if isinstance(obj, dict):
                    return obj.get(key)
                return getattr(obj, key, None)
            lines = [f"Failure predictor for {model} — trained on final-test outcomes."]
            perf = []
            for key, label, fmt in (("accuracy", "accuracy", "{:.0%}"),
                                    ("f1", "F1", "{:.2f}"),
                                    ("auc_roc", "AUC", "{:.2f}")):
                v = _g(m, key)
                if v:
                    perf.append(f"{label} " + fmt.format(v))
            n = getattr(pred, "training_samples", None)
            if n:
                perf.append(f"{n} training units")
            if perf:
                lines.append("Performance: " + ", ".join(perf) + ".")
            fi = getattr(pred, "feature_importance", None) or {}
            if fi:
                top = sorted(fi.items(), key=lambda kv: -abs(kv[1]))[:3]
                lines.append("Strongest signals: "
                             + ", ".join(f"{k} ({v:.0%})" for k, v in top) + ".")
            lines.append("Diagnostic only — linearity remains the disposition rule.")
            return "\n".join(lines)
        except Exception as exc:
            raise exc
    return _load


class PredictorPanel(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, load_fn: Optional[Callable[[str], str]] = None,
                 db=None, **kwargs):
        super().__init__(master, fg_color=theme.CARD, corner_radius=theme.RADIUS_MD, **kwargs)
        self.theme = theme
        self._load_fn = load_fn or (_default_load(db) if db is not None else None)
        self._model: Optional[str] = None
        self._expanded = False
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(side="top", fill="x")
        ctk.CTkLabel(header, text="Predictor (diagnostic — not part of daily flow)",
                     font=theme.font(theme.SIZE_CAPTION, "bold"), text_color=theme.TEXT_SECONDARY)\
            .pack(side="left", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
        self._toggle_btn = ctk.CTkButton(header, text="Show", width=60, fg_color=theme.CARD,
                                         hover_color=theme.ELEVATED, text_color=theme.TEXT_SECONDARY,
                                         command=self.toggle, corner_radius=theme.RADIUS_SM)
        self._toggle_btn.pack(side="right", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
        self._body = ctk.CTkFrame(self, fg_color="transparent")
        self._body_label = ctk.CTkLabel(self._body, text="", font=theme.font(theme.SIZE_BODY),
                                        text_color=theme.TEXT_SECONDARY, wraplength=700, justify="left")
        self._body_label.pack(padx=theme.SPACE_SM, pady=theme.SPACE_SM)
        self._ui = resolve_dispatcher(master)
        self._load_gen = 0  # drop stale results when the model changes mid-load

    def set_model(self, model: str) -> None:
        self._model = model
        if self._expanded:
            self._load()

    def toggle(self) -> None:
        if self._expanded:
            self._body.pack_forget()
            self._toggle_btn.configure(text="Show")
        else:
            self._body.pack(side="top", fill="x")
            self._toggle_btn.configure(text="Hide")
            self._load()
        self._expanded = not self._expanded

    def _load(self) -> None:
        if not self._model or self._load_fn is None:
            self._body_label.configure(text="No predictor available.")
            return
        # load_fn reaches the shared MLManager, which may unpickle every
        # trained predictor from disk — far too heavy for the UI thread
        # (and it re-runs after the manager's cache expires).
        model = self._model
        self._load_gen += 1
        gen = self._load_gen
        self._body_label.configure(text="Loading predictor…")

        def work():
            try:
                text = self._load_fn(model)
            except Exception:
                text = f"No predictor for {model}. Train it in Settings → ML Training."

            def apply():
                try:
                    if gen == self._load_gen and self._body_label.winfo_exists():
                        self._body_label.configure(text=text)
                except Exception:
                    pass
            if self._ui is not None:
                self._ui.post(apply)
            else:
                apply()  # tests: main thread, no dispatcher
        threading.Thread(target=work, daemon=True).start()
