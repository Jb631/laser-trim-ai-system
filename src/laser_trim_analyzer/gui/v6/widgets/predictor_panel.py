"""Spec 3c — PredictorPanel: demoted per-unit predictor. Lazy, clearly diagnostic, graceful."""
from typing import Callable, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


def _default_load(db):
    def _load(model: str) -> str:
        # Best-effort read of whatever the existing predictor exposes for the model.
        try:
            from laser_trim_analyzer.ml import get_shared_ml_manager
            mgr = get_shared_ml_manager(db)
            pred = getattr(mgr, "predictors", {}).get(model)
            if pred is None:
                raise LookupError("no trained predictor for this model")
            return f"Predictor loaded for {model} (diagnostic — not part of daily flow)."
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
        try:
            self._body_label.configure(text=self._load_fn(self._model))
        except Exception:
            self._body_label.configure(
                text=f"No predictor for {self._model}. Train it in Settings → ML Training.")
