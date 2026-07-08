"""Spec 3d — TrainingModal: drift training with progress. train_fn injectable for tests."""
import threading
from typing import Callable, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class TrainingModal(ctk.CTkToplevel):
    def __init__(self, master, theme: ThemeManager, db, preset: str,
                 train_fn: Optional[Callable] = None):
        super().__init__(master)
        self.theme = theme
        self.db = db
        self.preset = preset
        if train_fn is None:
            from laser_trim_analyzer.ml.drift_training import train_drift_detector
            train_fn = train_drift_detector
        self._train_fn = train_fn
        # Resolve the app's UiDispatcher NOW (main thread) — master may be a
        # nested frame, not V6App itself.
        from laser_trim_analyzer.gui.v6.ui_dispatch import resolve_dispatcher
        self._ui = resolve_dispatcher(master)
        self.title("Training drift detector")
        self.geometry("420x190")
        self.configure(fg_color=theme.SURFACE)
        self.transient(master)
        ctk.CTkLabel(self, text="Training drift detector", font=theme.font(theme.SIZE_HEADING, "bold"),
                     text_color=theme.TEXT_PRIMARY).pack(pady=(theme.SPACE_LG, theme.SPACE_SM))
        self._status = ctk.CTkLabel(self, text="Preparing…", font=theme.font(theme.SIZE_BODY),
                                    text_color=theme.TEXT_SECONDARY)
        self._status.pack(pady=theme.SPACE_SM)
        self._bar = ctk.CTkProgressBar(self, progress_color=theme.ACCENT, fg_color=theme.CARD)
        self._bar.pack(fill="x", padx=theme.SPACE_LG, pady=theme.SPACE_SM)
        self._bar.set(0)
        ctk.CTkButton(self, text="Close", fg_color=theme.CARD, hover_color=theme.ELEVATED,
                      text_color=theme.TEXT_PRIMARY, command=self.destroy,
                      corner_radius=theme.RADIUS_SM).pack(pady=theme.SPACE_MD)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def start(self) -> None:
        threading.Thread(target=self._run_training, daemon=True).start()

    def _run_training(self) -> None:
        try:
            self._train_fn(self.db, self.preset, progress_callback=self._on_progress)
        except Exception as exc:
            self._safe(lambda: self._status.configure(text=f"Training failed: {exc}"))
            return
        self._safe(lambda: self._status.configure(text="Training complete."))
        self._safe(lambda: self._bar.set(1.0))
        self._safe(lambda: self.after(750, self.destroy))

    def _on_progress(self, model: str, done: int, total: int) -> None:
        self._safe(lambda: self._status.configure(text=f"Training {done + 1} / {total} ({model})"))
        self._safe(lambda: self._bar.set((done + 1) / max(total, 1)))

    def _safe(self, fn) -> None:
        """Marshal fn to the UI thread; called from the training worker.

        Routes through the app's UiDispatcher (master is V6App) so no Tk call
        happens on the worker. Fallback runs inline for tests that drive the
        modal from the main thread with no dispatcher attached.
        """
        def guarded():
            try:
                if self.winfo_exists():
                    fn()
            except Exception:
                pass

        if self._ui is not None:
            self._ui.post(guarded)
        else:
            # No dispatcher (tests drive from the main thread): run inline.
            guarded()
