"""Findings -- every model's process findings in one ranked list (the front door).

Recoverable units a year first; findings that cannot state a gain after them,
by volume. Reads the cache only: nothing here computes anything.
"""
import logging
import threading
from typing import Any, Dict, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.page_base import PageBase

logger = logging.getLogger(__name__)


class FindingsPage(PageBase):
    page_title = "Findings"

    def __init__(self, master, *, theme, app, page_title="Findings"):
        self._rows: List[Dict[str, Any]] = []
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    def build_content(self, parent):
        self._zone_header(parent, "WHAT TO CHANGE, BIGGEST FIRST",
                          "units a year recoverable, then by volume — click a row to open the model")
        self._list = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self._list.pack(side="top", fill="both", expand=True)

    # ---- data ----
    def reload_now(self):
        """Synchronous reload + apply (the test path)."""
        self._apply(self._query())

    def on_show(self):
        """Load on a background thread; apply on the Tk thread via safe_after."""
        def work():
            data = self._query()
            self.safe_after(lambda: self._apply(data))
        threading.Thread(target=work, daemon=True).start()

    def _query(self) -> Dict[str, Any]:
        """{"rows": [...], "errors": {...}, "failed": None} on success, or on ANY exception
        (a database error, say) {"rows": [], "errors": {}, "failed": "ExcType: message"} --
        a failed load must never come back looking like an empty, healthy result."""
        try:
            return {"rows": self.app.db.get_process_findings(),
                    "errors": self.app.db.get_process_errors(), "failed": None}
        except Exception as exc:
            logger.exception("Findings page: load failed")
            return {"rows": [], "errors": {}, "failed": f"{type(exc).__name__}: {exc}"}

    def _apply(self, data: Any) -> None:
        if isinstance(data, list):                 # back-compat: callers may still pass a plain row list
            data = {"rows": data, "errors": {}, "failed": None}
        t = self.theme
        rows = data.get("rows") or []
        errors = data.get("errors") or {}
        failed = data.get("failed")
        self._rows = rows
        for child in self._list.winfo_children():
            child.destroy()
        if failed:
            ctk.CTkLabel(self._list, font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY, anchor="w",
                         justify="left", wraplength=900,
                         text=f"Findings could not be loaded ({failed}). This is an error, not an "
                              f"empty list — the log has the details."
                         ).pack(fill="x", pady=t.SPACE_SM)
            return
        if not rows:
            ctk.CTkLabel(self._list, font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY, anchor="w",
                         justify="left", wraplength=900,
                         text="No findings yet. They are worked out after each ingest that saves trim files; "
                              "a model with nothing worth acting on does not appear here."
                         ).pack(fill="x", pady=t.SPACE_SM)
        else:
            for f in rows:
                upy = f.get("units_per_year")
                gain = f"{upy:,.0f} units a year" if upy is not None else "no gain claimed"
                ctk.CTkButton(
                    self._list, anchor="w", fg_color=t.CARD, hover_color=t.ACCENT_HOVER,
                    text_color=t.TEXT_PRIMARY, font=t.font(t.SIZE_BODY), corner_radius=8,
                    text=f"{f.get('model', '')}   ·   {f.get('title', '')}\n"
                         f"{f.get('category', '')}  ·  lever: {f.get('lever_label', '')} "
                         f"({f.get('lead_time', '')})  ·  {gain}",
                    command=lambda m=f.get("model"): self._open(m),
                ).pack(fill="x", pady=(0, t.SPACE_SM))
        if errors:
            names = sorted(errors)
            shown = ", ".join(names[:10]) + (" …" if len(names) > 10 else "")
            ctk.CTkLabel(self._list, font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY, anchor="w",
                         justify="left", wraplength=900,
                         text=f"{len(names)} model(s) could not be fully worked out on the last refresh, "
                              f"so they may be missing from this list: {shown}. Open one to see what failed."
                         ).pack(fill="x", pady=t.SPACE_SM)

    def _open(self, model: str) -> None:
        self.app.set_model_route(model)
        self.app.show_page("model")
