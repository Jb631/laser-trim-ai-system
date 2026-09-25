"""Findings -- every model's process findings, grouped by what to do about them (the front door).

Four groups (findings/presentation.py): change a setting to raise yield, laser time you could
save, tests to check, and what changed. Reads the cache only: nothing here computes anything.
"""
import logging
import threading
from typing import Any, Dict, List

import customtkinter as ctk

from laser_trim_analyzer.findings import presentation as P
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets import blocks
from laser_trim_analyzer.gui.v6.widgets.findings_view import FindingsView

logger = logging.getLogger(__name__)


class FindingsPage(PageBase):
    page_title = "Findings"

    def __init__(self, master, *, theme, app, page_title="Findings"):
        self._rows: List[Dict[str, Any]] = []
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    def build_content(self, parent):
        t = self.theme
        self._notices = ctk.CTkFrame(parent, fg_color="transparent")
        self._notices.pack(side="top", fill="x")
        self._list = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self._list.pack(side="top", fill="both", expand=True)
        self._view = FindingsView(self._list, t, on_open=self._open, include_empty=True)
        self._view.pack(fill="x")

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
        """{"rows", "errors", "failed", "errors_failed"}. TWO reads, guarded separately:

        * the list itself fails -> `failed` = "ExcType: message". A failed load must never come
          back looking like an empty, healthy result.
        * only "which models could not be worked out" fails -> the list is still true, so it is
          still shown; `errors_failed` says the OTHER thing is unknown. One try block around both
          used to replace a good list with an error page.
        """
        out: Dict[str, Any] = {"rows": [], "errors": {}, "failed": None, "errors_failed": None}
        try:
            out["rows"] = self.app.db.get_process_findings()
        except Exception as exc:
            logger.exception("Findings page: load failed")
            out["failed"] = f"{type(exc).__name__}: {exc}"
            return out
        try:
            out["errors"] = self.app.db.get_process_errors()
        except Exception as exc:
            logger.exception("Findings page: could not read which models failed")
            out["errors_failed"] = f"{type(exc).__name__}: {exc}"
        return out

    def _apply(self, data: Any) -> None:
        if isinstance(data, list):                 # back-compat: callers may still pass a plain row list
            data = {"rows": data, "errors": {}, "failed": None}
        t = self.theme
        rows = data.get("rows") or []
        errors = data.get("errors") or {}
        self._rows = rows
        for child in self._notices.winfo_children():
            child.destroy()
        if data.get("failed"):
            self.set_caption("")
            self._view.set_findings([])
            self._view.pack_forget()
            blocks.banner(self._notices, t,
                          f"Findings could not be loaded ({data['failed']}). This is an error, not an "
                          f"empty list — the log has the details.").pack(fill="x", pady=t.SPACE_SM)
            return
        if data.get("errors_failed"):
            blocks.banner(self._notices, t, P.errors_unknown_notice(data["errors_failed"])
                          ).pack(fill="x", pady=(0, t.SPACE_SM))
        if errors:
            blocks.banner(self._notices, t, P.errors_notice(errors)
                          ).pack(fill="x", pady=(0, t.SPACE_SM))
        if not rows:
            self.set_caption("")
            blocks.banner(self._notices, t,
                          "No findings yet. They are worked out after each ingest that saves trim files; "
                          "a model with nothing worth acting on does not appear here.", tone="quiet"
                          ).pack(fill="x", pady=t.SPACE_SM)
            self._view.set_findings([])
            self._view.pack_forget()
            return
        # Only reached with real rows to show -- no wasted pack-then-immediately-forget
        # cycle on the empty/failed paths above (review, 2026-09-24). winfo_manager(), not
        # winfo_ismapped(): "is it laid out" is the question, and a window that is minimised
        # (or withdrawn, as in the tests) has nothing mapped whether the view is packed or not.
        if self._view.winfo_manager() == "":
            self._view.pack(fill="x")
        self.set_caption(P.caption(P.arrange(rows), rows))
        self._view.set_findings(rows)

    def _open(self, model: str) -> None:
        self.app.set_model_route(model, tab="findings")
        self.app.show_page("model")
