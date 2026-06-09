"""Spec 3e — ProcessProgressSection: bar + 5 counters + failure list."""
from typing import Dict, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

_BUCKETS = [("passed", "Passed"), ("warnings", "Warnings"), ("failed", "Failed"),
            ("skipped", "Skipped"), ("errors", "Errors")]


class ProcessProgressSection(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._counters: Dict[str, int] = {k: 0 for k, _ in _BUCKETS}
        self._failures: List[str] = []
        self._status = ctk.CTkLabel(self, text="Ready", font=theme.font(theme.SIZE_BODY),
                                    text_color=theme.TEXT_PRIMARY, anchor="w")
        self._status.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        self._bar = ctk.CTkProgressBar(self, progress_color=theme.ACCENT, fg_color=theme.CARD)
        self._bar.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        self._bar.set(0)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        self._labels: Dict[str, ctk.CTkLabel] = {}
        colors = {"passed": theme.TEXT_PRIMARY, "warnings": theme.TIER_WARNING,
                  "failed": theme.TIER_OOC, "skipped": theme.TEXT_SECONDARY, "errors": theme.TIER_OOC}
        for key, label in _BUCKETS:
            lbl = ctk.CTkLabel(row, text=f"{label}: 0", font=theme.font(theme.SIZE_BODY, "bold"),
                               text_color=colors[key])
            lbl.pack(side="left", padx=(0, theme.SPACE_LG))
            self._labels[key] = lbl
        self._fail_label = ctk.CTkLabel(self, text="Recent failures:", font=theme.font(theme.SIZE_CAPTION, "bold"),
                                        text_color=theme.TEXT_SECONDARY, anchor="w")
        self._fail_box = ctk.CTkTextbox(self, height=90, fg_color=theme.CARD,
                                        text_color=theme.TEXT_SECONDARY, font=theme.font(theme.SIZE_CAPTION))

    def set_progress(self, current: int, total: int, current_filename: str = "") -> None:
        self._bar.set(current / max(total, 1))
        txt = f"Processing {current} / {total}"
        if current_filename:
            txt += f": {current_filename}"
        self._status.configure(text=txt)

    def increment(self, key: str, reason: str = "") -> None:
        if key not in self._counters:
            return
        self._counters[key] += 1
        self._labels[key].configure(text=f"{dict(_BUCKETS)[key]}: {self._counters[key]}")
        if reason and key in ("failed", "errors"):
            self._failures.append(reason)
            self._render_failures()

    def set_final(self, summary) -> None:
        """Authoritative counts from BatchSummary (reconciles the live tally)."""
        self._counters = {"passed": summary.passed, "warnings": summary.warnings,
                          "failed": summary.failed, "skipped": summary.skipped, "errors": summary.errors}
        for key, label in _BUCKETS:
            self._labels[key].configure(text=f"{label}: {self._counters[key]}")
        self._bar.set(1.0)
        self._status.configure(text=(f"Complete: {summary.passed} passed, {summary.warnings} warnings, "
                                     f"{summary.failed} failed, {summary.skipped} skipped, "
                                     f"{summary.errors} errors."))

    def set_idle(self, message: str) -> None:
        self._status.configure(text=message)
        self._bar.set(0)

    def reset(self) -> None:
        self._counters = {k: 0 for k, _ in _BUCKETS}
        self._failures = []
        for key, label in _BUCKETS:
            self._labels[key].configure(text=f"{label}: 0")
        self._bar.set(0)
        self._status.configure(text="Ready")
        if self._fail_label.winfo_ismapped():
            self._fail_label.pack_forget()
            self._fail_box.pack_forget()

    def _render_failures(self) -> None:
        if not self._fail_label.winfo_ismapped():
            self._fail_label.pack(side="top", fill="x", pady=(self.theme.SPACE_SM, 0))
            self._fail_box.pack(side="top", fill="x")
        self._fail_box.configure(state="normal")
        self._fail_box.delete("1.0", "end")
        for line in self._failures[-10:]:
            self._fail_box.insert("end", line + "\n")
        self._fail_box.configure(state="disabled")
