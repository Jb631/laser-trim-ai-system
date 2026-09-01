"""Ingest Folders — the remembered list behind HOME's "Process everything new".

Spec: docs/superpowers/specs/2026-08-29-app-shape-investigate-design.md §1/§3.
Configured once here, walked top-to-bottom by HOME: the laser folders, then
the Final Test folder. Order is the processing order, which is why this is a
reorderable list and not a set.

Two behaviours this section is built around:

  * A folder that cannot be reached is REPORTED, never dropped. Adding an
    offline share succeeds — shares go down, and a list that silently refuses
    them would leave the batch quietly processing 2 of 3 folders. The status
    line names what is unreachable, right where the list is edited.
  * Every edit persists immediately. There is no Save button because there is
    no half-edited state worth keeping: the list IS the setting, and losing it
    to a forgotten click is worse than an extra write of a tiny YAML file.

Reachability checks touch the network (a dead SMB path blocks for the mount
timeout), so they run on a worker and post the result back through
`post_ui` — the section itself never reads or writes Tk off the main thread.
"""
import threading
from tkinter import filedialog
from typing import List, Optional

import customtkinter as ctk

from laser_trim_analyzer.config import (
    IngestConfig,
    missing_ingest_folders,
    normalize_ingest_path,
)
from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.ui_dispatch import post_ui


class IngestFoldersSection:
    """Controller + view. The mutations live on IngestConfig; this owns the rows."""

    def __init__(self, parent, theme: ThemeManager, app):
        self.theme = theme
        self.app = app
        self._rows: List[ctk.CTkFrame] = []
        self._problems: dict = {}          # folder -> reason (last check)
        t = theme

        ctk.CTkLabel(parent, justify="left", wraplength=640, anchor="w",
                     font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY,
                     text=("Home's “Process everything new” runs these folders in "
                           "order — laser folders first, Final Test last. Add each "
                           "one once; a folder that is offline is still processed "
                           "next time and is called out below, never skipped in "
                           "silence."))\
            .pack(side="top", fill="x", anchor="w", pady=(0, t.SPACE_MD))

        self._list = ctk.CTkFrame(parent, fg_color="transparent")
        self._list.pack(side="top", fill="x")

        add_row = ctk.CTkFrame(parent, fg_color="transparent")
        add_row.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))
        self._entry = ctk.CTkEntry(
            add_row, font=t.font(t.SIZE_BODY),
            placeholder_text=r"\\192.168.66.9\Public\LaserTrim  (or Browse…)")
        self._entry.pack(side="left", fill="x", expand=True, padx=(0, t.SPACE_SM))
        self._entry.bind("<Return>", lambda _e: self._add_from_entry())
        ctk.CTkButton(add_row, text="Add", width=70, fg_color=t.ACCENT,
                      hover_color=t.ACCENT_HOVER, text_color=t.TEXT_INVERSE,
                      corner_radius=t.RADIUS_SM, command=self._add_from_entry)\
            .pack(side="left", padx=(0, t.SPACE_XS))
        ctk.CTkButton(add_row, text="Browse…", width=90, fg_color=t.CARD,
                      hover_color=t.ELEVATED, text_color=t.TEXT_PRIMARY,
                      corner_radius=t.RADIUS_SM, command=self._browse)\
            .pack(side="left")

        self._status = ctk.CTkLabel(parent, text="", justify="left", anchor="w",
                                    wraplength=640, font=t.font(t.SIZE_CAPTION),
                                    text_color=t.TEXT_SECONDARY)
        self._status.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))

        ctk.CTkButton(parent, text="Check folders now", width=150,
                      fg_color=t.CARD, hover_color=t.ELEVATED,
                      text_color=t.TEXT_PRIMARY, corner_radius=t.RADIUS_SM,
                      command=self.check_folders)\
            .pack(side="top", anchor="w", pady=(t.SPACE_SM, 0))

        self._render()
        self.check_folders()

    # ---- config access ----------------------------------------------------
    @property
    def _cfg(self) -> IngestConfig:
        cfg = getattr(self.app.config, "ingest", None)
        if cfg is None:                      # a Config built before this field
            cfg = IngestConfig()
            self.app.config.ingest = cfg
        return cfg

    def paths(self) -> List[str]:
        return list(self._cfg.folders)

    def _persist(self) -> None:
        try:
            self.app.config.save()
        except Exception:
            # Same posture as the other settings sections: a config that will
            # not write must not take the UI down with it.
            pass

    # ---- mutations (return None on success, a message on refusal) ----------
    def add_folder(self, path: str) -> Optional[str]:
        p = normalize_ingest_path(path)
        if not p:
            return "Enter a folder path first."
        if self._cfg.index_of(p) is not None:
            return f"“{p}” is already on the list."
        self._cfg.add(p)
        self._persist()
        self._render()
        self.check_folders()
        return None

    def remove(self, index: int) -> None:
        if 0 <= index < len(self._cfg.folders):
            self._cfg.remove(self._cfg.folders[index])
            self._persist()
            self._render()
            self.check_folders()

    def move(self, index: int, delta: int) -> None:
        if self._cfg.move(index, delta):
            self._persist()
            self._render()

    # ---- status -----------------------------------------------------------
    def status_text(self) -> str:
        return self._status.cget("text")

    def check_folders(self) -> None:
        """Verify reachability off the Tk thread; paint the verdict on it."""
        folders = self.paths()               # read config on THIS thread

        def work():
            try:
                problems = dict(missing_ingest_folders(folders))
            except Exception as exc:
                problems = {"": f"check failed: {exc}"}
            post_ui(self.app, lambda: self._apply_status(problems))

        if getattr(self.app, "ui", None) is None:
            # Tests / standalone: stay synchronous so the status is readable
            # immediately after the call instead of at some later tick.
            self._apply_status(dict(missing_ingest_folders(folders)))
            return
        threading.Thread(target=work, daemon=True).start()

    def _apply_status(self, problems: dict) -> None:
        self._problems = problems
        n = len(self.paths())
        if n == 0:
            self._set_status("No folders configured — add the laser folders and "
                             "the Final Test folder above.")
            return
        noun = "folder" if n == 1 else "folders"
        txt = f"{n} {noun} configured, processed in this order."
        if problems:
            bad = "; ".join(f"{p or '(blank)'} — {why}" for p, why in problems.items())
            txt += f"  ⚠ {len(problems)} unreachable right now: {bad}"
        self._set_status(txt)
        self._render()

    def _set_status(self, text: str) -> None:
        try:
            if self._status.winfo_exists():
                self._status.configure(text=text)
        except Exception:
            pass

    # ---- view -------------------------------------------------------------
    def _add_from_entry(self) -> None:
        value = self._entry.get()
        msg = self.add_folder(value)
        if msg:
            self._set_status(msg)
        else:
            self._entry.delete(0, "end")

    def _browse(self) -> None:
        path = filedialog.askdirectory(title="Select a folder to ingest")
        if path:
            msg = self.add_folder(path)
            if msg:
                self._set_status(msg)

    def _render(self) -> None:
        t = self.theme
        for w in self._rows:
            try:
                w.destroy()
            except Exception:
                pass
        self._rows = []
        folders = self.paths()
        if not folders:
            row = ctk.CTkLabel(self._list, anchor="w", font=t.font(t.SIZE_BODY),
                               text_color=t.TEXT_SECONDARY,
                               text="(none yet — Home will point you back here)")
            row.pack(side="top", fill="x", pady=t.SPACE_XS)
            self._rows.append(row)
            return
        last = len(folders) - 1
        for i, folder in enumerate(folders):
            row = ctk.CTkFrame(self._list, fg_color=t.SURFACE,
                               corner_radius=t.RADIUS_SM)
            row.pack(side="top", fill="x", pady=2)
            ctk.CTkLabel(row, text=f"{i + 1}.", width=24, anchor="w",
                         font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY)\
                .pack(side="left", padx=(t.SPACE_SM, 0), pady=t.SPACE_XS)
            problem = self._problems.get(folder)
            ctk.CTkLabel(row, text=folder + (f"   ⚠ {problem}" if problem else ""),
                         anchor="w", font=t.font(t.SIZE_BODY),
                         text_color=t.TIER_WARNING if problem else t.TEXT_PRIMARY)\
                .pack(side="left", fill="x", expand=True, padx=(t.SPACE_XS, t.SPACE_SM))
            for text, delta, enabled in (("▲", -1, i > 0), ("▼", +1, i < last)):
                ctk.CTkButton(row, text=text, width=30, fg_color="transparent",
                              hover_color=t.ELEVATED, corner_radius=t.RADIUS_SM,
                              text_color=t.TEXT_SECONDARY if enabled else t.TEXT_DISABLED,
                              state="normal" if enabled else "disabled",
                              command=lambda idx=i, d=delta: self.move(idx, d))\
                    .pack(side="left", padx=1, pady=2)
            ctk.CTkButton(row, text="✕", width=30, fg_color="transparent",
                          hover_color=t.ELEVATED, text_color=t.TEXT_SECONDARY,
                          corner_radius=t.RADIUS_SM,
                          command=lambda idx=i: self.remove(idx))\
                .pack(side="left", padx=(1, t.SPACE_SM), pady=2)
            self._rows.append(row)


def build_ingest_folders_section(parent, theme: ThemeManager, app) -> IngestFoldersSection:
    """Settings-card entry point. Returns the controller so it stays testable."""
    return IngestFoldersSection(parent, theme=theme, app=app)
