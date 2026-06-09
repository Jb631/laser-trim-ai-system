"""Spec 3d — Database Cleanup: scan health + category purge + reset skipped (real V5 port).

Destructive operations go through the verified db.preview_cleanup / db.execute_cleanup
methods (never re-implemented here) and are gated behind a confirm dialog.
"""
import threading
from datetime import datetime
from typing import Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

# (UI key, checkbox label, execute_cleanup kwarg)
_CATEGORIES = [
    ("non_mps", "Non-MPS models (not in your MPS list)", "delete_non_mps"),
    ("suspect", "Suspect quality (flagged dirty)", "delete_suspect_quality"),
    ("unknown", "Unknown system/model", "delete_unknown"),
    ("error", "Error status", "delete_error_status"),
    ("no_tracks", "No tracks", "delete_no_tracks"),
    ("misclassified_ft", "Misclassified Final-Test", "delete_misclassified_ft"),
]


def build_cleanup_options(*, non_mps, before_date_enabled, date_str, suspect, unknown,
                          error, no_tracks, misclassified_ft, mps_models) -> Optional[dict]:
    """Build preview/execute_cleanup kwargs from UI state (pure port of V5
    _get_cleanup_options). Returns None when nothing is selected, when 'non-MPS'
    is checked but no MPS list is configured, or when the date is invalid."""
    opts = {
        "delete_non_mps": bool(non_mps),
        "mps_models": None,
        "delete_before_date": None,
        "delete_suspect_quality": bool(suspect),
        "delete_unknown": bool(unknown),
        "delete_error_status": bool(error),
        "delete_no_tracks": bool(no_tracks),
        "delete_misclassified_ft": bool(misclassified_ft),
    }
    if non_mps:
        opts["mps_models"] = list(mps_models or [])
        if not opts["mps_models"]:
            return None
    if before_date_enabled:
        try:
            opts["delete_before_date"] = datetime.strptime((date_str or "").strip(), "%Y-%m-%d")
        except ValueError:
            return None
    if not any([opts["delete_non_mps"], opts["delete_before_date"], opts["delete_suspect_quality"],
                opts["delete_unknown"], opts["delete_error_status"], opts["delete_no_tracks"],
                opts["delete_misclassified_ft"]]):
        return None
    return opts


def build_database_cleanup_section(parent, theme: ThemeManager, app) -> None:
    t = theme
    db = app.db

    ctk.CTkLabel(parent, justify="left", wraplength=640, anchor="w", font=t.font(t.SIZE_BODY),
                 text_color=t.TEXT_SECONDARY,
                 text=("Scan for dirty/contaminated records, then optionally purge by category. "
                       "Deletes are permanent and run through the same filters as the preview."))\
        .pack(side="top", fill="x", anchor="w", pady=(0, t.SPACE_MD))

    status = ctk.CTkLabel(parent, text="", justify="left", font=t.font(t.SIZE_CAPTION),
                          text_color=t.TEXT_SECONDARY, anchor="w")
    status.pack(side="top", fill="x")

    def _async(work_fn):
        def runner():
            try:
                msg = work_fn()
            except Exception as exc:
                msg = f"Error: {exc}"
            try:
                if status.winfo_exists():
                    status.after(0, lambda: status.winfo_exists() and status.configure(text=msg))
            except Exception:
                pass
        threading.Thread(target=runner, daemon=True).start()

    def _scan():
        status.configure(text="Scanning…")

        def work():
            health = db.scan_database_health()
            lines = [f"{health.get('total_dirty_records', 0)} records with issues"]
            for info in (health.get("issues") or {}).values():
                lines.append(f"  • {info['label']}: {info['count']}")
            return "\n".join(lines) if len(lines) > 1 else "Database is clean."
        _async(work)

    ctk.CTkButton(parent, text="Scan database", command=_scan, fg_color=t.CARD,
                  hover_color=t.ELEVATED, text_color=t.TEXT_PRIMARY,
                  corner_radius=t.RADIUS_SM).pack(side="top", anchor="w", pady=(0, t.SPACE_MD))

    # Category checkboxes.
    cvars = {}
    for key, label, _kw in _CATEGORIES:
        var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(parent, text=label, variable=var, font=t.font(t.SIZE_BODY),
                        text_color=t.TEXT_PRIMARY, fg_color=t.ACCENT,
                        hover_color=t.ACCENT_HOVER).pack(side="top", anchor="w", pady=1)
        cvars[key] = var

    date_row = ctk.CTkFrame(parent, fg_color="transparent")
    date_row.pack(side="top", fill="x", pady=t.SPACE_XS)
    date_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(date_row, text="Before date (YYYY-MM-DD):", variable=date_var,
                    font=t.font(t.SIZE_BODY), text_color=t.TEXT_PRIMARY, fg_color=t.ACCENT,
                    hover_color=t.ACCENT_HOVER).pack(side="left")
    date_entry = ctk.CTkEntry(date_row, width=120, fg_color=t.SURFACE, border_color=t.BORDER,
                              text_color=t.TEXT_PRIMARY)
    date_entry.pack(side="left", padx=(t.SPACE_SM, 0))

    def _current_options():
        return build_cleanup_options(
            non_mps=cvars["non_mps"].get(), before_date_enabled=date_var.get(),
            date_str=date_entry.get(), suspect=cvars["suspect"].get(),
            unknown=cvars["unknown"].get(), error=cvars["error"].get(),
            no_tracks=cvars["no_tracks"].get(), misclassified_ft=cvars["misclassified_ft"].get(),
            mps_models=getattr(app.config.active_models, "mps_models", []))

    def _preview():
        opts = _current_options()
        if opts is None:
            status.configure(text="Select at least one category (non-MPS needs an MPS list; "
                                  "before-date needs a valid YYYY-MM-DD).")
            return
        status.configure(text="Previewing…")
        _async(lambda: (lambda p: f"Would delete {p['records_to_delete']} of {p['total_records']} "
                        f"records.")(db.preview_cleanup(**opts)))

    def _execute():
        opts = _current_options()
        if opts is None:
            status.configure(text="Select at least one category first.")
            return
        from tkinter import messagebox
        if not messagebox.askyesno("Confirm cleanup",
                                   "Permanently delete the selected records? This cannot be undone."):
            return
        status.configure(text="Deleting…")
        _async(lambda: (lambda r: f"Deleted {sum(r.values()) if isinstance(r, dict) else r} records.")
               (db.execute_cleanup(**opts)))

    btns = ctk.CTkFrame(parent, fg_color="transparent")
    btns.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))
    ctk.CTkButton(btns, text="Preview", command=_preview, fg_color=t.CARD, hover_color=t.ELEVATED,
                  text_color=t.TEXT_PRIMARY, corner_radius=t.RADIUS_SM).pack(side="left")
    ctk.CTkButton(btns, text="Clear selected", command=_execute, fg_color=t.TIER_OOC,
                  hover_color=t.TIER_DRIFT, text_color=t.TEXT_PRIMARY,
                  corner_radius=t.RADIUS_SM).pack(side="left", padx=(t.SPACE_SM, 0))

    def _reset_skipped():
        from tkinter import messagebox

        # Count off-thread, then confirm on the Tk thread, then reset off-thread.
        def runner():
            try:
                count = db.count_skipped_files()
            except Exception as exc:
                status.after(0, lambda: status.winfo_exists() and status.configure(text=f"Error: {exc}"))
                return
            def confirm_and_run():
                if not status.winfo_exists():
                    return
                if count == 0:
                    status.configure(text="No skipped files to reset.")
                    return
                if not messagebox.askyesno("Reset skipped files",
                                           f"Reset {count} skipped files so they get reprocessed next run?"):
                    return
                _async(lambda: f"Reset {db.reset_skipped_files()} skipped files.")
            status.after(0, confirm_and_run)
        threading.Thread(target=runner, daemon=True).start()

    ctk.CTkButton(parent, text="Reset skipped files", command=_reset_skipped, fg_color=t.CARD,
                  hover_color=t.ELEVATED, text_color=t.TEXT_PRIMARY,
                  corner_radius=t.RADIUS_SM).pack(side="top", anchor="w", pady=(t.SPACE_MD, 0))
