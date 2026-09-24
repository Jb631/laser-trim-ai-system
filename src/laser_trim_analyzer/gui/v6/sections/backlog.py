"""Settings — Backlog: one open-order upload sets both the active-models list and
each model's current price (James, 2026-09-20). Replaces the old separate
"Active Models (MPS)" and "Pricing" sections — see core/backlog.py for the parser
and docs/decisions/2026-09-ledger-decisions.md (the E1 brief) for the full design.

`mps_models` stays the ONE list every other screen reads (dashboard cost
priorities, triage, trends, compare, analyze, database cleanup...). This section
only maintains it, as sorted(backlog_models | pinned_models), through
rebuild_active_list() below — called from both the upload path and the pin-save
path so the two can never drift apart.

An upload REPLACES backlog_models/backlog_open_qty (this week's list) but MERGES
model_prices (a model with no open orders this week keeps its last known price —
it's still useful for cost analytics).
"""
import threading
from datetime import date
from pathlib import Path

import customtkinter as ctk

from laser_trim_analyzer.core.backlog import Backlog, BacklogFormatError, parse_backlog
from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.ui_dispatch import post_ui


def rebuild_active_list(cfg) -> None:
    """mps_models is what every screen reads: the last backlog's models plus the manual pins."""
    cfg.mps_models = sorted(set(cfg.backlog_models or []) | set(cfg.pinned_models or []))


def _apply_parsed_backlog(app, backlog: Backlog, source_name: str) -> str:
    """The mutation half of an upload, given an already-parsed Backlog. Split out
    of apply_backlog() so the section can parse once, keep the Backlog object
    around to list unmatched items in the UI, and still funnel the actual state
    change through one place."""
    cfg = app.config.active_models
    cfg.backlog_models = list(backlog.models)
    cfg.backlog_open_qty = dict(backlog.open_qty)
    merged_prices = dict(cfg.model_prices or {})
    merged_prices.update(backlog.prices)          # MERGE: absent-this-week keeps its old price
    cfg.model_prices = merged_prices
    cfg.backlog_source = f"{source_name} · uploaded {date.today().isoformat()}"
    rebuild_active_list(cfg)
    app.config.save()
    return _full_summary(cfg.backlog_source, backlog)


def apply_backlog(app, df, source_name: str) -> str:
    """Parse `df` (already read from the uploaded file) as a backlog export and
    apply it to app.config: REPLACES backlog_models/backlog_open_qty, MERGES
    model_prices, sets backlog_source, rebuilds mps_models, saves, and returns
    the summary line.

    Tk-free and pure aside from the DB read and the config save, so it's directly
    testable without a file dialog or a worker thread.

    Raises BacklogFormatError verbatim on a file that isn't a backlog export;
    parse_backlog runs BEFORE anything on app.config is touched, so a raise here
    leaves the config completely unchanged and unsaved.
    """
    known = app.db.get_known_models()
    backlog = parse_backlog(df, known)
    return _apply_parsed_backlog(app, backlog, source_name)


def _full_summary(source_line: str, backlog: Backlog) -> str:
    """The summary shown right after a successful upload, while the parsed
    Backlog (and so its unmatched count) is still at hand."""
    return (f"{source_line} — {len(backlog.models)} active models · "
            f"{backlog.matched_units:,} open units · {len(backlog.prices)} priced · "
            f"{len(backlog.unmatched)} backlog items not recognised (add-ons such as "
            f"FAI/LAT/TEST UNITS, and products the app does not track)")


def _summary_from_config(cfg) -> str:
    """Rebuild the summary from persisted config alone — used when the section
    opens (e.g. after an app restart), before any upload has happened this
    session. The four config fields don't include the unmatched-items list (it's
    the raw file's Item IDs, not app state worth persisting), so this version
    omits that clause; it comes back as soon as the next upload runs."""
    if not cfg.backlog_source and not cfg.backlog_models:
        return "No backlog uploaded yet — upload one below to set active models and pricing."
    n_models = len(cfg.backlog_models or [])
    n_units = sum((cfg.backlog_open_qty or {}).values())
    prices = cfg.model_prices or {}
    n_priced = sum(1 for m in (cfg.backlog_models or []) if m in prices)
    source = cfg.backlog_source or "(backlog)"
    return (f"{source} — {n_models} active models · {n_units:,} open units · "
            f"{n_priced} priced")


def _migrate_hand_pinned_list(cfg) -> None:
    """One-time upgrade path: before this section existed, mps_models WAS the
    hand-pinned list. If neither pinned_models nor backlog_models has ever been
    set, whatever is in mps_models now was hand-pinned — move it to
    pinned_models so it survives the first backlog upload. Not saved here (per
    the E1 design, docs/decisions/2026-09-ledger-decisions.md); it is saved with whatever the user does next."""
    if cfg.mps_models and not cfg.pinned_models and not cfg.backlog_models:
        cfg.pinned_models = list(cfg.mps_models)


def build_backlog_section(parent, theme: ThemeManager, app) -> None:
    t = theme
    cfg = app.config.active_models
    _migrate_hand_pinned_list(cfg)
    _unmatched = {"items": []}   # populated after each upload this session; not persisted

    ctk.CTkLabel(parent, justify="left", wraplength=640, anchor="w", font=t.font(t.SIZE_BODY),
                 text_color=t.TEXT_SECONDARY,
                 text=("Upload your current open-order backlog to set the active-models list "
                       "and each model's price in one step. Active = a known model with an open "
                       "balance on this backlog; price = the unit price on its most recent order. "
                       "Each upload REPLACES the backlog-derived part of the active-models list "
                       "(pinned models below are unaffected); prices MERGE, so a model with no "
                       "open orders this week keeps its last known price."))\
        .pack(side="top", fill="x", anchor="w", pady=(0, t.SPACE_MD))

    summary = ctk.CTkLabel(parent, text=_summary_from_config(cfg), font=t.font(t.SIZE_CAPTION),
                           text_color=t.TEXT_SECONDARY, anchor="w", justify="left", wraplength=640)
    summary.pack(side="top", fill="x", pady=(0, t.SPACE_XS))

    status = ctk.CTkLabel(parent, text="", font=t.font(t.SIZE_CAPTION),
                          text_color=t.TEXT_SECONDARY, anchor="w")

    unmatched_box = ctk.CTkTextbox(parent, height=90, font=t.font(t.SIZE_CAPTION))

    def _render_unmatched():
        unmatched_box.configure(state="normal")
        unmatched_box.delete("1.0", "end")
        items = _unmatched["items"]
        unmatched_box.insert(
            "1.0", "\n".join(items) if items
            else "(none yet this session — shown here again after each upload)")
        unmatched_box.configure(state="disabled")

    def _toggle_unmatched():
        if unmatched_box.winfo_ismapped():
            unmatched_box.pack_forget()
            toggle_btn.configure(text="Show unrecognised items")
        else:
            _render_unmatched()
            unmatched_box.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
            toggle_btn.configure(text="Hide unrecognised items")

    def _upload():
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select current backlog export",
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv"), ("All files", "*.*")])
        if not path:
            return
        status.configure(text="Reading backlog…")

        def work():
            try:
                import pandas as pd
                p = Path(path)
                df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
                known = app.db.get_known_models()
                backlog = parse_backlog(df, known)
                summary_text = _apply_parsed_backlog(app, backlog, p.name)
                unmatched_items = list(backlog.unmatched)
                msg = "Backlog applied."
                failed = False
            except BacklogFormatError as exc:
                msg, summary_text, unmatched_items, failed = str(exc), None, None, True
            except Exception as exc:
                msg = f"Backlog import failed: {exc}"
                summary_text, unmatched_items, failed = None, None, True

            def apply_ui():
                if status.winfo_exists():
                    status.configure(text=msg)
                if not failed:
                    if summary.winfo_exists():
                        summary.configure(text=summary_text)
                    _unmatched["items"] = unmatched_items or []
                    if unmatched_box.winfo_exists() and unmatched_box.winfo_ismapped():
                        _render_unmatched()
            post_ui(app, apply_ui)
        threading.Thread(target=work, daemon=True).start()

    upload_btn = ctk.CTkButton(parent, text="Upload current backlog…", command=_upload,
                               fg_color=t.ACCENT, hover_color=t.ACCENT_HOVER,
                               text_color=t.TEXT_INVERSE, corner_radius=t.RADIUS_SM)
    upload_btn.pack(side="top", anchor="w", pady=(0, t.SPACE_XS))

    toggle_btn = ctk.CTkButton(parent, text="Show unrecognised items", command=_toggle_unmatched,
                               fg_color=t.CARD, hover_color=t.ELEVATED, text_color=t.TEXT_PRIMARY,
                               corner_radius=t.RADIUS_SM)
    toggle_btn.pack(side="top", anchor="w", pady=(0, t.SPACE_SM))

    status.pack(side="top", fill="x", pady=(0, t.SPACE_SM))

    ctk.CTkLabel(parent, text="Pinned models — active even with no open orders, one per line:",
                font=t.font(t.SIZE_BODY), text_color=t.TEXT_PRIMARY, anchor="w")\
        .pack(side="top", fill="x", pady=(t.SPACE_SM, t.SPACE_XS))
    pinned_box = ctk.CTkTextbox(parent, height=100, font=t.font(t.SIZE_BODY))
    pinned_box.pack(side="top", fill="x")
    if cfg.pinned_models:
        pinned_box.insert("1.0", "\n".join(cfg.pinned_models))

    def _entry_row(label, value):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(side="top", fill="x", pady=t.SPACE_XS)
        ctk.CTkLabel(frame, text=label, width=180, anchor="w", font=t.font(t.SIZE_BODY),
                     text_color=t.TEXT_SECONDARY).pack(side="left")
        entry = ctk.CTkEntry(frame, width=120, fg_color=t.SURFACE, border_color=t.BORDER,
                             text_color=t.TEXT_PRIMARY)
        entry.insert(0, str(value))
        entry.pack(side="left")
        return entry

    recent_days_entry = _entry_row("Recent days (1–365)", cfg.recent_days)
    cost_ratio_entry = _entry_row("Cost ratio (0.01–1.0)", cfg.cost_ratio)

    save_status = ctk.CTkLabel(parent, text="", font=t.font(t.SIZE_CAPTION),
                               text_color=t.TEXT_SECONDARY, anchor="w")

    def _pinned_from_box():
        seen, out = set(), []
        for line in pinned_box.get("1.0", "end").splitlines():
            m = line.strip()
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def _save():
        cfg.pinned_models = _pinned_from_box()
        try:
            cfg.recent_days = max(1, min(365, int(recent_days_entry.get())))
        except (ValueError, TypeError):
            pass
        try:
            cfg.cost_ratio = max(0.01, min(1.0, float(cost_ratio_entry.get())))
        except (ValueError, TypeError):
            pass
        rebuild_active_list(cfg)
        try:
            app.config.save()
            save_status.configure(text="Saved.")
        except Exception as exc:
            save_status.configure(text=f"Save failed: {exc}")
        summary.configure(text=_summary_from_config(cfg))

    ctk.CTkButton(parent, text="Save", command=_save, fg_color=t.ACCENT, hover_color=t.ACCENT_HOVER,
                 text_color=t.TEXT_INVERSE, corner_radius=t.RADIUS_SM)\
        .pack(side="top", anchor="w", pady=(t.SPACE_SM, 0))
    save_status.pack(side="top", fill="x")

    ctk.CTkLabel(parent, justify="left", wraplength=640, anchor="w", font=t.font(t.SIZE_CAPTION),
                 text_color=t.TEXT_DISABLED,
                 text=("Database Cleanup's “non-MPS” option treats every model not on "
                       "this list as removable. With the list now coming from the current "
                       "backlog, that means any model with no open orders this week — do not "
                       "use it to tidy up."))\
        .pack(side="top", fill="x", anchor="w", pady=(t.SPACE_MD, 0))
