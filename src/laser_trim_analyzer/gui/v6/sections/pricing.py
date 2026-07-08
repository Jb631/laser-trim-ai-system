"""Spec 3d — Pricing: cost ratio + recent-days config + model-price import (real V5 port)."""
import threading

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.ui_dispatch import post_ui

_MODEL_COLS = ('model', 'item id', 'item_id', 'itemid', 'part', 'part number')
_PRICE_COLS = ('price', 'unit price', 'unit_price', 'unitprice', 'cost', 'unit cost')


def extract_model_prices(df) -> dict:
    """Flexible model/price column match → {model: representative_price}. Pure port of
    V5 _import_pricing's core so it's testable without a file dialog. Uses the most
    common non-zero price per model (median tiebreak). Returns {} if columns missing."""
    model_col = price_col = None
    for col in df.columns:
        cl = str(col).lower()
        if cl in _MODEL_COLS:
            model_col = col
        elif cl in _PRICE_COLS:
            price_col = col
    if model_col is None or price_col is None:
        return {}
    prices = {}
    for model_val, group in df.groupby(model_col):
        model = str(model_val).strip()
        if not model:
            continue
        non_zero = group[group[price_col] > 0][price_col]
        if len(non_zero) > 0:
            mode_vals = non_zero.mode()
            prices[model] = float(mode_vals.iloc[0]) if len(mode_vals) > 0 else float(non_zero.median())
    return prices


def build_pricing_section(parent, theme: ThemeManager, app) -> None:
    t = theme
    am = app.config.active_models

    ctk.CTkLabel(parent, justify="left", wraplength=640, anchor="w", font=t.font(t.SIZE_BODY),
                 text_color=t.TEXT_SECONDARY,
                 text=("Cost model for impact analysis. Cost ratio is the fraction of unit price lost "
                       "to a late (final-test) failure; recent-days bounds the 'recent' window. Import "
                       "per-model prices from an Excel/CSV with Model/Item-ID and Unit-Price columns."))\
        .pack(side="top", fill="x", anchor="w", pady=(0, t.SPACE_MD))

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

    cost_ratio_entry = _entry_row("Cost ratio (0.01–1.0)", am.cost_ratio)
    recent_days_entry = _entry_row("Recent days (1–365)", am.recent_days)

    count = ctk.CTkLabel(parent, text="", font=t.font(t.SIZE_CAPTION),
                         text_color=t.TEXT_SECONDARY, anchor="w")
    count.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))

    def _summary():
        n = len(app.config.active_models.model_prices or {})
        return f"{n} models with pricing" if n else "No pricing data loaded"

    count.configure(text=_summary())

    status = ctk.CTkLabel(parent, text="", font=t.font(t.SIZE_CAPTION),
                          text_color=t.TEXT_SECONDARY, anchor="w")
    status.pack(side="top", fill="x")

    def _save():
        try:
            am.cost_ratio = max(0.01, min(1.0, float(cost_ratio_entry.get())))
        except (ValueError, TypeError):
            pass
        try:
            am.recent_days = max(1, min(365, int(recent_days_entry.get())))
        except (ValueError, TypeError):
            pass
        try:
            app.config.save()
            status.configure(text="Saved cost settings.")
        except Exception as exc:
            status.configure(text=f"Save failed: {exc}")

    def _import():
        from tkinter import filedialog
        from pathlib import Path
        path = filedialog.askopenfilename(
            title="Select pricing file",
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv"), ("All files", "*.*")])
        if not path:
            return
        status.configure(text="Importing…")

        def work():
            try:
                import pandas as pd
                p = Path(path)
                df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
                prices = extract_model_prices(df)
                if not prices:
                    msg = "No model/price columns found in file."
                else:
                    existing = app.config.active_models.model_prices or {}
                    existing.update(prices)
                    app.config.active_models.model_prices = existing
                    app.config.save()
                    msg = f"Imported {len(prices)} prices ({len(existing)} total)."
            except Exception as exc:
                msg = f"Import failed: {exc}"

            def apply():
                if status.winfo_exists():
                    status.configure(text=msg)
                if count.winfo_exists():
                    count.configure(text=_summary())
            post_ui(app, apply)
        threading.Thread(target=work, daemon=True).start()

    def _clear():
        app.config.active_models.model_prices = {}
        try:
            app.config.save()
        except Exception:
            pass
        count.configure(text=_summary())
        status.configure(text="Cleared all pricing.")

    btns = ctk.CTkFrame(parent, fg_color="transparent")
    btns.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))
    ctk.CTkButton(btns, text="Save cost settings", command=_save, fg_color=t.ACCENT,
                  hover_color=t.ACCENT_HOVER, text_color=t.TEXT_INVERSE,
                  corner_radius=t.RADIUS_SM).pack(side="left")
    ctk.CTkButton(btns, text="Import pricing", command=_import, fg_color=t.CARD,
                  hover_color=t.ELEVATED, text_color=t.TEXT_PRIMARY,
                  corner_radius=t.RADIUS_SM).pack(side="left", padx=(t.SPACE_SM, 0))
    ctk.CTkButton(btns, text="Clear pricing", command=_clear, fg_color=t.CARD,
                  hover_color=t.ELEVATED, text_color=t.TEXT_PRIMARY,
                  corner_radius=t.RADIUS_SM).pack(side="left", padx=(t.SPACE_SM, 0))
