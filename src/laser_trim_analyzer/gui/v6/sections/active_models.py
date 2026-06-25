"""Active Models (MPS): pin which models count as 'active' for Triage's default focus.

Triage's Active view hides legacy/quiet models so 'what to look at today' isn't drowned
by parts last run in 2016. A model is active if it has data within the recency window OR
it's pinned here. Pin models you're currently running that may have gone quiet in the
loaded data. Persists to config.active_models (mps_models, recent_days); Triage reads it.
"""
import threading

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.manager import active_model_set, list_known_models


def build_active_models_section(parent, theme: ThemeManager, app) -> None:
    t = theme
    cfg = app.config.active_models

    ctk.CTkLabel(parent, justify="left", wraplength=640, anchor="w", font=t.font(t.SIZE_BODY),
                 text_color=t.TEXT_SECONDARY,
                 text=("Triage focuses on ACTIVE models. A model is active if it has data within "
                       "the recency window below, OR you pin it on the MPS list. Pin models you're "
                       "running now that may have gone quiet in the loaded data so they still "
                       "appear in Triage's default (Active) view."))\
        .pack(side="top", fill="x", anchor="w", pady=(0, t.SPACE_MD))

    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
    ctk.CTkLabel(row, text="Recency window (days):", font=t.font(t.SIZE_BODY),
                 text_color=t.TEXT_PRIMARY).pack(side="left", padx=(0, t.SPACE_SM))
    days_entry = ctk.CTkEntry(row, width=80, font=t.font(t.SIZE_BODY))
    days_entry.insert(0, str(getattr(cfg, "recent_days", 90)))
    days_entry.pack(side="left")
    days_entry.bind("<Return>", lambda e: refresh_status())

    ctk.CTkLabel(parent, text="Pinned models (one per line):", font=t.font(t.SIZE_BODY),
                 text_color=t.TEXT_PRIMARY, anchor="w")\
        .pack(side="top", fill="x", pady=(t.SPACE_SM, t.SPACE_XS))
    box = ctk.CTkTextbox(parent, height=140, font=t.font(t.SIZE_BODY))
    box.pack(side="top", fill="x")
    if cfg.mps_models:
        box.insert("1.0", "\n".join(cfg.mps_models))

    status = ctk.CTkLabel(parent, text="", font=t.font(t.SIZE_CAPTION),
                          text_color=t.TEXT_SECONDARY, anchor="w")
    status.pack(side="top", fill="x", pady=(t.SPACE_XS, t.SPACE_SM))

    def _pinned_from_box():
        seen, out = set(), []
        for line in box.get("1.0", "end").splitlines():
            m = line.strip()
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def refresh_status():
        def work():
            try:
                days = max(1, int((days_entry.get() or "90").strip()))
            except ValueError:
                days = 90
            try:
                auto = active_model_set(app.db, recent_days=days, mps_models=[])
                known = {m.model for m in list_known_models(app.db)}
            except Exception:
                auto, known = set(), set()
            pinned = set(_pinned_from_box())
            total_active = auto | pinned
            extra = pinned - auto                  # pinned models recency wouldn't catch
            unknown = pinned - known if known else set()  # typos / not in DB yet
            txt = (f"{len(total_active)} active of {len(known)} known models "
                   f"— {len(auto)} by recency, {len(extra)} pinned-only.")
            if unknown:
                sample = ", ".join(sorted(unknown)[:5])
                txt += f"  ⚠ {len(unknown)} pinned not found in data: {sample}"
            try:
                if status.winfo_exists():
                    status.after(0, lambda: status.winfo_exists() and status.configure(text=txt))
            except Exception:
                pass
        threading.Thread(target=work, daemon=True).start()

    save_btn = ctk.CTkButton(parent, text="Save", fg_color=t.ACCENT, hover_color=t.ACCENT_HOVER,
                             text_color=t.TEXT_INVERSE, corner_radius=t.RADIUS_SM)

    def save():
        cfg.mps_models = _pinned_from_box()
        try:
            cfg.recent_days = max(1, int((days_entry.get() or "90").strip()))
        except ValueError:
            cfg.recent_days = 90
        try:
            app.config.save()
        except Exception:
            pass
        save_btn.configure(text="Saved ✓")
        save_btn.after(1500, lambda: save_btn.winfo_exists() and save_btn.configure(text="Save"))
        refresh_status()

    save_btn.configure(command=save)
    save_btn.pack(side="top", anchor="w")
    refresh_status()
