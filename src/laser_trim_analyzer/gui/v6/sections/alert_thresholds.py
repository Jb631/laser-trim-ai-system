"""Spec 3d — Alert Thresholds: sensitivity preset + live preview + Save. The primary tool
for reducing the false-positive rate ('flags everything')."""
import threading

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.ui_dispatch import post_ui
from laser_trim_analyzer.gui.v6.widgets.sensitivity_slider import SensitivitySlider
from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS, metric_label
from laser_trim_analyzer.ml.manager import apply_sensitivity_preset, preview_alert_count


def build_alert_thresholds_section(parent, theme: ThemeManager, app) -> None:
    t = theme
    current = getattr(app.config.ml, "drift_sensitivity", "standard")
    state = {"selected": current, "after_id": None}

    ctk.CTkLabel(parent, justify="left", wraplength=640, anchor="w", font=t.font(t.SIZE_BODY),
                 text_color=t.TEXT_SECONDARY,
                 text=("Tighter presets reduce false positives but may miss true drift; looser presets "
                       "surface more at the cost of noise. Preview shows how many models each preset "
                       "would flag against current data."))\
        .pack(side="top", fill="x", anchor="w", pady=(0, t.SPACE_MD))

    preview = ctk.CTkLabel(parent, text="Would flag: — Warning, — Drift, — Out-of-Control",
                           font=t.font(t.SIZE_BODY, "bold"), text_color=t.TEXT_PRIMARY, anchor="w")
    preview.pack(side="top", anchor="w", pady=(0, t.SPACE_SM))

    def refresh_preview(preset):
        def work():
            try:
                c = preview_alert_count(app.db, preset)
            except Exception:
                c = {"warning": 0, "drift": 0, "out_of_control": 0}
            txt = (f"Would flag at '{preset}': {c['warning']} Warning, "
                   f"{c['drift']} Drift, {c['out_of_control']} Out-of-Control")
            post_ui(app, lambda: preview.winfo_exists() and preview.configure(text=txt))
        threading.Thread(target=work, daemon=True).start()

    def on_change(preset):
        state["selected"] = preset
        if state["after_id"] is not None:
            try:
                preview.after_cancel(state["after_id"])
            except Exception:
                pass
        state["after_id"] = preview.after(200, lambda: refresh_preview(preset))

    SensitivitySlider(parent, theme=t, initial=current, on_change=on_change)\
        .pack(side="top", fill="x", pady=(0, t.SPACE_MD))

    # Educational: what each watched metric means — plain language, since this
    # list is the only place a user meets the full metric set (jargon sweep).
    _GLOSS = {
    "linearity_fail_fraction": "share of a lot's units failing linearity — catches lots that fail MORE units even when averages look normal",
        "ft_fail_fraction": "share of a FINAL-TEST lot failing — watches the last, most expensive station",
        "escape_fraction": "share of trim-PASSED units that then failed final test — the trim verdict losing its predictive power",
        "untrimmed_error_max": "worst pre-trim linearity error — incoming element quality",
        "untrimmed_sigma_gradient": "pre-trim noise level of the element",
        "sigma_gradient": "post-trim noise level",
        "untrimmed_resistance": "element resistance before trim",
        "linearity_error": "worst post-trim error vs spec (the customer requirement)",
        "measured_electrical_angle": "measured electrical angle of the element",
        "trim_pass_count": "how many trim passes units needed",
        "resistance_change_percent": "how much trimming moved resistance",
        "max_smoothness_value": "worst output-smoothness deviation",
        "composite_trim_risk_score": "ML score combining the trim-effort signals above",
    }
    desc = ctk.CTkFrame(parent, fg_color="transparent")
    desc.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
    # Same three-section order as the Model page (2026-07-13 design pass).
    from laser_trim_analyzer.ml.drift_types import METRIC_GROUPS
    for group_title, _group_gloss, metrics in METRIC_GROUPS:
        ctk.CTkLabel(desc, text=group_title, font=t.font(t.SIZE_CAPTION, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w")\
            .pack(side="top", fill="x", pady=(t.SPACE_XS, 0))
        for m in metrics:
            if m not in WATCHED_METRICS:
                continue
            gloss = _GLOSS.get(m)
            text = f"• {metric_label(m)}" + (f" — {gloss}" if gloss else "")
            ctk.CTkLabel(desc, text=text, font=t.font(t.SIZE_CAPTION),
                         text_color=t.TEXT_SECONDARY, anchor="w").pack(side="top", fill="x")

    save_btn = ctk.CTkButton(parent, text="Apply preset", fg_color=t.ACCENT, hover_color=t.ACCENT_HOVER,
                             text_color=t.TEXT_INVERSE, corner_radius=t.RADIUS_SM)

    def save():
        preset = state["selected"]
        app.config.ml.drift_sensitivity = preset
        try:
            app.config.save()
        except Exception:
            pass
        # Recompute thresholds so Triage reflects the new preset (get_drifting_models ignores its arg).
        threading.Thread(target=lambda: apply_sensitivity_preset(app.db, preset), daemon=True).start()
        save_btn.configure(text="Applied ✓ (recomputing thresholds…)")
        save_btn.after(2500, lambda: save_btn.winfo_exists() and save_btn.configure(text="Apply preset"))

    save_btn.configure(command=save)
    save_btn.pack(side="top", anchor="w")
    refresh_preview(current)
