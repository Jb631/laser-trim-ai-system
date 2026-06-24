"""Spec 3d — Per-model Specs: edit linearity spec + exclude_points (real V5 port).

exclude_points feeds ML correctness (it can flip a unit's FAIL<->PASS labeling,
per CLAUDE.md), so this is a correctness feature, not cosmetic. The human<->JSON
conversion is factored into build_spec_save_data so it can be tested without Tk.
"""
import customtkinter as ctk

from laser_trim_analyzer.core.analyzer import (
    format_exclude_points, human_to_exclude_json, parse_exclude_points)
from laser_trim_analyzer.gui.v6.theme import ThemeManager


def build_spec_save_data(model, linearity_spec_text, linearity_spec_pct,
                         exclude_trim, exclude_ft) -> dict:
    """Build a db.save_model_spec() data dict from raw field strings.

    Converts the human exclude-point strings ('0-2, 48-50') to JSON storage via
    human_to_exclude_json, and the % string to float. Empty fields become None.
    """
    def _f(v):
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        try:
            return float(v)
        except ValueError:
            return None

    return {
        "model": (model or "").strip(),
        "linearity_spec_text": (linearity_spec_text or "").strip() or None,
        "linearity_spec_pct": _f(linearity_spec_pct),
        "exclude_points": human_to_exclude_json(exclude_trim or ""),
        "exclude_points_ft": human_to_exclude_json(exclude_ft or ""),
    }


def _display_excludes(stored_json) -> str:
    """Stored exclude JSON -> human display ('0-2, 48-50')."""
    return format_exclude_points(parse_exclude_points(stored_json))


def build_per_model_specs_section(parent, theme: ThemeManager, app) -> None:
    t = theme
    db = app.db

    ctk.CTkLabel(parent, justify="left", wraplength=640, anchor="w", font=t.font(t.SIZE_BODY),
                 text_color=t.TEXT_SECONDARY,
                 text=("Per-model linearity spec and excluded measurement points. Excluded points "
                       "are removed before pass/fail evaluation, so they directly affect ML labels "
                       "— edit with care. Format: '0-2, 48-50'."))\
        .pack(side="top", fill="x", anchor="w", pady=(0, t.SPACE_MD))

    def _model_values():
        try:
            return [s["model"] for s in db.get_all_model_specs()]
        except Exception:
            return []

    fields = {}

    def _row(label):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(side="top", fill="x", pady=t.SPACE_XS)
        ctk.CTkLabel(frame, text=label, width=180, anchor="w", font=t.font(t.SIZE_BODY),
                     text_color=t.TEXT_SECONDARY).pack(side="left")
        entry = ctk.CTkEntry(frame, fg_color=t.SURFACE, border_color=t.BORDER,
                             text_color=t.TEXT_PRIMARY)
        entry.pack(side="left", fill="x", expand=True)
        return entry

    model_box = ctk.CTkComboBox(parent, values=_model_values(), fg_color=t.SURFACE,
                                border_color=t.BORDER, button_color=t.ACCENT,
                                button_hover_color=t.ACCENT_HOVER, text_color=t.TEXT_PRIMARY,
                                command=lambda choice: _load(choice))
    model_box.set("")
    model_box.pack(side="top", fill="x", pady=(0, t.SPACE_SM))

    # --- Bulk import: load/refresh ALL model specs from the master reference sheet
    # (Model Reference workbook). Reuses db.import_model_specs_from_excel, which merges
    # (updates existing, adds new, never deletes) and auto-detects columns.
    def _import_sheet():
        from tkinter import filedialog
        import threading
        path = filedialog.askopenfilename(
            title="Select model reference spec sheet",
            filetypes=[("Excel", "*.xlsx"), ("Excel 97-2003", "*.xls")])
        if not path:
            return
        import_status.configure(text="Importing…")

        def work():
            try:
                res = db.import_model_specs_from_excel(path)
                msg = (f"Imported: {res.get('added', 0)} added, "
                       f"{res.get('updated', 0)} updated, {res.get('skipped', 0)} skipped.")
            except Exception as exc:
                msg = f"Import failed: {exc}"

            def done():
                import_status.configure(text=msg)
                try:
                    model_box.configure(values=_model_values())
                except Exception:
                    pass
            try:
                import_status.after(0, done)
            except Exception:
                pass
        threading.Thread(target=work, daemon=True).start()

    imp_row = ctk.CTkFrame(parent, fg_color="transparent")
    imp_row.pack(side="top", fill="x", pady=(0, t.SPACE_XS))
    ctk.CTkButton(imp_row, text="Import spec sheet…", command=_import_sheet,
                  fg_color=t.ACCENT, hover_color=t.ACCENT_HOVER, text_color=t.TEXT_INVERSE,
                  corner_radius=t.RADIUS_SM).pack(side="left")
    ctk.CTkLabel(imp_row, text="  Bulk load/refresh every model from your master sheet "
                               "(merges — never deletes).",
                 font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY).pack(side="left")
    import_status = ctk.CTkLabel(parent, text="", font=t.font(t.SIZE_CAPTION),
                                 text_color=t.TEXT_SECONDARY, anchor="w")
    import_status.pack(side="top", fill="x", pady=(0, t.SPACE_SM))

    fields["linearity_spec_text"] = _row("Linearity spec (text)")
    fields["linearity_spec_pct"] = _row("Linearity spec (%)")
    fields["exclude_points"] = _row("Exclude points (Trim)")
    fields["exclude_points_ft"] = _row("Exclude points (FT)")

    status = ctk.CTkLabel(parent, text="", font=t.font(t.SIZE_CAPTION),
                          text_color=t.TEXT_SECONDARY, anchor="w")
    status.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))

    def _set(entry, value):
        entry.delete(0, "end")
        if value is not None:
            entry.insert(0, str(value))

    def _load(model):
        spec = None
        try:
            spec = db.get_model_spec(model)
        except Exception:
            spec = None
        if not spec:
            status.configure(text=f"No saved spec for {model} — fill in to create one.")
            for key in ("linearity_spec_text", "linearity_spec_pct", "exclude_points", "exclude_points_ft"):
                _set(fields[key], "")
            return
        _set(fields["linearity_spec_text"], spec.get("linearity_spec_text"))
        _set(fields["linearity_spec_pct"], spec.get("linearity_spec_pct"))
        _set(fields["exclude_points"], _display_excludes(spec.get("exclude_points")))
        _set(fields["exclude_points_ft"], _display_excludes(spec.get("exclude_points_ft")))
        status.configure(text=f"Loaded {model}.")

    def _save():
        data = build_spec_save_data(
            model_box.get(), fields["linearity_spec_text"].get(), fields["linearity_spec_pct"].get(),
            fields["exclude_points"].get(), fields["exclude_points_ft"].get())
        if not data["model"]:
            status.configure(text="Enter a model name first.")
            return
        try:
            _id, updated = db.save_model_spec(data)
            model_box.configure(values=_model_values())
            status.configure(text=f"{'Updated' if updated else 'Created'} {data['model']}.")
        except Exception as exc:
            status.configure(text=f"Save failed: {exc}")

    def _delete():
        model = model_box.get().strip()
        if not model:
            status.configure(text="Enter a model name first.")
            return
        try:
            ok = db.delete_model_spec(model)
            model_box.configure(values=_model_values())
            status.configure(text=f"Deleted {model}." if ok else f"No spec for {model}.")
        except Exception as exc:
            status.configure(text=f"Delete failed: {exc}")

    btns = ctk.CTkFrame(parent, fg_color="transparent")
    btns.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))
    ctk.CTkButton(btns, text="Save spec", command=_save, fg_color=t.ACCENT, hover_color=t.ACCENT_HOVER,
                  text_color=t.TEXT_INVERSE, corner_radius=t.RADIUS_SM).pack(side="left")
    ctk.CTkButton(btns, text="Delete spec", command=_delete, fg_color=t.CARD, hover_color=t.ELEVATED,
                  text_color=t.TEXT_PRIMARY, corner_radius=t.RADIUS_SM).pack(side="left", padx=(t.SPACE_SM, 0))
