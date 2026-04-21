"""
Specs Page - Model Engineering Specifications.

View and edit model specs: element type, product class, linearity type,
resistance range, electrical angle, etc.
"""

import customtkinter as ctk
import logging
from tkinter import messagebox
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class SpecsPage(ctk.CTkFrame):
    """
    Model Specs page for viewing and editing engineering specifications.

    Features:
    - Scrollable table of all model specs
    - Search/filter by model number
    - Add/Edit/Delete model specs
    - Inline detail panel for editing
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._specs: List[Dict[str, Any]] = []
        self._filtered_specs: List[Dict[str, Any]] = []
        self._selected_model: Optional[str] = None
        self._row_frames: List[ctk.CTkFrame] = []

        self._create_ui()

    def _create_ui(self):
        """Create the specs page UI."""
        self.grid_columnconfigure(0, weight=3)  # Table
        self.grid_columnconfigure(1, weight=2)  # Edit panel
        self.grid_rowconfigure(2, weight=1)

        # Header row
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")
        header_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header_frame,
            text="Model Specifications",
            font=ctk.CTkFont(size=24, weight="bold")
        ).grid(row=0, column=0, sticky="w")

        self._count_label = ctk.CTkLabel(
            header_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self._count_label.grid(row=0, column=1, padx=20, sticky="w")

        ctk.CTkButton(
            header_frame,
            text="+ Add Model",
            width=120,
            command=self._add_new
        ).grid(row=0, column=2, padx=5)

        self._discrepancy_btn = ctk.CTkButton(
            header_frame,
            text="Check Discrepancies",
            width=160,
            fg_color="#2980b9",
            command=self._check_discrepancies
        )
        self._discrepancy_btn.grid(row=0, column=3, padx=5)

        # Search bar
        search_frame = ctk.CTkFrame(self, fg_color="transparent")
        search_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="ew")
        search_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(search_frame, text="Search:").grid(row=0, column=0, padx=(0, 10))
        self._search_var = ctk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._on_search())
        self._search_entry = ctk.CTkEntry(
            search_frame,
            textvariable=self._search_var,
            placeholder_text="Filter by model number...",
            width=300
        )
        self._search_entry.grid(row=0, column=1, sticky="w")

        # Table area (left)
        self._table_frame = ctk.CTkScrollableFrame(self)
        self._table_frame.grid(row=2, column=0, sticky="nsew", padx=(20, 10), pady=(0, 20))
        self._table_frame.grid_columnconfigure(0, weight=1)

        # Column headers
        self._create_table_header()

        # Edit panel (right)
        self._edit_frame = ctk.CTkScrollableFrame(self)
        self._edit_frame.grid(row=2, column=1, sticky="nsew", padx=(10, 20), pady=(0, 20))
        self._edit_frame.grid_columnconfigure(1, weight=1)

        self._create_edit_panel()

    def _create_table_header(self):
        """Create table column headers."""
        header = ctk.CTkFrame(self._table_frame, fg_color=("gray80", "gray25"))
        header.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        header.grid_columnconfigure(0, weight=2)
        header.grid_columnconfigure(1, weight=2)
        header.grid_columnconfigure(2, weight=2)
        header.grid_columnconfigure(3, weight=2)
        header.grid_columnconfigure(4, weight=3)

        cols = ["Model", "Element Type", "Product Class", "Linearity", "Resistance"]
        for i, col in enumerate(cols):
            ctk.CTkLabel(
                header, text=col,
                font=ctk.CTkFont(size=12, weight="bold"),
                anchor="w"
            ).grid(row=0, column=i, padx=8, pady=6, sticky="w")

    def _create_edit_panel(self):
        """Create the detail edit panel."""
        ctk.CTkLabel(
            self._edit_frame,
            text="Model Details",
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 15), sticky="w")

        # Fields
        self._edit_fields = {}
        fields = [
            ("model", "Model Number", "entry"),
            ("element_type", "Element Type", "combo",
             ["Ceramic", "Winding", "Black", "Blue", "G11", "Infinatron"]),
            ("product_class", "Product Class", "combo",
             ["Runner", "Low Vol", "Space", "Panel", "Rotary"]),
            ("linearity_type", "Linearity Type", "combo",
             ["Absolute", "Independent", "Term Base", "Zero-Based", "VR Max", "Custom"]),
            ("linearity_spec_text", "Linearity Spec Text", "entry"),
            ("linearity_spec_pct", "Linearity Spec %", "entry"),
            ("total_resistance_min", "Resistance Min (Ω)", "entry"),
            ("total_resistance_max", "Resistance Max (Ω)", "entry"),
            ("electrical_angle", "Electrical Angle", "entry"),
            ("electrical_angle_tol", "Angle Tolerance", "entry"),
            # How to interpret electrical_angle_tol. The slope-from-tolerance
            # rule reads this to decide whether slope can be adjusted
            # symmetrically, one-sided, etc. Leave blank when there's no
            # tolerance on the print.
            ("electrical_angle_tol_type", "Tol Type", "combo",
             ["", "symmetric", "min", "max", "range", "bilateral"]),
            ("electrical_angle_unit", "Angle Unit", "combo", ["in", "deg"]),
            ("output_smoothness", "Output Smoothness", "entry"),
            # Open = visible resistive element, Closed = enclosed. Imported
            # from the Open/Closed column on the Model Reference sheet.
            ("open_closed", "Open/Closed", "combo", ["", "Open", "Closed"]),
            # Pipe-separated alternate model numbers that should resolve to
            # this spec. Example: "2001621501 | 1621501-R1"
            ("aliases", "Aliases (pipe-separated)", "entry"),
            ("notes", "Notes", "textbox"),
        ]

        for i, field_def in enumerate(fields):
            field_name = field_def[0]
            label = field_def[1]
            field_type = field_def[2]

            ctk.CTkLabel(
                self._edit_frame, text=label,
                font=ctk.CTkFont(size=12),
                anchor="w"
            ).grid(row=i + 1, column=0, padx=10, pady=4, sticky="w")

            if field_type == "entry":
                widget = ctk.CTkEntry(self._edit_frame, width=200)
                widget.grid(row=i + 1, column=1, padx=10, pady=4, sticky="ew")
            elif field_type == "combo":
                values = field_def[3]
                widget = ctk.CTkComboBox(
                    self._edit_frame, values=[""] + values, width=200
                )
                widget.grid(row=i + 1, column=1, padx=10, pady=4, sticky="ew")
            elif field_type == "textbox":
                widget = ctk.CTkTextbox(self._edit_frame, height=60, width=200)
                widget.grid(row=i + 1, column=1, padx=10, pady=4, sticky="ew")

            self._edit_fields[field_name] = (field_type, widget)

        # Buttons
        btn_row = len(fields) + 1
        btn_frame = ctk.CTkFrame(self._edit_frame, fg_color="transparent")
        btn_frame.grid(row=btn_row, column=0, columnspan=2, padx=10, pady=15, sticky="ew")

        ctk.CTkButton(
            btn_frame, text="Save", width=100,
            fg_color="#27ae60", hover_color="#219a52",
            command=self._save_spec
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_frame, text="Delete", width=100,
            fg_color="#e74c3c", hover_color="#c0392b",
            command=self._delete_spec
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_frame, text="Clear", width=100,
            fg_color="transparent", border_width=1,
            command=self._clear_edit_panel
        ).pack(side="left")

    def on_show(self):
        """Called when page becomes visible."""
        from laser_trim_analyzer.utils.threads import get_thread_manager

        def _load_all():
            try:
                from laser_trim_analyzer.database import get_database
                db = get_database()
                specs = db.get_all_model_specs()
                discrepancies = db.get_spec_discrepancies()
                self.after(0, lambda: self._on_specs_loaded(specs, discrepancies))
            except Exception as e:
                logger.error(f"Failed to load specs: {e}")
                self.after(0, lambda: self._on_specs_loaded([], []))

        get_thread_manager().start_thread(target=_load_all, name="specs-load")

    def _on_specs_loaded(self, specs, discrepancies):
        """Handle specs loaded from background thread."""
        if not self.winfo_exists():
            return
        self._specs = specs
        self._on_search()

        count = len(discrepancies) if discrepancies else 0
        if count > 0:
            self._discrepancy_btn.configure(
                text=f"Discrepancies ({count})",
                fg_color="#e74c3c"
            )
        else:
            self._discrepancy_btn.configure(
                text="Check Discrepancies",
                fg_color="#2980b9"
            )

    def on_hide(self):
        """Called when page is hidden - free 300+ rendered row widgets so
        the next page renders smoothly. Without this, the row frames stay
        alive in the geometry manager and the GUI feels sluggish for the
        rest of the session.
        """
        try:
            for frame in self._row_frames:
                try:
                    frame.destroy()
                except Exception:
                    pass
            self._row_frames.clear()
        except Exception as e:
            logger.debug(f"Specs on_hide cleanup warning: {e}")

    def _load_specs(self):
        """Load all specs from database."""
        try:
            from laser_trim_analyzer.database import get_database
            db = get_database()
            self._specs = db.get_all_model_specs()
            self._on_search()  # Apply current filter
        except Exception as e:
            logger.error(f"Failed to load model specs: {e}")
            self._specs = []
            self._update_table([])

    def _on_search(self):
        """Filter table rows by search text."""
        query = self._search_var.get().strip().lower()
        if query:
            self._filtered_specs = [
                s for s in self._specs
                if query in s["model"].lower()
                or (s.get("element_type") and query in s["element_type"].lower())
                or (s.get("product_class") and query in s["product_class"].lower())
            ]
        else:
            self._filtered_specs = self._specs[:]
        self._update_table(self._filtered_specs)

    def _update_table(self, specs: List[Dict[str, Any]]):
        """Rebuild the table with given specs."""
        # Clear existing rows
        for frame in self._row_frames:
            frame.destroy()
        self._row_frames.clear()

        # Count incomplete specs for the summary
        incomplete = sum(1 for s in specs if not s.get("element_type") or not s.get("linearity_type"))
        count_text = f"{len(specs)} models"
        if incomplete > 0:
            count_text += f" ({incomplete} incomplete — shown in red)"
        self._count_label.configure(text=count_text)

        for i, spec in enumerate(specs):
            bg = ("gray90", "gray17") if i % 2 == 0 else ("gray85", "gray20")
            row = ctk.CTkFrame(self._table_frame, fg_color=bg, corner_radius=0)
            row.grid(row=i + 1, column=0, sticky="ew", pady=0)
            row.grid_columnconfigure(0, weight=2)
            row.grid_columnconfigure(1, weight=2)
            row.grid_columnconfigure(2, weight=2)
            row.grid_columnconfigure(3, weight=2)
            row.grid_columnconfigure(4, weight=3)

            model = spec["model"]

            # Highlight incomplete specs
            missing = []
            if not spec.get("element_type"):
                missing.append("element")
            if not spec.get("linearity_type"):
                missing.append("linearity")

            model_color = "#e74c3c" if missing else ("gray10", "gray90")

            vals = [
                spec["model"],
                spec.get("element_type") or "—",
                spec.get("product_class") or "—",
                spec.get("linearity_type") or "—",
                self._format_resistance(spec),
            ]

            for j, val in enumerate(vals):
                color = model_color if j == 0 and missing else ("gray10", "gray90")
                lbl = ctk.CTkLabel(
                    row, text=str(val),
                    font=ctk.CTkFont(size=12),
                    anchor="w",
                    text_color=color,
                    cursor="hand2"
                )
                lbl.grid(row=0, column=j, padx=8, pady=5, sticky="w")
                lbl.bind("<Button-1>", lambda e, m=model: self._on_select(m))

            row.bind("<Button-1>", lambda e, m=model: self._on_select(m))
            self._row_frames.append(row)

    def _format_resistance(self, spec: Dict[str, Any]) -> str:
        """Format resistance range for display."""
        r_min = spec.get("total_resistance_min")
        r_max = spec.get("total_resistance_max")
        if r_min is not None and r_max is not None:
            return f"{r_min:,.0f} - {r_max:,.0f} Ω"
        return "—"

    def _on_select(self, model: str):
        """Show selected model in edit panel."""
        self._selected_model = model
        spec = next((s for s in self._specs if s["model"] == model), None)
        if not spec:
            return

        for field_name, (field_type, widget) in self._edit_fields.items():
            value = spec.get(field_name)
            if value is None:
                value = ""
            else:
                value = str(value)

            if field_type == "entry":
                widget.delete(0, "end")
                widget.insert(0, value)
            elif field_type == "combo":
                widget.set(value)
            elif field_type == "textbox":
                widget.delete("1.0", "end")
                widget.insert("1.0", value)

    def _add_new(self):
        """Clear edit panel for new model entry."""
        self._selected_model = None
        self._clear_edit_panel()
        # Focus the model number field
        _, widget = self._edit_fields["model"]
        widget.focus_set()

    def _clear_edit_panel(self):
        """Clear all fields in the edit panel."""
        self._selected_model = None
        for field_name, (field_type, widget) in self._edit_fields.items():
            if field_type == "entry":
                widget.delete(0, "end")
            elif field_type == "combo":
                widget.set("")
            elif field_type == "textbox":
                widget.delete("1.0", "end")

    def _save_spec(self):
        """Save the current edit panel data to DB."""
        data = {}
        for field_name, (field_type, widget) in self._edit_fields.items():
            if field_type == "entry":
                val = widget.get().strip()
            elif field_type == "combo":
                val = widget.get().strip()
            elif field_type == "textbox":
                val = widget.get("1.0", "end").strip()
            else:
                val = ""

            # Convert numeric fields
            if field_name in ("linearity_spec_pct", "total_resistance_min",
                              "total_resistance_max", "electrical_angle",
                              "electrical_angle_tol"):
                if val:
                    try:
                        val = float(val)
                    except ValueError:
                        messagebox.showerror("Invalid Input", f"{field_name} must be a number")
                        return
                else:
                    val = None

            # Convert empty strings to None
            if val == "":
                val = None

            data[field_name] = val

        if not data.get("model"):
            messagebox.showerror("Missing Model", "Model number is required")
            return

        try:
            from laser_trim_analyzer.database import get_database
            db = get_database()
            db.save_model_spec(data)
            self._load_specs()
            # Re-select the saved model
            self._on_select(data["model"])
        except Exception as e:
            logger.error(f"Failed to save model spec: {e}")
            messagebox.showerror("Save Error", str(e))

    def _delete_spec(self):
        """Delete the currently selected model spec."""
        if not self._selected_model:
            messagebox.showinfo("No Selection", "Select a model to delete")
            return

        if not messagebox.askyesno(
            "Confirm Delete",
            f"Delete spec for model '{self._selected_model}'?\n\nThis cannot be undone."
        ):
            return

        try:
            from laser_trim_analyzer.database import get_database
            db = get_database()
            db.delete_model_spec(self._selected_model)
            self._clear_edit_panel()
            self._load_specs()
        except Exception as e:
            logger.error(f"Failed to delete model spec: {e}")
            messagebox.showerror("Delete Error", str(e))

    def _check_discrepancies(self):
        """Check for spec discrepancies between file-parsed and reference specs."""
        try:
            from laser_trim_analyzer.database import get_database
            db = get_database()
            discrepancies = db.get_spec_discrepancies()

            if not discrepancies:
                messagebox.showinfo(
                    "Spec Check",
                    "No discrepancies found.\n\n"
                    "All file-parsed linearity specs match model reference specs within 5%."
                )
                return

            # Build report
            lines = [f"Found {len(discrepancies)} spec discrepancies:\n"]
            for d in discrepancies[:20]:  # Limit to top 20
                lines.append(
                    f"  {d['model']}: file={d['file_spec_avg']:.4f}, "
                    f"ref={d['reference_spec_decimal']:.4f} "
                    f"({d['difference_pct']:.0f}% off, {d['sample_count']} samples)"
                )
            if len(discrepancies) > 20:
                lines.append(f"\n  ... and {len(discrepancies) - 20} more")

            messagebox.showinfo("Spec Discrepancies", "\n".join(lines))
        except Exception as e:
            logger.error(f"Failed to check discrepancies: {e}")
            messagebox.showerror("Error", str(e))

    def _check_discrepancy_count(self):
        """Auto-check discrepancy count on page load and update button badge."""
        try:
            from laser_trim_analyzer.database import get_database
            db = get_database()
            discrepancies = db.get_spec_discrepancies()
            count = len(discrepancies) if discrepancies else 0
            if count > 0:
                self._discrepancy_btn.configure(
                    text=f"Discrepancies ({count})",
                    fg_color="#e74c3c"
                )
            else:
                self._discrepancy_btn.configure(
                    text="Check Discrepancies",
                    fg_color="#2980b9"
                )
        except Exception:
            pass
