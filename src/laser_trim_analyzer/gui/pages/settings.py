"""
Settings Page - Configuration.

Manage database path, export location, ML training, theme.
"""

import customtkinter as ctk
import logging
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Optional, Any

from laser_trim_analyzer.utils.threads import get_thread_manager

logger = logging.getLogger(__name__)


class SettingsPage(ctk.CTkFrame):
    """
    Settings page for configuration.

    Features:
    - Database path selector
    - Default export location
    - ML training trigger with status
    - Theme toggle (dark/light)
    - Version info
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.export_path: Optional[Path] = None
        self._ml_manager: Optional[Any] = None  # Set during training

        self._create_ui()

    def _create_ui(self):
        """Create the settings page UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkLabel(
            self,
            text="Settings",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        header.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        # Settings container
        container = ctk.CTkScrollableFrame(self)
        container.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        container.grid_columnconfigure(0, weight=1)

        # Database section
        self._create_database_section(container)

        # Export section
        self._create_export_section(container)

        # Processing section
        self._create_processing_section(container)

        # ML section
        self._create_ml_section(container)

        # Active Models section
        self._create_active_models_section(container)

        # Model Specs Import section
        self._create_model_specs_import_section(container)

        # Database Cleanup section
        self._create_database_cleanup_section(container)

        # Appearance section
        self._create_appearance_section(container)

        # About section
        self._create_about_section(container)

    def _create_database_section(self, container):
        """Create database settings section."""
        frame = ctk.CTkFrame(container)
        frame.grid(sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            frame,
            text="Database",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="w")

        # Current path
        ctk.CTkLabel(frame, text="Database Path:").grid(row=1, column=0, padx=15, pady=5, sticky="w")

        # Show relative path to avoid exposing full local filesystem path in screenshots
        try:
            from pathlib import Path
            db_display_path = str(Path(self.app.config.database.path).relative_to(Path.cwd()))
        except (ValueError, TypeError):
            db_display_path = Path(self.app.config.database.path).name
        self.db_path_label = ctk.CTkLabel(
            frame,
            text=db_display_path,
            text_color="gray",
            wraplength=400
        )
        self.db_path_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkButton(
            frame,
            text="Change",
            command=self._change_database_path,
            width=80
        ).grid(row=1, column=2, padx=15, pady=5)

        # Database info
        self.db_info_label = ctk.CTkLabel(
            frame,
            text="",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        )
        self.db_info_label.grid(row=2, column=0, columnspan=3, padx=15, pady=(0, 15), sticky="w")

    def _create_export_section(self, container):
        """Create export settings section."""
        frame = ctk.CTkFrame(container)
        frame.grid(sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            frame,
            text="Export",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="w")

        # Default export location
        ctk.CTkLabel(frame, text="Default Export Location:").grid(row=1, column=0, padx=15, pady=5, sticky="w")

        self.export_path_label = ctk.CTkLabel(
            frame,
            text="Not set (will ask each time)",
            text_color="gray",
            wraplength=400
        )
        self.export_path_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkButton(
            frame,
            text="Set",
            command=self._set_export_path,
            width=80
        ).grid(row=1, column=2, padx=15, pady=5)

    def _create_processing_section(self, container):
        """Create processing settings section."""
        frame = ctk.CTkFrame(container)
        frame.grid(sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            frame,
            text="Processing",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=2, padx=15, pady=(15, 10), sticky="w")

        settings = [
            ("Batch Size:", str(self.app.config.processing.batch_size)),
            ("Turbo Mode Threshold:", f"{self.app.config.processing.turbo_mode_threshold} files"),
        ]

        for i, (label, value) in enumerate(settings, start=1):
            ctk.CTkLabel(frame, text=label).grid(row=i, column=0, padx=15, pady=5, sticky="w")
            ctk.CTkLabel(frame, text=value, text_color="gray").grid(row=i, column=1, padx=10, pady=(5, 15 if i == len(settings) else 5), sticky="w")

    def _create_ml_section(self, container):
        """Create ML settings section with per-model training."""
        frame = ctk.CTkFrame(container)
        frame.grid(sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            frame,
            text="Machine Learning (Per-Model)",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="w")

        # ML enabled checkbox
        self.ml_enabled_var = ctk.BooleanVar(value=self.app.config.ml.enabled)
        ctk.CTkCheckBox(
            frame,
            text="Enable ML Features (per-model thresholds, failure prediction, drift detection)",
            variable=self.ml_enabled_var,
            command=self._toggle_ml
        ).grid(row=1, column=0, columnspan=3, padx=15, pady=5, sticky="w")

        # Train models button
        self.train_btn = ctk.CTkButton(
            frame,
            text="Train Models",
            command=self._train_models,
            width=130,
            fg_color="#2980b9"
        )
        self.train_btn.grid(row=2, column=0, padx=15, pady=(10, 5), sticky="w")

        # Apply to DB button
        self.apply_btn = ctk.CTkButton(
            frame,
            text="Apply to DB",
            command=self._apply_ml_to_db,
            width=130,
            fg_color="#27ae60"
        )
        self.apply_btn.grid(row=2, column=1, padx=5, pady=(10, 5), sticky="w")

        self.ml_status_label = ctk.CTkLabel(
            frame,
            text="Status: Not trained",
            text_color="gray"
        )
        self.ml_status_label.grid(row=2, column=2, padx=15, pady=(10, 5), sticky="w")

        # Training requirements note
        ctk.CTkLabel(
            frame,
            text="Train Models: Learns per-model thresholds, failure predictors, and drift baselines.\n"
                 "Apply to DB: Updates all records with learned thresholds and failure probabilities.",
            text_color="gray",
            font=ctk.CTkFont(size=11),
            justify="left"
        ).grid(row=3, column=0, columnspan=3, padx=15, pady=(0, 10), sticky="w")

        # Progress bar (hidden initially)
        self.ml_progress = ctk.CTkProgressBar(frame)
        self.ml_progress.grid(row=4, column=0, columnspan=3, padx=15, pady=5, sticky="ew")
        self.ml_progress.set(0)
        self.ml_progress.grid_remove()  # Hide initially

        self.ml_progress_label = ctk.CTkLabel(
            frame,
            text="",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        )
        self.ml_progress_label.grid(row=5, column=0, columnspan=3, padx=15, pady=(0, 5), sticky="w")
        self.ml_progress_label.grid_remove()  # Hide initially

        # Model status summary (collapsed by default)
        self.model_status_label = ctk.CTkLabel(
            frame,
            text="",
            text_color="gray",
            font=ctk.CTkFont(size=11),
            justify="left"
        )
        self.model_status_label.grid(row=6, column=0, columnspan=3, padx=15, pady=(5, 5), sticky="w")

        # ML Staleness indicator
        self.ml_staleness_label = ctk.CTkLabel(
            frame,
            text="",
            text_color="gray",
            font=ctk.CTkFont(size=11),
            justify="left"
        )
        self.ml_staleness_label.grid(row=7, column=0, columnspan=3, padx=15, pady=(0, 15), sticky="w")

    def _create_active_models_section(self, container):
        """Create Active Models (MPS) configuration section."""
        frame = ctk.CTkFrame(container)
        frame.grid(sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(0, weight=1)

        # Title
        title = ctk.CTkLabel(
            frame,
            text="Active Models (MPS Schedule)",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

        # Description
        desc = ctk.CTkLabel(
            frame,
            text="Models on MPS schedule are prioritized in dropdowns and Trends page alerts.\nEnter one model number per line. Recently active models (with data in last N days) are also prioritized.",
            text_color="gray",
            justify="left",
            font=ctk.CTkFont(size=11)
        )
        desc.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="w")

        # Text area for model list
        self.mps_textbox = ctk.CTkTextbox(
            frame,
            height=120,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.mps_textbox.grid(row=2, column=0, padx=15, pady=5, sticky="ew")

        # Load current models
        current_models = self.app.config.active_models.mps_models
        if current_models:
            self.mps_textbox.insert("1.0", "\n".join(current_models))

        # Button row
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.grid(row=3, column=0, padx=15, pady=5, sticky="ew")

        save_btn = ctk.CTkButton(
            btn_frame,
            text="Save",
            width=80,
            command=self._save_mps_models
        )
        save_btn.pack(side="left", padx=(0, 10))

        clear_btn = ctk.CTkButton(
            btn_frame,
            text="Clear All",
            width=80,
            fg_color="gray",
            command=self._clear_mps_models
        )
        clear_btn.pack(side="left")

        # Count label
        self.mps_count_label = ctk.CTkLabel(
            btn_frame,
            text=f"{len(current_models)} models configured",
            text_color="gray"
        )
        self.mps_count_label.pack(side="right")

        # Recent days setting
        recent_frame = ctk.CTkFrame(frame, fg_color="transparent")
        recent_frame.grid(row=4, column=0, padx=15, pady=(10, 15), sticky="w")

        ctk.CTkLabel(
            recent_frame,
            text="Also prioritize models with data in last"
        ).pack(side="left")

        self.recent_days_entry = ctk.CTkEntry(
            recent_frame,
            width=50,
            justify="center"
        )
        self.recent_days_entry.pack(side="left", padx=5)
        self.recent_days_entry.insert(0, str(self.app.config.active_models.recent_days))

        ctk.CTkLabel(recent_frame, text="days").pack(side="left")

        # --- Pricing Section ---
        pricing_label = ctk.CTkLabel(
            frame,
            text="Model Pricing",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        pricing_label.grid(row=5, column=0, padx=15, pady=(15, 5), sticky="w")

        pricing_desc = ctk.CTkLabel(
            frame,
            text="Import unit prices from Excel/CSV (columns: model/Item ID, price/Unit Price).\n"
                 "Prices are used for cost impact analysis on Dashboard and Trends pages.",
            text_color="gray",
            justify="left",
            font=ctk.CTkFont(size=11)
        )
        pricing_desc.grid(row=6, column=0, padx=15, pady=(0, 5), sticky="w")

        pricing_btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        pricing_btn_frame.grid(row=7, column=0, padx=15, pady=5, sticky="ew")

        ctk.CTkButton(
            pricing_btn_frame,
            text="Import Pricing from File",
            width=160,
            command=self._import_pricing,
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            pricing_btn_frame,
            text="Clear Pricing",
            width=100,
            fg_color="gray",
            command=self._clear_pricing,
        ).pack(side="left", padx=(0, 10))

        self.pricing_count_label = ctk.CTkLabel(
            pricing_btn_frame,
            text=self._pricing_summary_text(),
            text_color="gray",
            font=ctk.CTkFont(size=11)
        )
        self.pricing_count_label.pack(side="right")

        # Cost ratio setting
        cost_frame = ctk.CTkFrame(frame, fg_color="transparent")
        cost_frame.grid(row=8, column=0, padx=15, pady=(5, 15), sticky="w")

        ctk.CTkLabel(
            cost_frame,
            text="Cost ratio (fraction of unit price = invested cost):"
        ).pack(side="left")

        self.cost_ratio_entry = ctk.CTkEntry(cost_frame, width=60, justify="center")
        self.cost_ratio_entry.pack(side="left", padx=5)
        self.cost_ratio_entry.insert(0, str(self.app.config.active_models.cost_ratio))

        ctk.CTkLabel(cost_frame, text="(default 0.50 = 50%)").pack(side="left")

    def _pricing_summary_text(self) -> str:
        """Get summary text for pricing status."""
        prices = self.app.config.active_models.model_prices
        if not prices:
            return "No pricing data loaded"
        return f"{len(prices)} models with pricing"

    def _save_mps_models(self):
        """Save MPS model list to config."""
        text = self.mps_textbox.get("1.0", "end-1c")
        models = [m.strip() for m in text.split("\n") if m.strip()]

        # Remove duplicates while preserving order
        seen = set()
        unique_models = []
        for m in models:
            if m not in seen:
                seen.add(m)
                unique_models.append(m)

        self.app.config.active_models.mps_models = unique_models

        # Save recent days
        try:
            days = int(self.recent_days_entry.get())
            self.app.config.active_models.recent_days = max(1, min(365, days))
        except ValueError:
            pass

        # Save cost ratio
        try:
            ratio = float(self.cost_ratio_entry.get())
            self.app.config.active_models.cost_ratio = max(0.01, min(1.0, ratio))
        except (ValueError, AttributeError):
            pass

        self.app.config.save()
        self.mps_count_label.configure(text=f"{len(unique_models)} models configured")

        messagebox.showinfo("Saved", f"MPS model list saved ({len(unique_models)} models)")

    def _clear_mps_models(self):
        """Clear all MPS models."""
        self.mps_textbox.delete("1.0", "end")
        self.mps_count_label.configure(text="0 models configured")

    def _import_pricing(self):
        """Import model pricing from an Excel or CSV file."""
        file_path = filedialog.askopenfilename(
            title="Select Pricing File",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ]
        )
        if not file_path:
            return

        try:
            import pandas as pd

            file_path = Path(file_path)
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            # Find model and price columns (flexible matching)
            model_col = None
            price_col = None
            for col in df.columns:
                col_lower = str(col).lower()
                if col_lower in ('model', 'item id', 'item_id', 'itemid', 'part', 'part number'):
                    model_col = col
                elif col_lower in ('price', 'unit price', 'unit_price', 'unitprice', 'cost', 'unit cost'):
                    price_col = col

            if model_col is None or price_col is None:
                messagebox.showerror(
                    "Column Not Found",
                    f"Could not find model and price columns.\n\n"
                    f"Found columns: {list(df.columns)}\n\n"
                    f"Expected: 'Item ID'/'Model' and 'Unit Price'/'Price'"
                )
                return

            # Extract unique model → price mapping
            # Use the most common non-zero price for each model (handles duplicates)
            prices = {}
            for model_val, group in df.groupby(model_col):
                model = str(model_val).strip()
                if not model:
                    continue
                non_zero = group[group[price_col] > 0][price_col]
                if len(non_zero) > 0:
                    # Use mode (most common) price, or median if no clear mode
                    mode_vals = non_zero.mode()
                    prices[model] = float(mode_vals.iloc[0]) if len(mode_vals) > 0 else float(non_zero.median())

            if not prices:
                messagebox.showwarning("No Pricing Data", "No valid model/price pairs found in file.")
                return

            # Merge with existing prices (new values overwrite)
            existing = self.app.config.active_models.model_prices
            existing.update(prices)
            self.app.config.active_models.model_prices = existing
            self.app.config.save()

            self.pricing_count_label.configure(text=self._pricing_summary_text())
            messagebox.showinfo(
                "Pricing Imported",
                f"Imported {len(prices)} model prices from {file_path.name}.\n"
                f"Total: {len(existing)} models with pricing."
            )

        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import pricing:\n{e}")

    def _clear_pricing(self):
        """Clear all pricing data."""
        self.app.config.active_models.model_prices = {}
        self.app.config.save()
        self.pricing_count_label.configure(text=self._pricing_summary_text())
        messagebox.showinfo("Cleared", "All pricing data cleared.")

    def _create_model_specs_import_section(self, container):
        """Create model specs import section."""
        frame = ctk.CTkFrame(container)
        frame.grid(sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text="Model Specifications",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

        ctk.CTkLabel(
            frame, text="Import model engineering specs from the reference Excel file.",
            font=ctk.CTkFont(size=11), text_color="gray"
        ).grid(row=1, column=0, padx=15, pady=(0, 10), sticky="w")

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.grid(row=2, column=0, padx=15, pady=(0, 15), sticky="w")

        ctk.CTkButton(
            btn_frame,
            text="Import from Excel",
            width=150,
            command=self._import_model_specs,
        ).pack(side="left", padx=(0, 10))

        self._import_specs_label = ctk.CTkLabel(
            btn_frame, text="", text_color="gray", font=ctk.CTkFont(size=11)
        )
        self._import_specs_label.pack(side="left")

    def _import_model_specs(self):
        """Import model specs from Excel file."""
        file_path = filedialog.askopenfilename(
            title="Select Model Reference Excel",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if not file_path:
            return

        try:
            from laser_trim_analyzer.database import get_database
            db = get_database()
            result = db.import_model_specs_from_excel(file_path)
            msg = f"Added: {result['added']}, Updated: {result['updated']}"
            if result['skipped']:
                msg += f", Skipped: {result['skipped']}"
            self._import_specs_label.configure(text=msg, text_color="#27ae60")
            messagebox.showinfo("Import Complete", msg)
        except Exception as e:
            logger.error(f"Model specs import failed: {e}")
            self._import_specs_label.configure(text=f"Error: {e}", text_color="#e74c3c")
            messagebox.showerror("Import Error", str(e))

    def _create_database_cleanup_section(self, container):
        """Create database cleanup section for removing legacy/contaminated records."""
        frame = ctk.CTkFrame(container)
        frame.grid(sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)

        # Title
        ctk.CTkLabel(
            frame,
            text="Database Cleanup",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=3, padx=15, pady=(15, 5), sticky="w")

        # Description
        ctk.CTkLabel(
            frame,
            text="Scan finds dirty data, then use checkboxes to select what to delete.\n"
                 "Backup your database first: copy the .db file from the path shown above.",
            text_color="gray",
            justify="left",
            font=ctk.CTkFont(size=11)
        ).grid(row=1, column=0, columnspan=3, padx=15, pady=(0, 10), sticky="w")

        # --- Step 1: Scan button ---
        scan_frame = ctk.CTkFrame(frame, fg_color="transparent")
        scan_frame.grid(row=2, column=0, columnspan=3, padx=15, pady=(5, 5), sticky="w")

        self.scan_btn = ctk.CTkButton(
            scan_frame,
            text="Scan Database",
            width=140,
            fg_color="#2980b9",
            hover_color="#3498db",
            command=self._scan_database,
        )
        self.scan_btn.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(
            scan_frame,
            text="Step 1: Find dirty records (Unknown models, missing dates, bad data, etc.)",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        ).pack(side="left")

        # Scan results area
        self.scan_results_label = ctk.CTkLabel(
            frame,
            text="",
            text_color="gray",
            justify="left",
            font=ctk.CTkFont(size=11)
        )
        self.scan_results_label.grid(
            row=3, column=0, columnspan=3, padx=15, pady=(0, 10), sticky="w"
        )

        # --- Step 2: Cleanup options ---
        ctk.CTkLabel(
            frame,
            text="Step 2: Select what to delete",
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=4, column=0, columnspan=3, padx=15, pady=(5, 3), sticky="w")

        # Option 1: Delete non-MPS models
        self.cleanup_non_mps_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            frame,
            text="Records for models NOT in MPS list",
            variable=self.cleanup_non_mps_var,
        ).grid(row=5, column=0, columnspan=3, padx=15, pady=3, sticky="w")

        # Option 2: Delete before date
        self.cleanup_date_var = ctk.BooleanVar(value=False)
        date_row = ctk.CTkFrame(frame, fg_color="transparent")
        date_row.grid(row=6, column=0, columnspan=3, padx=15, pady=3, sticky="w")

        ctk.CTkCheckBox(
            date_row,
            text="Records older than:",
            variable=self.cleanup_date_var,
        ).pack(side="left")

        self.cleanup_date_entry = ctk.CTkEntry(
            date_row,
            width=100,
            placeholder_text="YYYY-MM-DD"
        )
        self.cleanup_date_entry.pack(side="left", padx=(10, 0))

        # Option 3: Delete suspect quality (now works after scan)
        self.cleanup_suspect_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            frame,
            text="Suspect data quality (run Scan first to flag records)",
            variable=self.cleanup_suspect_var,
        ).grid(row=7, column=0, columnspan=3, padx=15, pady=3, sticky="w")

        # Option 4: Delete Unknown model/serial
        self.cleanup_unknown_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            frame,
            text="Records with Unknown model or serial",
            variable=self.cleanup_unknown_var,
        ).grid(row=8, column=0, columnspan=3, padx=15, pady=3, sticky="w")

        # Option 5: Delete ERROR status
        self.cleanup_error_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            frame,
            text="Records with ERROR status (processing failed)",
            variable=self.cleanup_error_var,
        ).grid(row=9, column=0, columnspan=3, padx=15, pady=3, sticky="w")

        # Option 6: Delete empty analyses (no track data)
        self.cleanup_no_tracks_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            frame,
            text="Empty analyses (no track data)",
            variable=self.cleanup_no_tracks_var,
        ).grid(row=10, column=0, columnspan=3, padx=15, pady=3, sticky="w")

        # Option 7: Delete misclassified FT files (Test Station, Redundant, Primary, Final)
        self.cleanup_misclassified_ft_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            frame,
            text="Misclassified Final Test files (Test Station, Redundant, Primary)",
            variable=self.cleanup_misclassified_ft_var,
        ).grid(row=11, column=0, columnspan=3, padx=15, pady=3, sticky="w")

        # Button row
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.grid(row=12, column=0, columnspan=3, padx=15, pady=(10, 5), sticky="w")

        ctk.CTkButton(
            btn_frame,
            text="Preview",
            width=100,
            command=self._preview_cleanup,
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_frame,
            text="Delete Records",
            width=120,
            fg_color="#c0392b",
            hover_color="#e74c3c",
            command=self._execute_cleanup,
        ).pack(side="left")

        # Preview results area
        self.cleanup_preview_label = ctk.CTkLabel(
            frame,
            text="",
            text_color="gray",
            justify="left",
            font=ctk.CTkFont(size=11)
        )
        self.cleanup_preview_label.grid(
            row=13, column=0, columnspan=3, padx=15, pady=(5, 10), sticky="w"
        )

        # --- Reset skipped files ---
        reset_frame = ctk.CTkFrame(frame, fg_color="transparent")
        reset_frame.grid(row=14, column=0, columnspan=3, padx=15, pady=(5, 15), sticky="w")

        self.reset_skipped_btn = ctk.CTkButton(
            reset_frame,
            text="Reset Skipped Files",
            width=150,
            fg_color="#e67e22",
            hover_color="#f39c12",
            command=self._reset_skipped_files,
        )
        self.reset_skipped_btn.pack(side="left", padx=(0, 10))

        self.skipped_count_label = ctk.CTkLabel(
            reset_frame,
            text="",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        )
        self.skipped_count_label.pack(side="left")

        # Load skipped count on init
        self.after(500, self._update_skipped_count)

    def _scan_database(self):
        """Scan database for dirty data and flag suspect records (runs in background)."""
        self.scan_results_label.configure(text="Scanning... (this may take a moment)")
        self.scan_btn.configure(state="disabled")

        def _do_scan():
            from laser_trim_analyzer.database import get_database
            db = get_database()

            validate_result = db.retroactive_validate()
            health = db.scan_database_health()
            return validate_result, health

        def _on_done(result):
            try:
                validate_result, health = result
                lines = [
                    f"Scanned {validate_result['scanned']} records - "
                    f"found {health['total_dirty_records']} with issues "
                    f"({validate_result['flagged']} newly flagged)"
                ]

                if health["issues"]:
                    for key, info in health["issues"].items():
                        lines.append(f"  {info['label']}: {info['count']}")
                else:
                    lines.append("  Database is clean!")

                self.after(0, lambda: self.scan_results_label.configure(text="\n".join(lines)) if self.winfo_exists() else None)
            except Exception as e:
                self.after(0, lambda: self.scan_results_label.configure(text=f"Scan error: {e}") if self.winfo_exists() else None)
            finally:
                self.after(0, lambda: self.scan_btn.configure(state="normal") if self.winfo_exists() else None)

        def _run():
            try:
                result = _do_scan()
                _on_done(result)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                logger.error(f"Scan database error:\n{tb}")
                # Show traceback in UI for debugging
                msg = f"Scan error: {e}\n\n{tb[-500:]}"
                self.after(0, lambda: self.scan_results_label.configure(text=msg) if self.winfo_exists() else None)
                self.after(0, lambda: self.scan_btn.configure(state="normal") if self.winfo_exists() else None)

        get_thread_manager().start_thread(target=_run, name="settings-scan-db")

    def _update_skipped_count(self):
        """Update the skipped file count label (runs DB query in background)."""
        def _fetch():
            try:
                from laser_trim_analyzer.database import get_database
                db = get_database()
                count = db.count_skipped_files()
                self.after(0, lambda: self.skipped_count_label.configure(
                    text=f"{count} skipped files — clears them so they get reprocessed next run"
                ) if self.winfo_exists() else None)
            except Exception:
                self.after(0, lambda: self.skipped_count_label.configure(
                    text=""
                ) if self.winfo_exists() else None)

        get_thread_manager().start_thread(target=_fetch, name="settings-skipped-count")

    def _reset_skipped_files(self):
        """Reset skipped non-trim files so they can be reprocessed."""
        from laser_trim_analyzer.database import get_database
        db = get_database()

        count = db.count_skipped_files()
        if count == 0:
            messagebox.showinfo("No Skipped Files", "There are no skipped files to reset.")
            return

        confirm = messagebox.askyesno(
            "Reset Skipped Files",
            f"This will reset {count} skipped files so they get re-evaluated\n"
            f"on the next processing run.\n\n"
            f"Use this after fixing files that were previously detected as\n"
            f"non-trim (wrong format, missing sheets, etc.).\n\n"
            f"Continue?",
        )
        if not confirm:
            return

        cleared = db.reset_skipped_files()
        self._update_skipped_count()
        messagebox.showinfo(
            "Files Reset",
            f"Reset {cleared} files. They will be re-evaluated next time\n"
            f"you process that folder."
        )

    def _get_cleanup_options(self):
        """Get the current cleanup options from the UI checkboxes."""
        from datetime import datetime

        options = {
            "delete_non_mps": self.cleanup_non_mps_var.get(),
            "mps_models": None,
            "delete_before_date": None,
            "delete_suspect_quality": self.cleanup_suspect_var.get(),
            "delete_unknown": self.cleanup_unknown_var.get(),
            "delete_error_status": self.cleanup_error_var.get(),
            "delete_no_tracks": self.cleanup_no_tracks_var.get(),
            "delete_misclassified_ft": self.cleanup_misclassified_ft_var.get(),
        }

        # Get MPS models from config
        if options["delete_non_mps"]:
            options["mps_models"] = self.app.config.active_models.mps_models
            if not options["mps_models"]:
                messagebox.showwarning(
                    "No MPS Models",
                    "No MPS models configured. Add models in the Active Models section first."
                )
                return None

        # Parse date
        if options.get("delete_before_date") is None and self.cleanup_date_var.get():
            date_str = self.cleanup_date_entry.get().strip()
            if not date_str:
                messagebox.showwarning("No Date", "Enter a date in YYYY-MM-DD format.")
                return None
            try:
                options["delete_before_date"] = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                messagebox.showerror("Invalid Date", f"'{date_str}' is not a valid date.\nUse YYYY-MM-DD format.")
                return None

        # Check at least one option selected
        has_option = any([
            options["delete_non_mps"], options["delete_before_date"],
            options["delete_suspect_quality"], options["delete_unknown"],
            options["delete_error_status"], options["delete_no_tracks"],
            options["delete_misclassified_ft"],
        ])
        if not has_option:
            messagebox.showinfo("No Options", "Select at least one cleanup option.")
            return None

        return options

    def _preview_cleanup(self):
        """Preview what would be deleted without actually deleting."""
        options = self._get_cleanup_options()
        if not options:
            return

        from laser_trim_analyzer.database import get_database
        db = get_database()

        preview = db.preview_cleanup(**options)

        # Format preview results
        lines = [f"Preview: {preview['records_to_delete']} of {preview['total_records']} records would be deleted"]

        reason_labels = {
            "non_mps_models": "Non-MPS models",
            "before_date": "Before date",
            "suspect_quality": "Suspect quality",
            "unknown_model_serial": "Unknown model/serial",
            "error_status": "ERROR status",
            "no_tracks": "No track data",
            "misclassified_ft": "Misclassified FT files",
        }

        for reason, info in preview.get("by_reason", {}).items():
            label = reason_labels.get(reason, reason)
            if reason == "non_mps_models":
                models_str = ", ".join(info["models"][:10])
                if len(info["models"]) > 10:
                    models_str += f" ... (+{len(info['models']) - 10} more)"
                lines.append(f"  {label}: {info['count']} records ({len(info['models'])} models: {models_str})")
            elif reason == "before_date":
                lines.append(f"  Before {info['date']}: {info['count']} records")
            else:
                lines.append(f"  {label}: {info['count']} records")

        self.cleanup_preview_label.configure(text="\n".join(lines))

    def _execute_cleanup(self):
        """Execute the cleanup after confirmation."""
        options = self._get_cleanup_options()
        if not options:
            return

        # First run preview to get counts
        from laser_trim_analyzer.database import get_database
        db = get_database()

        preview = db.preview_cleanup(**options)

        if preview["records_to_delete"] == 0:
            messagebox.showinfo("Nothing to Delete", "No records match the selected criteria.")
            return

        # Backup reminder + confirmation
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"This will permanently delete {preview['records_to_delete']} records "
            f"(out of {preview['total_records']} total).\n\n"
            f"Have you backed up your database?\n"
            f"(Copy the .db file before proceeding)\n\n"
            f"This cannot be undone. Continue?",
            icon="warning"
        )
        if not confirm:
            return

        # Execute
        result = db.execute_cleanup(**options)

        self.cleanup_preview_label.configure(
            text=f"Deleted: {result['analyses']} analyses, {result['tracks']} tracks, "
                 f"{result['alerts']} alerts (files stay marked so they won't reprocess)"
        )
        messagebox.showinfo(
            "Cleanup Complete",
            f"Deleted {result['analyses']} analysis records and associated data.\n"
            f"Cleaned files won't be reprocessed.\n"
            f"Refresh Dashboard/Trends to see updated counts."
        )

    def _create_appearance_section(self, container):
        """Create appearance settings section."""
        frame = ctk.CTkFrame(container)
        frame.grid(sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            frame,
            text="Appearance",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=2, padx=15, pady=(15, 10), sticky="w")

        ctk.CTkLabel(frame, text="Theme:").grid(row=1, column=0, padx=15, pady=(5, 15), sticky="w")

        self.theme_dropdown = ctk.CTkOptionMenu(
            frame,
            values=["Dark", "Light", "System"],
            command=self._change_theme,
            width=150
        )
        self.theme_dropdown.set(self.app.config.gui.theme.capitalize())
        self.theme_dropdown.grid(row=1, column=1, padx=10, pady=(5, 15), sticky="w")

    def _create_about_section(self, container):
        """Create about section."""
        frame = ctk.CTkFrame(container)
        frame.grid(sticky="ew", padx=10, pady=10)

        title = ctk.CTkLabel(
            frame,
            text="About",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.pack(padx=15, pady=(15, 10), anchor="w")

        version_info = ctk.CTkLabel(
            frame,
            text=f"Laser Trim Analyzer v{self.app.config.version}\n\n"
                 "A streamlined, ML-integrated application for laser trim data analysis.\n\n"
                 "Key Features:\n"
                 "  • Per-model ML threshold optimization\n"
                 "  • Drift detection with CUSUM/EWMA\n"
                 "  • Memory-safe processing for 8GB systems\n"
                 "  • Excel export with batch processing",
            text_color="gray",
            justify="left"
        )
        version_info.pack(padx=15, pady=(0, 15), anchor="w")

    def _change_database_path(self):
        """Change the database path."""
        path = filedialog.asksaveasfilename(
            title="Select Database Location",
            defaultextension=".db",
            initialfile="laser_trim.db",
            filetypes=[("SQLite Database", "*.db"), ("All files", "*.*")]
        )
        if path:
            self.app.config.database.path = Path(path)
            self.app.config.save()
            self.db_path_label.configure(text=str(path))
            logger.info(f"Database path changed to: {path}")
            messagebox.showinfo(
                "Database Path Changed",
                f"Database path changed to:\n{path}\n\nRestart required for changes to take effect."
            )

    def _set_export_path(self):
        """Set the default export location."""
        path = filedialog.askdirectory(title="Select Default Export Location")
        if path:
            self.export_path = Path(path)
            self.export_path_label.configure(text=str(path))
            if hasattr(self.app.config, 'export_path'):
                self.app.config.export_path = str(path)
                self.app.config.save()
            logger.info(f"Default export path set to: {path}")

    def _toggle_ml(self):
        """Toggle ML features."""
        self.app.config.ml.enabled = self.ml_enabled_var.get()
        self.app.config.save()
        logger.info(f"ML features {'enabled' if self.app.config.ml.enabled else 'disabled'}")

    def _train_models(self):
        """Trigger per-model ML training."""
        self.train_btn.configure(state="disabled")
        self.apply_btn.configure(state="disabled")
        self.ml_status_label.configure(text="Status: Training...", text_color="gray")
        self.ml_progress.grid()  # Show progress bar
        self.ml_progress_label.grid()
        self.ml_progress.set(0)

        get_thread_manager().start_thread(target=self._run_training, name="ml-training")

    def _run_training(self):
        """Run per-model ML training in background thread."""
        try:
            from laser_trim_analyzer.database import get_database
            from laser_trim_analyzer.ml import MLManager, TrainingProgress

            db = get_database()

            # Create ML manager
            ml_manager = MLManager(db)

            # Progress callback
            def on_progress(progress: TrainingProgress):
                if progress.models_total > 0:
                    pct = progress.models_complete / progress.models_total
                    self.after(0, lambda p=pct, m=progress.message: self._update_training_progress(p, m))

            # Train all models
            results = ml_manager.train_all_models(
                min_samples=20,
                progress_callback=on_progress
            )

            # Save trained state
            ml_manager.save_all()

            # Store manager for apply step
            self._ml_manager = ml_manager

            # Count results
            trained = sum(1 for r in results.values() if r.success and r.threshold_calculated)
            predictors = sum(1 for r in results.values() if r.predictor_trained)
            drift = sum(1 for r in results.values() if r.drift_baseline_set)

            msg = f"Trained {trained} models ({predictors} predictors, {drift} drift baselines)"
            status_details = self._format_training_status(results)

            self.after(0, lambda: self._on_training_complete(True, msg, status_details))

        except Exception as e:
            logger.exception(f"ML training failed: {e}")
            self.after(0, lambda: self._on_training_complete(False, str(e)))

    def _update_training_progress(self, progress: float, message: str):
        """Update training progress bar."""
        if not self.winfo_exists():
            return
        self.ml_progress.set(progress)
        self.ml_progress_label.configure(text=message)

    def _format_training_status(self, results: dict) -> str:
        """Format training results for display."""
        lines = []
        for model_name, result in sorted(results.items()):
            if result.success:
                thresh = f"T={result.threshold_value:.5f}" if result.threshold_value else "T=N/A"
                pred = "P" if result.predictor_trained else "-"
                drift = "D" if result.drift_baseline_set else "-"
                lines.append(f"  {model_name}: {thresh} [{pred}{drift}] ({result.n_samples} samples)")
            else:
                lines.append(f"  {model_name}: {result.error or 'insufficient data'}")
        return "\n".join(lines[:10])  # Limit to first 10 for display

    def _apply_ml_to_db(self):
        """Apply learned ML to database."""
        if self._ml_manager is None or not self._ml_manager.trained_models:
            from tkinter import messagebox
            messagebox.showwarning(
                "No Trained Models",
                "Please train models first before applying to database."
            )
            return

        self.train_btn.configure(state="disabled")
        self.apply_btn.configure(state="disabled")
        self.ml_status_label.configure(text="Status: Applying...", text_color="gray")
        self.ml_progress.grid()
        self.ml_progress_label.grid()
        self.ml_progress.set(0)

        get_thread_manager().start_thread(target=self._run_apply_ml, name="apply-ml")

    def _run_apply_ml(self):
        """Run ML application in background thread."""
        try:
            from laser_trim_analyzer.ml import ApplyProgress

            # Progress callback
            def on_progress(progress: ApplyProgress):
                if progress.records_total > 0:
                    pct = progress.records_complete / progress.records_total
                    self.after(0, lambda p=pct, m=progress.message: self._update_training_progress(p, m))

            # Apply to database
            counts = self._ml_manager.apply_to_database(
                progress_callback=on_progress,
                run_drift_detection=True
            )

            # Format result message
            msg = f"Updated {counts['updated']} tracks"
            if counts['drift_alerts']:
                msg += f", {len(counts['drift_alerts'])} drift alerts"
            if counts['errors'] > 0:
                msg += f", {counts['errors']} errors"

            self.after(0, lambda: self._on_apply_complete(True, msg))

        except Exception as e:
            logger.exception(f"ML apply failed: {e}")
            self.after(0, lambda: self._on_apply_complete(False, str(e)))

    def _refresh_staleness(self):
        """Refresh the ML staleness label with current data."""
        try:
            from laser_trim_analyzer.database import get_database
            db = get_database()
            staleness = db.get_ml_staleness()
            needs_retrain = [s for s in staleness if s["needs_retrain"]]
            if needs_retrain:
                models_str = ", ".join(s["model"] for s in needs_retrain[:5])
                if len(needs_retrain) > 5:
                    models_str += f" (+{len(needs_retrain) - 5} more)"
                self.ml_staleness_label.configure(
                    text=f"Retrain needed: {len(needs_retrain)} models with 50+ new records — {models_str}",
                    text_color="#f39c12"
                )
            elif staleness:
                self.ml_staleness_label.configure(
                    text=f"All {len(staleness)} trained models are up to date",
                    text_color="#27ae60"
                )
            else:
                self.ml_staleness_label.configure(text="", text_color="gray")
        except Exception:
            self.ml_staleness_label.configure(text="", text_color="gray")

    def _on_apply_complete(self, success: bool, message: str):
        """Handle apply completion."""
        if not self.winfo_exists():
            return
        self.train_btn.configure(state="normal")
        self.apply_btn.configure(state="normal")
        self.ml_progress.grid_remove()
        self.ml_progress_label.grid_remove()

        if success:
            self.ml_status_label.configure(text=f"Applied: {message}", text_color="#27ae60")
            logger.info(f"ML apply successful: {message}")
        else:
            self.ml_status_label.configure(text=f"Apply failed: {message[:40]}...", text_color="#e74c3c")
            logger.error(f"ML apply failed: {message}")

    def _on_training_complete(self, success: bool, message: str, status_details: str = ""):
        """Handle training completion."""
        if not self.winfo_exists():
            return
        self.train_btn.configure(state="normal")
        self.apply_btn.configure(state="normal")
        self.ml_progress.grid_remove()
        self.ml_progress_label.grid_remove()

        if success:
            self.ml_status_label.configure(text=f"Status: {message}", text_color="#27ae60")
            if status_details:
                self.model_status_label.configure(text=status_details)
            logger.info(f"ML training successful: {message}")
            # Refresh staleness label so "Retrain needed" clears
            self._refresh_staleness()
        else:
            self.ml_status_label.configure(text=f"Status: Failed - {message[:40]}...", text_color="#e74c3c")
            logger.error(f"ML training failed: {message}")

    def _change_theme(self, theme: str):
        """Change the application theme."""
        theme_lower = theme.lower()
        ctk.set_appearance_mode(theme_lower)
        self.app.config.gui.theme = theme_lower
        logger.info(f"Theme changed to: {theme_lower}")

    def _update_database_info(self):
        """Update database info display (runs DB query in background)."""
        from laser_trim_analyzer.utils.threads import get_thread_manager

        def _fetch():
            try:
                from laser_trim_analyzer.database import get_database
                db = get_database()
                count = db.get_record_count()
                total = count.get("analyses", 0)
                self.after(0, lambda: self.db_info_label.configure(
                    text=f"Connected - {total} files in database"
                ) if self.winfo_exists() else None)
            except Exception as e:
                self.after(0, lambda: self.db_info_label.configure(
                    text=f"Not connected: {str(e)[:30]}"
                ) if self.winfo_exists() else None)

        get_thread_manager().start_thread(target=_fetch, name="settings-db-info")

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Settings page shown")
        # Restore export path from config
        if self.app.config.export_path:
            self.export_path = Path(self.app.config.export_path)
            self.export_path_label.configure(text=self.app.config.export_path)
        # Update database info
        self._update_database_info()

        # Check ML model status from database (background — MLManager.load_all can be slow)
        def _load_ml_status():
            try:
                from laser_trim_analyzer.database import get_database
                from laser_trim_analyzer.ml import MLManager

                db = get_database()
                ml_manager = MLManager(db)
                ml_manager.load_all()

                def _update_ui():
                    if not self.winfo_exists():
                        return
                    if ml_manager.trained_models:
                        self._ml_manager = ml_manager
                        self.ml_status_label.configure(
                            text=f"Status: {len(ml_manager.trained_models)} models trained",
                            text_color="#27ae60"
                        )
                    else:
                        self.ml_status_label.configure(text="Status: Not trained", text_color="gray")
                    self._refresh_staleness()

                self.after(0, _update_ui)
            except Exception:
                self.after(0, lambda: self.ml_status_label.configure(
                    text="Status: Not trained", text_color="gray"
                ) if self.winfo_exists() else None)

        get_thread_manager().start_thread(target=_load_ml_status, name="settings-ml-status")
