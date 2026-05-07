# Output Smoothness Stats & Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-model statistics panel and Excel export to the Output Smoothness page.

**Architecture:** New DB query aggregates smoothness stats grouped by model. The smoothness page gets a stats panel (table or card) inserted between the top bar and the results area, plus an export button that writes one-worksheet-per-model Excel files using openpyxl.

**Tech Stack:** SQLAlchemy (existing), openpyxl (existing), customtkinter (existing)

**Spec:** `docs/superpowers/specs/2026-04-21-smoothness-stats-export-design.md`

---

### Task 1: Add `get_smoothness_stats_by_model()` to DatabaseManager

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py` (insert after `get_smoothness_stats` at line ~6423)

- [ ] **Step 1: Add the query method**

Insert after the existing `get_smoothness_stats()` method (line 6423):

```python
def get_smoothness_stats_by_model(
    self, model: Optional[str] = None, days_back: int = 90
) -> List[Dict[str, Any]]:
    """Per-model Output Smoothness statistics.

    When model is None, returns one row per model sorted by pass_rate
    ascending (worst first). When model is given, returns a single-element
    list for that model.
    """
    from laser_trim_analyzer.database.models import SmoothnessResult as DBSmoothnessResult

    with self.session() as session:
        cutoff = datetime.now() - timedelta(days=days_back)
        query = session.query(
            DBSmoothnessResult.model,
            func.count(DBSmoothnessResult.id).label("count"),
            func.sum(
                case((DBSmoothnessResult.smoothness_pass == True, 1), else_=0)
            ).label("passed"),
            func.avg(DBSmoothnessResult.max_smoothness_value).label("avg_max"),
            func.max(DBSmoothnessResult.max_smoothness_value).label("worst"),
            func.avg(DBSmoothnessResult.smoothness_spec).label("spec"),
        ).filter(
            DBSmoothnessResult.file_date >= cutoff,
        )
        if model:
            query = query.filter(DBSmoothnessResult.model == model)
        query = query.group_by(DBSmoothnessResult.model)

        rows = query.all()
        results = []
        for r in rows:
            count = r.count or 0
            passed = r.passed or 0
            avg_max = r.avg_max
            spec = r.spec
            margin = round(spec - avg_max, 4) if spec and avg_max else None
            results.append({
                "model": r.model,
                "count": count,
                "passed": passed,
                "pass_rate": round(passed / count * 100, 1) if count else 0,
                "avg_max_smoothness": round(avg_max, 4) if avg_max else None,
                "worst_case": round(r.worst, 4) if r.worst else None,
                "spec_limit": round(spec, 4) if spec else None,
                "margin": margin,
            })
        # Sort worst-first by pass_rate, then by margin
        results.sort(key=lambda x: (x["pass_rate"], x["margin"] or 0))
        return results
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Smoke-test against live DB**

Run:
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from laser_trim_analyzer.database.manager import DatabaseManager
db = DatabaseManager('data/analysis.db')
stats = db.get_smoothness_stats_by_model()
for s in stats[:5]:
    print(s)
print(f'Total models: {len(stats)}')
# Single model
single = db.get_smoothness_stats_by_model(model='8275')
print(f'Single: {single}')
"
```

Expected: list of dicts with model, count, pass_rate, margin, etc.

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py
git commit -m "feat(smoothness): add get_smoothness_stats_by_model query"
```

---

### Task 2: Add stats panel to smoothness page

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/smoothness.py`

This task restructures the layout to add a stats panel between the top bar and the content area, and wires up the data loading.

- [ ] **Step 1: Update `_create_layout` to add stats panel and export button**

Replace the full `_create_layout` method with:

```python
def _create_layout(self):
    """Create the page layout."""
    self.grid_rowconfigure(2, weight=1)  # content row gets the stretch
    self.grid_columnconfigure(0, weight=1)

    # Row 0: Top bar with filters + export
    bar = ctk.CTkFrame(self)
    bar.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))

    ctk.CTkLabel(bar, text="Output Smoothness",
                font=ctk.CTkFont(size=18, weight="bold")).pack(side="left", padx=10)

    ctk.CTkLabel(bar, text="Model:").pack(side="left", padx=(20, 5))
    self.model_dropdown = ScrollableComboBox(
        bar,
        values=["All Models"],
        command=lambda _: self._load_results(),
        width=120
    )
    self.model_dropdown.set("All Models")
    self.model_dropdown.pack(side="left", padx=5)

    ctk.CTkButton(bar, text="Refresh", width=80,
                 command=self._load_results).pack(side="left", padx=10)

    ctk.CTkButton(bar, text="Export to Excel", width=120,
                 command=self._export_to_excel).pack(side="left", padx=5)

    self.stats_label = ctk.CTkLabel(bar, text="", text_color="gray")
    self.stats_label.pack(side="right", padx=10)

    # Row 1: Stats panel
    self.stats_panel = ctk.CTkFrame(self)
    self.stats_panel.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
    self.stats_panel.grid_columnconfigure(0, weight=1)

    # Scrollable table for all-models view
    self.stats_table_frame = ctk.CTkScrollableFrame(self.stats_panel, height=180)
    self.stats_table_frame.grid(row=0, column=0, sticky="ew")
    self.stats_table_frame.grid_columnconfigure(0, weight=1)

    # Single-model card (hidden by default)
    self.stats_card_frame = ctk.CTkFrame(self.stats_panel)

    # Row 2: Main content (results list + detail)
    content = ctk.CTkFrame(self)
    content.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
    content.grid_rowconfigure(0, weight=1)
    content.grid_columnconfigure(0, weight=1)
    content.grid_columnconfigure(1, weight=2)

    # Results list (left)
    list_frame = ctk.CTkFrame(content)
    list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
    list_frame.grid_rowconfigure(0, weight=1)
    list_frame.grid_columnconfigure(0, weight=1)

    self.results_scroll = ctk.CTkScrollableFrame(list_frame)
    self.results_scroll.grid(row=0, column=0, sticky="nsew")
    self.results_scroll.grid_columnconfigure(0, weight=1)

    page_frame = ctk.CTkFrame(list_frame)
    page_frame.grid(row=1, column=0, sticky="ew", pady=5)
    self.prev_btn = ctk.CTkButton(page_frame, text="<", width=30, command=self._prev_page)
    self.prev_btn.pack(side="left", padx=5)
    self.page_label = ctk.CTkLabel(page_frame, text="Page 1/1")
    self.page_label.pack(side="left", padx=5)
    self.next_btn = ctk.CTkButton(page_frame, text=">", width=30, command=self._next_page)
    self.next_btn.pack(side="left", padx=5)

    # Detail panel (right)
    detail = ctk.CTkFrame(content)
    detail.grid(row=0, column=1, sticky="nsew")
    detail.grid_rowconfigure(1, weight=1)
    detail.grid_columnconfigure(0, weight=1)

    self.info_text = ctk.CTkTextbox(detail, height=120)
    self.info_text.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

    self.chart_frame = ctk.CTkFrame(detail)
    self.chart_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
```

- [ ] **Step 2: Update `_load_results` to also fetch per-model stats**

Replace `_load_results` with:

```python
def _load_results(self):
    """Load smoothness results and per-model stats from database."""
    model = self.model_dropdown.get()
    if model == "All Models":
        model = None

    def _do_load():
        try:
            db = get_database()
            results = db.search_smoothness_results(model=model, limit=500)
            stats = db.get_smoothness_stats()
            model_stats = db.get_smoothness_stats_by_model(model=model)
            return results, stats, model_stats
        except Exception as e:
            logger.error(f"Error loading smoothness results: {e}")
            return [], {}, []

    def _on_done(data):
        results, stats, model_stats = data
        def _apply():
            if not self.winfo_exists():
                return
            self.results_list = results
            self._current_page = 0
            self._total_pages = max(1, (len(results) + self._page_size - 1) // self._page_size)
            self._display_results()
            self._display_stats(stats)
            self._display_model_stats(model_stats, is_single=model is not None)
        self.after(0, _apply)

    get_thread_manager().start_thread(
        target=lambda: _on_done(_do_load()),
        name="smoothness-load"
    )
```

- [ ] **Step 3: Add `_display_model_stats` method for the comparison table and single-model card**

Add after `_display_stats`:

```python
def _display_model_stats(self, model_stats: List[Dict[str, Any]], is_single: bool = False):
    """Display per-model stats as a table or single-model card."""
    # Clear both views
    for w in self.stats_table_frame.winfo_children():
        w.destroy()
    for w in self.stats_card_frame.winfo_children():
        w.destroy()

    if not model_stats:
        self.stats_card_frame.grid_remove()
        self.stats_table_frame.grid()
        ctk.CTkLabel(self.stats_table_frame, text="No smoothness data",
                    text_color="gray").grid(row=0, column=0, pady=10)
        return

    if is_single:
        # Single-model card view
        self.stats_table_frame.grid_remove()
        self.stats_card_frame.grid(row=0, column=0, sticky="ew")
        self._build_single_model_card(model_stats[0])
    else:
        # All-models comparison table
        self.stats_card_frame.grid_remove()
        self.stats_table_frame.grid()
        self._build_comparison_table(model_stats)

def _build_comparison_table(self, model_stats: List[Dict[str, Any]]):
    """Build the all-models comparison table."""
    frame = self.stats_table_frame
    headers = ["Model", "Count", "Pass Rate", "Avg Max", "Worst Case", "Spec Limit", "Margin"]
    for col, header in enumerate(headers):
        ctk.CTkLabel(frame, text=header, font=ctk.CTkFont(size=11, weight="bold"),
                    text_color="#aaaaaa").grid(row=0, column=col, padx=6, pady=(4, 2), sticky="w")
    frame.grid_columnconfigure(0, weight=1)

    for row_idx, s in enumerate(model_stats, start=1):
        # Pass rate color
        pr = s["pass_rate"]
        pr_color = "#e74c3c" if pr < 80 else "#f39c12" if pr < 95 else "#27ae60"

        # Margin color
        margin = s["margin"]
        spec = s["spec_limit"]
        if margin is not None and spec is not None and spec > 0:
            margin_pct = margin / spec
            m_color = "#e74c3c" if margin_pct < 0 else "#f39c12" if margin_pct < 0.20 else "#27ae60"
        else:
            m_color = "gray"

        model_label = ctk.CTkLabel(frame, text=s["model"], font=ctk.CTkFont(size=11),
                                   cursor="hand2", text_color="#3498db")
        model_label.grid(row=row_idx, column=0, padx=6, pady=1, sticky="w")
        model_label.bind("<Button-1>", lambda e, m=s["model"]: self._select_model(m))

        ctk.CTkLabel(frame, text=str(s["count"]),
                    font=ctk.CTkFont(size=11)).grid(row=row_idx, column=1, padx=6, pady=1, sticky="w")
        ctk.CTkLabel(frame, text=f"{s['passed']}/{s['count']} ({pr:.0f}%)",
                    font=ctk.CTkFont(size=11), text_color=pr_color
                    ).grid(row=row_idx, column=2, padx=6, pady=1, sticky="w")
        ctk.CTkLabel(frame, text=f"{s['avg_max_smoothness']:.4f}" if s["avg_max_smoothness"] else "—",
                    font=ctk.CTkFont(size=11)).grid(row=row_idx, column=3, padx=6, pady=1, sticky="w")
        ctk.CTkLabel(frame, text=f"{s['worst_case']:.4f}" if s["worst_case"] else "—",
                    font=ctk.CTkFont(size=11)).grid(row=row_idx, column=4, padx=6, pady=1, sticky="w")
        ctk.CTkLabel(frame, text=f"{spec:.4f}" if spec else "—",
                    font=ctk.CTkFont(size=11)).grid(row=row_idx, column=5, padx=6, pady=1, sticky="w")
        ctk.CTkLabel(frame, text=f"{margin:.4f}" if margin is not None else "—",
                    font=ctk.CTkFont(size=11), text_color=m_color
                    ).grid(row=row_idx, column=6, padx=6, pady=1, sticky="w")

def _build_single_model_card(self, s: Dict[str, Any]):
    """Build the single-model summary card."""
    frame = self.stats_card_frame

    pr = s["pass_rate"]
    pr_color = "#e74c3c" if pr < 80 else "#f39c12" if pr < 95 else "#27ae60"

    margin = s["margin"]
    spec = s["spec_limit"]
    if margin is not None and spec is not None and spec > 0:
        margin_pct = margin / spec
        m_color = "#e74c3c" if margin_pct < 0 else "#f39c12" if margin_pct < 0.20 else "#27ae60"
    else:
        m_color = "gray"

    items = [
        ("Pass Rate", f"{s['passed']}/{s['count']} ({pr:.0f}%)", pr_color),
        ("Avg Max Smoothness", f"{s['avg_max_smoothness']:.4f}" if s["avg_max_smoothness"] else "—", None),
        ("Worst Case", f"{s['worst_case']:.4f}" if s["worst_case"] else "—", None),
        ("Spec Limit", f"{spec:.4f}" if spec else "—", None),
        ("Margin", f"{margin:.4f}" if margin is not None else "—", m_color),
    ]
    for col, (label, value, color) in enumerate(items):
        f = ctk.CTkFrame(frame, fg_color="transparent")
        f.pack(side="left", padx=15, pady=8)
        ctk.CTkLabel(f, text=label, font=ctk.CTkFont(size=10),
                    text_color="#aaaaaa").pack()
        ctk.CTkLabel(f, text=value, font=ctk.CTkFont(size=14, weight="bold"),
                    text_color=color or ("white" if ctk.get_appearance_mode() == "Dark" else "black")
                    ).pack()

def _select_model(self, model: str):
    """Select a model from the comparison table."""
    self.model_dropdown.set(model)
    self._load_results()
```

- [ ] **Step 4: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/smoothness.py').read()); print('OK')"`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/smoothness.py
git commit -m "feat(smoothness): add per-model stats panel with comparison table"
```

---

### Task 3: Add Excel export to smoothness page

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/smoothness.py`

- [ ] **Step 1: Add the `_export_to_excel` method**

Add after `_select_model`:

```python
def _export_to_excel(self):
    """Export smoothness data to Excel with one worksheet per model."""
    from tkinter import filedialog, messagebox
    from datetime import datetime

    model_filter = self.model_dropdown.get()
    if model_filter == "All Models":
        model_filter = None

    # Get data
    try:
        db = get_database()
        results = db.search_smoothness_results(model=model_filter, limit=10000)
        model_stats = db.get_smoothness_stats_by_model(model=model_filter)
    except Exception as e:
        logger.error(f"Export data fetch failed: {e}")
        messagebox.showerror("Export Error", str(e))
        return

    if not results:
        messagebox.showinfo("Export", "No smoothness data to export.")
        return

    # Ask for save location
    default_name = f"Output_Smoothness_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
    file_path = filedialog.asksaveasfilename(
        title="Export Smoothness to Excel",
        defaultextension=".xlsx",
        initialfile=default_name,
        initialdir=getattr(self.app.config, 'export_path', None),
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    if not file_path:
        return

    try:
        self._write_smoothness_excel(file_path, results, model_stats)
        messagebox.showinfo("Export Complete", f"Saved to:\n{file_path}")
    except Exception as e:
        logger.error(f"Smoothness export failed: {e}")
        messagebox.showerror("Export Error", str(e))

def _write_smoothness_excel(
    self,
    file_path: str,
    results: List[Dict[str, Any]],
    model_stats: List[Dict[str, Any]],
):
    """Write smoothness results to Excel with one sheet per model."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # Group results by model
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)

    # Build a stats lookup
    stats_lookup = {s["model"]: s for s in model_stats}

    headers = [
        "Model", "Serial", "Element Label", "Date", "Status",
        "Spec Limit", "Max Smoothness", "Avg Smoothness",
        "Linked Trim ID", "Match Confidence",
    ]
    header_font = Font(bold=True)
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    pass_font = Font(color="27AE60")
    fail_font = Font(color="E74C3C")

    for model_name in sorted(by_model.keys()):
        model_results = by_model[model_name]
        # Excel sheet names max 31 chars
        sheet_name = model_name[:31]
        ws = wb.create_sheet(title=sheet_name)

        # Header row
        for col, h in enumerate(headers, start=1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.font = header_font
            cell.fill = header_fill

        # Data rows (sorted by date descending)
        model_results.sort(key=lambda x: x.get("file_date") or "", reverse=True)
        for row_idx, r in enumerate(model_results, start=2):
            fd = r.get("file_date")
            date_str = fd.strftime("%Y-%m-%d") if hasattr(fd, "strftime") else str(fd)[:10] if fd else ""
            status = r.get("overall_status", "UNKNOWN")

            ws.cell(row=row_idx, column=1, value=r["model"])
            ws.cell(row=row_idx, column=2, value=r["serial"])
            ws.cell(row=row_idx, column=3, value=r.get("element_label") or "")
            ws.cell(row=row_idx, column=4, value=date_str)
            status_cell = ws.cell(row=row_idx, column=5, value=status)
            status_cell.font = pass_font if status == "PASS" else fail_font
            ws.cell(row=row_idx, column=6, value=r.get("smoothness_spec"))
            ws.cell(row=row_idx, column=7, value=r.get("max_smoothness_value"))
            ws.cell(row=row_idx, column=8, value=r.get("avg_smoothness_value"))
            ws.cell(row=row_idx, column=9, value=r.get("linked_trim_id"))
            conf = r.get("match_confidence")
            ws.cell(row=row_idx, column=10, value=f"{conf:.0%}" if conf else "")

        # Summary row
        summary_row = len(model_results) + 3
        ws.cell(row=summary_row, column=1, value="Summary").font = Font(bold=True)

        s = stats_lookup.get(model_name, {})
        summary_items = [
            ("Count", s.get("count")),
            ("Pass Rate", f"{s['pass_rate']:.1f}%" if s.get("pass_rate") is not None else None),
            ("Avg Max Smoothness", s.get("avg_max_smoothness")),
            ("Worst Case", s.get("worst_case")),
            ("Spec Limit", s.get("spec_limit")),
            ("Margin", s.get("margin")),
        ]
        for i, (label, value) in enumerate(summary_items):
            ws.cell(row=summary_row + 1 + i, column=1, value=label).font = Font(italic=True)
            ws.cell(row=summary_row + 1 + i, column=2, value=value)

        # Auto-width columns
        for col_idx in range(1, len(headers) + 1):
            max_len = len(headers[col_idx - 1])
            for row_idx in range(2, len(model_results) + 2):
                val = ws.cell(row=row_idx, column=col_idx).value
                if val:
                    max_len = max(max_len, len(str(val)))
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 30)

    wb.save(file_path)
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/smoothness.py').read()); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Smoke-test the export function offline**

Run:
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from laser_trim_analyzer.database import get_database
db = get_database()
results = db.search_smoothness_results(limit=500)
stats = db.get_smoothness_stats_by_model()
print(f'Results: {len(results)}, Models with stats: {len(stats)}')

# Test the Excel writing directly
from laser_trim_analyzer.gui.pages.smoothness import SmoothnessPage
# Can't instantiate GUI, but we can test the static-ish parts
from openpyxl import Workbook
print('openpyxl import OK')
print('Models in results:', sorted(set(r['model'] for r in results)))
"
```

Expected: prints result counts and model list without errors.

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/smoothness.py
git commit -m "feat(smoothness): add Excel export with one worksheet per model"
```

---

### Task 4: Verify full integration

- [ ] **Step 1: Syntax check all modified files**

Run:
```bash
python3 -c "
import ast
for f in ['src/laser_trim_analyzer/database/manager.py', 'src/laser_trim_analyzer/gui/pages/smoothness.py']:
    ast.parse(open(f).read())
    print(f'{f}: OK')
"
```

Expected: both OK

- [ ] **Step 2: Verify import chain works**

Run:
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.gui.pages.smoothness import SmoothnessPage
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 3: Commit any remaining fixes**

If any fixes were needed, commit them:
```bash
git add -u
git commit -m "fix(smoothness): integration fixes for stats and export"
```
