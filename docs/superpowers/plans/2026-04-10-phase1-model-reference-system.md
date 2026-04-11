# Phase 1: Model Reference System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a model_specs database table, Excel import, dedicated Specs page for editing, Analyze page enrichment, and Dashboard/Trends filtering by element type and product class.

**Architecture:** New `ModelSpec` SQLAlchemy model joined to existing `analysis_results` by model name (no FK). New Specs page follows existing page pattern. Import logic parses the reference Excel into the DB. Dashboard/Trends queries get optional filter parameters.

**Tech Stack:** SQLAlchemy 2.0, customtkinter, openpyxl (for xlsx import), existing app patterns

**Dev Environment Notes:**
- No pytest installed — use `python3 -c "import ast; ast.parse(open('file').read())"` for syntax checks
- No pydantic — use dataclasses or plain dicts
- Runtime: `python3` (not `python`)
- Working directory: `/Users/jb631/projects/laser-trim-ai-system-v5/`
- Branch: `v5-upgrade`

---

### Task 1: Add ModelSpec Database Model

**Files:**
- Modify: `src/laser_trim_analyzer/database/models.py`

- [ ] **Step 1: Add ModelSpec class to database models**

Add after the `ModelMLState` class (around line 1290):

```python
class ModelSpec(Base):
    """Model engineering specifications — element type, linearity type, resistance, angle, etc."""
    __tablename__ = "model_specs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model = Column(String(50), unique=True, nullable=False, index=True)
    element_type = Column(String(30), nullable=True)
    product_class = Column(String(20), nullable=True)
    linearity_type = Column(String(30), nullable=True)
    linearity_spec_text = Column(String(100), nullable=True)
    linearity_spec_pct = Column(Float, nullable=True)
    total_resistance_min = Column(Float, nullable=True)
    total_resistance_max = Column(Float, nullable=True)
    electrical_angle = Column(Float, nullable=True)
    electrical_angle_tol = Column(Float, nullable=True)
    electrical_angle_unit = Column(String(10), nullable=True)
    output_smoothness = Column(String(50), nullable=True)
    circuit_type = Column(String(10), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    def __repr__(self):
        return f"<ModelSpec(model='{self.model}', element='{self.element_type}', class='{self.product_class}')>"
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/models.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/laser_trim_analyzer/database/models.py
git commit -m "feat: add ModelSpec database model for engineering specs"
```

---

### Task 2: Add Table Migration and CRUD Methods

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py`

- [ ] **Step 1: Import ModelSpec and add table creation**

Add `ModelSpec` to the imports from `database.models` (around line 19). Then in the `_ensure_tables()` or `__init__` migration section, add:

```python
# Create model_specs table if it doesn't exist
ModelSpec.__table__.create(self.engine, checkfirst=True)
```

- [ ] **Step 2: Add CRUD methods for model specs**

Add these methods to `DatabaseManager`:

```python
def get_all_model_specs(self) -> List[Dict[str, Any]]:
    """Get all model specs as dicts."""
    with self.session() as session:
        specs = session.query(ModelSpec).order_by(ModelSpec.model).all()
        return [
            {
                "id": s.id,
                "model": s.model,
                "element_type": s.element_type,
                "product_class": s.product_class,
                "linearity_type": s.linearity_type,
                "linearity_spec_text": s.linearity_spec_text,
                "linearity_spec_pct": s.linearity_spec_pct,
                "total_resistance_min": s.total_resistance_min,
                "total_resistance_max": s.total_resistance_max,
                "electrical_angle": s.electrical_angle,
                "electrical_angle_tol": s.electrical_angle_tol,
                "electrical_angle_unit": s.electrical_angle_unit,
                "output_smoothness": s.output_smoothness,
                "circuit_type": s.circuit_type,
                "notes": s.notes,
            }
            for s in specs
        ]

def get_model_spec(self, model: str) -> Optional[Dict[str, Any]]:
    """Get spec for a specific model."""
    with self.session() as session:
        spec = session.query(ModelSpec).filter(
            ModelSpec.model == model
        ).first()
        if not spec:
            return None
        return {
            "id": spec.id,
            "model": spec.model,
            "element_type": spec.element_type,
            "product_class": spec.product_class,
            "linearity_type": spec.linearity_type,
            "linearity_spec_text": spec.linearity_spec_text,
            "linearity_spec_pct": spec.linearity_spec_pct,
            "total_resistance_min": spec.total_resistance_min,
            "total_resistance_max": spec.total_resistance_max,
            "electrical_angle": spec.electrical_angle,
            "electrical_angle_tol": spec.electrical_angle_tol,
            "electrical_angle_unit": spec.electrical_angle_unit,
            "output_smoothness": spec.output_smoothness,
            "circuit_type": spec.circuit_type,
            "notes": spec.notes,
        }

def save_model_spec(self, data: Dict[str, Any]) -> int:
    """Create or update a model spec. Returns the spec ID."""
    with self._write_lock:
        with self.session() as session:
            existing = session.query(ModelSpec).filter(
                ModelSpec.model == data["model"]
            ).first()

            if existing:
                for key, value in data.items():
                    if key not in ("id", "model", "created_at"):
                        setattr(existing, key, value)
                existing.updated_at = utc_now()
                session.flush()
                return existing.id
            else:
                spec = ModelSpec(**{k: v for k, v in data.items() if k != "id"})
                session.add(spec)
                session.flush()
                return spec.id

def delete_model_spec(self, model: str) -> bool:
    """Delete a model spec. Returns True if found and deleted."""
    with self._write_lock:
        with self.session() as session:
            spec = session.query(ModelSpec).filter(
                ModelSpec.model == model
            ).first()
            if spec:
                session.delete(spec)
                return True
            return False

def get_distinct_element_types(self) -> List[str]:
    """Get all distinct element types from model_specs."""
    with self.session() as session:
        results = session.query(ModelSpec.element_type).filter(
            ModelSpec.element_type.isnot(None)
        ).distinct().order_by(ModelSpec.element_type).all()
        return [r[0] for r in results]

def get_distinct_product_classes(self) -> List[str]:
    """Get all distinct product classes from model_specs."""
    with self.session() as session:
        results = session.query(ModelSpec.product_class).filter(
            ModelSpec.product_class.isnot(None)
        ).distinct().order_by(ModelSpec.product_class).all()
        return [r[0] for r in results]
```

- [ ] **Step 3: Add import_model_specs_from_excel method**

```python
def import_model_specs_from_excel(self, file_path: str) -> Dict[str, int]:
    """
    Import model specs from the reference Excel file.
    Merges: updates existing, adds new, never deletes.

    Returns: {"updated": N, "added": N, "skipped": N}
    """
    import re
    import openpyxl

    wb = openpyxl.load_workbook(file_path, read_only=True)
    result = {"updated": 0, "added": 0, "skipped": 0}

    # Collect data from all three sheets
    model_data = {}  # model -> dict of fields

    # Sheet 1: Model Reference (primary, most complete)
    if "Model Reference" in wb.sheetnames:
        ws = wb["Model Reference"]
        for row in ws.iter_rows(min_row=2, values_only=True):
            model = str(row[1]).strip() if row[1] else None
            if not model:
                continue

            # Skip Customer (row[2]) and Operator (row[9])
            element_type = str(row[3]).strip() if row[3] else None
            linearity_text = str(row[4]).strip() if row[4] else None
            resistance_text = str(row[5]).strip() if row[5] else None
            angle_text = str(row[6]).strip() if row[6] else None
            smoothness = str(row[7]).strip() if row[7] else None
            circuit = str(row[8]).strip() if row[8] else None
            product_class = str(row[10]).strip() if row[10] else None

            # Parse linearity type from text
            linearity_type = None
            linearity_pct = None
            if linearity_text:
                # Extract type: look for (Absolute), (Independent), etc.
                type_match = re.search(
                    r'\((Absolute|Independent|Term Base|Zero-Based|VR Max)\)',
                    linearity_text, re.IGNORECASE
                )
                if type_match:
                    linearity_type = type_match.group(1)
                elif any(kw in linearity_text.lower() for kw in
                         ['see chart', 'see table', 'function', 'trim according']):
                    linearity_type = "Custom"

                # Extract percentage: look for ± N.N% or +/- N.N%
                pct_match = re.search(r'[±]\s*(\d+\.?\d*)\s*%', linearity_text)
                if not pct_match:
                    pct_match = re.search(r'\+/?-?\s*\.?(\d+\.?\d*)\s*%', linearity_text)
                if pct_match:
                    try:
                        linearity_pct = float(pct_match.group(1))
                    except ValueError:
                        pass

            # Parse resistance: "950 - 1,050 Ω" → min=950, max=1050
            r_min = None
            r_max = None
            if resistance_text:
                r_match = re.search(
                    r'([\d,]+\.?\d*)\s*[-–]\s*([\d,]+\.?\d*)',
                    resistance_text
                )
                if r_match:
                    try:
                        r_min = float(r_match.group(1).replace(',', ''))
                        r_max = float(r_match.group(2).replace(',', ''))
                    except ValueError:
                        pass

            # Parse angle: '1.31" ± .005"' or '240° ± 2°'
            angle_val = None
            angle_tol = None
            angle_unit = None
            if angle_text:
                # Try inches format: N.NN" ± .NNN"
                a_match = re.search(r'([\d.]+)"?\s*[±]\s*\.?([\d.]+)', angle_text)
                if a_match:
                    try:
                        angle_val = float(a_match.group(1))
                        angle_tol = float('0.' + a_match.group(2)) if '.' not in a_match.group(2) else float(a_match.group(2))
                        angle_unit = "in"
                    except ValueError:
                        pass
                if angle_val is None:
                    # Try just a number (degrees or inches)
                    num_match = re.search(r'([\d.]+)', angle_text)
                    if num_match:
                        try:
                            angle_val = float(num_match.group(1))
                            angle_unit = "in" if '"' in angle_text else "deg"
                        except ValueError:
                            pass

            # Clean element type
            if element_type and element_type.startswith('N/A'):
                element_type = element_type  # Keep as-is for switches/wipers

            model_data[model] = {
                "model": model,
                "element_type": element_type,
                "product_class": product_class,
                "linearity_type": linearity_type,
                "linearity_spec_text": linearity_text,
                "linearity_spec_pct": linearity_pct,
                "total_resistance_min": r_min,
                "total_resistance_max": r_max,
                "electrical_angle": angle_val,
                "electrical_angle_tol": angle_tol,
                "electrical_angle_unit": angle_unit,
                "output_smoothness": smoothness if smoothness and smoothness != 'None' else None,
                "circuit_type": circuit if circuit and circuit != 'None' else None,
            }

    # Sheet 2: Element Type (supplement — broader coverage)
    if "Element Type" in wb.sheetnames:
        ws = wb["Element Type"]
        for row in ws.iter_rows(min_row=2, values_only=True):
            model = str(row[0]).strip() if row[0] else None
            etype = str(row[1]).strip() if row[1] else None
            if model and etype:
                if model not in model_data:
                    model_data[model] = {"model": model, "element_type": etype}
                elif not model_data[model].get("element_type"):
                    model_data[model]["element_type"] = etype

    # Sheet 3: Product Class (supplement — broadest coverage)
    if "Product Class" in wb.sheetnames:
        ws = wb["Product Class"]
        for row in ws.iter_rows(min_row=2, values_only=True):
            model = str(row[0]).strip() if row[0] else None
            pclass = str(row[1]).strip() if row[1] else None
            if model and pclass:
                if model not in model_data:
                    model_data[model] = {"model": model, "product_class": pclass}
                elif not model_data[model].get("product_class"):
                    model_data[model]["product_class"] = pclass

    wb.close()

    # Save to database (merge logic)
    for model, data in model_data.items():
        try:
            with self.session() as session:
                existing = session.query(ModelSpec).filter(
                    ModelSpec.model == model
                ).first()

            if existing:
                self.save_model_spec(data)
                result["updated"] += 1
            else:
                self.save_model_spec(data)
                result["added"] += 1
        except Exception as e:
            logger.warning(f"Skipping model spec {model}: {e}")
            result["skipped"] += 1

    logger.info(
        f"Model specs import: {result['added']} added, "
        f"{result['updated']} updated, {result['skipped']} skipped"
    )
    return result
```

- [ ] **Step 4: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('OK')"
```

- [ ] **Step 5: Test import with the reference Excel**

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from laser_trim_analyzer.database.manager import DatabaseManager
db = DatabaseManager('data/test_specs.db')
result = db.import_model_specs_from_excel('work files/work files/model_reference_table_cleaned.xlsx')
print(result)
specs = db.get_all_model_specs()
print(f'Total specs: {len(specs)}')
for s in specs[:5]:
    print(f'  {s[\"model\"]}: element={s[\"element_type\"]}, class={s[\"product_class\"]}, lin={s[\"linearity_type\"]}')
import os; os.remove('data/test_specs.db')
"
```

Expected: `{'added': ~350, 'updated': 0, 'skipped': 0}` with spec details printed.

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py
git commit -m "feat: add model spec CRUD methods and Excel import"
```

---

### Task 3: Create Specs Page

**Files:**
- Create: `src/laser_trim_analyzer/gui/pages/specs.py`
- Modify: `src/laser_trim_analyzer/app.py`

- [ ] **Step 1: Create the Specs page**

Create `src/laser_trim_analyzer/gui/pages/specs.py` — a customtkinter page with:
- Scrollable table of all model specs
- Search box to filter by model number
- Add/Edit/Delete functionality
- Detail panel for editing a selected model

The page follows the same pattern as other pages (extends `ctk.CTkFrame`, has `on_show()` method).

This is a large file (~500 lines). Key sections:
- `__init__`: Create the layout with search bar, table, and edit panel
- `on_show()`: Load all specs from DB and populate table
- `_load_specs()`: Query DB and populate the table widget
- `_on_search()`: Filter table rows by search text
- `_on_select()`: Show selected model in edit panel
- `_save_spec()`: Save edited spec back to DB
- `_add_new()`: Clear edit panel for new model entry
- `_delete_spec()`: Delete with confirmation

Since this is a substantial UI file, create the basic skeleton first with the table and search, then add editing in the next step.

- [ ] **Step 2: Add Specs page to app navigation**

In `src/laser_trim_analyzer/app.py`, add the Specs page:

In the `nav_items` list (around line 90), add before the export entry:
```python
("specs", "Model Specs", 8),
```

Shift export and settings rows down by 1.

In `_create_pages()` (around line 148), add the import and page creation:
```python
from laser_trim_analyzer.gui.pages.specs import SpecsPage
self._pages["specs"] = SpecsPage(self.main_frame, self)
```

- [ ] **Step 3: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/specs.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/app.py').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/specs.py src/laser_trim_analyzer/app.py
git commit -m "feat: add Model Specs page with table view and editing"
```

---

### Task 4: Add Import Button to Settings Page

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/settings.py`

- [ ] **Step 1: Add import section to Settings**

In `_create_database_section()` or as a new section, add an "Import Model Specs" button with a file picker:

```python
def _create_model_specs_import_section(self, container):
    """Create model specs import section."""
    frame = ctk.CTkFrame(container)
    frame.grid(sticky="ew", padx=10, pady=10)
    frame.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
        frame, text="Model Specifications",
        font=ctk.CTkFont(size=16, weight="bold")
    ).grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

    btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
    btn_frame.grid(row=1, column=0, padx=15, pady=5, sticky="w")

    ctk.CTkButton(
        btn_frame,
        text="Import from Excel",
        width=150,
        command=self._import_model_specs,
    ).pack(side="left", padx=(0, 10))

    self.import_specs_label = ctk.CTkLabel(
        btn_frame, text="", text_color="gray", font=ctk.CTkFont(size=11)
    )
    self.import_specs_label.pack(side="left")

def _import_model_specs(self):
    """Import model specs from Excel file."""
    from tkinter import filedialog, messagebox
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
        self.import_specs_label.configure(text=msg, text_color="#27ae60")
        messagebox.showinfo("Import Complete", msg)
    except Exception as e:
        self.import_specs_label.configure(text=f"Error: {e}", text_color="#e74c3c")
```

Call `_create_model_specs_import_section(container)` from the main settings layout method.

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/settings.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/settings.py
git commit -m "feat: add model specs Excel import button to Settings"
```

---

### Task 5: Enrich Analyze Page with Model Specs

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/analyze.py`

- [ ] **Step 1: Add model spec lookup to the analysis display**

In the `_update_file_info()` or equivalent method that populates the File Info tab, after displaying the existing analysis metadata, look up the model spec and append it:

```python
# Look up model specs
try:
    from laser_trim_analyzer.database import get_database
    db = get_database()
    spec = db.get_model_spec(analysis.model)
    if spec:
        lines.append("")
        lines.append("─" * 40)
        lines.append("  MODEL SPECIFICATIONS")
        lines.append("─" * 40)
        if spec["element_type"]:
            lines.append(f"  Element Type:    {spec['element_type']}")
        if spec["product_class"]:
            lines.append(f"  Product Class:   {spec['product_class']}")
        if spec["linearity_type"]:
            lines.append(f"  Linearity Type:  {spec['linearity_type']}")
        if spec["linearity_spec_text"]:
            lines.append(f"  Linearity Spec:  {spec['linearity_spec_text']}")
        if spec["total_resistance_min"] and spec["total_resistance_max"]:
            lines.append(f"  Resistance:      {spec['total_resistance_min']:.0f} - {spec['total_resistance_max']:.0f} Ω")
        if spec["electrical_angle"]:
            angle_str = f"{spec['electrical_angle']}"
            if spec["electrical_angle_tol"]:
                angle_str += f" ± {spec['electrical_angle_tol']}"
            if spec["electrical_angle_unit"]:
                angle_str += f" {spec['electrical_angle_unit']}"
            lines.append(f"  Elec. Angle:     {angle_str}")
        if spec["output_smoothness"]:
            lines.append(f"  Smoothness:      {spec['output_smoothness']}")
        if spec["circuit_type"]:
            lines.append(f"  Circuit:         {spec['circuit_type']}")
except Exception:
    pass  # Model specs not available — no problem
```

Find the exact insertion point by reading the current `_update_file_info` method and locating the end of the existing metadata display.

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/analyze.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/analyze.py
git commit -m "feat: show model specs in Analyze page File Info tab"
```

---

### Task 6: Add Dashboard Filtering by Element Type / Product Class

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py`
- Modify: `src/laser_trim_analyzer/gui/pages/dashboard.py`

- [ ] **Step 1: Add filter parameters to dashboard stats query**

Find `get_dashboard_stats()` in `manager.py`. Add optional `element_type` and `product_class` parameters. When provided, add a subquery join to `model_specs`:

```python
def get_dashboard_stats(self, days_back: int = 30, element_type: str = None, product_class: str = None) -> Dict[str, Any]:
```

Add model filtering logic at the start of the query building:

```python
# Build base query with optional model spec filters
model_filter_ids = None
if element_type or product_class:
    with self.session() as session:
        q = session.query(ModelSpec.model)
        if element_type:
            q = q.filter(ModelSpec.element_type == element_type)
        if product_class:
            q = q.filter(ModelSpec.product_class == product_class)
        filter_models = [r[0] for r in q.all()]
        # Use model name filter on analysis queries
```

Then add `.filter(DBAnalysisResult.model.in_(filter_models))` to each subquery when `filter_models` is set.

- [ ] **Step 2: Add filter dropdowns to Dashboard page**

In the Dashboard page, add Element Type and Product Class dropdowns at the top. When changed, re-run the data load with the filter applied.

```python
# Filter frame at top of dashboard
filter_frame = ctk.CTkFrame(self, fg_color="transparent")
filter_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))

ctk.CTkLabel(filter_frame, text="Filter:").pack(side="left", padx=(0, 5))

self.element_filter = ctk.CTkComboBox(
    filter_frame,
    values=["All Element Types"],
    command=self._on_filter_changed,
    width=150
)
self.element_filter.pack(side="left", padx=5)

self.class_filter = ctk.CTkComboBox(
    filter_frame,
    values=["All Product Classes"],
    command=self._on_filter_changed,
    width=150
)
self.class_filter.pack(side="left", padx=5)
```

Populate the dropdown values from `db.get_distinct_element_types()` and `db.get_distinct_product_classes()` in `on_show()`.

- [ ] **Step 3: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/dashboard.py').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py src/laser_trim_analyzer/gui/pages/dashboard.py
git commit -m "feat: add element type and product class filtering to Dashboard"
```

---

### Task 7: Add Trends Page Filtering

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py`
- Modify: `src/laser_trim_analyzer/database/manager.py`

- [ ] **Step 1: Add filter parameters to trends queries**

Find the trends query methods in `manager.py` (e.g., `get_linearity_daily_trend`, `get_model_trend_data`, etc.). Add the same `element_type` and `product_class` optional parameters with model_specs join filtering, following the same pattern as Task 6.

- [ ] **Step 2: Add filter dropdowns to Trends page**

Add the same Element Type and Product Class dropdowns to the Trends page header, following the same pattern as Task 6. Wire them to re-run the trend data queries when changed.

- [ ] **Step 3: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/trends.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/trends.py src/laser_trim_analyzer/database/manager.py
git commit -m "feat: add element type and product class filtering to Trends"
```

---

### Task 8: Add Performance Breakdown Charts to Dashboard

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py`
- Modify: `src/laser_trim_analyzer/gui/pages/dashboard.py`

- [ ] **Step 1: Add query for pass rate by category**

Add to `manager.py`:

```python
def get_pass_rate_by_category(self, category: str = "element_type", days_back: int = 30) -> List[Dict[str, Any]]:
    """
    Get pass rate grouped by element_type or product_class.

    Args:
        category: "element_type" or "product_class"
        days_back: Number of days to look back

    Returns:
        List of dicts with keys: category, total, passed, pass_rate
    """
    with self.session() as session:
        cutoff = datetime.now() - timedelta(days=days_back)
        spec_col = ModelSpec.element_type if category == "element_type" else ModelSpec.product_class

        results = session.query(
            spec_col.label("category"),
            func.count(DBAnalysisResult.id).label("total"),
            func.sum(
                case(
                    (DBAnalysisResult.overall_status == DBStatusType.PASS, 1),
                    else_=0
                )
            ).label("passed")
        ).join(
            ModelSpec,
            DBAnalysisResult.model == ModelSpec.model
        ).filter(
            DBAnalysisResult.file_date >= cutoff,
            spec_col.isnot(None)
        ).group_by(spec_col).all()

        return [
            {
                "category": r.category,
                "total": r.total,
                "passed": r.passed,
                "pass_rate": (r.passed / r.total * 100) if r.total > 0 else 0
            }
            for r in results
        ]
```

- [ ] **Step 2: Add breakdown bar charts to Dashboard**

Add a "Performance by Category" section to the dashboard that shows two bar charts: one for element type and one for product class. Use the existing chart widget pattern or matplotlib directly.

- [ ] **Step 3: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/dashboard.py').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py src/laser_trim_analyzer/gui/pages/dashboard.py
git commit -m "feat: add pass rate by element type and product class charts"
```

---

### Task 9: Version Bump and Final Verification

**Files:**
- Modify: `src/laser_trim_analyzer/utils/constants.py`

- [ ] **Step 1: Bump version to 5.0.0**

In `constants.py`, change:
```python
APP_VERSION: Final[str] = "5.0.0"
```

- [ ] **Step 2: Full syntax check on all modified files**

```bash
python3 -c "
import ast, os
files = [
    'src/laser_trim_analyzer/database/models.py',
    'src/laser_trim_analyzer/database/manager.py',
    'src/laser_trim_analyzer/gui/pages/specs.py',
    'src/laser_trim_analyzer/gui/pages/settings.py',
    'src/laser_trim_analyzer/gui/pages/analyze.py',
    'src/laser_trim_analyzer/gui/pages/dashboard.py',
    'src/laser_trim_analyzer/gui/pages/trends.py',
    'src/laser_trim_analyzer/app.py',
    'src/laser_trim_analyzer/utils/constants.py',
]
for f in files:
    ast.parse(open(f).read())
    print(f'OK: {f}')
print('All files pass syntax check')
"
```

- [ ] **Step 3: Commit and push**

```bash
git add -A
git commit -m "feat: Phase 1 complete — Model Reference System v5.0.0"
git push origin v5-upgrade
```
