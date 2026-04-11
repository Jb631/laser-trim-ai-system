# Phase 3: Format Audit, Parsing Fixes & Output Smoothness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix pre-trim chart scaling, improve FT matching quality, cross-verify spec limits against model_specs, and integrate Output Smoothness as a new data source with its own parser, DB table, and Smoothness page (8th page).

**Architecture:** Pre-trim chart fix in ChartWidget. FT matching improvements in DatabaseManager. Cross-format spec verification adds a comparison query. Output Smoothness gets a new `SmoothnessParser`, `SmoothnessResult`/`SmoothnessTrack` DB models, CRUD methods, and a `SmoothnessPage` GUI page. The parser detects OS files by `_OS_` in filename.

**Tech Stack:** SQLAlchemy 2.0, customtkinter, openpyxl (for xlsx parsing), matplotlib, existing app patterns

**Dev Environment Notes:**
- No pytest installed — use `python3 -c "import ast; ast.parse(open('file').read())"` for syntax checks
- No pydantic — use dataclasses or plain dicts
- Runtime: `python3` (not `python`)
- Working directory: `/Users/jb631/projects/laser-trim-ai-system-v5/`
- Branch: `v5-upgrade`

**Assumes:** Phase 1 (Model Reference System) and Phase 2 are complete. `model_specs` table exists with `output_smoothness` column. All existing pages and DB models are functional.

---

## Part A: Pre-Trim Chart Scaling Fix

### Task 1: Fix Pre-Trim vs Post-Trim Y-Axis Scale Mismatch

**Problem:** When pre-trim (untrimmed) error range is much larger than post-trim (trimmed) error range, the untrimmed data overwhelms the chart and makes the trimmed data appear flat against the spec limits. This makes it impossible to visually assess post-trim quality.

**Solution:** Use dual Y-axes (twinx) when the untrimmed error range exceeds 3x the trimmed+spec range. The left axis shows trimmed data + spec limits at proper scale, the right axis shows untrimmed data. When ranges are comparable, keep single axis (current behavior).

**Files:**
- Modify: `src/laser_trim_analyzer/gui/widgets/chart.py`

- [ ] **Step 1: Add scale comparison logic to `plot_error_vs_position`**

In `src/laser_trim_analyzer/gui/widgets/chart.py`, modify the `plot_error_vs_position` method (around line 177). After the existing code that plots untrimmed data (around line 228), add scale-detection logic. The key change is:

1. Before plotting, calculate the Y-ranges of both datasets
2. If untrimmed range > 3x the trimmed+spec range, use a secondary Y-axis (twinx)
3. Otherwise, keep current single-axis behavior

Replace the entire `plot_error_vs_position` method with the updated version:

```python
def plot_error_vs_position(
    self,
    positions: List[float],
    trimmed_errors: List[float],
    upper_limits: Optional[List[float]] = None,
    lower_limits: Optional[List[float]] = None,
    untrimmed_positions: Optional[List[float]] = None,
    untrimmed_errors: Optional[List[float]] = None,
    offset: float = 0.0,
    title: str = "Error vs Position Analysis",
    fail_points: Optional[List[int]] = None,
    serial_number: Optional[str] = None,
    trim_date: Optional[str] = None,
    trim_improvement_percent: Optional[float] = None,
) -> None:
    """
    Plot error vs position - the main analysis chart.

    Auto-detects when pre-trim error range overwhelms post-trim and
    switches to dual Y-axes so both datasets are readable.
    """
    self.clear()
    ax = self.figure.add_subplot(111)
    self._style_axis(ax)

    # Apply offset to trimmed errors
    shifted_errors = [e + offset for e in trimmed_errors]

    # Determine if we need dual Y-axes
    use_dual_axis = False
    ax2 = None

    if untrimmed_positions and untrimmed_errors:
        # Calculate ranges
        trimmed_range = max(shifted_errors) - min(shifted_errors) if shifted_errors else 0
        untrimmed_range = max(untrimmed_errors) - min(untrimmed_errors) if untrimmed_errors else 0

        # Include spec limits in trimmed range calculation
        spec_max = trimmed_range
        if upper_limits and lower_limits:
            valid_upper = [u for u in upper_limits if u is not None]
            valid_lower = [l for l in lower_limits if l is not None]
            if valid_upper and valid_lower:
                spec_range = max(valid_upper) - min(valid_lower)
                spec_max = max(trimmed_range, spec_range)

        # Use dual axes when untrimmed range is > 3x the trimmed+spec range
        if spec_max > 0 and untrimmed_range > 3.0 * spec_max:
            use_dual_axis = True
            ax2 = ax.twinx()
            self._style_axis(ax2)
            # Make right axis label distinct
            ax2.set_ylabel('Pre-Trim Error', fontsize=self.style.font_size,
                          color=COLORS['untrimmed'])
            ax2.tick_params(axis='y', colors=COLORS['untrimmed'])

    # Plot untrimmed data
    if untrimmed_positions and untrimmed_errors:
        label_text = 'Untrimmed'
        if trim_improvement_percent is not None:
            label_text = f'Untrimmed (trim improved {trim_improvement_percent:.0f}%)'

        target_ax = ax2 if use_dual_axis else ax
        target_ax.plot(
            untrimmed_positions, untrimmed_errors,
            color=COLORS['untrimmed'],
            linestyle='--',
            linewidth=self.style.line_width,
            alpha=0.7 if not use_dual_axis else 0.5,
            label=label_text
        )

    # Plot trimmed data
    ax.plot(
        positions, shifted_errors,
        color=COLORS['trimmed'],
        linewidth=self.style.line_width,
        label=f'Trimmed (offset: {offset:.6f})'
    )

    # Plot specification limits (handle None = no limit at that position)
    if upper_limits and lower_limits:
        upper_plot = np.array([u if u is not None else np.nan for u in upper_limits])
        lower_plot = np.array([l if l is not None else np.nan for l in lower_limits])
        pos_array = np.array(positions[:len(upper_limits)])

        ax.plot(
            pos_array, upper_plot,
            color=COLORS['spec_limit'],
            linestyle='--',
            linewidth=1,
            alpha=0.8,
            label='Spec Limits'
        )
        ax.plot(
            pos_array, lower_plot,
            color=COLORS['spec_limit'],
            linestyle='--',
            linewidth=1,
            alpha=0.8
        )

        # Fill between limits only where both are defined
        ax.fill_between(
            pos_array,
            lower_plot,
            upper_plot,
            alpha=0.1,
            color=COLORS['spec_limit'],
            where=~np.isnan(upper_plot) & ~np.isnan(lower_plot)
        )

    # Mark fail points
    if fail_points:
        fail_x = [positions[i] for i in fail_points if i < len(positions)]
        fail_y = [shifted_errors[i] for i in fail_points if i < len(shifted_errors)]
        ax.scatter(
            fail_x, fail_y,
            color=COLORS['fail'],
            marker='x',
            s=50,
            linewidths=2,
            label='Fail Points',
            zorder=5
        )

    # Styling
    ax.set_xlabel('Position', fontsize=self.style.font_size)
    ax.set_ylabel('Error', fontsize=self.style.font_size)
    ax.set_title(title, fontsize=self.style.title_size)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])

    # Combine legends from both axes
    if use_dual_axis and ax2:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                 loc='best', fontsize=self.style.font_size - 2)
        # Add "(dual scale)" indicator to title
        ax.set_title(f"{title} (dual scale)", fontsize=self.style.title_size)
    else:
        ax.legend(loc='best', fontsize=self.style.font_size - 2)

    # Add SN and Trim Date info box in upper right corner
    if serial_number or trim_date:
        info_lines = []
        if serial_number:
            info_lines.append(f"SN: {serial_number}")
        if trim_date:
            info_lines.append(f"Trim Date: {trim_date}")
        info_text = "\n".join(info_lines)

        text_color = COLORS['text'] if self.style.dark_mode else 'black'
        bg_color = '#3d3d3d' if self.style.dark_mode else 'lightyellow'
        ax.text(0.98, 0.98, info_text,
               transform=ax.transAxes, fontsize=self.style.font_size - 1,
               va='top', ha='right',
               bbox=dict(boxstyle='round,pad=0.4', facecolor=bg_color, alpha=0.9, edgecolor='gray'),
               color=text_color)

    self.figure.tight_layout()
    self.canvas.draw()
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/widgets/chart.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/laser_trim_analyzer/gui/widgets/chart.py
git commit -m "fix: auto-scale pre-trim chart with dual Y-axis when ranges differ >3x"
```

---

## Part B: FT Matching Improvements

### Task 2: Add Match Quality Indicators to FT Results Display

**Problem:** Users cannot see *why* a match was made or *why* it was missed. The Compare page shows match_confidence as a number but doesn't explain whether it was exact serial, fuzzy serial, or model variant match.

**Solution:** Add a `match_method` field to FinalTestResult that records which matching tier succeeded. Display this on the Compare page alongside confidence.

**Files:**
- Modify: `src/laser_trim_analyzer/database/models.py`
- Modify: `src/laser_trim_analyzer/database/manager.py`

- [ ] **Step 1: Add match_method column to FinalTestResult**

In `src/laser_trim_analyzer/database/models.py`, add after the `days_since_trim` column (around line 1039):

```python
    match_method = Column(String(30))  # 'exact', 'fuzzy_serial', 'model_variant', or None
```

- [ ] **Step 2: Add migration for match_method column**

In `src/laser_trim_analyzer/database/manager.py`, in `_run_migrations()`, add after the existing migrations (before the `needs_rematch` block):

```python
            # Migration: Add match_method column to final_test_results
            try:
                session.execute(text("SELECT match_method FROM final_test_results LIMIT 1"))
            except OperationalError:
                logger.info("Running migration: Adding match_method column to final_test_results")
                try:
                    session.execute(text("ALTER TABLE final_test_results ADD COLUMN match_method VARCHAR(30)"))
                    session.commit()
                    logger.info("Migration completed: Added match_method column")
                except Exception as e:
                    logger.warning(f"match_method migration warning (may already exist): {e}")
```

- [ ] **Step 3: Update `_find_matching_trim` to return match method**

Modify the `_find_matching_trim` method signature and return value to include match_method. Change the return type from `Tuple[Optional[int], Optional[float], Optional[int]]` to `Tuple[Optional[int], Optional[float], Optional[int], Optional[str]]`.

Update each return statement:
- Attempt 1 (exact serial): return `match.id, confidence, days_diff, "exact"`
- Attempt 2 (fuzzy serial): return `trim_id, confidence, days_diff, "fuzzy_serial"`
- Attempt 3 (model variant): return `trim_id, confidence, days_diff, "model_variant"`
- No match: return `None, None, None, None`

- [ ] **Step 4: Update `save_final_test` to store match_method**

In `save_final_test`, update the `_find_matching_trim` call to unpack 4 values:

```python
linked_trim_id, match_confidence, days_since_trim, match_method = self._find_matching_trim(
    session,
    metadata.get("model"),
    metadata.get("serial"),
    metadata.get("file_date") or metadata.get("test_date")
)
```

And add `match_method=match_method` to the `DBFinalTestResult` constructor.

- [ ] **Step 5: Update `rematch_final_tests` to store match_method**

In `rematch_final_tests`, update the `_find_matching_trim` call to unpack 4 values:

```python
new_trim_id, new_confidence, new_days, new_method = self._find_matching_trim(
    session, ft.model, ft.serial, test_date
)
```

And add `ft.match_method = new_method` alongside the existing updates.

- [ ] **Step 6: Update `get_final_test` and `search_final_tests` to include match_method**

In `get_final_test`, add to the result dict:
```python
"match_method": result.match_method,
```

In `search_final_tests`, add to each pair dict:
```python
"match_method": ft.match_method,
```

- [ ] **Step 7: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/models.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('OK')"
```

- [ ] **Step 8: Commit**

```bash
git add src/laser_trim_analyzer/database/models.py src/laser_trim_analyzer/database/manager.py
git commit -m "feat: add match_method tracking to FT matching pipeline"
```

---

### Task 3: Improve Serial Normalization Edge Cases

**Problem:** Serial normalization in `_normalize_serial` may strip meaningful suffixes. For example, serials like `25D` (track D on unit 25) or `31R`/`31L` (right/left) lose the track/position indicator. The current code strips ALL trailing single letters, which causes false matches between different tracks of the same unit.

**Solution:** Only strip known track-indicator suffixes (A/B for dual-track, P/R for primary/redundant). Leave other suffixes (D, L, etc.) intact since they may be meaningful serial identifiers.

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py`

- [ ] **Step 1: Refine `_normalize_serial` to be more selective**

Replace the existing `_normalize_serial` method (around line 3546):

```python
@staticmethod
def _normalize_serial(serial: str) -> str:
    """
    Normalize a serial number for fuzzy matching.

    Handles common formatting differences between trim and FT files:
    - Strip leading zeros (007 -> 7)
    - Lowercase
    - Strip whitespace
    - Remove common prefixes (sn, s/n, #)
    - Strip known track-position suffixes only (A/B for dual-track,
      P/R for primary/redundant, T for test)
      e.g. 32A -> 32, 10P -> 10
    - Do NOT strip other letters (25D, 31L, 31R stay as-is since
      D/L/R may be meaningful serial identifiers)
    """
    import re
    s = serial.lower().strip()
    # Remove common prefixes
    s = re.sub(r'^(sn|s/n|s\.n\.|#)\s*', '', s)
    # Strip only known track-indicator suffixes:
    # a/b = dual-track positions, p/r = primary/redundant, t = test
    s = re.sub(r'^(\d+)[abprt]$', r'\1', s)
    # Strip leading zeros (but keep at least one digit for "0" itself)
    s = s.lstrip('0') or '0'
    return s
```

- [ ] **Step 2: Add `_normalize_serial_aggressive` for fallback matching**

Add a second, more aggressive normalizer that strips all trailing letters. This is used as a fallback in `_find_matching_trim` when the selective normalizer fails:

```python
@staticmethod
def _normalize_serial_aggressive(serial: str) -> str:
    """
    Aggressively normalize a serial number — strips ALL trailing letters.

    Used as a fallback when selective normalization fails to find a match.
    May produce false matches (e.g. 25D matches 25E) but increases recall.
    """
    import re
    s = serial.lower().strip()
    s = re.sub(r'^(sn|s/n|s\.n\.|#)\s*', '', s)
    # Strip ANY trailing single letter
    s = re.sub(r'^(\d+)[a-z]$', r'\1', s)
    s = s.lstrip('0') or '0'
    return s
```

- [ ] **Step 3: Update `_find_matching_trim` to use two-tier serial normalization**

In `_find_matching_trim`, after Attempt 2 (fuzzy serial with selective normalization) fails, add an Attempt 2b that uses aggressive normalization:

```python
        # Attempt 2b: Exact model + aggressively normalized serial (strips all trailing letters)
        ft_serial_aggressive = self._normalize_serial_aggressive(serial)
        if ft_serial_aggressive != ft_serial_norm:
            for trim_id, trim_serial, trim_date in model_trims:
                if trim_serial and self._normalize_serial_aggressive(trim_serial) == ft_serial_aggressive:
                    days_diff = (test_date - trim_date).days
                    # Lower confidence for aggressive normalization
                    confidence = self._calculate_match_confidence(days_diff, exact_serial=False) * 0.90
                    logger.debug(
                        f"Aggressive fuzzy match: FT serial '{serial}' -> trim serial '{trim_serial}' "
                        f"(aggressive norm: '{ft_serial_aggressive}'), {days_diff} days"
                    )
                    return trim_id, confidence, days_diff, "fuzzy_serial_aggressive"
```

Insert this between Attempt 2 and Attempt 3.

- [ ] **Step 4: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py
git commit -m "fix: refine serial normalization — selective suffix stripping with aggressive fallback"
```

---

### Task 4: Add Match Quality Display to Compare Page

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/compare.py`

- [ ] **Step 1: Show match method and confidence in the comparison detail panel**

Find the section in the Compare page where match_confidence is displayed (search for `match_confidence` in compare.py). Add display of the match_method next to it.

Where the confidence is shown, add human-readable labels:

```python
# Match quality display
method_labels = {
    "exact": "Exact Match",
    "fuzzy_serial": "Fuzzy Serial",
    "fuzzy_serial_aggressive": "Fuzzy Serial (aggressive)",
    "model_variant": "Model Variant",
}
match_method = pair.get("match_method", "unknown")
method_label = method_labels.get(match_method, match_method or "Unknown")
confidence = pair.get("match_confidence", 0) or 0

# Color-code confidence
if confidence >= 0.85:
    conf_color = "#27ae60"  # Green
elif confidence >= 0.70:
    conf_color = "#f39c12"  # Orange
else:
    conf_color = "#e74c3c"  # Red
```

Display both the method and confidence value in the detail area, e.g.:
```
Match: Fuzzy Serial (87%)
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/compare.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/compare.py
git commit -m "feat: show match method and color-coded confidence on Compare page"
```

---

## Part C: Cross-Format Spec Verification

### Task 5: Add Spec Limit Cross-Check Query

**Problem:** Different file formats (System A, System B, Final Test) parse linearity spec limits from file metadata. These may not match the engineering specification stored in `model_specs`. We need a way to detect mismatches.

**Solution:** Add a query that compares `track_results.linearity_spec` values against `model_specs.linearity_spec_pct` for each model. Flag models where the file-parsed spec diverges from the reference spec.

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py`

- [ ] **Step 1: Add `get_spec_discrepancies` method**

Add to `DatabaseManager`:

```python
def get_spec_discrepancies(self, tolerance_pct: float = 5.0) -> List[Dict[str, Any]]:
    """
    Compare file-parsed linearity specs against model_specs reference.

    Flags models where the spec parsed from trim files differs from the
    engineering reference by more than tolerance_pct percent.

    Args:
        tolerance_pct: Percentage tolerance for mismatch detection

    Returns:
        List of dicts with model, file_spec, reference_spec, difference_pct
    """
    from laser_trim_analyzer.database.models import ModelSpec

    results = []

    with self.session() as session:
        # Get distinct file-parsed specs per model
        file_specs = (
            session.query(
                DBAnalysisResult.model,
                func.avg(DBTrackResult.linearity_spec).label('avg_file_spec'),
                func.min(DBTrackResult.linearity_spec).label('min_file_spec'),
                func.max(DBTrackResult.linearity_spec).label('max_file_spec'),
                func.count(DBTrackResult.id).label('sample_count'),
            )
            .join(DBTrackResult, DBAnalysisResult.id == DBTrackResult.analysis_id)
            .filter(
                DBTrackResult.linearity_spec.isnot(None),
                DBTrackResult.linearity_spec > 0,
            )
            .group_by(DBAnalysisResult.model)
            .all()
        )

        for row in file_specs:
            # Look up reference spec
            ref = session.query(ModelSpec).filter(
                ModelSpec.model == row.model
            ).first()

            if not ref or not ref.linearity_spec_pct:
                continue

            ref_spec = ref.linearity_spec_pct / 100.0  # Convert % to decimal
            file_spec = row.avg_file_spec

            if ref_spec > 0:
                diff_pct = abs(file_spec - ref_spec) / ref_spec * 100
            else:
                diff_pct = 0

            if diff_pct > tolerance_pct:
                results.append({
                    "model": row.model,
                    "file_spec_avg": round(file_spec, 6),
                    "file_spec_min": round(row.min_file_spec, 6),
                    "file_spec_max": round(row.max_file_spec, 6),
                    "reference_spec_pct": ref.linearity_spec_pct,
                    "reference_spec_decimal": round(ref_spec, 6),
                    "difference_pct": round(diff_pct, 1),
                    "sample_count": row.sample_count,
                    "linearity_type": ref.linearity_type,
                })

    # Sort by biggest discrepancy first
    results.sort(key=lambda x: x["difference_pct"], reverse=True)
    return results
```

- [ ] **Step 2: Add spec discrepancy display to Specs page (or Quality Health page)**

Add a "Spec Discrepancies" section to the Specs page that calls `get_spec_discrepancies()` and shows a table of mismatches. This helps the user verify that file-parsed specs match engineering references.

Find the appropriate location in the Specs page (or create a new tab/section) and add:

```python
def _load_discrepancies(self):
    """Load spec discrepancy data."""
    try:
        from laser_trim_analyzer.database import get_database
        db = get_database()
        discrepancies = db.get_spec_discrepancies()
        # Display in table or text widget
        if discrepancies:
            self._show_discrepancy_table(discrepancies)
        else:
            self._show_no_discrepancies()
    except Exception as e:
        logger.warning(f"Could not load spec discrepancies: {e}")
```

- [ ] **Step 3: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py
git commit -m "feat: add cross-format spec verification query"
```

---

## Part D: Output Smoothness Integration

### Task 6: Add SmoothnessResult and SmoothnessTrack Database Models

**Files:**
- Modify: `src/laser_trim_analyzer/database/models.py`

- [ ] **Step 1: Add SmoothnessResult and SmoothnessTrack classes**

Add after the `FinalTestTrack` class (around line 1193). The Output Smoothness file format uses:
- Filename pattern: `model-snSerial_OS_date_time.xlsx` (sometimes with `_Primary_` or `_Redundant_` before `_OS_`)
- Sheets: "Test Data", "Report", "Rev History"
- "Test Data" sheet contains position vs output smoothness measurements

```python
class SmoothnessResult(Base):
    """
    Output Smoothness test file-level results.

    Stores results from output smoothness testing station.
    Linked to AnalysisResult by model + serial + date proximity.
    """
    __tablename__ = 'smoothness_results'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # File identification
    filename = Column(String(255), nullable=False)
    file_path = Column(Text)
    file_hash = Column(String(64))
    file_date = Column(DateTime)

    # Basic properties
    model = Column(String(50), nullable=False)
    serial = Column(String(100), nullable=False)
    element_label = Column(String(30))  # 'Primary', 'Redundant', or None

    # Test results
    test_date = Column(DateTime)
    overall_status = Column(Enum(StatusType), nullable=False)

    # Smoothness metrics (from Report sheet)
    smoothness_spec = Column(Float)          # Spec limit (from model_specs or file)
    max_smoothness_value = Column(Float)     # Worst (highest) smoothness reading
    avg_smoothness_value = Column(Float)     # Average smoothness value
    smoothness_pass = Column(Boolean)        # Overall pass/fail

    # Processing metadata
    timestamp = Column(DateTime, default=utc_now, nullable=False)
    processing_time = Column(Float)
    software_version = Column(String(20))

    # Link to matching trim result
    linked_trim_id = Column(Integer, ForeignKey('analysis_results.id', ondelete='SET NULL'))
    match_confidence = Column(Float)
    match_method = Column(String(30))
    days_since_trim = Column(Integer)

    # Relationships
    linked_trim = relationship("AnalysisResult", backref="smoothness_results", foreign_keys=[linked_trim_id])
    tracks = relationship(
        "SmoothnessTrack",
        back_populates="smoothness_result",
        cascade="all, delete-orphan",
        lazy="select"
    )

    # Indexes
    __table_args__ = (
        Index('idx_os_filename_date', 'filename', 'file_date'),
        Index('idx_os_model_serial', 'model', 'serial'),
        Index('idx_os_model_serial_date', 'model', 'serial', 'file_date'),
        Index('idx_os_timestamp', 'timestamp'),
        Index('idx_os_status', 'overall_status'),
        Index('idx_os_linked_trim', 'linked_trim_id'),
        UniqueConstraint('filename', 'file_date', 'model', 'serial', name='uq_smoothness_file'),
        CheckConstraint("LENGTH(TRIM(filename)) > 0", name='check_os_filename_not_empty'),
        CheckConstraint("LENGTH(TRIM(model)) > 0", name='check_os_model_not_empty'),
        CheckConstraint("LENGTH(TRIM(serial)) > 0", name='check_os_serial_not_empty'),
    )

    @validates('filename')
    def validate_filename(self, key, filename):
        if not filename or not filename.strip():
            raise ValueError("Filename cannot be empty")
        return filename.strip()

    @validates('model')
    def validate_model(self, key, model):
        if not model or not model.strip():
            raise ValueError("Model cannot be empty")
        return model.strip()

    @validates('serial')
    def validate_serial(self, key, serial):
        if not serial or not serial.strip():
            raise ValueError("Serial cannot be empty")
        return serial.strip()

    def __repr__(self):
        return f"<SmoothnessResult(id={self.id}, model='{self.model}', serial='{self.serial}', pass={self.smoothness_pass})>"


class SmoothnessTrack(Base):
    """
    Output Smoothness track-level data.

    Stores position vs smoothness measurement arrays for charting.
    """
    __tablename__ = 'smoothness_tracks'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign key
    smoothness_id = Column(Integer, ForeignKey('smoothness_results.id'), nullable=False)

    # Track identification
    track_id = Column(String(20), nullable=False, default='default')
    status = Column(Enum(StatusType), nullable=False)

    # Smoothness data
    smoothness_spec = Column(Float)
    max_smoothness = Column(Float)
    avg_smoothness = Column(Float)
    smoothness_pass = Column(Boolean)

    # Raw data for charts (JSON arrays)
    position_data = Column(SafeJSON, nullable=True)
    smoothness_data = Column(SafeJSON, nullable=True)  # Smoothness values at each position
    upper_limit_data = Column(SafeJSON, nullable=True)  # Spec limit at each position

    # Relationships
    smoothness_result = relationship("SmoothnessResult", back_populates="tracks")

    # Indexes
    __table_args__ = (
        Index('idx_ost_smoothness', 'smoothness_id', 'track_id'),
        Index('idx_ost_status', 'status'),
        UniqueConstraint('smoothness_id', 'track_id', name='uq_smoothness_track'),
        CheckConstraint("LENGTH(TRIM(track_id)) > 0", name='check_ost_track_id_not_empty'),
    )

    @validates('track_id')
    def validate_track_id(self, key, track_id):
        if not track_id or not track_id.strip():
            raise ValueError("Track ID cannot be empty")
        return track_id.strip()

    def __repr__(self):
        return f"<SmoothnessTrack(id={self.id}, smoothness_id={self.smoothness_id}, track_id='{self.track_id}')>"
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/models.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/laser_trim_analyzer/database/models.py
git commit -m "feat: add SmoothnessResult and SmoothnessTrack database models"
```

---

### Task 7: Create Output Smoothness Parser

**Files:**
- Create: `src/laser_trim_analyzer/core/smoothness_parser.py`

The OS file format from the sample files:
- Filename: `model-snSerial[_Primary|_Redundant]_OS_date_time.xlsx`
- Date format in filename: `M-D-YYYY` (e.g., `4-10-2026`)
- Time format in filename: `H-MM-SS AM/PM` (e.g., `8-07-07 AM`)
- Sheets: "Test Data" (measurement data), "Report" (summary), "Rev History"
- Extensions: `.xlsx` and `.xls`

- [ ] **Step 1: Create the smoothness parser**

Create `src/laser_trim_analyzer/core/smoothness_parser.py`:

```python
"""
Output Smoothness file parser for Laser Trim Analyzer.

Parses output smoothness test files (.xlsx/.xls) from the test station.
Filename pattern: model-snSerial[_Primary|_Redundant]_OS_date_time.xlsx
Sheets: Test Data, Report, Rev History
"""

import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Extensions supported
OS_EXTENSIONS = {'.xlsx', '.xls'}

# Filename pattern: model-snSerial[_element]_OS_date_time.ext
# Examples:
#   8275-sn75_OS_4-10-2026_8-07-07 AM.xlsx
#   8080-6-sn1023_Primary_OS_3-20-2026_1-39-30 PM.xlsx
#   8867-05-sn10_OS_8-1-2025_11-01 AM.xls
OS_FILENAME_PATTERN = re.compile(
    r'^(.+?)-sn(.+?)(?:_(Primary|Redundant))?_OS_'
    r'(\d{1,2}-\d{1,2}-\d{4})_'
    r'(\d{1,2}-\d{2}(?:-\d{2})?\s*(?:AM|PM))'
    r'\.(xlsx?)',
    re.IGNORECASE
)


def is_smoothness_file(filename: str) -> bool:
    """Check if a filename matches the output smoothness pattern."""
    return '_OS_' in filename and Path(filename).suffix.lower() in OS_EXTENSIONS


class SmoothnessParser:
    """
    Parser for Output Smoothness Excel files.

    Handles:
    - Filename parsing (model, serial, element label, date)
    - Test Data sheet extraction (positions, smoothness values)
    - Report sheet extraction (pass/fail, spec limits)
    """

    def __init__(self):
        pass

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse an Output Smoothness Excel file.

        Args:
            file_path: Path to the .xlsx/.xls file

        Returns:
            Dictionary with:
            - metadata: Dict with model, serial, element_label, test_date, etc.
            - tracks: List of track data dicts with positions, smoothness values
            - file_hash: SHA256 hash for deduplication
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in OS_EXTENSIONS:
            raise ValueError(f"Not a supported smoothness file: {file_path}")

        logger.info(f"Parsing Output Smoothness file: {file_path.name}")

        # Calculate file hash
        file_hash = self._calculate_hash(file_path)

        # Parse filename for metadata
        metadata = self._parse_filename(file_path.name)
        metadata["filename"] = file_path.name
        metadata["file_path"] = str(file_path)

        # Parse the Excel content
        tracks = []
        try:
            with pd.ExcelFile(file_path) as xl:
                sheet_names = xl.sheet_names

                # Parse Test Data sheet (primary data source)
                if "Test Data" in sheet_names:
                    tracks = self._parse_test_data_sheet(xl)
                elif len(sheet_names) > 0:
                    # Try the first sheet as fallback
                    tracks = self._parse_test_data_sheet(xl, sheet_name=sheet_names[0])

                # Parse Report sheet for summary/spec info
                if "Report" in sheet_names:
                    report_data = self._parse_report_sheet(xl)
                    metadata.update(report_data)

        except Exception as e:
            logger.error(f"Error parsing smoothness file {file_path.name}: {e}")
            raise

        return {
            "metadata": metadata,
            "tracks": tracks,
            "file_hash": file_hash,
        }

    def _parse_filename(self, filename: str) -> Dict[str, Any]:
        """
        Extract model, serial, element label, and date from filename.

        Examples:
            8275-sn75_OS_4-10-2026_8-07-07 AM.xlsx
            -> model=8275, serial=75, element_label=None, file_date=2026-04-10

            8080-6-sn1023_Primary_OS_3-20-2026_1-39-30 PM.xlsx
            -> model=8080-6, serial=1023, element_label=Primary, file_date=2026-03-20
        """
        result = {
            "model": "Unknown",
            "serial": "Unknown",
            "element_label": None,
            "file_date": None,
            "test_date": None,
        }

        match = OS_FILENAME_PATTERN.match(filename)
        if match:
            result["model"] = match.group(1)
            result["serial"] = match.group(2)
            result["element_label"] = match.group(3)  # Primary/Redundant or None

            # Parse date: M-D-YYYY
            try:
                date_str = match.group(4)
                result["file_date"] = datetime.strptime(date_str, "%m-%d-%Y")
            except ValueError:
                logger.warning(f"Could not parse date from filename: {filename}")

            # Parse time: H-MM-SS AM/PM or H-MM AM/PM
            try:
                time_str = match.group(5).strip()
                # Normalize time format
                # Handle both "8-07-07 AM" and "11-01 AM" formats
                parts = time_str.replace(' ', '-').split('-')
                if len(parts) >= 4:
                    # H-MM-SS-AM/PM
                    h, m, s, ampm = parts[0], parts[1], parts[2], parts[3]
                    time_parsed = datetime.strptime(f"{h}:{m}:{s} {ampm}", "%I:%M:%S %p")
                elif len(parts) == 3:
                    # H-MM-AM/PM
                    h, m, ampm = parts[0], parts[1], parts[2]
                    time_parsed = datetime.strptime(f"{h}:{m} {ampm}", "%I:%M %p")
                else:
                    time_parsed = None

                if time_parsed and result["file_date"]:
                    result["test_date"] = result["file_date"].replace(
                        hour=time_parsed.hour,
                        minute=time_parsed.minute,
                        second=time_parsed.second if len(parts) >= 4 else 0,
                    )
            except (ValueError, IndexError):
                logger.warning(f"Could not parse time from filename: {filename}")
        else:
            # Fallback: try to extract model and serial from non-standard names
            stem = Path(filename).stem
            # Look for _OS_ marker and parse before it
            os_idx = stem.find('_OS_')
            if os_idx > 0:
                prefix = stem[:os_idx]
                # Try sn pattern
                sn_match = re.match(r'^(.+?)-sn(.+?)(?:_(Primary|Redundant))?$', prefix, re.IGNORECASE)
                if sn_match:
                    result["model"] = sn_match.group(1)
                    result["serial"] = sn_match.group(2)
                    result["element_label"] = sn_match.group(3)

        return result

    def _parse_test_data_sheet(
        self,
        xl: pd.ExcelFile,
        sheet_name: str = "Test Data"
    ) -> List[Dict[str, Any]]:
        """
        Parse the Test Data sheet for position vs smoothness data.

        Returns list of track dicts, each with:
        - track_id: 'default' or element-specific
        - positions: List[float]
        - smoothness_values: List[float]
        - smoothness_spec: float (if found)
        - max_smoothness: float
        - avg_smoothness: float
        - smoothness_pass: bool
        """
        try:
            df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
        except Exception as e:
            logger.warning(f"Could not read sheet '{sheet_name}': {e}")
            return []

        if df.empty:
            return []

        tracks = []

        # Strategy: look for columns with position data and smoothness data
        # The test station typically outputs columns like:
        # Position (or Angle or Travel), Smoothness (or Output Smoothness)
        # Try to find header row first
        header_row = None
        for i in range(min(20, len(df))):
            row_values = [str(v).lower().strip() for v in df.iloc[i] if pd.notna(v)]
            # Look for position-related headers
            if any(kw in val for val in row_values for kw in ['position', 'angle', 'travel', 'degrees', 'inches']):
                header_row = i
                break

        if header_row is not None:
            # Re-read with header
            df_data = pd.read_excel(xl, sheet_name=sheet_name, header=header_row)
            df_data.columns = [str(c).strip() for c in df_data.columns]

            # Find position column
            pos_col = None
            for col in df_data.columns:
                col_lower = col.lower()
                if any(kw in col_lower for kw in ['position', 'angle', 'travel', 'degrees', 'inches']):
                    pos_col = col
                    break

            # Find smoothness column(s)
            smooth_cols = []
            spec_value = None
            for col in df_data.columns:
                col_lower = col.lower()
                if 'smooth' in col_lower or 'output' in col_lower:
                    smooth_cols.append(col)
                if 'spec' in col_lower or 'limit' in col_lower:
                    # Try to extract spec value from this column
                    valid_vals = df_data[col].dropna()
                    if len(valid_vals) > 0:
                        try:
                            spec_value = float(valid_vals.iloc[0])
                        except (ValueError, TypeError):
                            pass

            if pos_col and smooth_cols:
                positions = pd.to_numeric(df_data[pos_col], errors='coerce').dropna().tolist()

                for sc in smooth_cols:
                    values = pd.to_numeric(df_data[sc], errors='coerce').dropna().tolist()
                    if not values:
                        continue

                    # Align lengths
                    min_len = min(len(positions), len(values))
                    pos = positions[:min_len]
                    vals = values[:min_len]

                    max_val = max(vals) if vals else 0
                    avg_val = sum(vals) / len(vals) if vals else 0

                    # Determine pass/fail from spec
                    passes = True
                    if spec_value is not None and spec_value > 0:
                        passes = max_val <= spec_value

                    track = {
                        "track_id": "default",
                        "positions": pos,
                        "smoothness_values": vals,
                        "smoothness_spec": spec_value,
                        "max_smoothness": max_val,
                        "avg_smoothness": avg_val,
                        "smoothness_pass": passes,
                    }
                    tracks.append(track)
        else:
            # No header found — try to extract numeric columns
            # Assume first numeric column is position, second is smoothness
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                positions = df[numeric_cols[0]].dropna().tolist()
                values = df[numeric_cols[1]].dropna().tolist()
                min_len = min(len(positions), len(values))
                if min_len > 0:
                    pos = positions[:min_len]
                    vals = values[:min_len]
                    tracks.append({
                        "track_id": "default",
                        "positions": pos,
                        "smoothness_values": vals,
                        "smoothness_spec": None,
                        "max_smoothness": max(vals),
                        "avg_smoothness": sum(vals) / len(vals),
                        "smoothness_pass": None,  # Unknown without spec
                    })

        return tracks

    def _parse_report_sheet(self, xl: pd.ExcelFile) -> Dict[str, Any]:
        """
        Parse the Report sheet for summary info.

        Returns dict with any extracted metadata:
        - smoothness_spec, overall_pass, operator, etc.
        """
        result = {}
        try:
            df = pd.read_excel(xl, sheet_name="Report", header=None)
            if df.empty:
                return result

            # Scan for key-value pairs in the report
            for i in range(len(df)):
                for j in range(len(df.columns) - 1):
                    label = str(df.iloc[i, j]).strip().lower() if pd.notna(df.iloc[i, j]) else ""
                    value = df.iloc[i, j + 1] if j + 1 < len(df.columns) and pd.notna(df.iloc[i, j + 1]) else None

                    if value is None:
                        continue

                    if 'spec' in label and 'smooth' in label:
                        try:
                            result["smoothness_spec"] = float(value)
                        except (ValueError, TypeError):
                            pass
                    elif label in ('result', 'pass/fail', 'status'):
                        val_str = str(value).strip().lower()
                        result["overall_pass"] = val_str in ('pass', 'passed', 'ok', 'yes', 'true')
                    elif label in ('operator', 'tester'):
                        result["operator"] = str(value).strip()
                    elif 'model' in label:
                        result["report_model"] = str(value).strip()
                    elif 'serial' in label:
                        result["report_serial"] = str(value).strip()

        except Exception as e:
            logger.warning(f"Could not parse Report sheet: {e}")

        return result

    @staticmethod
    def _calculate_hash(file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/smoothness_parser.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/laser_trim_analyzer/core/smoothness_parser.py
git commit -m "feat: add Output Smoothness file parser"
```

---

### Task 8: Add Smoothness CRUD Methods to DatabaseManager

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py`

- [ ] **Step 1: Add SmoothnessResult import to manager.py**

At the top of `manager.py`, add `SmoothnessResult` and `SmoothnessTrack` to the imports from `database.models` (around line 28). Since these are new models that may not exist in older DB schemas, use a deferred import pattern:

```python
# Add to the existing imports block:
# from laser_trim_analyzer.database.models import ...SmoothnessResult, SmoothnessTrack...
# OR use deferred imports in the methods themselves (safer for backwards compatibility)
```

Use deferred imports inside the methods to avoid import errors on older databases.

- [ ] **Step 2: Add `save_smoothness_result` method**

Add to `DatabaseManager`, after the Final Test methods section:

```python
# =========================================================================
# Output Smoothness Methods
# =========================================================================

def save_smoothness_result(
    self,
    metadata: Dict[str, Any],
    tracks: List[Dict[str, Any]],
    file_hash: str
) -> int:
    """
    Save an Output Smoothness result to the database.

    Args:
        metadata: Dict with filename, model, serial, element_label, test_date, etc.
        tracks: List of track data dicts with positions, smoothness_values, etc.
        file_hash: SHA256 hash of the file

    Returns:
        ID of saved SmoothnessResult
    """
    from laser_trim_analyzer.database.models import (
        SmoothnessResult as DBSmoothnessResult,
        SmoothnessTrack as DBSmoothnessTrack,
    )

    with self._write_lock:
        try:
            with self.session() as session:
                # Check for duplicate by file_hash
                existing = (
                    session.query(DBSmoothnessResult)
                    .filter(DBSmoothnessResult.file_hash == file_hash)
                    .first()
                )
                if existing:
                    logger.info(f"Smoothness result already exists: {metadata.get('filename')}")
                    return existing.id

                # Determine overall status
                overall_status = DBStatusType.PASS
                for track in tracks:
                    if track.get("smoothness_pass") is False:
                        overall_status = DBStatusType.FAIL
                        break

                # Calculate overall metrics
                max_smoothness = max(
                    (t.get("max_smoothness", 0) or 0 for t in tracks), default=0
                )
                avg_smoothness = sum(
                    t.get("avg_smoothness", 0) or 0 for t in tracks
                ) / len(tracks) if tracks else 0

                smoothness_spec = metadata.get("smoothness_spec") or (
                    tracks[0].get("smoothness_spec") if tracks else None
                )

                smoothness_pass = all(
                    t.get("smoothness_pass", True) for t in tracks
                ) if tracks else None

                # Find matching trim result (reuse existing FT matching logic)
                linked_trim_id, match_confidence, days_since_trim, match_method = self._find_matching_trim(
                    session,
                    metadata.get("model"),
                    metadata.get("serial"),
                    metadata.get("file_date") or metadata.get("test_date")
                )

                # Create SmoothnessResult
                db_result = DBSmoothnessResult(
                    filename=metadata.get("filename", "unknown"),
                    file_path=str(metadata.get("file_path", "")),
                    file_hash=file_hash,
                    file_date=metadata.get("file_date"),
                    model=metadata.get("model", "unknown"),
                    serial=metadata.get("serial", "unknown"),
                    element_label=metadata.get("element_label"),
                    test_date=metadata.get("test_date"),
                    overall_status=overall_status,
                    smoothness_spec=smoothness_spec,
                    max_smoothness_value=max_smoothness,
                    avg_smoothness_value=avg_smoothness,
                    smoothness_pass=smoothness_pass,
                    linked_trim_id=linked_trim_id,
                    match_confidence=match_confidence,
                    match_method=match_method,
                    days_since_trim=days_since_trim,
                )

                session.add(db_result)
                session.flush()
                result_id = db_result.id

                # Add tracks
                for track_data in tracks:
                    db_track = DBSmoothnessTrack(
                        smoothness_id=result_id,
                        track_id=track_data.get("track_id", "default"),
                        status=DBStatusType.PASS if track_data.get("smoothness_pass", True) else DBStatusType.FAIL,
                        smoothness_spec=track_data.get("smoothness_spec"),
                        max_smoothness=track_data.get("max_smoothness"),
                        avg_smoothness=track_data.get("avg_smoothness"),
                        smoothness_pass=track_data.get("smoothness_pass"),
                        position_data=track_data.get("positions"),
                        smoothness_data=track_data.get("smoothness_values"),
                        upper_limit_data=(
                            [track_data["smoothness_spec"]] * len(track_data.get("positions", []))
                            if track_data.get("smoothness_spec")
                            else None
                        ),
                    )
                    session.add(db_track)

                session.commit()
                logger.info(
                    f"Saved Smoothness: {metadata.get('filename')} "
                    f"(ID: {result_id}, linked_trim: {linked_trim_id})"
                )
                return result_id

        except IntegrityError:
            logger.warning(f"Smoothness duplicate detected: {metadata.get('filename')}")
            with self.session() as session:
                existing = (
                    session.query(DBSmoothnessResult)
                    .filter(DBSmoothnessResult.file_hash == file_hash)
                    .first()
                )
                if existing:
                    return existing.id
            raise

def search_smoothness_results(
    self,
    model: Optional[str] = None,
    serial: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    limit: int = 500
) -> List[Dict[str, Any]]:
    """
    Search Output Smoothness results with filters.
    """
    from laser_trim_analyzer.database.models import (
        SmoothnessResult as DBSmoothnessResult,
    )

    with self.session() as session:
        query = session.query(DBSmoothnessResult).options(
            joinedload(DBSmoothnessResult.linked_trim)
        )

        if model and model != "All Models":
            query = query.filter(DBSmoothnessResult.model == model)

        if serial and serial.strip():
            serial_pattern = f"%{serial.strip()}%"
            query = query.filter(
                func.lower(DBSmoothnessResult.serial).like(func.lower(serial_pattern))
            )

        if date_from:
            query = query.filter(DBSmoothnessResult.file_date >= date_from)
        if date_to:
            end_of_day = date_to.replace(hour=23, minute=59, second=59)
            query = query.filter(DBSmoothnessResult.file_date <= end_of_day)

        results = query.order_by(desc(DBSmoothnessResult.file_date)).limit(limit).all()

        return [
            {
                "id": r.id,
                "filename": r.filename,
                "model": r.model,
                "serial": r.serial,
                "element_label": r.element_label,
                "file_date": r.file_date,
                "test_date": r.test_date,
                "overall_status": r.overall_status.value if r.overall_status else "UNKNOWN",
                "smoothness_spec": r.smoothness_spec,
                "max_smoothness_value": r.max_smoothness_value,
                "smoothness_pass": r.smoothness_pass,
                "linked_trim_id": r.linked_trim_id,
                "match_confidence": r.match_confidence,
                "match_method": r.match_method,
                "days_since_trim": r.days_since_trim,
                "linked_trim": {
                    "model": r.linked_trim.model,
                    "serial": r.linked_trim.serial,
                    "file_date": r.linked_trim.file_date,
                    "overall_status": r.linked_trim.overall_status.value if r.linked_trim.overall_status else "UNKNOWN",
                } if r.linked_trim else None,
            }
            for r in results
        ]

def get_smoothness_result(self, result_id: int) -> Optional[Dict[str, Any]]:
    """Get a single Output Smoothness result by ID with tracks."""
    from laser_trim_analyzer.database.models import (
        SmoothnessResult as DBSmoothnessResult,
        SmoothnessTrack as DBSmoothnessTrack,
    )

    with self.session() as session:
        result = session.query(DBSmoothnessResult).filter(
            DBSmoothnessResult.id == result_id
        ).first()

        if not result:
            return None

        tracks = session.query(DBSmoothnessTrack).filter(
            DBSmoothnessTrack.smoothness_id == result_id
        ).all()

        linked_trim = None
        if result.linked_trim_id:
            linked_trim = self._get_analysis_summary(session, result.linked_trim_id)

        return {
            "id": result.id,
            "filename": result.filename,
            "model": result.model,
            "serial": result.serial,
            "element_label": result.element_label,
            "file_date": result.file_date,
            "test_date": result.test_date,
            "overall_status": result.overall_status.value if result.overall_status else "UNKNOWN",
            "smoothness_spec": result.smoothness_spec,
            "max_smoothness_value": result.max_smoothness_value,
            "avg_smoothness_value": result.avg_smoothness_value,
            "smoothness_pass": result.smoothness_pass,
            "linked_trim_id": result.linked_trim_id,
            "match_confidence": result.match_confidence,
            "match_method": result.match_method,
            "linked_trim": linked_trim,
            "tracks": [
                {
                    "track_id": t.track_id,
                    "status": t.status.value if t.status else "UNKNOWN",
                    "smoothness_spec": t.smoothness_spec,
                    "max_smoothness": t.max_smoothness,
                    "avg_smoothness": t.avg_smoothness,
                    "smoothness_pass": t.smoothness_pass,
                    "positions": t.position_data or [],
                    "smoothness_values": t.smoothness_data or [],
                    "upper_limits": t.upper_limit_data or [],
                }
                for t in tracks
            ],
        }

def get_smoothness_stats(self, days_back: int = 90) -> Dict[str, Any]:
    """Get Output Smoothness dashboard statistics."""
    from laser_trim_analyzer.database.models import (
        SmoothnessResult as DBSmoothnessResult,
    )

    with self.session() as session:
        cutoff_date = datetime.now() - timedelta(days=days_back)

        total = (
            session.query(func.count(DBSmoothnessResult.id))
            .filter(DBSmoothnessResult.file_date >= cutoff_date)
            .scalar()
        ) or 0

        if total == 0:
            return {"total": 0, "pass_rate": 0, "linked_count": 0, "link_rate": 0}

        passed = (
            session.query(func.count(DBSmoothnessResult.id))
            .filter(
                DBSmoothnessResult.file_date >= cutoff_date,
                DBSmoothnessResult.smoothness_pass == True,
            )
            .scalar()
        ) or 0

        linked = (
            session.query(func.count(DBSmoothnessResult.id))
            .filter(
                DBSmoothnessResult.file_date >= cutoff_date,
                DBSmoothnessResult.linked_trim_id.isnot(None),
            )
            .scalar()
        ) or 0

        return {
            "total": total,
            "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
            "linked_count": linked,
            "link_rate": round(linked / total * 100, 1) if total > 0 else 0,
        }
```

- [ ] **Step 3: Add smoothness file detection to `is_file_processed`**

In the `is_file_processed` method, add a check for SmoothnessResult after the existing FinalTestResult check:

```python
            # Also check Output Smoothness files
            from laser_trim_analyzer.database.models import SmoothnessResult as DBSmoothnessResult
            try:
                exists = (
                    session.query(DBSmoothnessResult)
                    .filter(DBSmoothnessResult.file_hash == file_hash)
                    .first()
                ) is not None
                if exists:
                    return True
            except Exception:
                pass  # Table may not exist yet
```

- [ ] **Step 4: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py
git commit -m "feat: add Output Smoothness CRUD methods and stats query"
```

---

### Task 9: Add Smoothness File Processing to Processor

**Files:**
- Modify: `src/laser_trim_analyzer/core/processor.py`
- Modify: `src/laser_trim_analyzer/core/parser.py`

- [ ] **Step 1: Add smoothness detection to file type detection**

In `src/laser_trim_analyzer/core/parser.py`, find the `detect_file_type` function and add smoothness detection. Add the `_OS_` filename check before the existing Final Test checks:

```python
# Check for Output Smoothness files (before FT check)
if '_OS_' in file_path.name:
    return 'smoothness'
```

This must come early since `_OS_` is a definitive marker for output smoothness files.

- [ ] **Step 2: Add smoothness processing to Processor**

In `src/laser_trim_analyzer/core/processor.py`, import the SmoothnessParser and add processing logic.

Add import at the top (after FinalTestParser import):

```python
from laser_trim_analyzer.core.smoothness_parser import SmoothnessParser, is_smoothness_file
```

Add to `__init__`:

```python
self.smoothness_parser = SmoothnessParser()
```

In the `process_file` method, add smoothness handling after the Final Test check but before the trim file processing:

```python
# Check for Output Smoothness files
if is_smoothness_file(file_path.name):
    return self._process_smoothness_file(file_path, start_time)
```

Add the processing method:

```python
def _process_smoothness_file(self, file_path: Path, start_time: float) -> AnalysisResult:
    """
    Process an Output Smoothness file.

    Parses the file, saves to database, and returns a marker result.
    """
    from laser_trim_analyzer.database import get_database

    try:
        parsed = self.smoothness_parser.parse_file(file_path)

        metadata = parsed["metadata"]
        tracks = parsed["tracks"]
        file_hash = parsed["file_hash"]

        # Save to database
        db = get_database()
        result_id = db.save_smoothness_result(
            metadata=metadata,
            tracks=tracks,
            file_hash=file_hash
        )

        processing_time = time.time() - start_time

        # Create minimal metadata for result
        minimal_metadata = FileMetadata(
            filename=metadata.get("filename", file_path.name),
            file_path=str(file_path),
            model=metadata.get("model", "unknown"),
            serial=metadata.get("serial", "unknown"),
            system=SystemType.UNKNOWN,
            file_date=metadata.get("file_date"),
        )

        # Determine status
        overall_status = AnalysisStatus.PASS
        if any(not t.get("smoothness_pass", True) for t in tracks):
            overall_status = AnalysisStatus.FAIL

        # Create minimal track data for display
        analyzed_tracks = []
        for track in tracks:
            track_data = TrackData(
                track_id=track.get("track_id", "default"),
                status=AnalysisStatus.PASS if track.get("smoothness_pass", True) else AnalysisStatus.FAIL,
                travel_length=1.0,
                linearity_spec=0.01,
                sigma_gradient=0.0,
                sigma_threshold=0.01,
                sigma_pass=True,
                optimal_offset=0.0,
                linearity_error=0.0,
                linearity_pass=True,
                linearity_fail_points=0,
            )
            analyzed_tracks.append(track_data)

        if not analyzed_tracks:
            # Create a dummy track so the result is valid
            analyzed_tracks.append(TrackData(
                track_id="default",
                status=AnalysisStatus.PASS,
                travel_length=1.0,
                linearity_spec=0.01,
                sigma_gradient=0.0,
                sigma_threshold=0.01,
                sigma_pass=True,
                optimal_offset=0.0,
                linearity_error=0.0,
                linearity_pass=True,
                linearity_fail_points=0,
            ))

        result = AnalysisResult(
            metadata=minimal_metadata,
            overall_status=overall_status,
            processing_time=processing_time,
            tracks=analyzed_tracks,
        )

        # Mark as smoothness file for special handling
        result.file_type = "smoothness"
        result.smoothness_id = result_id

        logger.info(
            f"Processed Smoothness: {file_path.name} - {overall_status.value} "
            f"(ID: {result_id}, {processing_time:.2f}s)"
        )

        return result

    except Exception as e:
        logger.exception(f"Error processing Smoothness {file_path.name}: {e}")
        return self._create_error_result(
            self._create_minimal_metadata(file_path),
            f"Smoothness error: {e}",
            start_time
        )
```

- [ ] **Step 3: Update `save_analysis` to skip smoothness files**

In `manager.py`'s `save_analysis` method, add the same skip logic used for Final Test:

```python
# Skip Smoothness files - they're already saved in processor
if getattr(analysis, 'file_type', 'trim') == 'smoothness':
    logger.debug(f"Skipping save_analysis for Smoothness: {analysis.metadata.filename}")
    return getattr(analysis, 'smoothness_id', -1) or -1
```

Add the same check to `save_batch`.

- [ ] **Step 4: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/processor.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/parser.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('OK')"
```

- [ ] **Step 5: Test smoothness file detection**

```bash
python3 -c "
from laser_trim_analyzer.core.smoothness_parser import is_smoothness_file, SmoothnessParser

# Test detection
assert is_smoothness_file('8275-sn75_OS_4-10-2026_8-07-07 AM.xlsx')
assert is_smoothness_file('8080-6-sn1023_Primary_OS_3-20-2026_1-39-30 PM.xlsx')
assert not is_smoothness_file('8275_sn75_4-10-2026.xlsx')
print('Detection OK')

# Test filename parsing
p = SmoothnessParser()
meta = p._parse_filename('8275-sn75_OS_4-10-2026_8-07-07 AM.xlsx')
assert meta['model'] == '8275'
assert meta['serial'] == '75'
assert meta['element_label'] is None
print(f'Parsed: model={meta[\"model\"]}, serial={meta[\"serial\"]}')

meta2 = p._parse_filename('8080-6-sn1023_Primary_OS_3-20-2026_1-39-30 PM.xlsx')
assert meta2['model'] == '8080-6'
assert meta2['serial'] == '1023'
assert meta2['element_label'] == 'Primary'
print(f'Parsed: model={meta2[\"model\"]}, serial={meta2[\"serial\"]}, element={meta2[\"element_label\"]}')

print('All tests passed')
"
```

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/core/processor.py src/laser_trim_analyzer/core/parser.py src/laser_trim_analyzer/database/manager.py
git commit -m "feat: integrate smoothness file processing into batch pipeline"
```

---

### Task 10: Add Smoothness Chart Method to ChartWidget

**Files:**
- Modify: `src/laser_trim_analyzer/gui/widgets/chart.py`

- [ ] **Step 1: Add `plot_smoothness` method to ChartWidget**

Add after the existing `plot_error_vs_position` method:

```python
def plot_smoothness(
    self,
    positions: List[float],
    smoothness_values: List[float],
    spec_limit: Optional[float] = None,
    title: str = "Output Smoothness",
    serial_number: Optional[str] = None,
    test_date: Optional[str] = None,
    element_label: Optional[str] = None,
) -> None:
    """
    Plot output smoothness vs position.

    Args:
        positions: Position values (travel or angle)
        smoothness_values: Smoothness measurements at each position
        spec_limit: Smoothness specification limit (horizontal line)
        title: Chart title
        serial_number: Serial number to display
        test_date: Test date to display
        element_label: 'Primary', 'Redundant', or None
    """
    self.clear()
    ax = self.figure.add_subplot(111)
    self._style_axis(ax)

    # Plot smoothness data
    label = 'Smoothness'
    if element_label:
        label = f'{element_label} Smoothness'

    ax.plot(
        positions, smoothness_values,
        color=COLORS['trimmed'],
        linewidth=self.style.line_width,
        label=label
    )

    # Plot spec limit
    if spec_limit is not None:
        ax.axhline(
            y=spec_limit,
            color=COLORS['spec_limit'],
            linestyle='--',
            linewidth=1,
            alpha=0.8,
            label=f'Spec Limit ({spec_limit})'
        )

        # Fill above spec limit (fail region)
        ax.fill_between(
            positions,
            spec_limit,
            max(max(smoothness_values), spec_limit * 1.1),
            alpha=0.05,
            color=COLORS['fail'],
        )

        # Mark points that exceed spec
        fail_x = [p for p, v in zip(positions, smoothness_values) if v > spec_limit]
        fail_y = [v for v in smoothness_values if v > spec_limit]
        if fail_x:
            ax.scatter(
                fail_x, fail_y,
                color=COLORS['fail'],
                marker='x',
                s=50,
                linewidths=2,
                label=f'Exceeds Spec ({len(fail_x)} pts)',
                zorder=5
            )

    # Styling
    ax.set_xlabel('Position', fontsize=self.style.font_size)
    ax.set_ylabel('Output Smoothness', fontsize=self.style.font_size)
    ax.set_title(title, fontsize=self.style.title_size)
    ax.legend(loc='best', fontsize=self.style.font_size - 2)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])

    # Info box
    if serial_number or test_date:
        info_lines = []
        if serial_number:
            info_lines.append(f"SN: {serial_number}")
        if test_date:
            info_lines.append(f"Date: {test_date}")
        if element_label:
            info_lines.append(f"Element: {element_label}")
        info_text = "\n".join(info_lines)

        text_color = COLORS['text'] if self.style.dark_mode else 'black'
        bg_color = '#3d3d3d' if self.style.dark_mode else 'lightyellow'
        ax.text(0.98, 0.98, info_text,
               transform=ax.transAxes, fontsize=self.style.font_size - 1,
               va='top', ha='right',
               bbox=dict(boxstyle='round,pad=0.4', facecolor=bg_color, alpha=0.9, edgecolor='gray'),
               color=text_color)

    self.figure.tight_layout()
    self.canvas.draw()
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/widgets/chart.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/laser_trim_analyzer/gui/widgets/chart.py
git commit -m "feat: add plot_smoothness chart method"
```

---

### Task 11: Create Smoothness Page (8th Page)

**Files:**
- Create: `src/laser_trim_analyzer/gui/pages/smoothness.py`

- [ ] **Step 1: Create the Smoothness page**

Create `src/laser_trim_analyzer/gui/pages/smoothness.py`. Follow the same pattern as `compare.py` (Final Test page):
- Filter controls at top (model dropdown, date range)
- Scrollable results list with model, serial, date, pass/fail
- Detail panel showing smoothness chart and metadata
- Pagination for large result sets

```python
"""
Smoothness Page - Output Smoothness test results with charts.

Displays Output Smoothness test data, links to trim results,
and shows smoothness vs position charts.
"""

import customtkinter as ctk
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from laser_trim_analyzer.database import get_database
from laser_trim_analyzer.config import get_config
from laser_trim_analyzer.utils.threads import get_thread_manager
from laser_trim_analyzer.gui.widgets.scrollable_combobox import ScrollableComboBox

if TYPE_CHECKING:
    from laser_trim_analyzer.gui.widgets.chart import ChartWidget

logger = logging.getLogger(__name__)


class SmoothnessPage(ctk.CTkFrame):
    """
    Smoothness page for Output Smoothness test results.

    Features:
    - Browse smoothness results from database
    - Filter by model and date
    - View smoothness vs position charts
    - Show linked trim data and match quality
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.current_result: Optional[Dict[str, Any]] = None
        self.results_list: List[Dict[str, Any]] = []

        # Pagination
        self._page_size = 20
        self._current_page = 0
        self._total_pages = 1

        # Chart widget (lazy loaded)
        self._chart: Optional['ChartWidget'] = None

        self._create_layout()

    def _create_layout(self):
        """Create the page layout."""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Top bar with filters
        self._create_filter_bar()

        # Main content: results list on left, detail/chart on right
        content = ctk.CTkFrame(self)
        content.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=2)

        # Results list (left)
        self._create_results_list(content)

        # Detail panel (right)
        self._create_detail_panel(content)

    def _create_filter_bar(self):
        """Create the filter bar at top."""
        bar = ctk.CTkFrame(self)
        bar.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))

        ctk.CTkLabel(bar, text="Output Smoothness",
                    font=ctk.CTkFont(size=18, weight="bold")).pack(side="left", padx=10)

        # Model filter
        ctk.CTkLabel(bar, text="Model:").pack(side="left", padx=(20, 5))
        self.model_var = ctk.StringVar(value="All Models")
        self.model_dropdown = ScrollableComboBox(
            bar, variable=self.model_var,
            values=["All Models"],
            command=lambda _: self._load_results(),
            width=120
        )
        self.model_dropdown.pack(side="left", padx=5)

        # Refresh button
        ctk.CTkButton(bar, text="Refresh", width=80,
                     command=self._load_results).pack(side="left", padx=10)

        # Stats label
        self.stats_label = ctk.CTkLabel(bar, text="", text_color="gray")
        self.stats_label.pack(side="right", padx=10)

    def _create_results_list(self, parent):
        """Create the scrollable results list."""
        list_frame = ctk.CTkFrame(parent)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        # Scrollable frame for results
        self.results_scroll = ctk.CTkScrollableFrame(list_frame)
        self.results_scroll.grid(row=0, column=0, sticky="nsew")
        self.results_scroll.grid_columnconfigure(0, weight=1)

        # Pagination controls
        page_frame = ctk.CTkFrame(list_frame)
        page_frame.grid(row=1, column=0, sticky="ew", pady=5)

        self.prev_btn = ctk.CTkButton(page_frame, text="<", width=30,
                                      command=self._prev_page)
        self.prev_btn.pack(side="left", padx=5)

        self.page_label = ctk.CTkLabel(page_frame, text="Page 1/1")
        self.page_label.pack(side="left", padx=5)

        self.next_btn = ctk.CTkButton(page_frame, text=">", width=30,
                                      command=self._next_page)
        self.next_btn.pack(side="left", padx=5)

    def _create_detail_panel(self, parent):
        """Create the detail panel with chart and info."""
        detail = ctk.CTkFrame(parent)
        detail.grid(row=0, column=1, sticky="nsew")
        detail.grid_rowconfigure(1, weight=1)
        detail.grid_columnconfigure(0, weight=1)

        # Info section
        self.info_text = ctk.CTkTextbox(detail, height=120)
        self.info_text.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Chart section
        self.chart_frame = ctk.CTkFrame(detail)
        self.chart_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    def _get_chart(self) -> 'ChartWidget':
        """Lazy-load the chart widget."""
        if self._chart is None:
            from laser_trim_analyzer.gui.widgets.chart import ChartWidget
            self._chart = ChartWidget(self.chart_frame)
            self._chart.pack(fill="both", expand=True)
        return self._chart

    def on_show(self):
        """Called when page becomes visible."""
        self._load_models()
        self._load_results()

    def on_hide(self):
        """Called when page is hidden — release chart memory."""
        if self._chart:
            self._chart.clear()

    def _load_models(self):
        """Load model list for dropdown."""
        try:
            db = get_database()
            models = db.get_models_list()
            values = ["All Models"] + models
            self.model_dropdown.configure(values=values)
        except Exception as e:
            logger.warning(f"Could not load models: {e}")

    def _load_results(self):
        """Load smoothness results from database."""
        def _do_load():
            try:
                db = get_database()
                model = self.model_var.get()
                if model == "All Models":
                    model = None
                results = db.search_smoothness_results(model=model, limit=500)
                stats = db.get_smoothness_stats()
                return results, stats
            except Exception as e:
                logger.error(f"Error loading smoothness results: {e}")
                return [], {}

        def _on_loaded(future):
            try:
                results, stats = future.result()
                self.results_list = results
                self._current_page = 0
                self._total_pages = max(1, (len(results) + self._page_size - 1) // self._page_size)
                self.after(0, lambda: self._display_results())
                self.after(0, lambda: self._display_stats(stats))
            except Exception as e:
                logger.error(f"Error displaying smoothness results: {e}")

        get_thread_manager().start_thread(
            "smoothness_load", _do_load, callback=_on_loaded
        )

    def _display_results(self):
        """Display current page of results."""
        # Clear existing
        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        start = self._current_page * self._page_size
        end = start + self._page_size
        page_results = self.results_list[start:end]

        if not page_results:
            ctk.CTkLabel(self.results_scroll, text="No results found",
                        text_color="gray").grid(row=0, column=0, pady=20)
            return

        for i, result in enumerate(page_results):
            self._create_result_row(i, result)

        self.page_label.configure(text=f"Page {self._current_page + 1}/{self._total_pages}")

    def _create_result_row(self, row: int, result: Dict[str, Any]):
        """Create a single result row in the list."""
        status = result.get("overall_status", "UNKNOWN")
        is_pass = status == "PASS"

        frame = ctk.CTkFrame(self.results_scroll, cursor="hand2")
        frame.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        frame.grid_columnconfigure(1, weight=1)

        # Status indicator
        color = "#27ae60" if is_pass else "#e74c3c"
        ctk.CTkLabel(frame, text="  ", width=8,
                    fg_color=color, corner_radius=2).grid(row=0, column=0, padx=(2, 5), pady=2)

        # Model + Serial
        text = f"{result['model']} SN{result['serial']}"
        if result.get("element_label"):
            text += f" ({result['element_label']})"
        ctk.CTkLabel(frame, text=text, font=ctk.CTkFont(size=12)).grid(
            row=0, column=1, sticky="w", padx=2)

        # Date
        date_str = ""
        if result.get("file_date"):
            date_str = result["file_date"].strftime("%Y-%m-%d")
        ctk.CTkLabel(frame, text=date_str, text_color="gray",
                    font=ctk.CTkFont(size=11)).grid(row=0, column=2, padx=5)

        # Click handler
        frame.bind("<Button-1>", lambda e, r=result: self._on_result_selected(r))
        for child in frame.winfo_children():
            child.bind("<Button-1>", lambda e, r=result: self._on_result_selected(r))

    def _on_result_selected(self, result: Dict[str, Any]):
        """Handle result selection — load detail and chart."""
        self.current_result = result

        def _do_load():
            db = get_database()
            return db.get_smoothness_result(result["id"])

        def _on_loaded(future):
            try:
                detail = future.result()
                if detail:
                    self.after(0, lambda: self._display_detail(detail))
            except Exception as e:
                logger.error(f"Error loading smoothness detail: {e}")

        get_thread_manager().start_thread(
            "smoothness_detail", _do_load, callback=_on_loaded
        )

    def _display_detail(self, detail: Dict[str, Any]):
        """Display selected result detail and chart."""
        # Update info text
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")

        lines = []
        lines.append(f"Model: {detail['model']}    Serial: {detail['serial']}")
        if detail.get("element_label"):
            lines.append(f"Element: {detail['element_label']}")
        lines.append(f"Status: {detail['overall_status']}")
        if detail.get("smoothness_spec"):
            lines.append(f"Spec Limit: {detail['smoothness_spec']}")
        if detail.get("max_smoothness_value") is not None:
            lines.append(f"Max Smoothness: {detail['max_smoothness_value']:.4f}")
        if detail.get("linked_trim"):
            lt = detail["linked_trim"]
            lines.append(f"Linked Trim: {lt['model']} SN{lt['serial']} ({lt['overall_status']})")
            if detail.get("match_confidence"):
                lines.append(f"Match: {detail.get('match_method', 'unknown')} ({detail['match_confidence']:.0%})")

        self.info_text.insert("1.0", "\n".join(lines))
        self.info_text.configure(state="disabled")

        # Plot chart
        if detail.get("tracks"):
            track = detail["tracks"][0]
            positions = track.get("positions", [])
            values = track.get("smoothness_values", [])

            if positions and values:
                chart = self._get_chart()
                chart.plot_smoothness(
                    positions=positions,
                    smoothness_values=values,
                    spec_limit=track.get("smoothness_spec") or detail.get("smoothness_spec"),
                    title=f"Output Smoothness - {detail['model']} SN{detail['serial']}",
                    serial_number=detail["serial"],
                    test_date=detail["test_date"].strftime("%Y-%m-%d") if detail.get("test_date") else None,
                    element_label=detail.get("element_label"),
                )

    def _display_stats(self, stats: Dict[str, Any]):
        """Display summary stats in the stats label."""
        if not stats or stats.get("total", 0) == 0:
            self.stats_label.configure(text="No data")
            return

        text = (
            f"Total: {stats['total']}  |  "
            f"Pass: {stats['pass_rate']:.0f}%  |  "
            f"Linked: {stats['link_rate']:.0f}%"
        )
        self.stats_label.configure(text=text)

    def _prev_page(self):
        if self._current_page > 0:
            self._current_page -= 1
            self._display_results()

    def _next_page(self):
        if self._current_page < self._total_pages - 1:
            self._current_page += 1
            self._display_results()
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/smoothness.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/smoothness.py
git commit -m "feat: add Smoothness page for output smoothness test results"
```

---

### Task 12: Add Smoothness Page to App Navigation

**Files:**
- Modify: `src/laser_trim_analyzer/app.py`

- [ ] **Step 1: Add Smoothness to nav_items**

In `src/laser_trim_analyzer/app.py`, add the Smoothness page to the `nav_items` list (around line 90).

Current nav_items:
```python
nav_items = [
    ("dashboard", "Dashboard", 2),
    ("quality_health", "Quality Health", 3),
    ("process", "Process Files", 4),
    ("analyze", "Analyze Trim", 5),
    ("compare", "Final Test", 6),
    ("trends", "Trends", 7),
    ("export", "Export", 8),
]
```

Add Smoothness after Final Test and shift the remaining items down:
```python
nav_items = [
    ("dashboard", "Dashboard", 2),
    ("quality_health", "Quality Health", 3),
    ("process", "Process Files", 4),
    ("analyze", "Analyze Trim", 5),
    ("compare", "Final Test", 6),
    ("smoothness", "Smoothness", 7),
    ("trends", "Trends", 8),
    ("export", "Export", 9),
]
```

Update the spacer row in `_create_sidebar` to account for the new button:
```python
self.sidebar.grid_rowconfigure(10, weight=1)  # Spacer row (adjusted for 8 nav items)
```

And update the Settings button row:
```python
settings_btn.grid(row=11, column=0, padx=10, pady=(5, 20), sticky="ew")
```

- [ ] **Step 2: Add Smoothness page to `_create_pages`**

In `_create_pages`, add the import and page creation:

```python
from laser_trim_analyzer.gui.pages.smoothness import SmoothnessPage
```

And in the page creation block:
```python
self._pages["smoothness"] = SmoothnessPage(self.main_frame, self)
```

- [ ] **Step 3: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/app.py').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/app.py
git commit -m "feat: add Smoothness page to navigation (8th page)"
```

---

### Task 13: Final Verification and Version Bump

**Files:**
- Modify: `src/laser_trim_analyzer/utils/constants.py` (if version bump needed)

- [ ] **Step 1: Full syntax check on all modified and created files**

```bash
python3 -c "
import ast, os
files = [
    'src/laser_trim_analyzer/gui/widgets/chart.py',
    'src/laser_trim_analyzer/database/models.py',
    'src/laser_trim_analyzer/database/manager.py',
    'src/laser_trim_analyzer/core/parser.py',
    'src/laser_trim_analyzer/core/processor.py',
    'src/laser_trim_analyzer/core/smoothness_parser.py',
    'src/laser_trim_analyzer/gui/pages/compare.py',
    'src/laser_trim_analyzer/gui/pages/smoothness.py',
    'src/laser_trim_analyzer/app.py',
]
for f in files:
    ast.parse(open(f).read())
    print(f'OK: {f}')
print('All files pass syntax check')
"
```

- [ ] **Step 2: Test smoothness parser with sample files**

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from laser_trim_analyzer.core.smoothness_parser import SmoothnessParser, is_smoothness_file

p = SmoothnessParser()

# Test with a real sample file
import os
sample_dir = 'work files/work files/Sample_Base_2026-04-10/Smoothness_Sample_2026-04-10/Test Station'
if os.path.exists(sample_dir):
    for root, dirs, files in os.walk(sample_dir):
        for f in files[:3]:  # Test first 3 files
            if is_smoothness_file(f):
                fpath = os.path.join(root, f)
                result = p.parse_file(fpath)
                meta = result['metadata']
                tracks = result['tracks']
                print(f'{f}: model={meta[\"model\"]}, serial={meta[\"serial\"]}, '
                      f'element={meta.get(\"element_label\")}, tracks={len(tracks)}')
                if tracks:
                    t = tracks[0]
                    print(f'  positions={len(t.get(\"positions\", []))}, '
                          f'max_smooth={t.get(\"max_smoothness\")}, '
                          f'pass={t.get(\"smoothness_pass\")}')
        break
print('Sample file test complete')
"
```

- [ ] **Step 3: Commit final**

```bash
git add -A
git commit -m "feat: Phase 3 complete — format audit, FT matching fixes, output smoothness integration"
```
