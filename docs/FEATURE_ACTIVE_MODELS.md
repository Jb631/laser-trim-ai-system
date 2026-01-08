# Feature: Active Model Prioritization

**Status:** Planned
**Created:** 2026-01-07
**Priority:** Enhancement

---

## Overview

Add ability to mark models as "active" based on MPS (Master Production Schedule) and prioritize them across all application pages. Inactive/legacy models are shown at the bottom of lists, not hidden.

### Problem
Many models in the database are old/inactive, making it harder to find current production models in dropdowns and lists.

### Solution
Three-tier model priority system:
1. **MPS Active** - Models on MPS schedule (user-managed list, updated quarterly)
2. **Recently Active** - Models with data processed in last 90 days (auto-detected)
3. **Inactive** - All other models (shown at bottom of lists)

---

## User Workflow

1. Open Settings page → "Active Models" section
2. Paste model numbers from MPS spreadsheet (one per line)
3. Click Save
4. All dropdowns across the app now prioritize those models
5. Update quarterly when MPS changes

---

## Implementation Plan

### Step 1: Add ActiveModelsConfig to config.py

**File:** `src/laser_trim_analyzer/config.py`

Add new dataclass after existing config classes (~line 45):

```python
@dataclass
class ActiveModelsConfig:
    """Configuration for active model prioritization."""
    mps_models: List[str] = field(default_factory=list)  # User-managed MPS list
    recent_days: int = 90  # Days to consider "recently active"
    prioritize_active: bool = True  # Enable/disable feature

    def is_mps_model(self, model: str) -> bool:
        """Check if model is on MPS schedule."""
        return model in self.mps_models
```

Update `AppConfig` dataclass to include:
```python
active_models: ActiveModelsConfig = field(default_factory=ActiveModelsConfig)
```

Update `_load_config()` and `save()` methods to handle new section.

---

### Step 2: Add Database Method for Prioritized Model List

**File:** `src/laser_trim_analyzer/database/manager.py`

Add new method after `get_models_list()` (~line 1660):

```python
def get_models_list_prioritized(
    self,
    mps_models: List[str] = None,
    recent_days: int = 90
) -> List[Dict[str, Any]]:
    """
    Get models sorted by priority: MPS → Recently Active → Inactive.

    Args:
        mps_models: List of models on MPS schedule
        recent_days: Days threshold for "recently active"

    Returns:
        List of dicts: [{'model': str, 'status': 'mps'|'active'|'inactive', 'count': int}]
    """
    mps_models = mps_models or []
    mps_set = set(mps_models)
    cutoff_date = datetime.now() - timedelta(days=recent_days)

    with self.session() as session:
        # Get all models with their latest activity date and count
        results = (
            session.query(
                DBAnalysisResult.model,
                func.max(DBAnalysisResult.file_date).label('latest_date'),
                func.count(DBAnalysisResult.id).label('count')
            )
            .filter(DBAnalysisResult.model.isnot(None))
            .group_by(DBAnalysisResult.model)
            .all()
        )

        # Categorize models
        mps_list = []
        active_list = []
        inactive_list = []

        for model, latest_date, count in results:
            if not model:
                continue

            entry = {
                'model': model,
                'count': count,
                'latest_date': latest_date
            }

            if model in mps_set:
                entry['status'] = 'mps'
                mps_list.append(entry)
            elif latest_date and latest_date >= cutoff_date:
                entry['status'] = 'active'
                active_list.append(entry)
            else:
                entry['status'] = 'inactive'
                inactive_list.append(entry)

        # Sort each group alphabetically
        mps_list.sort(key=lambda x: x['model'])
        active_list.sort(key=lambda x: x['model'])
        inactive_list.sort(key=lambda x: x['model'])

        return mps_list + active_list + inactive_list
```

Add helper method for simple string list:

```python
def get_models_list_prioritized_simple(
    self,
    mps_models: List[str] = None,
    recent_days: int = 90,
    include_status_suffix: bool = True
) -> List[str]:
    """
    Get prioritized model list as simple strings for dropdowns.

    Args:
        mps_models: List of models on MPS schedule
        recent_days: Days threshold for "recently active"
        include_status_suffix: If True, append " (inactive)" to inactive models

    Returns:
        List of model strings, prioritized
    """
    models = self.get_models_list_prioritized(mps_models, recent_days)

    result = []
    for m in models:
        if include_status_suffix and m['status'] == 'inactive':
            result.append(f"{m['model']} (inactive)")
        else:
            result.append(m['model'])

    return result
```

---

### Step 3: Add Active Models Section to Settings Page

**File:** `src/laser_trim_analyzer/gui/pages/settings.py`

Add new section in `_create_ui()` after existing sections (~line 280):

```python
def _create_active_models_section(self, container):
    """Create Active Models (MPS) configuration section."""
    frame = ctk.CTkFrame(container)
    frame.grid(sticky="ew", padx=10, pady=10)

    # Title
    title = ctk.CTkLabel(
        frame,
        text="Active Models (MPS Schedule)",
        font=ctk.CTkFont(size=16, weight="bold")
    )
    title.pack(padx=15, pady=(15, 5), anchor="w")

    # Description
    desc = ctk.CTkLabel(
        frame,
        text="Models on MPS schedule are prioritized in all dropdowns.\nEnter one model per line.",
        text_color="gray",
        justify="left"
    )
    desc.pack(padx=15, pady=(0, 10), anchor="w")

    # Text area for model list
    self.mps_textbox = ctk.CTkTextbox(
        frame,
        height=150,
        width=300,
        font=ctk.CTkFont(family="Consolas", size=12)
    )
    self.mps_textbox.pack(padx=15, pady=5, fill="x")

    # Load current models
    current_models = self.app.config.active_models.mps_models
    if current_models:
        self.mps_textbox.insert("1.0", "\n".join(current_models))

    # Button row
    btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
    btn_frame.pack(padx=15, pady=5, fill="x")

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
    recent_frame.pack(padx=15, pady=(10, 15), fill="x")

    ctk.CTkLabel(
        recent_frame,
        text="Also prioritize recently active (last"
    ).pack(side="left")

    self.recent_days_entry = ctk.CTkEntry(
        recent_frame,
        width=50,
        justify="center"
    )
    self.recent_days_entry.pack(side="left", padx=5)
    self.recent_days_entry.insert(0, str(self.app.config.active_models.recent_days))

    ctk.CTkLabel(recent_frame, text="days)").pack(side="left")

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

    self.app.config.save()
    self.mps_count_label.configure(text=f"{len(unique_models)} models configured")

    messagebox.showinfo("Saved", f"MPS model list saved ({len(unique_models)} models)")

def _clear_mps_models(self):
    """Clear all MPS models."""
    self.mps_textbox.delete("1.0", "end")
    self.mps_count_label.configure(text="0 models configured")
```

---

### Step 4: Update Model Dropdowns in All Pages

Update each page to use the prioritized model list.

#### 4a. Analyze Page

**File:** `src/laser_trim_analyzer/gui/pages/analyze.py`

Find `_update_filters()` method (~line 576). Replace:
```python
models = db.get_models_list()
```

With:
```python
config = get_config()
if config.active_models.prioritize_active:
    models = db.get_models_list_prioritized_simple(
        mps_models=config.active_models.mps_models,
        recent_days=config.active_models.recent_days
    )
else:
    models = db.get_models_list()
```

Add import at top:
```python
from laser_trim_analyzer.config import get_config
```

#### 4b. Trends Page

**File:** `src/laser_trim_analyzer/gui/pages/trends.py`

Find model dropdown update (~line 964). Apply same pattern.

#### 4c. Compare Page

**File:** `src/laser_trim_analyzer/gui/pages/compare.py`

Find model filter update. Apply same pattern.
Note: May need to query FinalTestResult table instead.

#### 4d. Export Page

**File:** `src/laser_trim_analyzer/gui/pages/export.py`

Find `_update_model_dropdown()` (~line 261). Apply same pattern.

---

### Step 5: Handle Selection with Status Suffix

When a model is selected that has " (inactive)" suffix, strip it before querying:

```python
def _get_clean_model(self, model_value: str) -> str:
    """Remove status suffix from model value."""
    if model_value.endswith(" (inactive)"):
        return model_value[:-11]  # Remove " (inactive)"
    return model_value
```

Apply in all filter callbacks:
```python
model = self._get_clean_model(self.model_filter.get())
if model and model != "All Models":
    query = query.filter(DBAnalysisResult.model == model)
```

---

## Files to Modify Summary

| File | Changes |
|------|---------|
| `config.py` | Add `ActiveModelsConfig` dataclass, update `AppConfig` |
| `database/manager.py` | Add `get_models_list_prioritized()` and `_simple()` methods |
| `gui/pages/settings.py` | Add Active Models section with text area |
| `gui/pages/analyze.py` | Use prioritized list, handle suffix |
| `gui/pages/trends.py` | Use prioritized list, handle suffix |
| `gui/pages/compare.py` | Use prioritized list, handle suffix |
| `gui/pages/export.py` | Use prioritized list, handle suffix |

---

## Config File Format

After implementation, `./data/config.yaml` will include:

```yaml
active_models:
  mps_models:
    - "8340-1"
    - "6828"
    - "2475"
    - "1091701"
  recent_days: 90
  prioritize_active: true
```

---

## Testing Checklist

- [ ] Add 5+ models to MPS list in Settings
- [ ] Verify models appear first in Analyze page dropdown
- [ ] Verify models appear first in Trends page dropdown
- [ ] Verify models appear first in Compare page dropdown
- [ ] Verify models appear first in Export page dropdown
- [ ] Verify inactive models show "(inactive)" suffix
- [ ] Verify selecting inactive model still works (filters correctly)
- [ ] Verify config persists after restart
- [ ] Verify Clear All button works
- [ ] Verify recent_days setting works

---

## Future Enhancements

1. **Visual separator** - Add divider line between MPS/Active/Inactive groups
2. **Star indicator** - Show ★ next to MPS models instead of relying on position
3. **Import from file** - Button to load models from CSV/text file
4. **Sync with external** - Auto-sync with MPS system if API available
5. **Per-page toggle** - Option to hide inactive models entirely on some pages
