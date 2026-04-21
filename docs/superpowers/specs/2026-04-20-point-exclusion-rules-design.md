# Point Exclusion Rules — Design Spec

**Date:** 2026-04-20
**Status:** Approved

## Problem

Some models have data points (typically first/last rows in the trim spreadsheet) that should not count toward linearity pass/fail. These are mechanical artifacts (wiper settling at endpoints) that always fail but don't represent real quality issues. Currently there's no way to tell the system to ignore them.

## Requirements

- Per-model list of point indices to exclude from linearity fail-point counting
- Supports individual indices and ranges (endpoints now, arbitrary middle points later)
- Excluded points are still displayed on charts (grayed out) — not deleted from raw data
- Analysis metrics note how many points were excluded
- Existing models without exclusion rules are unaffected

## Storage

Add `exclude_points` column (TEXT, nullable) to `model_specs` table.

Stores JSON. Two supported formats:

```json
{"exclude": [0, 1, 49, 50]}
{"exclude": [[0, 2], [48, 50]]}
```

Individual integers and `[start, end]` inclusive ranges can be mixed:
```json
{"exclude": [0, [3, 5], 49]}
```

Empty/NULL means no exclusions.

## Parsing

Human-friendly input format in the UI: `0-2, 48-50` or `0, 1, 49, 50`

Parse rules:
- `N` → single index N
- `N-M` → inclusive range [N, M]
- Comma-separated, whitespace-tolerant
- Stored as JSON on save, displayed as human-friendly string on load

## Analyzer Changes

### `_count_fail_points(errors, upper_limits, lower_limits, exclude_indices=None)`

Add optional `exclude_indices: set[int]` parameter. When checking each point:
- If index is in `exclude_indices`, skip it (don't count as fail, don't count as checked)
- The point's error is still computed and stored in data arrays

### `_calculate_linearity(positions, errors, upper_limits, lower_limits, ..., exclude_indices=None)`

Pass `exclude_indices` through to `_count_fail_points`. The `linearity_pass = fail_points == 0` check uses the filtered count.

### `_calculate_failure_margins(shifted_errors, upper_limits, lower_limits, exclude_indices=None)`

Skip excluded indices in margin calculations too — they shouldn't inflate max_violation or skew margin_to_spec.

### `analyze_track(track_data, ...)`

Look up the model's exclusion rule from the `exclude_points` field passed via track_data. Convert the JSON format into a `set[int]` and pass down.

## Processor Changes

### `processor.py`

When building track_data dict before calling analyzer, look up the model's spec and include `exclude_indices` if defined. The spec lookup already happens for linearity_type/angle_spec — add exclude_points to the same path.

## Database Changes

### `database/models.py`

Add to ModelSpec:
```python
exclude_points = Column(Text)  # JSON: {"exclude": [0, 1, [48, 50]]}
```

### `database/manager.py`

- Migration: `ALTER TABLE model_specs ADD COLUMN exclude_points TEXT`
- `get_model_spec()` already returns all columns — no query change needed
- `save_model_spec()` already saves all fields — no change needed

## GUI Changes

### Specs Page (`gui/pages/specs.py`)

Add "Exclude Points" field to the edit panel:
- Label: "Exclude Points"
- Widget: CTkEntry with placeholder "e.g. 0-2, 48-50"
- On save: parse human-friendly string → JSON, store in exclude_points column
- On load: parse JSON → human-friendly string for display

### Analyze Page (`gui/pages/analyze.py`)

When rendering the error chart for a track:
- Look up the current result's model spec for exclude_points
- Excluded fail points rendered as gray hollow circles instead of red filled circles
- Add a note in the metrics panel: "N points excluded per model spec" (only if N > 0)

## Data Flow

```
User sets "0-2, 48-50" on Specs page
  → Parsed to {"exclude": [[0, 2], [48, 50]]}
  → Saved to model_specs.exclude_points

File processed:
  Processor looks up model spec
  → Passes exclude_indices={0, 1, 2, 48, 49, 50} to Analyzer
  → _count_fail_points skips those indices
  → linearity_pass based on remaining points only
  → Saved to DB as normal (raw data unchanged)

Analyze page displays result:
  → Chart shows all points
  → Excluded fail points shown as gray markers
  → Metrics note: "6 points excluded"
```

## What Doesn't Change

- Raw data arrays (position_data, error_data, upper/lower limits) are never modified
- Export includes all points with exclusion noted
- TrackResult DB schema unchanged — no new columns on track_results
- Final Test parser unaffected (exclusions are trim-analysis only)
- Existing records without exclusion rules work exactly as before
