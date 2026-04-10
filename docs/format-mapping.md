# File Format Mapping — All Three Systems

**Generated:** 2026-04-10
**Source:** Sample_Base_2026-04-10 (1,276 valid files across LTS, DLTS, Test Station)

---

## System A (DLTS) — SEC1 TRK Sheets

### Data Sheet Layout (SEC1 TRK1 0 / SEC1 TRK1 N TRM1)

Row 0 = headers. Data starts at row 1.

| Col | Index | Header | Content |
|-----|-------|--------|---------|
| D | 3 | Meas Volts | Measured voltage |
| E | 4 | Index | Point index (0, 1, 2...) |
| F | 5 | Theory Volts | Theoretical voltage |
| G | 6 | Lin Error | Linearity error (meas - theory) |
| H | 7 | Position | Angular position (degrees) |
| I | 8 | Upper Lin Lim | Upper spec limit |
| J | 9 | Lower Lin Lim | Lower spec limit |
| K | 10 | pass/fail? | 42 = pass, error value = fail |
| L | 11 | trim target | Target voltage for trim |
| O | 14 | Zero-Start | 0-based travel distance |

### Metadata (Rows 0-9, Cols A-B in data sheets)

| Row | Col A label | Col B value | Notes |
|-----|-------------|-------------|-------|
| 0 | model# | Model number | |
| 1 | shop# | Serial | Can be numeric or string (e.g., "2B") |
| 5 | date | Excel serial date | |
| 7 | **length value** | **Nominal electrical angle** | Design spec (e.g., 240.0 degrees) |
| 8 | **meas value** | **Measured electrical angle** | Actual measured (e.g., 242.364) |
| 9 | **resistance** | **Resistance (Ohms)** | Different on untrimmed vs trimmed sheets |

### Track Parameters Sheet (angle tolerance)

| Row | Label | Value |
|-----|-------|-------|
| 6 | Length Minimum | Min acceptable angle |
| 7 | Length Theoretical | Nominal angle |
| 8 | Length Maximum | Max acceptable angle |

### Current Constants vs Reality

| Constant | Current | Reads | Should Be |
|----------|---------|-------|-----------|
| `measured_electrical_angle` | L1 | "trim target" (header text!) | **B9** (row 8 = "meas value") |
| `unit_length` | B26 | "Calc. Angle" (missing on pre-2020 files) | **B8** (row 7 = nominal angle) |
| `untrimmed_resistance` | B10 | resistance | Correct |
| `trimmed_resistance` | B10 | resistance | Correct |

### Multi-Track

TRK1 + TRK2 sheets. 22 models have 2 tracks. No models have 2 sections.

---

## System B (LTS) — test / Lin Error / Trim N Sheets

### Data Column Layout (same for test, Lin Error, and Trim N)

No header row. Row 0 = first data point.

| Col | Index | Content |
|-----|-------|---------|
| A | 0 | Measured voltage |
| B | 1 | Point index (0, 1, 2...) |
| C | 2 | Theoretical voltage |
| D | 3 | Linearity error |
| E | 4 | Position (degrees or inches) |
| F | 5 | Upper spec limit |
| G | 6 | Lower spec limit |
| H | 7 | Multiplier (always 1.0) |
| I | 8 | Cumulative travel position (empty on some old models like 6607) |
| R | 17 | **Resistance** (row 0 only) — untrimmed in "test", trimmed in "Lin Error" |

### Electrical Angle Location

| Location | Cell | Content | Always Present? |
|----------|------|---------|-----------------|
| test sheet, K1 | row 0, col 10 | **Measured electrical angle** | Yes |
| Lin Error, L1 | row 0, col 11 | **Theoretical electrical angle** | Yes |
| Model Params, row 2 | row 2, col 0 | Pot angle (total travel, negative) | Yes |
| Model Params, row 5 | row 5, col 0 | Min elec angle (tolerance low) | Newer templates only |
| Model Params, row 6 | row 6, col 0 | Max elec angle (tolerance high) | Newer templates only |

### Row 0 Metadata (cols N-O, newer templates only)

| Row | Col N value | Col O label |
|-----|-------------|-------------|
| 0 | degrees_between_points | "Degrees you want between points" |
| 2 | theoretical_angle | "Theoretical angle" |
| 3 | measured_angle | "Measured angle" |

### Current Constants vs Reality

| Constant | Current | Reads | Should Be |
|----------|---------|-------|-----------|
| `measured_electrical_angle` | B9 | Point index 8 (data value!) | **test sheet K1** (row 0, col 10) |
| `unit_length` | K1 | NaN (empty on Lin Error) | **Lin Error L1** (row 0, col 11) = theoretical angle |
| `untrimmed_resistance` | R1 | Correct (test sheet) | Correct |
| `trimmed_resistance` | R1 | Correct (Lin Error sheet) | Correct |

### Bowtie Spec Models

| Model | Pattern |
|-------|---------|
| 8232-1 | Linear taper: 0.0055 (center) to 0.0169 (ends) |
| 6607 | Step function: 0.2, 0.05, 0.025, 0.01 |
| 6126 | Two zones: 0.15 (ends), 0.05 (center) |

### Format Variations

| Feature | Old (1844205, 6607, 8232-1) | New (8340-1, 6952, 8895, 8887) |
|---------|----------------------------|-------------------------------|
| N/O metadata block | Missing or partial | Always present |
| Col I travel | Sometimes empty (6607) | Always populated |
| Model Params angle tolerance | Not present | Rows 5-6 |
| Trim Parameters | 12 or 26 rows | 12 rows |

---

## Final Test (Test Station) — Sheet1

### Data Columns (A-I) — Consistent across ALL models

Row 0 = first data point (NO header row).

| Col | Index | Content |
|-----|-------|---------|
| A | 0 | Measured voltage |
| B | 1 | Sample index |
| C | 2 | Theoretical voltage |
| D | 3 | Error (meas - theory, WITH compensation applied) |
| E | 4 | Position index |
| F | 5 | Error (duplicate of D, empty on some models) |
| G | 6 | Upper spec limit (can vary per row = bowtie) |
| H | 7 | Lower spec limit |
| I | 8 | Pass/fail flag |

### Metadata Cells (CONSISTENT across all models)

| Cell | Row,Col | Content | Notes |
|------|---------|---------|-------|
| L1 | 0,11 | Header text | "TEST RESULTS DATA FOR MODEL {model} SERIAL/SHOP NO:" |
| M1 | 0,12 | Serial number | |
| N1 | 0,13 | Test datetime | |
| L2 | 1,11 | Linearity result | "PASSED" or "FAILED" |
| N2 | 1,13 | Fail point count | |
| L3 | 2,11 | EA test result | "PASSED" or "FAILED" |
| L4 | 3,11 | Model number | |
| **M4** | **3,12** | **Compensation** | **Always present. The offset applied by test station.** |
| **M5** | **4,12** | **Second compensation** | Often 0, sometimes significant (7458-1: -11.08) |
| L11 | 10,11 | Pot angle | Total sweep (negative for CW) |
| L12 | 11,11 | Sample count | |
| L15 | 14,11 | Data row count | |
| **L19** | **18,11** | **Linearity type** | **"Absolute" (most) or "Independent" (8508)** |
| **M20** | **19,12** | **Nominal electrical angle** | Design spec value |
| L21 | 20,11 | Stop angle | |
| L35 | 34,11 | Pot type | "Linear" or "Rotary" |

### Data Table Sheet (EA tolerance)

| Row | Col B | Col C | Col D | Col E |
|-----|-------|-------|-------|-------|
| 0 | TEST | THEORY | TOLERANCE | MEASUREMENT |
| 2 | Resistance | Nominal | Tolerance | Measured |
| 3 | Linearity | -- | Spec | -- |
| 5 | **Electrical Angle** | -- | **EA tolerance** | -- |
| 6 | Output Smoothness | -- | Spec | Value |

---

## Output Smoothness — Test Data Sheet

### Consistent format across all models (.xlsx)

| Row | Col A | Col B | Col C |
|-----|-------|-------|-------|
| 0 | Model Parameters | Model Number | {model} |
| 1 | -- | Shop Number | {serial} |
| 2 | -- | Pot Type | Rotary / Linear |
| 3 | -- | Gear Ratio | {ratio} |
| 4 | Test Parameters | Input Voltage (V) | 10 |
| 5 | -- | Deviation Spec (%) | {spec} |
| 6 | -- | Electrical Travel | {travel} (deg or in) |
| 7 | -- | Sample Rate (Hz) | 1000 |

| Col | Index | Content |
|-----|-------|---------|
| D | 3 | Time (s) |
| E | 4 | Filtered Volts (V) |
| G | 6 | Max Deviation (V) — row 1 only |
| H | 7 | Max Spec Deviation (V) — row 1 only |
| I | 8 | Result — row 1 only (PASSED/FAILED) |

---

## Summary of Bugs Found

1. **System A `measured_electrical_angle = "L1"`** reads header text "trim target", not the angle. Should be **B9**.
2. **System A `unit_length = "B26"`** reads "Calc. Angle" which is missing on pre-2020 files. Should be **B8** (nominal angle).
3. **System B `measured_electrical_angle = "B9"`** reads point index 8 (data value), not the angle. Should read **test sheet K1**.
4. **System B `unit_length = "K1"`** reads NaN on Lin Error sheet. Should be **Lin Error L1** (theoretical angle).
