# Laser Trim Analyzer v5

Production quality analysis platform for potentiometer laser trim data. Built for AS9100-certified aerospace/defense manufacturing environments.

## Features

### Core Analysis
- **Multi-format Parsing**: System A, System B, and Final Test file support
- **Linearity-First Quality**: Zero-tolerance linearity spec enforcement with margin analysis
- **Sigma Gradient Analysis**: Pass/fail with ML-optimized thresholds
- **Cpk/Ppk Process Capability**: Statistical process capability metrics per model
- **Output Smoothness**: Smoothness test parsing and visualization
- **Trim Effectiveness**: Measures resistance change and improvement from laser trimming

### Operational Intelligence
- **Quality Health Dashboard**: At-a-glance quality status with actionable recommendations
- **Model Scorecard**: Single-page model health summary (pass rate, Cpk, drift, specs)
- **Near-Miss Detection**: Identifies parts passing but close to spec limits
- **Cost Impact Analysis**: Links quality metrics to scrap/rework costs
- **Linearity Prioritization**: Ranks models by quality impact for focused improvement

### Data & Analytics
- **SQLite Database**: Historical tracking with SQLAlchemy 2.0 ORM
- **Trend Analysis**: Model-level trends, alerts, and SPC control charts
- **Model Comparison**: Side-by-side model performance comparison
- **Excel Export**: Professional reports with executive summary option

### Machine Learning
- **Per-Model ML**: Threshold optimization trained on each model's data
- **Drift Detection**: CUSUM + EWMA algorithms for process drift alerting
- **Statistical Profiling**: Per-model statistical profiles for anomaly detection
- **Failure Prediction**: ML-based pass/fail prediction using Final Test ground truth

### Engineering
- **Model Specs Management**: View/edit engineering specifications per model
- **Final Test Matching**: Fuzzy matching links trim data to post-assembly test results

## GUI Pages

| Page | Purpose |
|------|---------|
| **Dashboard** | Linearity quality overview, Pareto charts, cost impact |
| **Process** | Scan and process trim data files into the database |
| **Analyze** | Deep-dive single-result analysis with charts |
| **Compare** | Side-by-side model comparison |
| **Trends** | Historical trends, SPC charts, linearity prioritization |
| **Quality Health** | Operational quality status with recommendations |
| **Scorecard** | Single-model health summary |
| **Smoothness** | Output smoothness test results and charts |
| **Specs** | Model engineering specifications management |
| **Settings** | Configuration, ML training, database maintenance |

## Quick Start

### Running from Source

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -e .

# Run the application
python src/__main__.py
```

### Deployed Executable (Windows)

```bash
deploy.bat
```

This creates a versioned folder (e.g., `LaserTrimAnalyzer-v5.0.0-YYYYMMDD/`) ready for distribution.

## Project Structure

```
src/laser_trim_analyzer/
├── __main__.py              # Entry point
├── app.py                   # Main application
├── config.py                # Configuration
├── core/
│   ├── parser.py            # Trim file parser (System A & B)
│   ├── final_test_parser.py # Final Test file parser
│   ├── smoothness_parser.py # Output Smoothness parser
│   ├── processor.py         # Analysis processor
│   ├── analyzer.py          # Sigma/linearity/trim effectiveness analysis
│   ├── cpk.py               # Cpk/Ppk process capability
│   └── models.py            # Data models
├── database/
│   ├── manager.py           # Database operations & queries
│   └── models.py            # SQLAlchemy ORM models
├── gui/
│   ├── app.py               # GUI application
│   ├── pages/               # All GUI pages (see table above)
│   └── widgets/             # Chart widget, scrollable combobox
├── ml/
│   ├── manager.py           # ML orchestration
│   ├── predictor.py         # Per-model failure prediction
│   ├── threshold_optimizer.py # Per-model threshold optimization
│   ├── drift_detector.py    # CUSUM + EWMA drift detection
│   └── profiler.py          # Per-model statistical profiling
├── export/
│   └── excel.py             # Excel export with executive summary
└── utils/
    ├── constants.py         # Application constants
    └── threads.py           # Thread management
```

## Configuration

- **Database**: `./data/analysis.db` (relative to application)
- **User settings**: `~/.laser_trim_analyzer/config.yaml`

No external config files required — self-contained for easy deployment.

## Requirements

- Python 3.10+
- See `pyproject.toml` for full dependency list

## Version History

- **v5.0.0** (2026-04): Spec-aware analysis engine, model reference system, output smoothness, Cpk/Ppk, scorecard page, specs management
- **v4.0.0** (2026-03): Linearity-first quality prioritization, operational analytics, cost impact, quality health dashboard, near-miss detection
- **v3.0.0** (2025-12): Complete redesign — simplified architecture (110 files to ~30), self-contained config, per-model ML, dark theme GUI
- **v2.2.9** (2025-08): Database configuration and analysis fixes

## License

MIT License - see [LICENSE](LICENSE) file.
