# Appendix — Raw feature & chart catalogs (one line per feature, nothing merged)

Companion to `V6_DESIGN_REVIEW.md`. These are the exhaustive inventories the
review matrix was built from, with code references. Where the matrix merged
navigation/pagination rows for readability, this file keeps every row.

## V5 feature catalog (164 features, 11 pages)

Format: `page | feature | what it does | code ref` — refs relative to
`src/laser_trim_analyzer/gui/pages/` unless noted.

### Dashboard (21)
```
Dashboard | Export Summary button | Exports executive quality summary to Excel (bg thread) | dashboard.py:88/1144
Dashboard | Refresh button | Reloads all dashboard stats from DB | dashboard.py:97
Dashboard | Element Type filter | Combobox filtering all stats by element type | dashboard.py:119
Dashboard | Product Class filter | Combobox filtering all stats by product class | dashboard.py:130
Dashboard | Attention cards (x4) | Clickable severity cards for models needing action → scorecard | dashboard.py:142/1230
Dashboard | Linearity Quality card | 90-day linearity pass rate + sigma/overall + sparkline | dashboard.py:156/703
Dashboard | Database Total card | All-time file count, pass/warn/fail, model count | dashboard.py:167/712
Dashboard | Sigma Process Health card | Sigma pass rate as leading indicator | dashboard.py:178/728
Dashboard | System A/B comparison | Per-system linearity/sigma/file counts | dashboard.py:199/784
Dashboard | Final Test info line | FT linearity pass %, test count, link-to-trim rate | dashboard.py:211/817
Dashboard | Prediction Accuracy line | Trim-vs-FT agreement, escape %, overkill % | dashboard.py:217/833
Dashboard | Near-Miss line | % of failures that are near-miss (1-3 fail pts) | dashboard.py:223/851
Dashboard | Cost Impact line | Estimated scrap cost from model pricing config | dashboard.py:229/869
Dashboard | Pass-rate trend P-chart | Linearity pass-rate trend chart | dashboard.py:239/974
Dashboard | Trend range dropdown | Switches chart window 30/90/180/365/All | dashboard.py:267/982
Dashboard | Recent Alerts textbox | Top-5 deduplicated alerts w/ severity icons | dashboard.py:312/905
Dashboard | Where to Focus panel | Top-5 priority models w/ pass%, fails, recommendation | dashboard.py:326/1057
Dashboard | Perf by Element Type | Color-coded pass rate per element type (90d) | dashboard.py:362/1190
Dashboard | Perf by Product Class | Color-coded pass rate per product class (90d) | dashboard.py:372/1214
Dashboard | Process Capability (Cpk) summary | Model counts by Cpk rating | dashboard.py:392/1320
Dashboard | Drift Alerts card (clickable) | Failure-mode counts + drift model list → Trends Drift | dashboard.py:402/1320
```

### Quality Health (5)
```
Quality Health | Active Models Only checkbox | Filters table/cards to active models | quality_health.py:80
Quality Health | Refresh button | Reloads per-model quality + trend + FT data | quality_health.py:89
Quality Health | Plant Quality banner | Overall status, declining/stable/improving counts, avg pass | quality_health.py:416
Quality Health | Models Needing Attention cards | Top-5: trim%, FT%, trend arrow, recommendation | quality_health.py:506/518
Quality Health | All Models Ranked table | Status·model·trim%·FT%·gap·trend·change·samples·action | quality_health.py:603/661
```

### Process Files (12)
```
Process | Incremental Mode checkbox | Only process files not already in DB | process.py:84
Process | Save to Database checkbox | Toggle saving results to DB | process.py:94
Process | Date filter entry | Process files from a modified-date onward | process.py:116
Process | Clear date filter (X) | Resets the date filter | process.py:105
Process | Select File(s) button | File dialog for individual .xls/.xlsx | process.py:135
Process | Select Folder button | Recursive background folder scan | process.py:142/312
Process | Start Processing button | Batch processing w/ progress (bg thread) | process.py:157/573
Process | Cancel button | Cancels processing or folder scan | process.py:168/861
Process | Export button | Exports batch results to Excel | process.py:179/871
Process | Selected Files list | Files grouped by folder | process.py:222/498
Process | Progress bar + ETA | Percent, files/sec, ETA | process.py:200/718
Process | Processing Results textbox | Live per-file status + summary | process.py:237/779
```

### Analyze Trim (20)
```
Analyze | Model filter combobox | Filter analyses by model | analyze.py:123
Analyze | Active Only checkbox | Hide inactive models | analyze.py:135
Analyze | Date From entry | Trim date start filter | analyze.py:148
Analyze | Date To entry | Trim date end filter | analyze.py:158
Analyze | Status filter dropdown | All/Pass/Warning/Fail | analyze.py:169
Analyze | Serial filter entry | Partial serial match | analyze.py:179
Analyze | Refresh button | Reload analyses | analyze.py:188
Analyze | Recent Analyses list | Paginated rows w/ status + anomaly + ML-risk badges | analyze.py:227/767
Analyze | Pagination prev/next | Page through results | analyze.py:235/251
Analyze | Status banner | Large PASS/FAIL/WARNING banner | analyze.py:270/888
Analyze | Re-analyze button | Reprocess original file, update record | analyze.py:289/1514
Analyze | View Scorecard button | Jump to model scorecard | analyze.py:301/1506
Analyze | Delete button | Delete selected record (confirm) | analyze.py:311/1606
Analyze | Export Chart button | 4-panel chart to PNG/PDF | analyze.py:323/1794
Analyze | Export Model button | All serials for model to Excel | analyze.py:333/1734
Analyze | Export SN button | Single result to Excel | analyze.py:343/1707
Analyze | Track selector | Per-track chart + Compare All Tracks | analyze.py:354/1424
Analyze | Chart tab | Error-vs-position / track comparison | analyze.py:367/948
Analyze | Metrics tab | Sigma/linearity/margins/anomaly/props | analyze.py:385/1079
Analyze | File Info tab | File meta, worst fail-zone, specs, optimization | analyze.py:392/1285
```

### Final Test / Compare (15)
```
Compare | Model filter combobox | Filter FT results by model | compare.py:192
Compare | Date From/To entries | Filter by file date | compare.py:204/213
Compare | Status filter dropdown | All/Pass/Fail | compare.py:225
Compare | Serial filter entry | Partial serial match | compare.py:236
Compare | Linked Only checkbox | Only FTs linked to trim | compare.py:245
Compare | Refresh button | Reload comparison pairs | compare.py:255
Compare | Re-match button | Re-run FT→trim matching | compare.py:268/1193
Compare | Fix Missing Tracks button | Re-parse files lacking track arrays | compare.py:279/1237
Compare | Export Chart button | Current comparison chart PNG/PDF | compare.py:290/1446
Compare | Export Selected button | Multi-page PDF of selected comparisons | compare.py:300/1489
Compare | FT Results list | Paginated rows w/ checkboxes + link status | compare.py:367/611
Compare | Select All/None | Bulk selection | compare.py:346/356
Compare | Pagination prev/next | Page through FT results | compare.py:374/386
Compare | Comparison Details panel | Model/serial/date/status/match confidence | compare.py:418/720
Compare | Trim-vs-FT overlay chart | Interactive overlay | compare.py:427/945
```

### Smoothness (10)
```
Smoothness | Model dropdown filter | Filter results by model | smoothness.py:50
Smoothness | Refresh button | Reload results + stats | smoothness.py:59
Smoothness | Export to Excel | One sheet per model | smoothness.py:62/493
Smoothness | Summary stats label | Total, pass %, link % | smoothness.py:65/662
Smoothness | All-models comparison table | Count·pass·avg-max·worst·spec·margin | smoothness.py:73/342
Smoothness | Single-model stats card | Pass/avg/worst/spec/margin tiles | smoothness.py:77/435
Smoothness | Results list (paginated) | Clickable rows w/ status dot | smoothness.py:96/213
Smoothness | Pagination prev/next | 20/page | smoothness.py:102/674
Smoothness | Detail info textbox | Model/serial/spec/max/link/match | smoothness.py:115/270
Smoothness | Smoothness chart | Smoothness-vs-position | smoothness.py:118/297
```

### Trends (32)
```
Trends | Trend-type segmented button | Standard / Drift / Trim Difficulty | trends.py:513/2753
Trends | Model dropdown | All Models vs specific | trends.py:529/1438
Trends | Active Only checkbox | Hide inactive | trends.py:541
Trends | Min samples entry | Filter models below N samples | trends.py:555
Trends | Min fail% entry | Filter below N% fail | trends.py:562
Trends | Trim Date dropdown | 30/90/365/All | trends.py:570/1535
Trends | Element filter | By element type | trends.py:581
Trends | Class filter | By product class | trends.py:594
Trends | Rolling Avg dropdown | 7/14/30/60-day (detail) | trends.py:606/1553
Trends | Refresh button | Reload | trends.py:615
Trends | Export PDF (summary) | Summary view PDF | trends.py:624/3893
Trends | Active Models Summary tiles | 9 stats | trends.py:717/2082
Trends | Focus This Week rows | Top-5 w/ recommendation | trends.py:778/806
Trends | Failure Severity chart | Fail-point-bucket bars | trends.py:877/897
Trends | Cost Impact chart | Scrap-cost bars top 15 | trends.py:941/957
Trends | ML Insights textbox | Trained counts, drift, attention list | trends.py:765/2427
Trends | ML View All Details dialog | ML status/drift/difficulty/alerts | trends.py:758/2530
Trends | Detail stat tiles | 9 per-model stats incl Cpk | trends.py:1153/2341
Trends | Sigma Gradient scatter | SPC scatter | trends.py:1176/1649
Trends | Chart Range dropdown | Zoom recent 7/14/30/60d | trends.py:1197/1565
Trends | Export PDF (detail) | Model detail PDF | trends.py:1206/3926
Trends | Linearity P-chart | P-chart + unit-yield overlay | trends.py:1224/1723
Trends | Sigma Distribution histogram | 20+ samples only | trends.py:1247/2395
Trends | Model Alerts textbox | Per-model alerts | trends.py:1282/2650
Trends | ML Recommendations textbox | Threshold/confidence/drift insights | trends.py:1297/2669
Trends | Drift model-filter dropdown | All vs single model | trends.py:2872/2920
Trends | ML/Process Drift toggle | Sub-view switch | trends.py:2882/2906
Trends | ML Drift table | Sortable, sparklines, row drill-in | trends.py:3382/3433
Trends | Process Drift metric tabs | Resistance/Angle/Trim Passes | trends.py:3613/3727
Trends | Process Drift table | Baseline/recent/Δ%/z/sparkline | trends.py:3572/3598
Trends | Single-model drift dashboard | Pills + 2×2 chart grid | trends.py:2937/2979
Trends | Trim Difficulty chart | Avg-trim-passes bars | trends.py:3740/3760
```

### Model Specs (9)
```
Specs | + Add Model button | New spec in edit panel | specs.py:65/447
Specs | Check Discrepancies button | File-parsed vs reference + badge | specs.py:72/543
Specs | Search entry | Live filter | specs.py:89/306
Specs | Pagination prev/next | 50/page | specs.py:101/321
Specs | Specs table | Clickable; incomplete highlighted red | specs.py:110/335
Specs | Edit panel (18 fields) | Element/class/linearity/resistance/angle/aliases/excludes/notes | specs.py:118/142
Specs | Save button | Persist w/ validation | specs.py:224/466
Specs | Delete button | Delete spec (confirm) | specs.py:230/521
Specs | Clear button | Clear panel | specs.py:236/455
```

### Export (11)
```
Export | Model filter combobox | Filter units by model | export.py:79
Export | Trim Date dropdown | Today/7/30/90/All | export.py:90
Export | Serial filter entry | Partial serial match | export.py:100
Export | Search button | Query exportable units | export.py:108/216
Export | Results count label | Result count | export.py:117
Export | Select All button | Check all | export.py:154/394
Export | Clear All button | Uncheck all | export.py:142/401
Export | Results list (checkboxes) | Multi-select rows | export.py:166/315
Export | Selected count label | Live count | export.py:176/408
Export | Export format radios | PNGs vs combined PDF | export.py:189/198
Export | Export Charts button | Render charts for selected units | export.py:207/442
```

### Scorecard (7)
```
Scorecard | Model selector | Pick model | scorecard.py:42
Scorecard | Refresh button | Reload | scorecard.py:50
Scorecard | Header section | Model + element/class/linearity type | scorecard.py:155
Scorecard | Key Metrics cards | Pass rate, units 90d, Cpk, avg error | scorecard.py:182
Scorecard | Specifications section | Linearity spec ± and type | scorecard.py:216
Scorecard | Process Capability section | Cpk/Ppk/Cp, mean/sigma, limits | scorecard.py:234
Scorecard | Drift Status section | Stable / drifting | scorecard.py:256
```

### Settings (22)
```
Settings | Change DB path | Pick SQLite location | settings.py:114/1152
Settings | DB info label | Connected status + count | settings.py:122/1383
Settings | Set export location | Default export folder | settings.py:154/1170
Settings | Batch/Turbo display | Read-only perf settings | settings.py:174
Settings | Enable ML checkbox | Toggle per-model ML | settings.py:197/1181
Settings | Train Models button | Train thresholds/predictors/drift | settings.py:206/1187
Settings | Apply to DB button | Write learned values to records | settings.py:216/1266
Settings | ML status + staleness | Trained counts, retrain-needed | settings.py:225/1315
Settings | Active Models (MPS) textbox | MPS list | settings.py:302
Settings | Save MPS button | Persist list + recent-days + cost-ratio | settings.py:318/428
Settings | Clear All MPS | Empty list | settings.py:326/462
Settings | Recent-days entry | Active-window days | settings.py:352
Settings | Import Pricing | Model→price from Excel/CSV | settings.py:383/467
Settings | Clear Pricing | Wipe pricing | settings.py:390/541
Settings | Cost ratio entry | Invested-cost fraction | settings.py:415
Settings | Import Model Specs | From reference Excel | settings.py:567/579
Settings | Scan Database | Flag dirty records | settings.py:647/806
Settings | Cleanup checkboxes (7) | non-MPS/date/suspect/unknown/error/no-tracks/misclassified-FT | settings.py:684-747
Settings | Preview cleanup | Count would-delete | settings.py:753/977
Settings | Delete Records | Execute cleanup (confirm) | settings.py:760/1029
Settings | Reset Skipped Files | Un-skip for reprocess | settings.py:785/872
Settings | Theme dropdown | Dark/Light/System | settings.py:1117/1376
```

Notes: Scorecard is reached via Dashboard attention cards / Analyze (no sidebar
entry). Dashboard's Pareto/confusion/escape-scatter charts are retired dead refs
(dashboard.py:320-322). Dead-but-present chart code in trends.py:
`_render_yield_trend` (3279) and the `_update_heatmap` cluster are unreachable.
ChartWidget methods with no live caller: plot_spc_control, plot_sample_scatter,
plot_line_chart, plot_pass_rate_bars, plot_trending_worse, plot_alert_summary,
plot_drift_chart, plot_pareto, plot_confusion_matrix, plot_heatmap,
plot_escape_scatter (chart.py).

## V6 feature catalog (5 pages + shell)

Format: `page | feature | what it does | code ref` — refs relative to
`src/laser_trim_analyzer/gui/v6/`.

### Shell
```
shell | Sidebar nav (5) | Dashboard/Triage/Process/Model/Settings + active stripe | sidebar.py:12,35
shell | Page header + actions | Title + right-aligned action area per page | page_base.py:78,90
shell | Model routing hint | Deep-link model+focus_metric across pages | app.py:57,68
shell | First-run auto-train gate | TrainingModal when data exists, untrained | app.py:88
shell | Startup drift catch-up | Advance detectors over new data | app.py:106
```

### Dashboard
```
dashboard | Window filter | 30d/90d/365d/All (default 90d) | dashboard_page.py:30
dashboard | Trim yield panel | Pass % + P/W/F/U/E counts + sparkline | dashboard_page.py:42; yield_panel.py:28
dashboard | Final-test yield panel | Same for FT | dashboard_page.py:44
dashboard | Company trend header+toggle | Weekly/Monthly period menu | dashboard_page.py:49,53
dashboard | Company trend chart | Pass-rate lines (company + A/B/C) + volume bars | company_trend_chart.py
dashboard | Lowest-yield models list | Ranked; min-5-units cap disclosed | worst_models_list.py:43
dashboard | Row click → Model | Routes model | dashboard_page.py:120
```

### Triage
> Note (2026-08-30): the card wall below (`Flagged cards zone` / `Model alert card`) was replaced by the FOCUS list — see `docs/superpowers/specs/2026-08-29-focus-spc-redesign-design.md`. Rows below kept as historical record, not rewritten.
```
triage | Scope toggle | Active / All models (default Active) | triage_page.py:33
triage | Flagged cards zone | "Needs attention (N)" scrollable card grid | flagged_cards_zone.py:17,30
triage | Model alert card | Tier color, alert type, worst metric, ±σ shift | model_alert_card.py:27
triage | Flagged empty state | "All models within tolerance" + last-processed | flagged_cards_zone.py:45
triage | Browse zone | All-models list + search | browse_zone.py:12
triage | Browse search + cap | Substring filter; 200 cap disclosed | browse_zone.py:22,50
triage | Browse row | Tier dot + model + last date; click → Model | browse_zone.py:57
triage | Card click → Model+focus | Deep-link with triggering metric | triage_page.py:86
```

### Process
```
process | Folder picker | Choose batch folder | folder_picker.py
process | Incremental checkbox | Skip processed (default on) | process_page.py:41
process | Start processing | Worker batch run | process_page.py:46
process | Progress section | Status + bar + 5 counters + failures | process_progress_section.py:12
process | Post-batch drift advance | Advance drift state for batch models | process_page.py:~130
process | Go to Triage | On completion | process_page.py:53,143
```

### Model
```
model | Model selector | Known models combobox | model_page.py:64,140
model | Window menu | 30d/90d/365d/All | model_page.py:70
model | Copy summary | Paste-ready drift summary (worker) | model_page.py:76,351
model | Export evidence pack | 3-sheet xlsx, full history | model_page.py:79,374
model | Empty state | "Pick a model…" | model_page.py:85
model | Metric pill row (9) | Tier-colored pills + σ shift; click refocuses | metric_pill_row.py:10
model | Focus chart | Selected metric SPC/trend | focus_chart.py
model | Predictor panel | Collapsible diagnostic (lazy, worker) | predictor_panel.py:26
model/Drift | Metrics table | 9 metrics: tier/alert/baseline/recent/shift | drift_metrics_tab.py:9,27
model/Smoothness | Trend + list | max-dev chart + serial·date·dev/spec·verdict | smoothness_tab.py
model/Units | Table + search + export | Recent units, serial search (500 cap), sortable | units_tab.py
model/Units | Row → unit modal | Per-unit chart | model_page.py:348
model/TrimFT | KPIs | Pass rates, escapes, overkills, agreement, serials | trim_ft_tab.py:21
model/History | Measure dropdown + 2 panels | Value scatter + monthly pass-rate | history_tab.py:52,74
model/UnitModal | Error-vs-position chart | Sweep + spec band + fail markers | unit_chart_modal.py:94,109
model/UnitModal | Save chart… | PNG/PDF/SVG | unit_chart_modal.py:57,117
model/UnitModal | No-sweep placeholder | Explains Warning-no-data units | unit_chart_modal.py:97
model/UnitModal | Multi-track note | "track 1 of N" | unit_chart_modal.py:107
```

### Settings (6 sections)
```
settings/AlertThresholds | Sensitivity slider | loose/standard/tight/strict | sensitivity_slider.py:8
settings/AlertThresholds | Live flag preview | Would-flag counts per tier | alert_thresholds.py:26,30
settings/AlertThresholds | Metric legend | Watched metrics list | alert_thresholds.py:56
settings/AlertThresholds | Save preset | Persist + apply thresholds (worker) | alert_thresholds.py:60,63
settings/ActiveModels | Recency window entry | Auto-active days | active_models.py:33
settings/ActiveModels | Pinned models textbox | MPS pins | active_models.py:41
settings/ActiveModels | Status label | Active counts + unknown-pin warning | active_models.py:59,77
settings/ActiveModels | Save | Persist pins + days | active_models.py:85
settings/PerModelSpecs | Model combobox | Load spec | per_model_specs.py:89,159
settings/PerModelSpecs | Import spec sheet | Bulk merge from master Excel | per_model_specs.py:100,130
settings/PerModelSpecs | 4 spec fields | Linearity text/%, exclude pts trim/FT | per_model_specs.py:140
settings/PerModelSpecs | Save / Delete | Persist/remove (worker) | per_model_specs.py:239,241
settings/MLTraining | Retrain drift | TrainingModal w/ preset | ml_training.py:24,44
settings/MLTraining | Retrain per-model ML | thresholds+predictor+profiler | ml_training.py:28,47
settings/Pricing | Cost ratio + recent days | Entries | pricing.py:60,61
settings/Pricing | Count label + Save/Import/Clear | Pricing management | pricing.py:63,138-144
settings/Cleanup | Scan database | Dirty-record counts | database_cleanup.py:80,91
settings/Cleanup | 6 category checkboxes + before-date | Purge filters | database_cleanup.py:16,97,104
settings/Cleanup | Preview / Clear selected | Count then delete (confirm) | database_cleanup.py:122-149
settings/Cleanup | Reset skipped files | Un-skip (count + confirm) | database_cleanup.py:153,176
```

### Evidence export
```
export | "Drift evidence" sheet | Per-metric tier/alert/baseline/recent/Shift(σ)/magnitude | evidence.py:104
export | "Unit history" sheet | Full record, 16 cols per unit/track | evidence.py:120
export | "Monthly summary" sheet | Units, P/W/F, pass %, 6 metric means per month | evidence.py:146
export | Copy summary text | Per-metric baseline-vs-recent + σ-shift + tier | evidence.py:60
```

## V6 chart catalog (5 charts, all matplotlib — no tk.Canvas anywhere)

```
FocusChart | Model focus + Smoothness tab | control/trend chart, SPC Rule-1 overlays, robust y-window, off-scale annotation | set_series(metric,dates,values,baseline…) | focus_chart.py:35
CompanyTrendChart | Dashboard | multi-line + twin-axis volume bars | set_data(trend,period) | company_trend_chart.py:49
MiniTrendChart | YieldPanel ×2 | sparkline, ≤48 buckets, fixed 0-100 | set_points(list) | mini_trend_chart.py:37
HistoryTab | Model History | 2 subplots: value scatter + monthly pass-rate | set_data(dict) → _render | history_tab.py:52,74
UnitChart | UnitChartModal (shared v5 ChartWidget) | error-vs-position + spec band + fail markers | plot_error_vs_position(…) | gui/widgets/chart.py:211
```

## V5 chart catalog (21 entries)

See review §(c) rows 7–17 for verdicts; full technical inventory with
extractability notes:

```
P-chart (dashboard + trends detail) | matplotlib | ChartWidget.plot_pchart chart.py:1724 | extractable
Text sparkline | dashboard | unicode blocks dashboard.py:23/691 | n/a
Error-vs-position | Analyze | chart.py:211 | extractable
Track comparison | Analyze multi-track | chart.py:817/863 | extractable
4-panel export chart | Analyze | analyze.py:1929-2189 | extractable
Trim-vs-FT overlay | Compare | compare.py:945 | semi-tangled
4-panel comparison export | Compare | compare.py:1746-2006 | extractable
Smoothness-vs-position | Smoothness | chart.py:533 | extractable
Failure Severity bars | Trends | trends.py:897 | tangled
Cost Impact bars | Trends | trends.py:957 | tangled
Sigma scatter | Trends | chart.py:1370 | extractable
Linearity P-chart + yield overlay | Trends | chart.py:1724/1854 | extractable
Sigma histogram | Trends | chart.py:662 | extractable
ML Drift sparklines | Trends table | tk.Canvas trends.py:296/321 | NOT headless-renderable
Process Drift sparklines | Trends table | tk.Canvas trends.py:3684 | NOT headless-renderable
Sigma drift panel | Trends single-model | trends.py fn 224 | extractable (page plumbing)
Smoothed panels ×3 | Trends single-model | trends.py fn 174 | extractable (page plumbing)
Trim Difficulty bars | Trends | trends.py:3760 | tangled
Export 4-panel (page) | Export | export.py:703-931 | extractable
Export multi-page PDF | Export | export.py:623 | extractable
Trends PDFs | export/trends_pdf.py | separate module | out of GUI scope
```
