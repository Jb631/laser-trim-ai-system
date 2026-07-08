# V6 Design Review — 2026-07-07

Laser Trim Analyzer: production quality-analysis platform for potentiometer laser-trim data (QA manager + engineers/MEs; drives drift triage, unit investigation, and process-direction decisions in an AS9100 shop).

**Version mapping (verified, not inferred):** V5 GUI = `src/laser_trim_analyzer/gui/pages/` (10 pages) + `gui/widgets/`; V6 GUI = `src/laser_trim_analyzer/gui/v6/` (5 pages). Both are working code sharing one core (`core/`, `database/`, `ml/`, `export/`) and one SQLite DB. V6 launches with `--v6` (branch `V6`; includes the uncommitted 2026-07-06/07 work: ui_dispatch, Company Trends, System C, evidence pack v2).

**Method.** Both apps were launched on the real machine against the real 3.2 GB database and every page was screen-captured (20 screenshots). Every matplotlib chart was ALSO rendered headlessly through the app's own plotting code (Tk embedding stubbed, chart logic untouched) — 24 PNGs in `charts/`. Every rendered number was recomputed independently with raw SQL + pandas (no app code): **22 of 24 checks matched exactly**; the 2 mismatches are a v5 Smoothness-table definition question, not math errors (see audit table). Edge cases (empty / single point / same-day retrims / None values / zero-σ constant / partial period) were executed against the real chart code — zero exceptions. The only visuals not headless-renderable are v5's two tk.Canvas sparkline columns (audited via live capture).

---

## (a) BLUF

**Keep v6 — modify in place; do not hybrid v5 pages back in and do not start v7.** The v6 architecture demonstrably absorbs new features at low cost (Company Trends, a third laser system, and a full-history export were all added in the last 48 hours as pure additions — no page rewrites), and every chart's math verified correct against raw SQL. What must change before you trust it for decisions is not design but a **trust layer**: the live system's top triage alert (8887, +16.3σ, verified arithmetically correct) is actually one corrupt data point on an MPS-pinned model idle for 321 days, shown with an empty default chart — fix staleness disclosure, outlier-robust "recent," the WARNING-status backfill, and window anchoring, and v6 is the durable platform.

---

## (b) Feature matrix

Status: kept = equivalent exists in v6 · changed = exists, different behavior · dropped = no v6 equivalent · new = v6-only. Assessment from live use + code + recompute.

### V5 Dashboard (11 pages → v6 Dashboard/Triage)

| # | Feature (v5) | V5 behavior | V6 behavior | Status | Assessment |
|---|---|---|---|---|---|
| 1 | Export Summary (Excel) | Executive workbook | none (per-model evidence pack instead) | dropped | Deliberate (mission cut). Works in v5. |
| 2 | Refresh button | manual reload | reload on page show + window change | changed | Improvement (no stale panel risk). |
| 3 | Element Type filter | filters all dashboard stats | none | dropped | Mission cut; company trend has per-system split instead. |
| 4 | Product Class filter | same | none | dropped | Same. |
| 5 | Attention cards (4) | clickable → scorecard | worst-models list + Triage cards → Model | changed | Improvement: routed to a richer page. |
| 6 | Linearity Quality card | 90d pass % + text sparkline | Trim yield panel + real sparkline | changed | Improvement (real chart, counts disclosed). |
| 7 | Database Total card | all-time counts | window-scoped counts (All available) | changed | Equivalent. |
| 8 | Sigma Process Health card | sigma pass % | pill/table per model; not company-level | changed | Minor gap: no company sigma headline. Acceptable — sigma is no longer the watched headline (D-SIGMA decision). |
| 9 | System A/B comparison | text lines (90d) | Company trend per-system lines (A/B/C) | changed | Improvement: trend vs snapshot; verified exact (A 90.2%/648, B 83.4%/603). |
| 10 | Final Test info line | FT pass/link rate text | FT yield panel | changed | Improvement. |
| 11 | Prediction Accuracy line | agreement/escape/overkill text | Model page Trim-vs-FT tab (per model) | changed | Company-level number dropped; per-model is the actionable one. |
| 12 | Near-Miss line | % failures near-miss | none | dropped | Mission cut; near-miss data still in DB. |
| 13 | Cost Impact line | $ scrap estimate | none (pricing config retained) | dropped | Mission cut ("engineer/ME app"). Revisit only if leadership reporting returns. |
| 14 | Pass-rate P-chart + range | daily P-chart, control limits | Company trend (yield+volume, no control limits) | changed | See chart audit #7/#3. P-chart's binomial limits were statistically nice; v6 favors volume context. Acceptable. |
| 15 | Recent Alerts textbox | top-5 dedup alerts | Triage page (cards, tiered) | changed | Improvement — alerts became a page with evidence. |
| 16 | Where to Focus panel | top-5 priority + recommendation text | worst-models list + Triage ordering | changed | Recommendation strings dropped; tier+σ-shift is more honest than canned advice. |
| 17 | Perf by Element/Class | color-coded category rates | none | dropped | Mission cut. |
| 18 | Cpk summary | counts by Cpk rating | none | dropped | See Deferred — Cpk queries still in manager. |
| 19 | Drift Alerts card | count + model names (old detector) | Triage (new multi-metric detector) | changed | Improvement, but note: v5 card said 34, v5 drift table said 6, v6 Triage says 39 — three detectors/filters coexist today. Retiring v5 kills the confusion. |

### V5 Quality Health / Scorecard

| # | Feature | V5 behavior | V6 behavior | Status | Assessment |
|---|---|---|---|---|---|
| 20 | Plant Quality banner | HEALTHY/…/NEEDS ATTENTION + trend counts | none | dropped | Partially covered by Dashboard yield + Triage count. A one-line "plant status" is cheap to add later if missed. |
| 21 | Models Needing Attention cards | trim% vs FT% + trend + recommendation | Triage cards (drift-based) | changed | Different lens (yield-gap vs drift). The trim-vs-FT GAP view (8340-1: trim 39.7% vs FT 94.2%) is genuinely useful and only partly covered by Model→Trim-vs-FT tab. See Should-fix S8. |
| 22 | All Models Ranked table | trim/FT/gap/trend/samples/action | worst-models list (trim/FT columns) | changed | Core kept; trend arrows + action text dropped. |
| 23 | Scorecard page (7 features) | per-model Cpk/Ppk/specs/drift text | Model page (drift table, history, units) | changed | Cpk/Ppk per model dropped — the one Scorecard item with no v6 home. Deferred. |

### V5 Process Files → v6 Process

| # | Feature | V5 | V6 | Status | Assessment |
|---|---|---|---|---|---|
| 24 | Incremental mode | checkbox | checkbox (default on) | kept | Now fast (stat fast-path, 2026-07-06). |
| 25 | Save-to-DB toggle | optional save | always saves | changed | Simplification; acceptable (processing without saving was a v5 foot-gun). |
| 26 | Date filter on files | process files from date X | none | dropped | Incremental mode covers the main use. |
| 27 | Select File(s) | individual file picker | folder only | dropped | Minor regression for one-off files — workaround: drop file in a folder. See S10. |
| 28 | Folder scan + cancel | recursive scan, cancellable | folder pick, no cancel | changed | Cancel missing; batches are resumable via incremental so low risk. |
| 29 | Progress + ETA | %, rate, ETA | progress bar + 5 counters + failures list | changed | ETA dropped; counters clearer. |
| 30 | Export batch results | Excel of batch | none | dropped | Mission cut. |
| 31 | Results textbox | per-file live log | recent-failures list | changed | Cleaner; full detail remains in log file. |
| 32 | (new) Go to Triage | — | appears on completion; drift state auto-advances after batch | new | The Job-1 loop closed (2026-07-06 fix). |

### V5 Analyze Trim → v6 Model → Units

| # | Feature | V5 | V6 | Status | Assessment |
|---|---|---|---|---|---|
| 33 | Model/date/status/serial filters | 6 filters + pagination | model selector + window + serial search (500 cap, sortable) | changed | Simpler; covers the workflow. Status filter dropped (S11 candidate). |
| 34 | Recent list w/ ML badges | anomaly/risk badges | status column only | changed | Predictor demoted by design (diagnostic panel). |
| 35 | Status banner | PASS/FAIL banner | modal title + row status | changed | Fine. |
| 36 | Re-analyze button | reprocess one file in place | none | dropped | Useful for corrections — v5 remains the tool until ported (Deferred D6). |
| 37 | Delete record | delete one analysis | Settings cleanup (bulk categories) | changed | Single-record delete dropped; acceptable (bulk covers hygiene; single delete is audit-risky anyway). |
| 38 | Export Chart (4-panel PNG/PDF) | comprehensive chart export | unit modal Save chart (PNG/PDF/SVG) | changed | v6 exports the chart only — no metrics/status panels. Your workflow ("charts for the team") works; the 4-panel version is richer. S5. |
| 39 | Export Model / Export SN (Excel) | per-model / per-unit workbooks | evidence pack (3 sheets, full history) | changed | Improvement (2026-07-07): full unit history + monthly summary + drift evidence, verified against raw SQL. |
| 40 | Track selector + comparison | per-track chart, compare-all | first track only, "track 1 of N" note | changed | REGRESSION for multi-track units — see Must-fix M6. |
| 41 | Metrics / File Info tabs | sigma/margins/anomaly/file meta | drift table + history stats | changed | Per-unit metrics detail thinner in v6; unit modal shows chart only. S5. |

### V5 Final Test (Compare) — no v6 page

| # | Feature | V5 | V6 | Status | Assessment |
|---|---|---|---|---|---|
| 42 | FT list + filters + pagination | browse FT results | Model→Trim-vs-FT tab (aggregates only) | changed | Browsing individual FT records: v5 only. |
| 43 | Re-match button | re-run FT↔trim matching | none | dropped | Operational tool — keep v5 available (D5). |
| 44 | Fix Missing Tracks | re-parse array-less records | none | dropped | On the finish plan (NULL-array fix pending); v5 remains the tool. |
| 45 | Trim-vs-FT overlay chart | overlay with spec band + match confidence | none (text KV only) | dropped | The best v5-only chart. Port recommended — S3. |
| 46 | Export selected comparisons (PDF) | multi-page PDF | none | dropped | Mission cut. |
| 47 | (new v6) Escape/Overkill KPIs | buried in dashboard text | per-model tab with serials disclosed | new | Improvement; verified live (8887: 0 escapes, 1 overkill, serial shown). |

### V5 Smoothness → v6 Model → Smoothness tab

| # | Feature | V5 | V6 | Status | Assessment |
|---|---|---|---|---|---|
| 48 | All-models comparison table | count/pass/avg-max/worst/spec/margin | none (per-model only) | dropped | Cross-model smoothness ranking gone; acceptable for the mission, revisit on demand. |
| 49 | Per-model stats card + list | tiles + paginated list | trend chart + recent list w/ verdicts | changed | Equivalent for investigation. |
| 50 | Smoothness-vs-position chart | per-unit sweep with exceedance markers | none (trend of max only) | dropped | Per-unit smoothness sweep is v5-only. Chart itself has a readability defect (audit #13). D7. |
| 51 | Smoothness Excel export | per-model sheets | monthly summary includes smoothness metric | changed | Thinner but adequate. |

### V5 Trends → v6 Dashboard/Triage/Model

| # | Feature | V5 | V6 | Status | Assessment |
|---|---|---|---|---|---|
| 52 | Standard/Drift/Difficulty views | 3-view segmented page | Dashboard + Triage + Model split | changed | Improvement: three jobs, three pages. |
| 53 | Summary tiles (9 stats) | active/samples/pass/at-risk/best/worst | Dashboard panels + Triage count | changed | Core kept. |
| 54 | Focus This Week (top-5 + advice) | ranked models + canned recommendations | Triage tier ordering | changed | See #16. |
| 55 | Failure Severity chart | fail-point-bucket bars | none | dropped | Near-miss analysis cut with it. Data (fail_points) still stored. |
| 56 | Cost Impact chart | top-15 scrap-$ bars | none | dropped | Mission cut. |
| 57 | ML Insights + dialog | trained counts, drift notes | Settings preview + Triage | changed | Fine. |
| 58 | Detail stat tiles (9) | per-model incl. Cpk, near-miss | Model pills + drift table + history stats | changed | Cpk again the gap. |
| 59 | Sigma scatter + range zoom | SPC scatter, pass/fail colors | FocusChart (any watched metric) | changed | Improvement: metric-generic, honest y-window (audit #1/#8). |
| 60 | Linearity P-chart + unit-yield overlay | per-model daily P-chart | History tab monthly pass-rate | changed | Coarser (monthly vs daily) but honest; unit-yield overlay dropped. |
| 61 | Sigma histogram (20+ samples) | distribution + spec lines | none | dropped | Distribution view has no v6 home. D8. |
| 62 | ML/Process drift tables + sparklines | sortable tables, canvas sparklines | Triage cards + Model drift table | changed | Improvement in evidence quality; sparklines-in-table dropped (fine). |
| 63 | Single-model drift dashboard (2×2) | sigma SPC + resistance/angle/retrim panels | FocusChart per metric + pills | changed | v6 is clearer (real dates vs bucket indices — audit #10). |
| 64 | Trim Difficulty chart | avg-passes bars + retrim% | trim_pass_count watched metric + exports | changed | Ranking view dropped (mission cut); the metric is still monitored. |
| 65 | Trends PDF exports (2) | summary/detail PDFs | none | dropped | Mission cut. |

### V5 Model Specs → v6 Settings → Per-model Specs

| # | Feature | V5 | V6 | Status | Assessment |
|---|---|---|---|---|---|
| 66 | Specs table + search + pagination | 50/page browser, incomplete flagged red | model combobox (pick one) | changed | Browsing/QA-ing the spec TABLE is weaker in v6; import merge covers bulk. S9. |
| 67 | 18-field editor | element/class/resistance/angle/aliases/notes | 4 fields (linearity text/%, exclude pts trim/FT) | changed | v6 edits only what the engine consumes. Aliases/notes etc. remain DB columns editable via import. Acceptable; document it. |
| 68 | Check Discrepancies | file-vs-reference comparison + badge | none | dropped | On finish plan (ingest-time check). D4. |
| 69 | Spec import from master sheet | none (manual) | bulk import/merge button | new | Improvement. |

### V5 Export page

| # | Feature | V5 | V6 | Status | Assessment |
|---|---|---|---|---|---|
| 70 | Search/select units + bulk chart export | PNG-per-unit or combined PDF | unit modal save (one at a time) | dropped | Bulk unit-chart export gone. Your stated workflow is per-unit for the team — acceptable; batch export is S6 if volume grows. |

### V5 Settings → v6 Settings

| # | Feature | V5 | V6 | Status | Assessment |
|---|---|---|---|---|---|
| 71 | DB path/export path/theme | path pickers, theme switch | none (config.yaml) | dropped | Low-value UI; config file covers it. |
| 72 | ML enable + Train + Apply-to-DB | v5 ML lifecycle | Retrain drift + Retrain per-model ML | changed | Apply-to-DB (bulk re-gating) unreachable in v6 — intentional (audit honesty). |
| 73 | ML status/staleness labels | trained counts, retrain-needed list | training modal progress only | changed | Staleness visibility dropped — S7 (small label). |
| 74 | MPS list + recent-days + cost ratio | textbox + entries | Active Models section + Pricing section | kept | Improvement (status line shows active counts, unknown-pin warning). |
| 75 | Pricing import/clear | Excel/CSV import | same | kept | Same code path, verified. |
| 76 | Scan DB + 7 cleanup filters + preview + delete + reset skipped | full hygiene suite | same 6 categories + before-date + preview + confirm + reset | kept | Parity (misclassified-FT etc.). |
| 77 | (new v6) Alert Thresholds | — | preset slider + live would-flag preview | new | Preview now mathematically consistent with training (2026-07-06 fix), verified live (14/16/64). |
| 78 | (new v6) Company trend | — | Dashboard section, weekly/monthly, A/B/C | new | Verified exact vs raw SQL. |
| 79 | (new v6) Startup drift catch-up + post-batch advance | — | automatic | new | Closes the frozen-triage failure mode. |
| 80 | (new v6) Evidence pack v2 | — | Drift evidence + full Unit history + Monthly summary | new | Verified: honest σ-shift matches UI; full record. |

---

## (c) Chart audit

Every chart in both UIs. Evidence = rendered PNG in `charts/` and/or live screenshot + independent recomputation. Verdicts: **correct** / **wrong** / **correct-but-misleading**.

| # | Chart | Where | Verdict | Evidence & notes | Fix |
|---|---|---|---|---|---|
| 1 | FocusChart (focus metric) | v6 Model | **correct-but-misleading (default window)** | `v6_focus_8887_linearity_all.png` matches live app; baseline 0.02983±0.02616, shift +16.32σ recomputed EXACT from raw SQL. SPC overlays correct; off-scale annotation honest. BUT: default 90d window is wall-clock-anchored → clicking the live +16.3σ Triage card showed "No measurements in the selected window." Alert points at a chart that renders empty. | Anchor windows to the model's latest file_date (like `compute_recent_means`) or auto-fall back to All with a notice. Must-fix M1. |
| 2 | FocusChart (Smoothness tab) | v6 Model | **correct** | `v6_smoothness_trend_1844205.png` (n=33, scatter mode — no fake continuity); live empty-state honest ("No smoothness records"). | Suppress the empty 0–1 axes frame when no data (cosmetic). |
| 3 | CompanyTrendChart | v6 Dashboard | **correct** | Weekly 90d + monthly All rendered (`v6_company_trend_*.png`) and live; all 8 weekly company points recomputed EXACT (e.g. W18 = 101/204 = 49.5%). Per-system split verified (A/B; C absent = correct, no LTS3 rows here). Edges: empty/single/partial all safe (`edge_company_*.png`). | Partial-period honesty: current week/month renders as a cliff (July = 0% from 2 units in the edge test). Hollow-marker or "(partial)" label for the last bucket. M5. Minor: isolated months render as floating dots (visible ~2015); legend can overlap data. |
| 4 | MiniTrendChart (×2 panels) | v6 Dashboard | **correct** | `v6_mini_trend_trim_90d.png`; 39 points; panel headline 59.8% = 748/1251 recomputed EXACT (and FT 90.6% = 1276/1408). Fixed 0–100 y is right for a sparkline. | None. |
| 5 | Unit chart (error-vs-position) | v6 Model → modal (shared ChartWidget) | **correct** | `v6_unit_chart_8887_36.png` (42 pts, 0 fails, spec band) + live Unit 37 capture; fail-point rule = zero-tolerance per-point spec check (code-verified `compute_fail_points`). | Legend `loc=best` can sit on the trace (live shot). v5's version also draws corrected trace + "trim improved %" — port that context (S5). Multi-track: only track 1 shown — M6. |
| 6 | HistoryTab (2 panels) | v6 Model | **correct** | `v6_history_8887.png` + live; angle stats n=43, mean 0.8205, σ=0.001757 recomputed EXACT. Robust 2–98pct y-window + off-scale note works. | Title/tick collision between stacked panels (cosmetic). Bottom panel is per-track monthly rate while Trim-vs-FT tab is per-unit — label the basis (S4). |
| 7 | Dashboard P-chart | v5 | **correct** | `v5_dashboard_pchart_daily.png` + live ("Data as of 05/28" disclosure is good practice). Variable binomial limits + violations marked. | None (v5-retained). Its "data as of" banner should be COPIED to v6 (M1). |
| 8 | Sigma scatter | v5 Trends detail | **correct** | `v5_trends_sigma_scatter_6607.png` (1,330 pts, threshold line). Dense but honest. | Superseded by FocusChart; retire with v5. |
| 9 | Sigma histogram | v5 Trends detail | **correct** | `v5_trends_sigma_histogram_6607.png`. | No v6 home (D8). |
| 10 | Single-model drift 2×2 | v5 Trends Drift | **correct-but-hard-to-read** | Live capture (8071-8): sigma SPC + 3 smoothed panels render fine, but x-axes are bucket INDICES (0–10), not dates — time is unlabeled. | Retire with v5; v6 FocusChart uses real dates. |
| 11 | ML/Process drift sparklines (tk.Canvas) | v5 Trends tables | **not renderable headless; correct live** | Live capture shows per-row sparklines drawing. | Retire with v5. |
| 12 | Failure Severity bars | v5 Trends summary | **correct** | Live capture (15,135 failing tracks bucketed; 47% in first bucket). | Dropped in v6 by mission; no action. |
| 13 | Smoothness-vs-position | v5 Smoothness | **correct-but-misleading (readability)** | Live capture SN20: 14,375 exceedance markers painted as a solid red mass — the trace is unreadable and "everything exceeds" is the only takeaway. | If ever ported: line + shaded exceedance region + count, not per-point X markers. |
| 14 | Trim-vs-FT overlay | v5 Compare | **correct** | Live capture (8895 SN22): trim raw/corrected + FT raw/corrected + spec band + match method/confidence disclosed. Best v5-only chart. | Port to v6 unit modal when an FT link exists — S3. |
| 15 | Trim Difficulty bars | v5 Trends | **correct** | Live capture: avg-passes bars + retrim% annotations. | Dropped by mission; metric still watched. |
| 16 | Cost Impact bars | v5 Trends summary | **not exercised** (pricing-dependent) | Live summary shows $38,020/90d line on dashboard; chart itself renders in v5 summary view (captured). Math not recomputed (pricing map × cost ratio — config-driven). | Out of v6 scope; if leadership reporting returns, recompute then. |
| 17 | Comprehensive 4-panel exports (Analyze/Compare/Export pages) | v5 export-only | **correct (code-reviewed, extractable)** | Same plotting functions as #5/#14 with info panels; not re-rendered (redundant with #5/#14 evidence). | S5/S6 if batch export returns. |
| 18 | Edge-case suite | both | **pass** | 12 edge renders, zero exceptions: `edge_focus_empty/single/retrims/constant-zero-std`, `edge_company_empty/single/partial`, `edge_mini_empty/single`, `edge_history_empty`, `edge_unit_no_sweep/single_point`, `edge_v5_pchart_empty/single`. Same-day retrims stack vertically at one x (honest); None values dropped; zero-σ constant renders STABLE-ish flat band (guard working). | None. |
| 19 | Smoothness comparison table stats | v5 Smoothness (table, not chart) | **definition mismatch** | UI shows 7844: Count 60, Avg Max 0.0346; raw SQL says 68 rows (57 distinct serials), avg 0.0317; worst 1.4070 matches. Not a math error — the page's model-stats query filters differently (unlabeled). | Label the basis (latest-per-unit?) or align with raw counts if ported. |

**Cross-cutting chart verdict:** no wrong numbers found anywhere — every mismatch traced to definition/labeling, not computation. The systemic risks are *contextual honesty* (stale data, partial periods, window anchoring), not math.

---

## (d) Must-fix before v6 ships (as the only UI)

**M1 — Data-vintage and staleness honesty.** The DB ends 2026-05-28; today the Dashboard silently presents "90d" windows anchored to wall-clock now, and Triage's top card was a model whose last unit ran 321 days ago (surfaced because it's MPS-pinned). Add: (a) a "Data through YYYY-MM-DD" line on Dashboard/Triage (v5's P-chart already does this — copy it), (b) a staleness chip on Triage cards and the Model header ("last data 321d ago"), (c) window anchoring to latest data (or auto-All fallback) on the Model focus chart. *Cost if skipped:* engineers walk into a chart that contradicts the alert that sent them there; trust in Triage dies in week one.

**M2 — Outlier-robust "recent" for drift evidence.** 8887's +16.3σ = one linearity_error value of 10.007 (≈340× baseline mean) inside a 24-row anchored window; without it the recent mean is 0.0414 (≈ +0.44σ). The CUSUM/EWMA tiers have their own guards, but the headline σ-shift on cards/pills/exports is a plain mean. Use median or a trimmed mean for `compute_recent_means`, and flag any recent window containing values > k× spec as "contains suspect data" instead of letting them set the headline. *Cost if skipped:* your loudest alerts are data-quality artifacts; real drift drowns.

**M3 — Ingest data-quality guard for scale anomalies.** That 10.007 should never have entered silently (zero-tolerance spec ⇒ linearity errors are O(0.01–0.1)). Flag values > N× the model's spec at ingest as suspect (`data_quality` column already exists). Feeds M2. *Cost if skipped:* M2 keeps treating symptoms.

**M4 — WARNING recompute backfill.** 42% of all rows (429 of 1,256 in the live 90d window) are WARNING with three historical meanings; some are linearity-FAILs. The Dashboard reads "59.8% pass" while actual FAIL is 5.9% — a leadership eye can't tell process failure from classification debt, and for a zero-tolerance customer spec that ambiguity is an AS9100 exposure. Run the status-recompute backfill (rule already correct in analyzer.py; runbook it like trim_pass_count) and add a one-line legend of what Warning means. *Cost if skipped:* every yield number in the app stays arguable.

**M5 — Partial-period marker on Company trend.** Current week/month renders as a cliff (edge-tested: a 2-unit July renders 0%). Hollow marker or "(partial)" on the last bucket. Trivial. *Cost if skipped:* monthly review starts with "why did yield crash" every month-start.

**M6 — Multi-track units in the unit modal.** Modal shows track 1 of N with a title note. Your product families are multi-track; an investigation view that hides tracks 2+ can miss the failing element. Add a track selector (v5 Analyze had one). *Cost if skipped:* Job 2a incomplete for exactly the units most worth investigating.

**M7 — Commit the working tree.** The stability/drift/System-C/trends work driving this review is uncommitted on `V6`. Commit before anything else touches the tree.

## (e) Should-fix (soon after)

- **S1 — Triage calibration story.** 39 flagged of ~51 active models (and Settings preview: 94/279 at standard) is still "flags most things." With M1–M3 the count will drop; then pick the preset that matches your review capacity and document it (the preview now tells the truth). Also surface *why* per card (already: metric + σ) plus staleness/pin badges (M1c covers most).
- **S2 — Startup contention.** Live launch: window appeared well after DB init (migrations + eager page build + section workers on a 3.2 GB DB; plus first-launch grant quirks on macOS). Defer section data-loads until a section expands; show the window before pages finish building.
- **S3 — Port the Trim-vs-FT overlay chart** into the v6 unit modal when an FT link exists (chart #14). It's the single v5 visual with unique investigative value.
- **S4 — Rate-basis labels.** History tab (per-track monthly) vs Trim-vs-FT tab (per-unit) can disagree; label each ("track basis" / "unit basis").
- **S5 — Unit modal parity.** Corrected trace + "trim improved %" + optional metrics side-panel (v5's 4-panel export had these); legend placement off-data.
- **S6 — Batch unit-chart export** (v5 Export page's job) if per-unit saving becomes tedious — evidence pack + loop is 80% of the code.
- **S7 — ML staleness label** in Settings (v5 had it; one query + one label).
- **S8 — Trim%-vs-FT% gap column** in the worst-models list (Quality Health's best idea, one query away — the 8340-1 39.7%/94.2% overkill pattern should be visible company-wide).
- **S9 — Spec browser table** in Settings (v5 Specs page's table with incomplete-spec highlighting) — AS9100 spec hygiene wants a browsable view, not just a per-model picker.
- **S10 — Single-file processing** (file picker next to folder picker).
- **S11 — Cosmetics:** 9th pill truncation ("omposite trim-ris"), History panel title overlap, smoothness empty-frame suppression, Process page counting UNTRIMMED under "Passed" (label it "Passed/Untrimmed" or add a 6th counter).

## (f) Deferred, with rationale

- **D1 — v5 page retirement.** Keep `--v5` launchable until M-items + S3 land and Fix Missing Tracks/Re-match have v6 homes. Zero cost to keep; it's already isolated (one shared widget import, verified).
- **D2 — Cost/exec reporting (Excel summary, cost charts).** Cut by the mission ("not leadership reporting"). Revisit only if that mission changes; the pricing config and queries survive either way.
- **D3 — Cpk/Ppk surfaces.** Queries exist (`get_model_cpk`, trend). Your stated workflow didn't include them; defer until asked-for, then it's a Model-page stat row, not a page.
- **D4 — Ingest-time spec discrepancy check.** Already on the finish plan; retrospective checker exists.
- **D5 — FT browse/re-match/Fix-Missing-Tracks in v6.** Wait for the NULL-array repair work (finish plan P2-2) so it's built once, correctly.
- **D6 — Re-analyze single file.** Niche; v5 covers it meanwhile.
- **D7 — Per-unit smoothness sweep chart.** Port only with the readability redesign (audit #13).
- **D8 — Distribution histograms.** Nice-to-have; drift engine + focus chart answer the daily questions.
- **D9 — manager.py decomposition (~8.9k lines) + old single-metric drift detector retirement.** Real debt, zero user-facing risk; do it with D1 since v5 pages are the old detector's only readers (it currently burns training time writing state nothing reads).

## (g) Lock-in decisions — expensive to reverse, flagged loudly

1. **SQLite + StaticPool + process-wide session lock.** Correct for a single-user desktop tool, and the 2026-07-06 work made the UI respect it. But it is a **hard single-user ceiling**: the moment a second user, a shared drive DB, or a background service appears, this needs WAL-multi-connection or a server DB, touching every session call site. Decide *now* that v6 is single-user by contract, or budget the migration.
2. **Identity-by-path for System C (LTS3 folder).** Cheap and it unblocked you, but it makes **folder naming an operational requirement**: an LTS3 file copied elsewhere silently becomes System B, and the migration retag keys on the same convention. Write the folder rule into the work instructions (AS9100 doc control), or eventually add an in-file marker if the laser vendor provides one.
3. **Two SystemType enums (core + DB) with manual mapping.** Every future system = 2 enums + 3 mapping sites + comparison buckets. Worked fine for C, but it's copy-paste-dependent. Acceptable; unify only when touching the mapper anyway.
4. **file_date day granularity as the universal time axis.** Within-day ordering is unknowable (the advance watermark now works around it via row ids). All charts/aggregations inherit day resolution permanently; if the lasers ever emit timestamps, capture them at parse-time — retrofitting is a full reprocess.
5. **model_metric_state stores only current thresholds/baselines (no history).** Preset changes and retrains overwrite in place, so "what would have alerted last March" is unanswerable — a traceability gap if drift alerts ever back a customer-facing quality decision. A small append-only audit table would close it; add before Triage output enters the QMS.
6. **Baseline = oldest-70%-of-history policy.** For long-history models the baseline embeds ancient process regimes (8887's includes 2024 data), and every retrain re-fixes it. If you later adopt "baseline = last known-good window," ALL tiers shift and historical comparability breaks. Choose the policy deliberately before tiers drive formal dispositions.
7. **UiDispatcher as the only legal cross-thread UI path.** The freeze class is gone *as long as* new code never calls Tk from workers. That's a convention, not a compiler check — one line in CLAUDE.md's dev rules keeps future-you honest.
8. **Evidence pack = full history by default.** Now the contract for "the model's record"; at 78k rows a giant model makes a heavy workbook. Fine today; add a row cap disclosure if models grow 10×.

---

## Verdict recap

V6's five-page design matches how the work actually flows (process → triage → investigate → configure), its charts are mathematically trustworthy (22/22 exact recomputations on everything v6 renders), and the last two days proved features land in it cheaply. The v5 features worth money are either already ported, on the S-list (FT overlay, gap column, spec browser), or deliberately out of mission. Fix the trust layer (M1–M7) and stop — there is no v7-shaped problem here.
