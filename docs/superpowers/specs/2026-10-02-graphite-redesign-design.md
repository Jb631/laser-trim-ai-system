# Graphite redesign — design (2026-10-02)

James, in the handoff he pasted on 2026-10-02: "im just not happy with the app, there is so much
going on its hard to see what is what, i dont like the way the app presents" · "i dont think its
colors or font its the overall ui" · "i like dark mode". The September facelift (TRACKER F1–F5)
restyled and reordered every page but removed nothing — seven sidebar destinations, and the Model
page stacks 12 metric pills, a stats table, a chart and 7 tabs. **This redesign removes.**

## Decisions James made (each from a rendered option with his real data)

1. **Graphite** (picked from three dark options in the btx_quality session): neutral near-black,
   white text, ONE blue accent, no navy, no teal; IBM Plex kept. bg `#0c0c0e` · card `#151518` ·
   card border `#26262b` · dividers `#1d1d21` · text `#ededed` · secondary `#8b8b93` · accent
   `#3b82f6` (the single primary button) · chart highlight `#60a5fa` · history bars `#2a3a58` ·
   worse/fail `#f87171` · better/pass `#4ade80`.
2. **Overview structure** (the approved mockup): a top bar replaces the sidebar — name · Overview ·
   Models · Settings · one blue "Process new files"; "N models need a look" as cards; "Everything
   else" as a plain list; metrics, findings, drift and history live inside a model.
3. **Which models get a card** (2026-10-02, drawn on the home copy): "keep all 16 cards (fail rate
   up, or a signal moved), each with its reason" — the union of the "Drifting now" fail-rate list
   (`ml/spc.compute_focus_list`) and the drift detector's flags (`get_drifting_models`), each card
   saying in one line why it is there. (13 on the 30 Sep work copy.)
4. **Model page = layout C, four tabs** ("i like c", from three layouts drawn on 8504-2): a header
   over **Summary · Units · Final test · History**.

## Rulings (mine — say if any is wrong)

- **Tokens, not a fork.** Graphite is a new value for every token in `gui/v6/theme.py`; no widget
  holds a hex colour today (grep: 0 outside theme.py), so the whole app turns at once. Derived
  tokens are measured by `tests/test_theme_contrast.py` (47/47 already). Lasers become violet
  `#a78bfa` / fuchsia `#e879f9` / pink `#f472b6`: the readability test keeps each 30° of hue from
  every colour that means something, and blue now means "act here". Text on the blue button is
  dark (white on `#3b82f6` is 3.7:1).
- **Navigation (James: "thats fine", 2026-10-02).** Top bar: Overview (key `home`), Models (key
  `model`), Settings (`settings`); keys unchanged so every deep link still works. **Triage is
  retired** — Overview's cards and list are what it showed. **Fleet Findings and Dashboard leave
  the bar** but stay one click away as two quiet links at the foot of Overview ("All findings",
  "Company trends"); findings appear inside each model's Summary. Process is reached by the blue
  button.
- **"Process new files"** runs the remembered folder list (what Home's button did) on the Process
  page, which now holds both runs — the remembered folders at the top, "a specific folder" below —
  and the progress. A finished run lands on Overview (it used to land on Triage).
- **Overview card:** model (+ "hand trim" tag for 8232-1 and 8340-1), units in the last 90 days,
  big pass % (90 days), "was X%" (the year before that), monthly pass bars (12 months, history
  colour, newest month highlighted), and the reason in red. Order: the fail-rate list's own rank
  first, then the detector's flags by tier. A model with only final-test data (8506) shows its
  final-test pass instead. Click → its Model page.
- **"Everything else":** active models only (trimmed in the last 90 days), busiest first: model ·
  units · pass % · steady / up N pts / down N pts (±5 points = steady). Inactive models (F5) sit in
  a collapsed line at the end — counted, never hidden.
- **Model page C.** Header: model, a status word (Drifting / Steady / Inactive), pass %, units,
  lasers. **Summary**: the headline in one sentence; the moving signal's run chart; "Also moving";
  "Worth changing" (the model's findings); "All 12 signals ▸" (today's drift table, folded).
  **Units**: today's Units tab, with Smoothness under it. **Final test**: final-test units, with
  trim vs final test under it. **History**: today's History tab. The 12 pills and the stats table
  go; their numbers live in "All 12 signals" and the Units tab.
- **Failures stay loud** (the standing rule): any load that fails is named in a banner, never drawn
  as "nothing needs a look".

## Not in this redesign
Model-name merges (J5, James to confirm), the hand-trim suspect line (J6), the composite back-fill
(J7), V5 (H4).
