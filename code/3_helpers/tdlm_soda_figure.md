# TDLM vs SODA Explainer Figure — Handoff

This is a working note for any future agent picking up `tdlm_soda_figure.py`. It
describes the current design, what every panel shows, the choices that were
made (and why), and the open issues / likely next steps.

## Goal

A single production-ready figure for a paper / talk that conveys the *gist* of
two methods for detecting fast neural sequences:

- **TDLM** (Liu et al. 2021, eLife) — lag-domain template-matching on MEG/EEG.
- **SODA** (Wittkuhn & Schuck 2021, Nat Comms) — slope-time-course +
  Lomb-Scargle on fMRI.

The figure is **conceptual**, not a literal reproduction of the papers' plots.

## Files

- `tdlm_soda_figure.py` — the figure script (single file, ~445 lines).
- Outputs: `tdlm_soda_methods.svg`, `tdlm_soda_methods.png` (300 dpi),
  `tdlm_soda_methods.pdf` (when built).
- Source papers in repo root: `Liu et al. - 2021 ...pdf`, `Wittkuhn and Schuck - 2021 ...pdf`.
- `3_SODA_explainer.py` — original SODA explainer that the SODA row in this
  figure was derived from (good reference for the data-generation model).

Method summaries (canonical, in agent memory):

- `~/.claude/projects/-home-simon-Desktop-tdlm-explainer/memory/tdlm_method.md`
- `~/.claude/projects/-home-simon-Desktop-tdlm-explainer/memory/soda_method.md`

## Layout

`figsize=(14, 6.4)`, 2 rows × 3 cols GridSpec, width ratios `[0.95, 1.40, 1.05]`.
Each row tells one method's story as **input → mechanism → output**:

```
        col 0 (input)        col 1 (mechanism)         col 2 (output)
row 0   a) TDLM X(t)         b) β matrices @ 40/90 ms  c) Z_F(Δt), Z_B(Δt)
row 1   d) SODA X(t)         e) per-TR slope fits      f) slope dynamic b(t)
```

Row labels "TDLM (MEG/EEG)" / "SODA (fMRI)" are placed vertically in the left
margin via `fig.text` (two separate calls per row — multiline + rotation
breaks SVG rendering).

## Shared style

- Arial sans-serif, base font size 9.5, top/right spines hidden.
- 300 dpi PNG + SVG output.
- Panel letters via `panel_title(ax, "a)", "...")`, bold, `loc="left"`.
- Colors:
  - `C_FWD = "#c0392b"` (forward / positive), `C_BWD = "#2c7fb8"` (backward).
  - **Shared tab10 palette across rows**: TDLM states A–D and SODA items A–E
    use `plt.cm.tab10(0..4)` so the "blue thing" is the same item across
    panels — this visual unification was an explicit user request.

## Panel-by-panel

### (a) TDLM input — `axA`

Four stacked Gaussian "bumps" (states A,B,C,D) along a 0–260 ms time axis
with **40 ms peak spacing** (peaks at 60/100/140/180 ms). Each ribbon is a
narrow Gaussian (σ=3 ms) plus light smoothed noise, with the matching state
letter colored on the left.

A faint curved arrow + italic "latent order" label hints at the hidden
sequence. Y axis is hidden (left spine off); xlabel says
"time (ms), sample rate ~100 Hz".

### (b) TDLM mechanism — `axB`  *(most-iterated panel)*

Conceptually: "at each lag Δt, build a 4×4 transition matrix β; sum the
evidence on the forward (upper off-diagonal) cells; that sum is the point on
the sequenceness curve at Δt."

Implementation:

- `axB` is a transparent host axes (`axis("off")`) — the matrices are
  `inset_axes` placed inside it; `axB.text` is used for the caption and the
  Σ labels.
- Two example matrices shown: **Δt = 40 ms ("high evidence")** and
  **Δt = 90 ms ("no evidence")**.
- `beta_at(lag, seed)` builds a toy β: forward (i, i+1) cells filled with
  `peak_strength * exp(-(lag - 40)² / (2·22²))` and a within-lag decay
  `[1.00, 0.92, 0.85]`; small Gaussian noise added everywhere.
- Forward cells highlighted with a 2-pt red rectangle.
- Tick labels are A/B/C/D on **both axes**, color-matched to the state palette
  (no axis tick lines).
- Two-line subtitle per matrix: `Δt = {lag} ms  ({tag})`, regular weight.
- Σ value (sum over the forward cells) printed below each matrix in bold red.
- Italic caption "example transition matrices for lag 40 ms & 90 ms" at the
  top of `axB` (y=0.99).

**Important coupling with panel (c):** the same `beta_at(lag, seed=lag)`
function is used to generate the gray dots in (c) at every 10-ms lag, and the
two bold black dots in (c) are exactly the Σ values shown in (b). Keep this
synchronization if you edit either side.

### (c) TDLM output — `axC`

- `Z_F(Δt)` (forward, red, `zf_clean`) — bell centered on 40 ms.
- `Z_B(Δt)` (backward, blue) — small noisy spline through random knots; near
  zero everywhere.
- Gray dots at every 10 ms lag (computed from `beta_at(lag, seed=lag).sum()`
  on the forward mask); 40 ms and 90 ms are emphasized with bold black dots.
- Annotation "replay time lag 40 ms" pointing at the 40 ms peak.
- x-range capped at 150 ms; legend top-right.
- Permutation threshold line is currently **commented out** (was removed at
  user request).

### (d) SODA input — `axD`

Five stacked, **heavily overlapping** Gaussian "HRF" responses (items A–E),
mirroring TDLM's stacked-ribbon look but with `ribbon_offset = 0.40` so the
ribbons clearly bleed into each other (this is the visual point: fMRI cannot
temporally separate them).

- Item centers `2.5 + 0.15·i` TRs, σ = 0.5 TR.
- Integer TRs 0..4 highlighted as faint gray vertical bands.
- Sample dots drawn on each ribbon at each integer TR — these are the values
  used in panel (e).
- xlabel: "time (TRs), sample rate ~0.8 Hz".

### (e) SODA mechanism — per-TR regressions

Nested GridSpec (`gsE_outer` 3 rows: title / row of 4 scatters / spacer;
`gsE_inner` 1×4 for the four mini-scatters). TRs 1..4 are plotted (TR 0 is
all-near-zero and uninformative).

- Per TR: scatter (probability − min) vs. serial position 1..5, color-coded
  by item (same tab10 palette).
- `fit_slope` does OLS on the min-subtracted data; sign is flipped so that
  **positive slope = forward** (matches the user's mental model). This was a
  back-and-forth: force-through-origin gave all-positive slopes and was
  rejected; the current "OLS after subtracting min" preserves the sign while
  baselining each plot at zero visually.
- Line color: red if `flipped > 0`, blue otherwise.
- Annotation `b_{TR} = ±0.xx` top-center.

### (f) SODA output — slope dynamic `axF`

- `slope_at(t)` evaluated densely along time → continuous curve (black).
- Filled red above zero, blue below zero (semantic colors).
- The four discrete slopes from panel (e) are overlaid as black dots labeled
  "TR1..TR4". Labels are always above the dot for consistency.
- "onset" (red, top-left) / "offset" (blue, bottom-left) annotations indicate
  the forward → backward swing.

  **Note on terminology**: "onset"/"offset" are informal here. The Wittkuhn &
  Schuck paper calls these the **forward** and **backward** periods. If
  rewording the caption / labels, prefer the paper's terminology.

## Design decisions worth preserving

1. **Conceptual, not literal.** The user repeatedly steered away from
   reproducing PDF figures. Keep the abstraction level high.
2. **2×3 input/mechanism/output structure.** Both rows obey it.
3. **Shared item color palette** across TDLM and SODA rows.
4. **(b) and (c) are coupled.** Same `beta_at` underlies both — gray dots
   must equal what the matrices would produce.
5. **Min-subtracted OLS, not through-origin**, in panel (e) — preserves slope
   sign.
6. **Panel B caption + Σ values** were carefully placed to avoid colliding
   with matrix titles/xlabels. If you change matrix size or `mini_y`/`mini_h`,
   re-check overlap.

## Open issues / likely next requests

- **F)'s "onset/offset" wording vs. "forward/backward"** — user flagged in
  the caption review that the paper uses "forward period" / "backward period".
  Consider switching the labels in `axF` (lines ~424–429) to match.
- **Caption text for the final paper** — last user-proposed caption is in the
  prior conversation summary. It needs:
  - typo fix "a empirical" → "an empirical"
  - rephrase the B-panel question so the lag is on the predicted side
    (e.g. "how much does an earlier reactivation of Y predict a later
    reactivation of X, Δt later?")
  - swap "onset/offset" wording per above
  - pick a single notation for the time lag (Δt, not alternating with "t")
- **No frequency-spectrum panel for SODA.** The original SODA paper makes a
  big deal of the Lomb-Scargle periodogram of `b(t)` to estimate replay
  speed. The current figure only shows the time-domain slope dynamic. If a
  reviewer asks "where's the frequency analysis?" we may want to add an
  inset spectrum to panel (f), or a small 7th panel.
- **TDLM (a) shows only forward replay.** No backward example exists in the
  figure. Probably fine for a "gist" figure, but worth knowing.
- **Permutation threshold** in (c) was removed; if statistical-inference
  framing is needed later, uncomment lines ~237–239.

## How to run

```bash
cd /home/simon/Desktop/tdlm-explainer
python tdlm_soda_figure.py
# writes tdlm_soda_methods.svg and tdlm_soda_methods.png
```

No external assets are loaded — everything is synthetic. Dependencies:
`numpy`, `scipy`, `matplotlib`, `seaborn` (only for the default theme; could
be dropped).

## User collaboration notes

- Prefers terse responses; do not pad with summaries.
- Will manually edit the script between turns — always re-read before editing.
- Pushes back hard on anything that looks like reproducing the source PDFs.
- Cares about visual consistency between the two rows (colors, ribbon style,
  axis treatment).
