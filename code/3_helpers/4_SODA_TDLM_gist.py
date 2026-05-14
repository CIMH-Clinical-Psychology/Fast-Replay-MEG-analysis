"""
Conceptual comparison of two methods for detecting fast neural sequences:

    TDLM (Liu et al. 2021, eLife)         — lag-domain template matching
    SODA (Wittkuhn & Schuck 2021, Nat Comms) — frequency-domain HRF unmasking

Each row tells one method's story in three steps: input → mechanism → output.
The figure aims to convey the *concept* of each method, not to reproduce the
plots from the original papers.
"""
from __future__ import annotations

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns


rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# Font scaling — multiply every font size in the figure by this factor.
# ----------------------------------------------------------------------
FONT_SCALAR = 1.5

# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "svg.fonttype": "path",
    "font.size": 9.5 * FONT_SCALAR,
    "axes.labelsize": 9.5 * FONT_SCALAR,
    "axes.titlesize": 10 * FONT_SCALAR,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8.5 * FONT_SCALAR,
    "ytick.labelsize": 8.5 * FONT_SCALAR,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 8 * FONT_SCALAR,
    "legend.frameon": False,
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.default": "regular",
})

# Semantic colors
C_FWD  = "#c0392b"
C_BWD  = "#2c7fb8"
C_REST = "#95a5a6"
# Shared palette for both rows: tab10 (seaborn default). Item/state A → blue,
# B → orange, C → green, D → red, E → purple. Used in panel (a) for TDLM
# states and in panel (d) for SODA items so colours line up across rows.
ITEM_COLORS = [plt.cm.tab10(i) for i in range(5)]
STATE_COLORS = ITEM_COLORS[:4]


def panel_title(ax, letter, text, *, pad=8):
    ax.set_title(f"{letter.upper()} {text}", loc="left", fontweight="bold", pad=pad)


def hrf_sine(t, A=0.55, lam=5.24, d=0.0, b=0.10):
    out = np.full_like(t, b, dtype=float)
    m = (t >= d) & (t <= d + lam)
    out[m] = (A / 2) * np.sin(2 * np.pi * (t[m] - d) / lam - 0.5 * np.pi) + b + A / 2
    return out


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(14, 6.4))
gs = GridSpec(
    2, 3, figure=fig,
    hspace=0.65, wspace=0.32,
    left=0.075, right=0.975, top=0.91, bottom=0.10,
    width_ratios=[0.95, 1.40, 1.05],
)

# Row labels in the left margin — separate calls to avoid multiline+rotation SVG bug
fig.text(0.018, 0.715, "TDLM", rotation=90, fontsize=12 * FONT_SCALAR, fontweight="bold",
         color="#222", ha="center", va="center")
fig.text(0.038, 0.715, "(MEG / EEG)", rotation=90, fontsize=8.5 * FONT_SCALAR,
         color="#666", ha="center", va="center")
fig.text(0.018, 0.275, "SODA", rotation=90, fontsize=12 * FONT_SCALAR, fontweight="bold",
         color="#222", ha="center", va="center")
fig.text(0.038, 0.275, "(fMRI)", rotation=90, fontsize=8.5 * FONT_SCALAR,
         color="#666", ha="center", va="center")


# ============================================================
# (a) TDLM input — decoded state probabilities, hidden order
# ============================================================
axA = fig.add_subplot(gs[0, 0])
state_names = list("ABCD")
t_ms = np.linspace(0, 260, 700)
peaks_ms = [60, 100, 140, 180]   # 40 ms spacing
sigma = 3

for i, p in enumerate(peaks_ms):
    bump = 0.85 * np.exp(-((t_ms - p) ** 2) / (2 * sigma ** 2))
    noise = 0.06 * np.convolve(rng.standard_normal(len(t_ms)), np.ones(18) / 18, mode="same")
    y0 = (3 - i) * 1.05
    axA.fill_between(t_ms, y0, y0 + bump + noise, color=STATE_COLORS[i], alpha=0.30, lw=0)
    axA.plot(t_ms, y0 + bump + noise, color=STATE_COLORS[i], lw=1.05)
    axA.text(-12, y0 + 0.3, state_names[i], color=STATE_COLORS[i],
             fontweight="bold", ha="right", va="center", fontsize=10 * FONT_SCALAR)

# subtle directional arrow indicating the latent sequence
axA.annotate("", xy=(peaks_ms[-1] + 5, 0.25), xytext=(peaks_ms[0] - 5, 3.35),
             arrowprops=dict(arrowstyle="->", color="0.45", lw=0.9,
                             connectionstyle="arc3,rad=-0.25"))
axA.text(158, 2.55, "latent",
         color="0.45", fontsize=8.5 * FONT_SCALAR, style="italic", ha="center")
axA.text(158, 2.15, "order",
         color="0.45", fontsize=8.5 * FONT_SCALAR, style="italic", ha="center")

axA.set_xlim(-20, 245)
axA.set_yticks([])
axA.spines["left"].set_visible(False)
axA.set_xlabel("time (ms), sample rate ~100 Hz")
panel_title(axA, "a)", r"Input: decoded states  $X(t)$")


# ============================================================
# (b) TDLM mechanism — β transition matrices at lags 10..80 ms;
# correct (forward) cells are highlighted; their sum at each lag
# is the value plotted on the sequenceness curve in panel (c).
# ============================================================
axB = fig.add_subplot(gs[0, 1])
axB.set_xlim(0, 1); axB.set_ylim(0, 1)
axB.axis("off")
panel_title(axB, "b)", r"Per time lag $\Delta t$: sum evidence $X(t)\,\rightarrow\,Y(t{+}\Delta t)$")

# caption sits at the very top of the panel area, well above the matrix titles
axB.text(0.50, 0.99,
         "example transition matrices for lag 40 ms & 90 ms",
         ha="center", va="top", fontsize=8.5 * FONT_SCALAR, color="0.30", style="italic")

# Underlying toy model used by both panel (b) and panel (c)
peak_lag, peak_strength, decay_ms = 40, 0.85, 22
fwd_mask = np.zeros((4, 4), dtype=bool)
fwd_mask[np.arange(3), np.arange(1, 4)] = True
decay_steps = [1.00, 0.92, 0.85]      # within-lag decay across A→B, B→C, C→D


def beta_at(lag_ms, seed=None):
    fwd = peak_strength * np.exp(-((lag_ms - peak_lag) ** 2) / (2 * decay_ms ** 2))
    M = np.zeros((4, 4))
    for k_, w in enumerate(decay_steps):
        M[k_, k_ + 1] = fwd * w
    if seed is not None:
        M = M + 0.04 * np.random.default_rng(seed).standard_normal((4, 4))
    return M


def zf_clean(lag_ms):
    fwd = peak_strength * np.exp(-((lag_ms - peak_lag) ** 2) / (2 * decay_ms ** 2))
    return fwd * sum(decay_steps)


# Two contrasting example lags
lags_to_show = [40, 90]
example_tags = ["high evidence", "no evidence"]

mini_w = 0.30
gap    = 0.16
total_w     = len(lags_to_show) * mini_w + (len(lags_to_show) - 1) * gap
left_start  = (1.0 - total_w) / 2
mini_y      = 0.22
mini_h      = 0.42

z_values = []
for k, (lag, tag) in enumerate(zip(lags_to_show, example_tags)):
    x0   = left_start + k * (mini_w + gap)
    beta = beta_at(lag, seed=int(lag))
    z_value = float(beta[fwd_mask].sum())
    z_values.append(z_value)

    ax_mat = axB.inset_axes([x0, mini_y, mini_w, mini_h])
    ax_mat.imshow(beta, cmap="RdBu_r", vmin=-0.9, vmax=0.9, aspect="equal")

    ax_mat.set_xticks(range(4))
    ax_mat.set_yticks(range(4))
    ax_mat.set_xticklabels(state_names, fontsize=9 * FONT_SCALAR, fontweight="bold")
    ax_mat.set_yticklabels(state_names, fontsize=9 * FONT_SCALAR, fontweight="bold")
    for tlab, col in zip(ax_mat.get_xticklabels(), STATE_COLORS):
        tlab.set_color(col)
    for tlab, col in zip(ax_mat.get_yticklabels(), STATE_COLORS):
        tlab.set_color(col)
    ax_mat.tick_params(axis="both", which="both", length=0)

    ax_mat.set_xlabel(r"state at $t + \Delta t$", fontsize=8.5 * FONT_SCALAR, labelpad=2)
    ax_mat.set_ylabel(r"state at $t$",            fontsize=8.5 * FONT_SCALAR, labelpad=2)

    # red borders around the correct (forward) transition cells
    for (i, j) in zip(np.arange(3), np.arange(1, 4)):
        ax_mat.add_patch(plt.Rectangle(
            (j - 0.5, i - 0.5), 1, 1, fill=False,
            edgecolor=C_FWD, lw=2.0, zorder=10))

    for s in ax_mat.spines.values(): s.set_linewidth(0.6)

    # Two-line subtitle (Δt label + tag), small enough to leave room for
    # the italic caption above.
    ax_mat.set_title(rf"$\Delta t = {lag}$ ms" + f"\n({tag})",
                     fontsize=9 * FONT_SCALAR, color="0.18",
                     fontweight="normal",
                     pad=4.5)

    # Σ value placed clearly below the matrix's xlabel
    axB.text(x0 + mini_w / 2, 0.0,
             rf"$\sum = {z_value:+.2f}$",
             ha="center", va="top", fontsize=11 * FONT_SCALAR,
             color=C_FWD, fontweight="bold")


# ============================================================
# (c) TDLM output — sequenceness vs lag (capped at 150 ms),
# with the eight Σ values from panel (b) overlaid as black dots.
# ============================================================
axC = fig.add_subplot(gs[0, 2])
lags = np.arange(0, 151, 1)
zf = np.array([zf_clean(L) for L in lags])
from scipy.interpolate import make_interp_spline
_knots = np.linspace(0, 150, 12)
_vals = 0.1 * rng.standard_normal(len(_knots))
zb = make_interp_spline(_knots, _vals, k=3)(lags.astype(float))

axC.plot(lags, zf, color=C_FWD, lw=1.8, label=r"forward $Z_F$")
axC.plot(lags, zb, color=C_BWD, lw=1.8, label=r"backward $Z_B$")

# threshold = 1.0
# axC.axhline(threshold, color="0.4", lw=0.8, ls="--")
# axC.text(148, threshold + 0.10, "perm. 95%", color="0.4", fontsize=7.5, ha="right")
axC.axhline(0, color="k", lw=0.4)

# gray dots at every 10-ms lag (the curve is built from these per-lag Σ values),
# with 40 ms and 90 ms emphasised in bold black to match the matrices in panel (b).
all_dot_lags = list(range(10, 151, 10))
for lag in all_dot_lags:
    if lag in lags_to_show:
        continue
    beta_dot = beta_at(lag, seed=int(lag))
    zv_dot = float(beta_dot[fwd_mask].sum())
    axC.scatter(lag, zv_dot, color="0.55", s=22, zorder=5,
                edgecolor="white", lw=0.5)

# bold black dots for the two example lags shown in panel (b)
for lag, zv in zip(lags_to_show, z_values):
    axC.scatter(lag, zv, color="k", s=48, zorder=6,
                edgecolor="white", lw=0.9)

# annotate the typical lag (the 40 ms point)
i_peak = lags_to_show.index(peak_lag)
axC.annotate("replay time lag 40 ms",
             xy=(peak_lag, z_values[i_peak]),
             xytext=(85, 2.5),
             fontsize=8.5 * FONT_SCALAR, color=C_FWD, ha="center",
             arrowprops=dict(arrowstyle="-", color=C_FWD, lw=0.6))

axC.set_xlim(0, 150)
axC.set_ylim(-0.6, 2.85)
axC.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88))
axC.set_xlabel(r"time lag  $\Delta t$  (ms)")
axC.set_ylabel(r"sequenceness")
panel_title(axC, "c)", r"Sequenceness per time lag")


# ============================================================
# SODA: data generation matches 3_SODA_explainer.py
# Each item i is a Gaussian "HRF response" centered at TR 2.5 + 0.15 i,
# σ_TR = 0.5. Five items → five overlapping bell curves shifted in time.
# ============================================================
ITEM_CENTERS = np.array([2.5 + 0.15 * i for i in range(5)])
ITEM_SIGMA   = 0.5
TRS          = np.arange(5)               # integer-TR sample points
SLOPE_TRS    = np.arange(1, 5)            # TRs at which we fit slopes (1..4)


def item_prob(i, t):
    return stats.norm.pdf(t, loc=ITEM_CENTERS[i], scale=ITEM_SIGMA)


# probability of each item (rows) at each integer TR (cols)
PROBS_AT_TR = np.stack([item_prob(i, TRS.astype(float)) for i in range(5)])


def fit_slope(x, y):
    """OLS slope after subtracting min(y) so each fit starts from a zero baseline.
    The slope is invariant to the constant shift, but visually the line on the
    plot has no intercept-like offset above the lowest data point."""
    y_shifted = y - y.min()
    s, _, _, _, _ = stats.linregress(x, y_shifted)
    return float(s)


def slope_at(t):
    """Sign-flipped slope of (probability vs serial position) at time t."""
    pos = np.arange(1, 6).astype(float)
    probs = np.array([item_prob(i, np.atleast_1d(t).astype(float))[0] for i in range(5)])
    return -fit_slope(pos, probs)


# ============================================================
# (d) SODA input — five stacked ribbons (TDLM-style) with very wide,
# heavily overlapping HRF responses
# ============================================================
axD = fig.add_subplot(gs[1, 0])

ribbon_offset = 0.40   # << much smaller than TDLM (a) → ribbons heavily overlap
PEAK = stats.norm.pdf(0, 0, ITEM_SIGMA)  # ≈ 0.798

t_dense = np.linspace(-0.4, 5.4, 600)
for i in range(5):
    p = item_prob(i, t_dense)
    y0 = (4 - i) * ribbon_offset
    axD.fill_between(t_dense, y0, y0 + p, color=ITEM_COLORS[i], alpha=0.32, lw=0)
    axD.plot(t_dense, y0 + p, color=ITEM_COLORS[i], lw=1.15)
    axD.scatter(TRS, y0 + PROBS_AT_TR[i],
                color=ITEM_COLORS[i], s=22, zorder=5,
                edgecolor="white", lw=0.5)
    axD.text(-0.30, y0 + 0.18, chr(65 + i),
             color=ITEM_COLORS[i], fontweight="bold",
             ha="right", va="center", fontsize=11 * FONT_SCALAR)

# integer-TR sampling bands across the full y range
ymax = 4 * ribbon_offset + PEAK + 0.10
for tr in TRS:
    axD.axvspan(tr - 0.05, tr + 0.05, color="0.5", alpha=0.18, lw=0, zorder=0)

axD.set_xlim(-0.5, 5.0)
axD.set_ylim(-0.05, ymax)
axD.set_xticks(TRS)
axD.set_yticks([])
axD.spines["left"].set_visible(False)
axD.set_xlabel("time (TRs), sample rate ~0.8 Hz")
panel_title(axD, "d)", r"Input: decoded states  $X(t)$")


# ============================================================
# (e) SODA mechanism — slope of (prob vs. position) at four TRs
# ============================================================
gsE_outer = GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs[1, 1],
    height_ratios=[0.30, 0.70, 0.15], hspace=0.05,
)
ax_title_e = fig.add_subplot(gsE_outer[0])
ax_title_e.axis("off")
panel_title(ax_title_e, "e)", "Per TR: regress probability on position")

gsE_inner = GridSpecFromSubplotSpec(1, 4, subplot_spec=gsE_outer[1], wspace=0.45)

slopes_at_TR = []
positions = np.arange(1, 6).astype(float)

for k, tr in enumerate(SLOPE_TRS):
    ax = fig.add_subplot(gsE_inner[k])
    probs = PROBS_AT_TR[:, tr]
    probs_shifted = probs - probs.min()    # << "remove min" baselines each TR at 0

    raw_slope, intercept, _, _, _ = stats.linregress(positions, probs_shifted)
    flipped = -raw_slope
    slopes_at_TR.append(flipped)
    line_color = C_FWD if flipped > 0 else C_BWD

    # OLS line on shifted data
    xx = np.array([0.4, 5.6])
    ax.plot(xx, raw_slope * xx + intercept,
            color=line_color, lw=1.4, ls="--", alpha=0.85)

    for i in range(5):
        ax.scatter(positions[i], probs_shifted[i], color=ITEM_COLORS[i],
                   s=34, edgecolor="white", lw=0.6, zorder=5)

    ax.axhline(0, color="0.7", lw=0.4)

    ax.set_xlim(0.4, 5.6)
    ax.set_ylim(-0.03, 0.55)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("position", fontsize=8.5 * FONT_SCALAR)
    ax.set_title(f"TR {tr}", fontsize=9.5 * FONT_SCALAR, pad=3)
    if k == 0:
        ax.set_ylabel(r"prob. $-$ min", fontsize=8.5 * FONT_SCALAR)
    else:
        ax.set_yticklabels([])

    ax.text(0.5, 0.97, rf"$b_{{{tr}}}={flipped:+.2f}$",
            transform=ax.transAxes, fontsize=8.2 * FONT_SCALAR,
            ha="center", va="top", color=line_color, fontweight="bold")


# ============================================================
# (f) SODA output — slope dynamic across all TRs
# ============================================================
axF = fig.add_subplot(gs[1, 2])
t_slope = np.linspace(0, 5, 200)
slope_curve = np.array([slope_at(t) for t in t_slope])

# fill above and below zero with semantic colours
axF.fill_between(t_slope, 0, slope_curve, where=slope_curve > 0,
                 color=C_FWD, alpha=0.20, lw=0)
axF.fill_between(t_slope, 0, slope_curve, where=slope_curve < 0,
                 color=C_BWD, alpha=0.20, lw=0)
axF.plot(t_slope, slope_curve, color="0.15", lw=1.8)
axF.axhline(0, color="k", lw=0.5)

# overlay the four discrete slopes from panel (e); each label sits at a
# distinct clock position around its dot so it doesn't overlap the curve.
TR_LABEL_POS = {
    1: (0.00,  0.014, "center", "bottom"),   # 12 o'clock
    2: (-0.10, 0.010, "right",  "bottom"),   # 11 o'clock
    3: (-0.12, 0.000, "right",  "center"),   # 9 o'clock
    4: (0.10, -0.010, "left",   "top"),      # 4 o'clock
}
for k, tr in enumerate(SLOPE_TRS):
    axF.scatter(tr, slopes_at_TR[k], color="k", s=42, zorder=6,
                edgecolor="white", lw=0.7)
    dx, dy, ha, va = TR_LABEL_POS[tr]
    axF.text(tr + dx, slopes_at_TR[k] + dy,
             f"TR{tr}",
             ha=ha, va=va,
             fontsize=8.5 * FONT_SCALAR, color="0.20")

# directional labels: positive slope = forward, negative slope = backward
axF.text(0.04, 0.92, "onset", transform=axF.transAxes,
         color=C_FWD, fontsize=10 * FONT_SCALAR, fontweight="bold",
         ha="left", va="top")
axF.text(0.04, 0.08, "offset", transform=axF.transAxes,
         color=C_BWD, fontsize=10 * FONT_SCALAR, fontweight="bold",
         ha="left", va="bottom")

axF.set_xlim(0, 5)
axF.set_ylim(-0.18, 0.16)
axF.set_xticks(np.arange(0, 6))
axF.set_xlabel("time (TRs)")
axF.set_ylabel(r"regression slope  $b(t)$")
panel_title(axF, "f)", "Slope dynamic $b(t)$")


# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
fig.savefig("tdlm_soda_methods.svg")
fig.savefig("tdlm_soda_methods.png", dpi=300)
print("Wrote tdlm_soda_methods.svg and tdlm_soda_methods.png")
