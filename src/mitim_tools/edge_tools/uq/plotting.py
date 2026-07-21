"""
edge_tools.uq.plotting
----------------------
Visualize the per-source uncertainty breakdown (``run_edge_uq`` ->
``source_breakdown``, or ``powerstate._edge_uq_breakdown``).

Correctness rule: uncertainties add in QUADRATURE, so only VARIANCES are
additive.  Every stacked/heatmap value here is a **variance** (or % of variance);
the absolute total is reported as a **std** label.  Stacking std would overstate
the total and mislead.

Two views:
  * ``plot_breakdown_bars``   -- one 100%-stacked variance bar per (quantity,
    rhoCP), total-sigma annotated on top.  Shows composition and where the
    dominant source shifts across the flux-match control points.
  * ``plot_breakdown_heatmap``-- sources x quantities, cell = % of variance.
    Compact overview across many quantities.

Colors: fixed source->color registry (Okabe-Ito, a published CVD-safe palette),
so a source keeps its color across every bar, panel, and figure.
"""

import numpy as np

# Okabe-Ito CVD-safe categorical hues, assigned to sources in a STABLE registry
# (a source always gets the same color).  Secondary/model sources use grays.
SOURCE_COLORS = {
    "LCFS ne":       "#0072B2",  # blue
    "LCFS te":       "#D55E00",  # vermillion
    "LCFS ti":       "#E69F00",  # orange
    "LCFS aLne":     "#56B4E9",  # sky blue
    "LCFS aLte":     "#CC79A7",  # reddish purple
    "LCFS aLti":     "#F0E442",  # yellow
    "aLy(PeretSSF)": "#009E73",  # bluish green
    "impurity D/V":  "#8B4513",  # sienna
    "source rate":   "#7F7F7F",  # gray
    "vtor":          "#000000",  # black
    "mtanh-fit":     "#BFBFBF",  # light gray
}
_FALLBACK = ["#4C4C4C", "#B0B0B0", "#7B68EE", "#2CA02C"]


def _color(source, i):
    return SOURCE_COLORS.get(source, _FALLBACK[i % len(_FALLBACK)])


def _rhocp_indices(powerstate, breakdown_key_shape_len):
    """Fine-grid indices nearest each rhoCP control point (or None -> use all)."""
    if powerstate is None or not hasattr(powerstate, "rhoCP"):
        return None, None
    rho = powerstate.plasma["rho"][0].detach().cpu().numpy()
    idx = [int(np.argmin(np.abs(rho - float(r)))) for r in powerstate.rhoCP]
    labels = [f"{float(r):.3g}" for r in powerstate.rhoCP]
    return idx, labels


def _as_np(v):
    return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)


def _sources_at(bd, idx):
    """{source: variance} at a single radial index, sorted by contribution desc."""
    out = {}
    for s, v in bd.items():
        if s == "__total__":
            continue
        arr = _as_np(v).reshape(-1)
        out[s] = float(arr[idx]) ** 2 if idx < arr.size else 0.0
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def plot_breakdown_bars(source_breakdown, keys, powerstate=None, ax=None,
                        normalize=True, title=None):
    """
    One 100%-stacked variance bar per (quantity, rhoCP).  Total sigma annotated.

    keys : list of observable keys to show (e.g. transport fluxes).
    normalize : True -> each bar is % of variance (composition); False -> absolute
                variance (comparable only within same-scale quantities).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.6 * len(keys) + 2.0, 4.2))
    else:
        fig = ax.figure

    # collect the ordered set of sources present (for a stable legend/color order)
    present = []
    for k in keys:
        for s in (source_breakdown.get(k) or {}):
            if s != "__total__" and s not in present:
                present.append(s)
    order = [s for s in SOURCE_COLORS if s in present] + \
            [s for s in present if s not in SOURCE_COLORS]

    xticks, xlabels, group_centers, group_names = [], [], [], []
    x = 0.0
    bar_w = 0.8
    for k in keys:
        bd = source_breakdown.get(k)
        if bd is None:
            continue
        idxs, cplabels = _rhocp_indices(powerstate, 1)
        tot_arr = _as_np(bd["__total__"]).reshape(-1)
        if idxs is None:
            idxs = list(range(tot_arr.size))
            cplabels = [str(i) for i in idxs]
        x0 = x
        for ii, idx in enumerate(idxs):
            var = _sources_at(bd, idx)
            total_var = sum(var.values())
            denom = total_var if (normalize and total_var > 0) else 1.0
            bottom = 0.0
            for si, s in enumerate(order):
                h = var.get(s, 0.0) / denom
                if h <= 0:
                    continue
                ax.bar(x, h, bar_w, bottom=bottom, color=_color(s, si),
                       edgecolor="white", linewidth=0.6,
                       label=s if (x == x0 or s not in _legend_seen(ax)) else None)
                bottom += h
            # total-sigma annotation on top
            sig = float(np.sqrt(total_var))
            ax.text(x, (1.02 if normalize else bottom * 1.02),
                    f"σ={sig:.2g}", ha="center", va="bottom",
                    fontsize=7, color="#333333", rotation=0)
            xticks.append(x); xlabels.append(cplabels[ii])
            x += 1.0
        group_centers.append((x0 + x - 1.0) / 2.0)
        group_names.append(_short(k))
        x += 0.7  # gap between quantities

    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=7, color="#555")
    for gc, gn in zip(group_centers, group_names):
        ax.text(gc, -0.14, gn, ha="center", va="top", fontsize=9,
                fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_ylabel("fraction of variance" if normalize else "variance")
    if normalize:
        ax.set_ylim(0, 1.15); ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_title(title or "Uncertainty source breakdown (variance) by rhoCP",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", lw=0.8)
    ax.set_axisbelow(True)
    # de-duplicated legend
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), fontsize=8, ncol=1,
              loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False)
    fig.tight_layout()
    return fig, ax


def _legend_seen(ax):
    return set(ax.get_legend_handles_labels()[1])


def source_order_for(source_breakdown, keys):
    """Stable source ordering (registry order first) across a set of keys."""
    present = []
    for k in keys:
        for s in (source_breakdown.get(k) or {}):
            if s != "__total__" and s not in present:
                present.append(s)
    return [s for s in SOURCE_COLORS if s in present] + \
           [s for s in present if s not in SOURCE_COLORS]


def stacked_breakdown_axes(ax, bd, idxs, cplabels, source_order,
                           normalize=True, ylabel=None, sigma_labels=True):
    """
    Plot ONE quantity's per-control-point 100%-stacked VARIANCE bars into ``ax``
    (for embedding beneath a flux panel).  ``idxs`` are the fine-grid indices of
    the control points, ``cplabels`` their tick labels.  Returns {label: handle}.
    """
    handles = {}
    if bd is None:
        ax.set_axis_off()
        return handles
    for xi, idx in enumerate(idxs):
        var = _sources_at(bd, idx)
        total = sum(var.values())
        denom = total if (normalize and total > 0) else 1.0
        bottom = 0.0
        for si, s in enumerate(source_order):
            h = var.get(s, 0.0) / denom
            if h <= 0:
                continue
            b = ax.bar(xi, h, 0.82, bottom=bottom, color=_color(s, si),
                       edgecolor="white", linewidth=0.5)
            handles.setdefault(s, b)
            bottom += h
        if sigma_labels:
            ax.text(xi, (1.02 if normalize else bottom * 1.02),
                    f"{np.sqrt(total):.1g}", ha="center", va="bottom",
                    fontsize=6, color="#555")
    ax.set_xticks(range(len(idxs)))
    ax.set_xticklabels(cplabels, fontsize=6.5, color="#666")
    if normalize:
        ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", ".5", "1"], fontsize=6.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2)
    ax.margins(x=0.05)
    return handles


def plot_breakdown_heatmap(source_breakdown, keys, powerstate=None, ax=None,
                           rhocp=None, title=None):
    """
    Heatmap: rows = sources, cols = quantities, cell = % of variance.

    rhocp : which control point (int index into rhoCP) to show; None -> RMS over
            all radial points.
    """
    import matplotlib.pyplot as plt

    present = []
    for k in keys:
        for s in (source_breakdown.get(k) or {}):
            if s != "__total__" and s not in present:
                present.append(s)
    rows = [s for s in SOURCE_COLORS if s in present] + \
           [s for s in present if s not in SOURCE_COLORS]

    M = np.full((len(rows), len(keys)), np.nan)
    for cj, k in enumerate(keys):
        bd = source_breakdown.get(k)
        if bd is None:
            continue
        tot = _as_np(bd["__total__"]).reshape(-1)
        for ri, s in enumerate(rows):
            v = _as_np(bd[s]).reshape(-1) if s in bd else None
            if v is None:
                continue
            if rhocp is not None and hasattr(powerstate, "rhoCP"):
                idxs, _ = _rhocp_indices(powerstate, 1)
                i = idxs[rhocp]
                frac = (v[i] ** 2) / max(tot[i] ** 2, 1e-30)
            else:
                frac = (v ** 2).sum() / max((tot ** 2).sum(), 1e-30)
            M[ri, cj] = 100.0 * frac

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.1 * len(keys) + 2.5, 0.5 * len(rows) + 1.5))
    else:
        fig = ax.figure
    # sequential single-hue (magnitude) -- light->dark blue
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([_short(k) for k in keys],
                                                        rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    for ri in range(len(rows)):
        for cj in range(len(keys)):
            if not np.isnan(M[ri, cj]) and M[ri, cj] >= 3:
                ax.text(cj, ri, f"{M[ri,cj]:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if M[ri, cj] > 55 else "#333")
    ax.set_title(title or ("% of variance" +
                 (f" @ rhoCP[{rhocp}]" if rhocp is not None else " (RMS over ρ)")),
                 fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="% variance")
    fig.tight_layout()
    return fig, ax


def _short(key):
    """Compact axis label for a plasma flux/observable key."""
    return (key.replace("1E20m2", "").replace("MWm2", "").replace("Jm2", "")
            .replace("_tr_turb", " turb").replace("_tr_neoc", " neoc")
            .replace("_tr", " tr"))
