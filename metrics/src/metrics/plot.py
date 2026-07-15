import math
import statistics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Literal
from .config import CORES, RESULTS_DIR, Core
from .collect import load
from .metric import Metric, METRICS, IPC, INST, IPC_FETCH, IPC_ISSUE, IPC_RETIREMENT, BPR, SIMPLE_METRICS, \
    I_CACHE_MISSES, I_CACHE_MISS_RATE, PSF_MISPREDICTION_RATE, PSF_MISPREDICTIONS, SOFT_FLUSHES, FLUSHES, FLUSH_RATE, \
    SOFT_FLUSH_RATE
import matplotlib.cm as cm

DisplayMode = Literal["absolute", "relative", "percent"]


def _geomean(vals: list[float]) -> float:
    """Geometric mean of a list of positive values."""
    pos = [v for v in vals if v > 0]
    if not pos:
        return 0.0
    return math.exp(sum(math.log(v) for v in pos) / len(pos))


def _gsd(vals: list[float]) -> float:
    """Geometric standard deviation. Returns a multiplicative factor >= 1."""
    pos = [v for v in vals if v > 0]
    if len(pos) < 2:
        return 1.0
    log_vals = [math.log(v) for v in pos]
    mean_log = sum(log_vals) / len(log_vals)
    variance = sum((x - mean_log) ** 2 for x in log_vals) / (len(log_vals) - 1)
    return math.exp(math.sqrt(variance))


def _fmt(v: float, mode: str, number_type: str = "float") -> str:
    if mode == "percent":
        return f"{v:+.1f}%"
    if mode == "relative":
        return f"{v:.2f}×"

    if number_type == "percentage":
        return f"{v * 100:.2f}%"
    elif number_type == "int":
        return f"{int(round(v))}"
    else:  # float
        return f"{v:.3f}"


def _formatter(mode: str, number_type: str = "float") -> mticker.Formatter:
    if mode == "percent":
        return mticker.FuncFormatter(lambda v, _: f"{v:+.1f}%")
    if mode == "relative":
        return mticker.FuncFormatter(lambda v, _: f"{v:.2f}×")

    if number_type == "percentage":
        return mticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
    elif number_type == "int":
        return mticker.FuncFormatter(lambda v, _: f"{int(round(v))}")
    else:  # float
        return mticker.FuncFormatter(lambda v, _: f"{v:.3f}")


def summarize(raw: dict[str, dict[str, float]], mode: str = "absolute", baseline: Core | None = None) -> dict[
    str, dict[str, float]]:
    def stats(vals):
        gm = _geomean(vals)
        gsd = _gsd(vals)
        return {
            "min": min(vals),
            "median": statistics.median(vals),
            "max": max(vals),
            "geomean": gm,
            "gsd": gsd,
        }

    if mode == "absolute" or baseline is None:
        return {cid: stats(list(scores.values())) for cid, scores in raw.items() if scores}

    base_scores = raw.get(baseline.id, {})

    out = {}
    for cid, scores in raw.items():
        normalized = []
        for benchmark, value in scores.items():
            base_value = base_scores.get(benchmark)
            if base_value is None or base_value == 0:
                continue
            if mode == "relative":
                normalized.append(value / base_value)
            else:  # percent
                normalized.append((value / base_value - 1) * 100)
        if normalized:
            out[cid] = stats(normalized)

    return out


def summarize_stacked(raws: list[dict[str, dict[str, float]]],
                      metrics: list[Metric]) -> dict[str, dict[str, dict[str, float]]]:
    """Returns {stat: {core_id: [per_metric_value]}} where all metrics come from the same benchmark.

    'geomean' entry holds per-metric geomeans across benchmarks.
    'gsd' entry holds per-metric GSDs (multiplicative factor).
    """

    core_ids = list(raws[0].keys())

    out = {stat: {cid: [] for cid in core_ids} for stat in ("min", "median", "max", "geomean", "gsd")}

    for cid in core_ids:
        # Get benchmarks present in all lanes
        benchmarks = list(raws[0][cid].keys())
        for r in raws[1:]:
            benchmarks = [b for b in benchmarks if b in r.get(cid, {})]

        # Compute per-benchmark sum across lanes
        sums = {b: sum(r[cid][b] for r in raws) for b in benchmarks}

        if not sums:
            continue

        vals = list(sums.values())
        min_bench = min(sums, key=sums.get)
        max_bench = max(sums, key=sums.get)
        median_val = statistics.median(vals)
        median_bench = min(sums, key=lambda b: abs(sums[b] - median_val))

        for stat, bench in [("min", min_bench), ("median", median_bench), ("max", max_bench)]:
            out[stat][cid] = [raws[i][cid][bench] for i in range(len(metrics))]

        # Geomean and GSD: computed per-metric lane across all benchmarks
        for i in range(len(metrics)):
            lane_vals = [raws[i][cid][b] for b in benchmarks if raws[i][cid].get(b, 0) > 0]
            out["geomean"][cid].append(_geomean(lane_vals) if lane_vals else 0.0)
            out["gsd"][cid].append(_gsd(lane_vals) if len(lane_vals) >= 2 else 1.0)

    return out


def _annotate(ax, bars, mode, number_type):
    for bar in bars:
        v = bar.get_height()
        nudge = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + nudge if v >= 0 else v - nudge,
                _fmt(v, mode, number_type),
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=7, color="#555")


def _y_label(metric_name: str, mode: str) -> str:
    return {
        "absolute": metric_name,
        "relative": f"{metric_name} (relative to baseline)",
        "percent": f"{metric_name} (% over baseline)",
    }[mode]


def plot(metric: Metric, cores: list[Core] = CORES, path: Path = RESULTS_DIR / "metrics", save_path=None,
         mode: DisplayMode = "absolute", baseline: Core | None = None) -> None:
    raw = load(cores, metric, path)
    data = summarize(raw, mode=mode, baseline=baseline)

    core_order = [c.id for c in cores if c.id in data]
    core_labels = [c.name for c in cores if c.id in data]

    mins = np.array([data[cid]["min"] for cid in core_order])
    medians = np.array([data[cid]["median"] for cid in core_order])
    maxs = np.array([data[cid]["max"] for cid in core_order])
    geomeans = np.array([data[cid]["geomean"] for cid in core_order])
    gsds = np.array([data[cid]["gsd"] for cid in core_order])

    # GSD error bars: [geomean/gsd, geomean*gsd] — asymmetric in linear space
    gm_err_lo = geomeans - geomeans / gsds
    gm_err_hi = geomeans * gsds - geomeans
    gm_yerr = np.array([gm_err_lo, gm_err_hi])

    n = len(core_order)
    x = np.arange(n)
    width = 0.2  # narrower to fit 4 bars

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_min = ax.bar(x - 1.5 * width, mins, width, label="Min.", color="#85B7EB")
    bars_med = ax.bar(x - 0.5 * width, medians, width, label="Median", color="#1D9E75")
    bars_max = ax.bar(x + 0.5 * width, maxs, width, label="Max.", color="#D85A30")
    bars_gm = ax.bar(x + 1.5 * width, geomeans, width, label="Geometric Mean", color="#8B5CF6")
    ax.errorbar(x + 1.5 * width, geomeans, yerr=gm_yerr,
                fmt="none", color="#4C1D95", linewidth=1.5, capsize=4, capthick=1.5,
                label="Geometric Mean ±1 Geometric SD")

    ax.set_xticks(x)
    ax.set_xticklabels(core_labels, rotation=15, ha="right")
    ax.set_ylabel(_y_label(metric.name, mode))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(_formatter(mode, metric.number_type))
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)

    if mode != "absolute":
        ax.axhline(0 if mode == "percent" else 1,
                   color="0.6", linewidth=0.8, linestyle="--")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0), frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    _annotate(ax, bars_min, mode, metric.number_type)
    _annotate(ax, bars_med, mode, metric.number_type)
    _annotate(ax, bars_max, mode, metric.number_type)
    _annotate(ax, bars_gm, mode, metric.number_type)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_stacked(metrics: list[Metric], cores: list[Core] = CORES, path: Path = RESULTS_DIR / "metrics", save_path=None,
                 mode: DisplayMode = "absolute", baseline: Core | None = None) -> None:
    raws = [load(cores, m, path) for m in metrics]

    core_order = [c.id for c in cores if c.id in raws[0]]
    core_labels = [c.name for c in cores if c.id in raws[0]]

    # Assume all stacked metrics share the same number type (e.g., floats for IPC)
    primary_number_type = metrics[0].number_type

    n = len(core_order)
    x = np.arange(n)
    width = 0.2  # narrower to fit 4 bars

    stat_colors = {
        "min": ["#185FA5", "#378ADD", "#85B7EB", "#D4E8F7"],
        "median": ["#0F6E56", "#1D9E75", "#5DCAA5", "#9FE1CB"],
        "max": ["#993C1D", "#D85A30", "#F0997B", "#F5C4B3"],
        "geomean": ["#4C1D95", "#7C3AED", "#A78BFA", "#DDD6FE"],
    }

    data = summarize_stacked(raws, metrics)

    fig, ax = plt.subplots(figsize=(10, 5))

    for stat, offset in [("min", -1.5 * width), ("median", -0.5 * width), ("max", 0.5 * width),
                         ("geomean", 1.5 * width)]:
        if stat == "geomean":
            # For geomean, plot as a single bar without splitting
            vals = np.array([sum(data[stat][cid]) if data[stat].get(cid) else 0.0 for cid in core_order])
            ax.bar(x + offset, vals, width, color=stat_colors["geomean"][0], label="Geomean")

            # Annotate total on top
            for xi, val in zip(x, vals):
                nudge = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
                ax.text(xi + offset, val + nudge, _fmt(val, mode, primary_number_type),
                        ha="center", va="bottom", fontsize=7, color="#555")
        else:
            bottom = np.zeros(n)
            for i, m in enumerate(metrics):
                vals = np.array([data[stat][cid][i] if data[stat][cid] else 0.0 for cid in core_order])
                label = m.name if stat == "min" else None
                ax.bar(x + offset, vals, width, bottom=bottom,
                       color=stat_colors[stat][i], label=label)
                bottom += vals

            # Annotate total on top of each stacked bar
            for xi, total in zip(x, bottom):
                nudge = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
                ax.text(xi + offset, total + nudge, _fmt(total, mode, primary_number_type),
                        ha="center", va="bottom", fontsize=7, color="#555")

    # Add GSD error bars on geomean bars (based on total geomean sum across lanes)
    gm_totals = np.array([
        sum(data["geomean"][cid]) if data["geomean"].get(cid) else 0.0
        for cid in core_order
    ])
    # Use the GSD of the first lane as a representative spread indicator
    gsds = np.array([
        data["gsd"][cid][0] if (data["gsd"].get(cid) and len(data["gsd"][cid]) > 0) else 1.0
        for cid in core_order
    ])
    gm_err_lo = gm_totals - gm_totals / gsds
    gm_err_hi = gm_totals * gsds - gm_totals
    ax.errorbar(x + 1.5 * width, gm_totals, yerr=np.array([gm_err_lo, gm_err_hi]),
                fmt="none", color="#2D0A66", linewidth=1.5, capsize=4, capthick=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(core_labels, rotation=15, ha="right")
    ax.set_ylabel("IPC")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(_formatter(mode, primary_number_type))
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    lane_handles = [Patch(color=stat_colors["median"][i], label=m.name) for i, m in enumerate(metrics)]
    stat_handles = [Patch(color=stat_colors["min"][0], label="Min."),
                    Patch(color=stat_colors["median"][0], label="Median"),
                    Patch(color=stat_colors["max"][0], label="Max."),
                    Patch(color=stat_colors["geomean"][0], label="Geomean (±1 GSD)")]

    all_handles = lane_handles + stat_handles
    fig.legend(handles=all_handles, loc="lower center", ncol=len(all_handles),
               bbox_to_anchor=(0.5, 0), frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_improvement(core1: Core, core2: Core, metric: Metric, path: Path = RESULTS_DIR / "metrics",
                     save_path=None) -> None:
    # Load raw data for the two specified cores
    raw = load([core1, core2], metric, path)

    # Extract overlapping benchmarks present in both cores
    data1 = raw.get(core1.id, {})
    data2 = raw.get(core2.id, {})
    benchmarks = [b for b in data1.keys() if b in data2]
    benchmarks.sort()

    if not benchmarks:
        print(f"No common benchmarks found for {core1.name} and {core2.name}.")
        return

    # Extract values
    val1s = np.array([data1[b] for b in benchmarks])
    val2s = np.array([data2[b] for b in benchmarks])

    # Calculate absolute improvements and percentage
    improvements = val2s - val1s
    pct_improvements = np.zeros_like(improvements, dtype=float)
    mask = val1s != 0
    pct_improvements[mask] = (improvements[mask] / val1s[mask]) * 100

    x = np.arange(len(benchmarks))
    width = 0.75  # Increased width to fit the bars closer together
    fig, ax = plt.subplots(figsize=(max(7, len(benchmarks) * 0.45), 5))  # More compact width mapping

    # Base bar for the first core
    bars_base = ax.bar(x, val1s, width, label=core1.name, color="#85B7EB")

    # Stacked bars for improvements (Positive in Green, Negative in Red)
    pos_improvements = np.maximum(improvements, 0)
    neg_improvements = np.minimum(improvements, 0)

    # Only add bars and legend entries if the event actually occurs
    if np.any(pos_improvements > 0):
        ax.bar(x, pos_improvements, width, bottom=val1s, label=f"{core2.name} Improvement", color="#1D9E75")

    if np.any(neg_improvements < 0):
        ax.bar(x, neg_improvements, width, bottom=val1s, label=f"{core2.name} Degradation", color="#D85A30")

    # Annotate with percentage improvement
    for i, pct in enumerate(pct_improvements):
        if val1s[i] == 0:
            continue

        nudge = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
        if pct > 0.1:  # Display positive improvements
            y_pos = val1s[i] + pos_improvements[i] + nudge
            ax.text(x[i], y_pos, f"+{pct:.1f}%", ha="center", va="bottom", fontsize=8, color="#0F6E56",
                    fontweight="bold")
        elif pct < -0.1:  # Display degradations
            y_pos = val1s[i] + neg_improvements[i] - nudge
            ax.text(x[i], y_pos, f"{pct:.1f}%", ha="center", va="top", fontsize=8, color="#993C1D", fontweight="bold")
        else:  # Basically equal
            y_pos = val1s[i] + nudge
            ax.text(x[i], y_pos, f"~0%", ha="center", va="bottom", fontsize=8, color="#555")

    # Formatting axes and legends
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=30, ha="right")
    ax.set_ylabel(metric.name)
    ax.set_title(f"{metric.name}: {core1.name} vs {core2.name}")

    # Clean up duplicate labels in legend if any exist
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(_formatter("absolute", metric.number_type))
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_single(metric: Metric, cores: list[Core] = CORES, path: Path = RESULTS_DIR / "metrics", save_path=None,
                mode: DisplayMode = "absolute", baseline: Core | None = None) -> None:
    """Plots a metric that has a single 'score' per core rather than per-benchmark values."""
    raw = load(cores, metric, path)

    # Extract the single "score" value and handle normalization if a baseline/mode is specified
    data = {}

    if mode == "absolute" or baseline is None:
        data = {cid: scores.get("score", 0.0) for cid, scores in raw.items() if scores}
    else:
        base_val = raw.get(baseline.id, {}).get("score", 0.0)
        for cid, scores in raw.items():
            if not scores:
                continue
            val = scores.get("score", 0.0)
            if base_val == 0:
                data[cid] = 0.0  # Avoid division by zero
            elif mode == "relative":
                data[cid] = val / base_val
            else:  # percent
                data[cid] = (val / base_val - 1) * 100

    core_order = [c.id for c in cores if c.id in data]
    core_labels = [c.name for c in cores if c.id in data]
    vals = np.array([data[cid] for cid in core_order])

    n = len(core_order)
    x = np.arange(n)
    width = 0.5  # Wider bar since it's just one per core

    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), 5))

    # Single bar plot for the cores
    bars = ax.bar(x, vals, width, color="#85B7EB")

    # Formatting axes
    ax.set_xticks(x)
    ax.set_xticklabels(core_labels, rotation=15, ha="right")
    ax.set_ylabel(_y_label(metric.name, mode))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(_formatter(mode, metric.number_type))
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)

    # Add a horizontal line at the baseline level if doing relative/percent comparisons
    if mode != "absolute":
        ax.axhline(0 if mode == "percent" else 1,
                   color="0.6", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    # Annotate the values on top of the bars
    _annotate(ax, bars, mode, metric.number_type)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()



def main():
    for metric in SIMPLE_METRICS:
        plot(metric, cores=CORES,
              save_path=f"plots/{metric.id}"
            )

    # plot_single(Metric("hardware_size", "Hardware size (mm²)", None, "float"), cores=CORES[1:], save_path=f"../../writing/report/figures/plots/hardware_size")

    # plot_single(Metric("critical_path", "Critical Path (ps)", None, "int"), cores=CORES[1:], save_path=f"../../writing/report/figures/plots/critical_path")

    # plot(PSF_MISPREDICTIONS)
    # plot(PSF_MISPREDICTION_RATE)
    # plot(IPC)

    # plot(FLUSHES)
    # plot(SOFT_FLUSHES)

    # plot(FLUSH_RATE)
    # plot(SOFT_FLUSH_RATE)
    # plot_improvement(
    #     core1=CORES[1],
    #     core2=CORES[-1],
    #     metric=IPC,
    #     save_path=f"../../writing/report/figures/plots/ipc_improvement"
    # )

    # plot_stacked(IPC_FETCH, cores=CORES[1:], save_path=f"../../writing/report/figures/plots/ipc_fetch")
    # plot_stacked(IPC_ISSUE, cores=CORES[1:], save_path=f"../../writing/report/figures/plots/ipc_issue")
    # plot_stacked(IPC_RETIREMENT, cores=CORES[1:], save_path=f"../../writing/report/figures/plots/ipc_retirement")

if __name__ == "__main__":
    main()