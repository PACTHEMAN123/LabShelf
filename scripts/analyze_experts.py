#!/usr/bin/env python3
"""Expert Activation Analysis Pipeline
====================================
Produces per-layer analysis folders with:
1. Heatmap of expert activation over time/step
2. Stacked area chart of expert activation over time/step
3. EPLB algorithm comparison — imbalance ratio with threshold-triggered re-placement

Each analysis: 2 x-axis modes (time, step) × 3 filter modes (prefill, decode, mix) = 6 figures

自动生成 by labshelf.py add-script
"""
import json
import sys
import os
import re
import ast
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────
IMBALANCE_THRESHOLDS = [1.5, 2.0]  # 比较不同阈值
HEATMAP_FREQ_CAP_PERCENTILE = 97  # clamp heatmap color at this percentile for visibility
STEP_RANGE = (1000, 1100)  # (m, n) 只取第 m~n 条记录，None 表示全部。例: STEP_RANGE = (100, 500)

# ──────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────
def parse_log_file(filepath):
    """Parse a layer log file. Returns list of dicts with keys: counts, time, is_decode, step."""
    records = []
    with open(filepath, "r") as f:
        for step_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Parse: [c0, c1, ...], time: <float>, is_decode: <bool>
            # Find the list part
            bracket_end = line.index("]") + 1
            counts_str = line[:bracket_end]
            rest = line[bracket_end:]

            counts = ast.literal_eval(counts_str)

            # Parse time
            time_match = re.search(r"time:\s*([\d.]+)", rest)
            time_val = float(time_match.group(1)) if time_match else step_idx

            # Parse is_decode
            decode_match = re.search(r"is_decode:\s*(True|False)", rest)
            is_decode = decode_match.group(1) == "True" if decode_match else True

            records.append({
                "counts": np.array(counts, dtype=np.float64),
                "time": time_val,
                "is_decode": is_decode,
                "step": step_idx,
            })
    return records


def filter_records(records, mode):
    """Filter records by mode: 'decode', 'prefill', 'mix'."""
    if mode == "decode":
        return [r for r in records if r["is_decode"]]
    elif mode == "prefill":
        return [r for r in records if not r["is_decode"]]
    else:  # mix
        return records


# ──────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────
def compute_imbalance_ratio(counts):
    """Imbalance ratio = max(counts) / mean(counts). Higher = more imbalanced."""
    mean_val = counts.mean()
    if mean_val == 0:
        return 0.0
    return counts.max() / mean_val



# ──────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────
def get_x_values(filtered_records, x_mode):
    """Get x-axis values based on mode."""
    if x_mode == "time":
        vals = np.array([r["time"] for r in filtered_records])
        # Normalize to relative time
        if len(vals) > 0:
            vals = vals - vals[0]
        return vals, "Relative Time (s)"
    else:
        return np.array([r["step"] for r in filtered_records]), "Step"


def make_title(base, x_mode, filter_mode, layer_idx):
    mode_label = {"decode": "Decode Only", "prefill": "Prefill Only", "mix": "All (Mixed)"}
    x_label = {"time": "vs Time", "step": "vs Step"}
    return f"Layer {layer_idx} | {base} {x_label[x_mode]} | {mode_label[filter_mode]}"


def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────
# Plot 1: Heatmap
# ──────────────────────────────────────────────────────────
def plot_heatmap(filtered_records, x_mode, filter_mode, layer_idx, output_dir):
    if len(filtered_records) < 2:
        return

    x_vals, x_label = get_x_values(filtered_records, x_mode)
    num_experts = len(filtered_records[0]["counts"])

    # Build matrix: rows=experts, cols=timesteps (raw counts)
    matrix = np.zeros((num_experts, len(filtered_records)))
    for i, r in enumerate(filtered_records):
        matrix[:, i] = r["counts"]

    # Clamp at percentile for better visibility
    nonzero = matrix[matrix > 0]
    if len(nonzero) > 0:
        vmax = np.percentile(nonzero, HEATMAP_FREQ_CAP_PERCENTILE)
    else:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(16, 10))

    # Use extent to map to actual x values
    extent = [x_vals[0], x_vals[-1], num_experts - 0.5, -0.5]
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", extent=extent,
                   interpolation="nearest", vmin=0, vmax=vmax)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Expert ID", fontsize=12)
    ax.set_title(make_title("Expert Activation Heatmap", x_mode, filter_mode, layer_idx),
                 fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Token Count", fontsize=11)

    fname = f"heatmap_{x_mode}_{filter_mode}.png"
    savefig(fig, os.path.join(output_dir, fname))


# ──────────────────────────────────────────────────────────
# Plot 2: Stacked area chart
# ──────────────────────────────────────────────────────────
def plot_stacked(filtered_records, x_mode, filter_mode, layer_idx, output_dir):
    if len(filtered_records) < 2:
        return

    x_vals, x_label = get_x_values(filtered_records, x_mode)
    num_experts = len(filtered_records[0]["counts"])

    # Build matrix
    matrix = np.zeros((num_experts, len(filtered_records)))
    for i, r in enumerate(filtered_records):
        total = r["counts"].sum()
        if total > 0:
            matrix[:, i] = r["counts"] / total
        else:
            matrix[:, i] = r["counts"]

    fig, ax = plt.subplots(figsize=(16, 8))

    # Stack all experts in ID order
    stack_data = matrix  # shape: (num_experts, timesteps)
    labels = [f"Expert {i}" for i in range(num_experts)]

    # Color map — golden ratio spacing for max adjacent contrast, muted tones
    golden_ratio = 0.618033988749895
    colors = []
    for i in range(num_experts):
        h = (i * golden_ratio) % 1.0
        # HSV → muted: lower saturation + moderate value
        rgb = mcolors.hsv_to_rgb([h, 0.55, 0.85])
        colors.append(rgb)

    ax.stackplot(x_vals, stack_data, labels=labels, colors=colors, alpha=0.85)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Activation Fraction", fontsize=12)
    ax.set_title(make_title("Expert Activation Stacked Area", x_mode, filter_mode, layer_idx),
                 fontsize=14, fontweight="bold")

    # Only show legend when expert count is manageable
    if num_experts <= 20:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, ncol=1)
    else:
        # Add colorbar-style reference instead of huge legend
        listed_cmap = mcolors.ListedColormap(colors)
        sm = plt.cm.ScalarMappable(cmap=listed_cmap, norm=plt.Normalize(0, num_experts - 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Expert ID", fontsize=11)

    fname = f"stacked_{x_mode}_{filter_mode}.png"
    savefig(fig, os.path.join(output_dir, fname))


# ──────────────────────────────────────────────────────────
# EPLB algorithms
# ──────────────────────────────────────────────────────────
def eplb_greedy(counts, num_groups=None):
    """Greedy EPLB: assign experts to least-loaded group."""
    n = len(counts)
    if num_groups is None:
        num_groups = max(1, int(np.sqrt(n)))
    sorted_indices = np.argsort(-counts)
    group_loads = np.zeros(num_groups)
    assignment = np.zeros(n, dtype=int)
    for expert_idx in sorted_indices:
        min_group = np.argmin(group_loads)
        assignment[expert_idx] = min_group
        group_loads[min_group] += counts[expert_idx]
    permutation = np.zeros(n, dtype=int)
    pos = 0
    for g in range(num_groups):
        experts_in_group = np.where(assignment == g)[0]
        experts_in_group = experts_in_group[np.argsort(-counts[experts_in_group])]
        for e in experts_in_group:
            permutation[e] = pos
            pos += 1
    return permutation


def eplb_balanced_partition(counts, num_groups=None):
    """Balanced partition EPLB: snake ordering for balance."""
    n = len(counts)
    if num_groups is None:
        num_groups = max(1, int(np.sqrt(n)))
    sorted_indices = np.argsort(-counts)
    permutation = np.zeros(n, dtype=int)
    groups = [[] for _ in range(num_groups)]
    for i, expert_idx in enumerate(sorted_indices):
        group_id = i % num_groups
        row = i // num_groups
        if row % 2 == 1:
            group_id = num_groups - 1 - group_id
        groups[group_id].append(expert_idx)
    pos = 0
    for g in range(num_groups):
        for e in groups[g]:
            permutation[e] = pos
            pos += 1
    return permutation


def eplb_frequency_aware(counts, num_groups=None):
    """Frequency-aware EPLB: same as greedy but intended for EMA-smoothed counts."""
    return eplb_greedy(counts, num_groups)


EPLB_ALGORITHMS = {
    "Greedy": eplb_greedy,
    "BalancedPartition": eplb_balanced_partition,
    "FrequencyAware": eplb_frequency_aware,
}


def compute_placement_quality(counts, permutation, num_groups=None):
    """max_group_load / mean_group_load. Lower = better."""
    n = len(counts)
    if num_groups is None:
        num_groups = max(1, int(np.sqrt(n)))
    group_loads = np.zeros(num_groups)
    group_size = (n + num_groups - 1) // num_groups
    for expert_idx in range(n):
        group_id = min(permutation[expert_idx] // group_size, num_groups - 1)
        group_loads[group_id] += counts[expert_idx]
    mean_load = group_loads.mean()
    if mean_load == 0:
        return 1.0
    return group_loads.max() / mean_load


# ──────────────────────────────────────────────────────────
# Plot 3: Imbalance ratio — compare EPLB algorithms
# ──────────────────────────────────────────────────────────
def plot_imbalance_compare(filtered_records, x_mode, filter_mode, layer_idx, output_dir,
                           threshold):
    if len(filtered_records) < 2:
        return

    x_vals, x_label = get_x_values(filtered_records, x_mode)
    n_experts = len(filtered_records[0]["counts"])

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]},
                             sharex=True)
    ax_main, ax_events = axes

    # Raw imbalance (no EPLB)
    raw_imbalance = np.array([compute_imbalance_ratio(r["counts"]) for r in filtered_records])
    ax_main.plot(x_vals, raw_imbalance, label="No EPLB", color="#999999",
                 linestyle="--", linewidth=1.2, alpha=0.7)

    # Per-algorithm: simulate threshold-triggered EPLB
    algo_colors = {
        "Greedy": "#e74c3c",
        "BalancedPartition": "#3498db",
        "FrequencyAware": "#2ecc71",
    }
    algo_styles = {
        "Greedy": "-",
        "BalancedPartition": "-",
        "FrequencyAware": "-.",
    }

    all_trigger_x = {}  # algo_name -> list of trigger x values

    for algo_name, algo_fn in EPLB_ALGORITHMS.items():
        current_perm = np.arange(n_experts)
        adjusted = []
        triggers_x = []
        ema_counts = None
        ema_alpha = 0.3

        for i, r in enumerate(filtered_records):
            counts = r["counts"]

            # Update EMA for frequency-aware
            if ema_counts is None:
                ema_counts = counts.copy()
            else:
                ema_counts = ema_alpha * counts + (1 - ema_alpha) * ema_counts

            adj_ir = compute_placement_quality(counts, current_perm)

            if adj_ir > threshold:
                input_counts = ema_counts if algo_name == "FrequencyAware" else counts
                current_perm = algo_fn(input_counts)
                adj_ir = compute_placement_quality(counts, current_perm)
                triggers_x.append(x_vals[i])

            adjusted.append(adj_ir)

        all_trigger_x[algo_name] = triggers_x
        adjusted = np.array(adjusted)

        ax_main.plot(x_vals, adjusted, label=f"{algo_name} ({len(triggers_x)} triggers)",
                     color=algo_colors.get(algo_name, "black"),
                     linestyle=algo_styles.get(algo_name, "-"),
                     linewidth=1.8, alpha=0.9)

    ax_main.axhline(y=threshold, color="#e67e22", linestyle=":", linewidth=1.5,
                    label=f"Threshold = {threshold:.2f}")

    ax_main.set_ylabel("Imbalance Ratio (group max / group mean)", fontsize=11)
    ax_main.set_title(
        make_title(f"EPLB Algorithm Comparison (threshold={threshold:.2f})",
                   x_mode, filter_mode, layer_idx),
        fontsize=14, fontweight="bold")
    ax_main.legend(fontsize=10, loc="upper right")
    ax_main.grid(True, alpha=0.3)

    # Bottom: trigger events per algorithm
    y_positions = {name: i for i, name in enumerate(EPLB_ALGORITHMS)}
    for algo_name, triggers_x in all_trigger_x.items():
        y = y_positions[algo_name]
        color = algo_colors.get(algo_name, "black")
        for tx in triggers_x:
            ax_events.plot(tx, y, "|", color=color, markersize=15, markeredgewidth=2)

    ax_events.set_yticks(list(y_positions.values()))
    ax_events.set_yticklabels(list(y_positions.keys()), fontsize=9)
    ax_events.set_xlabel(x_label, fontsize=12)
    ax_events.set_title("EPLB Trigger Events", fontsize=11)
    ax_events.set_ylim(-0.5, len(EPLB_ALGORITHMS) - 0.5)

    plt.tight_layout()
    fname = f"imbalance_compare_t{threshold:.1f}_{x_mode}_{filter_mode}.png"
    savefig(fig, os.path.join(output_dir, fname))



# ──────────────────────────────────────────────────────────
# HTML index
# ──────────────────────────────────────────────────────────
def generate_index(output_root, layer_indices):
    """Generate an HTML index for easy browsing."""
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Expert Activation Analysis</title>
<style>
body { font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #34495e; margin-top: 30px; }
h3 { color: #7f8c8d; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 15px; }
.card { background: white; border-radius: 8px; padding: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
.card img { width: 100%; border-radius: 4px; cursor: pointer; }
.card img:hover { transform: scale(1.02); transition: 0.2s; }
.card p { margin: 5px 0; font-size: 13px; color: #555; text-align: center; }
.nav { position: sticky; top: 0; background: #2c3e50; padding: 10px 20px; border-radius: 8px; margin-bottom: 20px; }
.nav a { color: white; margin-right: 15px; text-decoration: none; font-weight: bold; }
.nav a:hover { color: #3498db; }
details { margin: 10px 0; }
summary { cursor: pointer; font-weight: bold; font-size: 16px; color: #2c3e50; }
</style></head><body>
<h1>Expert Activation Analysis</h1>
<div class="nav">
"""
    for idx in layer_indices:
        html += f'<a href="#layer_{idx}">Layer {idx}</a>\n'
    html += "</div>\n"

    plot_types = [
        ("heatmap", "1. Expert Activation Heatmap"),
        ("stacked", "2. Stacked Area Chart"),
    ]
    x_modes = ["time", "step"]
    filter_modes = ["decode", "prefill", "mix"]

    for idx in layer_indices:
        layer_dir = f"layer_{idx}"
        html += f'<h2 id="layer_{idx}">Layer {idx}</h2>\n'

        for plot_prefix, plot_title in plot_types:
            html += f"<details open><summary>{plot_title}</summary>\n<div class='grid'>\n"
            for x_mode in x_modes:
                for fm in filter_modes:
                    fname = f"{plot_prefix}_{x_mode}_{fm}.png"
                    fpath = f"{layer_dir}/{fname}"
                    label = f"{x_mode.title()} / {fm.title()}"
                    html += f"""<div class="card">
<a href="{fpath}" target="_blank"><img src="{fpath}" alt="{label}"></a>
<p>{label}</p>
</div>\n"""
            html += "</div></details>\n"

        # EPLB comparison — per threshold
        for thresh in IMBALANCE_THRESHOLDS:
            title = f"3. EPLB Algorithm Comparison (threshold={thresh:.1f})"
            html += f"<details open><summary>{title}</summary>\n<div class='grid'>\n"
            for x_mode in x_modes:
                for fm in filter_modes:
                    fname = f"imbalance_compare_t{thresh:.1f}_{x_mode}_{fm}.png"
                    fpath = f"{layer_dir}/{fname}"
                    label = f"{x_mode.title()} / {fm.title()}"
                    html += f"""<div class="card">
<a href="{fpath}" target="_blank"><img src="{fpath}" alt="{label}"></a>
<p>{label}</p>
</div>\n"""
            html += "</div></details>\n"

    html += "</body></html>"

    index_path = os.path.join(output_root, "index.html")
    with open(index_path, "w") as f:
        f.write(html)
    print(f"  Saved index: {index_path}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    args = json.loads(sys.argv[1])
    exp_dir = Path(args["exp_dir"])
    output_dir = Path(args["output_dir"])
    inputs = args["inputs"]  # {"logical_name": "path/to/file", ...}

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print(inputs)

    dir_path = Path(next(iter(inputs.values())))

    layer_entries = {}

    for file in dir_path.glob("layer_*.log"):
        m = re.search(r"layer[_-]?(\d+)", file.name)
        if m:
            layer_entries[int(m.group(1))] = str(file)

    if not layer_entries:
        print(f"No layer files found in inputs. Available inputs: {list(inputs.keys())}")
        return

    layer_indices = sorted(layer_entries.keys())
    print(f"Found {len(layer_indices)} layer inputs: {['layer_' + str(i) for i in layer_indices]}")
    print(f"Imbalance thresholds: {IMBALANCE_THRESHOLDS}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    X_MODES = ["time", "step"]
    FILTER_MODES = ["decode", "prefill", "mix"]

    for layer_idx in layer_indices:
        filepath = layer_entries[layer_idx]

        print(f"\n{'='*60}")
        print(f"Processing Layer {layer_idx}: {filepath}")
        print(f"{'='*60}")

        records = parse_log_file(filepath)
        if STEP_RANGE is not None:
            m, n = STEP_RANGE
            records = records[m:n]
            print(f"  Loaded {len(records)} records (step range [{m}, {n}))")
        print(f"  Total: {len(records)} records "
              f"({sum(1 for r in records if r['is_decode'])} decode, "
              f"{sum(1 for r in records if not r['is_decode'])} prefill)")

        layer_dir = os.path.join(str(output_dir), f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)

        for x_mode in X_MODES:
            for filter_mode in FILTER_MODES:
                filtered = filter_records(records, filter_mode)
                if len(filtered) < 2:
                    print(f"  [SKIP] {x_mode}/{filter_mode}: only {len(filtered)} records")
                    continue

                print(f"\n  --- {x_mode} / {filter_mode} ({len(filtered)} records) ---")

                # 1. Heatmap
                plot_heatmap(filtered, x_mode, filter_mode, layer_idx, layer_dir)

                # 2. Stacked
                plot_stacked(filtered, x_mode, filter_mode, layer_idx, layer_dir)

                # 3. Imbalance — compare EPLB algorithms (per threshold)
                for thresh in IMBALANCE_THRESHOLDS:
                    plot_imbalance_compare(filtered, x_mode, filter_mode, layer_idx, layer_dir, threshold=thresh)

        # 层级汇总
        num_experts = len(records[0]["counts"])
        all_counts = np.array([r["counts"] for r in records])
        total_per_expert = all_counts.sum(axis=0)
        mean_ir = np.mean([compute_imbalance_ratio(r["counts"]) for r in records])
        most_active = int(np.argmax(total_per_expert))
        least_active = int(np.argmin(total_per_expert))
        print(f"\n  Layer {layer_idx} 完成 ({num_experts} experts, {len(records)} records)")
        print(f"    平均不均衡度: {mean_ir:.3f}")
        print(f"    最活跃专家: Expert {most_active} ({total_per_expert[most_active]:.0f} tokens)")
        print(f"    最不活跃专家: Expert {least_active} ({total_per_expert[least_active]:.0f} tokens)")
        print(f"    输出目录: {layer_dir}")

    # Generate summary index
    generate_index(str(output_dir), layer_indices)
    print(f"\n{'='*60}")
    print(f"完成: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
