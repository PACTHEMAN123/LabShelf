#!/usr/bin/env python3
"""EPLB Simulation Analysis Pipeline
====================================
Produces per-layer and system-level EPLB comparison plots:
1. Per-layer: No EPLB vs Static vs Dynamic EPLB
2. System-level: same comparison aggregated across all layers
3. Relative change: Dynamic vs Static EPLB

Each analysis: 1 x-axis mode (step) × 3 filter modes (prefill, decode, mix)

自动生成 by labshelf.py add-script
"""
import json
import sys
import os
import re
import ast
import shutil
import heapq
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────
EPLB_INTERVALS = [1, 2, 5, 200, 1000]  # compare different EPLB intervals
HIDDEN_SIZE = 5120
MOE_INTERMEDIATE_SIZE = 1536
NUM_EP_RANKS = 32  # number of EP ranks (parallelism degree)
EXTRA_SLOTS_PER_RANK = 1  # slots_per_rank = local_experts + extra
EPMOE_LATENCY = {
    "active_experts": list(range(1, 33)),
    "latency_ms": [
        0.327536,0.328576,0.329600,0.327296,0.354256,0.33784,0.329968,0.325616,
        0.326016,0.363072,0.405536,0.411616,0.433280,0.480448,0.516096,0.508736,
        0.549792,0.542576,0.575744,0.634432,0.637968,0.652320,0.691952,0.687616,
        0.719296,0.741088,0.743392,0.840064,0.8702912,0.827024,0.862720,0.913840
    ],
}
STEP_RANGE = (1000, 10000)  # (m, n) 只取第 m~n 条记录，None 表示全部
# STEP_RANGE = None

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
# Metrics / helpers
# ──────────────────────────────────────────────────────────
def compute_group_loads(counts, permutation, num_groups=None):
    """Compute per-group token loads given expert counts and permutation."""
    n = len(counts)
    if num_groups is None:
        num_groups = max(1, int(np.sqrt(n)))
    group_loads = np.zeros(num_groups)
    group_size = (n + num_groups - 1) // num_groups
    for expert_idx in range(n):
        group_id = min(permutation[expert_idx] // group_size, num_groups - 1)
        group_loads[group_id] += counts[expert_idx]
    return group_loads


def eplb_replica_placement(counts, num_groups=None, slots_per_rank=None):
    """EPLB with expert replication + greedy LPT placement.

    Step 1 (Replication): greedily assign extra slots to highest per-replica load experts.
    Step 2 (Placement): assign replicas to least-loaded ranks, largest first.

    Returns (rank_experts, replica_counts):
      rank_experts:  list[num_groups] of lists of expert IDs on each rank
      replica_counts: array[n] — number of replicas per expert
    """
    n = len(counts)
    if num_groups is None:
        num_groups = max(1, int(np.sqrt(n)))
    if slots_per_rank is None:
        slots_per_rank = (n + num_groups - 1) // num_groups

    total_slots = num_groups * slots_per_rank
    replica_counts = np.ones(n, dtype=int)

    # Step 1: Replication — max-heap by per-replica load (capped at num_groups)
    if total_slots > n:
        heap = [(-float(counts[i]), i) for i in range(n)]
        heapq.heapify(heap)
        assigned = n
        while assigned < total_slots:
            neg_load, eid = heapq.heappop(heap)
            if replica_counts[eid] >= num_groups:
                continue  # can't exceed rank count
            replica_counts[eid] += 1
            heapq.heappush(heap, (-float(counts[eid]) / replica_counts[eid], eid))
            assigned += 1

    # Step 2: Placement — sort replicas largest-first, assign to least-loaded rank
    # that doesn't already have this expert
    replicas = []
    for eid in range(n):
        slot_load = float(counts[eid]) / replica_counts[eid]
        for _ in range(replica_counts[eid]):
            replicas.append((slot_load, eid))
    replicas.sort(key=lambda x: -x[0])

    rank_loads = np.zeros(num_groups)
    rank_experts = [[] for _ in range(num_groups)]
    rank_has = [set() for _ in range(num_groups)]

    for slot_load, eid in replicas:
        best_rid, best_load = -1, float('inf')
        for rid in range(num_groups):
            if eid not in rank_has[rid] and rank_loads[rid] < best_load:
                best_load = rank_loads[rid]
                best_rid = rid
        if best_rid == -1:
            best_rid = int(np.argmin(rank_loads))
        rank_experts[best_rid].append(eid)
        rank_has[best_rid].add(eid)
        rank_loads[best_rid] += slot_load

    return rank_experts, replica_counts


def compute_group_loads_replica(counts, rank_experts, replica_counts):
    """Compute per-group loads with expert replication (load split across replicas)."""
    num_groups = len(rank_experts)
    group_loads = np.zeros(num_groups)
    for g in range(num_groups):
        for eid in rank_experts[g]:
            group_loads[g] += counts[eid] / replica_counts[eid]
    return group_loads


def load_latency_table():
    """Return an interpolation function: active_experts -> latency_ms from hardcoded data."""
    xs = np.array(EPMOE_LATENCY["active_experts"], dtype=float)
    ys = np.array(EPMOE_LATENCY["latency_ms"], dtype=float)

    def latency_fn(n_active):
        if n_active <= 0:
            return 0.0
        if n_active >= xs[-1]:
            return float(ys[-1])
        return float(np.interp(n_active, xs, ys))
    return latency_fn


def compute_rank_active_counts(counts, rank_experts):
    """Count active expert slots per rank. Each slot with counts[eid] > 0 is active."""
    active = np.zeros(len(rank_experts), dtype=int)
    for g in range(len(rank_experts)):
        for eid in rank_experts[g]:
            if counts[eid] > 0:
                active[g] += 1
    return active


# ──────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────
def get_x_values(filtered_records, x_mode):
    """Get x-axis values based on mode."""
    if x_mode == "time":
        vals = np.array([r["time"] for r in filtered_records])
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
# EPLB simulation helpers
# ──────────────────────────────────────────────────────────
def _compute_init_placement(pre_records, n_experts, num_groups, epg, slots_per_rank):
    """Compute initial EPLB placement from pre-window records.
    Returns (rk_experts, rk_replicas, agg_counts)."""
    if not pre_records:
        rk_experts = [[] for _ in range(num_groups)]
        for e in range(n_experts):
            rk_experts[min(e // epg, num_groups - 1)].append(e)
        return rk_experts, np.ones(n_experts, dtype=int), np.zeros(n_experts)
    agg = np.zeros(n_experts, dtype=np.float64)
    for r in pre_records:
        agg += r["counts"]
    rk_experts, rk_replicas = eplb_replica_placement(agg, num_groups, slots_per_rank)
    return rk_experts, rk_replicas, agg


def _simulate_no_eplb(filtered_records, n_experts, num_groups, latency_fn):
    """No EPLB: identity permutation throughout."""
    identity = np.arange(n_experts)
    epg = (n_experts + num_groups - 1) // num_groups
    identity_ranks = [list(range(g * epg, min((g + 1) * epg, n_experts)))
                      for g in range(num_groups)]
    n = len(filtered_records)
    time_arr = np.zeros(n)
    util_arr = np.zeros(n)
    max_load_arr = np.zeros(n)
    mean_load_arr = np.zeros(n)
    for i, r in enumerate(filtered_records):
        gl = compute_group_loads(r["counts"], identity, num_groups)
        max_l, mean_l = gl.max(), gl.mean()
        max_load_arr[i] = max_l
        mean_load_arr[i] = mean_l
        util_arr[i] = mean_l / max_l if max_l > 0 else 1.0
        ac = compute_rank_active_counts(r["counts"], identity_ranks)
        time_arr[i] = max(latency_fn(a) for a in ac)
    return time_arr, util_arr, max_load_arr, mean_load_arr


def _simulate_static_eplb(filtered_records, latency_fn, rk_experts, rk_replicas):
    """Static EPLB: fixed placement from pre-window stats, never re-adjusted."""
    n = len(filtered_records)
    time_arr = np.zeros(n)
    util_arr = np.zeros(n)
    max_load_arr = np.zeros(n)
    mean_load_arr = np.zeros(n)
    for i, r in enumerate(filtered_records):
        gl = compute_group_loads_replica(r["counts"], rk_experts, rk_replicas)
        max_l, mean_l = gl.max(), gl.mean()
        max_load_arr[i] = max_l
        mean_load_arr[i] = mean_l
        util_arr[i] = mean_l / max_l if max_l > 0 else 1.0
        ac = compute_rank_active_counts(r["counts"], rk_experts)
        time_arr[i] = max(latency_fn(a) for a in ac)
    return time_arr, util_arr, max_load_arr, mean_load_arr


def _simulate_dynamic_eplb(filtered_records, eplb_interval, n_experts, num_groups,
                            slots_per_rank, latency_fn,
                            init_rk_experts, init_rk_replicas):
    """Dynamic EPLB: every k steps, aggregate future window and run greedy EPLB."""
    n = len(filtered_records)
    time_arr = np.zeros(n)
    util_arr = np.zeros(n)
    max_load_arr = np.zeros(n)
    mean_load_arr = np.zeros(n)
    rk_experts = [list(g) for g in init_rk_experts]
    rk_replicas = init_rk_replicas.copy()

    for i, r in enumerate(filtered_records):
        if i % eplb_interval == 0:
            future_end = min(i + eplb_interval, n)
            agg = np.zeros(n_experts, dtype=np.float64)
            for j in range(i, future_end):
                agg += filtered_records[j]["counts"]
            rk_experts, rk_replicas = eplb_replica_placement(
                agg, num_groups, slots_per_rank)
        gl = compute_group_loads_replica(r["counts"], rk_experts, rk_replicas)
        max_l, mean_l = gl.max(), gl.mean()
        max_load_arr[i] = max_l
        mean_load_arr[i] = mean_l
        util_arr[i] = mean_l / max_l if max_l > 0 else 1.0
        ac = compute_rank_active_counts(r["counts"], rk_experts)
        time_arr[i] = max(latency_fn(a) for a in ac)
    return time_arr, util_arr, max_load_arr, mean_load_arr


def _compute_simulations(filtered, pre_filtered):
    """Compute all simulations for one (layer, filter_mode). Returns dict of results."""
    n_experts = len(filtered[0]["counts"])
    num_groups = NUM_EP_RANKS
    epg = (n_experts + num_groups - 1) // num_groups
    slots_per_rank = epg + EXTRA_SLOTS_PER_RANK
    latency_fn = load_latency_table()

    init_rk, init_rep, init_agg = _compute_init_placement(
        pre_filtered, n_experts, num_groups, epg, slots_per_rank)

    no = _simulate_no_eplb(filtered, n_experts, num_groups, latency_fn)
    st = _simulate_static_eplb(filtered, latency_fn, init_rk, init_rep)

    dy = {}
    for k in EPLB_INTERVALS:
        dy[k] = _simulate_dynamic_eplb(
            filtered, k, n_experts, num_groups,
            slots_per_rank, latency_fn, init_rk, init_rep)

    return {"no": no, "static": st, "dynamic": dy}


# ──────────────────────────────────────────────────────────
# Plot: EPLB comparison — No EPLB vs Static vs Dynamic (per-layer)
# ──────────────────────────────────────────────────────────
def plot_eplb_compare(sim, x_vals, x_label, x_mode, filter_mode,
                      layer_idx, output_dir, eplb_interval):
    """Per-layer: No EPLB vs Static EPLB vs Dynamic EPLB(k) for one interval."""
    no_time, no_util, no_ml, _ = sim["no"]
    st_time, st_util, st_ml, _ = sim["static"]
    dy_time, dy_util, dy_ml, _ = sim["dynamic"][eplb_interval]

    MB = 1024 ** 2
    activation_per_token = (HIDDEN_SIZE + 2 * MOE_INTERMEDIATE_SIZE) * 2
    no_mem = no_ml * activation_per_token / MB
    st_mem = st_ml * activation_per_token / MB
    dy_mem = dy_ml * activation_per_token / MB

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    lw = 1.2
    c_no, c_st, c_dy = "#e74c3c", "#3498db", "#9b59b6"

    ax1.plot(x_vals, no_time, label="No EPLB", color=c_no, linewidth=lw, alpha=0.8)
    ax1.plot(x_vals, st_time, label="Static EPLB", color=c_st, linewidth=lw, alpha=0.8)
    ax1.plot(x_vals, dy_time, label=f"Dynamic (k={eplb_interval})",
             color=c_dy, linewidth=lw, alpha=0.8)
    ax1.set_ylabel("MoE Time (ms)\n(latency lookup)", fontsize=11)
    ax1.set_title(
        make_title(f"EPLB Comparison (k={eplb_interval}, +{EXTRA_SLOTS_PER_RANK} slots)",
                   x_mode, filter_mode, layer_idx),
        fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

    ax2.plot(x_vals, no_util * 100, label="No EPLB", color=c_no, linewidth=lw, alpha=0.8)
    ax2.plot(x_vals, st_util * 100, label="Static EPLB", color=c_st, linewidth=lw, alpha=0.8)
    ax2.plot(x_vals, dy_util * 100, label=f"Dynamic (k={eplb_interval})",
             color=c_dy, linewidth=lw, alpha=0.8)
    ax2.set_ylabel("Flops Utilization (%)", fontsize=11)
    ax2.set_ylim(0, 105); ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)

    ax3.plot(x_vals, no_mem, label="No EPLB", color=c_no, linewidth=lw, alpha=0.8)
    ax3.plot(x_vals, st_mem, label="Static EPLB", color=c_st, linewidth=lw, alpha=0.8)
    ax3.plot(x_vals, dy_mem, label=f"Dynamic (k={eplb_interval})",
             color=c_dy, linewidth=lw, alpha=0.8)
    y_hi = max(no_mem.max(), st_mem.max(), dy_mem.max()) * 1.05
    ax3.set_ylim(0, y_hi)
    ax3.set_ylabel("Activation Memory per EP Group (MB)", fontsize=11)
    ax3.set_xlabel(x_label, fontsize=12)
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"eplb_k{eplb_interval}_{x_mode}_{filter_mode}.png"
    savefig(fig, os.path.join(output_dir, fname))


# ──────────────────────────────────────────────────────────
# Plot: System-level Dynamic vs Static — relative change
# ──────────────────────────────────────────────────────────
def plot_system_dynamic_vs_static(layer_sims, layer_indices, x_mode,
                                  filter_mode, output_dir):
    """System-level relative change: Dynamic EPLB(k) vs Static EPLB.

    System metrics per step:
      Time  = sum across layers (sequential forward)
      Memory = max across layers (peak activation, in max_load units)
      Utilization = sum(mean_load) / sum(max_load) across layers
    """
    n_steps = min(len(layer_sims[idx]["no"][0]) for idx in layer_indices)
    MB = 1024 ** 2
    activation_per_token = (HIDDEN_SIZE + 2 * MOE_INTERMEDIATE_SIZE) * 2

    # Aggregate static system metrics per step
    st_sys_time = np.zeros(n_steps)
    st_sys_mem = np.zeros(n_steps)
    st_sys_sum_mean = np.zeros(n_steps)
    st_sys_sum_max = np.zeros(n_steps)
    for idx in layer_indices:
        st_time, st_util, st_ml, st_mean = layer_sims[idx]["static"]
        st_sys_time += st_time[:n_steps]
        st_sys_mem = np.maximum(st_sys_mem, st_ml[:n_steps])
        st_sys_sum_mean += st_mean[:n_steps]
        st_sys_sum_max += st_ml[:n_steps]
    st_sys_mem_mb = st_sys_mem * activation_per_token / MB
    st_sys_util = np.where(st_sys_sum_max > 0, st_sys_sum_mean / st_sys_sum_max, 1.0)

    time_rel = []
    util_rel = []
    mem_rel = []

    for k in EPLB_INTERVALS:
        dy_sys_time = np.zeros(n_steps)
        dy_sys_mem = np.zeros(n_steps)
        dy_sys_sum_mean = np.zeros(n_steps)
        dy_sys_sum_max = np.zeros(n_steps)
        for idx in layer_indices:
            dy_time, dy_util, dy_ml, dy_mean = layer_sims[idx]["dynamic"][k]
            dy_sys_time += dy_time[:n_steps]
            dy_sys_mem = np.maximum(dy_sys_mem, dy_ml[:n_steps])
            dy_sys_sum_mean += dy_mean[:n_steps]
            dy_sys_sum_max += dy_ml[:n_steps]
        dy_sys_mem_mb = dy_sys_mem * activation_per_token / MB
        dy_sys_util = np.where(dy_sys_sum_max > 0, dy_sys_sum_mean / dy_sys_sum_max, 1.0)

        safe_time = np.where(st_sys_time > 0, st_sys_time, 1e-12)
        safe_util = np.where(st_sys_util > 0, st_sys_util, 1e-12)
        safe_mem = np.where(st_sys_mem_mb > 0, st_sys_mem_mb, 1e-12)

        time_rel.append((dy_sys_time - st_sys_time) / safe_time * 100)
        util_rel.append((dy_sys_util - st_sys_util) / safe_util * 100)
        mem_rel.append((dy_sys_mem_mb - st_sys_mem_mb) / safe_mem * 100)

    # Statistics
    def _stats(arr):
        return {
            "mean": np.mean(arr), "median": np.median(arr),
            "p5": np.percentile(arr, 5), "p95": np.percentile(arr, 95),
        }

    # ---- Plot ----
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(max(10, 3 * len(EPLB_INTERVALS)), 14))

    x_pos = np.arange(len(EPLB_INTERVALS))
    bar_width = 0.55
    dynamic_cmap = plt.cm.Purples
    colors = [dynamic_cmap(0.4 + 0.15 * i) for i in range(len(EPLB_INTERVALS))]

    mode_label = {"decode": "Decode Only", "prefill": "Prefill Only", "mix": "All (Mixed)"}
    n_layers = len(layer_indices)

    metric_data = [
        (ax1, time_rel, "System MoE Time", "Relative Change vs Static EPLB (%)\n(negative = faster)"),
        (ax2, util_rel, "System Utilization", "Relative Change vs Static EPLB (%)\n(positive = better)"),
        (ax3, mem_rel, "System Peak Memory", "Relative Change vs Static EPLB (%)\n(negative = less memory)"),
    ]

    for ax, rel_list, metric_name, ylabel in metric_data:
        means, p5s, p95s, medians = [], [], [], []
        for rel_arr in rel_list:
            s = _stats(rel_arr)
            means.append(s["mean"]); medians.append(s["median"])
            p5s.append(s["p5"]); p95s.append(s["p95"])

        means = np.array(means); medians = np.array(medians)
        p5s = np.array(p5s); p95s = np.array(p95s)
        err_low = np.maximum(means - p5s, 0)
        err_high = np.maximum(p95s - means, 0)

        bars = ax.bar(x_pos, means, bar_width, color=colors, edgecolor="white",
                      linewidth=0.5, yerr=[err_low, err_high],
                      capsize=5, error_kw={"linewidth": 1.2, "color": "#555"})
        ax.axhline(y=0, color="#999", linestyle="-", linewidth=0.8)

        for i, (bar, mean_val, med_val, p5_val, p95_val) in enumerate(
                zip(bars, means, medians, p5s, p95s)):
            va = "bottom" if mean_val >= 0 else "top"
            y_offset = max(abs(p95_val), abs(mean_val)) + 0.5 if mean_val >= 0 else min(-abs(p5_val), mean_val) - 0.5
            ax.text(bar.get_x() + bar.get_width() / 2, y_offset,
                    f"mean:{mean_val:+.2f}%\nmed:{med_val:+.2f}%",
                    ha="center", va=va, fontsize=8, fontweight="bold", color="#333")

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"k={k}" for k in EPLB_INTERVALS], fontsize=10)
        ax.set_ylim(-100, 100)
        ax.grid(True, alpha=0.3, axis="y")

    ax1.set_title(
        f"System ({n_layers} layers) | Dynamic EPLB vs Static "
        f"(+{EXTRA_SLOTS_PER_RANK} slots) | {mode_label[filter_mode]}",
        fontsize=14, fontweight="bold")
    ax3.set_xlabel("EPLB Re-balance Interval (k steps)", fontsize=11)

    plt.tight_layout()
    fname = f"eplb_system_vs_static_{x_mode}_{filter_mode}.png"
    savefig(fig, os.path.join(output_dir, fname))


# ──────────────────────────────────────────────────────────
# Plot: Dynamic vs Static EPLB — relative change statistics
# ──────────────────────────────────────────────────────────
def plot_eplb_dynamic_vs_static(sim, x_mode, filter_mode,
                                layer_idx, output_dir):
    """Per-layer: for each Dynamic EPLB(k), show relative change vs Static EPLB.

    3 subplots (time, utilization, memory). Each subplot is a grouped bar chart.
    Uses pre-computed simulation results from sim dict.
    """
    st_time, st_util, st_ml, _ = sim["static"]
    MB = 1024 ** 2
    activation_per_token = (HIDDEN_SIZE + 2 * MOE_INTERMEDIATE_SIZE) * 2
    st_mem = st_ml * activation_per_token / MB

    time_rel = []
    util_rel = []
    mem_rel = []

    for k in EPLB_INTERVALS:
        dy_time, dy_util, dy_ml, _ = sim["dynamic"][k]
        dy_mem = dy_ml * activation_per_token / MB

        # Relative change (%), avoid div-by-zero
        safe_st_time = np.where(st_time > 0, st_time, 1e-12)
        safe_st_util = np.where(st_util > 0, st_util, 1e-12)
        safe_st_mem = np.where(st_mem > 0, st_mem, 1e-12)

        time_rel.append((dy_time - st_time) / safe_st_time * 100)
        util_rel.append((dy_util - st_util) / safe_st_util * 100)
        mem_rel.append((dy_mem - st_mem) / safe_st_mem * 100)

    # Compute statistics per k
    def _stats(arr):
        return {
            "mean": np.mean(arr),
            "median": np.median(arr),
            "p5": np.percentile(arr, 5),
            "p95": np.percentile(arr, 95),
            "min": np.min(arr),
            "max": np.max(arr),
        }

    # ---- Plot ----
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(max(10, 3 * len(EPLB_INTERVALS)), 14))

    x_pos = np.arange(len(EPLB_INTERVALS))
    bar_width = 0.55
    dynamic_cmap = plt.cm.Purples
    colors = [dynamic_cmap(0.4 + 0.15 * i) for i in range(len(EPLB_INTERVALS))]

    mode_label = {"decode": "Decode Only", "prefill": "Prefill Only", "mix": "All (Mixed)"}

    metric_data = [
        (ax1, time_rel, "MoE Time", "Relative Change vs Static EPLB (%)\n(negative = faster)"),
        (ax2, util_rel, "Flops Utilization", "Relative Change vs Static EPLB (%)\n(positive = better)"),
        (ax3, mem_rel, "Activation Memory", "Relative Change vs Static EPLB (%)\n(negative = less memory)"),
    ]

    for ax, rel_list, metric_name, ylabel in metric_data:
        means = []
        medians = []
        p5s = []
        p95s = []
        for rel_arr in rel_list:
            s = _stats(rel_arr)
            means.append(s["mean"])
            medians.append(s["median"])
            p5s.append(s["p5"])
            p95s.append(s["p95"])

        means = np.array(means)
        medians = np.array(medians)
        p5s = np.array(p5s)
        p95s = np.array(p95s)

        # Error bars: p5 to p95 (clamp to non-negative)
        err_low = np.maximum(means - p5s, 0)
        err_high = np.maximum(p95s - means, 0)

        bars = ax.bar(x_pos, means, bar_width, color=colors, edgecolor="white",
                      linewidth=0.5, yerr=[err_low, err_high],
                      capsize=5, error_kw={"linewidth": 1.2, "color": "#555"})

        # Zero line
        ax.axhline(y=0, color="#999", linestyle="-", linewidth=0.8)

        # Annotate: mean and median on each bar
        for i, (bar, mean_val, med_val, p5_val, p95_val) in enumerate(
                zip(bars, means, medians, p5s, p95s)):
            # Mean label at bar top/bottom
            va = "bottom" if mean_val >= 0 else "top"
            y_offset = max(abs(p95_val), abs(mean_val)) + 0.5 if mean_val >= 0 else min(-abs(p5_val), mean_val) - 0.5
            ax.text(bar.get_x() + bar.get_width() / 2, y_offset,
                    f"mean:{mean_val:+.2f}%\nmed:{med_val:+.2f}%",
                    ha="center", va=va, fontsize=8, fontweight="bold",
                    color="#333")

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"k={k}" for k in EPLB_INTERVALS], fontsize=10)
        ax.set_ylim(-100, 100)
        ax.grid(True, alpha=0.3, axis="y")

    ax1.set_title(
        make_title(f"Dynamic EPLB vs Static (+{EXTRA_SLOTS_PER_RANK} slots)",
                   x_mode, filter_mode, layer_idx),
        fontsize=14, fontweight="bold")
    ax3.set_xlabel("EPLB Re-balance Interval (k steps)", fontsize=11)

    plt.tight_layout()
    fname = f"eplb_vs_static_{x_mode}_{filter_mode}.png"
    savefig(fig, os.path.join(output_dir, fname))


# ──────────────────────────────────────────────────────────
# HTML index
# ──────────────────────────────────────────────────────────
def generate_eplb_index(output_root, layer_indices):
    """Generate a separate HTML index for EPLB analysis."""
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>EPLB Simulation Analysis</title>
<style>
body { font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }
h1 { color: #2c3e50; border-bottom: 3px solid #9b59b6; padding-bottom: 10px; }
h2 { color: #34495e; margin-top: 30px; }
h3 { color: #7f8c8d; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 15px; }
.card { background: white; border-radius: 8px; padding: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
.card img { width: 100%; border-radius: 4px; cursor: pointer; }
.card img:hover { transform: scale(1.02); transition: 0.2s; }
.card p { margin: 5px 0; font-size: 13px; color: #555; text-align: center; }
.nav { position: sticky; top: 0; background: #2c3e50; padding: 10px 20px; border-radius: 8px; margin-bottom: 20px; }
.nav a { color: white; margin-right: 15px; text-decoration: none; font-weight: bold; }
.nav a:hover { color: #9b59b6; }
details { margin: 10px 0; }
summary { cursor: pointer; font-weight: bold; font-size: 16px; color: #2c3e50; }
</style></head><body>
<h1>EPLB Simulation Analysis</h1>
<div class="nav">
<a href="index.html">Main Index</a>
"""
    for idx in layer_indices:
        html += f'<a href="#layer_{idx}">Layer {idx}</a>\n'
    html += '<a href="#system">System</a>\n'
    html += "</div>\n"

    x_modes = ["step"]
    filter_modes = ["decode", "prefill", "mix"]

    for idx in layer_indices:
        layer_dir = f"layer_{idx}"
        html += f'<h2 id="layer_{idx}">Layer {idx}</h2>\n'

        # Dynamic vs Static relative change
        html += "<details open><summary>Dynamic vs Static EPLB — Relative Change</summary>\n<div class='grid'>\n"
        for x_mode in x_modes:
            for fm in filter_modes:
                fname = f"eplb_vs_static_{x_mode}_{fm}.png"
                fpath = f"{layer_dir}/{fname}"
                label = f"{x_mode.title()} / {fm.title()}"
                html += f"""<div class="card">
<a href="{fpath}" target="_blank"><img src="{fpath}" alt="{label}"></a>
<p>{label}</p>
</div>\n"""
        html += "</div></details>\n"

        # Per-k comparison
        for k in EPLB_INTERVALS:
            title = f"k={k}: No EPLB vs Static vs Dynamic"
            html += f"<details open><summary>{title}</summary>\n<div class='grid'>\n"
            for x_mode in x_modes:
                for fm in filter_modes:
                    fname = f"eplb_k{k}_{x_mode}_{fm}.png"
                    fpath = f"{layer_dir}/{fname}"
                    label = f"{x_mode.title()} / {fm.title()}"
                    html += f"""<div class="card">
<a href="{fpath}" target="_blank"><img src="{fpath}" alt="{label}"></a>
<p>{label}</p>
</div>\n"""
            html += "</div></details>\n"

    html += '<h2 id="system">System-Level</h2>\n'
    html += "<details open><summary>Dynamic vs Static EPLB — System Relative Change</summary>\n<div class='grid'>\n"
    for x_mode in x_modes:
        for fm in filter_modes:
            fname = f"eplb_system_vs_static_{x_mode}_{fm}.png"
            label = f"{x_mode.title()} / {fm.title()}"
            html += f"""<div class="card">
<a href="{fname}" target="_blank"><img src="{fname}" alt="{label}"></a>
<p>{label}</p>
</div>\n"""
    html += "</div></details>\n"

    html += "</body></html>"

    index_path = os.path.join(output_root, "eplb.html")
    with open(index_path, "w") as f:
        f.write(html)
    print(f"  Saved EPLB index: {index_path}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def _process_layer(args):
    """Worker: compute simulations + generate per-layer plots for one layer."""
    layer_idx, records, pre_records, output_dir = args

    layer_dir = os.path.join(output_dir, f"layer_{layer_idx}")
    os.makedirs(layer_dir, exist_ok=True)

    X_MODES = ["step"]
    FILTER_MODES = ["decode", "prefill", "mix"]

    sims = {}  # filter_mode -> sim dict
    for filter_mode in FILTER_MODES:
        filtered = filter_records(records, filter_mode)
        if len(filtered) < 2:
            continue
        pre_filtered = filter_records(pre_records, filter_mode)

        sim = _compute_simulations(filtered, pre_filtered)
        sims[filter_mode] = sim

        for x_mode in X_MODES:
            x_vals, x_label = get_x_values(filtered, x_mode)

            for k in EPLB_INTERVALS:
                plot_eplb_compare(sim, x_vals, x_label, x_mode, filter_mode,
                                 layer_idx, layer_dir, k)

            plot_eplb_dynamic_vs_static(sim, x_mode, filter_mode,
                                        layer_idx, layer_dir)

    print(f"  Layer {layer_idx} done.")
    return layer_idx, sims


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
    print(f"EPLB intervals: {EPLB_INTERVALS}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    # Parse all layers — keep both window and pre-window records
    all_records = {}
    all_pre_records = {}
    for layer_idx in layer_indices:
        filepath = layer_entries[layer_idx]
        full = parse_log_file(filepath)
        if STEP_RANGE is not None:
            m, n = STEP_RANGE
            all_records[layer_idx] = full[m:n]
            all_pre_records[layer_idx] = full[:m]
        else:
            all_records[layer_idx] = full
            all_pre_records[layer_idx] = []

    # Process layers in parallel — each worker computes simulations + per-layer plots
    tasks = [(idx, all_records[idx], all_pre_records[idx], str(output_dir))
             for idx in layer_indices]
    n_workers = min(len(layer_indices), cpu_count())
    print(f"Processing {len(layer_indices)} layers with {n_workers} workers...")

    with Pool(n_workers) as pool:
        results = pool.map(_process_layer, tasks)

    # Collect per-layer simulations for system-level plots
    all_sims = {layer_idx: sims for layer_idx, sims in results}

    # System-level EPLB comparison — uses cached per-layer results
    print(f"\n{'='*60}")
    print(f"System-level EPLB simulation ({len(layer_indices)} layers)")
    print(f"{'='*60}")

    X_MODES = ["step"]
    FILTER_MODES = ["decode", "prefill", "mix"]

    for x_mode in X_MODES:
        for filter_mode in FILTER_MODES:
            if all(filter_mode in all_sims[idx] for idx in layer_indices):
                layer_sims = {idx: all_sims[idx][filter_mode]
                              for idx in layer_indices}
                plot_system_dynamic_vs_static(layer_sims, layer_indices, x_mode,
                                             filter_mode, str(output_dir))

    # Generate EPLB HTML index
    generate_eplb_index(str(output_dir), layer_indices)
    print(f"\n{'='*60}")
    print(f"完成: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
