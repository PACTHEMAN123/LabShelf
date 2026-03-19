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
EP_SIZES = [8, 16, 32, 64]  # EP rank counts to sweep
EXTRA_SLOTS_RANGE = [0, 1, 2]  # extra slots per rank to sweep
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
def eplb_replica_placement(counts, num_groups, slots_per_rank):
    """EPLB with expert replication + greedy LPT placement.

    Step 1 (Replication): greedily assign extra slots to highest per-replica load experts.
    Step 2 (Placement): assign replicas to least-loaded ranks, largest first.

    Returns (rank_experts, replica_counts):
      rank_experts:  list[num_groups] of lists of expert IDs on each rank
      replica_counts: array[n] — number of replicas per expert
    """
    n = len(counts)

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


def load_latency_table(json_path):
    """Return a lookup function: (n_active_experts, batch_size) -> latency_ms.

    JSON format: {"active_experts": [1..32], "batch_sizes": [...], "latency_ms": [[...]]}
    where latency_ms has shape (len(active_experts), len(batch_sizes)).

    n_active_experts is an integer 1-32, used as direct row index (no interpolation).
    batch_size is interpolated piecewise-linearly along each row, with linear
    extrapolation beyond the profiled range.
    """
    with open(json_path) as f:
        data = json.load(f)
    batch_sizes = np.array(data["batch_sizes"], dtype=float)
    latency_ms = np.array(data["latency_ms"], dtype=float)
    row_map = {int(ae): i for i, ae in enumerate(data["active_experts"])}

    def latency_fn(n_active, batch_size):
        if n_active <= 0 or batch_size <= 0:
            return 0.0
        row = latency_ms[row_map[int(n_active)]]
        bs = float(batch_size)
        if bs <= batch_sizes[0]:
            slope = (row[1] - row[0]) / (batch_sizes[1] - batch_sizes[0])
            return float(row[0] + slope * (bs - batch_sizes[0]))
        if bs >= batch_sizes[-1]:
            slope = (row[-1] - row[-2]) / (batch_sizes[-1] - batch_sizes[-2])
            return float(row[-1] + slope * (bs - batch_sizes[-1]))
        return float(np.interp(bs, batch_sizes, row))
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


def _simulate(filtered_records, latency_fn, rk_experts, rk_replicas,
              n_experts=None, num_groups=None, slots_per_rank=None, eplb_interval=None):
    """Unified EPLB simulation.

    eplb_interval=None → static (fixed placement).
    eplb_interval=N   → dynamic (re-balance every N steps).
    For no-EPLB, pass identity placement with eplb_interval=None.
    """
    n = len(filtered_records)
    time_arr = np.zeros(n)
    util_arr = np.zeros(n)
    max_load_arr = np.zeros(n)
    mean_load_arr = np.zeros(n)
    rk_experts = [list(g) for g in rk_experts]
    rk_replicas = rk_replicas.copy()

    for i, r in enumerate(filtered_records):
        if eplb_interval and i % eplb_interval == 0:
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
        time_arr[i] = max(latency_fn(int(ac[g]), float(gl[g]))
                          for g in range(len(rk_experts)))
    return time_arr, util_arr, max_load_arr, mean_load_arr


def _compute_simulations(filtered, pre_filtered, num_ep_ranks, extra_slots_per_rank, latency_fn):
    """Compute all simulations for one (layer, filter_mode). Returns dict of results."""
    n_experts = len(filtered[0]["counts"])
    epg = (n_experts + num_ep_ranks - 1) // num_ep_ranks
    slots_per_rank = epg + extra_slots_per_rank

    # Identity placement for no-EPLB
    identity_ranks = [list(range(g * epg, min((g + 1) * epg, n_experts)))
                      for g in range(num_ep_ranks)]
    identity_replicas = np.ones(n_experts, dtype=int)

    init_rk, init_rep, _ = _compute_init_placement(
        pre_filtered, n_experts, num_ep_ranks, epg, slots_per_rank)

    no = _simulate(filtered, latency_fn, identity_ranks, identity_replicas)
    st = _simulate(filtered, latency_fn, init_rk, init_rep)
    dy = {k: _simulate(filtered, latency_fn, init_rk, init_rep,
                        n_experts, num_ep_ranks, slots_per_rank, k)
          for k in EPLB_INTERVALS}

    return {"no": no, "static": st, "dynamic": dy}


# ──────────────────────────────────────────────────────────
# Plot: EPLB comparison — No EPLB vs Static vs Dynamic (per-layer)
# ──────────────────────────────────────────────────────────
def plot_eplb_compare(sim, x_vals, x_label, x_mode, filter_mode,
                      layer_idx, output_dir, eplb_interval,
                      num_ep_ranks, extra_slots_per_rank):
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
        make_title(f"EPLB Comparison (EP={num_ep_ranks}, k={eplb_interval}, +{extra_slots_per_rank} slots)",
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
                                  filter_mode, output_dir,
                                  num_ep_ranks, extra_slots_per_rank):
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
        f"(EP={num_ep_ranks}, +{extra_slots_per_rank} slots) | {mode_label[filter_mode]}",
        fontsize=14, fontweight="bold")
    ax3.set_xlabel("EPLB Re-balance Interval (k steps)", fontsize=11)

    plt.tight_layout()
    fname = f"eplb_system_vs_static_{x_mode}_{filter_mode}.png"
    savefig(fig, os.path.join(output_dir, fname))


# ──────────────────────────────────────────────────────────
# Plot: Dynamic vs Static EPLB — relative change statistics
# ──────────────────────────────────────────────────────────
def plot_eplb_dynamic_vs_static(sim, x_mode, filter_mode,
                                layer_idx, output_dir,
                                num_ep_ranks, extra_slots_per_rank):
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
        make_title(f"Dynamic EPLB vs Static (EP={num_ep_ranks}, +{extra_slots_per_rank} slots)",
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
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>EPLB Simulation Analysis</title>
{_HTML_STYLE}</head><body>
<h1>EPLB Simulation Analysis</h1>
<div class="nav">
<a href="../sweep_index.html">Sweep Index</a>
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
def _extract_layer_summary(sim):
    """Extract scalar summary from per-layer simulation results.
    Returns dict: {k: {metric_rel: {mean, p5, p95}}}."""
    st_time, st_util, st_ml, _ = sim["static"]
    MB = 1024 ** 2
    activation_per_token = (HIDDEN_SIZE + 2 * MOE_INTERMEDIATE_SIZE) * 2
    st_mem = st_ml * activation_per_token / MB

    def _s(arr):
        return {"mean": float(np.mean(arr)), "p5": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95))}

    safe_t = np.where(st_time > 0, st_time, 1e-12)
    safe_u = np.where(st_util > 0, st_util, 1e-12)
    safe_m = np.where(st_mem > 0, st_mem, 1e-12)

    summary = {}
    for k in EPLB_INTERVALS:
        dy_time, dy_util, dy_ml, _ = sim["dynamic"][k]
        dy_mem = dy_ml * activation_per_token / MB

        t_rel = (dy_time - st_time) / safe_t * 100
        u_rel = (dy_util - st_util) / safe_u * 100
        m_rel = (dy_mem - st_mem) / safe_m * 100

        summary[k] = {"time_rel": _s(t_rel), "util_rel": _s(u_rel), "mem_rel": _s(m_rel)}
    return summary


def _extract_system_summary(layer_sims, layer_indices):
    """Extract system-level scalar summary across layers.
    Returns dict: {k: {metric_rel: {mean, p5, p95}}}."""
    n_steps = min(len(layer_sims[idx]["no"][0]) for idx in layer_indices)
    MB = 1024 ** 2
    activation_per_token = (HIDDEN_SIZE + 2 * MOE_INTERMEDIATE_SIZE) * 2

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

    def _s(arr):
        return {"mean": float(np.mean(arr)), "p5": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95))}

    safe_t = np.where(st_sys_time > 0, st_sys_time, 1e-12)
    safe_u = np.where(st_sys_util > 0, st_sys_util, 1e-12)
    safe_m = np.where(st_sys_mem_mb > 0, st_sys_mem_mb, 1e-12)

    summary = {}
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

        t_rel = (dy_sys_time - st_sys_time) / safe_t * 100
        u_rel = (dy_sys_util - st_sys_util) / safe_u * 100
        m_rel = (dy_sys_mem_mb - st_sys_mem_mb) / safe_m * 100

        summary[k] = {"time_rel": _s(t_rel), "util_rel": _s(u_rel), "mem_rel": _s(m_rel)}
    return summary


def _process_layer(args):
    """Worker: compute simulations for one layer (plotting skipped in sweep mode)."""
    layer_idx, records, pre_records, num_ep_ranks, extra_slots_per_rank, latency_json_path = args

    latency_fn = load_latency_table(latency_json_path)
    FILTER_MODES = ["decode", "prefill", "mix"]

    sims = {}
    layer_summary = {}
    for filter_mode in FILTER_MODES:
        filtered = filter_records(records, filter_mode)
        if len(filtered) < 2:
            continue
        pre_filtered = filter_records(pre_records, filter_mode)

        sim = _compute_simulations(filtered, pre_filtered, num_ep_ranks, extra_slots_per_rank, latency_fn)
        sims[filter_mode] = sim
        layer_summary[filter_mode] = _extract_layer_summary(sim)

    print(f"  Layer {layer_idx} done.")
    return layer_idx, sims, layer_summary


# ──────────────────────────────────────────────────────────
# Comparison plots (across combos)
# ──────────────────────────────────────────────────────────
def plot_comparison_bar(combo_summaries, combos, filter_mode, layer_idx, output_dir):
    """Per-layer comparison: grouped bar chart across combos for one filter_mode."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1,
        figsize=(max(12, 2.5 * len(EPLB_INTERVALS)), 14))

    n_k = len(EPLB_INTERVALS)
    n_combos = len(combos)
    bar_width = 0.8 / n_combos
    x_pos = np.arange(n_k)

    cmap = plt.cm.tab10
    combo_colors = [cmap(i / max(n_combos, 1)) for i in range(n_combos)]

    mode_label = {"decode": "Decode Only", "prefill": "Prefill Only", "mix": "All (Mixed)"}
    metrics = [
        ("time_rel", "Relative Change vs Static (%)\n(negative = faster)"),
        ("util_rel", "Relative Change vs Static (%)\n(positive = better)"),
        ("mem_rel", "Relative Change vs Static (%)\n(negative = less memory)"),
    ]

    for ax, (metric_key, ylabel) in zip([ax1, ax2, ax3], metrics):
        for ci, combo in enumerate(combos):
            summary = combo_summaries.get(combo, {})
            layer_data = summary.get(layer_idx, {}).get(filter_mode)
            if layer_data is None:
                continue

            means = np.array([layer_data[k][metric_key]["mean"] for k in EPLB_INTERVALS])
            p5s = np.array([layer_data[k][metric_key]["p5"] for k in EPLB_INTERVALS])
            p95s = np.array([layer_data[k][metric_key]["p95"] for k in EPLB_INTERVALS])
            err_low = np.maximum(means - p5s, 0)
            err_high = np.maximum(p95s - means, 0)

            offset = (ci - (n_combos - 1) / 2) * bar_width
            ep_size, extra = combo
            ax.bar(x_pos + offset, means, bar_width * 0.9,
                   color=combo_colors[ci], edgecolor="white", linewidth=0.5,
                   yerr=[err_low, err_high], capsize=3,
                   error_kw={"linewidth": 0.8, "color": "#555"},
                   label=f"EP={ep_size},+{extra}")

        ax.axhline(y=0, color="#999", linestyle="-", linewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"k={k}" for k in EPLB_INTERVALS], fontsize=10)
        ax.set_ylim(-100, 100)
        ax.legend(fontsize=8, ncol=min(n_combos, 4))
        ax.grid(True, alpha=0.3, axis="y")

    ax1.set_title(
        f"Layer {layer_idx} | Combo Comparison — Dynamic vs Static | {mode_label[filter_mode]}",
        fontsize=14, fontweight="bold")
    ax3.set_xlabel("EPLB Re-balance Interval (k steps)", fontsize=11)

    plt.tight_layout()
    savefig(fig, os.path.join(output_dir, f"compare_{filter_mode}.png"))


def plot_comparison_system_bar(combo_summaries, combos, filter_mode, output_dir):
    """System-level comparison: grouped bar chart across combos for one filter_mode."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1,
        figsize=(max(12, 2.5 * len(EPLB_INTERVALS)), 14))

    n_k = len(EPLB_INTERVALS)
    n_combos = len(combos)
    bar_width = 0.8 / n_combos
    x_pos = np.arange(n_k)

    cmap = plt.cm.tab10
    combo_colors = [cmap(i / max(n_combos, 1)) for i in range(n_combos)]

    mode_label = {"decode": "Decode Only", "prefill": "Prefill Only", "mix": "All (Mixed)"}
    metrics = [
        ("time_rel", "Relative Change vs Static (%)\n(negative = faster)"),
        ("util_rel", "Relative Change vs Static (%)\n(positive = better)"),
        ("mem_rel", "Relative Change vs Static (%)\n(negative = less memory)"),
    ]

    for ax, (metric_key, ylabel) in zip([ax1, ax2, ax3], metrics):
        for ci, combo in enumerate(combos):
            sys_data = combo_summaries.get(combo, {}).get("system", {}).get(filter_mode)
            if sys_data is None:
                continue

            means = np.array([sys_data[k][metric_key]["mean"] for k in EPLB_INTERVALS])
            p5s = np.array([sys_data[k][metric_key]["p5"] for k in EPLB_INTERVALS])
            p95s = np.array([sys_data[k][metric_key]["p95"] for k in EPLB_INTERVALS])
            err_low = np.maximum(means - p5s, 0)
            err_high = np.maximum(p95s - means, 0)

            offset = (ci - (n_combos - 1) / 2) * bar_width
            ep_size, extra = combo
            ax.bar(x_pos + offset, means, bar_width * 0.9,
                   color=combo_colors[ci], edgecolor="white", linewidth=0.5,
                   yerr=[err_low, err_high], capsize=3,
                   error_kw={"linewidth": 0.8, "color": "#555"},
                   label=f"EP={ep_size},+{extra}")

        ax.axhline(y=0, color="#999", linestyle="-", linewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"k={k}" for k in EPLB_INTERVALS], fontsize=10)
        ax.set_ylim(-100, 100)
        ax.legend(fontsize=8, ncol=min(n_combos, 4))
        ax.grid(True, alpha=0.3, axis="y")

    ax1.set_title(
        f"System | Combo Comparison — Dynamic vs Static | {mode_label[filter_mode]}",
        fontsize=14, fontweight="bold")
    ax3.set_xlabel("EPLB Re-balance Interval (k steps)", fontsize=11)

    plt.tight_layout()
    savefig(fig, os.path.join(output_dir, f"compare_system_{filter_mode}.png"))


# ──────────────────────────────────────────────────────────
# HTML generators for sweep
# ──────────────────────────────────────────────────────────
_HTML_STYLE = """<style>
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
</style>"""


def generate_comparison_index(output_dir, combos, layer_indices):
    """Generate HTML index for comparison plots."""
    filter_modes = ["decode", "prefill", "mix"]
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>EPLB Sweep Comparison</title>
{_HTML_STYLE}</head><body>
<h1>EPLB Sweep Comparison</h1>
<div class="nav">
<a href="../sweep_index.html">Sweep Index</a>
"""
    for idx in layer_indices:
        html += f'<a href="#layer_{idx}">Layer {idx}</a>\n'
    html += '<a href="#system">System</a>\n'
    html += "</div>\n"

    combo_str = ", ".join(f"EP={ep}+{ex}" for ep, ex in combos)
    html += f"<p>Combos: {combo_str}</p>\n"

    for idx in layer_indices:
        html += f'<h2 id="layer_{idx}">Layer {idx}</h2>\n'
        html += "<div class='grid'>\n"
        for fm in filter_modes:
            fname = f"layer_{idx}/compare_{fm}.png"
            html += f"""<div class="card">
<a href="{fname}" target="_blank"><img src="{fname}" alt="Layer {idx} {fm}"></a>
<p>{fm.title()}</p>
</div>\n"""
        html += "</div>\n"

    html += '<h2 id="system">System-Level</h2>\n<div class="grid">\n'
    for fm in filter_modes:
        fname = f"compare_system_{fm}.png"
        html += f"""<div class="card">
<a href="{fname}" target="_blank"><img src="{fname}" alt="System {fm}"></a>
<p>{fm.title()}</p>
</div>\n"""
    html += "</div>\n</body></html>"

    path = os.path.join(output_dir, "comparison.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  Saved comparison index: {path}")


def generate_sweep_index(output_dir, combos, layer_indices):
    """Generate top-level HTML index linking all combo results and comparison."""
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>EPLB Hyperparameter Sweep</title>
{_HTML_STYLE}</head><body>
<h1>EPLB Hyperparameter Sweep</h1>
<div class="nav">
<a href="comparison/comparison.html">Comparison</a>
"""
    for ep, ex in combos:
        tag = f"ep{ep}_extra{ex}"
        html += f'<a href="{tag}/eplb.html">EP={ep},+{ex}</a>\n'
    html += "</div>\n"

    html += "<h2>Configurations</h2>\n<ul>\n"
    for ep, ex in combos:
        tag = f"ep{ep}_extra{ex}"
        html += f'<li><a href="{tag}/eplb.html">EP={ep}, extra_slots={ex}</a></li>\n'
    html += "</ul>\n"

    html += '<h2>Cross-Configuration Comparison</h2>\n'
    html += '<p><a href="comparison/comparison.html">View comparison plots</a></p>\n'

    html += "</body></html>"

    path = os.path.join(output_dir, "sweep_index.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  Saved sweep index: {path}")


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

    # Detect trace directory (layer_*.log) and latency model (JSON) from inputs
    trace_dir = None
    latency_json_path = None
    for name, path in inputs.items():
        p = Path(path)
        if p.is_dir() and list(p.glob("layer_*.log")):
            trace_dir = p
        elif p.is_file() and p.suffix == ".json":
            latency_json_path = str(p)
        elif p.is_dir():
            jsons = list(p.glob("*.json"))
            if jsons:
                latency_json_path = str(jsons[0])

    if trace_dir is None:
        sys.exit("Error: no trace directory (containing layer_*.log) found in inputs")
    if latency_json_path is None:
        sys.exit("Error: no latency model JSON found in inputs")

    # Discover layers
    layer_entries = {}
    for file in trace_dir.glob("layer_*.log"):
        m = re.search(r"layer[_-]?(\d+)", file.name)
        if m:
            layer_entries[int(m.group(1))] = str(file)

    if not layer_entries:
        sys.exit(f"No layer files found in {trace_dir}")

    layer_indices = sorted(layer_entries.keys())
    combos = [(ep, ex) for ep in EP_SIZES for ex in EXTRA_SLOTS_RANGE]
    print(f"Found {len(layer_indices)} layers, latency model: {latency_json_path}")
    print(f"EPLB intervals: {EPLB_INTERVALS}")
    print(f"Sweep combos: {combos}")
    print(f"EP_SIZES: {EP_SIZES}, EXTRA_SLOTS_RANGE: {EXTRA_SLOTS_RANGE}")

    # Parse all layers once — shared across all combos
    all_records = {}
    all_pre_records = {}
    for layer_idx in layer_indices:
        full = parse_log_file(layer_entries[layer_idx])
        if STEP_RANGE is not None:
            m, n = STEP_RANGE
            all_records[layer_idx] = full[m:n]
            all_pre_records[layer_idx] = full[:m]
        else:
            all_records[layer_idx] = full
            all_pre_records[layer_idx] = []

    FILTER_MODES = ["decode", "prefill", "mix"]
    n_workers = min(len(layer_indices), cpu_count())
    combo_summaries = {}

    for ep_size, extra_slots in combos:
        combo_tag = f"ep{ep_size}_extra{extra_slots}"
        print(f"\n  Simulating {combo_tag}...")

        tasks = [(idx, all_records[idx], all_pre_records[idx],
                  ep_size, extra_slots, latency_json_path) for idx in layer_indices]

        with Pool(n_workers) as pool:
            results = pool.map(_process_layer, tasks)

        layer_summaries = {}
        all_sims = {}
        for layer_idx, sims, layer_summary in results:
            all_sims[layer_idx] = sims
            layer_summaries[layer_idx] = layer_summary

        system_summary = {}
        for filter_mode in FILTER_MODES:
            if all(filter_mode in all_sims[idx] for idx in layer_indices):
                layer_sims = {idx: all_sims[idx][filter_mode] for idx in layer_indices}
                system_summary[filter_mode] = _extract_system_summary(layer_sims, layer_indices)

        combo_summaries[(ep_size, extra_slots)] = {**layer_summaries, "system": system_summary}
        del all_sims, results
        print(f"  {combo_tag} done.")

    # Comparison plots only (per-combo plots skipped)
    print(f"\nGenerating comparison plots...")
    comp_dir = str(output_dir / "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    for filter_mode in FILTER_MODES:
        for idx in layer_indices:
            layer_comp_dir = os.path.join(comp_dir, f"layer_{idx}")
            os.makedirs(layer_comp_dir, exist_ok=True)
            plot_comparison_bar(combo_summaries, combos, filter_mode, idx, layer_comp_dir)
        plot_comparison_system_bar(combo_summaries, combos, filter_mode, comp_dir)

    generate_comparison_index(comp_dir, combos, layer_indices)

    print(f"\n完成: {output_dir}")


if __name__ == "__main__":
    main()
