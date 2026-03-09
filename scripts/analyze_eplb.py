#!/usr/bin/env python3
"""EPLB Simulation Analysis Pipeline
====================================
Produces per-layer and system-level EPLB comparison plots:
1. Per-layer: No EPLB vs Static EPLB vs Dynamic EPLB(k) — time, utilization, memory
2. System-level: same comparison aggregated across all layers
3. Win-rate comparison: bar chart showing how often each strategy wins per metric

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
EXTRA_SLOTS_PER_RANK = 2  # slots_per_rank = local_experts + extra
EMA_DECAY = 0.5           # EMA blend: decay * history + (1-decay) * current_window
GPU_TFLOPS = 990          # bf16 peak TFLOPS (e.g. H100=990, A100=312)
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

    # Step 1: Replication — max-heap by per-replica load
    if total_slots > n:
        heap = [(-float(counts[i]), i) for i in range(n)]
        heapq.heapify(heap)
        assigned = n
        while assigned < total_slots:
            neg_load, eid = heapq.heappop(heap)
            replica_counts[eid] += 1
            heapq.heappush(heap, (-float(counts[eid]) / replica_counts[eid], eid))
            assigned += 1

    # Step 2: Placement — sort replicas largest-first, assign to least-loaded rank
    replicas = []
    for eid in range(n):
        slot_load = float(counts[eid]) / replica_counts[eid]
        for _ in range(replica_counts[eid]):
            replicas.append((slot_load, eid))
    replicas.sort(key=lambda x: -x[0])

    rank_heap = [(0.0, rid) for rid in range(num_groups)]
    heapq.heapify(rank_heap)
    rank_experts = [[] for _ in range(num_groups)]

    for slot_load, eid in replicas:
        cur_load, rid = heapq.heappop(rank_heap)
        rank_experts[rid].append(eid)
        heapq.heappush(rank_heap, (cur_load + slot_load, rid))

    return rank_experts, replica_counts


def compute_group_loads_replica(counts, rank_experts, replica_counts):
    """Compute per-group loads with expert replication (load split across replicas)."""
    num_groups = len(rank_experts)
    group_loads = np.zeros(num_groups)
    for g in range(num_groups):
        for eid in rank_experts[g]:
            group_loads[g] += counts[eid] / replica_counts[eid]
    return group_loads


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


def _simulate_no_eplb(filtered_records, n_experts, num_groups, t_coeff):
    """No EPLB: identity permutation throughout."""
    identity = np.arange(n_experts)
    n = len(filtered_records)
    time_arr = np.zeros(n)
    util_arr = np.zeros(n)
    for i, r in enumerate(filtered_records):
        gl = compute_group_loads(r["counts"], identity, num_groups)
        max_l, mean_l = gl.max(), gl.mean()
        time_arr[i] = max_l * t_coeff
        util_arr[i] = mean_l / max_l if max_l > 0 else 1.0
    return time_arr, util_arr


def _simulate_static_eplb(filtered_records, t_coeff, rk_experts, rk_replicas):
    """Static EPLB: fixed placement from pre-window stats, never re-adjusted."""
    n = len(filtered_records)
    time_arr = np.zeros(n)
    util_arr = np.zeros(n)
    for i, r in enumerate(filtered_records):
        gl = compute_group_loads_replica(r["counts"], rk_experts, rk_replicas)
        max_l, mean_l = gl.max(), gl.mean()
        time_arr[i] = max_l * t_coeff
        util_arr[i] = mean_l / max_l if max_l > 0 else 1.0
    return time_arr, util_arr


def _simulate_dynamic_eplb(filtered_records, eplb_interval, n_experts, num_groups,
                            slots_per_rank, t_coeff,
                            init_rk_experts, init_rk_replicas, init_ema):
    """Dynamic EPLB: starts from init placement (pre-window), re-adjusts every k steps."""
    n = len(filtered_records)
    time_arr = np.zeros(n)
    util_arr = np.zeros(n)
    window_acc = np.zeros(n_experts, dtype=np.float64)
    rk_experts = [list(g) for g in init_rk_experts]
    rk_replicas = init_rk_replicas.copy()
    ema_history = init_ema.copy() if init_ema is not None and init_ema.sum() > 0 else None

    for i, r in enumerate(filtered_records):
        counts = r["counts"]
        window_acc += counts
        if i > 0 and i % eplb_interval == 0:
            if ema_history is None:
                ema_history = window_acc.copy()
            else:
                ema_history = EMA_DECAY * ema_history + (1 - EMA_DECAY) * window_acc
            rk_experts, rk_replicas = eplb_replica_placement(ema_history, num_groups, slots_per_rank)
            window_acc[:] = 0
        gl = compute_group_loads_replica(counts, rk_experts, rk_replicas)
        max_l, mean_l = gl.max(), gl.mean()
        time_arr[i] = max_l * t_coeff
        util_arr[i] = mean_l / max_l if max_l > 0 else 1.0
    return time_arr, util_arr


# ──────────────────────────────────────────────────────────
# Plot: EPLB comparison — No EPLB vs Static vs Dynamic (per-layer)
# ──────────────────────────────────────────────────────────
def plot_eplb_compare(filtered_records, pre_filtered, x_mode, filter_mode,
                      layer_idx, output_dir, eplb_interval):
    """Per-layer: No EPLB vs Static EPLB vs Dynamic EPLB(k) for one interval."""
    if len(filtered_records) < 2:
        return

    x_vals, x_label = get_x_values(filtered_records, x_mode)
    n_experts = len(filtered_records[0]["counts"])
    num_groups = max(1, int(np.sqrt(n_experts)))
    epg = (n_experts + num_groups - 1) // num_groups
    slots_per_rank = epg + EXTRA_SLOTS_PER_RANK
    flops_per_token = 2 * 2 * HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE
    t_coeff = flops_per_token / (GPU_TFLOPS * 1e12) * 1000

    # Initial placement from pre-window stats
    init_rk, init_rep, init_agg = _compute_init_placement(
        pre_filtered, n_experts, num_groups, epg, slots_per_rank)

    # Three scenarios
    no_time, no_util = _simulate_no_eplb(filtered_records, n_experts, num_groups, t_coeff)
    st_time, st_util = _simulate_static_eplb(filtered_records, t_coeff, init_rk, init_rep)
    dy_time, dy_util = _simulate_dynamic_eplb(
        filtered_records, eplb_interval, n_experts, num_groups,
        slots_per_rank, t_coeff, init_rk, init_rep, init_agg)
    ideal_time = np.array([r["counts"].sum() / num_groups * t_coeff for r in filtered_records])

    # Memory
    MB = 1024 ** 2
    weight_per_expert = 3 * HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE * 2
    activation_per_token = (HIDDEN_SIZE + 2 * MOE_INTERMEDIATE_SIZE) * 2
    base_weight_mb = epg * weight_per_expert / MB
    replica_weight_mb = slots_per_rank * weight_per_expert / MB
    no_mem = no_time / t_coeff * activation_per_token / MB
    st_mem = st_time / t_coeff * activation_per_token / MB
    dy_mem = dy_time / t_coeff * activation_per_token / MB
    ideal_mem = ideal_time / t_coeff * activation_per_token / MB

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    lw = 1.2
    c_no, c_st, c_dy, c_ideal = "#e74c3c", "#3498db", "#9b59b6", "#2ecc71"

    ax1.plot(x_vals, no_time, label="No EPLB", color=c_no, linewidth=lw, alpha=0.8)
    ax1.plot(x_vals, st_time, label="Static EPLB", color=c_st, linewidth=lw, alpha=0.8)
    ax1.plot(x_vals, dy_time, label=f"Dynamic EPLB (k={eplb_interval})",
             color=c_dy, linewidth=lw, alpha=0.8)
    ax1.plot(x_vals, ideal_time, label="Ideal", color=c_ideal,
             linewidth=1.0, linestyle="--", alpha=0.6)
    ax1.set_ylabel("MoE Time (ms)\n(2×GroupGEMM, {:.0f} TFLOPS)".format(GPU_TFLOPS), fontsize=11)
    ax1.set_title(
        make_title(f"EPLB Comparison (k={eplb_interval}, +{EXTRA_SLOTS_PER_RANK} slots)",
                   x_mode, filter_mode, layer_idx),
        fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

    ax2.plot(x_vals, no_util * 100, label="No EPLB", color=c_no, linewidth=lw, alpha=0.8)
    ax2.plot(x_vals, st_util * 100, label="Static EPLB", color=c_st, linewidth=lw, alpha=0.8)
    ax2.plot(x_vals, dy_util * 100, label=f"Dynamic EPLB (k={eplb_interval})",
             color=c_dy, linewidth=lw, alpha=0.8)
    ax2.axhline(y=100, color=c_ideal, linestyle="--", linewidth=1.0, alpha=0.6, label="Ideal (100%)")
    ax2.set_ylabel("Flops Utilization (%)", fontsize=11)
    ax2.set_ylim(0, 105); ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)

    ax3.plot(x_vals, no_mem, label="No EPLB", color=c_no, linewidth=lw, alpha=0.8)
    ax3.plot(x_vals, st_mem, label="Static EPLB", color=c_st, linewidth=lw, alpha=0.8)
    ax3.plot(x_vals, dy_mem, label=f"Dynamic EPLB (k={eplb_interval})",
             color=c_dy, linewidth=lw, alpha=0.8)
    ax3.plot(x_vals, ideal_mem, label="Ideal", color=c_ideal,
             linewidth=1.0, linestyle="--", alpha=0.6)
    y_hi = max(no_mem.max(), st_mem.max(), dy_mem.max()) * 1.05
    ax3.set_ylim(0, y_hi)
    ax3.set_ylabel("Activation Memory per EP Group (MB)", fontsize=11)
    ax3.set_xlabel(x_label, fontsize=12)
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"eplb_k{eplb_interval}_{x_mode}_{filter_mode}.png"
    savefig(fig, os.path.join(output_dir, fname))


# ──────────────────────────────────────────────────────────
# Plot: System-level EPLB — No EPLB vs Static vs Dynamic
# ──────────────────────────────────────────────────────────
def plot_eplb_compare_system(all_records, all_pre_records, layer_indices, x_mode,
                             filter_mode, output_dir, eplb_interval):
    """System-level EPLB: flattened (step, layer), one figure per k interval."""
    filtered_per_layer = {}
    pre_per_layer = {}
    for idx in layer_indices:
        filtered = filter_records(all_records[idx], filter_mode)
        if len(filtered) < 2:
            return
        filtered_per_layer[idx] = filtered
        pre_per_layer[idx] = filter_records(all_pre_records[idx], filter_mode)

    n_steps = min(len(filtered_per_layer[idx]) for idx in layer_indices)
    n_layers = len(layer_indices)
    total_points = n_steps * n_layers

    first_layer = layer_indices[0]
    n_experts = len(filtered_per_layer[first_layer][0]["counts"])
    num_groups = max(1, int(np.sqrt(n_experts)))
    epg = (n_experts + num_groups - 1) // num_groups
    slots_per_rank = epg + EXTRA_SLOTS_PER_RANK
    flops_per_token = 2 * 2 * HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE
    t_coeff = flops_per_token / (GPU_TFLOPS * 1e12) * 1000
    MB = 1024 ** 2
    weight_per_expert = 3 * HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE * 2
    activation_per_token = (HIDDEN_SIZE + 2 * MOE_INTERMEDIATE_SIZE) * 2
    base_weight_mb = epg * weight_per_expert / MB
    replica_weight_mb = slots_per_rank * weight_per_expert / MB

    identity = np.arange(n_experts)
    x_flat = np.arange(total_points)

    # Per-layer initial placement from pre-window
    layer_init_rk, layer_init_rep, layer_init_agg = {}, {}, {}
    for idx in layer_indices:
        rk, rep, agg = _compute_init_placement(
            pre_per_layer[idx], n_experts, num_groups, epg, slots_per_rank)
        layer_init_rk[idx] = rk
        layer_init_rep[idx] = rep
        layer_init_agg[idx] = agg

    # ---- No EPLB & Ideal ----
    no_time = np.zeros(total_points)
    no_util = np.zeros(total_points)
    no_mem = np.zeros(total_points)
    ideal_time = np.zeros(total_points)
    ideal_mem = np.zeros(total_points)
    for step_i in range(n_steps):
        for li, idx in enumerate(layer_indices):
            flat_i = step_i * n_layers + li
            counts = filtered_per_layer[idx][step_i]["counts"]
            gl = compute_group_loads(counts, identity, num_groups)
            max_l, mean_l = gl.max(), gl.mean()
            no_time[flat_i] = max_l * t_coeff
            no_util[flat_i] = mean_l / max_l if max_l > 0 else 1.0
            no_mem[flat_i] = max_l * activation_per_token / MB
            il = counts.sum() / num_groups
            ideal_time[flat_i] = il * t_coeff
            ideal_mem[flat_i] = il * activation_per_token / MB

    # ---- Static EPLB ----
    st_time = np.zeros(total_points)
    st_util = np.zeros(total_points)
    for step_i in range(n_steps):
        for li, idx in enumerate(layer_indices):
            flat_i = step_i * n_layers + li
            counts = filtered_per_layer[idx][step_i]["counts"]
            gl = compute_group_loads_replica(counts, layer_init_rk[idx], layer_init_rep[idx])
            max_l, mean_l = gl.max(), gl.mean()
            st_time[flat_i] = max_l * t_coeff
            st_util[flat_i] = mean_l / max_l if max_l > 0 else 1.0
    st_mem = st_time / t_coeff * activation_per_token / MB

    # ---- Dynamic EPLB ----
    dy_time = np.zeros(total_points)
    dy_util = np.zeros(total_points)
    layer_window = {idx: np.zeros(n_experts, dtype=np.float64) for idx in layer_indices}
    layer_ema = {idx: agg.copy() if agg.sum() > 0 else None
                 for idx, agg in layer_init_agg.items()}
    layer_rk = {idx: [list(g) for g in layer_init_rk[idx]] for idx in layer_indices}
    layer_rep = {idx: layer_init_rep[idx].copy() for idx in layer_indices}

    for step_i in range(n_steps):
        triggered = (step_i > 0 and step_i % eplb_interval == 0)
        if triggered:
            for idx in layer_indices:
                w = layer_window[idx]
                h = layer_ema[idx]
                layer_ema[idx] = w.copy() if h is None else EMA_DECAY * h + (1 - EMA_DECAY) * w
                rk_e, rk_r = eplb_replica_placement(layer_ema[idx], num_groups, slots_per_rank)
                layer_rk[idx] = rk_e
                layer_rep[idx] = rk_r
                layer_window[idx][:] = 0
        for li, idx in enumerate(layer_indices):
            flat_i = step_i * n_layers + li
            counts = filtered_per_layer[idx][step_i]["counts"]
            layer_window[idx] += counts
            gl = compute_group_loads_replica(counts, layer_rk[idx], layer_rep[idx])
            max_l, mean_l = gl.max(), gl.mean()
            dy_time[flat_i] = max_l * t_coeff
            dy_util[flat_i] = mean_l / max_l if max_l > 0 else 1.0
    dy_mem = dy_time / t_coeff * activation_per_token / MB

    # ---- Plot ----
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    lw = 0.8
    c_no, c_st, c_dy, c_ideal = "#e74c3c", "#3498db", "#9b59b6", "#2ecc71"

    def _step_boundaries(ax):
        for s in range(1, n_steps):
            ax.axvline(x=s * n_layers, color="#ddd", linestyle="-", alpha=0.15, linewidth=0.5)

    mode_label = {"decode": "Decode Only", "prefill": "Prefill Only", "mix": "All (Mixed)"}
    title_base = (f"System ({n_layers} layers) | EPLB Comparison (k={eplb_interval}, "
                  f"+{EXTRA_SLOTS_PER_RANK} slots) | {mode_label[filter_mode]}")
    x_label = f"Layer-Step (each step = {n_layers} layers)"

    ax1.plot(x_flat, no_time, label="No EPLB", color=c_no, linewidth=lw, alpha=0.8)
    ax1.plot(x_flat, st_time, label="Static EPLB", color=c_st, linewidth=lw, alpha=0.8)
    ax1.plot(x_flat, dy_time, label=f"Dynamic EPLB (k={eplb_interval})",
             color=c_dy, linewidth=lw, alpha=0.8)
    ax1.plot(x_flat, ideal_time, label="Ideal", color=c_ideal,
             linewidth=0.6, linestyle="--", alpha=0.5)
    _step_boundaries(ax1)
    ax1.set_ylabel("MoE Time per Layer (ms)", fontsize=11)
    ax1.set_title(title_base, fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

    ax2.plot(x_flat, no_util * 100, label="No EPLB", color=c_no, linewidth=lw, alpha=0.8)
    ax2.plot(x_flat, st_util * 100, label="Static EPLB", color=c_st, linewidth=lw, alpha=0.8)
    ax2.plot(x_flat, dy_util * 100, label=f"Dynamic EPLB (k={eplb_interval})",
             color=c_dy, linewidth=lw, alpha=0.8)
    ax2.axhline(y=100, color=c_ideal, linestyle="--", linewidth=1.0, alpha=0.6, label="Ideal (100%)")
    _step_boundaries(ax2)
    ax2.set_ylabel("Flops Utilization (%)", fontsize=11)
    ax2.set_ylim(0, 105); ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)

    ax3.plot(x_flat, no_mem, label="No EPLB", color=c_no, linewidth=lw, alpha=0.8)
    ax3.plot(x_flat, st_mem, label="Static EPLB", color=c_st, linewidth=lw, alpha=0.8)
    ax3.plot(x_flat, dy_mem, label=f"Dynamic EPLB (k={eplb_interval})",
             color=c_dy, linewidth=lw, alpha=0.8)
    ax3.plot(x_flat, ideal_mem, label="Ideal", color=c_ideal,
             linewidth=0.6, linestyle="--", alpha=0.5)
    _step_boundaries(ax3)
    y_hi = max(no_mem.max(), st_mem.max(), dy_mem.max()) * 1.05
    ax3.set_ylim(0, y_hi)
    ax3.set_ylabel("Activation Memory per EP Group (MB)", fontsize=11)
    ax3.set_xlabel(x_label, fontsize=12)
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"eplb_system_k{eplb_interval}_{x_mode}_{filter_mode}.png"
    savefig(fig, os.path.join(output_dir, fname))


# ──────────────────────────────────────────────────────────
# Plot: EPLB Frequency Win-Rate Comparison (per-layer)
# ──────────────────────────────────────────────────────────
def plot_eplb_winrate(filtered_records, pre_filtered, x_mode, filter_mode,
                      layer_idx, output_dir):
    """Per-layer win-rate bar chart: for each metric, which strategy wins most often."""
    if len(filtered_records) < 2:
        return

    n_experts = len(filtered_records[0]["counts"])
    num_groups = max(1, int(np.sqrt(n_experts)))
    epg = (n_experts + num_groups - 1) // num_groups
    slots_per_rank = epg + EXTRA_SLOTS_PER_RANK
    flops_per_token = 2 * 2 * HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE
    t_coeff = flops_per_token / (GPU_TFLOPS * 1e12) * 1000
    MB = 1024 ** 2
    activation_per_token = (HIDDEN_SIZE + 2 * MOE_INTERMEDIATE_SIZE) * 2

    # Initial placement from pre-window stats
    init_rk, init_rep, init_agg = _compute_init_placement(
        pre_filtered, n_experts, num_groups, epg, slots_per_rank)

    # Simulate all strategies
    no_time, no_util = _simulate_no_eplb(filtered_records, n_experts, num_groups, t_coeff)
    st_time, st_util = _simulate_static_eplb(filtered_records, t_coeff, init_rk, init_rep)

    strategy_names = ["No EPLB", "Static EPLB"]
    all_time = [no_time, st_time]
    all_util = [no_util, st_util]

    for k in EPLB_INTERVALS:
        dy_time, dy_util = _simulate_dynamic_eplb(
            filtered_records, k, n_experts, num_groups,
            slots_per_rank, t_coeff, init_rk, init_rep, init_agg)
        strategy_names.append(f"Dynamic (k={k})")
        all_time.append(dy_time)
        all_util.append(dy_util)

    # Compute memory from time
    all_mem = [t / t_coeff * activation_per_token / MB for t in all_time]

    n_steps = len(filtered_records)
    n_strategies = len(strategy_names)

    # Compute win counts per metric
    # Time: lowest wins; Utilization: highest wins; Memory: lowest wins
    time_wins = np.zeros(n_strategies)
    util_wins = np.zeros(n_strategies)
    mem_wins = np.zeros(n_strategies)

    for i in range(n_steps):
        time_vals = [all_time[s][i] for s in range(n_strategies)]
        util_vals = [all_util[s][i] for s in range(n_strategies)]
        mem_vals = [all_mem[s][i] for s in range(n_strategies)]

        time_wins[np.argmin(time_vals)] += 1
        util_wins[np.argmax(util_vals)] += 1
        mem_wins[np.argmin(mem_vals)] += 1

    # Convert to percentages
    time_pct = time_wins / n_steps * 100
    util_pct = util_wins / n_steps * 100
    mem_pct = mem_wins / n_steps * 100

    # Plot grouped bar chart
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

    # Colors for strategies
    base_colors = ["#e74c3c", "#3498db"]  # No EPLB, Static
    dynamic_cmap = plt.cm.Purples
    for i, k in enumerate(EPLB_INTERVALS):
        base_colors.append(dynamic_cmap(0.4 + 0.15 * i))

    x_pos = np.arange(n_strategies)
    bar_width = 0.6

    mode_label = {"decode": "Decode Only", "prefill": "Prefill Only", "mix": "All (Mixed)"}

    for ax, pct, metric_name, ylabel in [
        (ax1, time_pct, "MoE Time", "Win Rate (%) — Lowest Time"),
        (ax2, util_pct, "Flops Utilization", "Win Rate (%) — Highest Utilization"),
        (ax3, mem_pct, "Activation Memory", "Win Rate (%) — Lowest Memory"),
    ]:
        bars = ax.bar(x_pos, pct, bar_width, color=base_colors, edgecolor="white", linewidth=0.5)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(strategy_names, fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis="y")
        # Add percentage labels on bars
        for bar, val in zip(bars, pct):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_title(
        make_title(f"EPLB Win-Rate Comparison (+{EXTRA_SLOTS_PER_RANK} slots)",
                   x_mode, filter_mode, layer_idx),
        fontsize=14, fontweight="bold")

    plt.tight_layout()
    fname = f"eplb_winrate_{x_mode}_{filter_mode}.png"
    savefig(fig, os.path.join(output_dir, fname))


# ──────────────────────────────────────────────────────────
# Plot: Dynamic vs Static EPLB — relative change statistics
# ──────────────────────────────────────────────────────────
def plot_eplb_dynamic_vs_static(filtered_records, pre_filtered, x_mode, filter_mode,
                                layer_idx, output_dir):
    """Per-layer: for each Dynamic EPLB(k), show relative change vs Static EPLB.

    3 subplots (time, utilization, memory). Each subplot is a grouped bar chart:
    - X-axis: k values from EPLB_INTERVALS
    - Bars: mean relative change (%)
    - Error bars: p5–p95 range
    - Annotations: median, max improvement, max degradation
    """
    if len(filtered_records) < 2:
        return

    n_experts = len(filtered_records[0]["counts"])
    num_groups = max(1, int(np.sqrt(n_experts)))
    epg = (n_experts + num_groups - 1) // num_groups
    slots_per_rank = epg + EXTRA_SLOTS_PER_RANK
    flops_per_token = 2 * 2 * HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE
    t_coeff = flops_per_token / (GPU_TFLOPS * 1e12) * 1000
    MB = 1024 ** 2
    activation_per_token = (HIDDEN_SIZE + 2 * MOE_INTERMEDIATE_SIZE) * 2

    # Initial placement
    init_rk, init_rep, init_agg = _compute_init_placement(
        pre_filtered, n_experts, num_groups, epg, slots_per_rank)

    # Static EPLB baseline
    st_time, st_util = _simulate_static_eplb(filtered_records, t_coeff, init_rk, init_rep)
    st_mem = st_time / t_coeff * activation_per_token / MB

    # Per-k dynamic simulations — compute relative change vs static
    k_labels = [str(k) for k in EPLB_INTERVALS]
    # For each metric: list of arrays, one per k
    time_rel = []  # (dynamic - static) / static * 100, negative = better
    util_rel = []  # (dynamic - static) / static * 100, positive = better
    mem_rel = []   # (dynamic - static) / static * 100, negative = better

    for k in EPLB_INTERVALS:
        dy_time, dy_util = _simulate_dynamic_eplb(
            filtered_records, k, n_experts, num_groups,
            slots_per_rank, t_coeff, init_rk, init_rep, init_agg)
        dy_mem = dy_time / t_coeff * activation_per_token / MB

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

        # Error bars: p5 to p95
        err_low = means - p5s
        err_high = p95s - means

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
        ax.grid(True, alpha=0.3, axis="y")

    ax1.set_title(
        make_title(f"Dynamic EPLB vs Static EPLB (+{EXTRA_SLOTS_PER_RANK} slots)",
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

        # Win-rate chart
        html += "<details open><summary>Win-Rate Comparison</summary>\n<div class='grid'>\n"
        for x_mode in x_modes:
            for fm in filter_modes:
                fname = f"eplb_winrate_{x_mode}_{fm}.png"
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
    for k in EPLB_INTERVALS:
        title = f"k={k}: No EPLB vs Static vs Dynamic (System)"
        html += f"<details open><summary>{title}</summary>\n<div class='grid'>\n"
        for x_mode in x_modes:
            for fm in filter_modes:
                fname = f"eplb_system_k{k}_{x_mode}_{fm}.png"
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

    X_MODES = ["step"]
    FILTER_MODES = ["decode", "prefill", "mix"]

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

    for layer_idx in layer_indices:
        filepath = layer_entries[layer_idx]
        records = all_records[layer_idx]
        pre_records = all_pre_records[layer_idx]

        print(f"\n{'='*60}")
        print(f"Processing Layer {layer_idx}: {filepath}")
        print(f"{'='*60}")

        if STEP_RANGE is not None:
            m, n = STEP_RANGE
            print(f"  Window [{m}, {n}): {len(records)} records, "
                  f"pre-window: {len(pre_records)} records")
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
                pre_filtered = filter_records(pre_records, filter_mode)

                print(f"\n  --- {x_mode} / {filter_mode} ({len(filtered)} records) ---")

                # EPLB comparison — per k interval
                for k in EPLB_INTERVALS:
                    plot_eplb_compare(filtered, pre_filtered, x_mode, filter_mode,
                                     layer_idx, layer_dir, k)

                # Win-rate comparison
                plot_eplb_winrate(filtered, pre_filtered, x_mode, filter_mode,
                                 layer_idx, layer_dir)

                # Dynamic vs Static relative change
                plot_eplb_dynamic_vs_static(filtered, pre_filtered, x_mode, filter_mode,
                                            layer_idx, layer_dir)

    # System-level EPLB comparison — per k interval
    print(f"\n{'='*60}")
    print(f"System-level EPLB simulation ({len(layer_indices)} layers)")
    print(f"{'='*60}")
    for x_mode in X_MODES:
        for filter_mode in FILTER_MODES:
            for k in EPLB_INTERVALS:
                plot_eplb_compare_system(all_records, all_pre_records, layer_indices,
                                        x_mode, filter_mode, str(output_dir), k)

    # Generate EPLB HTML index
    generate_eplb_index(str(output_dir), layer_indices)
    print(f"\n{'='*60}")
    print(f"完成: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
