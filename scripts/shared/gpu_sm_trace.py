#!/usr/bin/env python3
"""
GPU SM Utilization Trace — nsys SQLite → Perfetto JSON

从实验 metadata 中加载 nsys 导出的 SQLite 数据，
分析 GPU SM 利用率并生成 Chrome Trace JSON（用 Perfetto 打开）。

使用前：
  1. nsys export --type sqlite --output trace.sqlite trace.nsys-rep
  2. python labshelf.py add-data <exp> trace.sqlite --name <DATA_KEY> --format sqlite
  3. 修改下方 CONFIG 中的参数
  4. python labshelf.py run-other <exp> <other_name>

自动生成 by labshelf.py add-other
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.shared.loaders import load_data
import yaml

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — 按需修改
# ══════════════════════════════════════════════════════════════════════════════

DATA_KEY    = "trace"       # metadata.yaml 中 data 段的逻辑名（sqlite 格式）
WINDOW_MS   = 0.1           # 时间窗口大小 (ms)
DEVICE_ID   = 0             # GPU device ID
NUM_SMS     = 108           # GPU SM 数量 (A100/A800 = 108)
NVTX_FILTER = None          # NVTX range 名称过滤，None 表示不过滤
START_MS    = None           # 起始时间偏移 (ms)，None 表示从头开始
END_MS      = None           # 结束时间偏移 (ms)，None 表示到末尾


# ══════════════════════════════════════════════════════════════════════════════
# GA100 / A100 / A800 SM resource limits  (override NUM_SMS if needed)
# ══════════════════════════════════════════════════════════════════════════════

GPU_SPEC = dict(
    sm_max_threads=2048,
    sm_max_blocks=32,
    sm_max_regs=65536,
    sm_max_smem=167936,   # bytes (GA100 = 164 KB)
    warp_size=32,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_kernels(conn: sqlite3.Connection, device_id: int) -> pd.DataFrame:
    """Load kernel records from CUPTI table."""
    query = """
        SELECT
            start, end,
            gridX, gridY, gridZ,
            blockX, blockY, blockZ,
            registersPerThread,
            staticSharedMemory,
            dynamicSharedMemory,
            shortName AS kernel_name,
            deviceId
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE deviceId = ?
        ORDER BY start
    """
    df = pd.read_sql(query, conn, params=(device_id,))
    if df.empty:
        sys.exit(f"[ERROR] No kernels found for deviceId={device_id}")
    return df


def get_nvtx_range(conn: sqlite3.Connection, nvtx_name: str):
    """Return (start_ns, end_ns) for the first matching NVTX range."""
    df = pd.read_sql(
        "SELECT start, end FROM NVTX_EVENTS WHERE text = ? LIMIT 1",
        conn, params=(nvtx_name,)
    )
    if df.empty:
        avail = pd.read_sql(
            "SELECT DISTINCT text FROM NVTX_EVENTS WHERE text IS NOT NULL LIMIT 30",
            conn
        )["text"].tolist()
        sys.exit(
            f"[ERROR] NVTX range '{nvtx_name}' not found.\n"
            f"Available ranges (first 30): {avail}"
        )
    row = df.iloc[0]
    return int(row["start"]), int(row["end"])


def theoretical_occupancy(block_size, regs_per_thread, smem_per_block, spec):
    """Compute theoretical SM occupancy (0.0 – 1.0) from launch parameters."""
    if block_size <= 0:
        return float("nan")

    warp_size      = spec["warp_size"]
    sm_max_threads = spec["sm_max_threads"]
    sm_max_blocks  = spec["sm_max_blocks"]
    sm_max_regs    = spec["sm_max_regs"]
    sm_max_smem    = spec["sm_max_smem"]
    sm_max_warps   = sm_max_threads // warp_size
    warps_per_block = max(1, (block_size + warp_size - 1) // warp_size)

    limit_threads = sm_max_threads // block_size
    limit_blocks  = sm_max_blocks
    limit_regs    = (sm_max_regs // (regs_per_thread * block_size)
                     if regs_per_thread > 0 else sm_max_blocks)
    limit_smem    = (sm_max_smem // smem_per_block
                     if smem_per_block > 0 else sm_max_blocks)

    blocks_per_sm = max(0, min(limit_threads, limit_blocks, limit_regs, limit_smem))
    warps_per_sm  = blocks_per_sm * warps_per_block
    return min(warps_per_sm / sm_max_warps, 1.0)


def build_trace(kernels: pd.DataFrame, window_ms: float,
                num_sms: int, spec: dict) -> pd.DataFrame:
    """
    For each time window, compute:
      - time_util       : fraction of window where >= 1 kernel was running (0-100%)
      - sm_active_pct   : approx SM active % (clipped to num_sms)
      - avg_occupancy   : weighted-avg theoretical occupancy of active kernels
      - concurrent_kernels : number of overlapping kernels
    """
    t_min_ns = kernels["start"].min()
    kernels = kernels.copy()
    kernels["start_ms"] = (kernels["start"] - t_min_ns) / 1e6
    kernels["end_ms"]   = (kernels["end"]   - t_min_ns) / 1e6
    kernels["block_size"]     = (kernels["blockX"] * kernels["blockY"] * kernels["blockZ"]).clip(lower=1)
    kernels["smem_per_block"] = kernels["staticSharedMemory"] + kernels["dynamicSharedMemory"]
    kernels["total_blocks"]   = (kernels["gridX"] * kernels["gridY"] * kernels["gridZ"]).clip(lower=1)

    kernels["occupancy"] = kernels.apply(
        lambda r: theoretical_occupancy(
            r["block_size"], r["registersPerThread"],
            r["smem_per_block"], spec
        ), axis=1
    )

    t_max = kernels["end_ms"].max()
    bins  = np.arange(0, t_max + window_ms, window_ms)

    records = []
    for i in range(len(bins) - 1):
        t0, t1 = bins[i], bins[i + 1]
        mask  = (kernels["start_ms"] < t1) & (kernels["end_ms"] > t0)
        chunk = kernels[mask]

        if chunk.empty:
            records.append({
                "time_ms": t0,
                "time_util": 0.0,
                "sm_active_pct": 0.0,
                "avg_occupancy": 0.0,
                "concurrent_kernels": 0,
            })
            continue

        overlap_ms = chunk.apply(
            lambda r: max(0.0, min(r["end_ms"], t1) - max(r["start_ms"], t0)),
            axis=1
        )
        total_ms = overlap_ms.sum()

        time_util = min(total_ms / window_ms * 100.0, 100.0)
        sm_active = (chunk["total_blocks"].clip(upper=num_sms) * overlap_ms).sum() / total_ms
        sm_active_pct = sm_active / num_sms * 100.0
        avg_occ = (chunk["occupancy"].fillna(0) * overlap_ms).sum() / total_ms

        records.append({
            "time_ms": t0,
            "time_util": time_util,
            "sm_active_pct": sm_active_pct,
            "avg_occupancy": avg_occ * 100.0,
            "concurrent_kernels": len(chunk),
        })

    return pd.DataFrame(records)


def export_chrome_trace(
    util_df: pd.DataFrame,
    kernels_df: pd.DataFrame,
    output_path: str,
    window_ms: float,
    num_sms: int,
):
    """
    Export as Chrome Tracing JSON for Perfetto.
    Open with  https://ui.perfetto.dev  (drag & drop)

    Tracks produced:
      PID 1 – Utilization counters
      PID 2 – Individual kernel spans
    """
    events = []

    # ── PID 1 : Counter tracks ───────────────────────────────────────────────
    PID_UTIL = 1
    events.append({"ph": "M", "pid": PID_UTIL, "tid": 0,
                   "name": "process_name",
                   "args": {"name": "GPU Utilization Counters"}})

    for _, row in util_df.iterrows():
        ts_us     = row["time_ms"] * 1_000.0
        is_active = 1 if row["time_util"] > 0 else 0

        events.append({
            "ph": "C", "pid": PID_UTIL, "tid": 0, "ts": ts_us,
            "name": "① GPU Active Rate  [mean = 整卡利用率]",
            "args": {"gpu_active_0_or_1": is_active}
        })
        events.append({
            "ph": "C", "pid": PID_UTIL, "tid": 0, "ts": ts_us,
            "name": "② SM Util Rate  [mean = SM利用率]",
            "args": {"sm_util_0_to_1": round(row["sm_active_pct"] / 100.0, 6)}
        })
        events.append({
            "ph": "C", "pid": PID_UTIL, "tid": 0, "ts": ts_us,
            "name": "③ Theoretical Occupancy (ref)",
            "args": {"occupancy_0_to_1": round(row["avg_occupancy"] / 100.0, 6)}
        })
        events.append({
            "ph": "C", "pid": PID_UTIL, "tid": 0, "ts": ts_us,
            "name": "④ Concurrent Kernels (ref)",
            "args": {"count": int(row["concurrent_kernels"])}
        })

    # 在最后一个 bucket 结束处补零
    last_ts_us = (util_df["time_ms"].iloc[-1] + window_ms) * 1_000.0
    for name, key, val in [
        ("① GPU Active Rate  [mean = 整卡利用率]",    "gpu_active_0_or_1", 0),
        ("② SM Util Rate  [mean = SM利用率]",         "sm_util_0_to_1",    0.0),
        ("③ Theoretical Occupancy (ref)",              "occupancy_0_to_1",  0.0),
        ("④ Concurrent Kernels (ref)",                 "count",             0),
    ]:
        events.append({"ph": "C", "pid": PID_UTIL, "tid": 0,
                       "ts": last_ts_us, "name": name, "args": {key: val}})

    # ── PID 2 : Individual kernel spans ──────────────────────────────────────
    PID_KERN = 2
    events.append({"ph": "M", "pid": PID_KERN, "tid": 0,
                   "name": "process_name",
                   "args": {"name": "CUDA Kernels"}})

    kernels_sorted = kernels_df.sort_values("start").reset_index(drop=True)
    lane_end = []

    for _, k in kernels_sorted.iterrows():
        start_ns = int(k["start"])
        end_ns   = int(k["end"])
        dur_us   = max(1, (end_ns - start_ns) // 1_000)
        ts_us    = start_ns / 1_000.0

        assigned = None
        for i, le in enumerate(lane_end):
            if start_ns >= le:
                assigned = i
                lane_end[i] = end_ns
                break
        if assigned is None:
            assigned = len(lane_end)
            lane_end.append(end_ns)

        events.append({"ph": "M", "pid": PID_KERN, "tid": assigned,
                       "name": "thread_name",
                       "args": {"name": f"Lane {assigned}"}})
        events.append({
            "ph": "X", "pid": PID_KERN, "tid": assigned,
            "ts": ts_us, "dur": dur_us,
            "name": str(k.get("kernel_name", "kernel")),
            "args": {
                "grid":   f"{int(k['gridX'])}x{int(k['gridY'])}x{int(k['gridZ'])}",
                "block":  f"{int(k['blockX'])}x{int(k['blockY'])}x{int(k['blockZ'])}",
                "regs":   int(k["registersPerThread"]),
                "smem_B": int(k["staticSharedMemory"]) + int(k["dynamicSharedMemory"]),
            }
        })

    # ── Assemble and write ───────────────────────────────────────────────────
    trace = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "otherData": {
            "window_ms":     window_ms,
            "num_sms":       num_sms,
            "total_kernels": len(kernels_df),
            "duration_ms":   round(util_df["time_ms"].max(), 3),
            "how_to_read": (
                "Select a time range in Perfetto. "
                "In the Details panel, look for counter mean values: "
                "'① GPU Active Rate' mean = 整卡利用率, "
                "'② SM Util Rate' mean = SM利用率."
            ),
        }
    }

    with open(output_path, "w") as f:
        json.dump(trace, f, separators=(",", ":"))

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[✓] Chrome trace saved → {output_path}  ({size_mb:.1f} MB)")
    print()
    print("  打开方式: https://ui.perfetto.dev  (drag & drop)")
    print()
    print("  ┌─ 如何读取利用率 ─────────────────────────────────────────────┐")
    print("  │  1. 在时间轴上框选感兴趣的时间范围                            │")
    print("  │  2. 底部 Details 面板 → 找到以下 counter track：              │")
    print("  │       '① GPU Active Rate'  →  mean 列 = 整卡利用率           │")
    print("  │       '② SM Util Rate'     →  mean 列 = SM 利用率            │")
    print("  └──────────────────────────────────────────────────────────────┘")


def print_summary(df: pd.DataFrame):
    total  = len(df)
    active = (df["time_util"] > 0).sum()
    print("\n── Summary ──────────────────────────────────────────")
    print(f"  Total windows       : {total}")
    print(f"  Active windows      : {active}  ({active/total*100:.1f}%)")
    print(f"  Avg time util       : {df['time_util'].mean():.1f}%")
    print(f"  Avg SM active (est) : {df['sm_active_pct'].mean():.1f}%")
    print(f"  Avg occupancy       : {df['avg_occupancy'].mean():.1f}%")
    print(f"  Peak time util      : {df['time_util'].max():.1f}%")
    low = df[df["time_util"] < 50]
    print(f"  Low-util windows (<50%) : {len(low)}  ({len(low)/total*100:.1f}%)")
    print("─────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    exp_dir = Path(__file__).resolve().parents[1]
    metadata = yaml.safe_load((exp_dir / "metadata.yaml").read_text())

    # --- 加载数据 ---
    conn = load_data(exp_dir, metadata, DATA_KEY)

    # --- load kernels ---
    kernels = load_kernels(conn, DEVICE_ID)
    print(f"[*] Loaded {len(kernels):,} kernels from device {DEVICE_ID}")

    t_min_ns = kernels["start"].min()

    # --- time filtering ---
    if NVTX_FILTER:
        ns0, ns1 = get_nvtx_range(conn, NVTX_FILTER)
        kernels = kernels[(kernels["start"] >= ns0) & (kernels["end"] <= ns1)]
        t_min_ns = ns0
        print(f"[*] Filtered to NVTX range '{NVTX_FILTER}': "
              f"{(ns1-ns0)/1e6:.2f} ms, {len(kernels):,} kernels")
    else:
        if START_MS is not None:
            ns0 = t_min_ns + int(START_MS * 1e6)
            kernels = kernels[kernels["end"] >= ns0]
            t_min_ns = ns0
        if END_MS is not None:
            ns1 = kernels["start"].min() + int(END_MS * 1e6)
            kernels = kernels[kernels["start"] <= ns1]

    if kernels.empty:
        sys.exit("[ERROR] No kernels remain after time filtering.")

    # re-zero timestamps
    kernels = kernels.copy()
    kernels["start"] = kernels["start"] - t_min_ns
    kernels["end"]   = kernels["end"]   - t_min_ns

    # --- build trace ---
    print(f"[*] Building trace  window={WINDOW_MS}ms  num_sms={NUM_SMS}")
    df = build_trace(kernels, WINDOW_MS, NUM_SMS, GPU_SPEC)

    print_summary(df)

    # --- 保存 ---
    output_path = exp_dir / "others" / OUTPUT_TRACE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_chrome_trace(
        util_df=df,
        kernels_df=kernels,
        output_path=str(output_path),
        window_ms=WINDOW_MS,
        num_sms=NUM_SMS,
    )

    conn.close()


if __name__ == "__main__":
    main()
