#!/usr/bin/env python3
"""autoresearch driver for h3.c M1 Max optimization loop.

Usage:
    python3 autoresearch.py baseline          # run baseline bench, log CSV
    python3 autoresearch.py sweep             # sweep rows/iters, log CSV
    python3 autoresearch.py log <hypothesis>  # append hypothesis entry
    python3 autoresearch.py status            # show results + open hypotheses

Logs live in ~/h3.c/research/ (CSV rows + HYPOTHESES.md). This is the
measure -> hypothesize -> patch -> A/B -> keep/revert loop driver.
"""
import csv, os, subprocess, sys, time, datetime

H3_DIR = os.path.expanduser("~/h3.c")
RESEARCH = os.path.join(H3_DIR, "research")
CSV_PATH = os.path.join(RESEARCH, "bench_results.csv")
HYP_PATH = os.path.join(RESEARCH, "HYPOTHESES.md")
DEVICE = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True, text=True).stdout.strip()

os.makedirs(RESEARCH, exist_ok=True)

def run_bench(rows, iters, label=""):
    """Run bench_m1, return parsed rows."""
    cmd = [os.path.join(H3_DIR, "h3_bench_m1"), str(rows), str(iters)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=H3_DIR)
    if proc.returncode != 0:
        print("BENCH FAILED:", proc.stderr[:2000])
        return []
    rows_out = []
    for line in proc.stdout.splitlines():
        if line.startswith("#") or line.startswith("op,"):
            continue
        parts = line.split(",")
        if len(parts) < 7:
            continue
        rows_out.append({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "device": DEVICE, "rows": rows, "iters": iters, "label": label,
            "op": parts[0], "wall_ms": float(parts[1]),
            "perf_gflops": float(parts[2]), "bandwidth_GBps": float(parts[3]),
            "gpu_ms": float(parts[4]), "wait_ms": float(parts[5]),
            "mps_dispatch": int(parts[6]),
            "direct_dispatch": int(parts[7]),
        })
    return rows_out

def append_csv(rows):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if new:
            w.writeheader()
        w.writerows(rows)

def baseline():
    rows = run_bench(2048, 10, "baseline-2048")
    rows += run_bench(512, 10, "baseline-512")
    append_csv(rows)
    print(f"logged {len(rows)} rows -> {CSV_PATH}")
    show_latest("mlp_bf16_fused")

def sweep():
    for r in (256, 512, 1024, 2048):
        append_csv(run_bench(r, 10, f"sweep-rows{r}"))
    print("sweep done")

def log_hyp(text):
    with open(HYP_PATH, "a") as f:
        f.write(f"\n## [{datetime.datetime.now().isoformat(timespec='minutes')}] {DEVICE}\n{text}\n")
    print("hypothesis logged")

def show_latest(op_filter=None):
    if not os.path.exists(CSV_PATH):
        print("no results yet"); return
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    for r in rows[-30:]:
        if op_filter and r["op"] != op_filter:
            continue
        print(f"{r['timestamp']} | {r['label']:16} | {r['op']:28} | "
              f"wall={float(r['wall_ms']):9.2f}ms | gpu={float(r['gpu_ms']):7.2f}ms")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if cmd == "baseline": baseline()
    elif cmd == "sweep": sweep()
    elif cmd == "log": log_hyp(" ".join(sys.argv[2:]))
    elif cmd == "status": show_latest()
    else: print(__doc__)
