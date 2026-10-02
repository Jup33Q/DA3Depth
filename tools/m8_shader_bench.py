"""M8 shader bench: DepthShaderKit kernel benchmarks at 4K (+ optional extra size).

Runs the release depthshader-bench with --json and gates on the roadmap acceptance
thresholds:
  affine 4K GPU-only  < 1 ms
  fuse   4K GPU-only  < 1 ms
  drag-sim step mean  < 16.6 ms  (60fps)

Usage: python3 tools/m8_shader_bench.py [--extra-size 1920x1080]
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "DepthShaderKit")
XCODE_DEVELOPER = "/Applications/Xcode.app/Contents/Developer"

GATES = [  # (kernel, metric, threshold ms) — only apply at 4K
    ("affine", "gpu_ms", 1.0),
    ("fuse", "gpu_ms", 1.0),
]
DRAG_GATE_MS = 16.6


def build():
    env = dict(os.environ, DEVELOPER_DIR=XCODE_DEVELOPER)
    r = subprocess.run(["swift", "build", "-c", "release"], cwd=PKG, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        print("M8 FAIL (build error)")
        sys.exit(1)


def run_bench(size):
    binary = os.path.join(PKG, ".build", "release", "depthshader-bench")
    cmd = [binary, "--json", "--size", size]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        print(f"M8 FAIL (bench crashed at {size})")
        sys.exit(1)
    return json.loads(r.stdout)


def report_and_gate(report, gate):
    ok = True
    w, h = report["width"], report["height"]
    print(f"\n--- shader bench ({w}x{h}, {report['iterations']} iters, best-of) ---")
    print(f"{'kernel':<10} {'wall ms':>9} {'gpu ms':>9} {'MPix/s':>10}")
    for k in report["kernels"]:
        gpu_ms = k.get("gpu_ms")
        gpu = f"{gpu_ms:9.3f}" if gpu_ms is not None else "      n/a"
        print(f"{k['name']:<10} {k['wall_ms']:9.3f} {gpu} {k['mpix_s']:10.1f}")
    d = report["drag_sim"]
    print(f"drag-sim  mean {d['mean_ms']:.2f} ms  p95 {d['p95_ms']:.2f} ms  "
          f"max {d['max_ms']:.2f} ms  ({d['steps']} x {d['step_deg']} deg)")
    if gate:
        for name, metric, limit in GATES:
            k = next(k for k in report["kernels"] if k["name"] == name)
            status = "PASS" if k[metric] < limit else "FAIL"
            ok = ok and k[metric] < limit
            print(f"gate: {name} {metric} {k[metric]:.3f} ms < {limit} ms  {status}")
        status = "PASS" if d["mean_ms"] < DRAG_GATE_MS else "FAIL"
        ok = ok and d["mean_ms"] < DRAG_GATE_MS
        print(f"gate: drag-sim mean {d['mean_ms']:.2f} ms < {DRAG_GATE_MS} ms  {status}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="M8 shader bench (DepthShaderKit)")
    ap.add_argument("--extra-size", metavar="WxH",
                    help="also bench at this size (no gates applied)")
    args = ap.parse_args()

    build()
    ok = report_and_gate(run_bench("3840x2160"), gate=True)
    if args.extra_size:
        report_and_gate(run_bench(args.extra_size), gate=False)

    print("M8 PASS" if ok else "M8 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
