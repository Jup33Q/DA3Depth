"""M7 shader parity: DepthShaderKit GPU kernels vs CPU references (metal compute).

Builds the DepthShaderKit package (release) and runs the depthshader-parity entry
point, which emits JSON; prints a per-case table and gates on it.

Gate: every case within its tolerance (exact cases require diff == 0).
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "DepthShaderKit")
XCODE_DEVELOPER = "/Applications/Xcode.app/Contents/Developer"


def build():
    env = dict(os.environ, DEVELOPER_DIR=XCODE_DEVELOPER)
    r = subprocess.run(["swift", "build", "-c", "release"], cwd=PKG, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        print("M7 FAIL (build error)")
        sys.exit(1)


def main():
    build()
    binary = os.path.join(PKG, ".build", "release", "depthshader-parity")
    r = subprocess.run([binary, "--json"], capture_output=True, text=True)
    if r.returncode not in (0, 1) or not r.stdout.strip():
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        print("M7 FAIL (parity runner crashed)")
        sys.exit(1)

    report = json.loads(r.stdout)
    print(f"{'case':<22} {'size':<26} {'max diff':>11} {'tol':>9}  result")
    print("-" * 78)
    for c in report["cases"]:
        print(f"{c['name']:<22} {c['size']:<26} {c['max_abs_diff']:>11.3e} "
              f"{c['tolerance']:>9.1e}  {'PASS' if c['pass'] else 'FAIL'}")

    n = len(report["cases"])
    if report["all_pass"]:
        print(f"M7 PASS ({n}/{n} cases)")
        sys.exit(0)
    failed = sum(1 for c in report["cases"] if not c["pass"])
    print(f"M7 FAIL ({failed} of {n} cases over tolerance)")
    sys.exit(1)


if __name__ == "__main__":
    main()
