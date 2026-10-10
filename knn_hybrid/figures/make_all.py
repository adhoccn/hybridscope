# -*- coding: utf-8 -*-
"""
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["fig1_pso_flowchart.py", "fig2_snr_curve.py", "fig3_accuracy_bars.py",
           "table1_runtime.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    a = ap.parse_args()
    for s in SCRIPTS:
        cmd = [sys.executable, os.path.join(HERE, s)]
        if a.regenerate and s != "fig1_pso_flowchart.py":
            cmd.append("--regenerate")
        print("\n>>>", " ".join(os.path.basename(c) for c in cmd))
        subprocess.run(cmd, check=True)
    print("\nall figures ->", os.path.join(HERE, "out"))


if __name__ == "__main__":
    main()
