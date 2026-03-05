import subprocess
import time
from datetime import timedelta
import platform
import torch


def print_banner(title):
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60 + "\n")


def run(cmd, label):
    print_banner(f"STARTING: {label}")

    start = time.time()
    subprocess.run(cmd, shell=True, check=True)
    end = time.time()

    elapsed = end - start

    print_banner(f"FINISHED: {label}")
    print(f"{label} runtime: {str(timedelta(seconds=int(elapsed)))}\n")

    return elapsed


def main():

    print_banner("PETS Artifact Reproduction (CIFAR-10)")
    print(f"Python version : {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")
    print("=" * 60)

    total_start = time.time()
    timings = {}

    COMMON = "--dataset cifar10 --model alexnet --attack all --epochs 30"

    # -------------------------------------------------
    # 1. HAMP
    # -------------------------------------------------
    print("\nNow running HAMP defense...")
    timings["HAMP"] = run(
        f"python integrated_run.py {COMMON} --defense hamp",
        "HAMP Defense"
    )

    # -------------------------------------------------
    # 2. DynaNoise
    # -------------------------------------------------
    print("\nNow running DynaNoise defense...")
    timings["DynaNoise"] = run(
        f"python integrated_run.py {COMMON} "
        "--defense dyna --bv 0.3 --ls 2.0 --t 10.0",
        "DynaNoise Defense"
    )

    total_end = time.time()
    total_elapsed = total_end - total_start

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    print_banner("TIMING SUMMARY")

    for k, v in timings.items():
        print(f"{k:<15} : {str(timedelta(seconds=int(v)))}")

    print("-" * 40)
    print(f"{'TOTAL':<15} : {str(timedelta(seconds=int(total_elapsed)))}")
    print("=" * 60 + "\n")

    with open("timing_summary.txt", "w") as f:
        f.write("PETS Artifact Timing Summary\n")
        f.write("=" * 60 + "\n")
        for k, v in timings.items():
            f.write(f"{k:<15} : {str(timedelta(seconds=int(v)))}\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'TOTAL':<15} : {str(timedelta(seconds=int(total_elapsed)))}\n")

    print("Artifact reproduction completed successfully.")
    print("Timing written to timing_summary.txt\n")


if __name__ == "__main__":
    main()