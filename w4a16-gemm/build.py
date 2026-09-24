"""Compile w4a16_gemm.hip into libw4a16.so next to this script with hipcc."""
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gfx1100,gfx1101,gfx1102", help="comma-separated gfx11 targets")
    args = parser.parse_args()
    cmd = ["hipcc", "-O3", "-shared", "-fPIC", *(f"--offload-arch={a}" for a in args.arch.split(",")),
           "-o", str(HERE / "libw4a16.so"), str(HERE / "w4a16_gemm.hip")]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
