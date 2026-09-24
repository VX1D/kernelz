"""Compile q4_0_skinny.hip into libq4_0_skinny.so next to this script with hipcc."""
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gfx1100,gfx1101,gfx1102", help="comma-separated gfx11 targets")
    args = parser.parse_args()
    cmd = ["hipcc", "-O3", "-shared", "-fPIC", *(f"--offload-arch={a}" for a in args.arch.split(",")),
           "-o", str(HERE / "libq4_0_skinny.so"), str(HERE / "q4_0_skinny.hip")]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
