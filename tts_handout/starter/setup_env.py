#!/usr/bin/env python3
"""One-shot environment setup for the voice-cloning assignment.

Run this BEFORE the hour starts (it downloads ~1 GB of Python packages):

    python setup_env.py          # install everything, then verify
    python setup_env.py --check  # verify only, no installs

You also need espeak-ng installed on your machine:
    macOS:    brew install espeak-ng
    Linux:    sudo apt install espeak-ng
    Windows:  installer from github.com/espeak-ng/espeak-ng/releases

When it prints READY, you are set.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys

# The versions this kit was built and tested against. numba/llvmlite are
# pinned because newer resolutions drag in unbuildable versions; setuptools
# is pinned because resemblyzer needs pkg_resources (removed in 81+).
PKGS = [
    "setuptools<81",
    "torch>=2.4,<3",
    "numba==0.60.*",
    "llvmlite==0.43.*",
    "kokoro==0.9.4",
    "resemblyzer==0.1.4",
    "soundfile>=0.12",
    # kokoro's G2P needs this spaCy model; it is not on PyPI
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
]


def die(msg):
    raise SystemExit(f"\nSETUP FAILED: {msg}")


def check_python():
    v = sys.version_info
    if not (3, 10) <= (v.major, v.minor) <= (3, 12):
        die(f"Python {v.major}.{v.minor} unsupported — use Python 3.10–3.12 "
            f"(3.12 recommended)")
    if sys.platform == "darwin" and platform.machine() == "x86_64":
        try:
            arm = subprocess.run(["sysctl", "-in", "hw.optional.arm64"],
                                 capture_output=True, text=True).stdout.strip()
        except Exception:
            arm = "0"
        if arm == "1":
            die("this is an x86_64 Python running under Rosetta on an "
                "Apple-Silicon Mac — it caps torch below what kokoro needs. "
                "Install a native arm64 Python and rerun.")
        die("Intel Macs are unsupported: torch wheels there stop at 2.2.2, "
            "and kokoro needs torch>=2.4. Use another laptop.")


def install():
    if subprocess.run([sys.executable, "-m", "pip", "--version"],
                      capture_output=True).returncode != 0:
        print(">> no pip in this environment — bootstrapping via ensurepip...")
        if subprocess.run([sys.executable, "-m", "ensurepip",
                           "--upgrade"]).returncode != 0:
            die("could not bootstrap pip — use a standard `python -m venv` "
                "environment")
    base = [sys.executable, "-m", "pip", "install"]
    if sys.platform.startswith("linux"):
        # CPU torch first, or pip pulls ~2.5 GB of CUDA wheels
        print(">> installing CPU-only torch (Linux)...")
        subprocess.run(base + ["torch>=2.4,<3", "--index-url",
                               "https://download.pytorch.org/whl/cpu"],
                       check=True)
    print(">> installing packages (this is the ~1 GB download)...")
    subprocess.run(base + PKGS, check=True)


def verify():
    print(">> verifying...")
    try:
        import torch, soundfile, numba                       # noqa: F401
        from resemblyzer import VoiceEncoder                 # noqa: F401
        import kokoro                                        # noqa: F401
        import en_core_web_sm                                # noqa: F401
    except Exception as e:
        die(f"import failed: {e}\nRerun `python setup_env.py` (without "
            f"--check) to install.")
    # espeak-ng has a ~160-char internal path buffer; a python env in a
    # deep folder (cloud-synced Documents etc.) crashes it with a cryptic
    # 'phontab: No such file' error. Catch that here instead.
    import espeakng_loader
    if len(espeakng_loader.get_data_path()) > 140:
        die("your Python environment sits in a folder path too deep for "
            "espeak-ng (>140 chars). Move the kit (and recreate your venv) "
            "somewhere shorter, e.g. ~/ttskit, and rerun.")
    print(f"   torch {torch.__version__}, kokoro OK, resemblyzer OK, "
          f"spaCy model OK")
    if not (shutil.which("espeak-ng") or shutil.which("espeak")):
        die("espeak-ng not found on PATH — install it (see header) and rerun "
            "with --check")
    print("   espeak-ng OK")
    assets = None
    for d in ("./kokoro_assets", "../kokoro_assets"):
        if os.path.isdir(d):
            assets = d
            break
    if assets:
        print(">> one-sentence synthesis smoke test...")
        os.environ.setdefault("KOKORO_DIR", assets)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import synth
        wav = synth.synthesize("Setup check complete.", os.path.join(
            assets, "voices", "af_heart.pt"))
        assert len(wav) > 0
        print(f"   synthesized {len(wav)/synth.SR:.1f}s of audio")
    else:
        print("   (kokoro_assets/ not found next to this script — synthesis "
              "smoke test skipped; run --check again from inside the kit)")
    print("\nREADY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only")
    args = ap.parse_args()
    check_python()
    if not args.check:
        install()
    verify()


if __name__ == "__main__":
    main()
