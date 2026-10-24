"""
check_clean_install.py - proves the project works from a brand-new virtual environment.

    python check_clean_install.py

Creates a throw-away venv, installs requirements-dev.txt, runs the whole test suite, then deletes
the venv. Takes a few minutes the first time (downloads MediaPipe). Needs internet for pip only.
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd, **kw):
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=HERE, **kw).returncode


def main():
    if not ((3, 9) <= sys.version_info[:2] <= (3, 12)):
        print(f"WARNING: Python {sys.version_info.major}.{sys.version_info.minor} is outside the supported 3.9-3.12 "
              "(mediapipe 0.10.14 has no wheels for it). Install must be expected to fail.")
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="spymotion_venv_") as tmp:
        venv = Path(tmp) / "venv"
        if run([sys.executable, "-m", "venv", str(venv)]):
            return 1
        py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if run([py, "-m", "pip", "install", "-q", "--disable-pip-version-check", "-r", "requirements-dev.txt"]):
            print("\nFAILED: dependency installation (see above).")
            return 1
        if run([py, "-c", "import mediapipe as mp, cv2, streamlit, scipy, altair; mp.solutions.pose.Pose; print('imports ok')"]):
            return 1
        code = run([py, "-m", "pytest"])
    print(f"\n{'PASSED' if code == 0 else 'FAILED'} clean-install check in {time.time() - t0:.0f}s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
