import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "PixelsToPlayers"
COLLECT_MODULES = ("numpy", "mediapipe")
COLLECT_BIN_MODULES = ("mediapipe",)
COLLECT_DATA_MODULES = ("mediapipe",)

def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)

def _python_is_usable(candidate: Path) -> bool:
    if not (candidate.exists() and os.access(candidate, os.X_OK)):
        return False
    try:
        subprocess.run(
            [str(candidate), "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False

def resolve_python(project_root: Path) -> Path:
    """
    Prefer the Poetry-managed virtualenv interpreter so PyInstaller sees all
    project dependencies. Fallback to the current interpreter if needed.
    """
    if os.environ.get("POETRY_ACTIVE") == "1":
        return Path(sys.executable)

    venv_dir = project_root / ".venv"
    if not venv_dir.is_dir():
        return Path(sys.executable)

    if os.name == "nt":
        candidate_rel_paths = [
            Path("Scripts") / "python.exe",
            Path("Scripts") / "python",
        ]
    else:
        candidate_rel_paths = [
            Path("bin") / "python3",
            Path("bin") / "python",
        ]
    candidates = [venv_dir / rel for rel in candidate_rel_paths]

    for candidate in candidates:
        if _python_is_usable(candidate):
            return candidate

    return Path(sys.executable)

def ensure_pyinstaller(py_exe: Path) -> bool:
    """
    Verify PyInstaller is available in the selected environment.
    """
    try:
        subprocess.run(
            [str(py_exe), "-c", "import PyInstaller"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "PyInstaller is not available in the selected environment.\n"
            "Install it via Poetry (e.g. `poetry add --group dev pyinstaller` "
            "or `poetry run pip install pyinstaller`) and retry."
        )
        return False

def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    windows_root = project_root / "platforms" / "windows"
    dist_root = windows_root / "dist"
    build_dir = windows_root / "build"
    py = resolve_python(project_root)

    if not ensure_pyinstaller(py):
        return 1

    dist_dir = dist_root / APP_NAME
    icon_path = project_root / "resources" / "favicon.ico"
    entry_point = project_root / "src" / "pixels_to_players" / "app.py"

    if not entry_point.exists():
        print(f"Entry point not found: {entry_point}")
        return 2

    # Always clean output dirs
    for p in (build_dir, dist_dir):
        if p.exists():
            print(f"Removing {p} ...")
            shutil.rmtree(p, ignore_errors=True)

    # Compose PyInstaller args
    args = [
        str(py), "-m", "PyInstaller",
        "--onedir",
        "--name", APP_NAME,
        "--clean",
        "-y",  # remove output directory without confirmation if it still exists
    ]
    args += ["--distpath", str(dist_root), "--workpath", str(build_dir)]
    for module in COLLECT_MODULES:
        args += ["--hidden-import", module, "--collect-all", module]
    for module in COLLECT_BIN_MODULES:
        args += ["--collect-binaries", module]
    for module in COLLECT_DATA_MODULES:
        args += ["--collect-data", module]
    if icon_path.exists():
        args += ["--icon", str(icon_path)]
    args.append(str(entry_point))

    # Run build
    try:
        run(args)
    except subprocess.CalledProcessError as e:
        print(f"Build failed with exit code {e.returncode}")
        return e.returncode

    out = dist_dir
    exe = out / f"{APP_NAME}.exe"
    print(f"\nBuild complete. Output: {out}")
    if not exe.exists():
        print(f"Expected executable not found at: {exe}")
        return 3
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
