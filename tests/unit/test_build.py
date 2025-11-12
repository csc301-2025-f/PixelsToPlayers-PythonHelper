"""Tests that validate PyInstaller builds bundled with Poetry dependencies."""

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

# Global variables

project_root = Path(__file__).resolve().parents[2]
"""Repository root derived from the test file location."""

exe = project_root / "dist" / "PixelsToPlayers" / "PixelsToPlayers.exe"
"""Path to the built executable under test."""

BUILD_SCRIPT = project_root / "build_exe.py"
"""Path to the PyInstaller driver that creates the PixelsToPlayers executable."""

def build_env() -> dict[str, str]:
    """Return a copy of the environment stripped of IDE-specific Python vars.
     Required for PyInstaller to run correctly."""
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    return env

def build_command() -> list[str]:
    """Determine the command needed to run build_exe.py in the active context."""
    # If we're already running inside Poetry or an activated virtualenv, just reuse that interpreter
    if os.environ.get("POETRY_ACTIVE") == "1":
        return [sys.executable, str(BUILD_SCRIPT)]

    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        current = Path(sys.executable).resolve()
        if Path(venv_path).resolve() in current.parents:
            return [sys.executable, str(BUILD_SCRIPT)]

    poetry = shutil.which("poetry")
    if poetry:
        return [poetry, "run", "python", str(BUILD_SCRIPT)]
    return [sys.executable, str(BUILD_SCRIPT)]


class TestExecutable(unittest.TestCase):
    """Integration tests that build the exe and exercise critical launch paths."""

    def test_build_script_and_executable_exists(self):
        """Build via the current interpreter/Poetry env and confirm the exe exists."""
        proc = subprocess.run(
            build_command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=build_env(),
            cwd=project_root,
        )
        assert proc.returncode == 0, f"build_exe.py failed with {proc.returncode}\nstdout={proc.stdout.decode()}\nstderr={proc.stderr.decode()}"
        assert exe.exists(), f"Executable not found at {exe}"

    def test_executable_starts_with_url(self):
        """Ensure the special 'test' path prints the diagnostic strings."""

        # Test URL
        url = "PixelsToPlayers://test?key1=value1&key2=value2&key2=value3"

        # Launch the executable with the URL; capture stdout/stderr via pipeline
        proc = subprocess.Popen(
            [str(exe), url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=build_env(),
            cwd=project_root,
        )
        try:
            outs, errs = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()
            raise AssertionError(f"Process timed out. stdout={outs!r} stderr={errs!r}")

        # Basic exit code check
        assert proc.returncode == 0, f"Non-zero exit: {proc.returncode}\nstdout={outs!r}\nstderr={errs!r}"

        # Assert stdout contains the diagnostic prints emitted in test path
        out = outs.decode(errors="replace")
        assert "handle:path=test" in out, f"Expected test path marker not found. stdout={out!r}"
        assert "handle:query=" in out, f"Expected query print not found. stdout={out!r}"
        assert "handle:keys=" in out, f"Expected keys print not found. stdout={out!r}"
        assert "handle:value_counts=" in out, f"Expected value counts print not found. stdout={out!r}"

    def test_executable_loads_modules_for_real_path(self):
        """Launch with a non-test path to verify heavy dependencies (numpy/cv2) load."""
        url = "PixelsToPlayers://run?key1=value1&key2=value2&key2=value3"
        proc = subprocess.Popen(
            [str(exe), url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=build_env(),
            cwd=project_root,
        )
        outs, errs = proc.communicate(timeout=10)
        out = outs.decode(errors="replace")
        err = errs.decode(errors="replace")
        print(out)
        print(err)
        assert "OpenCV bindings requires \"numpy\" package" not in out + err, (
            f"Executable failed to load numpy/cv2.\nstdout={out!r}\nstderr={err!r}"
        )

def suite() -> unittest.TestSuite:
    """Assemble tests in definition order so build runs before runtime checks."""
    test_names = [
        name for name in TestExecutable.__dict__
        if name.startswith("test_")
    ]
    test_suite = unittest.TestSuite()
    for name in test_names:
        test_suite.addTest(TestExecutable(name))
    return test_suite

def load_tests(loader, tests, pattern):
    """Hook for unittest discovery to enforce our custom suite ordering."""
    return suite()


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
