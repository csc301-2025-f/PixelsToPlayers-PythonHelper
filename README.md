# README.md

### Dependencies

First, install poetry through pipx [here](href:https://python-poetry.org/docs/#installing-with-pipx). Make sure that poetry is in your path.

Create a poetry venv running Python 3.10, [PyCharm tutorial](https://www.jetbrains.com/help/pycharm/poetry.html#poetry-env).

Run `poetry install` to install dependencies.

### Build / Installer

1. **Create the PyInstaller bundle**

   ```bash
   poetry run python platforms/windows/build_exe.py
   ```

   This wipes `platforms/windows/build/` and `platforms/windows/dist/PixelsToPlayers/`, then regenerates the one-dir build with the icon + hidden imports configured in `platforms/windows/build_exe.py`. The resulting `PixelsToPlayers.exe` lives in `platforms/windows/dist/PixelsToPlayers/`.

2. **Package the MSI with WiX 6**

   From PowerShell (recommended on Windows), run:

   ```powershell
   .\platforms\windows\build_installer.ps1
   ```

   The script ensures the WiX 6 CLI and UI extension are installed (`dotnet tool install --global wix` if needed), checks for `platforms/windows/dist/PixelsToPlayers/PixelsToPlayers.exe`, and invokes `wix build` with the v4 authoring in `platforms/windows/wix/PixelsToPlayers.wxs`. Pass `-OutputPath <path\to\PixelsToPlayers.msi>` (either absolute or relative to the repo root) to override the default `platforms/windows/dist/PixelsToPlayers.msi`.

The MSI installs per-user by default but lets the user change the directory (default `%LOCALAPPDATA%\PixelsToPlayers`) and registers the custom protocol `PixelsToPlayers://` under `HKCU\Software\Classes\PixelsToPlayers`.

### Dependency Management

Add dependencies using `poetry add <package>` when needed.
