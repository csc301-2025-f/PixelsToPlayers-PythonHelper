# README.md

### Dependencies

First, install poetry through pipx [here](href:https://python-poetry.org/docs/#installing-with-pipx). Make sure that poetry is in your path.

Create a poetry venv running Python 3.10, [PyCharm tutorial](https://www.jetbrains.com/help/pycharm/poetry.html#poetry-env) (sorry VS Code).

Run `poetry install` to install dependencies.

### Build
Run `make_app.bat` then `make_msi.bat`

`.msi` installer registers the url `PixelsToPlayers://` to `HKCU/Software/Classes/PixelsToPlayers`

Build output exe in `.\dist\` and msi installer in root dir.

### Dependency Management

Add dependencies using `poetry add <package>` when needed.