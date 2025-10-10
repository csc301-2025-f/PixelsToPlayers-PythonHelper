# README.md

### Dependencies

Managed by poetry but still trying to figure it out. Requires Python 3.12 >= version >= 3.10
because msilib package is deprecated in 3.13+.  

### Build
Run `make_app.bat` then `make_msi.bat`

`.msi` installer registers the url `PixelsToPlayers://` to `HKCU/Software/Classes/PixelsToPlayers`

build output exe in `.\dist\` and msi installer  in root dir.

WIP