@echo off
setlocal

REM Build the app first
call make_app.bat

REM Check that the exe exists before proceeding
if not exist "dist\PixelsToPlayers\PixelsToPlayers.exe" (
  echo Error: Built executable not found. Build may have failed.
  exit /b 1
)

REM Create the installer using the freshly built exe
py make_msi.py --exe dist\PixelsToPlayers\PixelsToPlayers.exe --name "MyApp" --version 1.2.3 --protocol PixelsToPlayers --out PixelsToPlayers-Setup.msi

endlocal