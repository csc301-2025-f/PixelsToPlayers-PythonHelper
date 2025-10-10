py -m pip install --upgrade pip
py -m pip install pyinstaller

py -m PyInstaller ^
  --onedir ^
  --name PixelsToPlayers ^
  --icon resources\favicon.ico ^
  src\PixelsToPlayers\app.py