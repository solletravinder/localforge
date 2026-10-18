@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo LocalForge environment not found. Run run.bat setup first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m pip install -q "pyinstaller>=6.10"
if errorlevel 1 goto :failed

".venv\Scripts\pyinstaller.exe" --noconfirm --clean --onefile --windowed --noupx ^
  --name LocalForge ^
  --icon assets\localforge.ico ^
  --add-data "assets;assets" ^
  desktop.py
if errorlevel 1 goto :failed

copy /Y "dist\LocalForge.exe" "LocalForge.exe" >nul
if errorlevel 1 goto :failed

echo.
echo Built LocalForge.exe
exit /b 0

:failed
echo.
echo Build failed.
pause
exit /b 1
