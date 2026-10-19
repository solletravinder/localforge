@echo off
setlocal
cd /d "%~dp0"

call build-exe.bat
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" build-portable.py
if errorlevel 1 goto :failed

set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
  echo Inno Setup 6 was not found.
  echo Install it with: winget install JRSoftware.InnoSetup
  pause
  exit /b 1
)

for /f "usebackq delims=" %%v in (`".venv\Scripts\python.exe" -c "import pathlib,tomllib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])"`) do set "VERSION=%%v"
if not defined VERSION set "VERSION=0.4.0"

"%ISCC%" /DMyAppVersion=%VERSION% installer.iss
if errorlevel 1 goto :failed

echo.
echo Installer written to dist\LocalForge-Setup-0.4.0.exe
exit /b 0

:failed
echo.
echo Portable installer build failed.
pause
exit /b 1
