@echo off
setlocal
cd /d "%~dp0"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
  .venv\Scripts\python -m pip install -U pip
  .venv\Scripts\pip install -e .
)
.venv\Scripts\lforge %*
