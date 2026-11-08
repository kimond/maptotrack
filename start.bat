@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv (
  echo Creating Python environment...
  %PY% -m venv .venv || (echo Python 3.10+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install -q --disable-pip-version-check -r requirements.txt || (echo Package install failed & pause & exit /b 1)
python -m app.server
pause
