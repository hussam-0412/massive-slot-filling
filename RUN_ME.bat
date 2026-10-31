@echo off
cd /d "%~dp0"
echo ==========================================================
echo   NLP Project 2 - Cross-lingual slot filling on MASSIVE
echo ==========================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [X] Python is not installed, or not on your PATH.
  echo     https://www.python.org/downloads/  - tick "Add python.exe to PATH"
  pause
  exit /b 1
)

echo [1/6] Python found:
python --version
echo.

echo [2/6] Setting up an isolated environment...
echo       This does NOT touch the Python you use for other projects.
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 (
    echo [X] Could not create the virtual environment.
    pause
    exit /b 1
  )
)
set PY=.venv\Scripts\python.exe
%PY% -m pip install --quiet --upgrade pip
%PY% -m pip install --quiet sklearn-crfsuite
if errorlevel 1 (
  echo [X] Install failed. Check your internet connection.
  pause
  exit /b 1
)
echo       done.
echo.

echo [3/6] Running the test suite...
%PY% src\test_pipeline.py
if errorlevel 1 (
  echo [X] Tests failed. Send Claude everything above.
  pause
  exit /b 1
)
echo.

echo [4/6] Downloading the real MASSIVE corpus...
echo       Official tarball, standard library only. A few minutes.
%PY% data\load_massive.py --locales tr-TR en-US --out data\massive
if errorlevel 1 (
  echo.
  echo [X] DOWNLOAD FAILED - everything below would be meaningless,
  echo     so this script is stopping here. Send Claude the error above.
  pause
  exit /b 1
)
echo.

echo [5/6] Dataset statistics ^(morphology evidence^)...
%PY% src\dataset_stats.py --data data\massive
if errorlevel 1 ( pause & exit /b 1 )
echo.

echo [6/6] Training baseline + HMM + CRF on Turkish...
%PY% src\run_all.py --data data\massive\tr-TR --methods baseline,hmm,crf
if errorlevel 1 ( pause & exit /b 1 )
echo.

echo ==========================================================
echo   DONE. Sanity checks before you trust any of this:
echo.
echo    - "slot types seen" should be 55, NOT single digits
echo    - "skipped utterances" should be near 0
echo    - train sentences should be ~11500, NOT 1200
echo    - if every score is exactly 1.0000, something is wrong
echo.
echo   Results are in the results\ folder.
echo   BERTurk needs a GPU, so that one runs on Colab later.
echo ==========================================================
pause
