@echo off
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
set LOG=results\final_test_log.txt

echo ==========================================================
echo   FINAL TEST RUN - read this before continuing
echo ==========================================================
echo.
echo   The test split is scored ONCE, at the very end.
echo   Running it repeatedly and picking the best number is
echo   exactly the protocol violation the course warns about.
echo.
echo   Only continue if you are finished tuning.
echo.
pause
echo.

if not exist "%PY%" (
  echo [X] Run RUN_ME.bat first.
  pause
  exit /b 1
)

echo Scoring TURKISH test split...
echo ===== TURKISH TEST ===== > "%LOG%"
%PY% src\run_all.py --data data\massive\tr-TR --methods baseline,hmm,crf --out results\tr-TR --final-test > results\_t.txt 2>&1
type results\_t.txt
type results\_t.txt >> "%LOG%"
echo.

echo Scoring ENGLISH test split...
echo. >> "%LOG%"
echo ===== ENGLISH TEST ===== >> "%LOG%"
%PY% src\run_all.py --data data\massive\en-US --methods baseline,hmm,crf --out results\en-US --final-test > results\_t.txt 2>&1
type results\_t.txt
type results\_t.txt >> "%LOG%"
del results\_t.txt >nul 2>nul
echo.

echo ==========================================================
echo   Saved to results\final_test_log.txt
echo   Send that file to Claude.
echo ==========================================================
pause
