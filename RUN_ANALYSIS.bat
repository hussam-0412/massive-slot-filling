@echo off
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
set LOG=results\analysis_log.txt
if exist "%LOG%" del "%LOG%"

echo ==========================================================
echo   Part 2 - full analysis. Takes roughly 15 minutes.
echo ==========================================================
echo.

if not exist "%PY%" (
  echo [X] Run RUN_ME.bat first.
  pause
  exit /b 1
)

echo [1/5] Filing the Turkish results...
if not exist "results\tr-TR" mkdir "results\tr-TR"
move /Y results\pred_*.json results\tr-TR\ >nul 2>nul
move /Y results\metrics.json results\tr-TR\ >nul 2>nul
echo       done.
echo.

echo [2/5] Training all three methods on ENGLISH ^(~4 min^)...
echo ===== ENGLISH ===== >> "%LOG%"
%PY% src\run_all.py --data data\massive\en-US --methods baseline,hmm,crf --out results\en-US > results\_tmp.txt 2>&1
type results\_tmp.txt
type results\_tmp.txt >> "%LOG%"
if errorlevel 1 ( echo [X] Failed & pause & exit /b 1 )
echo.

echo [3/5] Significance tests ^(Turkish: CRF vs HMM, CRF vs baseline^)...
echo. >> "%LOG%"
echo ===== SIGNIFICANCE: CRF vs HMM ===== >> "%LOG%"
%PY% src\significance.py --pred results\tr-TR\pred_crf_dev.json --compare results\tr-TR\pred_hmm_dev.json > results\_tmp.txt 2>&1
type results\_tmp.txt
type results\_tmp.txt >> "%LOG%"
echo. >> "%LOG%"
echo ===== SIGNIFICANCE: CRF vs BASELINE ===== >> "%LOG%"
%PY% src\significance.py --pred results\tr-TR\pred_crf_dev.json --compare results\tr-TR\pred_baseline_dev.json > results\_tmp.txt 2>&1
type results\_tmp.txt
type results\_tmp.txt >> "%LOG%"
echo.

echo [4/5] Stratified analysis - THE MORPHOLOGY EVIDENCE...
echo. >> "%LOG%"
echo ===== STRATIFIED: TURKISH ===== >> "%LOG%"
%PY% src\stratified.py --pred results\tr-TR\pred_crf_dev.json --train data\massive\tr-TR\train.conll --out results\tr-TR\stratified.json > results\_tmp.txt 2>&1
type results\_tmp.txt
type results\_tmp.txt >> "%LOG%"
echo. >> "%LOG%"
echo ===== STRATIFIED: ENGLISH ===== >> "%LOG%"
%PY% src\stratified.py --pred results\en-US\pred_crf_dev.json --train data\massive\en-US\train.conll --out results\en-US\stratified.json > results\_tmp.txt 2>&1
type results\_tmp.txt
type results\_tmp.txt >> "%LOG%"
echo.

echo [5/5] Cross-lingual transfer matrix ^(~7 min, trains CRF twice^)...
echo. >> "%LOG%"
echo ===== CROSS-LINGUAL MATRIX ===== >> "%LOG%"
%PY% src\crosslingual.py --data data\massive --method crf --out results\crosslingual.json > results\_tmp.txt 2>&1
type results\_tmp.txt
type results\_tmp.txt >> "%LOG%"
del results\_tmp.txt >nul 2>nul
echo.

echo ==========================================================
echo   DONE. Everything saved to:
echo       results\analysis_log.txt
echo   Send that one file to Claude.
echo.
echo   The number that matters most: the OOV penalty in
echo   step 4, Turkish vs English. That is your thesis.
echo ==========================================================
pause
