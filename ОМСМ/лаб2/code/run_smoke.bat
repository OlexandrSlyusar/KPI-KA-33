@echo off
rem Quick end-to-end check of the whole pipeline on a tiny VisDrone subset (2 epochs)
chcp 65001 >nul
cd /d "%~dp0"
title Lab2 - smoke test
set LAB2_SMOKE=1
set PYTHONUTF8=1
"%USERPROFILE%\omsm_lab2\venv\Scripts\python.exe" lab2_main.py
echo.
echo Smoke test finished with code %errorlevel%.
pause
