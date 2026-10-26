@echo off
rem Full lab run: data, training of all models, evaluation, bonuses, inference
chcp 65001 >nul
cd /d "%~dp0"
title Lab2 - full run
set PYTHONUTF8=1
"%USERPROFILE%\omsm_lab2\venv\Scripts\python.exe" lab2_main.py
echo.
echo Finished with code %errorlevel%.
pause
