@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Lab2 - update evaluation
set "PY=%USERPROFILE%\omsm_lab2\venv\Scripts\python.exe"
"%PY%" -m pip install "faster-coco-eval>=1.6" -c constraints.txt
if errorlevel 1 (echo [ERROR] install failed & pause & exit /b 1)
"%PY%" -c "import faster_coco_eval as f; print('faster-coco-eval', f.__version__ if hasattr(f,'__version__') else 'OK')"
if exist "outputs\eval_cache" (
  rmdir /s /q "outputs\eval_cache"
  echo Old evaluation cache removed.
)
echo.
echo UPDATE DONE. You can close this window.
pause
