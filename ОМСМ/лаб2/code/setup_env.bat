@echo off
rem Lab 2 (YOLO, VisDrone): Python venv in ASCII path (MIOpen cannot read Cyrillic paths) + PyTorch for AMD ROCm (Radeon RX 9060 XT = gfx1200)
chcp 65001 >nul
cd /d "%~dp0"
title Lab2 - environment setup
echo ==== Lab 2: environment setup ====
set "PYEXE="
set "P312=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"

set "VENV=%USERPROFILE%\omsm_lab2\venv"
if exist "%VENV%\Scripts\python.exe" goto have_venv

where py >nul 2>nul
if errorlevel 1 goto no_launcher
echo Installed Python versions:
py -0p
for %%v in (3.12 3.13 3.11) do (
  if not defined PYEXE (
    py -%%v -c "import sys" >nul 2>nul && for /f "delims=" %%p in ('py -%%v -c "import sys;print(sys.executable)"') do set "PYEXE=%%p"
  )
)
:no_launcher
if not defined PYEXE if exist "%P312%" set "PYEXE=%P312%"
if defined PYEXE goto make_venv

echo.
echo ==== Python 3.11+ not found: installing Python 3.12.10 (python.org, per-user) ====
curl.exe -L --fail -o "%TEMP%\python-3.12.10-amd64.exe" https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
if errorlevel 1 (echo [ERROR] download failed & pause & exit /b 1)
"%TEMP%\python-3.12.10-amd64.exe" /passive InstallAllUsers=0 PrependPath=0 Include_launcher=1 InstallLauncherAllUsers=0 Include_test=0 Shortcuts=0
if not exist "%P312%" (echo [ERROR] Python 3.12 install failed & pause & exit /b 1)
set "PYEXE=%P312%"

:make_venv
echo Using %PYEXE%
"%PYEXE%" --version
"%PYEXE%" -m venv "%VENV%"
if errorlevel 1 (echo [ERROR] venv failed & pause & exit /b 1)

:have_venv
set "PY=%VENV%\Scripts\python.exe"
"%PY%" --version
"%PY%" -m pip install --upgrade pip
echo.
echo ==== PyTorch 2.14 + ROCm 10.1 for gfx1200 (AMD index) ====
"%PY%" -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1200]==2.14.0+rocm10.1.0" "torchvision[device-gfx1200]==0.29.0a0+rocm10.1.0"
if errorlevel 1 (echo [ERROR] PyTorch ROCm install failed & pause & exit /b 1)
"%PY%" -m pip freeze | findstr /b /i "torch== torchvision==" > constraints.txt
type constraints.txt
echo.
echo ==== Other packages (torch pinned by constraints.txt) ====
"%PY%" -m pip install -r requirements.txt -c constraints.txt
if errorlevel 1 (echo [ERROR] requirements install failed & pause & exit /b 1)
echo.
echo ==== GPU check ====
"%PY%" check_gpu.py
echo.
echo SETUP DONE. You can close this window.
pause
