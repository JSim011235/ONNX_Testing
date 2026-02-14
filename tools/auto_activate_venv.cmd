@echo off
set "PROJECT_ROOT=C:\Users\Justi\Documents\TURTLE\OLSN\ONNX_Testing"
set "VENV_ACTIVATE=%PROJECT_ROOT%\venv312\Scripts\activate.bat"
echo %CD% | findstr /I /C:"%PROJECT_ROOT%" >nul
if %errorlevel%==0 if exist "%VENV_ACTIVATE%" call "%VENV_ACTIVATE%"
