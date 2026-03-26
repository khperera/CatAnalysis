@echo off
REM One-command build script for CatDetector exe (Windows)
REM Usage: build_exe.bat

echo === Installing dependencies ===
pip install -r requirements.txt
pip install pyinstaller
if errorlevel 1 goto error

echo === Building exe ===
pyinstaller CatApp.spec --clean
if errorlevel 1 goto error

echo.
echo === Done ===
echo Executable is in:  dist\CatDetector\
echo Run it with:       dist\CatDetector\CatDetector.exe
goto end

:error
echo.
echo BUILD FAILED. Check the output above for errors.
exit /b 1

:end
