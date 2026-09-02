@echo off
setlocal
cd /d "%~dp0"

set "TARGET=%~1"

if "%TARGET%"=="" (
    echo I-JEPA Leaf Disease Predictor
    echo ================================
    echo Drag and drop an image or a folder of images onto this file,
    echo or paste a path below.
    echo.
    set /p TARGET="Image or folder path: "
)

if "%TARGET%"=="" (
    echo No path given. Exiting.
    pause
    exit /b 1
)

python "%~dp0trained_model\predict.py" "%TARGET%"

echo.
echo Done. Results were saved next to the original image(s).
pause
