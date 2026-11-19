@echo off
title TF Prime Texture-Model-Animation Viewer
cd /d "%~dp0"
where python >nul 2>nul
if %errorlevel%==0 (
    python "scripts\tfp_tool.py" %*
) else (
    py -3 "scripts\tfp_tool.py" %*
)
if errorlevel 1 (
    echo.
    echo The tool closed with an error. If Python or a package is missing, run:
    echo     pip install -r requirements.txt
    pause
)
