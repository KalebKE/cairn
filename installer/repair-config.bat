@echo off
REM Cairn - Repair Claude Desktop Configuration
REM Run this if Claude Desktop doesn't show Cairn tools after installation.
REM
REM What it does:
REM   1. Adds Cairn to Claude Desktop's config file
REM   2. Shows whether it succeeded or failed
REM
REM Safe to run multiple times - it won't duplicate entries.

echo.
echo  Cairn - Repair Configuration
echo  ====================================
echo.

set "INSTALL_DIR=%LOCALAPPDATA%\Cairn"
set "PYTHON=%INSTALL_DIR%\python\python.exe"
set "SCRIPT=%INSTALL_DIR%\configure_claude.py"

REM Check that Cairn is installed
if not exist "%PYTHON%" (
    echo  ERROR: Cairn Python not found at:
    echo    %PYTHON%
    echo.
    echo  Please reinstall Cairn from https://tracqi.com
    echo.
    pause
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo  ERROR: Configuration script not found at:
    echo    %SCRIPT%
    echo.
    echo  Please reinstall Cairn from https://tracqi.com
    echo.
    pause
    exit /b 1
)

echo  Found Cairn at: %INSTALL_DIR%
echo  Configuring Claude Desktop...
echo.

"%PYTHON%" "%SCRIPT%" --install-dir "%INSTALL_DIR%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo  SUCCESS! Cairn has been configured for Claude Desktop.
    echo.
    echo  Next steps:
    echo    1. Quit Claude Desktop completely (right-click tray icon, Quit)
    echo    2. Reopen Claude Desktop
    echo    3. Start a new chat - you should see the tools icon
    echo.
) else (
    echo.
    echo  FAILED to configure Claude Desktop.
    echo.
    echo  Check the log file for details:
    echo    %INSTALL_DIR%\configure_claude.log
    echo.
    echo  For help, visit https://tracqi.com/docs
    echo.
)

pause
