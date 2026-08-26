@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   商机推送助手 - 本机隐私模式
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [首次运行] 正在创建本地 Python 环境...
    set "PY_LAUNCHER="
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.12 -c "import sys" >nul 2>nul
        if not errorlevel 1 set "PY_LAUNCHER=py -3.12"
        if not defined PY_LAUNCHER set "PY_LAUNCHER=py -3"
    )
    if not defined PY_LAUNCHER (
        where python >nul 2>nul
        if not errorlevel 1 set "PY_LAUNCHER=python"
    )
    if not defined PY_LAUNCHER goto :error

    !PY_LAUNCHER! -m venv .venv
    if errorlevel 1 goto :error

    echo [首次运行] 正在安装依赖，请保持网络连接...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto :error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :error
)

echo 正在启动，浏览器将自动打开 http://localhost:8502
echo 本窗口运行期间请勿关闭。
echo.
".venv\Scripts\python.exe" -m streamlit run opportunity_app.py --server.address localhost --server.port 8502 --browser.gatherUsageStats false
goto :end

:error
echo.
echo 启动失败。请确认已安装 Python 3.12，并将错误截图发给开发人员。
pause

:end
endlocal
