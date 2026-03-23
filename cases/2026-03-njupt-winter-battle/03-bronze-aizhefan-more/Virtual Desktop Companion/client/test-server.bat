@echo off

rem 进入脚本所在目录
cd /d %~dp0

echo ==============================================
echo       桌面宠物本地服务器测试脚本

echo ==============================================

rem 检查WSL2是否安装
echo 1. 检查WSL2是否安装...
wsl --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未检测到WSL2，请先安装WSL2
    echo 安装指南：https://docs.microsoft.com/zh-cn/windows/wsl/install
    pause
    exit /b 1
)

echo 检测到WSL2，版本：
wsl --version

echo.

echo 2. 测试WSL2命令执行...
echo 执行命令：wsl ls -la
echo ==============================================
wsl ls -la
echo ==============================================

if %errorlevel% neq 0 (
    echo 错误：WSL2命令执行失败
    pause
    exit /b 1
)

echo WSL2命令执行成功！

echo.

echo 3. 检查本地服务器是否运行...
echo 执行命令：curl http://localhost:3000/health

try {
    curl http://localhost:3000/health
} catch {
    echo 警告：本地服务器未运行，请先启动服务器
    echo 请运行 start-server.bat 启动服务器
}

echo.
echo 测试完成！
pause
