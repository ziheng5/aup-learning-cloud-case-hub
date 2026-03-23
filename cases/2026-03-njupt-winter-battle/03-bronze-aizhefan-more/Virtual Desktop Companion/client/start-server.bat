@echo off

rem 进入脚本所在目录
cd /d %~dp0

echo ==============================================
echo       桌面宠物本地服务器启动脚本

echo ==============================================

rem 检查Node.js是否安装
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未检测到Node.js，请先安装Node.js
    echo 下载地址：https://nodejs.org/
    pause
    exit /b 1
)

echo 检测到Node.js，版本：
node --version

echo.

echo 1. 初始化npm项目...
npm init -y

echo.

echo 2. 安装依赖...
npm install express

echo.

echo 3. 启动本地服务器...
echo 本地服务器将运行在 http://localhost:3000
node local-server.js

pause
