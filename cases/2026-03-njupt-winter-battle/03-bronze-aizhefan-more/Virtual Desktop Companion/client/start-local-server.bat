@echo off

rem 设置错误处理模式
setlocal enabledelayedexpansion

rem 进入脚本所在目录
cd /d %~dp0

echo ===============================================================================
echo                             桌面宠物本地服务器启动脚本

echo ===============================================================================

rem 1. 检查Node.js是否安装
echo 1. 检查Node.js环境...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未检测到Node.js，请先安装Node.js
    echo 下载地址：https://nodejs.org/
    echo 安装完成后重新运行此脚本
    pause
    exit /b 1
)

echo 检测到Node.js，版本：
for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
echo !NODE_VERSION!

echo.

rem 2. 检查package.json是否存在
echo 2. 检查项目配置...
if not exist "package.json" (
    echo 未找到package.json，正在初始化npm项目...
    npm init -y >nul 2>&1
    if %errorlevel% neq 0 (
        echo 错误：初始化npm项目失败
        pause
        exit /b 1
    )
    echo npm项目初始化成功
) else (
    echo 检测到package.json，跳过初始化
)

echo.

rem 3. 检查express依赖是否安装
echo 3. 检查依赖...
if not exist "node_modules\express" (
    echo 未找到express依赖，正在安装...
    npm install express >nul 2>&1
    if %errorlevel% neq 0 (
        echo 错误：安装express依赖失败
        echo 请检查网络连接后重试
        pause
        exit /b 1
    )
    echo express依赖安装成功
) else (
    echo 检测到express依赖，跳过安装
)

echo.

rem 4. 检查local-server.js是否存在
echo 4. 检查服务器文件...
if not exist "local-server.js" (
    echo 错误：未找到local-server.js文件
    echo 请确保此脚本与local-server.js在同一目录
    pause
    exit /b 1
) else (
    echo 检测到local-server.js文件
)

echo.

rem 5. 启动本地服务器
echo 5. 启动本地服务器...
echo ===============================================================================
echo 本地服务器将运行在 http://localhost:3000

echo 健康检查：http://localhost:3000/health

echo 命令执行：http://localhost:3000/execute

echo ===============================================================================
echo 正在启动服务器...
echo 服务器启动后，请勿关闭此窗口

echo 按 Ctrl+C 停止服务器

echo ===============================================================================
echo.

node local-server.js

rem 服务器停止后显示信息
echo.
echo ===============================================================================
echo 服务器已停止

echo 如需重新启动，请再次运行此脚本

echo ===============================================================================

pause
