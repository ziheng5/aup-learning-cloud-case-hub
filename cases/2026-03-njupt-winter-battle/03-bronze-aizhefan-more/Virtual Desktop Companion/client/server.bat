@echo off

cd /d %~dp0

if not exist "package.json" (
    npm init -y
)

if not exist "node_modules\express" (
    npm install express
)

node local-server.js

pause
