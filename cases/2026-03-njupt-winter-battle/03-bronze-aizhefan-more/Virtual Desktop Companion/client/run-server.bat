@echo off

cd /d %~dp0

npm init -y
npm install express
node local-server.js

pause
