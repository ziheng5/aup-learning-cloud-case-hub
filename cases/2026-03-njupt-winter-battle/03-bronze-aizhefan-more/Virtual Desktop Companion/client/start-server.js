const fs = require('fs');
const { execSync } = require('child_process');

console.log('=============================================');
console.log('        桌面宠物本地服务器启动脚本');
console.log('=============================================');

// 1. 检查package.json
if (!fs.existsSync('package.json')) {
    console.log('1. 创建package.json...');
    execSync('npm init -y', { stdio: 'inherit' });
} else {
    console.log('1. package.json已存在，跳过创建');
}

// 2. 检查express依赖
if (!fs.existsSync('node_modules/express')) {
    console.log('2. 安装express依赖...');
    execSync('npm install express', { stdio: 'inherit' });
} else {
    console.log('2. express依赖已安装，跳过安装');
}

// 3. 启动本地服务器
console.log('3. 启动本地服务器...');
console.log('本地服务器将运行在 http://localhost:3000');
console.log('按 Ctrl+C 停止服务器');
console.log('=============================================');

// 4. 运行本地服务器
const { spawn } = require('child_process');
const server = spawn('node', ['local-server.js'], { stdio: 'inherit' });

server.on('close', (code) => {
    console.log('=============================================');
    console.log(`本地服务器已停止，退出代码: ${code}`);
    console.log('如需重新启动，请再次运行此脚本');
    console.log('=============================================');
});
