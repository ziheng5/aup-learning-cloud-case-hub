const express = require('express');
const { exec } = require('child_process');
const app = express();
const port = 3000;

app.use(express.json());

// 转换 UTF-16 编码的输出为 UTF-8
function convertUTF16ToUTF8(buffer) {
  let result = '';
  for (let i = 0; i < buffer.length; i += 2) {
    const charCode = buffer.readUInt16LE(i);
    if (charCode !== 0) {
      result += String.fromCharCode(charCode);
    }
  }
  return result;
}

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.post('/execute', (req, res) => {
  const { command, timeout } = req.body;
  console.log('接收到命令执行请求:', command);
  
  const timeoutId = setTimeout(() => {
    console.log('命令执行超时:', command);
    res.json({
      command: command,
      stdout: '',
      stderr: '命令执行超时',
      returncode: 1,
      execution_time: 0
    });
  }, timeout * 1000);

  // 确保命令在WSL中执行，避免执行会进入交互式会话的命令
  let wslCommand;
  if (command === 'wsl') {
    // 对于单纯的wsl命令，执行wsl -l来列出可用的发行版
    wslCommand = 'wsl -l';
  } else if (command.startsWith('wsl ')) {
    wslCommand = command;
  } else {
    wslCommand = `wsl ${command}`;
  }
  
  console.log('执行的WSL命令:', wslCommand);
  
  exec(wslCommand, { encoding: 'buffer' }, (error, stdout, stderr) => {
    clearTimeout(timeoutId);
    
    // 转换输出编码
    const convertedStdout = convertUTF16ToUTF8(stdout);
    const convertedStderr = convertUTF16ToUTF8(stderr);
    
    console.log('命令执行结果:', {
      error: error,
      stdout: convertedStdout,
      stderr: convertedStderr
    });
    
    if (error) {
      res.json({
        command: wslCommand,
        stdout: convertedStdout,
        stderr: convertedStderr || error.message,
        returncode: error.code || 1,
        execution_time: 0
      });
    } else {
      res.json({
        command: wslCommand,
        stdout: convertedStdout,
        stderr: convertedStderr,
        returncode: 0,
        execution_time: 0
      });
    }
  });
});

app.listen(port, () => {
  console.log(`本地服务器运行在 http://localhost:${port}`);
});
