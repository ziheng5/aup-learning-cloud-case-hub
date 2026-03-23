# AI 桌面宠物客户端

基于 Electron + Vue 的跨平台桌面客户端

## 功能特性

- 🎭 **Live2D 支持** - 可爱的 Live2D 虚拟形象
- 💬 **实时对话** - 与 AI 进行自然对话
- ⌨️ **键盘控制** - AI 可直接控制键盘执行操作
- 🪟 **透明窗口** - 无边框透明窗口，支持拖拽

## 安装依赖

```bash
npm install
```

## 运行

```bash
# 开发模式
npm start

# 直接运行
electron .
```

## 打包

```bash
# Windows
npm run build:win

# macOS
npm run build:mac

# Linux
npm run build:linux
```

## 配置

修改 `app.js` 中的服务器地址：

```javascript
const SERVER_URL = '你的服务器地址';
```

## 项目结构

```
client/
├── main.js          # Electron 主进程
├── preload.js       # 预加载脚本
├── index.html       # 主页面
├── app.js           # Vue 应用逻辑
├── styles.css       # 样式文件
├── keyboard.js      # 键盘控制模块
├── local-server.js  # 本地服务器（WSL2命令执行）
└── package.json     # 项目配置
```

## 功能说明

### 1. Live2D 模型
- 支持多种表情和动作
- 实时响应AI情感状态
- 可自定义模型路径

### 2. 聊天功能
- 支持与服务器端AI对话
- 实时显示AI响应
- 支持流式输出

### 3. WSL2 命令执行
- 本地Node.js服务器处理WSL2命令
- 支持命令超时设置
- 自动转换编码格式

### 4. 模式切换
- 聊天模式：正常对话
- WSL2模式：执行WSL2命令

## 测试页面

项目包含多个测试页面：
- `api_test.html` - 纯API测试页面
- `quick_test.html` - 快速测试页面
- `test_live2d.html` - Live2D测试页面
- `live2d_test.html` - 纯Live2D模型测试

## 依赖说明

主要依赖：
- `electron` - 桌面应用框架
- `vue` - 前端框架
- `axios` - HTTP客户端
- `express` - 本地服务器
- `@pixi/live2d` - Live2D渲染引擎

## 注意事项

1. 确保服务器端正常运行
2. 本地服务器端口默认为3000
3. WSL2命令执行需要Windows系统支持WSL2
4. Live2D模型文件需要放在指定目录

## 故障排除

### 模型加载失败
- 检查模型路径是否正确
- 确认模型文件完整

### 服务器连接失败
- 检查服务器地址配置
- 确认服务器端正常运行
- 检查网络连接

### WSL2命令执行失败
- 确认本地服务器已启动
- 检查WSL2是否正常安装
- 查看本地服务器日志

## 许可证

MIT License
