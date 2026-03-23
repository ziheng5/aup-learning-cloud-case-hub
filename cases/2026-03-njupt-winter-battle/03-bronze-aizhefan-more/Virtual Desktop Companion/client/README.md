# AI 桌面宠物客户端

基于 Electron + Vue 的跨平台桌面客户端

## 功能特性

- ? **Live2D 支持** - 可加载 Live2D 虚拟形象
- ? **实时聊天** - 与 AI 助手自然对话
- ?? **键盘控制** - AI 可以直接控制键盘执行操作
- ? **透明窗口** - 无边框透明窗口，可拖拽

## 安装依赖

```bash
npm install
```

## 运行

```bash
# 开发模式
npm start

# 或者直接运行
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
└── package.json     # 项目配置
```
