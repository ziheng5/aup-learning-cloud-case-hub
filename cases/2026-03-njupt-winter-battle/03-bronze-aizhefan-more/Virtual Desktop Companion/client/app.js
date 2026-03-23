const { createApp } = Vue;

const SERVER_URL = 'https://carlos-poultry-kentucky-critics.trycloudflare.com';
const MODEL_PATH = './mao_pro_zh/mao_pro_zh/runtime/mao_pro.model3.json';
const LOCAL_SERVER_PORT = 3000;
const LOCAL_SERVER_URL = `http://localhost:${LOCAL_SERVER_PORT}`;

let live2dModel = null;
let pixiApp = null;
let localServerRunning = false;

console.log('Vue app initializing');

// 检查Vue是否存在
if (typeof Vue === 'undefined') {
    console.error('Vue库未加载');
} else {
    console.log('Vue库已加载，版本:', Vue.version);
}

// 尝试创建Vue实例
const app = createApp({
    data() {
        return {
            messages: [],
            inputMsg: '',
            sessionId: null,
            currentEmotion: 'neutral',
            showChat: false,
            isLoading: false,
            isDragging: false,
            dragStartX: 0,
            dragStartY: 0,
            modelLoaded: false,
            showCommandModal: false,
            currentCommand: '',
            commandResult: null,
            isExecutingCommand: false,
            commandTimeout: 300,
            localServerRunning: false,
            localServerError: null,
            currentMode: 'chat', // 模式：chat 或 wsl2
            showModeToggle: false,
            multiStepExecution: {
                isActive: false,
                steps: [],
                currentStep: 0,
                taskId: null,
                taskDescription: ''
            }
        }
    },

    computed: {
        emotionEmoji() {
            const emojis = {
                'happy': '?',
                'sad': '?',
                'angry': '?',
                'playful': '?',
                'philosophical': '?',
                'neutral': '?'
            };
            return emojis[this.currentEmotion] || '?';
        }
    },

    mounted() {
        this.loadSession();
        this.initBubble();
        this.initDrag();
        this.initLive2D();
    },

    methods: {
        async initLive2D() {
            try {
                const canvas = this.$refs.live2dCanvas;
                if (!canvas) {
                    console.error('Canvas not found');
                    return;
                }

                console.log('PIXI:', typeof PIXI);
                console.log('PIXI.live2d:', typeof PIXI?.live2d);
                console.log('Live2DModel:', typeof PIXI?.live2d?.Live2DModel);

                const canvasWidth = 400;
                const canvasHeight = 600;
                
                canvas.width = canvasWidth;
                canvas.height = canvasHeight;
                canvas.style.width = canvasWidth + 'px';
                canvas.style.height = canvasHeight + 'px';

                pixiApp = new PIXI.Application({
                    view: canvas,
                    autoStart: true,
                    transparent: true,
                    backgroundAlpha: 0,
                    clearBeforeRender: true,
                    antialias: true,
                    width: canvasWidth,
                    height: canvasHeight,
                    resolution: window.devicePixelRatio || 1,
                    autoDensity: true,
                    powerPreference: "high-performance"
                });

                const Live2DModel = PIXI.live2d.Live2DModel;
                await Live2DModel.registerTicker(PIXI.Ticker);
                
                console.log('??????????:', MODEL_PATH);
                live2dModel = await Live2DModel.from(MODEL_PATH);
                
                console.log('?????????, ????????');
                pixiApp.stage.addChild(live2dModel);

                console.log('????????:', live2dModel.width, 'x', live2dModel.height);
                
                const scaleX = canvasWidth / live2dModel.width;
                const scaleY = canvasHeight / live2dModel.height;
                const scale = Math.min(scaleX, scaleY);
                
                console.log('???????:', scale);
                
                const scaledWidth = live2dModel.width * scale;
                const scaledHeight = live2dModel.height * scale;
                
                live2dModel.scale.set(scale);
                live2dModel.x = (canvasWidth - scaledWidth) / 2;
                live2dModel.y = canvasHeight - scaledHeight;
                
                console.log('???????:', scaledWidth, 'x', scaledHeight);
                console.log('???��??:', live2dModel.x, live2dModel.y);

                this.modelLoaded = true;
                console.log('Live2D ????????!');

                this.startIdleMotion();
                
                // 检查本地服务器状态
                await this.checkLocalServer();

            } catch (error) {
                console.error('Live2D ???????:', error);
            }
        },

        async checkLocalServer() {
            console.log('Checking local server status');
            try {
                // 尝试访问本地服务器
                const response = await axios.get(`${LOCAL_SERVER_URL}/health`, { timeout: 2000 });
                if (response.status === 200) {
                    this.localServerRunning = true;
                    console.log('本地服务器已运行');
                }
            } catch (error) {
                this.localServerRunning = false;
                this.localServerError = error.message;
                console.log('本地服务器未运行，需要启动:', error.message);
                // 显示启动本地服务器的提示
                this.addMessage('ai', '本地服务器未运行，需要启动才能执行WSL2命令。请在本地运行以下命令启动服务器：\n\n```bash\n# 创建本地服务器文件\ncat > local-server.js << \'EOF\nconst express = require(\'express\');\nconst { exec } = require(\'child_process\');\nconst app = express();\nconst port = 3000;\n\napp.use(express.json());\n\napp.get(\'/health\', (req, res) => {\n  res.json({ status: \'ok\' });\n});\n\napp.post(\'/execute\', (req, res) => {\n  const { command, timeout } = req.body;\n  \n  const timeoutId = setTimeout(() => {\n    res.json({\n      command: command,\n      stdout: \'\',\n      stderr: \'命令执行超时\',\n      returncode: 1,\n      execution_time: 0\n    });\n  }, timeout * 1000);\n\n  exec(command, (error, stdout, stderr) => {\n    clearTimeout(timeoutId);\n    \n    if (error) {\n      res.json({\n        command: command,\n        stdout: stdout,\n        stderr: stderr || error.message,\n        returncode: error.code || 1,\n        execution_time: 0\n      });\n    } else {\n      res.json({\n        command: command,\n        stdout: stdout,\n        stderr: stderr,\n        returncode: 0,\n        execution_time: 0\n      });\n    }\n  });\n});\n\napp.listen(port, () => {\n  console.log(`本地服务器运行在 http://localhost:${port}`);\n});\nEOF\n\n# 安装依赖并启动服务器\nnpm init -y\nnpm install express\nnode local-server.js\n```');
            }
        },

        async startLocalServer() {
            // 由于浏览器安全限制，无法直接在浏览器中启动本地服务器
            // 这里只是提示用户手动启动
            this.addMessage('ai', '请在本地运行以下命令启动服务器：\n\n```bash\n# 创建本地服务器文件\ncat > local-server.js << \'EOF\nconst express = require(\'express\');\nconst { exec } = require(\'child_process\');\nconst app = express();\nconst port = 3000;\n\napp.use(express.json());\n\napp.get(\'/health\', (req, res) => {\n  res.json({ status: \'ok\' });\n});\n\napp.post(\'/execute\', (req, res) => {\n  const { command, timeout } = req.body;\n  \n  const timeoutId = setTimeout(() => {\n    res.json({\n      command: command,\n      stdout: \'\',\n      stderr: \'命令执行超时\',\n      returncode: 1,\n      execution_time: 0\n    });\n  }, timeout * 1000);\n\n  exec(`wsl ${command}`, (error, stdout, stderr) => {\n    clearTimeout(timeoutId);\n    \n    if (error) {\n      res.json({\n        command: command,\n        stdout: stdout,\n        stderr: stderr || error.message,\n        returncode: error.code || 1,\n        execution_time: 0\n      });\n    } else {\n      res.json({\n        command: command,\n        stdout: stdout,\n        stderr: stderr,\n        returncode: 0,\n        execution_time: 0\n      });\n    }\n  });\n});\n\napp.listen(port, () => {\n  console.log(`本地服务器运行在 http://localhost:${port}`);\n});\nEOF\n\n# 安装依赖并启动服务器\nnpm init -y\nnpm install express\nnode local-server.js\n```');
        },

        startIdleMotion() {
            if (!live2dModel) return;
            
            setInterval(() => {
                if (live2dModel && !this.showChat) {
                    const motions = ['mtn_01', 'mtn_02', 'mtn_03', 'mtn_04'];
                    const randomMotion = motions[Math.floor(Math.random() * motions.length)];
                    live2dModel.motion(randomMotion).catch(() => {});
                }
            }, 5000);
        },

        setExpression(expressionIndex) {
            if (!live2dModel) return;
            const expressions = ['exp_01', 'exp_02', 'exp_03', 'exp_04', 
                               'exp_05', 'exp_06', 'exp_07', 'exp_08'];
            const expName = expressions[expressionIndex] || expressions[0];
            live2dModel.expression(expName).catch(() => {});
        },

        playMotion(motionName) {
            if (!live2dModel) return;
            live2dModel.motion(motionName).catch(() => {});
        },

        updateLive2DByEmotion(emotion, live2dParams = null) {
            if (!live2dModel) return;
            
            const emotionMap = {
                'happy': { expression: 'exp_01', motion: 'special_01' },
                'sad': { expression: 'exp_02', motion: 'mtn_02' },
                'angry': { expression: 'exp_03', motion: 'mtn_03' },
                'playful': { expression: 'exp_04', motion: 'special_02' },
                'philosophical': { expression: 'exp_05', motion: 'mtn_04' },
                'neutral': { expression: 'exp_01', motion: 'mtn_01' },
                'thinking': { expression: 'exp_05', motion: 'mtn_04' },
                'excited': { expression: 'exp_01', motion: 'special_01' },
                'tired': { expression: 'exp_02', motion: 'mtn_02' },
                'anxious': { expression: 'exp_03', motion: 'mtn_03' }
            };

            const config = emotionMap[emotion] || emotionMap['neutral'];
            
            live2dModel.expression(config.expression).catch(() => {});
            live2dModel.motion(config.motion).catch(() => {});

            if (live2dParams && live2dParams.params) {
                this.applyLive2DParams(live2dParams.params);
            }
        },

        applyLive2DParams(params) {
            if (!live2dModel || !live2dModel.internalModel) return;

            try {
                const coreModel = live2dModel.internalModel.coreModel;
                
                if (params.eye_open !== undefined) {
                    coreModel.setParameterValueById('ParamEyeLOpen', params.eye_open);
                    coreModel.setParameterValueById('ParamEyeROpen', params.eye_open);
                }
                if (params.eye_x !== undefined) {
                    coreModel.setParameterValueById('ParamEyeBallX', params.eye_x);
                }
                if (params.eye_y !== undefined) {
                    coreModel.setParameterValueById('ParamEyeBallY', params.eye_y);
                }
                if (params.mouth_open !== undefined) {
                    coreModel.setParameterValueById('ParamMouthOpenY', params.mouth_open);
                }
                if (params.mouth_form !== undefined) {
                    coreModel.setParameterValueById('ParamMouthForm', params.mouth_form);
                }
                if (params.body_angle_x !== undefined) {
                    coreModel.setParameterValueById('ParamBodyAngleX', params.body_angle_x);
                }
                if (params.body_angle_y !== undefined) {
                    coreModel.setParameterValueById('ParamBodyAngleY', params.body_angle_y);
                }
                if (params.body_angle_z !== undefined) {
                    coreModel.setParameterValueById('ParamBodyAngleZ', params.body_angle_z);
                }
                if (params.breath !== undefined) {
                    coreModel.setParameterValueById('ParamBreath', params.breath);
                }
            } catch (e) {
                console.log('???Live2D???????:', e);
            }
        },

        async loadSession() {
            try {
                const response = await axios.post(`${SERVER_URL}/api/v1/session`);
                this.sessionId = response.data.session_id;
            } catch (error) {
                console.error('Session error:', error);
            }
        },

        initBubble() {
        },

        handleAvatarClick() {
            this.showChat = !this.showChat;
        },

        closeChat() {
            this.showChat = false;
        },

        async sendMsg() {
            if (!this.inputMsg.trim() || this.isLoading) return;

            const userMsg = this.inputMsg.trim();
            this.inputMsg = '';

            this.addMessage('user', userMsg);
            this.isLoading = true;

            try {
                if (this.currentMode === 'chat') {
                    // 聊天模式：发送到远程服务器
                    const response = await axios.post(`${SERVER_URL}/api/v1/chat`, {
                        session_id: this.sessionId,
                        message: userMsg,
                        stream: false,
                        mode: this.currentMode
                    }, { timeout: 60000 });

                    const data = response.data;
                    this.sessionId = data.session_id;
                    this.currentEmotion = data.emotion;

                    this.addMessage('ai', data.text);
                    this.bubbleText = data.text;

                    console.log('???????????Live2D????:', data.live2d_params);
                    console.log('???????:', data.emotion_vector);
                    
                    this.updateLive2DByEmotion(data.emotion, data.live2d_params);

                    // 处理WSL2命令意图
                    if (data.metadata && data.metadata.is_wsl2_intent) {
                        const detectedCommand = data.metadata.detected_command;
                        if (detectedCommand) {
                            // 如果检测到具体命令，自动弹出确认对话框
                            this.showCommandConfirmation(detectedCommand);
                        }
                    }
                    
                    // 处理WSL2模式下的命令
                    if (this.currentMode === 'wsl2' && data.wsl2_command) {
                        // 自动执行WSL2命令
                        this.currentCommand = data.wsl2_command.command;
                        await this.executeLocalWSLCommand();
                    }
                } else {
                    // WSL2命令模式：直接发送用户输入给服务器AI，由AI判断用户意图
                    try {
                        const response = await axios.post(`${SERVER_URL}/api/v1/chat`, {
                            session_id: this.sessionId,
                            message: userMsg,
                            stream: false,
                            mode: this.currentMode
                        }, { timeout: 60000 });

                        const data = response.data;
                        this.sessionId = data.session_id;
                        this.currentEmotion = data.emotion;

                        // 显示AI的回复
                        this.addMessage('ai', data.text);
                        this.bubbleText = data.text;

                        this.updateLive2DByEmotion(data.emotion, data.live2d_params || null);

                        // 处理WSL2模式下的命令
                        if (data.wsl2_command) {
                            // 显示命令解释
                            const commandDescription = data.wsl2_command.description || `执行命令: ${data.wsl2_command.command}`;
                            this.addMessage('ai', `我理解您想要：${commandDescription}`);
                            
                            // 自动执行WSL2命令
                            this.currentCommand = data.wsl2_command.command;
                            await this.executeLocalWSLCommand();
                        } else {
                            // 尝试从AI回复中提取命令
                            const command = this.extractCommandFromAIResponse(data.text);
                            if (command) {
                                // 如果从AI回复中提取到命令，执行命令
                                this.addMessage('ai', `我理解您想要执行命令: ${command}`);
                                this.currentCommand = command;
                                await this.executeLocalWSLCommand();
                            } else {
                                this.addMessage('ai', '抱歉，我没有理解您需要执行的具体命令，请尝试更明确地描述您的需求。');
                            }
                        }
                    } catch (error) {
                        console.error('AI server error:', error);
                        this.addMessage('ai', `连接AI服务器失败：${error.message}`);
                    }
                }

            } catch (error) {
                console.error('Send error:', error);
                this.addMessage('ai', `连接AI服务器失败：${error.message}`);
            } finally {
                this.isLoading = false;
            }
        },

        showCommandConfirmation(command) {
            this.currentCommand = command;
            this.showCommandModal = true;
        },

        async executeCommand() {
            if (!this.currentCommand.trim() || this.isExecutingCommand) return;

            this.isExecutingCommand = true;
            this.commandResult = null;

            try {
                const response = await axios.post(`${SERVER_URL}/api/v1/skill`, {
                    skill_name: 'wsl2',
                    params: {
                        action: 'execute_command',
                        command: this.currentCommand,
                        timeout: this.commandTimeout
                    }
                }, { timeout: this.commandTimeout * 1000 + 5000 });

                const data = response.data;
                if (data.success) {
                    this.commandResult = data.data;
                    this.addMessage('ai', this.formatCommandResult(data.data));
                } else {
                    this.addMessage('ai', `命令执行失败：${data.message}`);
                }

            } catch (error) {
                console.error('Command execution error:', error);
                this.addMessage('ai', `命令执行失败：${error.message}`);
            } finally {
                this.isExecutingCommand = false;
                this.showCommandModal = false;
            }
        },

        async executeLocalWSLCommand() {
            console.log('executeLocalWSLCommand called');
            if (!this.currentCommand.trim() || this.isExecutingCommand) {
                console.log('Command empty or already executing');
                return;
            }

            try {
                // 先关闭命令确认对话框
                console.log('Closing command modal');
                this.showCommandModal = false;
                
                // 等待对话框关闭
                await new Promise(resolve => setTimeout(resolve, 100));

                this.isExecutingCommand = true;
                this.commandResult = null;

                console.log('Executing command:', this.currentCommand);
                // 显示执行中的消息
                this.addMessage('ai', `正在执行命令：${this.currentCommand}...`);

                // 直接在本地执行 WSL2 命令
                const result = await this.runLocalCommand(this.currentCommand, this.commandTimeout);
                
                // 显示执行结果
                console.log('Command result:', result);
                this.addMessage('ai', this.formatCommandResult(result));

            } catch (error) {
                console.error('Command execution error:', error);
                this.addMessage('ai', `命令执行失败：${error.message}`);
            } finally {
                console.log('Command execution finished');
                this.isExecutingCommand = false;
                // 确保命令确认对话框关闭
                this.showCommandModal = false;
                console.log('Command modal closed');
            }
        },

        runLocalCommand(command, timeout) {
            return new Promise((resolve, reject) => {
                // 检查本地服务器是否运行
                if (!this.localServerRunning) {
                    // 本地服务器未运行，直接reject
                    reject(new Error('本地服务器未运行，请先启动本地服务器'));
                } else {
                    // 本地服务器已运行，执行命令
                    this.executeCommandOnLocalServer(command, timeout)
                        .then(resolve)
                        .catch(reject);
                }
            });
        },

        async executeCommandOnLocalServer(command, timeout) {
            try {
                const response = await axios.post(`${LOCAL_SERVER_URL}/execute`, {
                    command: command,
                    timeout: timeout
                });
                return response.data;
            } catch (error) {
                throw new Error(`执行命令失败：${error.message}`);
            }
        },

        formatCommandResult(result) {
            // 检查 result 是否存在
            if (!result) {
                return "命令执行失败：未返回结果";
            }
            
            let output = `命令执行结果：\n\n**命令：** ${result.command || '未知'}\n\n**退出码：** ${result.returncode || 0}\n\n**输出：**\n\`\`\``;
            if (result.stdout) {
                output += `\n${result.stdout}`;
            }
            if (result.stderr) {
                output += `\n${result.stderr}`;
            }
            output += `\n\`\`\``;
            
            // 检查 execution_time 是否存在
            if (result.execution_time !== undefined) {
                output += `\n\n**执行时间：** ${result.execution_time.toFixed(2)}秒`;
            } else {
                output += `\n\n**执行时间：** 未知`;
            }
            
            return output;
        },

        closeCommandModal() {
            this.showCommandModal = false;
            this.currentCommand = '';
            this.commandResult = null;
        },

        addMessage(role, content) {
            this.messages.push({
                id: Date.now(),
                role,
                content
            });

            this.$nextTick(() => {
                const container = this.$refs.chatMessages;
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            });
        },

        initDrag() {
            document.addEventListener('mouseup', this.stopDrag.bind(this));
        },

        startDrag(e) {
            if (this.isDragging) return;
            this.isDragging = true;
            this.dragStartX = e.screenX;
            this.dragStartY = e.screenY;
            document.addEventListener('mousemove', this.onDrag.bind(this));
        },

        onDrag(e) {
            if (!this.isDragging) return;

            const deltaX = e.screenX - this.dragStartX;
            const deltaY = e.screenY - this.dragStartY;

            if (window.electronAPI) {
                window.electronAPI.drag(deltaX, deltaY);
            }

            this.dragStartX = e.screenX;
            this.dragStartY = e.screenY;
        },

        stopDrag() {
            if (this.isDragging) {
                this.isDragging = false;
                document.removeEventListener('mousemove', this.onDrag.bind(this));
            }
        },

        toggleMode() {
            this.currentMode = this.currentMode === 'chat' ? 'wsl2' : 'chat';
            this.addMessage('ai', `已切换到${this.currentMode === 'chat' ? '聊天' : 'WSL2命令执行'}模式`);
        },

        toggleModeMenu() {
            this.showModeToggle = !this.showModeToggle;
        },

        // 开始多步骤执行
        startMultiStepExecution(taskDescription, aiResponse) {
            this.multiStepExecution = {
                isActive: true,
                steps: [],
                currentStep: 0,
                taskId: Date.now().toString(),
                taskDescription: taskDescription
            };

            // 从AI回复中提取步骤
            this.extractStepsFromResponse(aiResponse);
        },

        // 从AI回复中提取步骤
        extractStepsFromResponse(responseText) {
            // 简单的步骤提取逻辑
            const lines = responseText.split('\n');
            const steps = [];
            
            let currentStep = null;
            
            lines.forEach(line => {
                const stepMatch = line.match(/^\d+\.\s*步骤(\d+):\s*(.+)$/);
                const commandMatch = line.match(/^\s*命令：\s*(.+)$/);
                const codeMatch = line.match(/```(\w+)?\n([\s\S]*?)```/);
                
                if (stepMatch) {
                    if (currentStep) {
                        steps.push(currentStep);
                    }
                    currentStep = {
                        description: stepMatch[2].trim(),
                        command: ''
                    };
                } else if (commandMatch && currentStep) {
                    currentStep.command = commandMatch[1].trim();
                } else if (codeMatch && codeMatch[2]) {
                    // 直接从代码块中提取命令
                    const extractedCommand = codeMatch[2].trim();
                    steps.push({
                        description: this.multiStepExecution.taskDescription,
                        command: extractedCommand
                    });
                }
            });
            
            if (currentStep) {
                steps.push(currentStep);
            }

            // 如果没有提取到步骤，尝试从自然语言中提取命令
            if (steps.length === 0) {
                // 查找包含命令的文本
                const wslCommandMatch = responseText.match(/wsl\s+(.+)/i);
                if (wslCommandMatch && wslCommandMatch[1]) {
                    const extractedCommand = `wsl ${wslCommandMatch[1].trim()}`;
                    steps.push({
                        description: this.multiStepExecution.taskDescription,
                        command: extractedCommand
                    });
                } else {
                    // 尝试提取常见命令
                    const commonCommands = [
                        /ls\s*-la?/i,
                        /pwd/i,
                        /cd\s+(.+)/i,
                        /mkdir\s+(.+)/i,
                        /rm\s+(.+)/i,
                        /cat\s+(.+)/i,
                        /echo\s+(.+)\s*>\s*(.+)/i
                    ];
                    
                    for (const pattern of commonCommands) {
                        const match = responseText.match(pattern);
                        if (match) {
                            const extractedCommand = `wsl ${match[0].trim()}`;
                            steps.push({
                                description: this.multiStepExecution.taskDescription,
                                command: extractedCommand
                            });
                            break;
                        }
                    }
                }
            }

            this.multiStepExecution.steps = steps;
            
            if (steps.length > 0) {
                this.addMessage('ai', `开始执行任务：${this.multiStepExecution.taskDescription}`);
                this.executeNextStep();
            } else {
                // 如果还是没有提取到步骤，尝试使用默认命令
                const defaultCommands = {
                    '查看目录': 'wsl ls -la',
                    '查看文件': 'wsl ls -la',
                    '创建文件': 'wsl touch test.txt',
                    '运行程序': 'wsl ls -la'
                };
                
                let defaultCommand = null;
                for (const [key, value] of Object.entries(defaultCommands)) {
                    if (this.multiStepExecution.taskDescription.includes(key)) {
                        defaultCommand = value;
                        break;
                    }
                }
                
                if (defaultCommand) {
                    steps.push({
                        description: this.multiStepExecution.taskDescription,
                        command: defaultCommand
                    });
                    this.multiStepExecution.steps = steps;
                    this.addMessage('ai', `开始执行任务：${this.multiStepExecution.taskDescription}`);
                    this.executeNextStep();
                } else {
                    this.addMessage('ai', '抱歉，我没有理解您需要执行的具体步骤，请尝试更明确地描述您的需求。');
                    this.multiStepExecution.isActive = false;
                }
            }
        },

        // 执行下一步
        async executeNextStep() {
            if (!this.multiStepExecution.isActive) return;
            
            const currentStep = this.multiStepExecution.currentStep;
            const steps = this.multiStepExecution.steps;
            
            if (currentStep >= steps.length) {
                // 所有步骤执行完成
                this.addMessage('ai', `任务执行完成：${this.multiStepExecution.taskDescription}`);
                this.multiStepExecution.isActive = false;
                return;
            }

            const step = steps[currentStep];
            this.addMessage('ai', `执行步骤 ${currentStep + 1}/${steps.length}：${step.description}`);

            // 执行当前步骤的命令
            this.currentCommand = step.command;
            await this.executeLocalWSLCommand();

            // 执行完成后，进入下一步
            this.multiStepExecution.currentStep++;
            this.executeNextStep();
        },

        // 取消多步骤执行
        cancelMultiStepExecution() {
            this.multiStepExecution.isActive = false;
            this.addMessage('ai', `任务执行已取消：${this.multiStepExecution.taskDescription}`);
        },

        // 从用户输入中提取命令
        extractCommandFromUserInput(input) {
            // 检查是否包含具体命令
            const commandMatch = input.match(/执行\s+(.+?)\s*命令/i);
            if (commandMatch && commandMatch[1]) {
                return `wsl ${commandMatch[1].trim()}`;
            }
            
            // 检查是否直接输入了命令
            const directCommandMatch = input.match(/^\s*(ls|pwd|cd|mkdir|rm|cat|echo|cp|mv|chmod|chown|find|grep|sed|awk|sort|uniq|head|tail|diff|git|npm|yarn|pip|python|node|gcc|g\+\+|make|cmake|docker|docker-compose)\s+.*/i);
            if (directCommandMatch) {
                return `wsl ${input.trim()}`;
            }
            
            return null;
        },

        // 从AI回复中提取命令
        extractCommandFromAIResponse(response) {
            // 从代码块中提取命令
            const codeMatch = response.match(/```(\w+)?\n([\s\S]*?)```/);
            if (codeMatch && codeMatch[2]) {
                const extractedCommand = codeMatch[2].trim();
                return extractedCommand.startsWith('wsl ') ? extractedCommand : `wsl ${extractedCommand}`;
            }
            
            // 从文本中提取命令
            const commandMatch = response.match(/命令：\s*(.+)/i);
            if (commandMatch && commandMatch[1]) {
                const extractedCommand = commandMatch[1].trim();
                return extractedCommand.startsWith('wsl ') ? extractedCommand : `wsl ${extractedCommand}`;
            }
            
            // 从文本中提取直接的命令
            const directCommandMatch = response.match(/^\s*(ls|pwd|cd|mkdir|rm|cat|echo|cp|mv|chmod|chown|find|grep|sed|awk|sort|uniq|head|tail|diff|git|npm|yarn|pip|python|node|gcc|g\+\+|make|cmake|docker|docker-compose)\s+.*/i);
            if (directCommandMatch) {
                const extractedCommand = directCommandMatch[0].trim();
                return extractedCommand.startsWith('wsl ') ? extractedCommand : `wsl ${extractedCommand}`;
            }
            
            return null;
        },

        // 根据用户输入获取默认命令 - 现在完全依赖服务器端返回
        getCommandFromUserInput(input) {
            // 不再返回预设命令，完全依赖服务器端的返回结果
            return null;
        }
    }
});

console.log('Vue app created, mounting to #app');
try {
    app.mount('#app');
    console.log('Vue app mounted successfully');
} catch (error) {
    console.error('Vue app mounting failed:', error);
    console.error('错误堆栈:', error.stack);
}
