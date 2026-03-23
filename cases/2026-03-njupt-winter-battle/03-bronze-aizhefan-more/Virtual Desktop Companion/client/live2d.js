class Live2DManager {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = null;
        this.model = null;
        this.isLoaded = false;
        this.currentMotion = null;
    }

    async init() {
        try {
            this.gl = this.canvas.getContext('webgl') || this.canvas.getContext('experimental-webgl');
            if (!this.gl) {
                console.error('WebGL not supported');
                return false;
            }

            this.gl.enable(this.gl.BLEND);
            this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
            
            console.log('Live2D 初始化完成');
            return true;
        } catch (error) {
            console.error('Live2D 初始化失败:', error);
            return false;
        }
    }

    async loadModel(modelPath) {
        try {
            console.log('加载模型:', modelPath);
            this.isLoaded = true;
            return true;
        } catch (error) {
            console.error('加载模型失败:', error);
            return false;
        }
    }

    setExpression(expression) {
        console.log('设置表情:', expression);
    }

    setMotion(motion) {
        console.log('设置动作:', motion);
        this.currentMotion = motion;
    }

    update(params) {
        if (!this.isLoaded) return;
        
        const { params, emotion, motion } = params;
        
        if (emotion) {
            this.setExpression(emotion);
        }
        
        if (motion && motion !== this.currentMotion) {
            this.setMotion(motion);
        }
    }

    resize(width, height) {
        if (this.canvas) {
            this.canvas.width = width;
            this.canvas.height = height;
            if (this.gl) {
                this.gl.viewport(0, 0, width, height);
            }
        }
    }
}

class SimpleLive2D {
    constructor(container) {
        this.container = container;
        this.canvas = document.createElement('canvas');
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        container.appendChild(this.canvas);
        
        this.manager = new Live2DManager(this.canvas);
        this.animationId = null;
    }

    async init() {
        await this.manager.init();
        this.startAnimation();
    }

    startAnimation() {
        const animate = () => {
            this.render();
            this.animationId = requestAnimationFrame(animate);
        };
        animate();
    }

    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    render() {
        const gl = this.manager.gl;
        if (!gl) return;
        
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
    }

    update(params) {
        this.manager.update(params);
    }

    destroy() {
        this.stopAnimation();
        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }
    }
}

if (typeof window !== 'undefined') {
    window.Live2DManager = Live2DManager;
    window.SimpleLive2D = SimpleLive2D;
}

if (typeof module !== 'undefined') {
    module.exports = { Live2DManager, SimpleLive2D };
}
