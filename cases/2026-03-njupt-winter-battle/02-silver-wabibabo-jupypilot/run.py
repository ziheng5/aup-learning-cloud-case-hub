#!/usr/bin/env python3
"""
一键启动 JupyPilot UI 页面。

用法：
    python run.py              # 自动选择 voila 或 jupyter notebook
    python run.py --voila      # 强制用 voila（纯净 UI，无代码）
    python run.py --notebook   # 强制用 jupyter notebook
    python run.py --port 8899  # 指定端口
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 JupyPilot UI")
    parser.add_argument("--voila", action="store_true", help="用 voila 启动（纯净 UI）")
    parser.add_argument("--notebook", action="store_true", help="用 jupyter notebook 启动")
    parser.add_argument("--port", type=int, default=8866, help="端口号（默认 8866）")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    notebook = "app.ipynb"
    browser_flag = "--no-browser" if args.no_browser else ""

    if args.voila or (not args.notebook and shutil.which("voila")):
        # voila 模式：只显示 widget 输出，隐藏代码
        cmd = ["voila", notebook, f"--port={args.port}"]
        if args.no_browser:
            cmd.append("--no-browser")
        print(f"[JupyPilot] 用 voila 启动: http://localhost:{args.port}")
    else:
        # jupyter notebook 模式
        # 优先用 jupyter CLI 入口（pip install notebook 会注册此命令）
        jupyter_bin = shutil.which("jupyter")
        if jupyter_bin:
            cmd = [jupyter_bin, "notebook", notebook, f"--port={args.port}"]
        else:
            # 回退：直接调用 notebook 模块（notebook 7.x 支持）
            cmd = [sys.executable, "-m", "notebook", notebook, f"--port={args.port}"]
        if args.no_browser:
            cmd.append("--no-browser")
        print(f"[JupyPilot] 用 jupyter notebook 启动: http://localhost:{args.port}")

    print(f"   命令: {' '.join(cmd)}")
    print("   按 Ctrl+C 停止\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
