import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from config import settings


@dataclass
class SkillResponse:
    success: bool
    data: Any
    message: str
    skill_name: str
    execution_time: float


class BaseSkill:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    async def execute(self, params: Dict[str, Any]) -> SkillResponse:
        """执行技能"""
        start_time = time.time()
        try:
            result = await self._execute(params)
            execution_time = time.time() - start_time
            return SkillResponse(
                success=True,
                data=result,
                message=f"技能 {self.name} 执行成功",
                skill_name=self.name,
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return SkillResponse(
                success=False,
                data=None,
                message=f"技能 {self.name} 执行失败: {str(e)}",
                skill_name=self.name,
                execution_time=execution_time
            )
    
    async def _execute(self, params: Dict[str, Any]) -> Any:
        """子类实现具体执行逻辑"""
        raise NotImplementedError
    
    def get_info(self) -> Dict[str, Any]:
        """获取技能信息"""
        return {
            "name": self.name,
            "description": self.description
        }


class WSL2Skill(BaseSkill):
    def __init__(self):
        super().__init__(
            name="wsl2",
            description="WSL2相关操作和信息"
        )
    
    async def _execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行WSL2相关操作"""
        action = params.get("action", "info")
        
        if action == "info":
            return {
                "type": "info",
                "data": {
                    "version": "WSL2",
                    "description": "Windows Subsystem for Linux 2",
                    "features": [
                        "完整的Linux内核",
                        "更好的性能",
                        "Docker支持",
                        "GUI应用支持"
                    ]
                }
            }
        
        elif action == "commands":
            return {
                "type": "commands",
                "data": {
                    "basic": [
                        "wsl --list -v",
                        "wsl --set-default <distro>",
                        "wsl --shutdown",
                        "wsl --terminate <distro>",
                        "wsl -d <distro>"
                    ],
                    "advanced": [
                        "wsl --export <distro> <filename>",
                        "wsl --import <distro> <install_location> <filename>",
                        "wsl --set-version <distro> <version>",
                        "wsl --update"
                    ]
                }
            }
        
        elif action == "troubleshooting":
            return {
                "type": "troubleshooting",
                "data": {
                    "common_errors": [
                        {
                            "error": "0x80370102",
                            "solution": "检查Hyper-V是否正确安装"
                        },
                        {
                            "error": "0x800701bc",
                            "solution": "运行 'wsl --update' 更新WSL内核"
                        },
                        {
                            "error": "网络问题",
                            "solution": "尝试重启WSL或检查Windows防火墙设置"
                        }
                    ]
                }
            }
        
        elif action == "execute_command":
            command = params.get("command", "")
            if not self._is_safe_command(command):
                return {
                    "type": "error",
                    "data": {
                        "message": "命令执行被拒绝：不安全的命令"
                    }
                }
            
            try:
                result = await self._run_command(command, params.get("timeout", 300))
                return {
                    "type": "command_result",
                    "data": {
                        "command": command,
                        "stdout": result["stdout"],
                        "stderr": result["stderr"],
                        "returncode": result["returncode"],
                        "execution_time": result["execution_time"]
                    }
                }
            except Exception as e:
                return {
                    "type": "error",
                    "data": {
                        "message": f"命令执行失败：{str(e)}"
                    }
                }
        
        else:
            return {
                "type": "error",
                "data": {
                    "message": f"不支持的操作: {action}"
                }
            }
    
    def _is_safe_command(self, command: str) -> bool:
        """检查命令是否安全"""
        safe_commands = [
            "wsl", "wsl.exe",
            "ls", "pwd", "cd", "cat", "echo", "mkdir", "rmdir", "rm", "cp", "mv",
            "sudo", "apt", "apt-get", "aptitude", "dpkg", "apt-cache",
            "git", "python", "python3", "pip", "pip3", "node", "npm", "yarn",
            "docker", "docker-compose", "kubectl",
            "ssh", "scp", "curl", "wget", "ping", "traceroute", "netstat",
            "ps", "top", "htop", "df", "du", "free", "uname", "whoami"
        ]
        
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return False
        
        base_cmd = cmd_parts[0]
        return base_cmd in safe_commands
    
    async def _run_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """执行命令并返回结果"""
        import subprocess
        import asyncio
        import time
        
        start_time = time.time()
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            return {
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "returncode": process.returncode,
                "execution_time": execution_time
            }
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return {
                "stdout": "",
                "stderr": "命令执行超时",
                "returncode": -1,
                "execution_time": execution_time
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "stdout": "",
                "stderr": f"执行错误: {str(e)}",
                "returncode": -1,
                "execution_time": execution_time
            }


class SystemSkill(BaseSkill):
    def __init__(self):
        super().__init__(
            name="system",
            description="系统相关操作和信息"
        )
    
    async def _execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行系统相关操作"""
        action = params.get("action", "status")
        
        if action == "status":
            return {
                "type": "status",
                "data": {
                    "service": "AI Desktop Pet Server",
                    "version": "1.0.0",
                    "status": "running",
                    "timestamp": time.time()
                }
            }
        
        elif action == "config":
            return {
                "type": "config",
                "data": {
                    "ollama_model": settings.OLLAMA_MODEL,
                    "deepseek_enabled": settings.USE_DEEPSEEK_FOR_COMPLEX,
                    "working_memory_size": settings.WORKING_MEMORY_SIZE
                }
            }
        
        elif action == "memory":
            return {
                "type": "memory",
                "data": {
                    "working_memory_size": settings.WORKING_MEMORY_SIZE,
                    "long_term_memory_top_k": settings.LONG_TERM_MEMORY_TOP_K
                }
            }
        
        else:
            return {
                "type": "error",
                "data": {
                    "message": f"不支持的操作: {action}"
                }
            }


class RAGSkill(BaseSkill):
    def __init__(self):
        super().__init__(
            name="rag",
            description="专业知识检索和管理"
        )
    
    async def _execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行RAG相关操作"""
        from rag_system import rag_system
        
        action = params.get("action", "search")
        
        if action == "search":
            query = params.get("query", "")
            category = params.get("category", None)
            
            rag_system.initialize()
            knowledge_items = rag_system.search(query, category)
            
            return {
                "type": "search",
                "data": {
                    "query": query,
                    "category": category,
                    "results": knowledge_items
                }
            }
        
        elif action == "categories":
            rag_system.initialize()
            categories = rag_system.get_categories()
            
            return {
                "type": "categories",
                "data": {
                    "categories": categories
                }
            }
        
        elif action == "initialize":
            rag_system.initialize()
            return {
                "type": "initialize",
                "data": {
                    "message": "RAG系统初始化成功"
                }
            }
        
        else:
            return {
                "type": "error",
                "data": {
                    "message": f"不支持的操作: {action}"
                }
            }


class SkillSystem:
    def __init__(self):
        self.skills: Dict[str, BaseSkill] = {}
        self._register_skills()
    
    def _register_skills(self):
        """注册技能"""
        skills = [
            WSL2Skill(),
            SystemSkill(),
            RAGSkill()
        ]
        
        for skill in skills:
            self.skills[skill.name] = skill
    
    async def execute_skill(self, skill_name: str, params: Dict[str, Any]) -> SkillResponse:
        """执行技能"""
        if skill_name not in self.skills:
            return SkillResponse(
                success=False,
                data=None,
                message=f"技能 {skill_name} 不存在",
                skill_name=skill_name,
                execution_time=0.0
            )
        
        skill = self.skills[skill_name]
        return await skill.execute(params)
    
    def get_skill_info(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """获取技能信息"""
        if skill_name not in self.skills:
            return None
        return self.skills[skill_name].get_info()
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """列出所有技能"""
        return [skill.get_info() for skill in self.skills.values()]


# 全局技能系统实例
skill_system = SkillSystem()
