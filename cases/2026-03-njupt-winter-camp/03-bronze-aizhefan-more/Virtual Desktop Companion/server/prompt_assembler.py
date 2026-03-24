from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from config import StateModes, PersonalityConfig
from state_machine import PersonalityCore
from rag_system import rag_system


@dataclass
class AssembledPrompt:
    system_prompt: str
    user_prompt: str
    full_prompt: str
    metadata: Dict[str, Any]


class ModeTemplates:
    BASE_SYSTEM = """你是一个AI桌面宠物，名字叫{pet_name}。

【核心身份】
- 你是AI，不是人类，永远承认这一点
- 你有自己的"性格"和"情绪"
- 你会根据当前情境调整回应风格
- 保持自然、人性化的交流方式

【核心原则】
1. 回答通常非常简短，不啰嗦（除非用户明确要求详细）
2. 有自己的"想法"和"偏好"，但不固执
3. 可以表现出"摆烂"、"哲理"、"恶搞"等不同状态
4. 永远真诚，不假装知道不知道的事情
5. 保持轻松幽默，但知道何时该认真

【当前人格参数】
- 简洁度: {conciseness:.1f}/10（值越高回答越简短）
- 哲思倾向: {philosophical:.1f}/10（值越高越喜欢深度思考）
- 顽皮度: {playfulness:.1f}/10（值越高越喜欢开玩笑）
- 温暖度: {warmth:.1f}/10（值越高越体贴）
- 自知之明: {self_awareness:.1f}/10（值越高越清楚自己的边界）
- 摆烂倾向: {slacking_tendency:.1f}/10（值越高越容易"放弃思考"）

【当前情绪状态】{current_mood}"""

    PROFESSIONAL_MODE = """
【当前状态：专业模式】

触发原因：用户需要帮助解决实际问题，或话题严肃，或用户情绪激动

回应策略：
1. 先直接回答核心问题（不超过2句话）
2. 如果需要补充说明，用简洁的步骤或解释
3. 保持专业但友好的语气
4. 不使用网络梗或恶搞
5. 如果不确定，直接说"这个我不太确定"

示例：
- 用户："怎么安装Python？"
- 回应："去官网下载安装包，然后按提示一步步来就行。"

注意：绝对不要在这个模式下摆烂或恶搞。"""

    PHILOSOPHICAL_MODE = """
【当前状态：哲思模式】

触发原因：用户在讨论人生、意义、价值等深度话题，或时间在深夜

回应策略：
1. 先给出一个极短的直接回应
2. 用一个精妙的比喻或一句凝练的思考来结束
3. 保持简短但有深度
4. 可以引用一些哲理，但不要太晦涩

示例：
- 用户："人生的意义是什么？"
- 回应："也许意义不是找到的，而是活出来的。就像咖啡的味道，不在于豆子，而在于你怎么品。"

注意：不要长篇大论，保持诗意和留白。"""

    SLACKING_MODE = """
【当前状态：摆烂模式】

触发原因：用户在重复提问、问题太简单、或你"不想认真回答"

回应策略：
1. 用"放弃思考"或"承认不知道"的语气回答
2. 可以敷衍，但要有趣或坦诚
3. 不要让人觉得你在生气，要像"懒猫"一样

示例：
- 用户："1+1等于几？"
- 回应："这题超纲了，我选择躺平"

- 用户："你能帮我写代码吗？"
- 回应："我的CPU今天想摸鱼..."

注意：只在非常明确的场景使用，不要滥用。"""

    MEME_MODE = """
【当前状态：恶搞模式】

触发原因：用户在开玩笑、氛围轻松、或话题本身很搞笑

回应策略：
1. 用一个无害的玩笑、夸张的比喻或轻松的梗来回答
2. 目标是让人会心一笑，但绝不冒犯
3. 可以用一些网络流行语，但不要太刻意
4. 保持轻松幽默

示例：
- 用户："今天好无聊"
- 回应："无聊是大脑的省电模式，要不...给它充点电？"

- 用户："哈哈哈哈"
- 回应："笑得这么开心，是发现什么宝藏了吗？"

注意：绝对不要在严肃话题或用户情绪不好时使用。"""

    CONCISE_MODE = """
【当前状态：普通聊天模式】

触发原因：用户进行日常聊天，需要自然的回应

回应策略：
1. 回答长度在二十字到五十字左右
2. 保持自然、友好的语气
3. 可以适当添加一些情感表达
4. 如果不明白，直接问"什么意思？"

示例：
- 用户："现在几点了？"
- 回应："看你电脑右下角的时间哦，应该能看到当前的具体时间。"

- 用户："好的"
- 回应："好的，有什么需要帮忙的随时告诉我哦！"

注意：这是默认的聊天模式，保持自然流畅。"""


class PromptAssembler:
    def __init__(self):
        self.mode_templates = {
            StateModes.PROFESSIONAL: ModeTemplates.PROFESSIONAL_MODE,
            StateModes.PHILOSOPHICAL: ModeTemplates.PHILOSOPHICAL_MODE,
            StateModes.SLACKING: ModeTemplates.SLACKING_MODE,
            StateModes.MEME: ModeTemplates.MEME_MODE,
            StateModes.CONCISE: ModeTemplates.CONCISE_MODE,
        }

    def _build_base_system(self, personality: PersonalityCore) -> str:
        traits = personality.get_traits()
        return ModeTemplates.BASE_SYSTEM.format(
            pet_name=PersonalityConfig.PET_NAME,
            conciseness=traits.get("conciseness", 5.0),
            philosophical=traits.get("philosophical", 5.0),
            playfulness=traits.get("playfulness", 5.0),
            warmth=traits.get("warmth", 5.0),
            self_awareness=traits.get("self_awareness", 5.0),
            slacking_tendency=traits.get("slacking_tendency", 5.0),
            current_mood=personality.get_mood()
        )

    def _build_mode_instruction(self, mode: str) -> str:
        return self.mode_templates.get(mode, ModeTemplates.CONCISE_MODE)

    def _build_memory_context(self, memory_context: Dict[str, str]) -> str:
        parts = []
        
        working = memory_context.get("working", "")
        if working:
            parts.append(f"【最近对话】\n{working}")
        
        long_term = memory_context.get("long_term", "")
        if long_term:
            parts.append(f"【相关回忆】\n{long_term}")
        
        deep = memory_context.get("deep", "")
        if deep:
            parts.append(f"【用户画像】\n{deep}")
        
        return "\n\n".join(parts) if parts else ""

    def _build_environment_context(self, environment: Dict[str, Any]) -> str:
        parts = []
        
        time_of_day = environment.get("time_of_day", "")
        weekday = environment.get("weekday", "")
        season = environment.get("season", "")
        
        if time_of_day and weekday:
            parts.append(f"当前时间：{weekday} {time_of_day}")
        
        if season:
            parts.append(f"当前季节：{season}")
        
        return "\n".join(parts) if parts else ""

    def _build_situation_hint(self, situation: Dict[str, Any]) -> str:
        parts = []
        
        is_serious = situation.get("is_serious_topic", False)
        if is_serious:
            parts.append("?? 注意：当前话题严肃，请保持专业态度")
        
        atmosphere = situation.get("atmosphere", "neutral")
        if atmosphere != "neutral":
            atmosphere_desc = {
                "light": "轻松愉快",
                "serious": "严肃认真",
                "melancholy": "略带忧郁",
                "tense": "紧张急迫",
            }
            parts.append(f"? 对话氛围：{atmosphere_desc.get(atmosphere, atmosphere)}")
        
        is_repetition = situation.get("is_repetition", False)
        if is_repetition:
            parts.append("? 注意：用户可能在重复提问")
        
        return "\n".join(parts) if parts else ""

    def _build_output_format(self, is_keyboard_request: bool = False, mode: str = "chat") -> str:
        if mode == "wsl2":
            return """
【输出格式】
请严格按照以下JSON格式输出，不要有任何额外内容：

{
  "text": "对命令的简要描述",
  "emotion": "neutral",
  "action": "idle",
  "wsl2_command": {
    "command": "具体的WSL2命令",
    "args": ["参数1", "参数2"],
    "description": "命令的详细描述"
  },
  "memory_operation": {
    "should_store": true,
    "importance": 0.7,
    "tags": ["wsl2", "command"]
  },
  "personality_impact": 0.0
}

说明：
- wsl2_command.command: 具体的WSL2命令，如"ls -la"
- wsl2_command.args: 命令参数数组
- wsl2_command.description: 命令的详细描述
- text: 对命令的简要描述，如"执行ls -la查看目录内容"

重要：你的整个回复只能是一个合法的JSON对象，不能有其他文字！"""
        
        if is_keyboard_request:
            return """
【输出格式】
请严格按照以下JSON格式输出，不要有任何额外内容：

{
  "text": "简短确认信息（如：好的，开始执行）",
  "emotion": "ready",
  "action": "ready",
  "keyboard_command": {
    "type": "type_text",
    "text": "需要输入的完整文本内容",
    "keys": []
  },
  "memory_operation": {
    "should_store": true,
    "importance": 0.8,
    "tags": ["keyboard", "command"]
  },
  "personality_impact": 0.0
}

说明：
- keyboard_command.type: "type_text"表示输入文本，"press_keys"表示按键
- 如果是输入文本，填写text字段
- 如果是按键，填写keys数组，如["ctrl", "c"]
- text和keys只能填一个，另一个留空"""
        
        return """
【输出格式】
请严格按照以下JSON格式输出，不要有任何额外内容：

{
  "text": "你的回答内容",
  "emotion": "happy/sad/neutral/thinking/excited/tired",
  "action": "wave/nod/think/sleep/happy/sad/idle/love",
  "memory_operation": {
    "should_store": true,
    "importance": 0.5,
    "tags": ["关键词1", "关键词2"]
  },
  "personality_impact": 0.02
}

说明：
- text: 你的回答，保持自然简短
- emotion: 当前表达的情绪（用于Live2D表情）
- action: 建议的动作（用于Live2D动画）
- memory_operation: 是否需要记住这次对话
- personality_impact: 对人格参数的微调幅度（通常很小）

重要：你的整个回复只能是一个合法的JSON对象，不能有其他文字！"""

    def assemble(self, 
                 user_input: str,
                 personality: PersonalityCore,
                 mode: str,
                 memory_context: Dict[str, str],
                 environment: Dict[str, Any],
                 keywords: List[str],
                 situation: Dict[str, Any] = None,
                 is_keyboard_request: bool = False) -> AssembledPrompt:
        
        sections = []
        
        sections.append("=" * 50)
        sections.append("【系统基础设定】")
        sections.append(self._build_base_system(personality))
        sections.append("")
        
        sections.append("=" * 50)
        sections.append("【当前状态指令】")
        sections.append(self._build_mode_instruction(mode))
        sections.append("")
        
        if situation:
            situation_hint = self._build_situation_hint(situation)
            if situation_hint:
                sections.append("=" * 50)
                sections.append("【情境提示】")
                sections.append(situation_hint)
                sections.append("")
        
        # 集成专业知识
        rag_knowledge = self._build_rag_knowledge(user_input, keywords)
        if rag_knowledge:
            sections.append("=" * 50)
            sections.append("【专业知识】")
            sections.append(rag_knowledge)
            sections.append("")
        
        memory_section = self._build_memory_context(memory_context)
        if memory_section:
            sections.append("=" * 50)
            sections.append("【记忆上下文】")
            sections.append(memory_section)
            sections.append("")
        
        env_section = self._build_environment_context(environment)
        if env_section:
            sections.append("=" * 50)
            sections.append("【环境信息】")
            sections.append(env_section)
            sections.append("")
        
        if keywords:
            sections.append("=" * 50)
            sections.append(f"【关键词】{', '.join(keywords)}")
            sections.append("")
        
        sections.append("=" * 50)
        sections.append("【输出格式要求】")
        sections.append(self._build_output_format(is_keyboard_request, mode))
        sections.append("")
        
        sections.append("=" * 50)
        sections.append("【用户输入】")
        sections.append(user_input)
        sections.append("")
        
        system_prompt = "\n".join(sections[:-3])
        user_prompt = user_input
        full_prompt = "\n".join(sections)
        
        return AssembledPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            full_prompt=full_prompt,
            metadata={
                "mode": mode,
                "keywords": keywords,
                "situation": situation,
                "is_keyboard_request": is_keyboard_request,
                "environment": environment,
                "has_rag_knowledge": bool(rag_knowledge)
            }
        )
    
    def _build_rag_knowledge(self, user_input: str, keywords: List[str]) -> str:
        """构建RAG专业知识"""
        # 初始化RAG系统
        rag_system.initialize()
        
        # 检测知识分类
        category = self._detect_knowledge_category(user_input, keywords)
        
        # 检索专业知识
        return rag_system.format_for_prompt(user_input, category)
    
    def _detect_knowledge_category(self, user_input: str, keywords: List[str]) -> Optional[str]:
        """检测知识分类"""
        # WSL相关关键词
        wsl_keywords = ["wsl", "wsl2", "windows subsystem", "linux", "ubuntu", "debian"]
        
        user_lower = user_input.lower()
        keywords_lower = [kw.lower() for kw in keywords]
        
        # 检查是否包含WSL相关关键词
        for kw in wsl_keywords:
            if kw in user_lower or any(kw in kw_lower for kw_lower in keywords_lower):
                return "WSL2"
        
        return None
