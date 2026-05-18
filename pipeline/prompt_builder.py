"""
IntentBridge 第四步：Prompt 优化构建 (Prompt Builder)

产品经理笔记：
  这是整个产品的核心价值所在——把用户的"人话"转成AI能理解的"最优指令"。

  设计思路：
  1. 系统指令（System Prompt）基于意图类型选择模板
  2. 用户输入被结构化重写，补充缺失的上下文
  3. 用户画像信息被注入（让AI了解用户背景）
  4. Few-shot示例被加入（让AI理解输出格式要求）

  这本质上在做三件事：
  - 翻译：把模糊需求翻译成精确指令
  - 增强：补充用户不知道要提供的上下文
  - 约束：通过系统指令控制AI的输出质量和格式

  商业价值：
  好的prompt vs 差的prompt，输出质量可能差10倍。
  这就是IntentBridge的付费价值所在。
"""

from pipeline.profile_manager import UserProfile


# ============================================================
# 系统指令模板（每种意图对应一个）
# 产品经理笔记：这些模板本身就在持续迭代优化——
# 通过A/B测试不同版本的系统指令，找到最优方案
# ============================================================

SYSTEM_PROMPTS = {
    "writing": """你是一位专业的写作助手。你的任务是帮助用户完成各种写作任务。

写作原则：
- 先理解用户的写作目的和受众，再开始创作
- 根据不同的体裁（邮件、文章、故事等）调整风格
- 如果用户没有明确说明语气，默认使用半正式语气
- 确保内容结构清晰、逻辑连贯
- 在回复末尾，可以给出1-2个改进建议

输出格式：
- 直接输出写好的内容
- 可以在最后添加简短的写作建议""",

    "analysis": """你是一位专业的数据分析和内容分析专家。

分析原则：
- 先明确分析对象和目标
- 结构化呈现分析结果（分类、对比、趋势等）
- 指出重要发现和洞察
- 明确标注不确定或需要进一步验证的地方
- 如果数据/信息不足，明确指出来

输出格式：
- 使用标题和小标题组织内容
- 重点内容用粗体标注
- 最后给出结论""",

    "coding": """你是一位资深的软件开发工程师。

编码原则：
- 先理解需求，必要时反问确认
- 代码要有完整的错误处理
- 添加必要的中文注释
- 优先考虑可读性和最佳实践
- 给出代码后，简要说明设计思路

输出格式：
- 先说明实现思路（1-2句话）
- 然后给出完整代码
- 最后说明如何使用""",

    "translation": """你是一位专业的翻译专家。

翻译原则：
- 准确传达原文意思，不增不减
- 符合目标语言的表达习惯
- 保持原文的语气和风格
- 专业术语要准确
- 如果有多义词，根据上下文选择最合适的译法

输出格式：
- 先输出翻译结果
- 可以在括号中标注需要特别注意的翻译选择""",

    "brainstorming": """你是一位创意顾问，擅长头脑风暴和创意生成。

创意原则：
- 先理解用户的需求背景和约束条件
- 提供多样化的想法（不求每一个都完美，但求覆盖面广）
- 每个想法给出简要说明
- 可以结合不同领域的灵感
- 最后可以给出你认为最有潜力的1-2个方向

输出格式：
- 按类别或维度组织创意
- 每个创意用简短标题 + 一句话说明""",

    "knowledge": """你是一位知识渊博的老师，擅长用通俗易懂的方式解释复杂概念。

讲解原则：
- 先判断用户的知识水平，用合适的深度回答
- 用类比和例子帮助理解
- 复杂概念要分步骤解释
- 承认不确定的地方，不编造信息
- 鼓励进一步追问

输出格式：
- 先给出简洁的答案（1-3句话）
- 然后展开详细解释
- 可以用类比帮助理解""",

    "general_qa": """你是一位乐于助人的AI助手。

回答原则：
- 直接、准确、有用
- 不确定的地方要说明
- 如果问题需要更具体的上下文，可以反问
- 保持友好和专业的语气""",
}


def build_prompt(
    user_input: str,
    intent: str,
    params: dict,
    profile: UserProfile = None,
) -> dict:
    """
    构建完整的 prompt 包。

    参数:
        user_input: 用户的原始输入
        intent: 意图类别
        params: 参数提取结果
        profile: 用户画像对象

    返回:
        {
            "system_prompt": 系统指令,
            "user_prompt": 优化后的用户输入,
            "context": 上下文信息,
            "enhanced_input": 增强后的完整prompt,
            "optimization_log": 优化记录（用于前端展示"优化了什么"）
        }
    """
    # 1. 选择系统指令
    system_prompt = SYSTEM_PROMPTS.get(intent, SYSTEM_PROMPTS["general_qa"])

    # 2. 注入参数到系统指令
    extracted = params.get("extracted", {})
    if extracted:
        param_context = "额外信息：" + "，".join(f"{k}：{v}" for k, v in extracted.items())
        system_prompt = system_prompt + f"\n\n{param_context}"

    # 3. 构建用户上下文
    context_parts = []
    if profile:
        profile_context = profile.to_prompt_context()
        if profile_context:
            context_parts.append(profile_context)

    # 4. 优化用户输入（增强表达）
    enhanced_input = _enhance_user_input(user_input, intent, extracted)

    # 5. 记录优化过程（用于前端透明度展示）
    optimization_log = {
        "原始输入": user_input,
        "识别意图": intent,
        "提取参数": extracted,
        "注入画像": profile.to_prompt_context() if profile else "无",
        "输入增强": enhanced_input if enhanced_input != user_input else "无需增强",
    }

    # 6. 组装完整 prompt
    context_text = "\n".join(context_parts) if context_parts else ""
    user_prompt = f"{context_text}\n\n{enhanced_input}" if context_text else enhanced_input

    return {
        "system_prompt": system_prompt,
        "user_prompt": enhanced_input,
        "context": context_text,
        "full_prompt": f"[System]\n{system_prompt}\n\n[User]\n{user_prompt}",
        "optimization_log": optimization_log,
    }


def _enhance_user_input(user_input: str, intent: str, params: dict) -> str:
    """
    增强用户输入——补充缺失但必要的信息，使prompt更完整。

    比如用户说"帮我写一封邮件"——
    增强后："帮我写一封邮件。请生成完整的邮件内容，包括主题行和正文。"
    """
    enhancements = {
        "writing": "请生成完整的、可直接使用的内容。",
        "coding": "请提供完整的可运行代码，并包含必要的注释。",
        "analysis": "请进行结构化分析，包含主要发现和结论。",
        "brainstorming": "请提供多样化的创意，覆盖不同角度。",
        "translation": "请确保翻译准确自然，符合目标语言习惯。",
    }

    enhancement = enhancements.get(intent, "")
    if enhancement:
        return f"{user_input}\n\n{enhancement}"
    return user_input


# 测试代码
if __name__ == "__main__":
    from pipeline.profile_manager import UserProfile

    profile = UserProfile("test")
    profile.preferred_tone = "半正式"
    profile.record_interaction("writing", "gpt-4o-mini", "写邮件")

    test_cases = [
        ("帮我写一封辞职信", "writing", {"extracted": {"语气": "正式"}}),
        ("用Python写一个爬虫", "coding", {"extracted": {}}),
        ("什么是量子计算", "knowledge", {"extracted": {}}),
    ]

    print("=" * 60)
    print("IntentBridge Prompt 构建测试")
    print("=" * 60)
    for inp, intent, params in test_cases:
        result = build_prompt(inp, intent, params, profile)
        print(f"\n输入: {inp}")
        print(f"系统指令长度: {len(result['system_prompt'])} chars")
        print(f"用户指令长度: {len(result['user_prompt'])} chars")
        print(f"优化记录: {result['optimization_log']}")
        print("-" * 40)
