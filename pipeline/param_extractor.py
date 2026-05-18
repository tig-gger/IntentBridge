"""
IntentBridge 第二步：参数提取 (Parameter Extractor)

产品经理笔记：
  用户说"帮我写一封信"和"帮我写一封辞职信"，虽然都是"写作"意图，
  但需要的参数完全不同。这个模块负责：
  1. 从用户输入中提取关键参数（语气、长度、格式、主题等）
  2. 识别缺失的关键信息
  3. 标记需要向用户追问的缺失参数

  这是产品设计中"渐进式揭露"（Progressive Disclosure）原则的体现
  ——不是一次性让用户填所有参数，而是智能提取 + 按需追问。
"""

import re


# 每种意图需要关注的参数
INTENT_PARAMS = {
    "writing": {
        "required": [],
        "optional": ["genre", "tone", "length", "audience", "format"],
        "genre": {
            "label": "体裁",
            "options": ["邮件", "文章", "报告", "故事", "文案", "信函", "日记", "其他"],
            "prompt": "你想写什么类型的？邮件、文章、还是其他？",
        },
        "tone": {
            "label": "语气",
            "options": ["正式", "半正式", "轻松", "幽默", "专业"],
            "prompt": "你希望语气是正式还是轻松一些？",
        },
        "length": {
            "label": "篇幅",
            "options": ["短（100字以内）", "中（200-500字）", "长（500字以上）"],
            "prompt": "你希望大概写多长？",
        },
        "audience": {
            "label": "读者",
            "options": [],
            "prompt": "这篇文章的读者是谁？",
        },
    },
    "analysis": {
        "required": ["source"],
        "optional": ["depth", "format"],
        "source": {
            "label": "分析对象",
            "options": [],
            "prompt": "你需要分析什么？（文本、数据、还是文件？）",
        },
        "depth": {
            "label": "分析深度",
            "options": ["快速摘要", "详细分析", "深度研究"],
            "prompt": "你希望分析到什么深度？快速摘要还是详细分析？",
        },
    },
    "coding": {
        "required": ["language"],
        "optional": ["framework", "complexity"],
        "language": {
            "label": "编程语言",
            "options": ["Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "C++", "其他"],
            "prompt": "你用的是什么编程语言？",
        },
    },
    "brainstorming": {
        "required": [],
        "optional": ["quantity", "domain"],
        "quantity": {
            "label": "想法的数量",
            "options": ["3-5个", "5-10个", "10个以上"],
            "prompt": "你希望得到多少个点子？",
        },
    },
}


def extract_parameters(user_input: str, intent: str) -> dict:
    """
    从用户输入中提取结构化参数。

    参数:
        user_input: 用户原始输入
        intent: 意图类别

    返回:
        {
            "extracted": {参数名: 值},   # 已提取的参数
            "missing": [参数名],          # 缺少的重要参数
            "suggestions": [提示信息],    # 对用户的追问建议
        }
    """
    text = user_input.lower()
    result = {
        "extracted": {},
        "missing": [],
        "suggestions": [],
    }

    # 获取该意图对应的参数字典
    param_defs = INTENT_PARAMS.get(intent, INTENT_PARAMS.get("general_qa", {}))

    # 遍历所有可能参数
    for param_name in param_defs.get("required", []) + param_defs.get("optional", []):
        if param_name not in param_defs:
            continue

        param_info = param_defs[param_name]
        param_label = param_info.get("label", param_name)
        extracted_value = _extract_single_param(text, param_name, param_info)

        if extracted_value:
            result["extracted"][param_label] = extracted_value
        elif param_name in param_defs.get("required", []):
            # 必填参数缺失，需要追问
            result["missing"].append(param_name)
            result["suggestions"].append(param_info.get("prompt", f"请提供{param_label}"))

    return result


def _extract_single_param(text: str, param_name: str, param_info: dict) -> str | None:
    """尝试从文本中提取单个参数的值"""
    options = param_info.get("options", [])
    if not options:
        return None

    for option in options:
        option_lower = option.lower()
        # 检查选项是否出现在文本中
        if option_lower in text:
            return option

    return None


# 测试代码
if __name__ == "__main__":
    test_cases = [
        ("帮我写一封辞职信，语气正式", "writing"),
        ("用 Python 写一个爬虫", "coding"),
        ("给我一些创意点子", "brainstorming"),
    ]

    print("=" * 50)
    print("IntentBridge 参数提取测试")
    print("=" * 50)
    for inp, intent in test_cases:
        result = extract_parameters(inp, intent)
        print(f"\n输入: {inp} | 意图: {intent}")
        print(f"  提取参数: {result['extracted']}")
        if result["missing"]:
            print(f"  缺失参数: {result['missing']}")
        if result["suggestions"]:
            print(f"  追问建议: {result['suggestions']}")
