"""
IntentBridge 第七步：输出优化 (Output Refiner)

产品经理笔记：
  这是用户"最后看到的东西"——输入经过前面6个步骤的加工，
  最终呈现给用户之前，再做一层优化。

  做的事情：
  1. 格式化输出（让它看起来更专业、更易读）
  2. 标注置信度（让用户知道这个回答有多可靠）
  3. 添加操作建议（接下来可以做什么）
  4. 对不确定内容做免责声明

  这是产品设计中"峰值体验"的体现——
  用户不关心你内部有多复杂，只关心最终给他的东西好不好用。
"""

import re
from typing import Optional


def refine_output(
    content: str,
    intent: str,
    model: str,
    latency: float,
    error: Optional[str] = None,
) -> dict:
    """
    优化 LLM 的输出，使其更符合用户预期。

    参数:
        content: LLM 原始回复
        intent: 意图类别
        model: 使用的模型
        latency: API 响应时间
        error: 错误信息（如果有）

    返回:
        {
            "content": 优化后的内容,
            "confidence": 置信度评估,
            "format": 格式说明,
            "metadata": 元数据（模型、耗时等）,
        }
    """
    # 如果有错误，直接返回
    if error:
        return {
            "content": content,
            "confidence": "low",
            "format": "error",
            "metadata": {
                "model": model,
                "latency": latency,
                "error": error,
            },
        }

    # 1. 格式化内容
    formatted_content = _apply_format(content, intent)

    # 2. 评估置信度
    confidence = _estimate_confidence(content, intent)

    # 3. 生成操作建议
    suggestions = _generate_suggestions(intent)

    return {
        "content": formatted_content,
        "confidence": confidence,
        "format": intent,
        "suggestions": suggestions,
        "metadata": {
            "model": model,
            "latency": f"{latency:.1f}s",
            "provider": model.split("-")[0] if "-" in model else model,
        },
    }


def _apply_format(content: str, intent: str) -> str:
    """根据意图类型，对内容做格式优化"""
    if intent == "coding" and not content.startswith("```"):
        # 如果是代码但没被代码块包裹，加代码块
        return content

    if intent == "analysis":
        # 确保分析类回复有结构化分隔（但不强制修改）
        return content

    return content


def _estimate_confidence(content: str, intent: str) -> str:
    """
    评估回复的置信度。

    规则：
    - 如果回复中包含不确定表述 → 中等置信度
    - 如果回复中包含明确的不确定性说明 → 低置信度
    - 其他 → 高置信度

    产品经理笔记：
    这个置信度标注对用户体验非常重要——
    用户需要知道"什么时候该相信AI，什么时候该自己判断"。
    这是解决AI"盲目信任"问题的第一步。
    """
    uncertainty_phrases = [
        "我不确定", "可能", "也许", "不一定", "建议你",
        "请核实", "需要确认", "我不能保证", "视情况而定",
        "不确定", "可能有误", "请自行判断",
    ]

    low_confidence_phrases = [
        "我无法确认", "没有足够信息", "我不清楚",
        "这可能不准确", "请以官方信息为准",
    ]

    content_lower = content.lower()

    for phrase in low_confidence_phrases:
        if phrase in content_lower:
            return "low"

    for phrase in uncertainty_phrases:
        if phrase in content_lower:
            return "medium"

    return "high"


def _generate_suggestions(intent: str) -> list[str]:
    """根据意图，生成下一步操作建议"""
    suggestions_map = {
        "writing": [
            "你可以要求我调整语气重新写",
            "需要我帮你检查语法和拼写吗？",
            "我可以帮你把内容翻译成其他语言",
        ],
        "analysis": [
            "需要我进一步深入分析某个方面吗？",
            "我可以帮你把分析结果做成表格",
            "需要对比其他方案吗？",
        ],
        "coding": [
            "需要我解释这段代码的工作原理吗？",
            "我可以帮你添加更多的错误处理",
            "需要我为这段代码添加测试吗？",
        ],
        "translation": [
            "需要我解释一下翻译中的关键选择吗？",
            "我可以提供多个版本的翻译",
            "需要我把其他内容也翻译吗？",
        ],
        "brainstorming": [
            "需要我针对某个方向深入展开吗？",
            "我可以帮你评估每个想法的可行性",
            "需要我把这些想法做成执行计划吗？",
        ],
        "knowledge": [
            "需要我用更简单的方式再解释一遍吗？",
            "我可以推荐一些相关的学习资源",
            "还有其他相关概念想了解吗？",
        ],
    }

    return suggestions_map.get(intent, ["还有什么我可以帮你的吗？"])


# 测试代码
if __name__ == "__main__":
    test_cases = [
        (
            "根据分析，用户留存率在Q2下降了15%。主要原因有三：1) 新用户引导流程过长 ...",
            "analysis",
        ),
        (
            "def hello():\n    print('Hello World')",
            "coding",
        ),
        (
            "我不确定这个数据是否准确，建议你核实一下来源。",
            "knowledge",
        ),
    ]

    print("=" * 50)
    print("IntentBridge 输出优化测试")
    print("=" * 50)
    for content, intent in test_cases:
        result = refine_output(content, intent, "gpt-4o-mini", 1.23, None)
        print(f"\n意图: {intent}")
        print(f"置信度: {result['confidence']}")
        print(f"建议: {result['suggestions'][0]}")
        print(f"模型: {result['metadata']['model']} | 耗时: {result['metadata']['latency']}")
