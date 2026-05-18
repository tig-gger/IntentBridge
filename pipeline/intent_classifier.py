"""
IntentBridge 第一步：意图识别 (Intent Classifier)

产品经理笔记：
  意图识别是整个产品的核心——它是"理解用户"的第一步。
  我们不用复杂的机器学习模型，而是用 LLM + Few-shot 示例来做分类。
  这样做的好处是：
  1. 不需要标注大量训练数据
  2. 可以快速迭代分类体系
  3. 对新意图的泛化能力强

  当前分类体系（V1）：
  - writing: 写作创作（邮件、文章、文案等）
  - analysis: 分析总结（数据分析、文本总结等）
  - coding: 编程开发（代码编写、调试等）
  - translation: 翻译转换（语言翻译、格式转换等）
  - brainstorming: 创意构思（头脑风暴、方案策划等）
  - knowledge: 知识问答（事实查询、概念解释等）

  面试时可以讨论：
  为什么选这6类？——基于用户研究中的高频场景
  为什么不继续细分？——MVP阶段需要平衡准确率和覆盖度
  怎么评估分类质量？——准确率、召回率、用户纠错率
"""

import json
from pathlib import Path


def classify_intent(user_input: str) -> tuple:
    """
    基于规则快速判断用户意图类别。
    使用关键词匹配 + 优先级规则，不依赖外部 API。

    参数:
        user_input: 用户的原始输入文本

    返回:
        (intent_category, confidence, explanation)
        例如: ("writing", 0.85, "检测到关键词: 写, 邮件")
    """
    if not user_input or not user_input.strip():
        return ("general_qa", 0.5, "输入为空，默认分类为通用问答")

    text = user_input.lower().strip()

    # 加载意图示例数据（关键词和示例）
    examples_path = Path(__file__).parent.parent / "data" / "intent_examples.json"
    with open(examples_path, "r", encoding="utf-8") as f:
        intent_data = json.load(f)

    # 计算每个意图类别的匹配得分
    scores = {}
    matched_keywords = {}

    for intent, data in intent_data.items():
        score = 0.0
        matched = []

        # 关键词匹配（每个关键词 +0.15）
        for keyword in data["keywords"]:
            if keyword.lower() in text:
                score += 0.15
                matched.append(keyword)

        # 示例模糊匹配（高度相似的输入 +0.3）
        for example in data["examples"]:
            # 检查示例中的关键部分是否在用户输入中
            example_words = set(example.lower().split())
            input_words = set(text.split())
            overlap = len(example_words & input_words)
            if overlap >= len(example_words) * 0.6:  # 60% 以上词重叠
                score += 0.3
                matched.append(f"similar_to: {example[:20]}...")

        scores[intent] = min(score, 1.0)  # 最高 1.0
        matched_keywords[intent] = matched

    # 找出得分最高的类别
    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    # 如果没有任何匹配，使用规则兜底
    if best_score < 0.1:
        # 问句倾向 → knowledge
        if any(q in text for q in ["?", "？", "什么", "怎么", "为什么", "如何", "吗"]):
            return ("knowledge", 0.4, "检测到问句特征，归类为知识问答")
        # 默认 → general_qa
        return ("general_qa", 0.3, "未匹配到特定意图，使用通用问答")

    # 构建解释
    matched = matched_keywords[best_intent][:3]  # 最多展示 3 个
    explanation = f"检测到关键词: {', '.join(matched)}" if matched else "基于输入模式匹配"

    # 检查是否有多个类别得分接近（表示意图模糊，需要确认）
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) >= 2 and sorted_scores[0][1] - sorted_scores[1][1] < 0.15:
        # 得分差距小于 0.15，说明意图不够明确
        return (best_intent, best_score, f"{explanation}（意图不够明确，同时匹配了「{sorted_scores[1][0]}」）")

    return (best_intent, best_score, explanation)


def get_all_intents() -> dict:
    """返回所有意图类别及其描述（供前端展示用）"""
    examples_path = Path(__file__).parent.parent / "data" / "intent_examples.json"
    with open(examples_path, "r", encoding="utf-8") as f:
        intent_data = json.load(f)

    return {
        intent: {
            "label": data["label"],
            "description": data["description"],
            "examples": data["examples"][:3],  # 只展示前3个示例
        }
        for intent, data in intent_data.items()
    }


# 测试代码（直接运行此文件时执行）
if __name__ == "__main__":
    test_inputs = [
        "帮我写一封辞职信",
        "分析一下这份数据",
        "用 Python 写一个爬虫",
        "把这段话翻译成英文",
        "给我一些团建活动的创意",
        "什么是量子计算",
        "今天天气怎么样",
    ]
    print("=" * 50)
    print("IntentBridge 意图识别测试")
    print("=" * 50)
    for inp in test_inputs:
        intent, confidence, explanation = classify_intent(inp)
        print(f"\n输入: {inp}")
        print(f"  意图: {intent} (置信度: {confidence:.2f})")
        print(f"  原因: {explanation}")
