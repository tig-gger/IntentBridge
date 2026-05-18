"""
IntentBridge 第五步：模型路由 (Model Router)

产品经理笔记：
  这是产品的"成本优化引擎"。不同任务适合不同的模型——

  路由策略：
  - writing → Claude (写作能力更强)
  - coding → Claude (代码能力更强)
  - analysis → GPT-4o (数据分析更结构化)
  - translation → GPT-4o-mini (简单任务用便宜模型)
  - brainstorming → GPT-4o (创意需要更大模型)
  - knowledge → GPT-4o-mini (性价比最高)

  商业价值：
  通过智能路由，可以在保证质量的前提下降低 60-80% 的API成本。
  这是企业客户最愿意付费的功能之一。

  面试时可以讨论：
  - 路由策略怎么确定？→ 基于模型评测 + A/B测试
  - 怎么处理模型不可用？→ 降级策略
  - 成本 vs 质量的平衡怎么找到最优解？
"""

from utils.config import Config


# ============================================================
# 意图到模型的推荐映射
# key: 意图类别
# value: {
#   recommended: 推荐模型,
#   alternatives: 备选模型,
#   reason: 选择理由
# }
# ============================================================
ROUTING_TABLE = {
    "writing": {
        "recommended": "deepseek-chat",
        "alternatives": ["claude-sonnet-4-20250514", "gpt-4o", "gpt-4o-mini"],
        "reason": "DeepSeek 在中文创作上表现优异，性价比最高",
    },
    "coding": {
        "recommended": "deepseek-chat",
        "alternatives": ["claude-sonnet-4-20250514", "gpt-4o", "gpt-4o-mini"],
        "reason": "DeepSeek 代码能力强劲，价格仅为 GPT 的 1/10",
    },
    "analysis": {
        "recommended": "deepseek-chat",
        "alternatives": ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514"],
        "reason": "DeepSeek 长上下文能力适合分析类任务",
    },
    "translation": {
        "recommended": "deepseek-chat",
        "alternatives": ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"],
        "reason": "翻译任务用 DeepSeek 即可满足要求，成本最低",
    },
    "brainstorming": {
        "recommended": "gpt-4o",
        "alternatives": ["deepseek-chat", "claude-sonnet-4-20250514", "gpt-4o-mini"],
        "reason": "创意类任务用 GPT-4o 质量更稳定",
    },
    "knowledge": {
        "recommended": "deepseek-chat",
        "alternatives": ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"],
        "reason": "知识问答用 DeepSeek 性价比最高",
    },
    "general_qa": {
        "recommended": "deepseek-chat",
        "alternatives": ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"],
        "reason": "通用问答使用默认的轻量模型，DeepSeek 成本最低",
    },
}


def route_model(intent: str, available_models: list[str] = None) -> dict:
    """
    根据意图和可用模型，决策使用哪个模型。

    参数:
        intent: 意图类别
        available_models: 当前可用的模型列表

    返回:
        {
            "selected": 最终选定的模型,
            "reason": 选择理由,
            "alternatives": 备选方案,
            "cost_tier": "low" | "medium" | "high",
        }
    """
    if available_models is None:
        available_models = Config.get_available_models()

    # 获取该意图的路由配置
    route_config = ROUTING_TABLE.get(intent, ROUTING_TABLE["general_qa"])

    # 优先使用推荐模型
    selected = route_config["recommended"]
    reason = route_config["reason"]

    # 如果推荐模型不可用，选择第一个可用的备选
    if selected not in available_models:
        for alt in route_config["alternatives"]:
            if alt in available_models:
                selected = alt
                reason = f"推荐模型不可用，降级使用 {alt}"
                break
        else:
            # 所有推荐模型都不可用，使用第一个可用模型
            selected = available_models[0] if available_models else Config.DEFAULT_MODEL
            reason = f"使用默认模型 {selected}"

    # 成本分级
    model_config = Config.ENABLED_MODELS.get(selected, {})
    cost_tier = model_config.get("cost", "medium")

    return {
        "selected": selected,
        "reason": reason,
        "alternatives": route_config["alternatives"],
        "cost_tier": cost_tier,
    }


# 测试代码
if __name__ == "__main__":
    available = ["gpt-4o-mini", "gpt-4o"]

    print("=" * 50)
    print("IntentBridge 模型路由测试")
    print("=" * 50)
    print(f"可用模型: {available}\n")

    for intent in ["writing", "coding", "analysis", "translation", "brainstorming", "knowledge"]:
        result = route_model(intent, available)
        print(f"{intent:15s} → {result['selected']:25s} | 成本: {result['cost_tier']}")
        print(f"{'':15s}   理由: {result['reason']}")
        print()
