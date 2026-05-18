"""
IntentBridge 配置管理模块
从 .env 文件读取 API Key 等敏感配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（从项目根目录）
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # 如果 .env 不存在，尝试加载 .env.example 作为提示
    example_path = Path(__file__).parent.parent / ".env.example"
    if example_path.exists():
        print("⚠️  未找到 .env 文件，请复制 .env.example 为 .env 并填写 API Key")
        print(f"   执行: cp {example_path} {env_path}")

# ============================================================
# 配置项说明（产品经理视角）
# ============================================================
# DEFAULT_MODEL: 默认调用的模型
#   选择依据：平衡成本、速度、质量
#   - gpt-4o-mini: 性价比最高，适合大多数场景（$0.015/1M input）
#   - gpt-4o: 质量最好，适合复杂创作（$0.15/1M input）
#   - claude-sonnet-4-6: 适合写作和分析类任务
#
# ENABLED_MODELS: 模型路由时可用的模型列表
#   IntentBridge 会根据任务类型从这些模型中选择最合适的
# ============================================================

class Config:
    """应用配置，从环境变量读取"""

    # API Keys（支持两种写法：DEEPSEEK_API_KEY 或 deepseekkey）
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseekkey", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("openaikey", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("anthropickey", "")

    # DeepSeek API 地址
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    # 模型配置
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL") or os.getenv("defaultmodel", "deepseek-chat")

    # 可用模型列表（模型路由时会从这些中选择）
    ENABLED_MODELS = {
        "deepseek-chat": {"provider": "deepseek", "cost": "low", "speed": "fast", "quality": "good"},
        "gpt-4o-mini": {"provider": "openai", "cost": "low", "speed": "fast", "quality": "good"},
        "gpt-4o": {"provider": "openai", "cost": "high", "speed": "medium", "quality": "best"},
        "claude-sonnet-4-20250514": {"provider": "anthropic", "cost": "high", "speed": "medium", "quality": "best"},
    }

    @classmethod
    def is_configured(cls):
        """检查是否配置了至少一个 API Key"""
        return bool(cls.DEEPSEEK_API_KEY) or bool(cls.OPENAI_API_KEY) or bool(cls.ANTHROPIC_API_KEY)

    @classmethod
    def get_available_models(cls):
        """返回当前可用的模型列表"""
        available = []
        if cls.DEEPSEEK_API_KEY:
            available.append("deepseek-chat")
        if cls.OPENAI_API_KEY:
            available.extend(["gpt-4o-mini", "gpt-4o"])
        if cls.ANTHROPIC_API_KEY:
            available.append("claude-sonnet-4-20250514")
        return available if available else [cls.DEFAULT_MODEL]
