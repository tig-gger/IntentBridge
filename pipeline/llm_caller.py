"""
IntentBridge 第六步：LLM 调用 (LLM Caller)

产品经理笔记：
  这个模块是"技术封装层"——对外部模型的调用做了统一封装。
  这样上层业务逻辑不需要关心具体调的是哪个模型、哪个API。
  是典型的"适配器模式"。

  为什么这个设计对产品重要？
  - 模型切换对用户透明：今天用GPT-4o，明天换Claude，用户无感
  - 可以自由做A/B测试：同一个请求发给两个模型，比较效果
  - 容错：一个模型挂了自动切换到另一个
"""
import requests
import httpx
import time
from openai import OpenAI
from utils.config import Config


# ============================================================
# 模型调用配置
# 产品经理笔记：这些参数直接影响用户体验和成本
# - temperature: 控制创造性（0=保守，1=创意）
# - max_tokens: 控制回复长度（影响响应时间和成本）
# - timeout: 超时控制（影响用户体验）
# ============================================================
MODEL_PARAMS = {
    "deepseek-chat": {
        "temperature": 0.7,
        "max_tokens": 4096,
        "timeout": 45,
    },
    "gpt-4o-mini": {
        "temperature": 0.7,
        "max_tokens": 2048,
        "timeout": 30,
    },
    "gpt-4o": {
        "temperature": 0.7,
        "max_tokens": 4096,
        "timeout": 45,
    },
    "claude-sonnet-4-20250514": {
        "temperature": 0.7,
        "max_tokens": 4096,
        "timeout": 45,
    },
}


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    stream: bool = False,
) -> dict:
    """
    调用 LLM API 生成回复。

    参数:
        system_prompt: 系统指令
        user_prompt: 用户输入
        model: 模型名称（默认使用配置中的默认模型）
        stream: 是否使用流式输出

    返回:
        {
            "content": 模型回复内容,
            "model": 实际使用的模型,
            "latency": 响应时间（秒）,
            "tokens_in": 输入token数,
            "tokens_out": 输出token数,
            "provider": API提供商,
        }
    """
    if model is None:
        model = Config.DEFAULT_MODEL

    # 检测模型提供商
    if model.startswith("deepseek"):
        return _call_deepseek(system_prompt, user_prompt, model, stream)
    elif model.startswith("gpt"):
        return _call_openai(system_prompt, user_prompt, model, stream)
    elif model.startswith("claude"):
        return _call_anthropic(system_prompt, user_prompt, model, stream)
    else:
        # 默认为 OpenAI
        return _call_openai(system_prompt, user_prompt, model, stream)


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    stream: bool = False,
) -> dict:
    """调用 OpenAI API"""
    if not Config.OPENAI_API_KEY:
        return _demo_response(system_prompt, user_prompt, model,
                              "⚠️ 未配置 OPENAI_API_KEY")

    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    params = MODEL_PARAMS.get(model, MODEL_PARAMS["gpt-4o-mini"])

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            timeout=params["timeout"],
            stream=False,
        )

        latency = time.time() - start_time

        return {
            "content": response.choices[0].message.content or "",
            "model": model,
            "latency": round(latency, 2),
            "tokens_in": response.usage.prompt_tokens if response.usage else 0,
            "tokens_out": response.usage.completion_tokens if response.usage else 0,
            "provider": "openai",
            "error": None,
        }

    except Exception as e:
        latency = time.time() - start_time
        return {
            "content": f"调用 API 时出错：{str(e)}",
            "model": model,
            "latency": round(latency, 2),
            "tokens_in": 0,
            "tokens_out": 0,
            "provider": "openai",
            "error": str(e),
        }


def _call_deepseek(
    system_prompt: str,
    user_prompt: str,
    model: str,
    stream: bool = False,
) -> dict:
    """
    使用 requests 直接调用 DeepSeek API（避免 openai 库的代理问题）
    """
    if not Config.DEEPSEEK_API_KEY:
        return _demo_response(system_prompt, user_prompt, model, "未配置 DEEPSEEK_API_KEY")

    import requests
    import json
    import time

    # DeepSeek API 地址
    base_url = Config.DEEPSEEK_BASE_URL.rstrip('/')
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    params = MODEL_PARAMS.get(model, MODEL_PARAMS.get("deepseek-chat", {}))
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": params.get("temperature", 0.7),
        "max_tokens": params.get("max_tokens", 4096),
        "stream": stream
    }
    start_time = time.time()
    try:
        # 创建 session 并禁用环境代理
        session = requests.Session()
        session.trust_env = False   # 不读取 HTTP_PROXY 环境变量
        response = session.post(url, headers=headers, json=data, timeout=params.get("timeout", 60))
        latency = time.time() - start_time
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"] or ""
            return {
                "content": content,
                "model": model,
                "latency": round(latency, 2),
                "tokens_in": result.get("usage", {}).get("prompt_tokens", 0),
                "tokens_out": result.get("usage", {}).get("completion_tokens", 0),
                "provider": "deepseek",
                "error": None,
            }
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            return {
                "content": f"DeepSeek API 错误: {error_msg}",
                "model": model,
                "latency": round(latency, 2),
                "tokens_in": 0,
                "tokens_out": 0,
                "provider": "deepseek",
                "error": error_msg,
            }
    except Exception as e:
        latency = time.time() - start_time
        return {
            "content": f"调用 DeepSeek API 时出错: {str(e)}",
            "model": model,
            "latency": round(latency, 2),
            "tokens_in": 0,
            "tokens_out": 0,
            "provider": "deepseek",
            "error": str(e),
        }
def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str,
    stream: bool = False,
) -> dict:
    """调用 Anthropic (Claude) API"""
    if not Config.ANTHROPIC_API_KEY:
        return _demo_response(system_prompt, user_prompt, model,
                              "⚠️ 未配置 ANTHROPIC_API_KEY")

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        params = MODEL_PARAMS.get(model, MODEL_PARAMS["claude-sonnet-4-20250514"])

        start_time = time.time()

        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        )

        latency = time.time() - start_time

        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        return {
            "content": content,
            "model": model,
            "latency": round(latency, 2),
            "tokens_in": response.usage.input_tokens if response.usage else 0,
            "tokens_out": response.usage.output_tokens if response.usage else 0,
            "provider": "anthropic",
            "error": None,
        }

    except Exception as e:
        return {
            "content": f"调用 Anthropic API 时出错：{str(e)}",
            "model": model,
            "latency": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "provider": "anthropic",
            "error": str(e),
        }


def _demo_response(system_prompt: str, user_prompt: str, model: str, warning: str) -> dict:
    """
    当没有配置 API Key 时，返回模拟响应。
    这样你即使没有付费API也能看到整个pipeline的完整流程。
    """
    return {
        "content": (
            f"[演示模式 - IntentBridge Pipeline 完整运行]\n\n"
            f"你的输入经过以下处理：\n"
            f"1. ✅ 意图识别 → 已分类\n"
            f"2. ✅ 参数提取 → 已提取结构化参数\n"
            f"3. ✅ 用户画像 → 已加载\n"
            f"4. ✅ Prompt优化 → 已构建增强指令\n"
            f"5. ✅ 模型路由 → 已选择最优模型\n"
            f"6. ⏳ LLM调用 → {warning}\n\n"
            f"---\n"
            f"配置好 API Key 后，这里就会显示真实的 AI 回复。\n"
            f"请复制 .env.example 为 .env 并填入你的密钥。"
        ),
        "model": model,
        "latency": 0.01,
        "tokens_in": 0,
        "tokens_out": 0,
        "provider": "demo",
        "error": warning,
    }


# 测试代码
if __name__ == "__main__":
    print("=" * 50)
    print("IntentBridge LLM 调用测试")
    print("=" * 50)
    print("\n这个测试需要 API Key，当前模式：")
    print(f"  DeepSeek可用: {'✓' if Config.DEEPSEEK_API_KEY else '✗'}")
    print(f"  OpenAI可用: {'✓' if Config.OPENAI_API_KEY else '✗'}")
    print(f"  Anthropic可用: {'✓' if Config.ANTHROPIC_API_KEY else '✗'}")
    print("\n如无API Key，会自动进入演示模式")
