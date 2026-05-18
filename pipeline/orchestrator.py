"""
IntentBridge Pipeline 编排层

这是整个pipeline的"总导演"——把7个步骤按顺序串联起来。

产品经理笔记：
  设计这个编排层的原因：
  1. 任何一个步骤都可以独立升级/替换（比如今天用关键词分类，
     明天换成ML模型分类，其他步骤不受影响）
  2. 可以方便地加入监控和日志（每个步骤的耗时、成功率）
  3. 方便做A/B测试（比如两个不同版本的prompt_builder对比）

  7个步骤 → 用户看到的只有1个结果，这就是"技术封装"的产品价值。
"""

import time
from pipeline.intent_classifier import classify_intent
from pipeline.param_extractor import extract_parameters
from pipeline.profile_manager import get_user_profile
from pipeline.prompt_builder import build_prompt
from pipeline.model_router import route_model
from pipeline.llm_caller import call_llm
from pipeline.output_refiner import refine_output
from utils.config import Config


def run_pipeline(user_input: str, user_id: str = "default_user", verbose: bool = True) -> dict:
    """
    运行完整的 IntentBridge Pipeline。

    参数:
        user_input: 用户的原始输入
        user_id: 用户ID（用于加载不同的用户画像）
        verbose: 是否返回完整的pipeline日志

    返回:
        {
            "final_output": 最终展示给用户的内容,
            "confidence": 置信度,
            "suggestions": 操作建议,
            "metadata": 元数据（模型、耗时等）,
            "pipeline_log": 各步骤的详细日志（verbose=True时）,
        }
    """
    pipeline_start = time.time()
    pipeline_log = {}

    # ============================================================
    # Step 1: 意图识别
    # ============================================================
    step_start = time.time()
    intent, confidence, explanation = classify_intent(user_input)
    pipeline_log["intent_classification"] = {
        "耗时": f"{(time.time() - step_start)*1000:.0f}ms",
        "识别意图": intent,
        "置信度": f"{confidence:.0%}",
        "判断依据": explanation,
    }

    # ============================================================
    # Step 2: 参数提取
    # ============================================================
    step_start = time.time()
    params = extract_parameters(user_input, intent)
    pipeline_log["param_extraction"] = {
        "耗时": f"{(time.time() - step_start)*1000:.0f}ms",
        "提取参数": params["extracted"],
        "缺失参数": params["missing"],
    }

    # ============================================================
    # Step 3: 用户画像
    # ============================================================
    step_start = time.time()
    profile = get_user_profile(user_id)
    pipeline_log["profile"] = {
        "耗时": f"{(time.time() - step_start)*1000:.0f}ms",
        "用户历史次数": profile.get_profile_summary()["total_interactions"],
        "常用意图": profile.get_profile_summary()["most_used_intent"],
    }

    # ============================================================
    # Step 4: Prompt 优化
    # ============================================================
    step_start = time.time()
    prompt_package = build_prompt(user_input, intent, params, profile)
    pipeline_log["prompt_building"] = {
        "耗时": f"{(time.time() - step_start)*1000:.0f}ms",
        "优化详情": prompt_package["optimization_log"],
    }

    # ============================================================
    # Step 5: 模型路由
    # ============================================================
    step_start = time.time()
    routing = route_model(intent)
    pipeline_log["model_routing"] = {
        "耗时": f"{(time.time() - step_start)*1000:.0f}ms",
        "选择模型": routing["selected"],
        "选择理由": routing["reason"],
        "成本等级": routing["cost_tier"],
    }

    # ============================================================
    # Step 6: LLM 调用
    # ============================================================
    step_start = time.time()
    llm_response = call_llm(
        system_prompt=prompt_package["system_prompt"],
        user_prompt=prompt_package["user_prompt"],
        model=routing["selected"],
    )
    pipeline_log["llm_call"] = {
        "耗时": f"{llm_response['latency']:.1f}s",
        "模型": llm_response["model"],
        "输入Token": llm_response["tokens_in"],
        "输出Token": llm_response["tokens_out"],
        "API提供商": llm_response["provider"],
    }

    # ============================================================
    # Step 7: 输出优化
    # ============================================================
    step_start = time.time()
    refined = refine_output(
        content=llm_response["content"],
        intent=intent,
        model=llm_response["model"],
        latency=llm_response["latency"],
        error=llm_response.get("error"),
    )
    pipeline_log["output_refinement"] = {
        "耗时": f"{(time.time() - step_start)*1000:.0f}ms",
        "置信度": refined["confidence"],
    }

    # 记录到用户画像
    profile.record_interaction(
        intent=intent,
        model=llm_response["model"],
        user_input=user_input,
    )

    # 总耗时
    total_time = time.time() - pipeline_start
    pipeline_log["total"] = {
        "总耗时": f"{total_time:.1f}s",
        "pipeline状态": "✅ 完整运行" if not llm_response.get("error") or llm_response["provider"] == "demo" else "⚠️ 部分完成",
    }

    result = {
        "final_output": refined["content"],
        "confidence": refined["confidence"],
        "suggestions": refined["suggestions"],
        "metadata": {
            "intent": intent,
            "model": llm_response["model"],
            "latency": llm_response["latency"],
            "provider": llm_response["provider"],
        },
    }

    if verbose:
        result["pipeline_log"] = pipeline_log
        result["prompt_detail"] = {
            "system_prompt": prompt_package["system_prompt"],
            "user_prompt": prompt_package["user_prompt"],
        }

    return result
