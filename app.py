"""
IntentBridge —— AI 意图路由与交互优化层

让每个人都能用自然语言用好 AI。

运行方式（在终端中执行）：
    streamlit run app.py

如果你还没安装依赖：
    pip install -r requirements.txt
"""

import streamlit as st
from pipeline.orchestrator import run_pipeline
from pipeline.intent_classifier import get_all_intents
from utils.config import Config
def _display_pipeline_log(pipeline_log):
    """显示 Pipeline 处理日志"""
    with st.expander("🔍 查看 IntentBridge 处理过程", expanded=False):
        # 每步用列显示
        steps = list(pipeline_log.keys())

        # 总览
        total = pipeline_log.get("total", {})
        st.caption(f"⏱️ 总耗时: {total.get('总耗时', 'N/A')} | 状态: {total.get('pipeline状态', 'N/A')}")

        # 各步骤详情
        for step_name, step_data in pipeline_log.items():
            if step_name == "total":
                continue

            step_label = {
                "intent_classification": "① 意图识别",
                "param_extraction": "② 参数提取",
                "profile": "③ 用户画像",
                "prompt_building": "④ Prompt 优化",
                "model_routing": "⑤ 模型路由",
                "llm_call": "⑥ LLM 调用",
                "output_refinement": "⑦ 输出优化",
            }.get(step_name, step_name)

            with st.container():
                cols = st.columns([1, 3])
                with cols[0]:
                    st.markdown(f"**{step_label}**")
                    st.caption(step_data.get("耗时", ""))
                with cols[1]:
                    for key, val in step_data.items():
                        if key != "耗时":
                            if isinstance(val, dict):
                                for k, v in val.items():
                                    st.markdown(f"`{k}`: {v}")
                            else:
                                st.markdown(f"`{key}`: {val}")
            st.divider()


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="IntentBridge",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Session 状态初始化
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_pipeline" not in st.session_state:
    st.session_state.show_pipeline = False
if "feedback_stats" not in st.session_state:
    st.session_state.feedback_stats = {"positive": 0, "negative": 0, "total": 0}


# ============================================================
# 侧边栏：功能介绍 + 设置
# ============================================================
with st.sidebar:
    st.title("🌉 IntentBridge")
    st.caption("让每个人都能用自然语言用好 AI")

    st.divider()

    # 反馈统计
    if st.session_state.feedback_stats["total"] > 0:
        stats = st.session_state.feedback_stats
        pos_pct = int(stats["positive"] / stats["total"] * 100)
        st.subheader("📊 反馈统计")
        st.caption(f"👍 有用 {stats['positive']}  /  👎 没用 {stats['negative']}")
        st.progress(pos_pct / 100, text=f"满意度 {pos_pct}%")
        st.divider()

    # API 状态
    st.subheader("🔌 API 状态")
    api_ok = Config.is_configured()
    if api_ok:
        st.success("✅ API 已配置")
        if Config.OPENAI_API_KEY:
            st.caption(f"  OpenAI: 已就绪")
        if Config.ANTHROPIC_API_KEY:
            st.caption(f"  Anthropic: 已就绪")
    else:
        st.warning("⚠️ 未配置 API Key")
        st.caption("当前为演示模式，配置后可以使用真实 AI 回复")
        st.caption("请复制 .env.example 为 .env 并填入密钥")

    st.divider()

    # Pipeline 透明度开关
    st.subheader("🔍 透明度模式")
    st.session_state.show_pipeline = st.toggle(
        "显示 Pipeline 处理过程",
        value=st.session_state.show_pipeline,
        help="开启后，你会看到 IntentBridge 是如何一步步处理你的输入的",
    )

    st.divider()

    # 功能介绍
    st.subheader("🎯 能做什么")
    intents = get_all_intents()
    for intent, data in intents.items():
        with st.expander(f"📌 {data['label']}"):
            st.write(data["description"])
            st.caption("例如：" + "、".join(data["examples"]))

    st.divider()

    # 清空对话
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("IntentBridge v1.0 | 作品集项目")


# ============================================================
# 主界面：对话区
# ============================================================
st.title("💬 对 AI 说人话")
st.markdown(
    "直接输入你的需求，IntentBridge 会自动理解你的意图、优化指令、路由到最适合的 AI 模型。"
)

# 显示历史消息
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果开启透明度模式，显示 pipeline 日志
        if message["role"] == "assistant" and "pipeline_log" in message:
            if st.session_state.show_pipeline:
                _display_pipeline_log(message["pipeline_log"])
        # 反馈按钮（只对AI回复显示）
        if message["role"] == "assistant":
            fb = message.get("feedback")
            if fb is None:
                col1, col2, _ = st.columns([1, 1, 10])
                with col1:
                    if st.button("👍", key=f"up_{msg_idx}", help="这个回答有用"):
                        st.session_state.messages[msg_idx]["feedback"] = "positive"
                        st.session_state.feedback_stats["positive"] += 1
                        st.session_state.feedback_stats["total"] += 1
                        st.rerun()
                with col2:
                    if st.button("👎", key=f"down_{msg_idx}", help="这个回答没用"):
                        st.session_state.messages[msg_idx]["feedback"] = "negative"
                        st.session_state.feedback_stats["negative"] += 1
                        st.session_state.feedback_stats["total"] += 1
                        st.rerun()
            else:
                icon = "👍" if fb == "positive" else "👎"
                st.caption(f"✅ 已反馈 {icon}")

# 输入框
if prompt := st.chat_input("说你的需求，比如「帮我写一封给客户的邮件」..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 运行 Pipeline
    with st.chat_message("assistant"):
        with st.status("🤔 IntentBridge 正在处理...", expanded=True) as status:
            st.write("📡 正在分析你的意图...")

            # 调用 pipeline
            result = run_pipeline(prompt, verbose=True)

            # 更新状态
            if result["metadata"]["provider"] == "demo":
                status.update(
                    label="⚠️ 演示模式（未配置 API Key）",
                    state="complete",
                )
            else:
                status.update(
                    label=f"✅ 完成！意图: {result['metadata']['intent']} | "
                           f"模型: {result['metadata']['model']} | "
                           f"耗时: {result['metadata']['latency']:.1f}s",
                    state="complete",
                )

        # 显示最终输出
        st.markdown(result["final_output"])

        # 置信度标注
        conf_colors = {
            "high": "🟢",
            "medium": "🟡",
            "low": "🔴",
        }
        conf_label = {
            "high": "高置信度",
            "medium": "中等置信度（建议核实）",
            "low": "低置信度（请谨慎参考）",
        }
        st.caption(
            f"{conf_colors.get(result['confidence'], '⚪')} "
            f"置信度: {conf_label.get(result['confidence'], '未评估')} | "
            f"模型: {result['metadata']['model']}"
        )

        # 操作建议
        if result.get("suggestions"):
            with st.expander("💡 接下来可以..."):
                for s in result["suggestions"]:
                    st.write(f"- {s}")

        # 透明度模式：显示 Pipeline 详情
        if st.session_state.show_pipeline and "pipeline_log" in result:
            _display_pipeline_log(result["pipeline_log"])

            with st.expander("📝 查看完整 Prompt"):
                st.text_area("System Prompt", result["prompt_detail"]["system_prompt"], height=200)
                st.text_area("User Prompt", result["prompt_detail"]["user_prompt"], height=150)

        # 保存消息
        msg_entry = {
            "role": "assistant",
            "content": result["final_output"],
            "feedback": None,
        }
        if st.session_state.show_pipeline and "pipeline_log" in result:
            msg_entry["pipeline_log"] = result["pipeline_log"]

        st.session_state.messages.append(msg_entry)


# ============================================================
# 辅助函数
# ============================================================
