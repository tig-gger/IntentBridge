"""
IntentBridge 第三步：用户画像管理 (Profile Manager)

产品经理笔记：
  这是产品"个性化"能力的核心。好的AI产品应该越用越懂用户。
  这个模块目前是基础版本——在session内记录用户偏好。

  未来迭代方向：
  V1 (当前): Session内记忆，重启丢失
  V2: SQLite持久化存储（跨session记忆）
  V3: 用户画像学习——根据用户历史行为自动调整回答风格
  V4: 团队画像——企业版中为整个团队建立统一的使用规范

  面试时可以讨论：
  用户画像的隐私边界在哪里？
  如何在个性化和隐私之间做平衡？（GDPR合规）
  冷启动问题：新用户没有历史数据怎么办？
"""

import json
from datetime import datetime
from typing import Optional


class UserProfile:
    """
    用户画像类，记录用户的使用偏好和历史。

    每条画像记录包含：
    - preferred_tone: 偏好的语气
    - preferred_model: 偏好的模型
    - history: 最近的交互历史（最多20条）
    - usage_count: 各类意图的使用频次
    """

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.preferred_tone: Optional[str] = None
        self.preferred_model: Optional[str] = None
        self.history: list = []
        self.usage_count: dict = {}
        self.max_history = 20

    def record_interaction(self, intent: str, model: str, user_input: str, feedback: int = 0):
        """
        记录一次交互。

        参数:
            intent: 意图类别
            model: 使用的模型
            user_input: 用户输入
            feedback: 用户反馈（0=未评价, 1=好评, -1=差评）
        """
        # 记录意图使用频次
        self.usage_count[intent] = self.usage_count.get(intent, 0) + 1

        # 添加到历史
        self.history.append({
            "time": datetime.now().isoformat(),
            "intent": intent,
            "model": model,
            "input": user_input[:100],  # 只保存前100个字符
            "feedback": feedback,
        })

        # 限制历史长度
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_most_used_intent(self) -> Optional[str]:
        """返回使用最多的意图类别"""
        if not self.usage_count:
            return None
        return max(self.usage_count, key=self.usage_count.get)

    def get_profile_summary(self) -> dict:
        """返回用户画像摘要（供前端展示和Prompt构建用）"""
        summary = {
            "user_id": self.user_id,
            "preferred_tone": self.preferred_tone,
            "preferred_model": self.preferred_model,
            "most_used_intent": self.get_most_used_intent(),
            "total_interactions": len(self.history),
            "intent_distribution": dict(self.usage_count),
        }
        return summary

    def to_prompt_context(self) -> str:
        """
        将用户画像转为Prompt中的上下文信息。
        这部分会被注入到最终发给AI的prompt中。
        """
        context_parts = []

        if self.preferred_tone:
            context_parts.append(f"用户偏好的回答语气：{self.preferred_tone}")

        most_used = self.get_most_used_intent()
        if most_used:
            context_parts.append(f"用户常问的类型：{most_used}")

        if context_parts:
            return "用户背景信息：" + "；".join(context_parts)
        return ""


# 全局用户画像实例（session级别）
_current_profile = UserProfile()


def get_user_profile(user_id: str = "default_user") -> UserProfile:
    """获取（或创建）用户画像实例"""
    global _current_profile
    if _current_profile.user_id != user_id:
        _current_profile = UserProfile(user_id)
    return _current_profile


# 测试代码
if __name__ == "__main__":
    profile = UserProfile("test_user")

    # 模拟几次交互
    profile.record_interaction("writing", "gpt-4o-mini", "帮我写一封邮件")
    profile.record_interaction("coding", "gpt-4o-mini", "写一个Python函数")
    profile.record_interaction("writing", "gpt-4o-mini", "写一篇博客")
    profile.preferred_tone = "半正式"

    print("=" * 50)
    print("IntentBridge 用户画像测试")
    print("=" * 50)
    print(f"\n用户画像摘要：{json.dumps(profile.get_profile_summary(), ensure_ascii=False, indent=2)}")
    print(f"\nPrompt上下文注入：{profile.to_prompt_context()}")
