"""
呼吸训练干预模块 - 根据情感和压力等级生成个性化呼吸引导
"""


def generate_breathing_guidance(stress_level="low", emotion="平静"):
    """
    根据压力等级和情感状态生成呼吸训练处方

    Args:
        stress_level: 压力等级 (low/moderate/high/unknown)
        emotion: 当前检测到的情感

    Returns:
        dict: 呼吸训练方案
    """
    # 根据压力等级选择呼吸方案
    if stress_level == "high":
        return {
            "type": "box_breathing",
            "name": "箱式呼吸法（高强度放松）",
            "steps": [
                "缓慢吸气 4 秒",
                "屏住呼吸 4 秒",
                "缓慢呼气 4 秒",
                "屏住呼吸 4 秒",
            ],
            "duration": "8 分钟",
            "cycles": "每分钟 3-4 个循环",
            "instruction": (
                f"检测到您当前情绪为【{emotion}】，压力水平较高。"
                "建议您现在跟随以下节奏进行箱式呼吸：\n"
                "1. 用鼻子缓慢深吸气 4 秒，感受腹部隆起\n"
                "2. 屏住呼吸 4 秒，保持放松\n"
                "3. 用嘴缓慢呼气 4 秒，感受身体放松\n"
                "4. 屏住呼吸 4 秒\n"
                "重复以上步骤 8 分钟。箱式呼吸能有效激活副交感神经，"
                "帮助降低心率和血压，快速缓解焦虑和紧张。"
            ),
            "effect": "快速降低心率、缓解焦虑、稳定血压",
        }
    elif stress_level == "moderate":
        return {
            "type": "extended_breathing",
            "name": "4-7-8 呼吸法（中等放松）",
            "steps": [
                "缓慢吸气 4 秒",
                "屏住呼吸 7 秒",
                "缓慢呼气 8 秒",
            ],
            "duration": "5 分钟",
            "cycles": "每分钟约 3 个循环",
            "instruction": (
                f"检测到您当前情绪为【{emotion}】，有一定压力感。"
                "推荐您尝试 4-7-8 呼吸法来舒缓身心：\n"
                "1. 用鼻子缓慢深吸气 4 秒\n"
                "2. 屏住呼吸 7 秒，让氧气充分进入血液\n"
                "3. 用嘴缓缓呼气 8 秒，感受身体逐步放松\n"
                "重复以上步骤 5 分钟。这种呼吸方式能帮助调节自主神经系统，"
                "促进深度放松。"
            ),
            "effect": "调节自主神经、促进放松、改善睡眠质量",
        }
    else:
        return {
            "type": "box_breathing",
            "name": "日常呼吸调节",
            "steps": [
                "吸气 4 秒",
                "屏息 4 秒",
                "呼气 4 秒",
                "屏息 4 秒",
            ],
            "duration": "5 分钟",
            "cycles": "每分钟 3-4 个循环",
            "instruction": (
                f"您当前情绪状态良好（{emotion}）。"
                "建议每天花 5 分钟进行呼吸训练，有助于维持心血管健康、"
                "增强抗压能力。可随时使用箱式呼吸法进行日常放松。"
            ),
            "effect": "维持心血管健康、增强抗压能力",
        }
