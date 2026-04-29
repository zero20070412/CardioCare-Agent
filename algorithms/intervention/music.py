"""
音乐疗法干预模块 - 根据情感和压力等级生成个性化音乐推荐
"""


def recommend_music(emotion="平静", stress_level="low"):
    """
    根据情感状态和压力等级推荐音乐疗法

    Args:
        emotion: 当前检测到的情感
        stress_level: 压力等级 (low/moderate/high/unknown)

    Returns:
        dict: 音乐推荐方案
    """
    if stress_level == "high":
        return {
            "playlist_name": "深度放松 - 频率共振舒缓",
            "style": "Binaural Beats / 432Hz 纯音 / 白噪音",
            "tempo": "40-60 BPM（极慢节奏）",
            "reason": (
                f"检测到您当前情绪为【{emotion}】，压力水平较高。"
                "推荐低频共振音乐（432Hz 基准频率），这类音乐的声波频率"
                "与人体自然振动频率接近，能有效促进副交感神经活动，"
                "降低皮质醇水平，快速缓解紧张和焦虑。"
            ),
            "recommendations": [
                "Binaural Beats - Delta Wave (2-4Hz)",
                "432Hz 纯音冥想音乐",
                "自然白噪音（雨声/海浪/森林）",
                "日本筝/古琴慢板曲目",
            ],
            "listening_guide": (
                "建议佩戴耳机，在安静环境中闭眼聆听 15-20 分钟。"
                "将注意力放在呼吸和音乐上，让身体自然放松。"
            ),
            "duration": "15-20 分钟",
        }
    elif stress_level == "moderate":
        return {
            "playlist_name": "舒缓心房 - 轻音乐放松",
            "style": "Ambient / Light Classical / Lo-Fi",
            "tempo": "60-80 BPM（慢节奏）",
            "reason": (
                f"检测到您当前情绪为【{emotion}】，存在一定压力感。"
                "推荐舒缓的轻音乐和环境音乐，这类音乐能有效降低心率变异性中的"
                "压力指标，帮助身心恢复平衡状态。"
            ),
            "recommendations": [
                "Ambient 电子轻音乐",
                "Lo-Fi Chillhop",
                "轻柔的钢琴/吉他独奏",
                "森林鸟鸣 + 轻柔钢琴",
            ],
            "listening_guide": (
                "在日常活动中作为背景音乐聆听，配合深呼吸效果更佳。"
            ),
            "duration": "20-30 分钟",
        }
    else:
        return {
            "playlist_name": "愉悦节拍 - 积极音乐",
            "style": "Light Pop / Acoustic / Nature Sounds",
            "tempo": "80-100 BPM（中等节奏）",
            "reason": (
                f"您当前情绪状态良好（{emotion}）。"
                "推荐播放节奏适中、旋律积极的音乐，有助于维持正面情绪，"
                "同时增强心血管系统的弹性。"
            ),
            "recommendations": [
                "轻快原声吉他",
                "古典乐 - 莫扎特/维瓦尔第",
                "自然声音（溪流/鸟鸣）",
                "Acoustic Pop",
            ],
            "listening_guide": "可在日常工作、学习中作为背景音乐。保持愉悦心情。",
            "duration": "30 分钟以上",
        }
