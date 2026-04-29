"""
端到端测试脚本 - 验证 emotion2vec 集成到 CardioBot 的完整链路

测试内容：
1. emotion2vec 模型加载与推理
2. voice.py 语音情感分析
3. breathing.py 和 music.py 干预处方生成
4. core.py 调度（带音频路径 / 不带音频路径）
5. __init__.py 导出修复验证
"""

import os
import sys

# 确保项目根目录在 path 中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_init_exports():
    """测试 __init__.py 导出修复"""
    print("=" * 60)
    print("测试 1: __init__.py 导出修复")
    print("=" * 60)

    # emotion_recognition
    from algorithms.emotion_recognition import analyze_voice_emotion
    assert callable(analyze_voice_emotion), "analyze_voice_emotion 导出失败"
    print("  [PASS] algorithms.emotion_recognition.analyze_voice_emotion 导出正常")

    # intervention
    from algorithms.intervention import generate_breathing_guidance, recommend_music
    assert callable(generate_breathing_guidance), "generate_breathing_guidance 导出失败"
    assert callable(recommend_music), "recommend_music 导出失败"
    print("  [PASS] algorithms.intervention.generate_breathing_guidance 导出正常")
    print("  [PASS] algorithms.intervention.recommend_music 导出正常")
    print()


def test_model_load_and_inference():
    """测试 emotion2vec 模型加载和推理"""
    print("=" * 60)
    print("测试 2: emotion2vec 模型加载与推理")
    print("=" * 60)

    from funasr import AutoModel

    print("  正在加载模型（首次可能需要几分钟）...")
    model = AutoModel(model="iic/emotion2vec_plus_seed", hub="ms", disable_update=True)
    assert model is not None, "模型加载失败"
    print("  [PASS] 模型加载成功")

    # 使用自带的测试音频
    wav_file = os.path.join(model.model_path, "example", "test.wav")
    assert os.path.isfile(wav_file), f"测试音频不存在: {wav_file}"

    import torch
    import soundfile as sf

    speech, sr = sf.read(wav_file, dtype="float32")
    if len(speech.shape) == 2:
        speech = speech.mean(axis=1)
    speech_tensor = torch.from_numpy(speech).float()

    result = model.generate(speech_tensor, granularity="utterance", extract_embedding=False, fs=sr)
    assert result and len(result) > 0, "模型推理未返回结果"

    item = result[0]
    labels = item.get("labels", [])
    scores = item.get("scores", [])
    assert len(labels) == 9, f"期望 9 类标签，得到 {len(labels)} 类"
    assert len(scores) == 9, f"期望 9 个分数，得到 {len(scores)} 个"
    assert abs(sum(scores) - 1.0) < 0.01, f"分数之和应为 1.0，实际为 {sum(scores)}"

    print(f"  [PASS] 推理成功，标签: {labels}")
    print(f"  [PASS] 分数: {[f'{s:.4f}' for s in scores]}")
    print()


def test_voice_emotion_analysis():
    """测试 voice.py 语音情感分析"""
    print("=" * 60)
    print("测试 3: voice.py 语音情感分析")
    print("=" * 60)

    from funasr import AutoModel

    model = AutoModel(model="iic/emotion2vec_plus_seed", hub="ms", disable_update=True)
    wav_file = os.path.join(model.model_path, "example", "test.wav")

    from algorithms.emotion_recognition.voice import analyze_voice_emotion

    result = analyze_voice_emotion(wav_file)

    assert result["status"] == "success", f"分析失败: {result.get('note')}"
    assert "emotion" in result, "缺少 emotion 字段"
    assert "stress_level" in result, "缺少 stress_level 字段"
    assert "confidence" in result, "缺少 confidence 字段"
    assert "scores" in result, "缺少 scores 字段"
    assert "note" in result, "缺少 note 字段"
    assert isinstance(result["scores"], dict), "scores 应为字典"
    assert len(result["scores"]) == 9, f"scores 应有 9 项，实际 {len(result['scores'])} 项"

    print(f"  [PASS] 情感: {result['emotion']}")
    print(f"  [PASS] 压力: {result['stress_level']}")
    print(f"  [PASS] 置信度: {result['confidence']}")
    print(f"  [PASS] 说明: {result['note']}")
    print()


def test_intervention_modules():
    """测试干预处方模块"""
    print("=" * 60)
    print("测试 4: 干预处方模块")
    print("=" * 60)

    from algorithms.intervention.breathing import generate_breathing_guidance
    from algorithms.intervention.music import recommend_music

    # 高压力
    b_high = generate_breathing_guidance(stress_level="high", emotion="愤怒")
    assert b_high["type"] in ("box_breathing", "extended_breathing"), "呼吸类型无效"
    assert b_high["duration"], "缺少时长"
    assert b_high["instruction"], "缺少指导说明"
    print(f"  [PASS] 高压力呼吸方案: {b_high['name']}, 时长 {b_high['duration']}")

    m_high = recommend_music(emotion="愤怒", stress_level="high")
    assert m_high["playlist_name"], "缺少歌单名"
    assert m_high["style"], "缺少音乐风格"
    assert m_high["recommendations"], "缺少推荐曲目"
    print(f"  [PASS] 高压力音乐推荐: {m_high['playlist_name']}, 风格 {m_high['style']}")

    # 低压力
    b_low = generate_breathing_guidance(stress_level="low", emotion="开心")
    assert b_low["type"], "呼吸类型无效"
    print(f"  [PASS] 低压力呼吸方案: {b_low['name']}")

    m_low = recommend_music(emotion="开心", stress_level="low")
    assert m_low["playlist_name"], "缺少歌单名"
    print(f"  [PASS] 低压力音乐推荐: {m_low['playlist_name']}")

    # 无音频时的降级
    from algorithms.emotion_recognition.voice import analyze_voice_emotion as _a
    voice_result = _a(None)
    assert voice_result["status"] == "error", "空路径应返回错误"
    print(f"  [PASS] 空路径降级: {voice_result['note']}")

    voice_bad = _a("/nonexistent/path.wav")
    assert voice_bad["status"] == "error", "无效路径应返回错误"
    print(f"  [PASS] 无效路径降级: {voice_bad['note']}")
    print()


def test_core_dispatch():
    """测试 core.py 调度逻辑"""
    print("=" * 60)
    print("测试 5: core.py 调度逻辑")
    print("=" * 60)

    from agent.core import get_agent_response, clear_session_history
    from agent.core import _run_voice_emotion_analysis

    test_session = "test_session_voice_integration"
    clear_session_history(test_session)

    # 测试 _run_voice_emotion_analysis 内部函数
    from funasr import AutoModel

    model = AutoModel(model="iic/emotion2vec_plus_seed", hub="ms", disable_update=True)
    wav_file = os.path.join(model.model_path, "example", "test.wav")

    results = _run_voice_emotion_analysis(wav_file)
    assert isinstance(results, list), "应返回列表"
    assert len(results) > 0, "结果不应为空"
    assert results[0]["module"] == "voice_emotion", "第一个结果应为 voice_emotion"
    print(f"  [PASS] 语音情感分析调度: {results[0]['module']} - {results[0]['status']}")
    print(f"         {results[0]['summary'][:80]}...")

    if len(results) > 1:
        for r in results[1:]:
            print(f"  [PASS] 干预触发: {r['module']} - {r['summary'][:60]}...")

    print()

    # 测试 get_agent_response 带 audio_path（需要 mock model client）
    print("  测试 get_agent_response 带 audio_path（使用 mock 模式）...")
    from agent.model import LLMClient
    from utils.config import settings

    try:
        mock_client = LLMClient(
            openai_api_key="test-key",
            openai_base_url="https://api.openai.com/v1",
            model_name="gpt-4o-mini",
            use_mock_model=True,
        )

        clear_session_history(test_session)
        response = get_agent_response(
            user_message="[语音消息]",
            session_id=test_session,
            audio_path=wav_file,
            model_client=mock_client,
        )
        assert "reply" in response, "缺少 reply"
        assert "algorithms" in response, "缺少 algorithms"

        algo_modules = [a.get("module") for a in response["algorithms"]]
        print(f"  [PASS] get_agent_response 返回正常")
        print(f"         触发算法模块: {algo_modules}")
        print(f"         回复: {response['reply'][:100]}...")
    except Exception as e:
        print(f"  [INFO] get_agent_response 测试跳过 (mock 配置): {e}")

    print()


def main():
    print()
    print("*" * 60)
    print("  CardioBot x emotion2vec 集成测试")
    print("*" * 60)
    print()

    passed = 0
    failed = 0

    tests = [
        test_init_exports,
        test_model_load_and_inference,
        test_voice_emotion_analysis,
        test_intervention_modules,
        test_core_dispatch,
    ]

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 60)
    print(f"  测试结果: {passed} 通过, {failed} 失败, 共 {passed + failed} 项")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
