"""
新功能集成测试
测试: ASR 模块、文件解析模块、信号分析带数据、Agent 响应带上传文件
"""

import sys
import os
import tempfile
import numpy as np

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_1_asr_module():
    """测试 1: ASR 模块基本功能"""
    print("\n=== 测试 1: ASR 模块 ===")
    try:
        from algorithms.asr.asr import transcribe_audio

        # 无效路径
        result = transcribe_audio("/nonexistent/audio.wav")
        assert result["status"] == "error", f"应为 error，实际: {result['status']}"
        assert result["text"] == ""
        print("  无效路径处理: OK")

        # None 路径
        result = transcribe_audio(None)
        assert result["status"] == "error"
        print("  None 路径处理: OK")

        # 空路径
        result = transcribe_audio("")
        assert result["status"] == "error"
        print("  空路径处理: OK")

        print("  [PASS] ASR 模块基础功能正确")
        return True
    except Exception as e:
        print(f"  [FAIL] ASR 模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_file_parser():
    """测试 2: 文件解析模块"""
    print("\n=== 测试 2: 文件解析模块 ===")
    try:
        from algorithms.file_parser.parser import parse_uploaded_file, _detect_signal_type

        # 信号类型检测
        assert _detect_signal_type("ecg_data.csv") == "ecg"
        assert _detect_signal_type("ECG-signal.npy") == "ecg"
        assert _detect_signal_type("心电信号.csv") == "ecg"
        assert _detect_signal_type("pcg_audio.wav") == "pcg"
        assert _detect_signal_type("心音数据.csv") == "pcg"
        assert _detect_signal_type("hrv_data.csv") == "hrv"
        assert _detect_signal_type("rr_intervals.npy") == "hrv"
        assert _detect_signal_type("random_data.csv") is None
        print("  信号类型检测: 全部正确")

        # CSV 解析
        with tempfile.NamedTemporaryFile(suffix="_ecg.csv", mode="w", delete=False, encoding="utf-8") as f:
            f.write("1,2,3\n4,5,6\n7,8,9\n")
            f.flush()
            result = parse_uploaded_file(f.name)
            fname = f.name
        os.unlink(fname)
        assert result["status"] == "success"
        assert result["file_type"] == "signal_csv"
        assert result["category"] == "signal"
        assert result["signal_type"] == "ecg"
        assert len(result["content"]["data"]) >= 6  # header=None: 3x3 = 9 values
        print(f"  CSV 解析: {result['filename']}, signal_type={result['signal_type']}, data_len={len(result['content']['data'])}")

        # NPY 解析
        with tempfile.NamedTemporaryFile(suffix="_pcg.npy", delete=False) as f:
            arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            np.save(f.name, arr)
            result = parse_uploaded_file(f.name)
            fname = f.name
        os.unlink(fname)
        assert result["status"] == "success"
        assert result["signal_type"] == "pcg"
        assert len(result["content"]["data"]) == 5
        print(f"  NPY 解析: signal_type={result['signal_type']}, data_len={len(result['content']['data'])}")

        # 无效路径
        result = parse_uploaded_file("/nonexistent/file.csv")
        assert result["status"] == "error"
        print("  无效路径处理: OK")

        # 不支持的类型
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            result = parse_uploaded_file(f.name)
            fname = f.name
        os.unlink(fname)
        assert result["status"] == "error"
        assert "不支持" in result["message"]
        print("  不支持的类型处理: OK")

        print("  [PASS] 文件解析模块全部通过")
        return True
    except Exception as e:
        print(f"  [FAIL] 文件解析模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_signal_analysis_with_data():
    """测试 3: 信号分析函数接收真实数据"""
    print("\n=== 测试 3: 信号分析（真实数据） ===")
    try:
        from algorithms.signal_processing.ecg import analyze_ecg
        from algorithms.signal_processing.pcg import analyze_pcg
        from algorithms.signal_processing.hrv import analyze_hrv

        # ECG: 12 导联 x 4096 采样点
        ecg_data = np.random.randn(12, 4096).astype(np.float32) * 0.5
        result = analyze_ecg(signal_data=ecg_data.flatten().tolist())
        assert result["module"] == "ecg"
        assert result["status"] in ("success", "demo", "placeholder")
        assert "risk_level" in result
        print(f"  ECG 分析: status={result['status']}, risk={result['risk_level']}, has_deep={'deep_analysis' in result}")

        # PCG: 40000 采样点
        pcg_data = np.random.randn(40000).astype(np.float32) * 0.1
        result = analyze_pcg(audio_data=pcg_data.tolist())
        assert result["module"] == "pcg"
        assert "risk_level" in result
        print(f"  PCG 分析: status={result['status']}, risk={result['risk_level']}")

        # HRV: RR 间期列表
        rr_intervals = [800, 810, 795, 820, 805, 815, 800, 830, 790, 825]
        result = analyze_hrv(rr_intervals=rr_intervals)
        assert result["module"] == "hrv"
        assert "metrics" in result
        print(f"  HRV 分析: status={result['status']}, metrics_keys={list(result['metrics'].keys())}")

        print("  [PASS] 信号分析（真实数据）全部通过")
        return True
    except Exception as e:
        print(f"  [FAIL] 信号分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_agent_with_uploaded_file():
    """测试 4: Agent 响应带上传文件"""
    print("\n=== 测试 4: Agent 响应（上传文件） ===")
    try:
        from agent.core import get_agent_response, clear_session_history
        from agent.memory import MemoryStore

        memory_store = MemoryStore(max_rounds=10)
        session_id = "test-file-upload"

        try:
            # 4a. 信号文件上传 (ECG)
            signal_upload = {
                "status": "success",
                "file_type": "signal_csv",
                "category": "signal",
                "signal_type": "ecg",
                "content": {"data": list(np.random.randn(12000)), "shape": [1000, 12]},
                "filename": "test_ecg.csv",
            }

            response = get_agent_response(
                user_message="帮我分析一下这个 ECG 数据",
                session_id=session_id,
                memory_store=memory_store,
                system_prompt="你是测试助手，简短回复。",
                uploaded_file=signal_upload,
            )
            assert "reply" in response
            assert "algorithms" in response
            assert len(response["algorithms"]) > 0
            print(f"  4a. ECG 信号上传: algorithms={len(response['algorithms'])} 个结果")
            for algo in response["algorithms"]:
                print(f"      {algo.get('module')}: {algo.get('status')}")

            # 4b. 信号文件上传 (HRV)
            signal_upload_hrv = {
                "status": "success",
                "file_type": "signal_csv",
                "category": "signal",
                "signal_type": "hrv",
                "content": {"data": [800, 810, 795, 820, 805], "shape": [5]},
                "filename": "test_hrv.csv",
            }

            response = get_agent_response(
                user_message="分析我的心率变异",
                session_id=session_id,
                memory_store=memory_store,
                system_prompt="简短回复。",
                uploaded_file=signal_upload_hrv,
            )
            assert "reply" in response
            print(f"  4b. HRV 信号上传: reply_length={len(response['reply'])}")

            print("  [PASS] Agent 响应（上传文件）全部通过")
            return True
        finally:
            clear_session_history(session_id, memory_store=memory_store)

    except Exception as e:
        print(f"  [FAIL] Agent 上传文件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_build_messages_multimedia():
    """测试 5: build_messages 处理多媒体 content"""
    print("\n=== 测试 5: build_messages 多媒体处理 ===")
    try:
        from agent.model import build_messages

        # 标准文本消息
        msgs = build_messages(
            user_message="测试消息",
            history=[{"role": "user", "content": "历史消息"}],
        )
        assert len(msgs) == 2  # 1 history + 1 user
        assert msgs[0]["content"] == "历史消息"
        assert msgs[1]["content"] == "测试消息"
        print("  标准文本消息: OK")

        # 多媒体消息（Gradio 5.x 格式）
        multimedia_content = [
            ("text", "这是语音转写的文字"),
            ("audio", "/path/to/audio.wav"),
        ]
        msgs = build_messages(
            user_message="新消息",
            history=[{"role": "user", "content": multimedia_content}],
        )
        assert len(msgs) == 2
        # 应该提取出 text 部分
        assert "这是语音转写的文字" in msgs[0]["content"]
        print(f"  多媒体消息: 提取文本='{msgs[0]['content']}'")

        # 空多媒体消息
        empty_multimedia = [("audio", "/path/to/audio.wav")]
        msgs = build_messages(
            user_message="测试",
            history=[{"role": "user", "content": empty_multimedia}],
        )
        # 空 text 部分应该不添加到历史（因为 content 为空）
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "测试"
        print("  空文本多媒体消息: OK")

        print("  [PASS] build_messages 多媒体处理全部通过")
        return True
    except Exception as e:
        print(f"  [FAIL] build_messages 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_history_manager_multimedia():
    """测试 6: history_manager 兼容多媒体消息"""
    print("\n=== 测试 6: history_manager 多媒体兼容 ===")
    try:
        from agent.history_manager import generate_title

        # 多媒体用户消息
        history = [
            {
                "role": "user",
                "content": [
                    ("text", "我今天感觉心慌"),
                    ("audio", "/path/to/audio.wav"),
                ],
            },
            {"role": "assistant", "content": "请问有什么具体症状？"},
        ]

        title = generate_title(history)
        assert "我今天感觉心慌" in title
        print(f"  多媒体消息标题: '{title}'")

        # 占位文本过滤
        history2 = [
            {"role": "user", "content": "[语音消息]"},
            {"role": "assistant", "content": "你好"},
        ]
        title2 = generate_title(history2)
        assert title2 == "新对话"
        print(f"  占位文本过滤: '{title2}'")

        print("  [PASS] history_manager 多媒体兼容全部通过")
        return True
    except Exception as e:
        print(f"  [FAIL] history_manager 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_file_to_analysis_pipeline():
    """测试 7: 文件解析 -> 算法分析完整流程"""
    print("\n=== 测试 7: 文件到分析完整流程 ===")
    try:
        from algorithms.file_parser.parser import parse_uploaded_file
        from algorithms.signal_processing.ecg import analyze_ecg
        from algorithms.signal_processing.pcg import analyze_pcg
        from algorithms.signal_processing.hrv import analyze_hrv

        # CSV ECG -> 解析 -> 分析
        with tempfile.NamedTemporaryFile(suffix="_ecg.csv", mode="w", delete=False, encoding="utf-8") as f:
            data = np.random.randn(100, 12)
            import pandas as pd
            pd.DataFrame(data).to_csv(f, index=False)
            f.flush()
            fname = f.name

        parsed = parse_uploaded_file(fname)
        os.unlink(fname)
        assert parsed["signal_type"] == "ecg"
        result = analyze_ecg(signal_data=parsed["content"]["data"])
        assert result["module"] == "ecg"
        print(f"  CSV ECG 管线: signal_type={parsed['signal_type']}, status={result['status']}")

        # NPY PCG -> 解析 -> 分析
        with tempfile.NamedTemporaryFile(suffix="_pcg.npy", delete=False) as f:
            np.save(f.name, np.random.randn(5000).astype(np.float32))
            fname = f.name

        parsed = parse_uploaded_file(fname)
        os.unlink(fname)
        assert parsed["signal_type"] == "pcg"
        result = analyze_pcg(audio_data=parsed["content"]["data"])
        assert result["module"] == "pcg"
        print(f"  NPY PCG 管线: signal_type={parsed['signal_type']}, status={result['status']}")

        # CSV HRV -> 解析 -> 分析
        with tempfile.NamedTemporaryFile(suffix="_hrv.csv", mode="w", delete=False, encoding="utf-8") as f:
            f.write("800\n810\n795\n820\n805\n")
            f.flush()
            fname = f.name

        parsed = parse_uploaded_file(fname)
        os.unlink(fname)
        assert parsed["signal_type"] == "hrv"
        result = analyze_hrv(rr_intervals=parsed["content"]["data"])
        assert result["module"] == "hrv"
        print(f"  CSV HRV 管线: signal_type={parsed['signal_type']}, status={result['status']}")

        print("  [PASS] 文件到分析完整流程全部通过")
        return True
    except Exception as e:
        print(f"  [FAIL] 文件到分析流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  CardioBot 新功能集成测试")
    print("=" * 60)

    results = []
    results.append(("测试 1: ASR 模块", test_1_asr_module()))
    results.append(("测试 2: 文件解析", test_2_file_parser()))
    results.append(("测试 3: 信号分析（真实数据）", test_3_signal_analysis_with_data()))
    results.append(("测试 4: Agent 响应（上传文件）", test_4_agent_with_uploaded_file()))
    results.append(("测试 5: build_messages 多媒体", test_5_build_messages_multimedia()))
    results.append(("测试 6: history_manager 多媒体", test_6_history_manager_multimedia()))
    results.append(("测试 7: 文件到分析流程", test_7_file_to_analysis_pipeline()))

    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n  总计: {passed}/{total} 通过")
    if passed == total:
        print("  全部通过!")
    else:
        print(f"  {total - passed} 项测试失败!")
