"""
L-LSTrans 集成测试
测试 MultiModeforHeart 模型代码迁移和推理引擎集成
"""

import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_1_model_import():
    """测试 1: lstrans 模型代码可正常导入"""
    print("\n=== 测试 1: lstrans 模型代码导入 ===")
    try:
        from algorithms.deep_models.lstrans import (
            LSTransECG, LSTransPCG, LSNet, SKA,
            MultimodalStudentNet, AlignmentLayer,
        )
        print(f"  LSTransECG: {LSTransECG}")
        print(f"  LSTransPCG: {LSTransPCG}")
        print(f"  LSNet: {LSNet}")
        print(f"  SKA: {SKA}")
        print(f"  AlignmentLayer: {AlignmentLayer}")
        print("  [PASS] 所有模型类导入成功")
        return True
    except Exception as e:
        print(f"  [FAIL] 导入失败: {e}")
        return False


def test_2_inference_engine_init():
    """测试 2: 推理引擎初始化（无 checkpoint 模拟模式）"""
    print("\n=== 测试 2: 推理引擎初始化 ===")
    try:
        from algorithms.deep_models.inference_engine import DiagnosticInferenceEngine, get_inference_engine

        engine = get_inference_engine()
        assert engine is not None
        assert not engine.has_ecg_model
        assert not engine.has_pcg_model
        print(f"  引擎设备: {engine.device}")
        print(f"  ECG 模型已加载: {engine.has_ecg_model}")
        print(f"  PCG 模型已加载: {engine.has_pcg_model}")
        print("  [PASS] 推理引擎初始化成功（模拟模式）")
        return True
    except Exception as e:
        print(f"  [FAIL] 推理引擎初始化失败: {e}")
        return False


def test_3_ecg_demo_inference():
    """测试 3: ECG 模拟推理输出格式"""
    print("\n=== 测试 3: ECG 模拟推理 ===")
    try:
        from algorithms.deep_models.inference_engine import get_inference_engine

        engine = get_inference_engine()
        result = engine.analyze_ecg()

        # 验证输出结构
        required_keys = ["module", "status", "model_config", "features_summary",
                        "predictions", "top_classes", "risk_assessment", "note"]
        for key in required_keys:
            assert key in result, f"缺少键: {key}"

        assert result["module"] == "ecg_deep"
        assert result["status"] == "demo"
        assert "dim" in result["features_summary"]
        assert len(result["top_classes"]) > 0
        assert "risk_level" in result["risk_assessment"]

        print(f"  模块: {result['module']}")
        print(f"  状态: {result['status']}")
        print(f"  模型配置: {result['model_config']}")
        print(f"  特征维度: {result['features_summary']['dim']}")
        print(f"  Top1 分类: {result['top_classes'][0]['label']} ({result['top_classes'][0]['probability']:.1%})")
        print(f"  风险等级: {result['risk_assessment']['risk_level']}")
        print("  [PASS] ECG 模拟推理输出格式正确")
        return True
    except Exception as e:
        print(f"  [FAIL] ECG 模拟推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_pcg_demo_inference():
    """测试 4: PCG 模拟推理输出格式"""
    print("\n=== 测试 4: PCG 模拟推理 ===")
    try:
        from algorithms.deep_models.inference_engine import get_inference_engine

        engine = get_inference_engine()
        result = engine.analyze_pcg(location="MV")

        required_keys = ["module", "status", "model_config", "features_summary",
                        "predictions", "top_classes", "risk_assessment", "note"]
        for key in required_keys:
            assert key in result, f"缺少键: {key}"

        assert result["module"] == "pcg_deep"
        assert result["status"] == "demo"
        assert "正常" in result["predictions"]
        assert "异常" in result["predictions"]
        assert "auscultation_location" in result["risk_assessment"]

        print(f"  模块: {result['module']}")
        print(f"  状态: {result['status']}")
        print(f"  正常概率: {result['predictions']['正常']:.1%}")
        print(f"  异常概率: {result['predictions']['异常']:.1%}")
        print(f"  听诊位置: {result['risk_assessment']['auscultation_location']}")
        print("  [PASS] PCG 模拟推理输出格式正确")
        return True
    except Exception as e:
        print(f"  [FAIL] PCG 模拟推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_agent_dispatch():
    """测试 5: Agent 端到端（关键词触发 → 模型推理 → LLM 提示词构建）"""
    print("\n=== 测试 5: Agent 端到端流程 ===")
    try:
        from agent.core import get_agent_response, _run_deep_analysis, _format_deep_analysis_for_llm

        # 5a. 测试深度分析函数
        ecg_result = _run_deep_analysis("ecg")
        assert ecg_result["module"] == "ecg"
        print(f"  5a. ECG 深度分析: {ecg_result['status']}, risk={ecg_result.get('risk_level', 'N/A')}")

        pcg_result = _run_deep_analysis("pcg")
        assert pcg_result["module"] == "pcg"
        print(f"  5b. PCG 深度分析: {pcg_result['status']}, risk={pcg_result.get('risk_level', 'N/A')}")

        fusion_result = _run_deep_analysis("fusion")
        assert fusion_result["module"] == "transformer"
        print(f"  5c. 融合分析: {fusion_result['status']}, risk={fusion_result.get('risk_level', 'N/A')}")

        # 5d. 测试 LLM 提示词格式化
        lines = _format_deep_analysis_for_llm("ecg", ecg_result)
        assert len(lines) > 0
        print(f"  5d. ECG LLM 格式化: {len(lines)} 行")
        for line in lines:
            print(f"      {line}")

        lines = _format_deep_analysis_for_llm("pcg", pcg_result)
        assert len(lines) > 0
        print(f"  5e. PCG LLM 格式化: {len(lines)} 行")

        lines = _format_deep_analysis_for_llm("fusion", fusion_result)
        print(f"  5f. 融合 LLM 格式化: {len(lines)} 行, module=fusion, has_prediction={'prediction' in fusion_result}")
        if not lines:
            print(f"  DEBUG fusion_result keys: {list(fusion_result.keys())}")
        assert len(lines) > 0
        print(f"  5f. 融合 LLM 格式化: {len(lines)} 行")

        print("  [PASS] Agent 端到端流程正确")
        return True
    except Exception as e:
        print(f"  [FAIL] Agent 端到端流程失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_strip_bug_fix():
    """测试 6: 'list' object has no attribute 'strip' bug 修复"""
    print("\n=== 测试 6: .strip() bug 修复 ===")
    try:
        from agent.model import build_messages

        # 测试 list 类型的 content
        history_with_list = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": ["这是", "列表内容"]},  # list content
            {"role": "user", "content": "再说一下"},
        ]

        messages = build_messages(
            history=history_with_list,
            user_message="测试消息",
        )

        assert len(messages) == 4  # 3 history + 1 user
        assert messages[1]["role"] == "assistant"
        assert isinstance(messages[1]["content"], str)
        assert "这是" in messages[1]["content"]  # list 内容被合并为字符串
        print(f"  list content 处理: '{messages[1]['content']}'")
        print("  [PASS] list.strip() bug 已修复")
        return True
    except AttributeError as e:
        print(f"  [FAIL] .strip() bug 仍存在: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] 其他错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  L-LSTrans 集成测试")
    print("=" * 60)

    results = []
    results.append(("测试 1: 模型导入", test_1_model_import()))
    results.append(("测试 2: 推理引擎初始化", test_2_inference_engine_init()))
    results.append(("测试 3: ECG 模拟推理", test_3_ecg_demo_inference()))
    results.append(("测试 4: PCG 模拟推理", test_4_pcg_demo_inference()))
    results.append(("测试 5: Agent 端到端", test_5_agent_dispatch()))
    results.append(("测试 6: .strip() bug 修复", test_6_strip_bug_fix()))

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
