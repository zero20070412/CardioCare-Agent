import unittest

from agent.core import (
    build_model_input,
    clear_session_history,
    dispatch_algorithm_results,
    get_agent_response,
    get_session_history,
)
from agent.memory import MemoryStore


class AgentCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory_store = MemoryStore(max_rounds=4)
        self.session_id = "test-session"

    def test_get_agent_response_supports_multi_turn_history(self) -> None:
        first = get_agent_response(
            "请分析一下我的心电情况",
            session_id=self.session_id,
            memory_store=self.memory_store,
            system_prompt="你是测试助手。",
        )
        second = get_agent_response(
            "继续说明一下风险点",
            session_id=self.session_id,
            memory_store=self.memory_store,
            run_algorithms=False,
            system_prompt="你是测试助手。",
        )

        self.assertEqual(first["session_id"], self.session_id)
        self.assertIn("ecg", first["algorithm_results"])
        self.assertEqual(second["session_id"], self.session_id)
        self.assertGreaterEqual(len(second["history"]), 5)
        self.assertEqual(second["history"][-2]["role"], "user")
        self.assertEqual(second["history"][-2]["content"], "继续说明一下风险点")
        self.assertEqual(second["history"][-1]["role"], "assistant")

    def test_dispatch_algorithm_results_returns_requested_placeholders(self) -> None:
        results = dispatch_algorithm_results(
            user_message="普通问候",
            requested_modules=["ecg", "hrv", "cnn", "fusion"],
        )

        self.assertEqual(set(results.keys()), {"ecg", "hrv", "cnn", "fusion"})
        self.assertEqual(results["ecg"]["status"], "placeholder")
        self.assertEqual(results["hrv"]["module"], "hrv")
        self.assertEqual(results["cnn"]["module"], "cnn")
        self.assertEqual(results["fusion"]["module"], "transformer")

    def test_build_model_input_includes_algorithm_context_when_available(self) -> None:
        model_input = build_model_input(
            user_message="帮我总结一下",
            algorithm_results={
                "ecg": {
                    "status": "placeholder",
                    "risk_level": "low",
                    "summary": "ECG placeholder result",
                    "metrics": {"heart_rate_bpm": 72},
                }
            },
        )

        self.assertIn("ECG", model_input)
        self.assertIn("heart_rate_bpm=72", model_input)
        self.assertIn("帮我总结一下", model_input)

    def test_clear_session_history_can_keep_system_prompt(self) -> None:
        get_agent_response(
            "请分析一下我的HRV",
            session_id=self.session_id,
            memory_store=self.memory_store,
            system_prompt="你是测试助手。",
        )

        clear_session_history(
            self.session_id,
            memory_store=self.memory_store,
            keep_system_message=True,
        )
        history = get_session_history(
            self.session_id,
            memory_store=self.memory_store,
            include_system=True,
        )

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["role"], "system")
        self.assertEqual(history[0]["content"], "你是测试助手。")


if __name__ == "__main__":
    unittest.main()
