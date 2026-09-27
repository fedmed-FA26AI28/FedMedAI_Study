"""Calendar deadline checks must notice a clock jump across host suspension."""

import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

path = Path(__file__).resolve().parents[1] / "scripts/run_communication_study.py"
spec = importlib.util.spec_from_file_location("study_runner", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class DeadlineTests(unittest.TestCase):
    def test_completed_before_deadline(self):
        process = Mock(args=["worker"])
        process.wait.return_value = 0
        with patch.object(runner.time, "time", side_effect=[0, 0, 20]):
            self.assertEqual(runner.wait_with_deadline(process, 100), 0)
        process.wait.assert_called_once_with(timeout=30)

    def test_resume_past_deadline_after_wait_timeout(self):
        process = Mock(args=["worker"])
        process.wait.side_effect = subprocess.TimeoutExpired(["worker"], 30)
        with patch.object(runner.time, "time", side_effect=[0, 0, 1000]):
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.wait_with_deadline(process, 100)
        self.assertEqual(process.wait.call_count, 1)

    def test_completion_observed_past_deadline_is_not_accepted(self):
        process = Mock(args=["worker"])
        process.wait.return_value = 0
        with patch.object(runner.time, "time", side_effect=[0, 0, 1000]):
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.wait_with_deadline(process, 100)


if __name__ == "__main__":
    unittest.main()
