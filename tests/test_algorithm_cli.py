"""Offline regression checks for implemented algorithm choices in the unified CLI."""

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms import STRATEGIES
from scripts import run_experiment


class AlgorithmCliTests(unittest.TestCase):
    def test_implemented_algorithms_forward_without_starting_training(self):
        implemented = sorted(name for name, status in STRATEGIES.items() if status == "implemented")
        self.assertTrue({"head_prox", "coverage_prox", "logit_calibration",
                         "coverage_calibrated", "calibrated_fedprox"}.issubset(implemented))
        for algorithm in implemented:
            with self.subTest(algorithm=algorithm):
                arguments = ["run_experiment.py", "--config", "chosen.yaml",
                             "--algorithm", algorithm, "--partition", "dirichlet",
                             "--alpha", "0.3", "--seed", "2026", "--num-clients", "3",
                             "--rounds", "8", "--backend", "sequential"]
                with patch.object(sys, "argv", arguments), \
                        patch.object(run_experiment, "run_federated_simulation") as simulation, \
                        patch.object(run_experiment, "run_centralized") as centralized:
                    run_experiment.main()
                simulation.assert_called_once_with(
                    "chosen.yaml", partition_type_override="dirichlet", alpha_override=0.3,
                    seed_override=2026, algorithm_override=algorithm,
                    num_clients_override=3, num_rounds_override=8,
                    backend_override="sequential")
                centralized.assert_not_called()

    def test_planned_extensions_are_rejected_before_any_run(self):
        for algorithm in ("fednova", "proposed"):
            with self.subTest(algorithm=algorithm):
                self.assertEqual(STRATEGIES[algorithm], "planned")
                error = io.StringIO()
                with patch.object(sys, "argv", ["run_experiment.py", "--algorithm", algorithm]), \
                        patch.object(sys, "stderr", error), \
                        patch.object(run_experiment, "run_federated_simulation") as simulation, \
                        patch.object(run_experiment, "run_centralized") as centralized:
                    with self.assertRaises(SystemExit) as stopped:
                        run_experiment.main()
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn("invalid choice", error.getvalue())
                simulation.assert_not_called()
                centralized.assert_not_called()


if __name__ == "__main__":
    unittest.main()
