"""
SafeFlow-AI Automated Pipeline Test Suite
Validates:
1. Heterogeneous Indian Road Congress (IRC) PCU weights.
2. Webster cycle time conservation (sum(green) + 16s == 120s).
3. Fail-safe starvation guardrails (min green >= 15s).
4. Emergency preemption priority overrides (30s South wave).
5. Canvas rendering and image integrity across all operating scenarios.
"""

import unittest
import numpy as np
from backend.optimizer import (
    calculate_lane_pcu,
    compute_adaptive_green_splits,
    CYCLE_BUDGET_SEC,
    MIN_GREEN_SEC,
    MAX_GREEN_SEC
)
from backend.canvas import render_crossroad_canvas


class TestSafeFlowPipeline(unittest.TestCase):

    def test_pcu_calculation(self):
        counts = {"bus": 1, "car": 2, "motorcycle": 4}
        pcu = calculate_lane_pcu(counts)
        self.assertEqual(pcu, 7.0)

    def test_cycle_time_conservation(self):
        pcus = {"North": 20.0, "South": 20.0, "East": 10.0, "West": 10.0}
        result = compute_adaptive_green_splits(pcus)
        self.assertEqual(result["cycle_total"], CYCLE_BUDGET_SEC)
        self.assertEqual(sum(result["green_durations"].values()) + 16.0, CYCLE_BUDGET_SEC)

    def test_extreme_asymmetry_starvation_guardrail(self):
        pcus = {"North": 90.0, "South": 0.0, "East": 0.0, "West": 0.0}
        result = compute_adaptive_green_splits(pcus)
        for arm in ["South", "East", "West"]:
            self.assertGreaterEqual(result["green_durations"][arm], MIN_GREEN_SEC)
        self.assertEqual(result["cycle_total"], CYCLE_BUDGET_SEC)

    def test_emergency_preemption(self):
        pcus = {"North": 30.0, "South": 10.0, "East": 15.0, "West": 15.0}
        result = compute_adaptive_green_splits(pcus, emergency_lane="South")
        self.assertEqual(result["status"], "EMERGENCY_PREEMPTION")
        self.assertEqual(result["green_durations"]["South"], 30.0)
        self.assertEqual(result["cycle_total"], CYCLE_BUDGET_SEC)

    def test_canvas_rendering_integrity(self):
        splits = {"North": 30.0, "South": 25.0, "East": 25.0, "West": 24.0}
        pcus = {"North": 40.0, "South": 20.0, "East": 15.0, "West": 15.0}
        counts = {
            "North": {"car": 10, "bus": 2, "motorcycle": 4},
            "South": {"car": 5, "bus": 1, "motorcycle": 2},
            "East": {"car": 4, "bus": 1, "motorcycle": 2},
            "West": {"car": 3, "bus": 0, "motorcycle": 1}
        }
        for scenario in [
            "1. Standard Autonomous Multi-Arm Flow",
            "2. Emergency Ambulance Preemption (South Approach Priority)",
            "3. Collision / Incident Containment (North Diversion)"
        ]:
            img = render_crossroad_canvas(splits, scenario, counts, pcus)
            self.assertIsInstance(img, np.ndarray)
            self.assertEqual(img.shape, (640, 640, 3))


if __name__ == "__main__":
    unittest.main()
