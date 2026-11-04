"""
SafeFlow-AI Adaptive Signal Optimization Engine
Implements:
1. Heterogeneous Indian Road Congress Passenger Car Unit (PCU) formulation.
2. Dynamic Webster Equi-Saturation green split allocation under strict 120s cycle budget.
3. Pedestrian starvation guardrails (15s minimum green <= G <= 60s maximum green).
4. Deterministic emergency preemption arbiter (corridor priority wave).
"""

from typing import Dict, Any, Optional

# Indian Road Congress standard PCU weights for mixed traffic
PCU_WEIGHTS = {
    "car": 1.0,
    "bus": 3.0,
    "truck": 3.0,
    "motorcycle": 0.5,
    "autorickshaw": 0.8
}

# Core safety invariants
CYCLE_BUDGET_SEC: float = 120.0
INTER_GREEN_CLEARANCE_PER_PHASE_SEC: float = 4.0  # 3s Yellow + 1s All-Red
TOTAL_CLEARANCE_SEC: float = 16.0                # 4 phases * 4.0s = 16.0s
MAX_ALLOCABLE_GREEN_SEC: float = CYCLE_BUDGET_SEC - TOTAL_CLEARANCE_SEC  # 104.0s

MIN_GREEN_SEC: float = 15.0
MAX_GREEN_SEC: float = 60.0
EMERGENCY_GREEN_SEC: float = 30.0


def calculate_lane_pcu(vehicle_counts: Dict[str, int]) -> float:
    """
    Computes total Passenger Car Units (PCU) based on vehicle classification.
    """
    total_pcu = 0.0
    for vtype, count in vehicle_counts.items():
        weight = PCU_WEIGHTS.get(vtype.lower(), 1.0)
        total_pcu += count * weight
    return round(total_pcu, 2)


def compute_adaptive_green_splits(
    lane_pcus: Dict[str, float],
    emergency_lane: Optional[str] = None
) -> Dict[str, Any]:
    """
    Allocates green times across intersection arms (e.g. North, South, East, West).
    Strictly conserves total cycle time: sum(green_times) + 16s = 120s.
    """
    arms = list(lane_pcus.keys())
    if not arms:
        arms = ["North", "South", "East", "West"]
        lane_pcus = {arm: 10.0 for arm in arms}

    # CASE 1: Emergency Preemption Active (Ambulance, Fire, Police)
    if emergency_lane and emergency_lane in lane_pcus:
        green_splits = {}
        green_splits[emergency_lane] = EMERGENCY_GREEN_SEC  # 30.0s priority wave
        
        # Remaining green pool is distributed among non-emergency arms PROPORTIONAL to their PCU traffic!
        non_emg_arms = [a for a in arms if a != emergency_lane]
        total_non_emg_pcu = sum(lane_pcus.get(a, 0.0) for a in non_emg_arms)
        rem_green_pool = MAX_ALLOCABLE_GREEN_SEC - EMERGENCY_GREEN_SEC  # 104 - 30 = 74s
        
        if total_non_emg_pcu <= 0.0:
            total_non_emg_pcu = 1.0
            
        for a in non_emg_arms:
            share = lane_pcus.get(a, 0.0) / total_non_emg_pcu
            allocated = round(rem_green_pool * share, 1)
            # Enforce IRC 15s minimum guardrail if pool permits
            green_splits[a] = max(MIN_GREEN_SEC, allocated)

        # Ensure exact 104.0s green budget conservation
        current_sum = sum(green_splits.values())
        diff = round(MAX_ALLOCABLE_GREEN_SEC - current_sum, 1)
        if diff != 0:
            # Adjust the busiest non-emergency arm
            busiest = max(non_emg_arms, key=lambda x: lane_pcus.get(x, 0.0))
            green_splits[busiest] = round(green_splits[busiest] + diff, 1)

        return {
            "status": "EMERGENCY_PREEMPTION",
            "active_emergency_arm": emergency_lane,
            "green_durations": green_splits,
            "cycle_total": round(sum(green_splits.values()) + TOTAL_CLEARANCE_SEC, 1),
            "safety_holds": [a for a in arms if a != emergency_lane]
        }

    # CASE 2: Autonomous Dynamic Webster Equi-Saturation
    total_pcu = sum(lane_pcus.values())
    if total_pcu <= 0.0:
        total_pcu = 1.0  # Prevent division by zero

    # Initial proportional distribution of allocable green time
    base_min_sum = len(arms) * MIN_GREEN_SEC  # 4 * 15 = 60s
    distributable_pool = MAX_ALLOCABLE_GREEN_SEC - base_min_sum  # 104 - 60 = 44s

    green_splits = {}
    for arm, pcu in lane_pcus.items():
        share = pcu / total_pcu
        allocated = MIN_GREEN_SEC + (share * distributable_pool)
        # Apply strict upper bound
        clamped = min(MAX_GREEN_SEC, max(MIN_GREEN_SEC, allocated))
        green_splits[arm] = round(clamped, 1)

    # Re-normalize to ensure exact mathematical conservation to 120.0s
    total_allocated = sum(green_splits.values())
    slack = round(MAX_ALLOCABLE_GREEN_SEC - total_allocated, 2)

    # Adjust the heaviest loaded arm to absorb rounding slack without violating bounds
    heaviest_arm = max(lane_pcus, key=lane_pcus.get)
    green_splits[heaviest_arm] = round(green_splits[heaviest_arm] + slack, 1)

    return {
        "status": "AUTONOMOUS_ADAPTIVE",
        "active_emergency_arm": None,
        "green_durations": green_splits,
        "cycle_total": round(sum(green_splits.values()) + TOTAL_CLEARANCE_SEC, 1),
        "safety_holds": []
    }

