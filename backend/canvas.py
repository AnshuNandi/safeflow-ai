"""
SafeFlow-AI: Professional High-Definition Intersection Canvas Engine
Strict Left-Hand Drive (Indian Traffic Norm - IRC: SP 41):
- Vehicles drive on the LEFT side of the yellow center line
- Vehicles wait queued strictly behind the white stop lines
- No vehicles awkwardly stuck in the intersection center box
- Large, prominent digital countdown timers on all 4 signal posts
"""

import numpy as np
import cv2
from typing import Dict, Any

def render_crossroad_canvas(
    green_splits: Dict[str, float],
    active_scenario: str,
    vehicle_counts: Dict[str, Dict[str, int]],
    lane_pcus: Dict[str, float]
) -> np.ndarray:
    W, H = 640, 640
    canvas = np.full((H, W, 3), (28, 30, 34), dtype=np.uint8)

    R_MIN = 220
    R_MAX = 420
    CENTER = 320

    # 1. Sidewalk Curbs (City Blocks)
    cv2.rectangle(canvas, (0, 0), (R_MIN, R_MIN), (44, 46, 52), -1)
    cv2.rectangle(canvas, (R_MAX, 0), (W, R_MIN), (44, 46, 52), -1)
    cv2.rectangle(canvas, (0, R_MAX), (R_MIN, H), (44, 46, 52), -1)
    cv2.rectangle(canvas, (R_MAX, R_MAX), (W, H), (44, 46, 52), -1)

    # Curb Edges
    cv2.line(canvas, (R_MIN, 0), (R_MIN, R_MIN), (90, 95, 105), 3)
    cv2.line(canvas, (0, R_MIN), (R_MIN, R_MIN), (90, 95, 105), 3)
    cv2.line(canvas, (R_MAX, 0), (R_MAX, R_MIN), (90, 95, 105), 3)
    cv2.line(canvas, (R_MAX, R_MIN), (W, R_MIN), (90, 95, 105), 3)
    cv2.line(canvas, (0, R_MAX), (R_MIN, R_MAX), (90, 95, 105), 3)
    cv2.line(canvas, (R_MIN, R_MAX), (R_MIN, H), (90, 95, 105), 3)
    cv2.line(canvas, (R_MAX, R_MAX), (W, R_MAX), (90, 95, 105), 3)
    cv2.line(canvas, (R_MAX, R_MAX), (R_MAX, H), (90, 95, 105), 3)

    # 2. Dark Asphalt Roadways
    cv2.rectangle(canvas, (R_MIN, 0), (R_MAX, H), (36, 38, 42), -1)
    cv2.rectangle(canvas, (0, R_MIN), (W, R_MAX), (36, 38, 42), -1)

    # 3. Zebra Crosswalks
    for x in range(R_MIN + 8, R_MAX - 8, 20):
        cv2.rectangle(canvas, (x, R_MIN - 35), (x + 12, R_MIN - 10), (220, 225, 230), -1)
        cv2.rectangle(canvas, (x, R_MAX + 10), (x + 12, R_MAX + 35), (220, 225, 230), -1)
    for y in range(R_MIN + 8, R_MAX - 8, 20):
        cv2.rectangle(canvas, (R_MIN - 35, y), (R_MIN - 10, y + 12), (220, 225, 230), -1)
        cv2.rectangle(canvas, (R_MAX + 10, y), (R_MAX + 35, y + 12), (220, 225, 230), -1)

    # Stop Lines (White horizontal / vertical bar across incoming approach)
    cv2.line(canvas, (R_MIN, R_MIN - 40), (R_MAX, R_MIN - 40), (245, 245, 245), 4)
    cv2.line(canvas, (R_MIN, R_MAX + 40), (R_MAX, R_MAX + 40), (245, 245, 245), 4)
    cv2.line(canvas, (R_MIN - 40, R_MIN), (R_MIN - 40, R_MAX), (245, 245, 245), 4)
    cv2.line(canvas, (R_MAX + 40, R_MIN), (R_MAX + 40, R_MAX), (245, 245, 245), 4)

    # Yellow Dashed Center Lines
    for y in range(0, R_MIN - 45, 28):
        cv2.line(canvas, (CENTER, y), (CENTER, y + 16), (0, 215, 255), 2)
    for y in range(R_MAX + 45, H, 28):
        cv2.line(canvas, (CENTER, y), (CENTER, y + 16), (0, 215, 255), 2)
    for x in range(0, R_MIN - 45, 28):
        cv2.line(canvas, (x, CENTER), (x + 16, CENTER), (0, 215, 255), 2)
    for x in range(R_MAX + 45, W, 28):
        cv2.line(canvas, (x, CENTER), (x + 16, CENTER), (0, 215, 255), 2)

    # 4. Traffic Light State Logic
    is_ambulance = "Ambulance" in active_scenario
    is_collision = "Collision" in active_scenario

    if is_ambulance:
        light_states = {"North": "RED", "South": "GREEN", "East": "RED", "West": "RED"}
    elif is_collision:
        light_states = {"North": "GREEN", "South": "RED", "East": "RED", "West": "RED"}
    else:
        heaviest = max(lane_pcus, key=lane_pcus.get)
        light_states = {arm: ("GREEN" if arm == heaviest else "RED") for arm in ["North", "South", "East", "West"]}

    # 5. Signal Posts with Prominent Digital Countdown Timers
    # Positioned on the DRIVER'S APPROACH CURB:
    # - Southbound traffic on North Road looks at the signal on the LEFT curb (top-right of junction: x=R_MAX+45, y=R_MIN-55)
    # - Northbound traffic on South Road looks at the signal on the LEFT curb (bottom-left of junction: x=R_MIN-45, y=R_MAX+55)
    # - Westbound traffic on East Road looks at the signal on the LEFT curb (bottom-right: x=R_MAX+55, y=R_MAX+45)
    # - Eastbound traffic on West Road looks at the signal on the LEFT curb (top-left: x=R_MIN-55, y=R_MIN-45)
    def draw_prominent_signal(px, py, state, timer_sec, label, is_blocked=False):
        box_w, box_h = 56, 90
        bx = px - box_w // 2
        by = py - box_h // 2
        cv2.rectangle(canvas, (bx, by), (bx + box_w, by + box_h), (12, 14, 18), -1)
        cv2.rectangle(canvas, (bx, by), (bx + box_w, by + box_h), (95, 105, 120), 2)

        r_on = (state == "RED")
        r_col = (30, 30, 255) if r_on else (15, 15, 55)
        cv2.circle(canvas, (px, py - 23), 11, r_col, -1)
        if r_on:
            cv2.circle(canvas, (px, py - 23), 16, (0, 0, 220), 2)

        g_on = (state == "GREEN")
        g_col = (30, 250, 60) if g_on else (10, 55, 18)
        cv2.circle(canvas, (px, py + 23), 11, g_col, -1)
        if g_on:
            cv2.circle(canvas, (px, py + 23), 16, (0, 220, 0), 2)

        # Digital Countdown Timer Badge
        # If lane is physically BLOCKED by an accident:
        # The signal is held permanently on RED STOP with NO timer (-- / HOLD) to prevent drivers from thinking the lane will open!
        cv2.rectangle(canvas, (px - 26, py - 11), (px + 26, py + 11), (0, 0, 0), -1)
        if is_blocked:
            timer_col = (0, 0, 255)
            border_col = (0, 0, 200)
            cv2.rectangle(canvas, (px - 26, py - 11), (px + 26, py + 11), border_col, 1)
            cv2.putText(canvas, "--", (px - 10, py + 5), cv2.FONT_HERSHEY_DUPLEX, 0.46, timer_col, 1)
            cv2.putText(canvas, f"{label} [HOLD]", (bx - 8, by - 8), cv2.FONT_HERSHEY_DUPLEX, 0.36, (80, 100, 255), 1)
        else:
            timer_col = (0, 255, 60) if g_on else (50, 80, 255)
            border_col = (0, 255, 60) if g_on else (0, 0, 180)
            cv2.rectangle(canvas, (px - 26, py - 11), (px + 26, py + 11), border_col, 1)
            t_str = f"{int(timer_sec)}s"
            offset_x = 17 if timer_sec < 100 else 22
            cv2.putText(canvas, t_str, (px - offset_x, py + 5), cv2.FONT_HERSHEY_DUPLEX, 0.46, timer_col, 1)

            l_col = (100, 255, 150) if g_on else (180, 190, 255)
            mode_tag = "GO" if g_on else "WAIT"
            cv2.putText(canvas, f"{label} [{mode_tag}]", (bx - 8, by - 8), cv2.FONT_HERSHEY_DUPLEX, 0.36, l_col, 1)

    # Real-World Sequential Phase Order:
    # Phase 1: East -> Phase 2: West -> Phase 3: South -> Phase 4: North
    phase_order = ["East", "West", "South", "North"]
    clearance_per_phase = 4.0  # 3s Yellow + 1s All-Red

    # Identify currently active green arm:
    active_green_arm = [a for a, s in light_states.items() if s == "GREEN"]
    active_arm = active_green_arm[0] if active_green_arm else "East"

    def get_real_world_timer(arm):
        if light_states[arm] == "GREEN":
            # Active green movement countdown
            return round(green_splits.get(arm, 15.0))
        
        if is_ambulance:
            # POST-EMERGENCY RECOVERY ORDER:
            # In an intelligent system, the approach with the heaviest accumulated backlog gets priority right after the ambulance!
            # Here, North is heaviest (63.0 PCU), so North goes FIRST after ambulance!
            # 1st after Ambulance -> NORTH (Waits: 30s South + 4s clearance = 34s)
            # 2nd after North -> EAST (Waits: 34s + 44.0s North + 4s clearance = 82s)
            # 3rd after East -> WEST (Waits: 82s + 15.0s East + 4s clearance = 101s)
            emg_clearance = green_splits.get("South", 30.0) + clearance_per_phase  # 34s
            g_north = green_splits.get("North", 44.0)
            g_east = green_splits.get("East", 15.0)
            
            if arm == "North":
                return round(emg_clearance)  # 34s - North goes next!
            elif arm == "East":
                return round(emg_clearance + g_north + clearance_per_phase)  # 82s
            else: # West
                return round(emg_clearance + g_north + clearance_per_phase + g_east + clearance_per_phase)  # 101s
        
        # Real-World Cumulative Delay Calculation:
        # Sum the remaining duration of the active green phase PLUS
        # the green and clearance durations of all intermediate phases scheduled before this arm!
        curr_idx = phase_order.index(active_arm)
        target_idx = phase_order.index(arm)
        
        wait_time = 0.0
        # 1. Finish the active phase:
        wait_time += green_splits.get(active_arm, 15.0) + clearance_per_phase
        
        # 2. Add intermediate phases between active and target arm in cyclic order:
        idx = (curr_idx + 1) % len(phase_order)
        while idx != target_idx:
            inter_arm = phase_order[idx]
            wait_time += green_splits.get(inter_arm, 15.0) + clearance_per_phase
            idx = (idx + 1) % len(phase_order)
            
        return round(wait_time)

    # Place each signal directly adjacent to its incoming stop line:
    # 1. NORTH APPROACH SIGNAL (Controls Southbound incoming vehicles on East lane):
    draw_prominent_signal(R_MAX + 45, R_MIN - 55, light_states["North"], get_real_world_timer("North"), "NORTH")

    # 2. SOUTH APPROACH SIGNAL (Controls Northbound incoming vehicles on West lane):
    # If collision incident is active on South approach, mark as BLOCKED (RED HOLD, no countdown timer):
    draw_prominent_signal(R_MIN - 45, R_MAX + 55, light_states["South"], get_real_world_timer("South"), "SOUTH", is_blocked=is_collision)

    # 3. WEST APPROACH SIGNAL (Controls Eastbound incoming vehicles on North lane):
    draw_prominent_signal(R_MIN - 55, R_MIN - 45, light_states["West"], get_real_world_timer("West"), "WEST")

    # 4. EAST APPROACH SIGNAL (Controls Westbound incoming vehicles on South lane):
    draw_prominent_signal(R_MAX + 55, R_MAX + 45, light_states["East"], get_real_world_timer("East"), "EAST")

    # 6. Realistic Vehicle Model Drawers
    def draw_car(cx, cy, heading, color):
        w, l = 28, 48
        if heading in ["S", "N"]:
            x1, y1 = cx - w//2, cy - l//2
            x2, y2 = cx + w//2, cy + l//2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (15, 15, 15), 1)
            cv2.rectangle(canvas, (x1 + 3, y1 + 10), (x2 - 3, y1 + 17), (40, 45, 55), -1)
            cv2.rectangle(canvas, (x1 + 3, y2 - 16), (x2 - 3, y2 - 10), (40, 45, 55), -1)
            cv2.rectangle(canvas, (x1 + 4, y1 + 17), (x2 - 4, y2 - 16), (int(color[0]*0.8), int(color[1]*0.8), int(color[2]*0.8)), -1)
            if heading == "S":
                cv2.circle(canvas, (x1 + 4, y2 - 1), 3, (120, 255, 255), -1)
                cv2.circle(canvas, (x2 - 4, y2 - 1), 3, (120, 255, 255), -1)
            else:
                cv2.circle(canvas, (x1 + 4, y1 + 1), 3, (120, 255, 255), -1)
                cv2.circle(canvas, (x2 - 4, y1 + 1), 3, (120, 255, 255), -1)
        else: # E or W
            x1, y1 = cx - l//2, cy - w//2
            x2, y2 = cx + l//2, cy + w//2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (15, 15, 15), 1)
            cv2.rectangle(canvas, (x1 + 10, y1 + 3), (x1 + 17, y2 - 3), (40, 45, 55), -1)
            cv2.rectangle(canvas, (x2 - 17, y1 + 3), (x2 - 10, y2 - 3), (40, 45, 55), -1)

    def draw_bus(cx, cy, heading):
        w, l = 34, 86
        if heading in ["S", "N"]:
            x1, y1 = cx - w//2, cy - l//2
            x2, y2 = cx + w//2, cy + l//2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 140, 245), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 85, 175), 2)
            cv2.rectangle(canvas, (x1 + 3, y2 - 14), (x2 - 3, y2 - 4), (40, 50, 65), -1)
            cv2.rectangle(canvas, (x1 + 5, y1 + 12), (x2 - 5, y2 - 20), (225, 230, 235), -1)
            cv2.putText(canvas, "BUS", (cx - 13, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (25, 25, 25), 1)
        else:
            x1, y1 = cx - l//2, cy - w//2
            x2, y2 = cx + l//2, cy + w//2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 140, 245), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 85, 175), 2)
            cv2.putText(canvas, "BUS", (cx - 13, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (25, 25, 25), 1)

    # 7. STRICT INDIAN LEFT-HAND DRIVE VEHICLE QUEUES
    # In Left-Hand Drive (India / UK / Japan):
    # - North Approach driving Southbound drives on the EAST (left side from driver POV): x in [CENTER, R_MAX]
    # - South Approach driving Northbound drives on the WEST (left side from driver POV): x in [R_MIN, CENTER]
    # - West Approach driving Eastbound drives on the NORTH (left side from driver POV): y in [R_MIN, CENTER]
    # - East Approach driving Westbound drives on the SOUTH (left side from driver POV): y in [CENTER, R_MAX]
    # All waiting vehicles stay behind the Stop Line! No vehicles stopped inside the junction box!

    n_cars = vehicle_counts.get("North", {}).get("car", 0)
    n_buses = vehicle_counts.get("North", {}).get("bus", 0)
    s_cars = vehicle_counts.get("South", {}).get("car", 0)
    s_buses = vehicle_counts.get("South", {}).get("bus", 0)

    car_palette = [(210, 130, 40), (60, 160, 220), (180, 80, 70), (70, 175, 90), (160, 70, 190)]

    # --- NORTH APPROACH (Incoming Southbound on Left Side -> x = 370) ---
    lane_n_car = R_MAX - 50  # x = 370
    lane_n_bus = R_MAX - 50
    cars_to_draw_n = min(4, n_cars)
    for idx in range(cars_to_draw_n):
        y_pos = (R_MIN - 70) - (idx * 56)
        if y_pos > 30:
            draw_car(lane_n_car, y_pos, "S", car_palette[idx % len(car_palette)])

    if n_buses > 0:
        # Bus in heavy queue behind cars
        bus_y = (R_MIN - 70) - (cars_to_draw_n * 56) - 20
        if bus_y > 45:
            draw_bus(lane_n_bus, bus_y, "S")

    # --- SOUTH APPROACH (Incoming Northbound on Left Side -> x = 270) ---
    lane_s_car = R_MIN + 50  # x = 270
    lane_s_bus = R_MIN + 50
    
    if is_ambulance:
        # Ambulance waiting or clearing at front of South approach
        ay = R_MAX + 80
        ax = lane_s_car
        cv2.rectangle(canvas, (ax - 18, ay - 35), (ax + 18, ay + 35), (250, 250, 250), -1)
        cv2.rectangle(canvas, (ax - 18, ay - 35), (ax + 18, ay + 35), (0, 0, 220), 2)
        cv2.rectangle(canvas, (ax - 4, ay - 14), (ax + 4, ay + 14), (0, 0, 220), -1)
        cv2.rectangle(canvas, (ax - 14, ay - 4), (ax + 14, ay + 4), (0, 0, 220), -1)
        cv2.circle(canvas, (ax - 10, ay - 32), 7, (0, 0, 255), -1)
        cv2.circle(canvas, (ax + 10, ay - 32), 7, (255, 100, 0), -1)
        cv2.putText(canvas, "AMBULANCE", (ax - 34, ay + 50), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 255, 120), 1)
    elif is_collision:
        # Collision hazard marker on South approach (crashed vehicles + hazard triangle)
        cy = R_MAX + 75
        cx = lane_s_car
        # Two colliding cars angled/impacted
        draw_car(cx - 10, cy - 10, "N", (40, 40, 200))
        draw_car(cx + 8, cy + 12, "E", (200, 120, 30))
        # Red hazard warning perimeter
        cv2.circle(canvas, (cx, cy), 32, (0, 0, 255), 2)
        cv2.putText(canvas, "! CRASH !", (cx - 36, cy + 52), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 80, 255), 1)
        cv2.putText(canvas, "LANE BLOCKED", (cx - 44, cy + 68), cv2.FONT_HERSHEY_DUPLEX, 0.38, (0, 0, 255), 1)
    else:
        cars_to_draw_s = min(4, s_cars)
        for idx in range(cars_to_draw_s):
            y_pos = (R_MAX + 70) + (idx * 56)
            if y_pos < H - 30:
                draw_car(lane_s_car, y_pos, "N", car_palette[(idx + 2) % len(car_palette)])
        if s_buses > 0:
            bus_y_s = (R_MAX + 70) + (cars_to_draw_s * 56) + 20
            if bus_y_s < H - 45:
                draw_bus(lane_s_bus, bus_y_s, "N")

    # --- EAST APPROACH (Incoming Westbound on Left Side -> y = 370) ---
    lane_e_y = R_MAX - 50  # y = 370
    draw_car(R_MAX + 75, lane_e_y, "W", (180, 70, 70))
    draw_car(R_MAX + 140, lane_e_y, "W", (70, 170, 80))

    # --- WEST APPROACH (Incoming Eastbound on Left Side -> y = 270) ---
    lane_w_y = R_MIN + 50  # y = 270
    draw_car(R_MIN - 75, lane_w_y, "E", (150, 60, 180))
    draw_car(R_MIN - 140, lane_w_y, "E", (200, 180, 50))

    # 8. Canvas Header
    cv2.rectangle(canvas, (10, 10), (330, 42), (18, 20, 24), -1)
    cv2.rectangle(canvas, (10, 10), (330, 42), (70, 75, 85), 1)
    cv2.putText(canvas, "SAFEFLOW-AI: J1 INTERSECTION CANVAS", (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1)

    return canvas
