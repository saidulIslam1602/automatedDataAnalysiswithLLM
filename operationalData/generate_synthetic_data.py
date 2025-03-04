import math
import random
import time
import csv
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.animation as animation

##################################
# 1. Extended Data Generation    #
##################################
def generate_synthetic_data_extended(
    num_time_steps=500,
    max_missions=8
):
    """
    Generates synthetic data for a robot driving in a warehouse with additional fields:
        time_step, timestamp, robot_x, robot_y, orientation_rad, speed, packages_delivered,
        battery_level_percent, power_consumption_step, motor_currents, wheel_speeds,
        robot_temperature_C, fault_code, error_log, station_inventory,
        warehouse_temperature_C, warehouse_humidity_percent, obstacle_event, item_metadata,
        mission_queue_length, current_delivery_time_sec, lidar_data, camera_feed,
        imu_accel, imu_gyro, idle_time_step, throughput_packages_per_hour,
        energy_efficiency_m_per_wh

    The warehouse has more stations, and the robot can run more missions.
    """

    # ---------- Warehouse Stations (12 stations) ----------
    stations = {
        0:  (0.0,  0.0),
        1:  (4.0,  2.0),
        2:  (8.0,  1.0),
        3:  (2.0,  6.0),
        4:  (5.0,  9.0),
        5:  (10.0, 4.0),
        6:  (12.0, 8.0),
        7:  (1.0,  10.0),
        8:  (7.0,  11.0),
        9:  (14.0, 2.0),
        10: (3.0,  12.0),
        11: (11.0, 10.0)
    }

    # ---------- Missions Setup ----------
    # We will create random missions from the station set:
    station_ids = list(stations.keys())
    chosen_missions = []
    for _ in range(max_missions):
        pick = random.choice(station_ids)
        drop = random.choice(station_ids)
        while drop == pick:
            drop = random.choice(station_ids)
        chosen_missions.append((pick, drop))

    # Station inventory (random initial counts)
    station_inventory = {s: random.randint(2, 15) for s in stations}

    # Item metadata pool
    item_metadata_pool = [
        {"item_id": "A001", "weight": 1.2, "size": "small"},
        {"item_id": "B999", "weight": 3.4, "size": "medium"},
        {"item_id": "C123", "weight": 0.8, "size": "small"},
        {"item_id": "D777", "weight": 5.0, "size": "large"},
        {"item_id": "ZXY9", "weight": 2.2, "size": "medium"}
    ]

    # ---------- Robot State Initialization ----------
    current_station = 0
    robot_x, robot_y = stations[current_station]
    robot_orientation = 0.0
    robot_speed = 0.0
    packages_delivered = 0

    # Battery / Power
    battery_level_percent = 100.0
    total_energy_consumed = 0.0

    # Temperature
    robot_temperature_C = 30.0

    # For measuring total distance traveled (for energy efficiency)
    total_distance_traveled = 0.0

    # ---------- Missions / Queue -----------
    mission_queue = chosen_missions[:]  # copy
    mission_index = 0
    going_to_pickup = True
    if len(mission_queue) > 0:
        pick_up_station, drop_off_station = mission_queue[mission_index]
        current_item = random.choice(item_metadata_pool)
    else:
        pick_up_station, drop_off_station = (None, None)
        current_item = None

    # For timing
    start_time_sim = time.time()  # overall sim start
    mission_start_time = time.time()

    data_records = []

    # ---------- Simulation Loop -----------
    for t in range(num_time_steps):
        # If out of missions, the robot is idle
        if mission_index >= len(mission_queue):
            robot_speed = 0.0
            target_station = None
            idle_time_step = 1.0
        else:
            target_station = pick_up_station if going_to_pickup else drop_off_station
            idle_time_step = 0.0

        # ---------- Environment (random) ----------
        warehouse_temperature_C = random.uniform(15, 25)
        warehouse_humidity_percent = random.uniform(30, 60)

        # ---------- Obstacle / Unexpected Event ----------
        # Small chance of an obstacle
        obstacle_event = False
        if random.random() < 0.03 and mission_index < len(mission_queue):
            obstacle_event = True
            robot_speed = random.uniform(0.1, 0.3)  # slow down drastically
        else:
            robot_speed = random.uniform(0.5, 1.6)  # normal speed range

        # ---------- LIDAR / Camera / IMU (Random) ----------
        lidar_data = [random.uniform(0.0, 10.0) for _ in range(5)]
        camera_feed = f"frame_{t}.jpg"
        imu_accel = (
            random.uniform(-0.1, 0.1),
            random.uniform(-0.1, 0.1),
            9.8  # approximate gravity
        )
        imu_gyro = (
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01),
            0.0
        )

        # ---------- Orientation & Movement ----------
        old_x, old_y = robot_x, robot_y
        if target_station is not None:
            tx, ty = stations[target_station]
            robot_orientation = math.atan2(ty - robot_y, tx - robot_x)
            dist_to_target = math.hypot(tx - robot_x, ty - robot_y)

            if dist_to_target < robot_speed:
                # Arrive at station
                robot_x, robot_y = tx, ty

                if going_to_pickup:
                    # "Pick up" from station inventory
                    if station_inventory[pick_up_station] > 0:
                        station_inventory[pick_up_station] -= 1
                    going_to_pickup = False
                else:
                    # "Deliver" item
                    station_inventory[drop_off_station] += 1
                    packages_delivered += 1

                    # Mission completed
                    mission_index += 1
                    if mission_index < len(mission_queue):
                        pick_up_station, drop_off_station = mission_queue[mission_index]
                        going_to_pickup = True
                        current_item = random.choice(item_metadata_pool)
                        mission_start_time = time.time()
                    else:
                        current_item = None
            else:
                # Move fraction
                ratio = robot_speed / dist_to_target if dist_to_target > 0 else 0
                robot_x += ratio * (tx - robot_x)
                robot_y += ratio * (ty - robot_y)
        # else: no target => idle

        # ---------- Distance & Speed Stats ----------
        step_distance = math.hypot(robot_x - old_x, robot_y - old_y)
        total_distance_traveled += step_distance

        # ---------- Motor Currents / Wheel Speeds (Mock) ----------
        # For a differential drive with 2 wheels, approximate:
        wheel_speeds = [
            robot_speed + random.uniform(-0.1, 0.1),
            robot_speed + random.uniform(-0.1, 0.1)
        ]
        motor_currents = [abs(ws) * random.uniform(1.0, 2.0) for ws in wheel_speeds]

        # ---------- Robot Temperature Update (Mock) ----------
        if robot_speed > 0:
            robot_temperature_C += random.uniform(0.0, 0.03)  # slight heating
        else:
            robot_temperature_C -= random.uniform(0.0, 0.01)  # cooling if idle
        robot_temperature_C = max(20.0, min(robot_temperature_C, 90.0))

        # ---------- Battery / Power Consumption (Mock) ----------
        # E.g., deplete by a small fraction each step based on speed
        battery_depletion = 0.02 + 0.001 * robot_speed
        battery_level_percent = max(0.0, battery_level_percent - battery_depletion)
        power_consumption_step = battery_depletion * 5.0  # example factor
        total_energy_consumed += power_consumption_step

        # ---------- Fault Codes / Error Logs (Random) ----------
        fault_code = None
        error_log = None
        if random.random() < 0.015:
            fault_code = "F001"
            error_log = "Minor sensor glitch"
        elif random.random() < 0.01:
            fault_code = "F002"
            error_log = "Motor stall detected"

        # ---------- Mission Queue Length ----------
        mission_queue_length = len(mission_queue) - mission_index

        # ---------- Current Delivery Time ----------
        current_delivery_time_sec = None
        if mission_index < len(mission_queue):
            current_delivery_time_sec = time.time() - mission_start_time

        # ---------- Throughput & Energy Efficiency ----------
        elapsed_sim_time = time.time() - start_time_sim
        throughput_packages_per_hour = 0.0
        if elapsed_sim_time > 0:
            throughput_packages_per_hour = (packages_delivered / elapsed_sim_time) * 3600.0

        energy_efficiency_m_per_wh = 0.0
        if total_energy_consumed > 0:
            energy_efficiency_m_per_wh = total_distance_traveled / total_energy_consumed

        # ---------- Record Data for This Step ----------
        record = {
            "time_step": t,
            "timestamp": datetime.now().isoformat(),
            "robot_x": robot_x,
            "robot_y": robot_y,
            "orientation_rad": robot_orientation,
            "speed": robot_speed,
            "packages_delivered": packages_delivered,
            "battery_level_percent": battery_level_percent,
            "power_consumption_step": power_consumption_step,
            "motor_currents": motor_currents,
            "wheel_speeds": wheel_speeds,
            "robot_temperature_C": robot_temperature_C,
            "fault_code": fault_code,
            "error_log": error_log,
            "station_inventory": station_inventory.copy(),
            "warehouse_temperature_C": warehouse_temperature_C,
            "warehouse_humidity_percent": warehouse_humidity_percent,
            "obstacle_event": obstacle_event,
            "item_metadata": current_item,
            "mission_queue_length": mission_queue_length,
            "current_delivery_time_sec": current_delivery_time_sec,
            "lidar_data": lidar_data,
            "camera_feed": camera_feed,
            "imu_accel": imu_accel,
            "imu_gyro": imu_gyro,
            "idle_time_step": idle_time_step,
            "throughput_packages_per_hour": throughput_packages_per_hour,
            "energy_efficiency_m_per_wh": energy_efficiency_m_per_wh
        }

        data_records.append(record)

    return data_records, stations


##################################
# 2. Animation / Plotting Method #
##################################
def animate_simulation(data_records, stations, interval=300):
    """
    Animate the robot's path through the warehouse and display some side info in real-time.
    We won't display *all* fields (to keep it readable), but they're all in data_records.
    :param data_records: List of dicts containing extended robot data (including positions).
    :param stations: Dict of station_id -> (x, y)
    :param interval: Delay between frames in milliseconds.
    """

    # Prepare figure with two subplots:
    #  - ax_map: the warehouse map with stations & robot path
    #  - ax_info: text area for real-time updates
    fig, (ax_map, ax_info) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the warehouse stations on ax_map
    station_x = [coord[0] for coord in stations.values()]
    station_y = [coord[1] for coord in stations.values()]
    ax_map.scatter(station_x, station_y, c='black', marker='s', s=50, label="Stations")
    ax_map.set_title("Warehouse Map")
    ax_map.set_xlabel("X position")
    ax_map.set_ylabel("Y position")

    # Auto-scale to fit all stations nicely, with some margin
    min_x = min(station_x) - 2
    max_x = max(station_x) + 2
    min_y = min(station_y) - 2
    max_y = max(station_y) + 2
    ax_map.set_xlim(min_x, max_x)
    ax_map.set_ylim(min_y, max_y)
    ax_map.legend()

    # Robot path line & point
    robot_path_x = []
    robot_path_y = []
    path_line, = ax_map.plot([], [], 'r--', lw=1)         # path history
    robot_point, = ax_map.plot([], [], 'ro', markersize=8)  # current position

    # Set up the info subplot
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    ax_info.set_title("Robot Info")

    # We'll create a Text object that we update each frame
    info_text = ax_info.text(0.05, 0.95, "", transform=ax_info.transAxes,
                             fontsize=12, va='top')

    def init():
        path_line.set_data([], [])
        robot_point.set_data([], [])
        info_text.set_text("")
        return path_line, robot_point, info_text

    def update(frame_index):
        if frame_index >= len(data_records):
            return path_line, robot_point, info_text

        rec = data_records[frame_index]
        rx, ry = rec["robot_x"], rec["robot_y"]

        robot_path_x.append(rx)
        robot_path_y.append(ry)

        # Update path line and robot point
        path_line.set_data(robot_path_x, robot_path_y)
        robot_point.set_data([rx], [ry])  # must be sequences

        # Build info string (subset of fields for readability)
        info_str = (
            f"Time Step: {rec['time_step']}\n"
            f"Timestamp: {rec['timestamp']}\n"
            f"Pos: ({rx:.2f}, {ry:.2f})\n"
            f"Speed: {rec['speed']:.2f} m/s\n"
            f"Battery: {rec['battery_level_percent']:.1f}%\n"
            f"Packages Delivered: {rec['packages_delivered']}\n"
            f"Mission Queue Length: {rec['mission_queue_length']}\n"
            f"Obstacle?: {rec['obstacle_event']}\n"
        )

        # Show item metadata if any
        if rec["item_metadata"] is not None:
            item_md = rec["item_metadata"]
            info_str += (
                f"Item: {item_md['item_id']} "
                f"({item_md['weight']}kg, {item_md['size']})\n"
            )

        info_text.set_text(info_str)

        return path_line, robot_point, info_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(data_records),
        init_func=init, interval=interval, blit=False
    )

    plt.tight_layout()
    plt.show()


##################################
# 3. Main: Putting It Together   #
##################################
def main():
    # 1) Generate extended synthetic data:
    data_records, stations = generate_synthetic_data_extended(
        num_time_steps=600,  # simulate 600 steps
        max_missions=10      # more missions
    )

    # 2) Animate the robot:
    animate_simulation(data_records, stations, interval=300)

    # 3) (Optional) Write to CSV if desired:
    # fieldnames we want in the CSV (all fields you listed)
    fieldnames = [
        "time_step",
        "timestamp",
        "robot_x",
        "robot_y",
        "orientation_rad",
        "speed",
        "packages_delivered",
        "battery_level_percent",
        "power_consumption_step",
        "motor_currents",
        "wheel_speeds",
        "robot_temperature_C",
        "fault_code",
        "error_log",
        "station_inventory",
        "warehouse_temperature_C",
        "warehouse_humidity_percent",
        "obstacle_event",
        "item_metadata",
        "mission_queue_length",
        "current_delivery_time_sec",
        "lidar_data",
        "camera_feed",
        "imu_accel",
        "imu_gyro",
        "idle_time_step",
        "throughput_packages_per_hour",
        "energy_efficiency_m_per_wh"
    ]

    with open("extended_robot_data.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in data_records:
            # Convert complex types to strings for CSV
            row = {}
            for k in fieldnames:
                val = rec[k]
                if isinstance(val, (list, dict, tuple)):
                    val = str(val)
                row[k] = val
            writer.writerow(row)

if __name__ == "__main__":
    main()
