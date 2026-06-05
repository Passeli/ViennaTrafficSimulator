import math
import json
import sys

class Node:
    def __init__(self, node_id, next_node_to_exit):
        self.id = node_id
        self.next_node_to_exit = next_node_to_exit
        self.locked_by_car = -1

class Edge:
    def __init__(self, edge_id, start_node, end_node, length, max_speed):
        self.id = edge_id
        self.start_node = start_node
        self.end_node = end_node
        self.length = length
        self.max_speed = max_speed
        self.cars_on_edge = []
        self.spawn_queue = []

class Car:
    def __init__(self, car_id, start_edge, start_pos):
        self.id = car_id
        self.current_edge = start_edge
        self.position = start_pos  # Represents the FRONT BUMPER
        self.speed = 0.0
        self.state = 3  # 3 = In Garage

class EvacuationSim:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.cars = {}
        self.time = 0.0
        self.ticks = 0  # <--- Added Integer Tick Tracker

        # --- PHYSICAL CONSTANTS ---
        self.CAR_LENGTH = 4.5
        self.MIN_GAP = 0.5

    def append_car_to_edge(self, car_id, edge_id):
        self.edges[edge_id].cars_on_edge.append(car_id)

    def tick(self, dt=0.1):
        self.time += dt
        self.ticks += 1

        # ==========================================
        # 1. SPAWN STEP
        # ==========================================
        for edge_id, edge in self.edges.items():
            if edge.spawn_queue:
                safe_to_spawn = True

                if edge.cars_on_edge:
                    last_car_id = edge.cars_on_edge[-1]
                    last_car = self.cars[last_car_id]
                    tail_distance = (last_car.position - self.CAR_LENGTH) - 0.0
                    if tail_distance < self.MIN_GAP:
                        safe_to_spawn = False

                if safe_to_spawn:
                    new_car_id = edge.spawn_queue.pop(0)
                    new_car = self.cars[new_car_id]
                    new_car.state = 0
                    new_car.position = 0.0
                    new_car.speed = 0.0
                    self.append_car_to_edge(new_car_id, edge_id)

        # ==========================================
        # 2. PHYSICS STEP
        # ==========================================
        for edge_id, edge in self.edges.items():
            for i, car_id in enumerate(edge.cars_on_edge):
                car = self.cars[car_id]
                if car.state != 0:
                    continue

                time_gap = 1.5
                # 1. Check if we are following a car, or looking at an intersection
                if i > 0:
                    # Following a car: Target a 5.0m bumper-to-bumper gap
                    front_car = self.cars[edge.cars_on_edge[i - 1]]
                    front_distance = (front_car.position - self.CAR_LENGTH) - car.position
                    front_speed = front_car.speed
                    target_gap = self.MIN_GAP  # 0.5 meters
                else:
                    # Looking at the intersection: Pull right up to the line (0.0m gap)
                    front_distance = edge.length - car.position
                    front_speed = 0.0
                    target_gap = 0.0  # Stop perfectly at the line!

                # IDM calculates braking based on the specific target gap
                safe_dist = target_gap + (car.speed * time_gap)

                # 2. THE SENSOR LOOP (Lead Car Only)
                # ONLY the lead car (i == 0) triggers the intersection.
                # Followers remain in State 0 (driving) and let IDM handle the traffic jam.
                if i == 0 and (car.position + 0.1 >= edge.length):
                    car.position = edge.length - target_gap # Snap perfectly to the target
                    car.speed = 0.0
                    car.state = 1 # Enter Intersection Queue
                else:
                    if front_distance < safe_dist:
                        speed_diff = car.speed - front_speed
                        accel = -1.0 * (safe_dist / max(front_distance, 0.1)) * max(speed_diff, 1.0)
                    else:
                        speed_ratio = car.speed / edge.max_speed
                        accel = 2.0 * (1.0 - math.pow(speed_ratio, 4.0))

                    accel = max(-5.0, min(accel, 3.0))

                    car.speed = max(0.0, car.speed + accel * dt)
                    car.position += car.speed * dt

                # --- BULLETPROOF TELEMETRY (Using integer ticks) ---
                if self.ticks % 10 == 0:  # Exactly every 1.0 seconds
                    print(f"T: {self.time:5.1f}s | Edge: {edge_id:15} | Pos: {car.position:6.2f}m / {edge.length:6.2f}m | "
                          f"Spd: {car.speed:5.2f} m/s | SafeDist: {safe_dist:5.2f}m | "
                          f"DistToObstacle: {front_distance:5.2f}m | State: {car.state}")

        # ==========================================
        # 3. INTERSECTION SCHEDULING STEP
        # ==========================================
        for edge_id, edge in self.edges.items():
            if not edge.cars_on_edge:
                continue

            lead_car_id = edge.cars_on_edge[0]
            lead_car = self.cars[lead_car_id]

            if lead_car.state == 1:
                target_node = self.nodes[edge.end_node]

                if target_node.locked_by_car == -1 or target_node.locked_by_car == lead_car.id:
                    target_node.locked_by_car = lead_car.id
                    next_node_id = target_node.next_node_to_exit

                    if next_node_id == -1:
                        print(f"\n🎉 EXITED MAP! Car successfully evacuated at Node {target_node.id} at T={self.time:.1f}s!")
                        lead_car.state = 2
                        edge.cars_on_edge.pop(0)
                        target_node.locked_by_car = -1
                    else:
                        next_edge_id = f"{target_node.id}_{next_node_id}"

                        if next_edge_id in self.edges:
                            next_edge = self.edges[next_edge_id]

                            entrance_clear = True
                            if next_edge.cars_on_edge:
                                tail_car_id = next_edge.cars_on_edge[-1]
                                tail_car = self.cars[tail_car_id]
                                tail_gap = tail_car.position - self.CAR_LENGTH
                                if tail_gap < self.MIN_GAP:
                                    entrance_clear = False

                            if entrance_clear:
                                print(f"🔄 TURN: Car transferring from {edge_id} -> {next_edge_id}")
                                edge.cars_on_edge.pop(0)
                                lead_car.current_edge = next_edge_id
                                lead_car.position = 0.0
                                lead_car.speed = 0.0
                                lead_car.state = 0

                                self.append_car_to_edge(lead_car.id, next_edge_id)
                                target_node.locked_by_car = -1

def load_simulation_data(sim, nodes_file, edges_file):
    print(f"Loading Real Vienna Map Data...")
    try:
        with open(nodes_file, 'r', encoding='utf-8') as f:
            nodes_data = json.load(f)
        with open(edges_file, 'r', encoding='utf-8') as f:
            edges_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find files.")
        sys.exit(1)

    for n in nodes_data:
        sim.nodes[n['id']] = Node(n['id'], n['next_node_to_exit'])

    valid_start_edge = None

    for e in edges_data:
        edge_id = f"{e['u']}_{e['v']}"
        if edge_id in sim.edges:
            continue

        sim.edges[edge_id] = Edge(
            edge_id=edge_id,
            start_node=e['u'],
            end_node=e['v'],
            length=float(e['length']),
            max_speed=float(e['max_speed_ms'])
        )

        if not valid_start_edge and float(e['length']) > 50.0:
            target_node = sim.nodes[e['v']]
            if target_node.next_node_to_exit != -1:
                valid_start_edge = edge_id

    print(f"✅ Loaded {len(sim.nodes)} intersections and {len(sim.edges)} streets.")

    car = Car(car_id=0, start_edge=valid_start_edge, start_pos=0.0)
    sim.cars[0] = car
    sim.edges[valid_start_edge].spawn_queue.append(0)

    print(f"🚗 Spawning 1 Test Car on Edge: {valid_start_edge}")

def main():
    sim = EvacuationSim()
    load_simulation_data(sim, "vienna_nodes.json", "vienna_edges.json")

    print("\n--- STARTING FULL CITY EVACUATION ---")

    # INCREASED TIME LIMIT: 36,000 ticks = 1 Hour of simulated driving
    for _ in range(36000):
        sim.tick(dt=0.1)
        if sim.cars[0].state == 2:
            break

    if sim.cars[0].state != 2:
        print("\n❌ SIMULATION PAUSED: Car is stuck, or 1 hour was not enough time to evacuate!")

if __name__ == "__main__":
    main()