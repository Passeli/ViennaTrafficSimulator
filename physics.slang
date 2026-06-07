struct Node {
  float x;
  float y;
  int lock;
  int padding;
};

struct Edge {
  int start_node_idx;
  int end_node_idx;
  int next_edge_idx;
  float length;
  float max_speed;
  int head_car_idx;
  int garage_lock;
  int p1;
};

struct Car {
  int current_edge_idx;
  float position;
  float speed;
  int state;
  int next_car_idx;
  int p1;
  int p2;
  int p3;
};

struct PushConstants {
  float dt;
  uint num_cars;
  uint num_edges;
  uint padding;
};

[[vk::binding(0, 0)]]
RWStructuredBuffer<Node> nodes;
[[vk::binding(1, 0)]]
RWStructuredBuffer<Edge> edges;
[[vk::binding(2, 0)]]
RWStructuredBuffer<Car> cars;
[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID) {
  uint id = threadId.x;
  if (id >= pc.num_cars)
    return;

  Car car = cars[id];
  if (car.state == 2)
    return; // Car has already safely evacuated the city

  Edge edge = edges[car.current_edge_idx];

  // --- PHYSICAL CONSTANTS ---
  float CAR_LENGTH = 4.5;
  float MIN_GAP = 0.5;
  float TIME_GAP = 1.5;

  // ==========================================
  // PHASE 1: DRIVING STATE (State 0)
  // ==========================================
  if (car.state == 0) {
    float front_distance = edge.length - car.position;
    float front_speed = 0.0;
    bool is_lead_car = true;

    // Traverse the spatial linked list to check for vehicles ahead of us
    int current_search_id = edge.head_car_idx;
    while (current_search_id != -1) {
      if ((uint)current_search_id != id) {
        Car other_car = cars[current_search_id];

        // Ensure the car hasn't secretly moved to a new edge this frame!
        if (other_car.current_edge_idx == car.current_edge_idx) {
          if (other_car.state == 0 || other_car.state == 1) {
            float dist_to_other_front = other_car.position - car.position;

            // --- FIX 1: THE TIE BREAKER ---
            // If cars do manage to perfectly overlap, mathematically force one
            // to be "in front" so they don't ignore each other!
            if (dist_to_other_front > 0.0 ||
                (dist_to_other_front == 0.0 && (uint)current_search_id > id)) {

              // Calculate gap from our front bumper to their rear bumper
              float true_gap = dist_to_other_front - CAR_LENGTH;

              // Keep track of the closest vehicle directly in front of us
              if (true_gap < front_distance) {
                front_distance = true_gap;
                front_speed = other_car.speed;
                is_lead_car = false;
              }
            }
          }
        }
      }
      current_search_id = cars[current_search_id].next_car_idx;
    }

    // --- TARGET SELECTION ---
    float target_gap = is_lead_car ? 0.0 : MIN_GAP;
    float safe_dist = target_gap + (car.speed * TIME_GAP);

    // --- SENSOR LOOP TRAP (Lead Car Only) ---
    if (is_lead_car && (car.position + 0.1 >= edge.length)) {
      car.position = edge.length; // Snap perfectly to the stop line
      car.speed = 0.0;
      car.state = 1; // Turn RED and request intersection lock
    } else {
      // Intelligent Driver Model (IDM)
      float accel = 0.0;
      if (front_distance < safe_dist) {
        float speed_diff = car.speed - front_speed;
        accel = -1.0 * (safe_dist / max(front_distance, 0.1)) *
                max(speed_diff, 1.0);
      } else {
        float speed_ratio = car.speed / edge.max_speed;
        accel = 2.0 * (1.0 - pow(speed_ratio, 4.0));
      }

      accel = clamp(accel, -5.0, 3.0);
      car.speed = max(0.0, car.speed + accel * pc.dt);

      // --- FIX 2: THE ANTI-TUNNELING CLAMP ---
      float intended_movement = car.speed * pc.dt;

      // Mathematically prevent the car from moving past the bumper of the car
      // in front of it! We leave a 0.01m micro-gap to prevent true 0.0
      // overlapping floats.
      float max_allowed_movement = max(0.0, front_distance - 0.01);

      car.position += min(intended_movement, max_allowed_movement);
    }
  }

  // ==========================================
  // PHASE 2: INTERSECTION SCHEDULING (State 1)
  // ==========================================
  else if (car.state == 1) {
    int target_node = edge.end_node_idx;
    int next_edge = edge.next_edge_idx;

    // Perform an atomic lock check on the physical intersection lane
    int original_lock;
    InterlockedCompareExchange(nodes[target_node].lock, -1, (int)id,
                               original_lock);

    if (original_lock == -1 || original_lock == 0 || original_lock == (int)id ||
        pc.num_cars == 1) {

      if (next_edge == -1) {
        // EXITED MAP
        car.state = 2;
        int dummy;
        InterlockedExchange(nodes[target_node].lock, -1, dummy); // Flush lock
      } else {
        // --- SPILLBACK CHECK (Entrance Clear) ---
        bool entrance_clear = true;
        int next_search_id = edges[next_edge].head_car_idx;

        while (next_search_id != -1) {
          Car next_car = cars[next_search_id];
          if (next_car.state == 0 || next_car.state == 1) {
            // Check where their rear bumper is relative to the start of the
            // street (0.0)
            float tail_gap = next_car.position - CAR_LENGTH;
            if (tail_gap < MIN_GAP) {
              entrance_clear = false;
              break;
            }
          }
          next_search_id = cars[next_search_id].next_car_idx;
        }

        if (entrance_clear) {
          // Route the car onto its new street assignment
          car.current_edge_idx = next_edge;
          car.position = 0.0; // Front bumper hits the new street
          car.speed = 0.0;
          car.state = 0; // Return to open roadway physics engine

          int dummy;
          InterlockedExchange(nodes[target_node].lock, -1, dummy); // Flush lock
        }
      }
    }
  }

  // ==========================================
  // PHASE 3: GARAGE INITIALIZATION (State 3)
  // ==========================================
  else if (car.state == 3) {

    // --- ATOMIC GARAGE DOOR LOCK ---
    int original_lock;
    InterlockedCompareExchange(edges[car.current_edge_idx].garage_lock, -1,
                               (int)id, original_lock);

    if (original_lock == -1 || original_lock == (int)id) {

      bool entrance_clear = true;
      int next_search_id = edge.head_car_idx;

      while (next_search_id != -1) {
        Car next_car = cars[next_search_id];

        if (next_car.current_edge_idx == car.current_edge_idx &&
            (next_car.state == 0 || next_car.state == 1)) {
          float tail_gap = next_car.position - CAR_LENGTH;
          if (tail_gap < 6.0) { // Require 6 meters of space to pull out
            entrance_clear = false;
            break;
          }
        }
        next_search_id = cars[next_search_id].next_car_idx;
      }

      if (entrance_clear) {
        car.state = 0;
        car.position = 0.0;
        car.speed = 0.0;
      }
    }
  }

  // Commit local registry records back into global GPU VRAM
  cars[id] = car;
}
