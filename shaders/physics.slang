#include "common.slang"

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
  const uint id = threadId.x;
  if (id >= pc.num_cars)
    return;

  Car car = cars[id];

  if (car.state == CarState::Evacuated)
    return;

  const Edge edge = edges[car.current_edge_idx];

  // --- PHYSICAL CONSTANTS ---
  const float CAR_LENGTH = 4.5;
  const float MIN_GAP = 0.5;
  const float TIME_GAP = 1.5;

  // ==========================================
  // PHASE 1: DRIVING STATE (CarState::Driving)
  // ==========================================
  if (car.state == CarState::Driving) {
    float front_distance = edge.length - car.position;
    float front_speed = 0.0;
    bool is_lead_car = true;

    // Traverse the spatial linked list
    int current_search_id = edge.head_car_idx;
    int closest_front_car_id = -1;
    while (current_search_id != -1) {
      if ((uint)current_search_id != id) {
        Car other_car = cars[current_search_id];

        if (other_car.current_edge_idx == car.current_edge_idx) {
          if (other_car.state == CarState::Driving ||
              other_car.state == CarState::Queuing ||
              other_car.state == CarState::Stuck) {
            float dist_to_other_front = other_car.position - car.position;

            if (dist_to_other_front > 0.0 ||
                (dist_to_other_front == 0.0 && (uint)current_search_id > id)) {

              float true_gap = dist_to_other_front - CAR_LENGTH;

              if (true_gap < front_distance) {
                front_distance = true_gap;
                front_speed = other_car.speed;
                is_lead_car = false;
                closest_front_car_id = current_search_id;
              }
            }
          }
        }
      }
      current_search_id = cars[current_search_id].next_car_idx;
    }

    float target_gap = is_lead_car ? 0.0 : MIN_GAP;

    if (is_lead_car && (car.position + 0.1 >= edge.length)) {
      car.position = edge.length; // Snap perfectly to the stop line
      car.speed = 0.0;
      if (nodes[edge.end_node_idx].type == NodeType::Blind) {
        car.state = CarState::Stuck;
      } else {
        car.state = CarState::Queuing;
      }
    } else if (!is_lead_car && closest_front_car_id != -1 &&
               cars[closest_front_car_id].state == CarState::Stuck &&
               (front_distance <= MIN_GAP + 0.1)) {
      car.position = cars[closest_front_car_id].position - CAR_LENGTH - MIN_GAP;
      car.speed = 0.0;
      car.state = CarState::Stuck;
    } else {
      // Official Intelligent Driver Model (IDM)
      float speed_diff = is_lead_car ? car.speed : (car.speed - front_speed);

      // Snappy parameters for static stop lines vs safe parameters for followers
      // T is safe time headway (seconds)
      float T = is_lead_car ? 0.6 : TIME_GAP;
      // denom is 2 * sqrt(a * b) where desired acceleration a = 1.5 m/s^2.
      // Lead cars: b = 3.5 m/s^2 (firm braking) -> 2 * sqrt(1.5 * 3.5) = 4.5825
      // Followers: b = 2.0 m/s^2 (comfortable braking) -> 2 * sqrt(1.5 * 2.0) = 3.4641
      float denom = is_lead_car ? 4.5825 : 3.4641;

      float dynamic_gap = target_gap + (car.speed * T) +
                          (car.speed * speed_diff) / denom;
      dynamic_gap = max(dynamic_gap, target_gap);

      float speed_ratio = car.speed / edge.max_speed;
      float accel = 1.5 * (1.0 - pow(speed_ratio, 4.0) -
                           pow(dynamic_gap / max(front_distance, 0.1), 2.0));

      accel = clamp(accel, -5.0, 3.0);
      car.speed = max(0.0, car.speed + accel * pc.dt);

      car.position += car.speed * pc.dt;
    }
  }

  // ==========================================
  // PHASE 2: INTERSECTION SCHEDULING (CarState::Queuing)
  // ==========================================
  else if (car.state == CarState::Queuing) {
    int target_node = edge.end_node_idx;
    int next_edge = edge.next_edge_idx;

    int original_lock;
    InterlockedCompareExchange(nodes[target_node].lock, -1, (int)id,
                               original_lock);

    if (original_lock == -1 || original_lock == (int)id) {
      if (nodes[target_node].type == NodeType::OpenExit) {
        car.state = CarState::Evacuated;
        int dummy = 0;
        InterlockedExchange(nodes[target_node].lock, -1, dummy);
      } else if (next_edge != -1) {
        bool entrance_clear = true;
        bool blocked_by_stuck = false;
        int next_search_id = edges[next_edge].head_car_idx;

        while (next_search_id != -1) {
          Car next_car = cars[next_search_id];
          if (next_car.state == CarState::Driving ||
              next_car.state == CarState::Queuing ||
              next_car.state == CarState::Stuck) {
            float tail_gap = next_car.position - CAR_LENGTH;
            if (tail_gap < MIN_GAP) {
              entrance_clear = false;
              if (next_car.state == CarState::Stuck) {
                blocked_by_stuck = true;
              }
              break;
            }
          }
          next_search_id = cars[next_search_id].next_car_idx;
        }

        if (entrance_clear) {
          car.current_edge_idx = next_edge;
          car.position = 0.0;
          car.speed = 0.0;
          car.state = CarState::Driving;
          int dummy = 0;
          InterlockedExchange(nodes[target_node].lock, -1, dummy);
        } else {
          if (blocked_by_stuck) {
            car.state = CarState::Stuck;
          }
          int dummy = 0;
          InterlockedExchange(nodes[target_node].lock, -1, dummy);
        }
      } else {
        // No path to exit (e.g. exit closed or all exits closed)
        int dummy = 0;
        InterlockedExchange(nodes[target_node].lock, -1, dummy);
      }
    }
  }

  // ==========================================
  // PHASE 3: GARAGE INITIALIZATION (CarState::Garage)
  // ==========================================
  else if (car.state == CarState::Garage) {
    int original_lock;
    InterlockedCompareExchange(edges[car.current_edge_idx].garage_lock, -1,
                               (int)id, original_lock);

    if (original_lock == -1 || original_lock == (int)id) {
      bool entrance_clear = true;
      int next_search_id = edge.head_car_idx;

      while (next_search_id != -1) {
        Car next_car = cars[next_search_id];
        if (next_car.current_edge_idx == car.current_edge_idx &&
            (next_car.state == CarState::Driving ||
             next_car.state == CarState::Queuing ||
             next_car.state == CarState::Stuck)) {
          float tail_gap = next_car.position - CAR_LENGTH;
          if (tail_gap < 5.0) {
            entrance_clear = false;
            break;
          }
        }
        next_search_id = cars[next_search_id].next_car_idx;
      }

      if (entrance_clear) {
        car.state = CarState::Driving;
        car.position = 0.0;
        car.speed = 0.0;
      }
    }
  }
  cars[id] = car;
}
