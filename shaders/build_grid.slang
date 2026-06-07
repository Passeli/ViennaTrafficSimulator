#include "common.slang"

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

  cars[id].next_car_idx = -1;

  if (cars[id].state == CarState::Evacuated ||
      cars[id].state == CarState::Garage) {
    return;
  }

  int edge_idx = cars[id].current_edge_idx;

  // Atomically insert this car at the head of the street's linked list
  int original_head;
  InterlockedExchange(edges[edge_idx].head_car_idx, (int)id, original_head);

  cars[id].next_car_idx = original_head;
}
