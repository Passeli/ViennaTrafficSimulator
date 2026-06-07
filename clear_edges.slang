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

struct PushConstants {
  float dt;
  uint num_cars;
  uint num_edges;
  uint padding;
};

[[vk::binding(1, 0)]]
RWStructuredBuffer<Edge> edges;
[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID) {
  uint id = threadId.x;
  if (id < pc.num_edges) {
    edges[id].head_car_idx = -1;
    edges[id].garage_lock = -1;
  }
}
