struct Edge {
    int start_node_idx;
    int end_node_idx;
    int next_edge_idx;
    float length;
    float max_speed;
    int head_car_idx;
    int p1;
    int p2;
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

[[vk::binding(1, 0)]] RWStructuredBuffer<Edge> edges;
[[vk::binding(2, 0)]] RWStructuredBuffer<Car> cars;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    uint id = threadId.x;
    if (id >= pc.num_cars) return;

    cars[id].next_car_idx = -1; // Clear old pointer

    if (cars[id].state == 2 || cars[id].state == 3) return; // Don't track exited or garaged cars

    int edge_idx = cars[id].current_edge_idx;

    // Atomically insert this car at the head of the street's linked list
    int original_head;
    InterlockedExchange(edges[edge_idx].head_car_idx, (int)id, original_head);

    cars[id].next_car_idx = original_head;
}