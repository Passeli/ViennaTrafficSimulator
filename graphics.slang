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

[[vk::binding(0, 0)]] RWStructuredBuffer<Node> nodes;
[[vk::binding(1, 0)]] RWStructuredBuffer<Edge> edges;
[[vk::binding(2, 0)]] RWStructuredBuffer<Car> cars;

struct GraphicsPushData {
    float min_x;
    float max_x;
    float min_y;
    float max_y;
    float aspect_ratio;
    float p1;
    float p2;
    float p3; // <-- ADDED
};
[[vk::push_constant]] ConstantBuffer<GraphicsPushData> pc;

struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR;
};

// ==========================================
// VERTEX SHADER: Simplified & Optimized
// ==========================================
[shader("vertex")]
VertexOutput vertexMain(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID) {
    Car car = cars[instanceID];

    // --- EARLY EXIT (GPU Optimization) ---
    // If the car is evacuated (2) or in the garage (3), do no math.
    // Just collapse the vertex far off-screen and abort.
    if (car.state >= 2) {
        VertexOutput output;
        output.position = float4(2.0, 2.0, 0.0, 1.0);
        output.color = float3(0.0, 0.0, 0.0);
        return output;
    }

    Edge edge = edges[car.current_edge_idx];
    Node start_node = nodes[edge.start_node_idx];
    Node end_node = nodes[edge.end_node_idx];

    // Interpolate world position
    float t = edge.length > 0.0 ? (car.position / edge.length) : 0.0;
    float car_x = lerp(start_node.x, end_node.x, t);
    float car_y = lerp(start_node.y, end_node.y, t);

    // Convert to Screen Coordinates (NDC)
    float nx = ((car_x - pc.min_x) / (pc.max_x - pc.min_x)) * 2.0 - 1.0;
    float ny = ((car_y - pc.min_y) / (pc.max_y - pc.min_y)) * 2.0 - 1.0;

    // --- SIMPLIFIED COLOR LOGIC ---
    // 0 = Green (Driving), 1 = Red (Queuing)
    float3 debug_color = (car.state == 0) ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);

    // --- MESH OFFSETS (FRONT BUMPER ALIGNED) ---
    // Because car.position is the FRONT bumper, we draw the rectangle trailing backwards
    // Note how the X values are -2 and 0 (instead of -1 and 1)
    float size = 0.001;
    float2 offsets[6] = {
        float2(-2, -1), float2(0, -1), float2(-2, 1),
        float2( 0, -1), float2(0,  1), float2(-2, 1)
    };

    // Aspect ratio correction (1920x1080)
    float aspect = pc.aspect_ratio; // <-- UPDATED
    float rx = (offsets[vertexID].x * size) / aspect;
    float ry = offsets[vertexID].y * size;

    VertexOutput output;
    output.position = float4(nx + rx, -(ny + ry), 0.0, 1.0);
    output.color = debug_color;

    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    return float4(input.color, 1.0);
}