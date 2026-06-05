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
    float camera_x; float camera_y;
    float zoom_level; float aspect_ratio;
    float extent_width; float extent_height;
    float p1; float p2;
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

    // 1. Shift coordinate relative to camera position (Pan)
    float local_x = car_x - pc.camera_x;
    float local_y = car_y - pc.camera_y;

    // 2. Scale coordinate based on viewport dimension boundaries (Zoom)
    // We scale relative to the bounding width of Vienna
    float nx = (local_x * pc.zoom_level) / (pc.extent_width * 0.5);
    float ny = (local_y * pc.zoom_level) / (pc.extent_height * 0.5);

    // Apply aspect ratio correction to prevent non-square distortion
    nx /= pc.aspect_ratio;

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