struct Node { float x; float y; int lock; int padding; };
struct Edge { int start_node_idx; int end_node_idx; int next_edge_idx; float length; float max_speed; int head_car_idx; int p1; int p2; };

[[vk::binding(0, 0)]] RWStructuredBuffer<Node> nodes;
[[vk::binding(1, 0)]] RWStructuredBuffer<Edge> edges;

// --- MATCHED TO C++ EXACTLY ---
struct GraphicsPushData {
    float camera_x;
    float camera_y;
    float zoom_level;
    float aspect_ratio;
    float extent_width;
    float extent_height;
    float p1;
    float p2;
};
[[vk::push_constant]] ConstantBuffer<GraphicsPushData> pc;

struct VertexOutput {
    float4 position : SV_Position;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID) {
    Edge edge = edges[instanceID];

    // vertexID 0 gets the start node, vertexID 1 gets the end node
    Node n = (vertexID == 0) ? nodes[edge.start_node_idx] : nodes[edge.end_node_idx];

    // 1. Shift coordinate relative to camera position (Pan)
    float local_x = n.x - pc.camera_x;
    float local_y = n.y - pc.camera_y;

    // 2. Scale coordinate based on viewport dimensions (Zoom)
    float nx = (local_x * pc.zoom_level) / (pc.extent_width * 0.5);
    float ny = (local_y * pc.zoom_level) / (pc.extent_height * 0.5);

    // 3. Apply aspect ratio correction
    nx /= pc.aspect_ratio;

    VertexOutput output;
    // -ny inverts the Y axis so North is UP
    output.position = float4(nx, -ny, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fragmentMain() : SV_Target {
    return float4(0.2, 0.2, 0.25, 1.0); // A nice dim asphalt gray
}