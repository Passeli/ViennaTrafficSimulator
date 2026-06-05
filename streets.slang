struct Node { float x; float y; int lock; int padding; };
struct Edge { int start_node_idx; int end_node_idx; int next_edge_idx; float length; float max_speed; int head_car_idx; int p1; int p2; };

[[vk::binding(0, 0)]] RWStructuredBuffer<Node> nodes;
[[vk::binding(1, 0)]] RWStructuredBuffer<Edge> edges;

struct GraphicsPushData {
    float min_x; float max_x; float min_y; float max_y;
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

    float nx = ((n.x - pc.min_x) / (pc.max_x - pc.min_x)) * 2.0 - 1.0;
    float ny = ((n.y - pc.min_y) / (pc.max_y - pc.min_y)) * 2.0 - 1.0;

    VertexOutput output;
    output.position = float4(nx, -ny, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fragmentMain() : SV_Target {
    return float4(0.2, 0.2, 0.25, 1.0); // A nice dim asphalt gray
}