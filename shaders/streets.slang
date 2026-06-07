#include "common.slang"

[[vk::binding(0, 0)]]
RWStructuredBuffer<Node> nodes;
[[vk::binding(1, 0)]]
RWStructuredBuffer<Edge> edges;
[[vk::push_constant]]
ConstantBuffer<GraphicsConstants> pc;

struct VertexOutput {
  float4 position : SV_Position;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexID: SV_VertexID,
                        uint instanceID: SV_InstanceID) {
  Edge edge = edges[instanceID];

  // vertexID 0 gets the start node, vertexID 1 gets the end node
  Node n =
      (vertexID == 0) ? nodes[edge.start_node_idx] : nodes[edge.end_node_idx];

  // 1. Pan and Zoom Projection
  float local_x = (n.x - pc.camera_x) * pc.zoom_level;
  float local_y = (n.y - pc.camera_y) * pc.zoom_level;

  float nx = local_x / (pc.extent_width * 0.5) / pc.aspect_ratio;
  float ny = local_y / (pc.extent_height * 0.5);

  VertexOutput output;
  // -ny inverts the Y axis so North is UP
  output.position = float4(nx, -ny, 0.0, 1.0);
  return output;
}

[shader("fragment")]
float4 fragmentMain() : SV_Target {
  return float4(0.2, 0.2, 0.25, 1.0); // A nice dim asphalt gray
}
