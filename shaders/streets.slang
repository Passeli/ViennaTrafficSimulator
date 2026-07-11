#include "common.slang"

[[vk::binding(0, 0)]]
RWStructuredBuffer<Node> nodes;
[[vk::binding(1, 0)]]
RWStructuredBuffer<Edge> edges;
[[vk::push_constant]]
ConstantBuffer<GraphicsConstants> pc;

struct VertexOutput {
  float4 position : SV_Position;
  float3 color : COLOR;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexID: SV_VertexID,
                        uint instanceID: SV_InstanceID) {
  const Edge edge = edges[instanceID];

  // vertexID 0 gets the start node, vertexID 1 gets the end node
  const Node n =
      (vertexID == 0) ? nodes[edge.start_node_idx] : nodes[edge.end_node_idx];

  const float local_x = (n.x - pc.camera_x) * pc.zoom_level;
  const float local_y = (n.y - pc.camera_y) * pc.zoom_level;

  const float nx = local_x / (pc.extent_height * 0.5) / pc.aspect_ratio;
  const float ny = local_y / (pc.extent_height * 0.5);

  // color based on population density (spawn_capacity)
  const float density = clamp(float(edge.spawn_capacity) / 50., 0., 1.);

  // color scale: from dim asphalt gray (empty) to bright magenta/pink
  const float3 low_density_color = float3(0.2, 0.2, 0.25);
  const float3 high_density_color = float3(1., 0.2, 0.6);
  const float3 edge_color =
      lerp(low_density_color, high_density_color, density);

  VertexOutput output;
  output.position = float4(nx, -ny, 0., 1.);
  output.color = edge_color;
  return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
  return float4(input.color, 1.);
}
