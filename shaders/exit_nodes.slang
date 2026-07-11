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
  const Node n = nodes[edge.end_node_idx];

  if (n.type != NodeType::OpenExit && n.type != NodeType::ClosedExit) {
    VertexOutput output;
    output.position = float4(2., 2., 0., 1.);
    output.color = float3(0., 0., 0.);
    return output;
  }

  const float local_x = (n.x - pc.camera_x) * pc.zoom_level;
  const float local_y = (n.y - pc.camera_y) * pc.zoom_level;

  const float nx = local_x / (pc.extent_height * 0.5) / pc.aspect_ratio;
  const float ny = local_y / (pc.extent_height * 0.5);

  // colors: green for open exits, red for closed exits, yellow for blind nodes
  float3 node_color = float3(1., 0., 0.);
  if (n.type == NodeType::OpenExit) {
    node_color = float3(0., 0.9, 0.);
  } else if (n.type == NodeType::Blind) {
    node_color = float3(1., 0.9, 0.);
  }

  const float size = 0.008;
  const float2 offsets[6] = { float2(-1, -1), float2(1, -1), float2(-1, 1),
                              float2(1, -1),  float2(1, 1),  float2(-1, 1) };

  const float rx = (offsets[vertexID].x * size) / pc.aspect_ratio;
  const float ry = offsets[vertexID].y * size;

  VertexOutput output;
  output.position = float4(nx + rx, -(ny + ry), 0., 1.);
  output.color = node_color;

  return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
  return float4(input.color, 1.);
}
