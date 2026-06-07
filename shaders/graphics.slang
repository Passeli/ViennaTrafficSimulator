#include "common.slang"

[[vk::binding(0, 0)]]
RWStructuredBuffer<Node> nodes;
[[vk::binding(1, 0)]]
RWStructuredBuffer<Edge> edges;
[[vk::binding(2, 0)]]
RWStructuredBuffer<Car> cars;
[[vk::push_constant]]
ConstantBuffer<GraphicsConstants> pc;

struct VertexOutput {
  float4 position : SV_Position;
  float3 color : COLOR;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexID: SV_VertexID,
                        uint instanceID: SV_InstanceID) {
  const Car car = cars[instanceID];

  // If evacuated or garaged, collapse and exit
  if (car.state == CarState::Evacuated || car.state == CarState::Garage) {
    VertexOutput output;
    output.position = float4(2.0, 2.0, 0.0, 1.0);
    output.color = float3(0.0, 0.0, 0.0);
    return output;
  }

  const Edge edge = edges[car.current_edge_idx];
  const Node start_node = nodes[edge.start_node_idx];
  const Node end_node = nodes[edge.end_node_idx];

  // Interpolate world position
  const float t = edge.length > 0.0 ? (car.position / edge.length) : 0.0;
  const float car_x = lerp(start_node.x, end_node.x, t);
  const float car_y = lerp(start_node.y, end_node.y, t);

  // 1. Pan and Zoom Projection
  const float local_x = (car_x - pc.camera_x) * pc.zoom_level;
  const float local_y = (car_y - pc.camera_y) * pc.zoom_level;

  const float nx = local_x / (pc.extent_width * 0.5) / pc.aspect_ratio;
  const float ny = local_y / (pc.extent_height * 0.5);

  // Driving = Green, Queuing/Blocked = Red
  const float3 debug_color =
      (car.state == CarState::Driving) ? float3(0, 1, 0) : float3(1, 0, 0);

  const float size = 0.001;
  const float2 offsets[6] = { float2(-2, -1), float2(0, -1), float2(-2, 1),
                        float2(0, -1),  float2(0, 1),  float2(-2, 1) };

  const float rx = (offsets[vertexID].x * size) / pc.aspect_ratio;
  const float ry = offsets[vertexID].y * size;

  VertexOutput output;
  output.position = float4(nx + rx, -(ny + ry), 0.0, 1.0);
  output.color = debug_color;

  return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
  return float4(input.color, 1.0);
}
