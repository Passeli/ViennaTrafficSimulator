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

  const float nx = local_x / (pc.extent_height * 0.5) / pc.aspect_ratio;
  const float ny = local_y / (pc.extent_height * 0.5);

  const bool is_selected = (instanceID == (uint)pc.selected_car_id);

  float3 debug_color;
  if (is_selected) {
    debug_color = float3(1.0, 0.0, 1.0); // Neon Pink/Fuchsia
  } else if (car.state == CarState::Driving) {
    const float speed_ratio =
        clamp(car.speed / max(edge.max_speed, 1.0), 0.0, 1.0);
    debug_color = lerp(float3(0.95, 0.25, 0.15), float3(0.15, 0.85, 0.35),
                       speed_ratio); // Red to Green speed gradient
  } else if (car.state == CarState::Stuck) {
    debug_color = float3(1.0, 0.65, 0.0); // Amber/Orange
  } else {
    debug_color = float3(0.9, 0.1, 0.1); // Coral Red for Queuing/Blocked
  }

  const float size = is_selected ? 0.010 : 0.004;
  const float2 offsets[6] = { float2(-2, -1), float2(0, -1), float2(-2, 1),
                              float2(0, -1),  float2(0, 1),  float2(-2, 1) };

  float2 offset = offsets[vertexID];
  if (is_selected) {
    const float2 centered_offsets[6] = { float2(-1, -1), float2(1, -1),
                                         float2(-1, 1),  float2(1, -1),
                                         float2(1, 1),   float2(-1, 1) };
    offset = centered_offsets[vertexID];
  }

  const float rx = (offset.x * size) / pc.aspect_ratio;
  const float ry = offset.y * size;

  VertexOutput output;
  output.position = float4(nx + rx, -(ny + ry), 0.0, 1.0);
  output.color = debug_color;

  return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
  return float4(input.color, 1.0);
}
