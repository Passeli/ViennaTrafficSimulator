#ifndef BA_COMMON_HPP
#define BA_COMMON_HPP

#include <cstdint>

enum class CarState : std::int32_t {
    Driving = 0,
    Queuing = 1,
    Evacuated = 2,
    Garage = 3,
    Stuck = 4,
    Disabled = 5
};

enum class NodeType : std::int32_t {
    Normal = 0,
    OpenExit = 1,
    ClosedExit = 2,
    Blind = 3
};

struct GPU_Node {
    float x = 0.f;
    float y = 0.f;
    std::int32_t lock = -1;
    NodeType type = NodeType::Normal;
};

struct GPU_Edge {
    std::int32_t start_node_idx = 0;
    std::int32_t end_node_idx = 0;
    std::int32_t next_edge_idx = -1;
    float length = 0.f;
    float max_speed = 0.f;
    std::int32_t head_car_idx = -1;
    std::int32_t garage_lock = -1;
    std::int32_t spawn_capacity = 0;
};

struct GPU_Car {
    std::int32_t current_edge_idx = 0;
    float position = 0.f;
    float speed = 0.f;
    CarState state = CarState::Garage;
    std::int32_t next_car_idx = -1;
    std::int32_t padding0 = 0;
    std::int32_t padding1 = 0;
    std::int32_t padding2 = 0;
};

struct PushConstants {
    float dt = 0.f;
    std::uint32_t num_cars = 0;
    std::uint32_t num_edges = 0;
    std::int32_t padding = 0;
};

struct GraphicsConstants {
    float camera_x = 0.f;
    float camera_y = 0.f;
    float zoom_level = 1.f;
    float aspect_ratio = 1.f;
    float extent_width = 0.f;
    float extent_height = 0.f;
    std::int32_t selected_car_id = -1;
    std::int32_t padding = 0;
};

#endif // BA_COMMON_HPP
