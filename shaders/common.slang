enum CarState : int {
  Driving = 0,
  Queuing = 1,
  Evacuated = 2,
  Garage = 3,
  Stuck = 4,
  Disabled = 5
};

enum NodeType : int { Normal = 0, OpenExit = 1, ClosedExit = 2, Blind = 3 };

struct Node {
  float x;
  float y;
  int lock;
  NodeType type;
};

struct Edge {
  int start_node_idx;
  int end_node_idx;
  int next_edge_idx;
  float length;
  float max_speed;
  int head_car_idx;
  int garage_lock;
  int spawn_capacity;
};

struct Car {
  int current_edge_idx;
  float position;
  float speed;
  CarState state;
  int next_car_idx;
  int padding0;
  int padding1;
  int padding2;
};

struct PushConstants {
  float dt;
  uint num_cars;
  uint num_edges;
  int padding;
};

struct GraphicsConstants {
  float camera_x;
  float camera_y;
  float zoom_level;
  float aspect_ratio;
  float extent_width;
  float extent_height;
  int selected_car_id;
  int padding;
};
