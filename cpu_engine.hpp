#ifndef BA_CPU_ENGINE_HPP
#define BA_CPU_ENGINE_HPP

#include "common.hpp"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <queue>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <chrono>

class CPUThreadPool {
public:
    explicit CPUThreadPool(std::size_t num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::max(1u, std::thread::hardware_concurrency());
        }
        resize(num_threads);
    }

    ~CPUThreadPool() {
        stop_all();
    }

    void stop_all() {
        stop.store(true, std::memory_order_release);
        cv_task.notify_all();
        workers.clear();
    }

    void resize(const std::size_t num_threads) {
        stop_all();
        stop.store(false, std::memory_order_release);
        active_tasks = 0;
        while (!tasks.empty()) {
            tasks.pop();
        }

        workers.reserve(num_threads);
        for (std::size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this]() {
                worker_loop();
            });
        }
    }

    [[nodiscard]] std::size_t get_num_threads() const noexcept {
        return workers.size();
    }

    template<typename F>
    void parallel_for(const std::size_t total, F &&func) {
        if (total == 0) return;
        const std::size_t n_threads = workers.size();
        if (n_threads <= 1) {
            func(0, total);
            return;
        }

        const std::size_t chunk_size = (total + n_threads - 1) / n_threads;
        {
            std::lock_guard lock(queue_mutex);
            active_tasks = n_threads;
            for (std::size_t t = 0; t < n_threads; ++t) {
                const std::size_t start = t * chunk_size;
                const std::size_t end = std::min(start + chunk_size, total);
                tasks.emplace([&func, start, end]() {
                    if (start < end) {
                        func(start, end);
                    }
                });
            }
        }
        cv_task.notify_all();

        std::unique_lock lock(queue_mutex);
        cv_done.wait(lock, [this]() {
            return this->active_tasks == 0 && this->tasks.empty();
        });
    }

private:
    void worker_loop() {
        while (!stop.load(std::memory_order_acquire)) {
            std::function<void()> task;
            {
                std::unique_lock lock(queue_mutex);
                cv_task.wait(lock, [this] {
                    return stop.load(std::memory_order_acquire) || !tasks.empty();
                });

                if (stop.load(std::memory_order_acquire) && tasks.empty()) {
                    return;
                }

                if (!tasks.empty()) {
                    task = std::move(tasks.front());
                    tasks.pop();
                }
            }

            if (task) {
                task();
                {
                    std::lock_guard lock(queue_mutex);
                    --active_tasks;
                    if (active_tasks == 0 && tasks.empty()) {
                        cv_done.notify_all();
                    }
                }
            }
        }
    }

    std::vector<std::jthread> workers;
    std::queue<std::function<void()> > tasks;
    std::mutex queue_mutex;
    std::condition_variable cv_task;
    std::condition_variable cv_done;
    std::size_t active_tasks = 0;
    std::atomic<bool> stop{false};
};

class CPUEngine {
public:
    explicit CPUEngine(const std::size_t num_threads = 0)
        : thread_pool(num_threads) {
    }

    ~CPUEngine() = default;

    void set_num_threads(const std::size_t num_threads) {
        thread_pool.resize(num_threads);
    }

    [[nodiscard]] std::size_t get_num_threads() const noexcept {
        return thread_pool.get_num_threads();
    }

    struct StepTimings {
        double clear_edges_ms = 0.0;
        double build_grid_ms = 0.0;
        double physics_ms = 0.0;
        double total_step_ms = 0.0;
    };

    [[nodiscard]] const StepTimings &get_last_timings() const noexcept {
        return last_timings;
    }

    void step(
        std::vector<GPU_Node> &nodes,
        std::vector<GPU_Edge> &edges,
        std::vector<GPU_Car> &cars,
        const float dt
    ) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        clear_edges_pass(edges);
        const auto t1 = std::chrono::high_resolution_clock::now();
        build_grid_pass(edges, cars);
        const auto t2 = std::chrono::high_resolution_clock::now();
        physics_pass(nodes, edges, cars, dt);
        const auto t3 = std::chrono::high_resolution_clock::now();

        last_timings.clear_edges_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        last_timings.build_grid_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        last_timings.physics_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        last_timings.total_step_ms = std::chrono::duration<double, std::milli>(t3 - t0).count();
    }

private:
    void clear_edges_pass(std::vector<GPU_Edge> &edges) {
        thread_pool.parallel_for(edges.size(), [&edges](const std::size_t start, const std::size_t end) {
            for (std::size_t i = start; i < end; ++i) {
                edges[i].head_car_idx = -1;
                edges[i].garage_lock = -1;
            }
        });
    }

    void build_grid_pass(std::vector<GPU_Edge> &edges, std::vector<GPU_Car> &cars) {
        thread_pool.parallel_for(cars.size(), [&edges, &cars](const std::size_t start, const std::size_t end) {
            for (std::size_t id = start; id < end; ++id) {
                cars[id].next_car_idx = -1;

                if (cars[id].state == CarState::Evacuated || cars[id].state == CarState::Garage) {
                    continue;
                }

                const std::int32_t edge_idx = cars[id].current_edge_idx;
                if (edge_idx >= 0 && static_cast<std::size_t>(edge_idx) < edges.size()) {
                    std::atomic_ref head_ref(edges[edge_idx].head_car_idx);
                    const std::int32_t original_head = head_ref.exchange(
                        static_cast<std::int32_t>(id), std::memory_order_relaxed);
                    cars[id].next_car_idx = original_head;
                }
            }
        });
    }

    void physics_pass(
        std::vector<GPU_Node> &nodes,
        std::vector<GPU_Edge> &edges,
        std::vector<GPU_Car> &cars,
        const float dt
    ) {
        if (cars_snapshot.size() != cars.size()) {
            cars_snapshot.resize(cars.size());
        }
        std::memcpy(cars_snapshot.data(), cars.data(), sizeof(GPU_Car) * cars.size());

        constexpr float CAR_LENGTH = 4.5f;
        constexpr float MIN_GAP = 0.5f;
        constexpr float TIME_GAP = 1.5f;

        thread_pool.parallel_for(cars.size(), [&](const std::size_t start, const std::size_t end) {
            for (std::size_t id = start; id < end; ++id) {
                GPU_Car car = cars_snapshot[id];

                if (car.state == CarState::Evacuated || car.state == CarState::Disabled) {
                    cars[id] = car;
                    continue;
                }

                const std::int32_t edge_idx = car.current_edge_idx;
                if (edge_idx < 0 || static_cast<std::size_t>(edge_idx) >= edges.size()) {
                    cars[id] = car;
                    continue;
                }

                const GPU_Edge &edge = edges[edge_idx];

                // ==========================================
                // PHASE 1: DRIVING STATE
                // ==========================================
                if (car.state == CarState::Driving) {
                    float front_distance = edge.length - car.position;
                    float front_speed = 0.0f;
                    bool is_lead_car = true;
                    std::int32_t closest_front_car_id = -1;

                    // Traverse the spatial linked list
                    std::int32_t current_search_id = edge.head_car_idx;
                    while (current_search_id != -1) {
                        if (static_cast<std::size_t>(current_search_id) != id) {
                            const GPU_Car &other_car = cars_snapshot[current_search_id];
                            if (other_car.current_edge_idx == car.current_edge_idx) {
                                if (other_car.state == CarState::Driving ||
                                    other_car.state == CarState::Queuing ||
                                    other_car.state == CarState::Stuck) {
                                    float dist_to_other_front = other_car.position - car.position;
                                    if (dist_to_other_front > 0.0f ||
                                        (dist_to_other_front == 0.0f && static_cast<std::size_t>(current_search_id) >
                                         id)) {
                                        float true_gap = dist_to_other_front - CAR_LENGTH;
                                        if (true_gap < front_distance) {
                                            front_distance = true_gap;
                                            front_speed = other_car.speed;
                                            is_lead_car = false;
                                            closest_front_car_id = current_search_id;
                                        }
                                    }
                                }
                            }
                        }
                        current_search_id = cars_snapshot[current_search_id].next_car_idx;
                    }

                    float target_gap = is_lead_car ? 0.0f : MIN_GAP;

                    if (is_lead_car && (car.position + 0.1f >= edge.length)) {
                        car.position = edge.length; // Snap perfectly to stop line
                        car.speed = 0.0f;
                        if (nodes[edge.end_node_idx].type == NodeType::Blind) {
                            car.state = CarState::Stuck;
                        } else {
                            car.state = CarState::Queuing;
                        }
                    } else if (!is_lead_car && closest_front_car_id != -1 &&
                               cars_snapshot[closest_front_car_id].state == CarState::Stuck &&
                               (front_distance <= MIN_GAP + 0.1f)) {
                        car.position = cars_snapshot[closest_front_car_id].position - CAR_LENGTH - MIN_GAP;
                        car.speed = 0.0f;
                        car.state = CarState::Stuck;
                    } else {
                        // Intelligent Driver Model (IDM)
                        float speed_diff = is_lead_car ? car.speed : (car.speed - front_speed);
                        float T = is_lead_car ? 0.6f : TIME_GAP;
                        float denom = is_lead_car ? 4.5825f : 3.4641f;

                        float dynamic_gap = target_gap + (car.speed * T) + (car.speed * speed_diff) / denom;
                        dynamic_gap = std::max(dynamic_gap, target_gap);

                        float speed_ratio = car.speed / std::max(edge.max_speed, 0.1f);
                        float accel = 1.5f * (1.0f - std::pow(speed_ratio, 4.0f) -
                                              std::pow(dynamic_gap / std::max(front_distance, 0.1f), 2.0f));

                        accel = std::clamp(accel, -5.0f, 3.0f);
                        car.speed = std::max(0.0f, car.speed + accel * dt);
                        car.position += car.speed * dt;
                    }
                }
                // ==========================================
                // PHASE 2: QUEUING STATE
                // ==========================================
                else if (car.state == CarState::Queuing) {
                    const std::int32_t target_node = edge.end_node_idx;
                    const std::int32_t next_edge = edge.next_edge_idx;

                    std::atomic_ref<std::int32_t> lock_ref(nodes[target_node].lock);
                    std::int32_t expected = -1;

                    if (lock_ref.compare_exchange_strong(expected, static_cast<std::int32_t>(id),
                                                         std::memory_order_acq_rel)) {
                        if (nodes[target_node].type == NodeType::OpenExit) {
                            car.state = CarState::Evacuated;
                            lock_ref.store(-1, std::memory_order_release);
                        } else if (next_edge != -1 && static_cast<std::size_t>(next_edge) < edges.size()) {
                            bool entrance_clear = true;
                            bool blocked_by_stuck = false;
                            std::int32_t next_search_id = edges[next_edge].head_car_idx;

                            while (next_search_id != -1) {
                                const GPU_Car &next_car = cars_snapshot[next_search_id];
                                if (next_car.state == CarState::Driving ||
                                    next_car.state == CarState::Queuing ||
                                    next_car.state == CarState::Stuck) {
                                    float tail_gap = next_car.position - CAR_LENGTH;
                                    if (tail_gap < MIN_GAP) {
                                        entrance_clear = false;
                                        if (next_car.state == CarState::Stuck) {
                                            blocked_by_stuck = true;
                                        }
                                        break;
                                    }
                                }
                                next_search_id = cars_snapshot[next_search_id].next_car_idx;
                            }

                            if (entrance_clear) {
                                car.current_edge_idx = next_edge;
                                car.position = 0.0f;
                                car.speed = 0.0f;
                                car.state = CarState::Driving;
                            } else if (blocked_by_stuck) {
                                car.state = CarState::Stuck;
                            }
                            lock_ref.store(-1, std::memory_order_release);
                        } else {
                            // No path to exit
                            car.state = CarState::Stuck;
                            lock_ref.store(-1, std::memory_order_release);
                        }
                    }
                }
                // ==========================================
                // PHASE 3: GARAGE STATE
                // ==========================================
                else if (car.state == CarState::Garage) {
                    std::atomic_ref<std::int32_t> garage_ref(edges[car.current_edge_idx].garage_lock);
                    std::int32_t expected = -1;

                    if (garage_ref.compare_exchange_strong(expected, static_cast<std::int32_t>(id),
                                                           std::memory_order_acq_rel)) {
                        bool entrance_clear = true;
                        std::int32_t next_search_id = edge.head_car_idx;

                        while (next_search_id != -1) {
                            const GPU_Car &next_car = cars_snapshot[next_search_id];
                            if (next_car.current_edge_idx == car.current_edge_idx &&
                                (next_car.state == CarState::Driving ||
                                 next_car.state == CarState::Queuing ||
                                 next_car.state == CarState::Stuck)) {
                                float tail_gap = next_car.position - CAR_LENGTH;
                                if (tail_gap < 5.0f) {
                                    entrance_clear = false;
                                    break;
                                }
                            }
                            next_search_id = cars_snapshot[next_search_id].next_car_idx;
                        }

                        if (entrance_clear) {
                            car.state = CarState::Driving;
                            car.position = 0.0f;
                            car.speed = 0.0f;
                        }
                    }
                }
                // ==========================================
                // PHASE 4: STUCK STATE
                // ==========================================
                else if (car.state == CarState::Stuck) {
                    if (car.position >= edge.length - 0.1f) {
                        bool target_is_open_exit = (nodes[edge.end_node_idx].type == NodeType::OpenExit);
                        bool target_is_blind = (nodes[edge.end_node_idx].type == NodeType::Blind);
                        if (!target_is_blind && (edge.next_edge_idx != -1 || target_is_open_exit)) {
                            car.state = CarState::Queuing;
                        }
                    } else {
                        float front_distance = edge.length - car.position;
                        bool is_lead_car = true;
                        std::int32_t closest_front_car_id = -1;

                        std::int32_t current_search_id = edge.head_car_idx;
                        while (current_search_id != -1) {
                            if (static_cast<std::size_t>(current_search_id) != id) {
                                const GPU_Car &other_car = cars_snapshot[current_search_id];
                                if (other_car.current_edge_idx == car.current_edge_idx) {
                                    if (other_car.state == CarState::Driving ||
                                        other_car.state == CarState::Queuing ||
                                        other_car.state == CarState::Stuck) {
                                        float dist_to_other_front = other_car.position - car.position;
                                        if (dist_to_other_front > 0.0f ||
                                            (dist_to_other_front == 0.0f && static_cast<std::size_t>(current_search_id)
                                             > id)) {
                                            float true_gap = dist_to_other_front - CAR_LENGTH;
                                            if (true_gap < front_distance) {
                                                front_distance = true_gap;
                                                is_lead_car = false;
                                                closest_front_car_id = current_search_id;
                                            }
                                        }
                                    }
                                }
                            }
                            current_search_id = cars_snapshot[current_search_id].next_car_idx;
                        }

                        if (is_lead_car ||
                            (closest_front_car_id != -1 &&
                             cars_snapshot[closest_front_car_id].state != CarState::Stuck)) {
                            car.state = CarState::Driving;
                        }
                    }
                }

                cars[id] = car;
            }
        });
    }

    CPUThreadPool thread_pool;
    std::vector<GPU_Car> cars_snapshot;
    StepTimings last_timings;
};

#endif // BA_CPU_ENGINE_HPP
