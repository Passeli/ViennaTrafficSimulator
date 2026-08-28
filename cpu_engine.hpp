#ifndef BA_CPU_ENGINE_HPP
#define BA_CPU_ENGINE_HPP

#include "common.hpp"
#include <span>
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
    explicit CPUThreadPool(std::size_t numThreads = 0) {
        if (numThreads == 0) {
            numThreads = std::max(1u, std::thread::hardware_concurrency());
        }
        resize(numThreads);
    }

    ~CPUThreadPool() {
        stopAll();
    }

    void stopAll() {
        stop.store(true, std::memory_order_release);
        cvTask.notify_all();
        workers.clear();
    }

    void resize(const std::size_t numThreads) {
        stopAll();
        stop.store(false, std::memory_order_release);
        activeTasks = 0;
        while (!tasks.empty()) {
            tasks.pop();
        }

        workers.reserve(numThreads);
        for (std::size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                workerLoop();
            });
        }
    }

    [[nodiscard]] std::size_t getNumThreads() const noexcept {
        return workers.size();
    }

    template<typename F>
    void parallelFor(const std::size_t total, F &&func) {
        if (total == 0) {
            return;
        }
        const std::size_t nThreads = workers.size();
        if (nThreads <= 1) {
            func(0, total);
            return;
        }

        const std::size_t chunkSize = (total + nThreads - 1) / nThreads;
        {
            std::lock_guard lock{queueMutex};
            activeTasks = nThreads;
            for (std::size_t t = 0; t < nThreads; ++t) {
                const std::size_t start = t * chunkSize;
                const std::size_t end = std::min(start + chunkSize, total);
                tasks.emplace([&func, start, end] {
                    if (start < end) {
                        func(start, end);
                    }
                });
            }
        }
        cvTask.notify_all();

        std::unique_lock lock{queueMutex};
        cvDone.wait(lock, [this] {
            return activeTasks == 0 && tasks.empty();
        });
    }

private:
    void workerLoop() {
        while (!stop.load(std::memory_order_acquire)) {
            std::function<void()> task;
            {
                std::unique_lock lock{queueMutex};
                cvTask.wait(lock, [this] {
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
                    std::lock_guard lock{queueMutex};
                    --activeTasks;
                    if (activeTasks == 0 && tasks.empty()) {
                        cvDone.notify_all();
                    }
                }
            }
        }
    }

    std::vector<std::jthread> workers;
    std::queue<std::function<void()> > tasks;
    std::mutex queueMutex;
    std::condition_variable cvTask;
    std::condition_variable cvDone;
    std::size_t activeTasks = 0;
    std::atomic<bool> stop = false;
};

class CPUEngine {
public:
    explicit CPUEngine(const std::size_t numThreads = 0)
        : threadPool{numThreads} {
    }

    ~CPUEngine() = default;

    void setNumThreads(const std::size_t numThreads) {
        threadPool.resize(numThreads);
    }

    [[nodiscard]] std::size_t getNumThreads() const noexcept {
        return threadPool.getNumThreads();
    }

    struct StepTimings {
        double clearEdgesMs = 0.;
        double buildGridMs = 0.;
        double physicsMs = 0.;
        double totalStepMs = 0.;
    };

    [[nodiscard]] const StepTimings &getLastTimings() const noexcept {
        return lastTimings;
    }

    void step(const std::span<GPU_Node> nodes, const std::span<GPU_Edge> edges, const std::span<GPU_Car> cars,
              const float dt) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        clearEdgesPass(edges);
        const auto t1 = std::chrono::high_resolution_clock::now();
        buildGridPass(edges, cars);
        const auto t2 = std::chrono::high_resolution_clock::now();
        physicsPass(nodes, edges, cars, dt);
        const auto t3 = std::chrono::high_resolution_clock::now();

        lastTimings.clearEdgesMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        lastTimings.buildGridMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
        lastTimings.physicsMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
        lastTimings.totalStepMs = std::chrono::duration<double, std::milli>(t3 - t0).count();
    }

private:
    void clearEdgesPass(const std::span<GPU_Edge> edges) {
        threadPool.parallelFor(edges.size(), [edges](const std::size_t start, const std::size_t end) {
            for (std::size_t i = start; i < end; ++i) {
                edges[i].head_car_idx = -1;
                edges[i].garage_lock = -1;
            }
        });
    }

    void buildGridPass(const std::span<GPU_Edge> edges, const std::span<GPU_Car> cars) {
        threadPool.parallelFor(cars.size(), [edges, cars](const std::size_t start, const std::size_t end) {
            for (std::size_t id = start; id < end; ++id) {
                cars[id].next_car_idx = -1;

                if (cars[id].state == CarState::Evacuated || cars[id].state == CarState::Garage) {
                    continue;
                }

                const std::int32_t edgeIdx = cars[id].current_edge_idx;
                if (edgeIdx >= 0 && static_cast<std::size_t>(edgeIdx) < edges.size()) {
                    std::atomic_ref headRef{edges[edgeIdx].head_car_idx};
                    const std::int32_t originalHead = headRef.exchange(
                        static_cast<std::int32_t>(id), std::memory_order_relaxed);
                    cars[id].next_car_idx = originalHead;
                }
            }
        });
    }

    void physicsPass(
        std::span<GPU_Node> nodes,
        std::span<GPU_Edge> edges,
        std::span<GPU_Car> cars,
        const float dt
    ) {
        if (carsSnapshot.size() != cars.size()) {
            carsSnapshot.resize(cars.size());
        }
        std::memcpy(carsSnapshot.data(), cars.data(), sizeof(GPU_Car) * cars.size());

        constexpr float CAR_LENGTH = 4.5f;
        constexpr float MIN_GAP = 0.5f;
        constexpr float TIME_GAP = 1.5f;

        threadPool.parallelFor(cars.size(),
                               [this, nodes, edges, cars, dt](const std::size_t start, const std::size_t end) {
                                   for (std::size_t id = start; id < end; ++id) {
                                       GPU_Car car = carsSnapshot[id];

                                       if (car.state == CarState::Evacuated || car.state == CarState::Disabled) {
                                           cars[id] = car;
                                           continue;
                                       }

                                       const std::int32_t edgeIdx = car.current_edge_idx;
                                       if (edgeIdx < 0 || static_cast<std::size_t>(edgeIdx) >= edges.size()) {
                                           cars[id] = car;
                                           continue;
                                       }

                                       const GPU_Edge &edge = edges[edgeIdx];

                                       // ==========================================
                                       // PHASE 1: DRIVING STATE
                                       // ==========================================
                                       if (car.state == CarState::Driving) {
                                           float frontDistance = edge.length - car.position;
                                           float frontSpeed = 0.f;
                                           bool isLeadCar = true;
                                           std::int32_t closestFrontCarId = -1;

                                           // Traverse the spatial linked list
                                           std::int32_t currentSearchId = edge.head_car_idx;
                                           while (currentSearchId != -1) {
                                               if (static_cast<std::size_t>(currentSearchId) != id) {
                                                   const GPU_Car &otherCar = carsSnapshot[currentSearchId];
                                                   if (otherCar.current_edge_idx == car.current_edge_idx) {
                                                       if (otherCar.state == CarState::Driving ||
                                                           otherCar.state == CarState::Queuing ||
                                                           otherCar.state == CarState::Stuck) {
                                                           const float distToOtherFront =
                                                                   otherCar.position - car.position;
                                                           if (distToOtherFront > 0.f ||
                                                               (distToOtherFront == 0.f && static_cast<std::size_t>(
                                                                    currentSearchId) >
                                                                id)) {
                                                               const float trueGap = distToOtherFront - CAR_LENGTH;
                                                               if (trueGap < frontDistance) {
                                                                   frontDistance = trueGap;
                                                                   frontSpeed = otherCar.speed;
                                                                   isLeadCar = false;
                                                                   closestFrontCarId = currentSearchId;
                                                               }
                                                           }
                                                       }
                                                   }
                                               }
                                               currentSearchId = carsSnapshot[currentSearchId].next_car_idx;
                                           }

                                           float targetGap = isLeadCar ? 0.f : MIN_GAP;

                                           if (isLeadCar && car.position + 0.1f >= edge.length) {
                                               car.position = edge.length; // Snap perfectly to stop line
                                               car.speed = 0.f;
                                               if (nodes[edge.end_node_idx].type == NodeType::Blind) {
                                                   car.state = CarState::Stuck;
                                               } else {
                                                   car.state = CarState::Queuing;
                                               }
                                           } else if (!isLeadCar && closestFrontCarId != -1 &&
                                                      carsSnapshot[closestFrontCarId].state == CarState::Stuck &&
                                                      (frontDistance <= MIN_GAP + 0.1f)) {
                                               car.position =
                                                       carsSnapshot[closestFrontCarId].position - CAR_LENGTH - MIN_GAP;
                                               car.speed = 0.f;
                                               car.state = CarState::Stuck;
                                           } else {
                                               // Intelligent Driver Model (IDM)
                                               const float speedDiff = isLeadCar ? car.speed : (car.speed - frontSpeed);
                                               const float T = isLeadCar ? 0.6f : TIME_GAP;
                                               const float denom = isLeadCar ? 4.5825f : 3.4641f;

                                               float dynamicGap =
                                                       targetGap + car.speed * T + car.speed * speedDiff / denom;
                                               dynamicGap = std::max(dynamicGap, targetGap);

                                               const float speedRatio = car.speed / std::max(edge.max_speed, 0.1f);
                                               float accel = 1.5f * (1.f - std::pow(speedRatio, 4.f) -
                                                                     std::pow(
                                                                         dynamicGap / std::max(frontDistance, 0.1f),
                                                                         2.f));

                                               accel = std::clamp(accel, -5.f, 3.f);
                                               car.speed = std::max(0.f, car.speed + accel * dt);
                                               car.position += car.speed * dt;
                                           }
                                       }
                                       // ==========================================
                                       // PHASE 2: QUEUING STATE
                                       // ==========================================
                                       else if (car.state == CarState::Queuing) {
                                           const std::int32_t targetNode = edge.end_node_idx;
                                           const std::int32_t nextEdge = edge.next_edge_idx;

                                           std::atomic_ref lockRef{nodes[targetNode].lock};
                                           std::int32_t expected = -1;

                                           if (lockRef.compare_exchange_strong(expected, static_cast<std::int32_t>(id),
                                                                               std::memory_order_acq_rel)) {
                                               if (nodes[targetNode].type == NodeType::OpenExit) {
                                                   car.state = CarState::Evacuated;
                                                   lockRef.store(-1, std::memory_order_release);
                                               } else if (nextEdge != -1 && static_cast<std::size_t>(nextEdge) < edges.
                                                          size()) {
                                                   bool entranceClear = true;
                                                   bool blockedByStuck = false;
                                                   std::int32_t nextSearchId = edges[nextEdge].head_car_idx;

                                                   while (nextSearchId != -1) {
                                                       const GPU_Car &nextCar = carsSnapshot[nextSearchId];
                                                       if (nextCar.state == CarState::Driving ||
                                                           nextCar.state == CarState::Queuing ||
                                                           nextCar.state == CarState::Stuck) {
                                                           const float tailGap = nextCar.position - CAR_LENGTH;
                                                           if (tailGap < MIN_GAP) {
                                                               entranceClear = false;
                                                               if (nextCar.state == CarState::Stuck) {
                                                                   blockedByStuck = true;
                                                               }
                                                               break;
                                                           }
                                                       }
                                                       nextSearchId = carsSnapshot[nextSearchId].next_car_idx;
                                                   }

                                                   if (entranceClear) {
                                                       car.current_edge_idx = nextEdge;
                                                       car.position = 0.f;
                                                       car.speed = 0.f;
                                                       car.state = CarState::Driving;
                                                   } else if (blockedByStuck) {
                                                       car.state = CarState::Stuck;
                                                   }
                                                   lockRef.store(-1, std::memory_order_release);
                                               } else {
                                                   // No path to exit
                                                   car.state = CarState::Stuck;
                                                   lockRef.store(-1, std::memory_order_release);
                                               }
                                           }
                                       }
                                       // ==========================================
                                       // PHASE 3: GARAGE STATE
                                       // ==========================================
                                       else if (car.state == CarState::Garage) {
                                           std::atomic_ref garageRef{edges[car.current_edge_idx].garage_lock};
                                           std::int32_t expected = -1;

                                           if (garageRef.compare_exchange_strong(
                                               expected, static_cast<std::int32_t>(id),
                                               std::memory_order_acq_rel)) {
                                               bool entranceClear = true;
                                               std::int32_t nextSearchId = edge.head_car_idx;

                                               while (nextSearchId != -1) {
                                                   const GPU_Car &nextCar = carsSnapshot[nextSearchId];
                                                   if (nextCar.current_edge_idx == car.current_edge_idx &&
                                                       (nextCar.state == CarState::Driving ||
                                                        nextCar.state == CarState::Queuing ||
                                                        nextCar.state == CarState::Stuck)) {
                                                       const float tailGap = nextCar.position - CAR_LENGTH;
                                                       if (tailGap < 5.f) {
                                                           entranceClear = false;
                                                           break;
                                                       }
                                                   }
                                                   nextSearchId = carsSnapshot[nextSearchId].next_car_idx;
                                               }

                                               if (entranceClear) {
                                                   car.state = CarState::Driving;
                                                   car.position = 0.f;
                                                   car.speed = 0.f;
                                               }
                                           }
                                       }
                                       // ==========================================
                                       // PHASE 4: STUCK STATE
                                       // ==========================================
                                       else if (car.state == CarState::Stuck) {
                                           if (car.position >= edge.length - 0.1f) {
                                               const bool targetIsOpenExit = (
                                                   nodes[edge.end_node_idx].type == NodeType::OpenExit);
                                               const bool targetIsBlind = (
                                                   nodes[edge.end_node_idx].type == NodeType::Blind);
                                               if (!targetIsBlind && (edge.next_edge_idx != -1 || targetIsOpenExit)) {
                                                   car.state = CarState::Queuing;
                                               }
                                           } else {
                                               float frontDistance = edge.length - car.position;
                                               bool isLeadCar = true;
                                               std::int32_t closestFrontCarId = -1;

                                               std::int32_t currentSearchId = edge.head_car_idx;
                                               while (currentSearchId != -1) {
                                                   if (static_cast<std::size_t>(currentSearchId) != id) {
                                                       const GPU_Car &otherCar = carsSnapshot[currentSearchId];
                                                       if (otherCar.current_edge_idx == car.current_edge_idx) {
                                                           if (otherCar.state == CarState::Driving ||
                                                               otherCar.state == CarState::Queuing ||
                                                               otherCar.state == CarState::Stuck) {
                                                               const float distToOtherFront =
                                                                       otherCar.position - car.position;
                                                               if (distToOtherFront > 0.f ||
                                                                   (distToOtherFront == 0.f && static_cast<std::size_t>(
                                                                        currentSearchId)
                                                                    > id)) {
                                                                   const float trueGap = distToOtherFront - CAR_LENGTH;
                                                                   if (trueGap < frontDistance) {
                                                                       frontDistance = trueGap;
                                                                       isLeadCar = false;
                                                                       closestFrontCarId = currentSearchId;
                                                                   }
                                                               }
                                                           }
                                                       }
                                                   }
                                                   currentSearchId = carsSnapshot[currentSearchId].next_car_idx;
                                               }

                                               if (isLeadCar ||
                                                   (closestFrontCarId != -1 &&
                                                    carsSnapshot[closestFrontCarId].state != CarState::Stuck)) {
                                                   car.state = CarState::Driving;
                                               }
                                           }
                                       }

                                       cars[id] = car;
                                   }
                               });
    }

    CPUThreadPool threadPool;
    std::vector<GPU_Car> carsSnapshot;
    StepTimings lastTimings;
};

#endif // BA_CPU_ENGINE_HPP
