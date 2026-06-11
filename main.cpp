#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>
#include <vector>
#include <span>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <print>
#include <cstdint>
#include <memory>
#include <iostream>
#include <algorithm>
#include <queue>
#include <set>
#include <chrono>

#include "common.hpp"

template<typename T>
auto loadBinaryData(const std::string_view filepath) -> std::vector<T> {
    std::ifstream file{std::string(filepath), std::ios::binary | std::ios::ate};

    if (!file.is_open()) {
        throw std::runtime_error{"Failed to open: " + std::string(filepath)};
    }

    const std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    const std::size_t elementsToRead = static_cast<std::size_t>(fileSize) / sizeof(T);
    std::vector<T> buffer(elementsToRead);

    if (elementsToRead > 0) {
        file.read(reinterpret_cast<char *>(buffer.data()), fileSize);
    }

    return buffer;
}

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

class EvacuationEngine {
public:
    void run() {
        initWindow();
        initVulkan();
        loadMapDataAndCreateBuffers();
        createComputePipeline();

        // Refactored startup sequence to support swapchain recreation
        createSwapchain();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();

        initImGui();

        mainLoop();
    }

    ~EvacuationEngine() {
        if (device) {
            device->waitIdle();
            if (imguiInitialized) {
                ImGui_ImplVulkan_Shutdown();
                ImGui_ImplGlfw_Shutdown();
                ImGui::DestroyContext();
            }
        }
        if (window != nullptr) {
            glfwDestroyWindow(window);
            glfwTerminate();
        }
    }

private:
    GLFWwindow *window = nullptr;

    std::unique_ptr<vk::raii::Context> context;
    std::unique_ptr<vk::raii::Instance> instance;
    std::unique_ptr<vk::raii::PhysicalDevice> physicalDevice;
    std::unique_ptr<vk::raii::Device> device;

    std::uint32_t queueFamilyIndex = 0;
    std::unique_ptr<vk::raii::Queue> queue;
    std::unique_ptr<vk::raii::CommandPool> commandPool;

    std::unique_ptr<vk::raii::Buffer> nodeBuffer;
    std::unique_ptr<vk::raii::DeviceMemory> nodeMemory;
    std::unique_ptr<vk::raii::Buffer> edgeBuffer;
    std::unique_ptr<vk::raii::DeviceMemory> edgeMemory;
    std::unique_ptr<vk::raii::Buffer> carBuffer;
    std::unique_ptr<vk::raii::DeviceMemory> carMemory;

    std::unique_ptr<vk::raii::DescriptorSetLayout> computeDescriptorSetLayout;
    std::unique_ptr<vk::raii::PipelineLayout> computePipelineLayout;
    std::unique_ptr<vk::raii::Pipeline> clearEdgesPipeline;
    std::unique_ptr<vk::raii::Pipeline> buildGridPipeline;
    std::unique_ptr<vk::raii::Pipeline> physicsPipeline;
    std::unique_ptr<vk::raii::DescriptorPool> descriptorPool;
    std::unique_ptr<vk::raii::DescriptorPool> imguiPool;
    std::vector<vk::raii::DescriptorSet> computeDescriptorSets;

    std::uint32_t totalCars = 0;
    std::uint32_t totalEdges = 0;
    const std::uint32_t WIDTH = 800;
    const std::uint32_t HEIGHT = 600;

    std::unique_ptr<vk::raii::SurfaceKHR> surface;
    std::unique_ptr<vk::raii::SwapchainKHR> swapchain;
    std::vector<vk::Image> swapchainImages;
    std::vector<vk::raii::ImageView> swapchainImageViews;
    std::unique_ptr<vk::raii::RenderPass> renderPass;
    std::vector<vk::raii::Framebuffer> framebuffers;
    vk::Format swapchainImageFormat{};
    vk::Extent2D swapchainExtent;

    std::unique_ptr<vk::raii::PipelineLayout> graphicsPipelineLayout;
    std::unique_ptr<vk::raii::Pipeline> graphicsPipeline;
    std::unique_ptr<vk::raii::Pipeline> streetPipeline;
    std::unique_ptr<vk::raii::Pipeline> exitNodePipeline;
    GraphicsConstants mapBounds;

    // --- SIMULATION CONTROLS ---
    bool isPaused = true;
    std::int32_t simSpeed = 1;
    bool framebufferResized = false;

    // --- CAMERA INTERACTION STATE ---
    bool isDragging = false;
    double lastMouseX = 0.;
    double lastMouseY = 0.;

    // --- MARQUEE SELECTION STATE ---
    bool isSelecting = false;
    double startMouseX = 0.;
    double startMouseY = 0.;
    double currentMouseX = 0.;
    double currentMouseY = 0.;

    // --- INSPECT STATE ---
    bool isInspecting = false;
    double inspectMouseX = 0.;
    double inspectMouseY = 0.;

    // --- CPU SHADOW COPIES ---
    std::vector<GPU_Car> cpuCars;
    std::vector<GPU_Edge> cpuEdges;
    std::vector<GPU_Node> cpuNodes;

    // --- READBACK BUFFERS (Host Visible) ---
    std::unique_ptr<vk::raii::Buffer> carReadbackBuffer;
    std::unique_ptr<vk::raii::DeviceMemory> carReadbackMemory;
    std::unique_ptr<vk::raii::Buffer> edgeReadbackBuffer;
    std::unique_ptr<vk::raii::DeviceMemory> edgeReadbackMemory;
    std::unique_ptr<vk::raii::Buffer> nodeReadbackBuffer;
    std::unique_ptr<vk::raii::DeviceMemory> nodeReadbackMemory;

    // --- UI STATE ---
    std::int32_t selectedCarId = 0;
    bool imguiInitialized = false;

    std::int32_t statGarage = 0;
    std::int32_t statRoad = 0;
    std::int32_t statEvacuated = 0;
    std::int32_t statStuck = 0;
    std::int32_t statDisabled = 0;
    float statAvgSpeed = 0.f;
    float simTime = 0.f;
    float participationPercentage = 100.f;
    bool hasStarted = false;

    // --- METRICS HISTORY ---
    std::vector<float> flowrateHistory;
    std::vector<float> evacuatedHistory;
    float lastRecordTime = 0.f;
    std::int32_t lastEvacuatedCount = 0;

    // --- GRAPH ROUTING DATA (NEW) ---
    struct IncomingEdge {
        std::int32_t source_node;
        std::int32_t edge_idx;
        float travel_time;
    };

    std::vector<std::vector<IncomingEdge> > reverseGraph;
    std::vector<std::vector<std::int32_t> > forwardGraph;
    std::vector<std::int32_t> allExitNodes;
    std::vector<bool> isExitOpen;

    // --- ROUTING FUNCTIONS (NEW) ---
    void recalculateGPS() {
        std::println("Recalculating City-Wide GPS Routes...");
        const auto startTime = std::chrono::high_resolution_clock::now();

        // 0. Count cars and detect stuck cars on each edge
        std::vector<std::int32_t> edge_car_counts(totalEdges, 0);
        std::vector<bool> edge_has_stuck(totalEdges, false);
        if (!cpuCars.empty()) {
            for (const auto &car: cpuCars) {
                if (car.state == CarState::Driving || car.state == CarState::Queuing || car.state == CarState::Stuck) {
                    if (car.current_edge_idx >= 0 && car.current_edge_idx < static_cast<std::int32_t>(totalEdges)) {
                        edge_car_counts[car.current_edge_idx]++;
                        if (car.state == CarState::Stuck) {
                            edge_has_stuck[car.current_edge_idx] = true;
                        }
                    }
                }
            }
        }

        // 1. Reset all edges to have no destination and update exit-edge padding
        for (auto &edge: cpuEdges) {
            edge.next_edge_idx = -1;
        }

        // Update node types for exits
        for (std::size_t i = 0; i < allExitNodes.size(); ++i) {
            const std::int32_t nodeIdx = allExitNodes[i];
            cpuNodes[nodeIdx].type = isExitOpen[i] ? NodeType::OpenExit : NodeType::ClosedExit;
        }

        std::vector<float> min_travel_time(cpuNodes.size(), std::numeric_limits<float>::max());
        std::vector<std::int32_t> next_node_to_exit(cpuNodes.size(), -1);

        // Priority Queue: {travel_time, node_idx}
        using NodeRecord = std::pair<float, std::int32_t>;
        std::priority_queue<NodeRecord, std::vector<NodeRecord>, std::greater<> > pq;

        // 2. Seed the algorithm with all CURRENTLY OPEN exits
        for (std::size_t i = 0; i < allExitNodes.size(); ++i) {
            if (isExitOpen[i]) {
                std::int32_t exit_idx = allExitNodes[i];
                min_travel_time[exit_idx] = 0.f;
                pq.emplace(0.f, exit_idx);
            }
        }

        // 3. Run Dijkstra backwards
        while (!pq.empty()) {
            auto [current_time, current_node] = pq.top();
            pq.pop();

            if (current_time > min_travel_time[current_node]) continue;

            for (const auto &incoming: reverseGraph[current_node]) {
                float travel_time = incoming.travel_time;
                if (!cpuCars.empty()) {
                    if (edge_has_stuck[incoming.edge_idx]) {
                        travel_time = 1e6f; // Blocked street penalty
                    } else {
                        // Estimate edge vehicle capacity based on 5. meters per car
                        const float capacity = std::max(cpuEdges[incoming.edge_idx].length / 5.f, 1.f);
                        const float congestion = static_cast<float>(edge_car_counts[incoming.edge_idx]) / capacity;
                        // BPR (Bureau of Public Roads) congestion function
                        travel_time = incoming.travel_time * (1.f + 4.f * std::pow(congestion, 4.f));
                    }
                }
                float new_time = current_time + travel_time;

                if (new_time < min_travel_time[incoming.source_node]) {
                    min_travel_time[incoming.source_node] = new_time;
                    next_node_to_exit[incoming.source_node] = current_node;
                    pq.emplace(new_time, incoming.source_node);
                }
            }
        }

        // 4. Bake the fast O(1) paths into the Edge array
        for (std::int32_t i = 0; i < totalEdges; ++i) {
            const std::int32_t end_node = cpuEdges[i].end_node_idx;
            const std::int32_t target_node = next_node_to_exit[end_node];

            if (target_node != -1) {
                for (const std::int32_t j: forwardGraph[end_node]) {
                    if (cpuEdges[j].end_node_idx == target_node) {
                        cpuEdges[i].next_edge_idx = j;
                        break;
                    }
                }
            }
        }

        const auto endTime = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> ms = endTime - startTime;
        std::println("Dijkstra Complete in {:.2f} ms!", ms.count());
    }

    void triggerDynamicReroute() {
        // 1. Force the GPU to finish its current frame and pause
        device->waitIdle();
        const bool wasPaused = isPaused;
        isPaused = true;

        // 2. Run the CPU Pathfinding
        recalculateGPS();

        // 3. Upload the newly updated cpuEdges and cpuNodes arrays to VRAM
        {
            const vk::DeviceSize bufferSize = sizeof(GPU_Edge) * totalEdges;
            auto [stagingBuffer, stagingMemory] = createBuffer(
                bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );
            void *mappedData{stagingMemory->mapMemory(0, bufferSize)};
            std::memcpy(mappedData, cpuEdges.data(), bufferSize);
            stagingMemory->unmapMemory();
            copyBuffer(*stagingBuffer, *edgeBuffer, bufferSize);
        }

        {
            const vk::DeviceSize bufferSize = sizeof(GPU_Node) * cpuNodes.size();
            auto [stagingBuffer, stagingMemory] = createBuffer(
                bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );
            void *mappedData{stagingMemory->mapMemory(0, bufferSize)};
            std::memcpy(mappedData, cpuNodes.data(), bufferSize);
            stagingMemory->unmapMemory();
            copyBuffer(*stagingBuffer, *nodeBuffer, bufferSize);
        }

        std::println("GPS Reroute and Node Types Uploaded to VRAM!");

        // 4. Resume the simulation
        isPaused = wasPaused;
    }

    // --- VULKAN ENGINE ---
    void recreateSwapchain() {
        std::int32_t width = 0;
        std::int32_t height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            // Pause execution if window is minimized
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device->waitIdle(); // Ensure GPU isn't using old swapchain resources

        // Destroy the old objects first
        framebuffers.clear();
        swapchainImageViews.clear();
        swapchain.reset();

        // Rebuild them with the new dimensions
        createSwapchain();
        createFramebuffers();

        // Update aspect ratio for rendering
        mapBounds.aspect_ratio = static_cast<float>(swapchainExtent.width) / static_cast<float>(swapchainExtent.height);
    }

    static void framebufferResizeCallback(GLFWwindow *window, std::int32_t width, std::int32_t height) {
        auto *engine = static_cast<EvacuationEngine *>(glfwGetWindowUserPointer(window));
        engine->framebufferResized = true;
    }

    void initImGui() {
        // 1. Create a massive descriptor pool specifically for ImGui
        std::array<vk::DescriptorPoolSize, 11> poolSizes = {
            {
                {vk::DescriptorType::eSampler, 1000},
                {vk::DescriptorType::eCombinedImageSampler, 1000},
                {vk::DescriptorType::eSampledImage, 1000},
                {vk::DescriptorType::eStorageImage, 1000},
                {vk::DescriptorType::eUniformTexelBuffer, 1000},
                {vk::DescriptorType::eStorageTexelBuffer, 1000},
                {vk::DescriptorType::eUniformBuffer, 1000},
                {vk::DescriptorType::eStorageBuffer, 1000},
                {vk::DescriptorType::eUniformBufferDynamic, 1000},
                {vk::DescriptorType::eStorageBufferDynamic, 1000},
                {vk::DescriptorType::eInputAttachment, 1000}
            }
        };

        vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = 1000, .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };
        imguiPool = std::make_unique<vk::raii::DescriptorPool>(*device, poolInfo);

        // 2. Initialize the core ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        const ImGuiIO &io = ImGui::GetIO();
        (void) io;
        ImGui::StyleColorsDark();

        // 3. Initialize the GLFW and Vulkan Backends
        ImGui_ImplGlfw_InitForVulkan(window, true);

        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = **instance;
        init_info.PhysicalDevice = **physicalDevice;
        init_info.Device = **device;
        init_info.QueueFamily = queueFamilyIndex;
        init_info.Queue = **queue;
        init_info.PipelineCache = VK_NULL_HANDLE;
        init_info.DescriptorPool = **imguiPool;
        init_info.MinImageCount = 2;
        init_info.ImageCount = static_cast<std::uint32_t>(swapchainImages.size());
        init_info.Allocator = nullptr;
        init_info.CheckVkResultFn = nullptr;
        init_info.PipelineInfoMain.RenderPass = **renderPass;
        init_info.PipelineInfoMain.Subpass = 0;
        init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

        ImGui_ImplVulkan_Init(&init_info);
        imguiInitialized = true;
    }

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
        window = glfwCreateWindow(static_cast<std::int32_t>(WIDTH), static_cast<std::int32_t>(HEIGHT),
                                  "Vienna Evacuation Simulator", nullptr, nullptr);

        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

        // A. SCROLL CALLBACK (ZOOM IN / OUT)
        glfwSetScrollCallback(window, [](GLFWwindow *win, double xoffset, const double yoffset) {
            // --- THE FIX: Ignore scroll if hovering over the UI ---
            if (ImGui::GetIO().WantCaptureMouse) return;

            auto *engine = static_cast<EvacuationEngine *>(glfwGetWindowUserPointer(win));
            if (yoffset > 0) {
                engine->mapBounds.zoom_level *= 1.15f;
            } else {
                engine->mapBounds.zoom_level /= 1.15f;
            }
            engine->mapBounds.zoom_level = std::clamp(engine->mapBounds.zoom_level, 0.5f, 500.f);
        });

        // B. MOUSE BUTTON CALLBACK (START / STOP DRAG & PICKING)
        glfwSetMouseButtonCallback(
            window, [](GLFWwindow *win, const std::int32_t button, const std::int32_t action, std::int32_t mods) {
                if (ImGui::GetIO().WantCaptureMouse) return;

                auto *engine = static_cast<EvacuationEngine *>(glfwGetWindowUserPointer(win));

                if (button == GLFW_MOUSE_BUTTON_LEFT) {
                    if (action == GLFW_PRESS) {
                        if (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                            glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
                            engine->isSelecting = true;
                            glfwGetCursorPos(win, &engine->startMouseX, &engine->startMouseY);
                            engine->currentMouseX = engine->startMouseX;
                            engine->currentMouseY = engine->startMouseY;
                        } else {
                            engine->isDragging = true;
                            glfwGetCursorPos(win, &engine->lastMouseX, &engine->lastMouseY);
                        }
                    } else if (action == GLFW_RELEASE) {
                        if (engine->isSelecting) {
                            engine->isSelecting = false;
                            engine->applyMarqueeSelection();
                        }
                        engine->isDragging = false;
                    }
                }

                // Right-Click = Inspect Car
                if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                    if (action == GLFW_PRESS) {
                        engine->isInspecting = true;
                        glfwGetCursorPos(win, &engine->inspectMouseX, &engine->inspectMouseY);

                        float worldX = 0.f;
                        float worldY = 0.f;
                        engine->screenToWorld(engine->inspectMouseX, engine->inspectMouseY, worldX, worldY);
                        engine->takeSnapshot();
                        engine->selectClosestCar(worldX, worldY);
                    } else if (action == GLFW_RELEASE) {
                        engine->isInspecting = false;
                    }
                }
            });

        // C. CURSOR POSITION CALLBACK (PANNING THE MAP / UPDATING MARQUEE)
        glfwSetCursorPosCallback(window, [](GLFWwindow *win, const double xpos, const double ypos) {
            auto *engine = static_cast<EvacuationEngine *>(glfwGetWindowUserPointer(win));
            if (engine->isSelecting) {
                engine->currentMouseX = xpos;
                engine->currentMouseY = ypos;
            } else if (engine->isInspecting) {
                engine->inspectMouseX = xpos;
                engine->inspectMouseY = ypos;
            } else if (engine->isDragging) {
                const double deltaX = xpos - engine->lastMouseX;
                const double deltaY = ypos - engine->lastMouseY;

                const float screenFactorX = (engine->mapBounds.extent_height / static_cast<float>(engine->
                                                 swapchainExtent.height));
                const float screenFactorY = (engine->mapBounds.extent_height / static_cast<float>(engine->
                                                 swapchainExtent.height));

                engine->mapBounds.camera_x -= static_cast<float>(deltaX) * (
                    screenFactorX / engine->mapBounds.zoom_level);
                engine->mapBounds.camera_y += static_cast<float>(deltaY) * (
                    screenFactorY / engine->mapBounds.zoom_level);

                engine->lastMouseX = xpos;
                engine->lastMouseY = ypos;
            }
        });

        // D. KEYBOARD CALLBACK
        glfwSetKeyCallback(
            window, [](GLFWwindow *win, const std::int32_t key, const std::int32_t scancode, const std::int32_t action,
                       const std::int32_t mods) {
                // --- THE FIX: Ignore keyboard shortcuts if typing inside an ImGui text box ---
                if (ImGui::GetIO().WantCaptureKeyboard) return;

                auto *engine = static_cast<EvacuationEngine *>(glfwGetWindowUserPointer(win));
                if (action == GLFW_PRESS) {
                    if (key == GLFW_KEY_SPACE) {
                        engine->isPaused = !engine->isPaused;
                        if (engine->isPaused) {
                            engine->takeSnapshot();
                        }
                    } else if (key == GLFW_KEY_UP || key == GLFW_KEY_RIGHT) {
                        engine->simSpeed = std::min(engine->simSpeed * 2, 256);
                    } else if (key == GLFW_KEY_DOWN || key == GLFW_KEY_LEFT) {
                        engine->simSpeed = std::max(engine->simSpeed / 2, 1);
                    }
                }
            });
    }

    void initVulkan() {
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
        context = std::make_unique<vk::raii::Context>();

        constexpr vk::ApplicationInfo appInfo{
            .pApplicationName = "Vienna Evacuation", .applicationVersion = 1,
            .pEngineName = "No Engine", .engineVersion = 1, .apiVersion = VK_API_VERSION_1_3
        };

        std::uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        const vk::InstanceCreateInfo createInfo{
            .pApplicationInfo = &appInfo, .enabledExtensionCount = glfwExtensionCount,
            .ppEnabledExtensionNames = glfwExtensions
        };

        instance = std::make_unique<vk::raii::Instance>(*context, createInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(**instance);

        const vk::raii::PhysicalDevices physicalDevices{*instance};
        physicalDevice = std::make_unique<vk::raii::PhysicalDevice>(physicalDevices.front());

        const std::vector<vk::QueueFamilyProperties> queueFamilies{physicalDevice->getQueueFamilyProperties()};
        for (std::uint32_t i = 0; i < queueFamilies.size(); ++i) {
            if ((queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
                (queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute)) {
                queueFamilyIndex = i;
                break;
            }
        }

        constexpr float queuePriority{1.f};
        const vk::DeviceQueueCreateInfo queueCreateInfo{
            .queueFamilyIndex = queueFamilyIndex, .queueCount = 1, .pQueuePriorities = &queuePriority
        };

        constexpr std::array<const char *, 2> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME, "VK_KHR_shader_draw_parameters"
        };

        vk::PhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.vertexPipelineStoresAndAtomics = VK_TRUE;
        deviceFeatures.fragmentStoresAndAtomics = VK_TRUE;

        const vk::DeviceCreateInfo deviceCreateInfo{
            .queueCreateInfoCount = 1, .pQueueCreateInfos = &queueCreateInfo,
            .enabledExtensionCount = static_cast<std::uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(), .pEnabledFeatures = &deviceFeatures
        };

        device = std::make_unique<vk::raii::Device>(*physicalDevice, deviceCreateInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(**device);
        queue = std::make_unique<vk::raii::Queue>(*device, queueFamilyIndex, 0);

        const vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueFamilyIndex
        };
        commandPool = std::make_unique<vk::raii::CommandPool>(*device, poolInfo);
    }

    [[nodiscard]] auto findMemoryType(const std::uint32_t typeFilter,
                                      const vk::MemoryPropertyFlags properties) const -> std::uint32_t {
        const vk::PhysicalDeviceMemoryProperties memProperties{physicalDevice->getMemoryProperties()};
        for (std::uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error{"Failed to find suitable memory type!"};
    }

    [[nodiscard]] auto createBuffer(const vk::DeviceSize size, const vk::BufferUsageFlags usage,
                                    const vk::MemoryPropertyFlags properties) const -> std::pair<std::unique_ptr<
        vk::raii::Buffer>, std::unique_ptr<vk::raii::DeviceMemory> > {
        const vk::BufferCreateInfo bufferInfo{.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};
        auto newBuffer{std::make_unique<vk::raii::Buffer>(*device, bufferInfo)};

        const vk::MemoryRequirements memRequirements{newBuffer->getMemoryRequirements()};
        const vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };
        auto newMemory{std::make_unique<vk::raii::DeviceMemory>(*device, allocInfo)};

        newBuffer->bindMemory(**newMemory, 0);
        return {std::move(newBuffer), std::move(newMemory)};
    }

    void copyBuffer(const vk::raii::Buffer &srcBuffer, const vk::raii::Buffer &dstBuffer,
                    const vk::DeviceSize size) const {
        const vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = **commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1
        };
        const vk::raii::CommandBuffers commandBuffers{*device, allocInfo};
        const vk::raii::CommandBuffer &cmdBuffer{commandBuffers.front()};

        constexpr vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
        cmdBuffer.begin(beginInfo);
        const vk::BufferCopy copyRegion{.srcOffset = 0, .dstOffset = 0, .size = size};
        cmdBuffer.copyBuffer(*srcBuffer, *dstBuffer, copyRegion);
        cmdBuffer.end();

        const vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &(*cmdBuffer)};
        queue->submit(submitInfo, nullptr);
        queue->waitIdle();
    }

    template<typename T>
    void uploadVectorToGPU(std::span<const T> data, std::unique_ptr<vk::raii::Buffer> &outBuffer,
                           std::unique_ptr<vk::raii::DeviceMemory> &outMemory) {
        const vk::DeviceSize bufferSize = sizeof(T) * data.size();

        auto [stagingBuffer, stagingMemory] = createBuffer(
            bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        void *mappedData{stagingMemory->mapMemory(0, bufferSize)};
        std::memcpy(mappedData, data.data(), static_cast<std::size_t>(bufferSize));
        stagingMemory->unmapMemory();

        auto [deviceLocalBuffer, deviceLocalMemory] = createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        outBuffer = std::move(deviceLocalBuffer);
        outMemory = std::move(deviceLocalMemory);
        copyBuffer(*stagingBuffer, *outBuffer, bufferSize);
    }

    void applyParticipation() {
        device->waitIdle();

        auto rawCars = loadBinaryData<GPU_Car>("vulkan_cars.bin");
        std::vector<std::int32_t> cars_spawned_per_edge(cpuEdges.size(), 0);
        std::vector<std::int32_t> target_per_edge(cpuEdges.size(), 0);

        float fractional_cars = 0.f;
        for (std::size_t i = 0; i < cpuEdges.size(); ++i) {
            const float ideal_cars = static_cast<float>(cpuEdges[i].spawn_capacity) * (participationPercentage / 100.f);
            fractional_cars += ideal_cars;
            const auto spawn_now = static_cast<std::int32_t>(fractional_cars);
            target_per_edge[i] = spawn_now;
            fractional_cars -= static_cast<float>(spawn_now);
        }

        for (auto &c: rawCars) {
            c.next_car_idx = -1;

            const std::int32_t edge_idx = c.current_edge_idx;
            if (edge_idx >= 0 && edge_idx < cpuEdges.size()) {
                if (cars_spawned_per_edge[edge_idx] < target_per_edge[edge_idx]) {
                    cars_spawned_per_edge[edge_idx]++;
                } else {
                    c.state = CarState::Disabled; // Disable this car
                }
            }
        }

        uploadVectorToGPU<GPU_Car>(rawCars, carBuffer, carMemory);
        cpuCars = rawCars;
        updateDescriptorSets();
        takeSnapshot();
    }

    void loadMapDataAndCreateBuffers() {
        std::println("Loading binary data into RAM...");

        auto nodes = loadBinaryData<GPU_Node>("vulkan_nodes.bin");
        auto rawEdges = loadBinaryData<GPU_Edge>("vulkan_edges.bin");
        auto rawCars = loadBinaryData<GPU_Car>("vulkan_cars.bin");

        for (auto &n: nodes) {
            n.lock = -1;
            // Multiply X coordinates by cos(48.2082) to fix map projection distortion
            n.x *= 0.6664f;
        }

        for (auto &e: rawEdges) {
            e.head_car_idx = -1;
            e.garage_lock = -1;
        }
        for (auto &c: rawCars) c.next_car_idx = -1;

        totalCars = static_cast<std::uint32_t>(rawCars.size());
        totalEdges = static_cast<std::uint32_t>(rawEdges.size());

        uploadVectorToGPU<GPU_Node>(nodes, nodeBuffer, nodeMemory);
        uploadVectorToGPU<GPU_Edge>(rawEdges, edgeBuffer, edgeMemory);
        uploadVectorToGPU<GPU_Car>(rawCars, carBuffer, carMemory);

        // --- 1. Find the boundaries using local variables ---
        float temp_min_x = 100000000.f;
        float temp_max_x = -100000000.f;
        float temp_min_y = 100000000.f;
        float temp_max_y = -100000000.f;

        for (const auto &n: nodes) {
            if (n.x < temp_min_x) temp_min_x = n.x;
            if (n.x > temp_max_x) temp_max_x = n.x;
            if (n.y < temp_min_y) temp_min_y = n.y;
            if (n.y > temp_max_y) temp_max_y = n.y;
        }

        // --- 2. Setup the Camera (mapBounds) ---
        const float width_meters = temp_max_x - temp_min_x;
        const float height_meters = temp_max_y - temp_min_y;

        mapBounds.camera_x = temp_min_x + (width_meters * 0.5f);
        mapBounds.camera_y = temp_min_y + (height_meters * 0.5f);
        mapBounds.zoom_level = 1.f;
        mapBounds.extent_width = width_meters;
        mapBounds.extent_height = height_meters;
        mapBounds.aspect_ratio = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);

        // Initialize CPU vectors to hold the data
        cpuCars = rawCars;
        cpuEdges = rawEdges;
        cpuNodes = nodes;

        // Create Readback Buffers (Host Visible + Coherent so the CPU can read them)
        auto createReadback = [&](const vk::DeviceSize size, std::unique_ptr<vk::raii::Buffer> &buf,
                                  std::unique_ptr<vk::raii::DeviceMemory> &mem) {
            auto [b, m] = createBuffer(size, vk::BufferUsageFlagBits::eTransferDst,
                                       vk::MemoryPropertyFlagBits::eHostVisible |
                                       vk::MemoryPropertyFlagBits::eHostCoherent);
            buf = std::move(b);
            mem = std::move(m);
        };

        createReadback(sizeof(GPU_Car) * totalCars, carReadbackBuffer, carReadbackMemory);
        createReadback(sizeof(GPU_Edge) * totalEdges, edgeReadbackBuffer, edgeReadbackMemory);
        createReadback(sizeof(GPU_Node) * nodes.size(), nodeReadbackBuffer, nodeReadbackMemory);

        // --- NEW: BUILD ROUTING GRAPHS ---
        reverseGraph.resize(cpuNodes.size());
        forwardGraph.resize(cpuNodes.size());
        std::set<std::int32_t> exitNodeSet;

        for (std::int32_t i = 0; i < totalEdges; ++i) {
            const GPU_Edge &edge = cpuEdges[i];
            const float travel_time = edge.length / std::max(edge.max_speed, 0.1f);

            reverseGraph[edge.end_node_idx].push_back({
                .source_node = edge.start_node_idx,
                .edge_idx = i,
                .travel_time = travel_time
            });
            forwardGraph[edge.start_node_idx].push_back(i);
        }

        // Detect exits only from nodes marked as exits in the binary file
        for (std::size_t i = 0; i < cpuNodes.size(); ++i) {
            if (cpuNodes[i].type == NodeType::OpenExit || cpuNodes[i].type == NodeType::ClosedExit) {
                exitNodeSet.insert(static_cast<std::int32_t>(i));
            }
        }

        allExitNodes.assign(exitNodeSet.begin(), exitNodeSet.end());
        isExitOpen.resize(allExitNodes.size(), true);
    }

    void createComputePipeline() {
        const auto clearEdgesCode = loadBinaryData<std::uint32_t>("clear_edges.spv");
        const vk::ShaderModuleCreateInfo clearEdgesInfo{
            .codeSize = clearEdgesCode.size() * 4, .pCode = clearEdgesCode.data()
        };
        const vk::raii::ShaderModule clearEdgesShader{*device, clearEdgesInfo};

        const auto buildGridCode = loadBinaryData<std::uint32_t>("build_grid.spv");
        const vk::ShaderModuleCreateInfo buildGridInfo{
            .codeSize = buildGridCode.size() * 4, .pCode = buildGridCode.data()
        };
        const vk::raii::ShaderModule buildGridShader{*device, buildGridInfo};

        const auto physicsCode = loadBinaryData<std::uint32_t>("physics.spv");
        const vk::ShaderModuleCreateInfo physShaderInfo{
            .codeSize = physicsCode.size() * 4, .pCode = physicsCode.data()
        };
        const vk::raii::ShaderModule physShader{*device, physShaderInfo};

        constexpr std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {
            {
                {
                    .binding = 0, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1,
                    .stageFlags = vk::ShaderStageFlagBits::eAll
                },
                {
                    .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1,
                    .stageFlags = vk::ShaderStageFlagBits::eAll
                },
                {
                    .binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1,
                    .stageFlags = vk::ShaderStageFlagBits::eAll
                }
            }
        };

        const vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = static_cast<std::uint32_t>(bindings.size()), .pBindings = bindings.data()
        };
        computeDescriptorSetLayout = std::make_unique<vk::raii::DescriptorSetLayout>(*device, layoutInfo);

        constexpr vk::PushConstantRange pushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(PushConstants)
        };

        const vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1, .pSetLayouts = &(**computeDescriptorSetLayout), .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange
        };
        computePipelineLayout = std::make_unique<vk::raii::PipelineLayout>(*device, pipelineLayoutInfo);

        auto mainEntryPoint = "main";

        const vk::ComputePipelineCreateInfo cePipelineInfo{
            .flags = {},
            .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = *clearEdgesShader, .pName = mainEntryPoint},
            .layout = **computePipelineLayout
        };
        clearEdgesPipeline = std::make_unique<vk::raii::Pipeline>(*device, nullptr, cePipelineInfo);

        const vk::ComputePipelineCreateInfo bgPipelineInfo{
            .flags = {},
            .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = *buildGridShader, .pName = mainEntryPoint},
            .layout = **computePipelineLayout
        };
        buildGridPipeline = std::make_unique<vk::raii::Pipeline>(*device, nullptr, bgPipelineInfo);

        const vk::ComputePipelineCreateInfo physPipelineInfo{
            .flags = {},
            .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = *physShader, .pName = mainEntryPoint},
            .layout = **computePipelineLayout
        };
        physicsPipeline = std::make_unique<vk::raii::Pipeline>(*device, nullptr, physPipelineInfo);

        constexpr vk::DescriptorPoolSize poolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 3};
        const vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, .maxSets = 1, .poolSizeCount = 1,
            .pPoolSizes = &poolSize
        };
        descriptorPool = std::make_unique<vk::raii::DescriptorPool>(*device, poolInfo);

        const vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = **descriptorPool, .descriptorSetCount = 1, .pSetLayouts = &(**computeDescriptorSetLayout)
        };
        computeDescriptorSets = vk::raii::DescriptorSets{*device, allocInfo};

        updateDescriptorSets();
    }

    void updateDescriptorSets() const {
        const vk::DescriptorBufferInfo nodeBufferInfo{.buffer = **nodeBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
        const vk::DescriptorBufferInfo edgeBufferInfo{.buffer = **edgeBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
        const vk::DescriptorBufferInfo carBufferInfo{.buffer = **carBuffer, .offset = 0, .range = VK_WHOLE_SIZE};

        const std::array<vk::WriteDescriptorSet, 3> descriptorWrites = {
            {
                {
                    .dstSet = *computeDescriptorSets[0], .dstBinding = 0, .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &nodeBufferInfo
                },
                {
                    .dstSet = *computeDescriptorSets[0], .dstBinding = 1, .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &edgeBufferInfo
                },
                {
                    .dstSet = *computeDescriptorSets[0], .dstBinding = 2, .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &carBufferInfo
                }
            }
        };
        device->updateDescriptorSets(descriptorWrites, nullptr);
    }

    void createSwapchain() {
        if (!surface) {
            VkSurfaceKHR c_surface;
            if (glfwCreateWindowSurface(**instance, window, nullptr, &c_surface) != VK_SUCCESS)
                throw std::runtime_error{"Failed to create window surface!"};
            surface = std::make_unique<vk::raii::SurfaceKHR>(*instance, c_surface);
        }

        const vk::SurfaceCapabilitiesKHR capabilities{physicalDevice->getSurfaceCapabilitiesKHR(**surface)};
        const std::vector<vk::SurfaceFormatKHR> formats{physicalDevice->getSurfaceFormatsKHR(**surface)};

        vk::SurfaceFormatKHR surfaceFormat = formats[0];
        for (const auto &availableFormat: formats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace ==
                vk::ColorSpaceKHR::eSrgbNonlinear) {
                surfaceFormat = availableFormat;
                break;
            }
        }
        swapchainImageFormat = surfaceFormat.format;

        // FETCH TRUE DIMENSIONS FROM GLFW (Fixes Extent Validation Error)
        std::int32_t width = 0;
        std::int32_t height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        swapchainExtent = vk::Extent2D{
            std::clamp(static_cast<std::uint32_t>(width), capabilities.minImageExtent.width,
                       capabilities.maxImageExtent.width),
            std::clamp(static_cast<std::uint32_t>(height), capabilities.minImageExtent.height,
                       capabilities.maxImageExtent.height)
        };

        std::uint32_t imageCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
            imageCount = capabilities.maxImageCount;

        const vk::SwapchainCreateInfoKHR createInfo{
            .flags = {}, .surface = **surface, .minImageCount = imageCount, .imageFormat = surfaceFormat.format,
            .imageColorSpace = surfaceFormat.colorSpace, .imageExtent = swapchainExtent, .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment, .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = capabilities.currentTransform, .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = vk::PresentModeKHR::eFifo, .clipped = VK_TRUE
        };

        swapchain = std::make_unique<vk::raii::SwapchainKHR>(*device, createInfo);
        swapchainImages = swapchain->getImages();

        for (const auto &image: swapchainImages) {
            constexpr vk::ComponentMapping components{
                .r = vk::ComponentSwizzle::eIdentity, .g = vk::ComponentSwizzle::eIdentity,
                .b = vk::ComponentSwizzle::eIdentity, .a = vk::ComponentSwizzle::eIdentity
            };
            constexpr vk::ImageSubresourceRange subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0,
                .layerCount = 1
            };
            const vk::ImageViewCreateInfo viewInfo{
                .flags = {}, .image = image, .viewType = vk::ImageViewType::e2D, .format = swapchainImageFormat,
                .components = components, .subresourceRange = subresourceRange
            };
            swapchainImageViews.emplace_back(*device, viewInfo);
        }

        // Update aspect ratio for rendering
        mapBounds.aspect_ratio = static_cast<float>(swapchainExtent.width) / static_cast<float>(swapchainExtent.height);
    }

    void createRenderPass() {
        const vk::AttachmentDescription colorAttachment{
            .flags = {}, .format = swapchainImageFormat, .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare, .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined, .finalLayout = vk::ImageLayout::ePresentSrcKHR
        };
        constexpr vk::AttachmentReference colorAttachmentRef{
            .attachment = 0, .layout = vk::ImageLayout::eColorAttachmentOptimal
        };
        const vk::SubpassDescription subpass{
            .flags = {}, .pipelineBindPoint = vk::PipelineBindPoint::eGraphics, .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentRef
        };
        const vk::RenderPassCreateInfo renderPassInfo{
            .flags = {}, .attachmentCount = 1, .pAttachments = &colorAttachment, .subpassCount = 1,
            .pSubpasses = &subpass
        };

        renderPass = std::make_unique<vk::raii::RenderPass>(*device, renderPassInfo);
    }

    void createFramebuffers() {
        for (const auto &imageView: swapchainImageViews) {
            const std::array<vk::ImageView, 1> attachments = {*imageView};
            const vk::FramebufferCreateInfo framebufferInfo{
                .flags = {}, .renderPass = **renderPass, .attachmentCount = 1, .pAttachments = attachments.data(),
                .width = swapchainExtent.width, .height = swapchainExtent.height, .layers = 1
            };
            framebuffers.emplace_back(*device, framebufferInfo);
        }
    }

    void createGraphicsPipeline() {
        const auto vertCode = loadBinaryData<std::uint32_t>("graphics_vert.spv");
        const auto fragCode = loadBinaryData<std::uint32_t>("graphics_frag.spv");

        const vk::raii::ShaderModule vertModule{
            *device, vk::ShaderModuleCreateInfo{.codeSize = vertCode.size() * 4, .pCode = vertCode.data()}
        };
        const vk::raii::ShaderModule fragModule{
            *device, vk::ShaderModuleCreateInfo{.codeSize = fragCode.size() * 4, .pCode = fragCode.data()}
        };

        const std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .flags = {}, .stage = vk::ShaderStageFlagBits::eVertex, .module = *vertModule, .pName = "main"
            },
            vk::PipelineShaderStageCreateInfo{
                .flags = {}, .stage = vk::ShaderStageFlagBits::eFragment, .module = *fragModule, .pName = "main"
            }
        };

        constexpr vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
        constexpr vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList
        };

        // --- NEW: DYNAMIC STATES ---
        // This allows us to resize the window WITHOUT rebuilding the entire pipeline
        constexpr std::array<vk::DynamicState, 2> dynamicStates = {
            vk::DynamicState::eViewport, vk::DynamicState::eScissor
        };
        const vk::PipelineDynamicStateCreateInfo dynamicStateInfo{
            .dynamicStateCount = 2, .pDynamicStates = dynamicStates.data()
        };

        // Viewports are now ignored at build time due to dynamic state, but struct must exist
        constexpr vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

        constexpr vk::PipelineRasterizationStateCreateInfo rasterizer{
            .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eNone, .lineWidth = 1.f
        };
        constexpr vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1
        };
        constexpr vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = VK_FALSE,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };
        const vk::PipelineColorBlendStateCreateInfo colorBlending{
            .attachmentCount = 1, .pAttachments = &colorBlendAttachment
        };
        constexpr vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = VK_FALSE, .depthWriteEnable = VK_FALSE, .depthCompareOp = vk::CompareOp::eLessOrEqual,
            .stencilTestEnable = VK_FALSE
        };

        constexpr vk::PushConstantRange pushRange{
            .stageFlags = vk::ShaderStageFlagBits::eAllGraphics, .offset = 0, .size = sizeof(GraphicsConstants)
        };
        const vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1, .pSetLayouts = &(**computeDescriptorSetLayout), .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushRange
        };
        graphicsPipelineLayout = std::make_unique<vk::raii::PipelineLayout>(*device, pipelineLayoutInfo);

        const vk::GraphicsPipelineCreateInfo pipelineInfo{
            .flags = {}, .stageCount = 2, .pStages = shaderStages.data(), .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly, .pViewportState = &viewportState, .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling, .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicStateInfo, .layout = **graphicsPipelineLayout, .renderPass = **renderPass,
            .subpass = 0
        };
        graphicsPipeline = std::make_unique<vk::raii::Pipeline>(*device, nullptr, pipelineInfo);

        // --- STREET PIPELINE ---
        const auto streetVertCode = loadBinaryData<std::uint32_t>("streets_vert.spv");
        const auto streetFragCode = loadBinaryData<std::uint32_t>("streets_frag.spv");
        const vk::raii::ShaderModule streetVertModule{
            *device, vk::ShaderModuleCreateInfo{.codeSize = streetVertCode.size() * 4, .pCode = streetVertCode.data()}
        };
        const vk::raii::ShaderModule streetFragModule{
            *device, vk::ShaderModuleCreateInfo{.codeSize = streetFragCode.size() * 4, .pCode = streetFragCode.data()}
        };

        const std::array<vk::PipelineShaderStageCreateInfo, 2> streetShaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .flags = {}, .stage = vk::ShaderStageFlagBits::eVertex, .module = *streetVertModule, .pName = "main"
            },
            vk::PipelineShaderStageCreateInfo{
                .flags = {}, .stage = vk::ShaderStageFlagBits::eFragment, .module = *streetFragModule, .pName = "main"
            }
        };

        constexpr vk::PipelineInputAssemblyStateCreateInfo lineAssembly{.topology = vk::PrimitiveTopology::eLineList};

        vk::GraphicsPipelineCreateInfo streetPipelineInfo = pipelineInfo;
        streetPipelineInfo.pStages = streetShaderStages.data();
        streetPipelineInfo.pInputAssemblyState = &lineAssembly;

        streetPipeline = std::make_unique<vk::raii::Pipeline>(*device, nullptr, streetPipelineInfo);

        // --- EXIT NODE PIPELINE ---
        const auto exitNodeVertCode = loadBinaryData<std::uint32_t>("exit_nodes_vert.spv");
        const auto exitNodeFragCode = loadBinaryData<std::uint32_t>("exit_nodes_frag.spv");
        const vk::raii::ShaderModule exitNodeVertModule{
            *device,
            vk::ShaderModuleCreateInfo{.codeSize = exitNodeVertCode.size() * 4, .pCode = exitNodeVertCode.data()}
        };
        const vk::raii::ShaderModule exitNodeFragModule{
            *device,
            vk::ShaderModuleCreateInfo{.codeSize = exitNodeFragCode.size() * 4, .pCode = exitNodeFragCode.data()}
        };

        const std::array<vk::PipelineShaderStageCreateInfo, 2> exitNodeShaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .flags = {}, .stage = vk::ShaderStageFlagBits::eVertex, .module = *exitNodeVertModule, .pName = "main"
            },
            vk::PipelineShaderStageCreateInfo{
                .flags = {}, .stage = vk::ShaderStageFlagBits::eFragment, .module = *exitNodeFragModule, .pName = "main"
            }
        };

        vk::GraphicsPipelineCreateInfo exitNodePipelineInfo = pipelineInfo;
        exitNodePipelineInfo.pStages = exitNodeShaderStages.data();

        exitNodePipeline = std::make_unique<vk::raii::Pipeline>(*device, nullptr, exitNodePipelineInfo);
    }

    void mainLoop() {
        const vk::CommandBufferAllocateInfo cmdAllocInfo{
            .commandPool = **commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1
        };
        const vk::raii::CommandBuffers cmdBuffers{*device, cmdAllocInfo};
        const vk::raii::CommandBuffer &cmd{cmdBuffers.front()};

        const vk::raii::Semaphore imageAvailableSemaphore{*device, vk::SemaphoreCreateInfo{}};
        const vk::raii::Semaphore renderFinishedSemaphore{*device, vk::SemaphoreCreateInfo{}};
        constexpr vk::ClearValue clearColor{
            .color = vk::ClearColorValue{std::array<float, 4>{0.05f, 0.05f, 0.1f, 1.f}}
        };

        double lastFrameTime = glfwGetTime();
        double accumulatedSimTimeQueue = 0.;

        // Take an initial snapshot so the UI has accurate stats on startup
        takeSnapshot();

        double lastSnapshotSimTime = simTime;

        while (glfwWindowShouldClose(window) == 0) {
            glfwPollEvents();
            if (glfwWindowShouldClose(window)) break;

            const double currentFrameTime = glfwGetTime();
            const double dt_real = currentFrameTime - lastFrameTime;
            lastFrameTime = currentFrameTime;

            // Periodically take a snapshot to update statistics, run dynamic GPS rerouting, and check if finished
            // Runs exactly every 60. seconds of simulation time, fully independent of real-world time
            if (!isPaused && (simTime - lastSnapshotSimTime >= 60.)) {
                takeSnapshot();

                // Recalculate routes dynamically based on new car counts and congestion
                recalculateGPS();

                // Upload the newly updated cpuEdges array (specifically next_edge_idx) to VRAM
                {
                    const vk::DeviceSize bufferSize = sizeof(GPU_Edge) * totalEdges;
                    auto [stagingBuffer, stagingMemory] = createBuffer(
                        bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
                    );
                    void *mappedData{stagingMemory->mapMemory(0, bufferSize)};
                    std::memcpy(mappedData, cpuEdges.data(), bufferSize);
                    stagingMemory->unmapMemory();
                    copyBuffer(*stagingBuffer, *edgeBuffer, bufferSize);
                }

                lastSnapshotSimTime = simTime;

                // Stop simulation if finished (no active cars on the road and no cars left in the garage)
                if (statRoad == 0 && statGarage == 0) {
                    isPaused = true;
                }
            }

            // Read back the selected car's updated data, and its current edge/node, from the previous frame's copy
            if (!isPaused && selectedCarId >= 0 && selectedCarId < static_cast<std::int32_t>(totalCars) && !cpuCars.
                empty()) {
                {
                    const vk::DeviceSize offset = sizeof(GPU_Car) * selectedCarId;
                    const void *mappedData = carReadbackMemory->mapMemory(offset, sizeof(GPU_Car));
                    std::memcpy(&cpuCars[selectedCarId], mappedData, sizeof(GPU_Car));
                    carReadbackMemory->unmapMemory();
                }

                const std::int32_t edgeIdx = cpuCars[selectedCarId].current_edge_idx;
                if (edgeIdx >= 0 && edgeIdx < static_cast<std::int32_t>(totalEdges)) {
                    const vk::DeviceSize offset = sizeof(GPU_Edge) * edgeIdx;
                    const void *mappedData = edgeReadbackMemory->mapMemory(offset, sizeof(GPU_Edge));
                    std::memcpy(&cpuEdges[edgeIdx], mappedData, sizeof(GPU_Edge));
                    edgeReadbackMemory->unmapMemory();

                    const std::int32_t nodeIdx = cpuEdges[edgeIdx].end_node_idx;
                    if (nodeIdx >= 0 && nodeIdx < static_cast<std::int32_t>(cpuNodes.size())) {
                        const vk::DeviceSize nodeOffset = sizeof(GPU_Node) * nodeIdx;
                        const void *mappedNodeData = nodeReadbackMemory->mapMemory(nodeOffset, sizeof(GPU_Node));
                        std::memcpy(&cpuNodes[nodeIdx], mappedNodeData, sizeof(GPU_Node));
                        nodeReadbackMemory->unmapMemory();
                    }
                }
            }

            // --- THE FIX: Recreate the swapchain when the window resizes ---
            if (framebufferResized) {
                framebufferResized = false;
                recreateSwapchain();
            }

            std::uint32_t imageIndex;
            try {
                auto [result, index] = swapchain->acquireNextImage(UINT64_MAX, *imageAvailableSemaphore, nullptr);
                // Suboptimal usually means the window is resizing but hasn't settled yet
                if (result == vk::Result::eSuboptimalKHR) {
                    framebufferResized = true;
                }
                imageIndex = index;
            } catch (const vk::OutOfDateKHRError &) {
                // OutOfDate means the swapchain is completely invalid (e.g., monitor changed, maximized)
                framebufferResized = false;
                recreateSwapchain();
                continue; // Restart the loop immediately with the new valid swapchain
            }

            // --- 1. START IMGUI FRAME ---
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // --- 2. BUILD YOUR UI ---
            ImGui::Begin("Simulation Control Panel");
            ImGui::Text("Performance: %.3f ms/frame (%.1f FPS)", 1000.f / ImGui::GetIO().Framerate,
                        ImGui::GetIO().Framerate);
            ImGui::Separator();

            // ==========================================
            // --- EMERGENCY SCENARIO CONTROL ---
            // ==========================================
            ImGui::Separator();
            ImGui::Text("--- EMERGENCY SCENARIO CONTROL ---");

            bool routingChanged = false;


            if (ImGui::Button("Open All Exits")) {
                std::ranges::fill(isExitOpen, true);
                routingChanged = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Close All Exits")) {
                std::ranges::fill(isExitOpen, false);
                routingChanged = true;
            }

            ImGui::Spacing();

            ImGui::BeginDisabled(hasStarted);
            ImGui::SliderFloat("Participation (%)", &participationPercentage, 0.f, 100.f, "%.1f%%");
            ImGui::EndDisabled();

            ImGui::Spacing();

            if (routingChanged) {
                triggerDynamicReroute();
            }
            // ==========================================

            ImGui::Separator();

            if (ImGui::Button(isPaused ? "Play Simulation" : "Pause Simulation", ImVec2(-1.f, 40.f))) {
                if (!hasStarted) {
                    applyParticipation();
                    hasStarted = true;
                }
                isPaused = !isPaused;
                if (isPaused) {
                    takeSnapshot();
                }
            }
            ImGui::Spacing();
            ImGui::SliderInt("Simulation Speed", &simSpeed, 1, 256, "%d x");

            ImGui::Separator();
            ImGui::Text("--- EVACUATION METRICS ---");
            {
                std::int32_t hours = static_cast<std::int32_t>(simTime) / 3600;
                std::int32_t minutes = (static_cast<std::int32_t>(simTime) % 3600) / 60;
                float seconds = std::fmod(simTime, 60.f);
                ImGui::Text("Simulation Time: %02d:%02d:%04.1f", hours, minutes, seconds);
            }
            ImGui::Text("Waiting in Garage: %d", statGarage);
            ImGui::Text("Active on Road: %d", statRoad);
            ImGui::Text("Safely Evacuated: %d", statEvacuated);
            ImGui::Text("Stuck Cars: %d", statStuck);
            ImGui::Text("City Average Speed: %.1f km/h", statAvgSpeed * 3.6f);

            // Progress Bar!
            std::int32_t participatingCars = static_cast<std::int32_t>(totalCars) - statDisabled;
            float progress = participatingCars > 0
                                 ? static_cast<float>(statEvacuated) / static_cast<float>(participatingCars)
                                 : 0.f;
            std::string progressText = std::format("Evacuation Progress: {:.1f}%", progress * 100.f);
            ImGui::ProgressBar(progress, ImVec2(-1.f, 0.f), progressText.c_str());

            // Update Metrics History (Every 60 simulation seconds)
            if (simTime - lastRecordTime >= 60.f) {
                const auto flow = static_cast<float>(statEvacuated - lastEvacuatedCount); // cars per minute
                flowrateHistory.push_back(flow);
                evacuatedHistory.push_back(static_cast<float>(statEvacuated));
                lastEvacuatedCount = statEvacuated;
                lastRecordTime = simTime;
            }

            if (!flowrateHistory.empty()) {
                ImGui::Spacing();
                ImGui::Text("--- CHARTS ---");
                // Box chart (Histogram) for flow rate
                std::string flowText = std::format("{:.0f} cars/min", flowrateHistory.back());
                ImGui::PlotHistogram("Flowrate (cars/min)", flowrateHistory.data(),
                                     static_cast<std::int32_t>(flowrateHistory.size()),
                                     0, flowText.c_str(), 0.f, std::numeric_limits<float>::max(), ImVec2(0.f, 120.f));

                // Line chart for total evacuated
                std::string evacText = std::format("{:.0f} evacuated", evacuatedHistory.back());
                ImGui::PlotLines("Total Evacuated", evacuatedHistory.data(),
                                 static_cast<std::int32_t>(evacuatedHistory.size()),
                                 0, evacText.c_str(), 0.f, std::numeric_limits<float>::max(), ImVec2(0.f, 120.f));
            }

            if (!cpuCars.empty()) {
                ImGui::Spacing();
                selectedCarId = std::clamp(selectedCarId, 0, static_cast<std::int32_t>(totalCars) - 1);

                GPU_Car &car = cpuCars[selectedCarId];

                ImGui::BeginChild("CarData", ImVec2(0, 150), true);
                ImGui::Text("--- CAR %d ---", selectedCarId);

                const char *stateStr = "Unknown";
                if (car.state == CarState::Driving) stateStr = "Driving";
                if (car.state == CarState::Queuing) stateStr = "Queuing at Intersection";
                if (car.state == CarState::Evacuated) stateStr = "Evacuated!";
                if (car.state == CarState::Garage) stateStr = "In Garage";
                if (car.state == CarState::Stuck) stateStr = "Stuck";

                ImGui::Text("State: %s (%d)", stateStr, car.state);
                ImGui::Text("Speed: %.2f m/s (%.1f km/h)", car.speed, car.speed * 3.6f);
                ImGui::Text("Position: %.2f meters", car.position);
                ImGui::Text("Current Edge ID: %d", car.current_edge_idx);
                ImGui::Text("Next Car in Linked List: %d", car.next_car_idx);
                ImGui::EndChild();

                // If the car is on an edge, show the edge data too!
                if (car.current_edge_idx != -1) {
                    GPU_Edge &edge = cpuEdges[car.current_edge_idx];
                    ImGui::BeginChild("EdgeData", ImVec2(0, 140), true);
                    ImGui::Text("--- EDGE %d ---", car.current_edge_idx);
                    ImGui::Text("Length: %.2f meters", edge.length);
                    ImGui::Text("Max Speed: %.1f km/h", edge.max_speed * 3.6f);
                    ImGui::Text("Head Car ID: %d", edge.head_car_idx);
                    ImGui::Text("Target Node ID: %d", edge.end_node_idx);
                    ImGui::Text("Spawn Capacity: %d", edge.spawn_capacity);
                    ImGui::EndChild();
                }
            }
            ImGui::End();

            if (isSelecting) {
                ImDrawList *drawList = ImGui::GetForegroundDrawList();
                ImVec2 p_min(
                    static_cast<float>(std::min(startMouseX, currentMouseX)),
                    static_cast<float>(std::min(startMouseY, currentMouseY))
                );
                ImVec2 p_max(
                    static_cast<float>(std::max(startMouseX, currentMouseX)),
                    static_cast<float>(std::max(startMouseY, currentMouseY))
                );
                drawList->AddRectFilled(p_min, p_max, IM_COL32(0, 150, 255, 60), 0.f);
                drawList->AddRect(p_min, p_max, IM_COL32(0, 150, 255, 255), 0.f, 0, 2.f);
            }

            if (isInspecting) {
                ImDrawList *drawList = ImGui::GetForegroundDrawList();
                ImVec2 center(static_cast<float>(inspectMouseX), static_cast<float>(inspectMouseY));
                float r = worldDistanceToPixels(20.f / 111300.f); // 20 meters selection radius in degrees
                drawList->AddCircleFilled(center, r, IM_COL32(255, 165, 0, 40), 64);
                drawList->AddCircle(center, r, IM_COL32(255, 165, 0, 180), 64, 2.f);
            }


            // --- 3. FINALIZE UI DATA ---
            ImGui::Render();

            constexpr vk::CommandBufferBeginInfo cmdBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
            cmd.begin(cmdBeginInfo);

            std::int32_t stepsToRun = 0;
            if (!isPaused) {
                const double capped_dt = std::min(dt_real, 0.033);
                accumulatedSimTimeQueue += capped_dt * simSpeed;
                while (accumulatedSimTimeQueue >= 0.1) {
                    stepsToRun++;
                    accumulatedSimTimeQueue -= 0.1;
                }
                if (stepsToRun > 256) {
                    stepsToRun = 256;
                    accumulatedSimTimeQueue = 0.;
                }
            }

            if (stepsToRun > 0) {
                simTime += static_cast<float>(stepsToRun) * 0.1f;
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, **computePipelineLayout, 0,
                                       {*computeDescriptorSets[0]}, nullptr);
                const PushConstants pushData{.dt = 0.1f, .num_cars = totalCars, .num_edges = totalEdges};
                cmd.pushConstants<PushConstants>(**computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                                 pushData);

                for (std::int32_t step = 0; step < stepsToRun; ++step) {
                    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, **clearEdgesPipeline);
                    cmd.dispatch((totalEdges + 255) / 256, 1, 1);

                    const vk::BufferMemoryBarrier edgeBarrier{
                        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                        .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .buffer = **edgeBuffer, .offset = 0, .size = VK_WHOLE_SIZE
                    };
                    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                        vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, edgeBarrier, nullptr);

                    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, **buildGridPipeline);
                    cmd.dispatch((totalCars + 255) / 256, 1, 1);

                    const vk::BufferMemoryBarrier carBarrier{
                        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                        .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .buffer = **carBuffer, .offset = 0, .size = VK_WHOLE_SIZE
                    };
                    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                        vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, carBarrier, nullptr);

                    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, **physicsPipeline);
                    cmd.dispatch((totalCars + 255) / 256, 1, 1);

                    vk::PipelineStageFlags dstStage = (step == stepsToRun - 1)
                                                          ? vk::PipelineStageFlagBits::eVertexShader
                                                          : vk::PipelineStageFlagBits::eComputeShader;
                    constexpr vk::MemoryBarrier stepBarrier{
                        .srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
                        .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite
                    };
                    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, dstStage, {}, stepBarrier, nullptr,
                                        nullptr);
                }

                // Copy the selected car's, its edge's, and target node's data back to the host visible buffer for dynamic inspection
                std::int32_t inspectEdgeIdx = -1;
                std::int32_t inspectNodeIdx = -1;
                if (selectedCarId >= 0 && selectedCarId < static_cast<std::int32_t>(totalCars) && !cpuCars.empty()) {
                    inspectEdgeIdx = cpuCars[selectedCarId].current_edge_idx;
                    if (inspectEdgeIdx >= 0 && inspectEdgeIdx < static_cast<std::int32_t>(totalEdges)) {
                        inspectNodeIdx = cpuEdges[inspectEdgeIdx].end_node_idx;
                    }
                }

                std::vector<vk::BufferMemoryBarrier> barriers;
                if (selectedCarId >= 0 && selectedCarId < static_cast<std::int32_t>(totalCars)) {
                    barriers.push_back(vk::BufferMemoryBarrier{
                        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .buffer = **carBuffer,
                        .offset = sizeof(GPU_Car) * selectedCarId,
                        .size = sizeof(GPU_Car)
                    });
                }
                if (inspectEdgeIdx >= 0 && inspectEdgeIdx < static_cast<std::int32_t>(totalEdges)) {
                    barriers.push_back(vk::BufferMemoryBarrier{
                        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .buffer = **edgeBuffer,
                        .offset = sizeof(GPU_Edge) * inspectEdgeIdx,
                        .size = sizeof(GPU_Edge)
                    });
                }
                if (inspectNodeIdx >= 0 && inspectNodeIdx < static_cast<std::int32_t>(cpuNodes.size())) {
                    barriers.push_back(vk::BufferMemoryBarrier{
                        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .buffer = **nodeBuffer,
                        .offset = sizeof(GPU_Node) * inspectNodeIdx,
                        .size = sizeof(GPU_Node)
                    });
                }

                if (!barriers.empty()) {
                    cmd.pipelineBarrier(
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eTransfer,
                        {},
                        nullptr,
                        barriers,
                        nullptr
                    );
                }

                if (selectedCarId >= 0 && selectedCarId < static_cast<std::int32_t>(totalCars)) {
                    const vk::BufferCopy copyRegion{
                        .srcOffset = sizeof(GPU_Car) * selectedCarId,
                        .dstOffset = sizeof(GPU_Car) * selectedCarId,
                        .size = sizeof(GPU_Car)
                    };
                    cmd.copyBuffer(**carBuffer, **carReadbackBuffer, copyRegion);
                }
                if (inspectEdgeIdx >= 0 && inspectEdgeIdx < static_cast<std::int32_t>(totalEdges)) {
                    const vk::BufferCopy copyRegion{
                        .srcOffset = sizeof(GPU_Edge) * inspectEdgeIdx,
                        .dstOffset = sizeof(GPU_Edge) * inspectEdgeIdx,
                        .size = sizeof(GPU_Edge)
                    };
                    cmd.copyBuffer(**edgeBuffer, **edgeReadbackBuffer, copyRegion);
                }
                if (inspectNodeIdx >= 0 && inspectNodeIdx < static_cast<std::int32_t>(cpuNodes.size())) {
                    const vk::BufferCopy copyRegion{
                        .srcOffset = sizeof(GPU_Node) * inspectNodeIdx,
                        .dstOffset = sizeof(GPU_Node) * inspectNodeIdx,
                        .size = sizeof(GPU_Node)
                    };
                    cmd.copyBuffer(**nodeBuffer, **nodeReadbackBuffer, copyRegion);
                }
            }

            const vk::RenderPassBeginInfo renderPassInfo{
                .renderPass = **renderPass, .framebuffer = *framebuffers[imageIndex],
                .renderArea = {.offset = {0, 0}, .extent = swapchainExtent}, .clearValueCount = 1,
                .pClearValues = &clearColor
            };
            cmd.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

            // Set Dynamic Viewport & Scissor
            const vk::Viewport dynamicViewport{
                .x = 0.f, .y = 0.f, .width = static_cast<float>(swapchainExtent.width),
                .height = static_cast<float>(swapchainExtent.height), .minDepth = 0.f, .maxDepth = 1.f
            };
            cmd.setViewport(0, dynamicViewport);
            const vk::Rect2D dynamicScissor{.offset = {0, 0}, .extent = swapchainExtent};
            cmd.setScissor(0, dynamicScissor);

            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, **streetPipeline);

            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **graphicsPipelineLayout, 0,
                                   {*computeDescriptorSets[0]}, nullptr);
            mapBounds.selected_car_id = selectedCarId;
            cmd.pushConstants<GraphicsConstants>(**graphicsPipelineLayout, vk::ShaderStageFlagBits::eAllGraphics, 0,
                                                 mapBounds);

            // Draw the Map
            cmd.draw(2, totalEdges, 0, 0);

            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, **graphicsPipeline);
            cmd.draw(6, totalCars, 0, 0);

            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, **exitNodePipeline);
            cmd.draw(6, totalEdges, 0, 0);

            // --- 4. DRAW IMGUI ON TOP OF THE MAP ---
            // ImGui uses its OWN internal pipeline and push constants.
            // By calling it LAST, it ignores the mapBounds we pushed above!
            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *cmd);

            cmd.endRenderPass();
            cmd.end();

            constexpr std::array<vk::PipelineStageFlags, 1> waitStages = {
                vk::PipelineStageFlagBits::eColorAttachmentOutput
            };
            const vk::SubmitInfo submitInfo{
                .waitSemaphoreCount = 1, .pWaitSemaphores = &(*imageAvailableSemaphore),
                .pWaitDstStageMask = waitStages.data(), .commandBufferCount = 1, .pCommandBuffers = &(*cmd),
                .signalSemaphoreCount = 1, .pSignalSemaphores = &(*renderFinishedSemaphore)
            };
            queue->submit(submitInfo, nullptr);

            const vk::PresentInfoKHR presentInfo{
                .waitSemaphoreCount = 1, .pWaitSemaphores = &(*renderFinishedSemaphore), .swapchainCount = 1,
                .pSwapchains = &(**swapchain), .pImageIndices = &imageIndex
            };

            try {
                auto presentResult = queue->presentKHR(presentInfo);
                if (presentResult == vk::Result::eSuboptimalKHR || framebufferResized) {
                    framebufferResized = false;
                    recreateSwapchain();
                }
            } catch (const vk::OutOfDateKHRError &) {
                framebufferResized = false;
                recreateSwapchain();
            }

            queue->waitIdle();
        }
        device->waitIdle();
    }

    void takeSnapshot() {
        // 1. Record the copy commands
        const vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = **commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1
        };
        const vk::raii::CommandBuffers cmdBuffers{*device, allocInfo};
        const vk::raii::CommandBuffer &cmd = cmdBuffers.front();

        cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        const vk::BufferCopy carCopy{.srcOffset = 0, .dstOffset = 0, .size = sizeof(GPU_Car) * totalCars};
        cmd.copyBuffer(**carBuffer, **carReadbackBuffer, carCopy);

        const vk::BufferCopy edgeCopy{.srcOffset = 0, .dstOffset = 0, .size = sizeof(GPU_Edge) * totalEdges};
        cmd.copyBuffer(**edgeBuffer, **edgeReadbackBuffer, edgeCopy);

        const vk::BufferCopy nodeCopy{.srcOffset = 0, .dstOffset = 0, .size = sizeof(GPU_Node) * cpuNodes.size()};
        cmd.copyBuffer(**nodeBuffer, **nodeReadbackBuffer, nodeCopy);

        cmd.end();

        // 2. Submit and wait for the GPU to finish copying
        queue->submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &(*cmd)}, nullptr);
        queue->waitIdle();

        // 3. Map the memory and copy it into our C++ vectors
        const void *mappedCars = carReadbackMemory->mapMemory(0, sizeof(GPU_Car) * totalCars);
        std::memcpy(cpuCars.data(), mappedCars, sizeof(GPU_Car) * totalCars);
        carReadbackMemory->unmapMemory();

        const void *mappedEdges = edgeReadbackMemory->mapMemory(0, sizeof(GPU_Edge) * totalEdges);
        std::memcpy(cpuEdges.data(), mappedEdges, sizeof(GPU_Edge) * totalEdges);
        edgeReadbackMemory->unmapMemory();

        const void *mappedNodes = nodeReadbackMemory->mapMemory(0, sizeof(GPU_Node) * cpuNodes.size());
        std::memcpy(cpuNodes.data(), mappedNodes, sizeof(GPU_Node) * cpuNodes.size());
        nodeReadbackMemory->unmapMemory();

        statGarage = 0;
        statRoad = 0;
        statEvacuated = 0;
        statStuck = 0;
        statDisabled = 0;
        statAvgSpeed = 0.f;
        double totalSpeed = 0.;

        for (const auto &car: cpuCars) {
            if (car.state == CarState::Driving || car.state == CarState::Queuing) {
                statRoad++;
                totalSpeed += car.speed;
            } else if (car.state == CarState::Evacuated) {
                statEvacuated++;
            } else if (car.state == CarState::Garage) {
                statGarage++;
            } else if (car.state == CarState::Stuck) {
                statStuck++;
            } else if (car.state == CarState::Disabled) {
                statDisabled++;
            }
        }
        statAvgSpeed = statRoad > 0 ? static_cast<float>(totalSpeed / statRoad) : 0.f;
    }

    // Converts raw GLFW window pixels into Vienna World Coordinates (Meters)
    void screenToWorld(const double screenX, const double screenY, float &outWorldX, float &outWorldY) const {
        // 1. Convert Screen Pixels to Normalized Device Coordinates (NDC: -1. to 1.)
        float ndcX = (static_cast<float>(screenX) / static_cast<float>(swapchainExtent.width)) * 2.f - 1.f;
        // Vulkan's Y axis points down, but our world map points up, so we invert Y
        const float ndcY = -((static_cast<float>(screenY) / static_cast<float>(swapchainExtent.height)) * 2.f - 1.f);

        // 2. Reverse the Aspect Ratio correction
        ndcX *= mapBounds.aspect_ratio;

        // 3. Reverse the Zoom and Extent scaling
        const float localX = (ndcX * (mapBounds.extent_height * 0.5f)) / mapBounds.zoom_level;
        const float localY = (ndcY * (mapBounds.extent_height * 0.5f)) / mapBounds.zoom_level;

        // 4. Reverse the Camera Pan
        outWorldX = localX + mapBounds.camera_x;
        outWorldY = localY + mapBounds.camera_y;
    }

    [[nodiscard]] float worldDistanceToPixels(const float distanceMeters) const {
        return distanceMeters * mapBounds.zoom_level * static_cast<float>(swapchainExtent.height) / mapBounds.
               extent_height;
    }

    void selectClosestCar(const float worldX, const float worldY) {
        if (cpuCars.empty()) return; // Must take a snapshot first!

        std::int32_t closestCarId = -1;
        float closestDistSq = std::numeric_limits<float>::max();

        for (std::size_t i = 0; i < cpuCars.size(); ++i) {
            const GPU_Car &car = cpuCars[i];

            // Only select cars that are actually on the road
            if (car.state == CarState::Evacuated || car.state == CarState::Garage)
                continue;

            const GPU_Edge &edge = cpuEdges[car.current_edge_idx];
            const GPU_Node &startNode = cpuNodes[edge.start_node_idx];
            const GPU_Node &endNode = cpuNodes[edge.end_node_idx];

            // Interpolate car's world position along the edge
            const float t = (edge.length > 0.f) ? (car.position / edge.length) : 0.f;
            const float carWorldX = startNode.x + t * (endNode.x - startNode.x);
            const float carWorldY = startNode.y + t * (endNode.y - startNode.y);

            // Calculate distance squared to the mouse click
            const float dx = carWorldX - worldX;
            const float dy = carWorldY - worldY;
            const float distSq = (dx * dx) + (dy * dy);

            // Find the absolute closest car
            if (distSq < closestDistSq) {
                closestDistSq = distSq;
                closestCarId = static_cast<std::int32_t>(i);
            }
        }

        // If we clicked reasonably close to a car (e.g., within 20 meters), select it!
        constexpr float thresholdDegrees = 20.f / 111300.f;
        if (closestCarId != -1 && closestDistSq < thresholdDegrees * thresholdDegrees) {
            selectedCarId = closestCarId;
            std::println("Selected Car ID: {}", selectedCarId);
        }
    }

    void applyMarqueeSelection() {
        if (cpuNodes.empty() || allExitNodes.empty()) return;

        float startWorldX = 0.f;
        float startWorldY = 0.f;
        float currentWorldX = 0.f;
        float currentWorldY = 0.f;
        screenToWorld(startMouseX, startMouseY, startWorldX, startWorldY);
        screenToWorld(currentMouseX, currentMouseY, currentWorldX, currentWorldY);

        const double dragDistX = std::abs(startMouseX - currentMouseX);
        const double dragDistY = std::abs(startMouseY - currentMouseY);

        if (dragDistX < 4. && dragDistY < 4.) {
            std::int32_t closestExitIdx = -1;
            float closestDistSq = std::numeric_limits<float>::max();

            for (std::size_t i = 0; i < allExitNodes.size(); ++i) {
                const float ex = cpuNodes[allExitNodes[i]].x;
                const float ey = cpuNodes[allExitNodes[i]].y;
                const float dx = ex - startWorldX;
                const float dy = ey - startWorldY;
                const float distSq = (dx * dx) + (dy * dy);
                if (distSq < closestDistSq) {
                    closestDistSq = distSq;
                    closestExitIdx = static_cast<std::int32_t>(i);
                }
            }

            constexpr float thresholdDegrees = 50.f / 111300.f;
            if (closestExitIdx != -1 && closestDistSq < thresholdDegrees * thresholdDegrees) {
                isExitOpen[closestExitIdx] = !isExitOpen[closestExitIdx];
                triggerDynamicReroute();
            }
            return;
        }

        const float minX = std::min(startWorldX, currentWorldX);
        const float maxX = std::max(startWorldX, currentWorldX);
        const float minY = std::min(startWorldY, currentWorldY);
        const float maxY = std::max(startWorldY, currentWorldY);

        bool changed = false;
        for (std::size_t i = 0; i < allExitNodes.size(); ++i) {
            const float x = cpuNodes[allExitNodes[i]].x;
            const float y = cpuNodes[allExitNodes[i]].y;

            if (x >= minX && x <= maxX && y >= minY && y <= maxY) {
                isExitOpen[i] = !isExitOpen[i];
                changed = true;
            }
        }

        if (changed) {
            triggerDynamicReroute();
        }
    }
};

auto main() -> std::int32_t {
    try {
        EvacuationEngine engine{};
        engine.run();
    } catch (const std::exception &e) {
        std::println(std::cerr, "Fatal GPU Error: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
