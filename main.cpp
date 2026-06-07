#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>
#include <print>
#include <cstdint>
#include <memory>
#include <iostream>
#include <algorithm>

// --- 1. THE DATA STRUCTS (Aligned to 16 bytes) ---

struct GPU_Node {
    float x{};
    float y{};
    int32_t lock{-1};
    int32_t padding{};
};

struct GPU_Edge {
    int32_t start_node_idx{};
    int32_t end_node_idx{};
    int32_t next_edge_idx{};
    float length{};
    float max_speed{};
    int32_t head_car_idx{-1};
    int32_t garage_lock{-1};
    int32_t padding1{};
};

struct GPU_Car {
    int32_t current_edge_idx{};
    float position{};
    float speed{};
    int32_t state{};
    int32_t next_car_idx{-1};
    int32_t padding1{};
    int32_t padding2{};
    int32_t padding3{};
};

struct PushData {
    float dt{};
    uint32_t num_cars{};
    uint32_t num_edges{};
    uint32_t padding{};
};

// ADDED: aspect_ratio to prevent cars from stretching when the window resizes!
struct GraphicsPushData {
    float camera_x{};
    float camera_y{};
    float zoom_level{1.0f};
    float aspect_ratio{};
    float extent_width{}; // Bounding box size of the city in meters
    float extent_height{};
    float padding1{};
    float padding2{};
};

template<typename T>
auto loadBinaryData(const std::string &filepath) -> std::vector<T> {
    std::ifstream file{filepath, std::ios::binary | std::ios::ate};

    if (!file.is_open()) {
        throw std::runtime_error{"Failed to open: " + filepath};
    }

    const std::streamsize fileSize{file.tellg()};
    file.seekg(0, std::ios::beg);

    const size_t elementsToRead{static_cast<size_t>(fileSize) / sizeof(T)};
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
            ImGui_ImplVulkan_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
        }
        if (window != nullptr) {
            glfwDestroyWindow(window);
            glfwTerminate();
        }
    }

private:
    GLFWwindow *window{nullptr};

    std::unique_ptr<vk::raii::Context> context;
    std::unique_ptr<vk::raii::Instance> instance;
    std::unique_ptr<vk::raii::PhysicalDevice> physicalDevice;
    std::unique_ptr<vk::raii::Device> device;

    uint32_t queueFamilyIndex{0};
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

    uint32_t totalCars{0};
    uint32_t totalEdges{0};
    const uint32_t WIDTH{1920};
    const uint32_t HEIGHT{1080};

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
    GraphicsPushData mapBounds;

    // --- SIMULATION CONTROLS ---
    bool isPaused{false};
    int32_t simSpeed{1};
    bool framebufferResized{false};

    // --- CAMERA INTERACTION STATE ---
    bool isDragging{false};
    double lastMouseX{0.0};
    double lastMouseY{0.0};

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
    int selectedCarId = 0;

    int statGarage = 0;
    int statRoad = 0;
    int statEvacuated = 0;
    float statAvgSpeed = 0.0f;

    // --- RECREATION LOGIC ---
    void recreateSwapchain() {
        int width = 0, height = 0;
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

    static void framebufferResizeCallback(GLFWwindow *window, int width, int height) {
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
            .maxSets = 1000,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
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
        init_info.ImageCount = static_cast<uint32_t>(swapchainImages.size());
        init_info.Allocator = nullptr;
        init_info.CheckVkResultFn = nullptr;

        // --- THE FIX: Use the new PipelineInfoMain nested struct ---
        init_info.PipelineInfoMain.RenderPass = **renderPass;
        init_info.PipelineInfoMain.Subpass = 0;
        init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

        ImGui_ImplVulkan_Init(&init_info);

        // --- THE FIX: Manual font uploading is no longer required! ---
        // ImGui now automatically uploads fonts to the GPU on the first ImGui_ImplVulkan_NewFrame() call.
    }

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
        window = glfwCreateWindow(static_cast<int32_t>(WIDTH), static_cast<int32_t>(HEIGHT),
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
            engine->mapBounds.zoom_level = std::clamp(engine->mapBounds.zoom_level, 0.5f, 500.0f);
        });

        // B. MOUSE BUTTON CALLBACK (START / STOP DRAG & PICKING)
        glfwSetMouseButtonCallback(window, [](GLFWwindow *win, const int button, const int action, int mods) {
            if (ImGui::GetIO().WantCaptureMouse) return;

            auto *engine = static_cast<EvacuationEngine *>(glfwGetWindowUserPointer(win));

            // Left Click = Pan the Map
            if (button == GLFW_MOUSE_BUTTON_LEFT) {
                if (action == GLFW_PRESS) {
                    engine->isDragging = true;
                    glfwGetCursorPos(win, &engine->lastMouseX, &engine->lastMouseY);
                } else if (action == GLFW_RELEASE) {
                    engine->isDragging = false;
                }
            }

            // Right Click = Inspect Car
            if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
                double mouseX, mouseY;
                glfwGetCursorPos(win, &mouseX, &mouseY);

                float worldX, worldY;
                engine->screenToWorld(mouseX, mouseY, worldX, worldY);
                engine->selectClosestCar(worldX, worldY);
            }
        });

        // C. CURSOR POSITION CALLBACK (PANNING THE MAP)
        glfwSetCursorPosCallback(window, [](GLFWwindow *win, const double xpos, const double ypos) {
            // (We don't need the ImGui check here because isDragging won't be true if the press was blocked above!)
            auto *engine = static_cast<EvacuationEngine *>(glfwGetWindowUserPointer(win));
            if (engine->isDragging) {
                const double deltaX = xpos - engine->lastMouseX;
                const double deltaY = ypos - engine->lastMouseY;

                const float screenFactorX = (engine->mapBounds.extent_width / static_cast<float>(engine->WIDTH));
                const float screenFactorY = (engine->mapBounds.extent_height / static_cast<float>(engine->HEIGHT));

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
            window, [](GLFWwindow *win, const int key, const int scancode, const int action, const int mods) {
                // --- THE FIX: Ignore keyboard shortcuts if typing inside an ImGui text box ---
                if (ImGui::GetIO().WantCaptureKeyboard) return;

                auto *engine = static_cast<EvacuationEngine *>(glfwGetWindowUserPointer(win));
                if (action == GLFW_PRESS) {
                    if (key == GLFW_KEY_SPACE) {
                        engine->isPaused = !engine->isPaused;
                        std::println("Simulation {}", engine->isPaused ? "PAUSED" : "RESUMED");
                    } else if (key == GLFW_KEY_UP || key == GLFW_KEY_RIGHT) {
                        engine->simSpeed = std::min(engine->simSpeed * 2, 64);
                        std::println("Speed: {}x", engine->simSpeed);
                    } else if (key == GLFW_KEY_DOWN || key == GLFW_KEY_LEFT) {
                        engine->simSpeed = std::max(engine->simSpeed / 2, 1);
                        std::println("Speed: {}x", engine->simSpeed);
                    }
                }
            });
    }

    void initVulkan() {
        std::println("Initializing Vulkan RAII Context...");
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
        context = std::make_unique<vk::raii::Context>();

        constexpr vk::ApplicationInfo appInfo{
            .pApplicationName = "Vienna Evacuation", .applicationVersion = 1,
            .pEngineName = "No Engine", .engineVersion = 1, .apiVersion = VK_API_VERSION_1_3
        };

        uint32_t glfwExtensionCount = 0;
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
        for (uint32_t i{0}; i < queueFamilies.size(); ++i) {
            if ((queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
                (queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute)) {
                queueFamilyIndex = i;
                break;
            }
        }

        constexpr float queuePriority{1.0f};
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
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
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

    [[nodiscard]] auto findMemoryType(const uint32_t typeFilter,
                                      const vk::MemoryPropertyFlags properties) const -> uint32_t {
        const vk::PhysicalDeviceMemoryProperties memProperties{physicalDevice->getMemoryProperties()};
        for (uint32_t i{0}; i < memProperties.memoryTypeCount; ++i) {
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
    void uploadVectorToGPU(const std::vector<T> &data, std::unique_ptr<vk::raii::Buffer> &outBuffer,
                           std::unique_ptr<vk::raii::DeviceMemory> &outMemory) {
        const vk::DeviceSize bufferSize{sizeof(T) * data.size()};

        auto [stagingBuffer, stagingMemory] = createBuffer(
            bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        void *mappedData{stagingMemory->mapMemory(0, bufferSize)};
        std::memcpy(mappedData, data.data(), static_cast<size_t>(bufferSize));
        stagingMemory->unmapMemory();

        auto [deviceLocalBuffer, deviceLocalMemory] = createBuffer(
            bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        outBuffer = std::move(deviceLocalBuffer);
        outMemory = std::move(deviceLocalMemory);
        copyBuffer(*stagingBuffer, *outBuffer, bufferSize);
    }

    void loadMapDataAndCreateBuffers() {
        std::println("Loading binary data into RAM...");

        auto nodes{loadBinaryData<GPU_Node>("vulkan_nodes.bin")};
        auto rawEdges{loadBinaryData<GPU_Edge>("vulkan_edges.bin")};
        auto rawCars{loadBinaryData<GPU_Car>("vulkan_cars.bin")};

        for (auto &n: nodes) n.lock = -1;
        for (auto &e: rawEdges) {
            e.head_car_idx = -1;
            e.garage_lock = -1;
        }
        for (auto &c: rawCars) c.next_car_idx = -1;

        totalCars = static_cast<uint32_t>(rawCars.size());
        totalEdges = static_cast<uint32_t>(rawEdges.size());

        uploadVectorToGPU(nodes, nodeBuffer, nodeMemory);
        uploadVectorToGPU(rawEdges, edgeBuffer, edgeMemory);
        uploadVectorToGPU(rawCars, carBuffer, carMemory);

        // --- 1. Find the boundaries using local variables ---
        float temp_min_x = 100000000.0f;
        float temp_max_x = -100000000.0f;
        float temp_min_y = 100000000.0f;
        float temp_max_y = -100000000.0f;

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
        mapBounds.zoom_level = 1.0f;
        mapBounds.extent_width = width_meters;
        mapBounds.extent_height = height_meters;
        mapBounds.aspect_ratio = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);

        // Initialize CPU vectors to hold the data
        cpuCars.resize(totalCars);
        cpuEdges.resize(totalEdges);
        cpuNodes.resize(nodes.size());

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
    }

    void createComputePipeline() {
        const auto clearEdgesCode{loadBinaryData<uint32_t>("clear_edges.spv")};
        const vk::ShaderModuleCreateInfo clearEdgesInfo{
            .codeSize = clearEdgesCode.size() * 4, .pCode = clearEdgesCode.data()
        };
        const vk::raii::ShaderModule clearEdgesShader{*device, clearEdgesInfo};

        const auto buildGridCode{loadBinaryData<uint32_t>("build_grid.spv")};
        const vk::ShaderModuleCreateInfo buildGridInfo{
            .codeSize = buildGridCode.size() * 4, .pCode = buildGridCode.data()
        };
        const vk::raii::ShaderModule buildGridShader{*device, buildGridInfo};

        const auto physicsCode{loadBinaryData<uint32_t>("physics.spv")};
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
            .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()
        };
        computeDescriptorSetLayout = std::make_unique<vk::raii::DescriptorSetLayout>(*device, layoutInfo);

        constexpr vk::PushConstantRange pushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(PushData)
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
                throw std::runtime_error
                        {"Failed to create window surface!"};
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
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        swapchainExtent = vk::Extent2D{
            std::clamp(static_cast<uint32_t>(width), capabilities.minImageExtent.width,
                       capabilities.maxImageExtent.width),
            std::clamp(static_cast<uint32_t>(height), capabilities.minImageExtent.height,
                       capabilities.maxImageExtent.height)
        };

        uint32_t imageCount = capabilities.minImageCount + 1;
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
        const auto vertCode{loadBinaryData<uint32_t>("graphics_vert.spv")};
        const auto fragCode{loadBinaryData<uint32_t>("graphics_frag.spv")};

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
            .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eNone, .lineWidth = 1.0f
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
            .stageFlags = vk::ShaderStageFlagBits::eAllGraphics, .offset = 0, .size = sizeof(GraphicsPushData)
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
        const auto streetVertCode{loadBinaryData<uint32_t>("streets_vert.spv")};
        const auto streetFragCode{loadBinaryData<uint32_t>("streets_frag.spv")};
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
            .color = vk::ClearColorValue{std::array<float, 4>{0.05f, 0.05f, 0.1f, 1.0f}}
        };

        while (glfwWindowShouldClose(window) == 0) {
            glfwPollEvents();

            // --- THE FIX: Recreate the swapchain when the window resizes ---
            if (framebufferResized) {
                framebufferResized = false;
                recreateSwapchain();
            }

            uint32_t imageIndex;
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
            ImGui::Text("Performance: %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                        ImGui::GetIO().Framerate);
            ImGui::Separator();

            if (ImGui::Button(isPaused ? "Resume Simulation" : "Pause Simulation")) {
                isPaused = !isPaused;
            }
            ImGui::SliderInt("Simulation Speed", &simSpeed, 1, 64, "%d x");

            // ==========================================
            // --- NEW: THE DATA INSPECTOR ---
            // ==========================================
            ImGui::Separator();
            ImGui::Text("GPU Memory Inspector");

            if (ImGui::Button("Download GPU Snapshot")) {
                takeSnapshot(); // Pulls the data from VRAM!
            }

            ImGui::Separator();
            ImGui::Text("--- EVACUATION METRICS ---");
            ImGui::Text("Waiting in Garage: %d", statGarage);
            ImGui::Text("Active on Road: %d", statRoad);
            ImGui::Text("Safely Evacuated: %d", statEvacuated);
            ImGui::Text("City Average Speed: %.1f km/h", statAvgSpeed * 3.6f);

            // Progress Bar!
            float progress = static_cast<float>(statEvacuated) / static_cast<float>(totalCars);
            ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), "Evacuation Progress");

            if (!cpuCars.empty()) {
                ImGui::Spacing();
                // Allow the user to type in a Car ID to inspect
                ImGui::InputInt("Inspect Car ID", &selectedCarId);
                selectedCarId = std::clamp(selectedCarId, 0, static_cast<int>(totalCars) - 1);

                GPU_Car &car = cpuCars[selectedCarId];

                ImGui::BeginChild("CarData", ImVec2(0, 150), true);
                ImGui::Text("--- CAR %d ---", selectedCarId);

                const char *stateStr = "Unknown";
                if (car.state == 0) stateStr = "Driving";
                if (car.state == 1) stateStr = "Queuing at Intersection";
                if (car.state == 2) stateStr = "Evacuated!";
                if (car.state == 3) stateStr = "In Garage";

                ImGui::Text("State: %s (%d)", stateStr, car.state);
                ImGui::Text("Speed: %.2f m/s (%.1f km/h)", car.speed, car.speed * 3.6f);
                ImGui::Text("Position: %.2f meters", car.position);
                ImGui::Text("Current Edge ID: %d", car.current_edge_idx);
                ImGui::Text("Next Car in Linked List: %d", car.next_car_idx);
                ImGui::EndChild();

                // If the car is on an edge, show the edge data too!
                if (car.current_edge_idx != -1) {
                    GPU_Edge &edge = cpuEdges[car.current_edge_idx];
                    ImGui::BeginChild("EdgeData", ImVec2(0, 120), true);
                    ImGui::Text("--- EDGE %d ---", car.current_edge_idx);
                    ImGui::Text("Length: %.2f meters", edge.length);
                    ImGui::Text("Max Speed: %.1f km/h", edge.max_speed * 3.6f);
                    ImGui::Text("Head Car ID: %d", edge.head_car_idx);
                    ImGui::Text("Target Node ID: %d", edge.end_node_idx);
                    ImGui::EndChild();

                    // Show the node data (Intersection Lock)
                    GPU_Node &node = cpuNodes[edge.end_node_idx];
                    ImGui::Text("Target Node Lock: %d", node.lock);
                }
            }
            ImGui::End();

            // --- 3. FINALIZE UI DATA ---
            ImGui::Render();

            // --- VULKAN COMMAND RECORDING ---
            constexpr vk::CommandBufferBeginInfo cmdBeginInfo{
                .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
            };
            cmd.begin(cmdBeginInfo);

            if (!isPaused) {
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, **computePipelineLayout, 0,
                                       {*computeDescriptorSets[0]}, nullptr);
                const PushData pushData{.dt = 0.1f, .num_cars = totalCars, .num_edges = totalEdges};
                cmd.pushConstants<PushData>(**computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pushData);

                for (int32_t step = 0; step < simSpeed; ++step) {
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

                    vk::PipelineStageFlags dstStage = (step == simSpeed - 1)
                                                          ? vk::PipelineStageFlagBits::eVertexShader
                                                          : vk::PipelineStageFlagBits::eComputeShader;
                    constexpr vk::MemoryBarrier stepBarrier{
                        .srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
                        .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite
                    };
                    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, dstStage, {}, stepBarrier, nullptr,
                                        nullptr);
                }
            }

            const vk::RenderPassBeginInfo renderPassInfo{
                .renderPass = **renderPass, .framebuffer = *framebuffers[imageIndex],
                .renderArea = {.offset = {0, 0}, .extent = swapchainExtent}, .clearValueCount = 1,
                .pClearValues = &clearColor
            };
            cmd.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

            // --- THE FIX: Bind the graphics pipeline and push camera constants FIRST ---
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, **streetPipeline);

            // Set Dynamic Viewport & Scissor
            const vk::Viewport dynamicViewport{
                .x = 0.0f, .y = 0.0f, .width = static_cast<float>(swapchainExtent.width),
                .height = static_cast<float>(swapchainExtent.height), .minDepth = 0.0f, .maxDepth = 1.0f
            };
            cmd.setViewport(0, dynamicViewport);
            const vk::Rect2D dynamicScissor{.offset = {0, 0}, .extent = swapchainExtent};
            cmd.setScissor(0, dynamicScissor);

            // Push the Camera Data (This only affects your pipelines, not ImGui!)
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **graphicsPipelineLayout, 0,
                                   {*computeDescriptorSets[0]}, nullptr);
            cmd.pushConstants<GraphicsPushData>(**graphicsPipelineLayout, vk::ShaderStageFlagBits::eAllGraphics, 0,
                                                mapBounds);

            // Draw the Map
            cmd.draw(2, totalEdges, 0, 0);

            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, **graphicsPipeline);
            cmd.draw(6, totalCars, 0, 0);

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

        // --- THE SHUTDOWN FIX ---
        // Force the CPU to wait until the GPU is completely finished drawing the last frame
        // BEFORE destroying the objects
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

        std::println("Snapshot downloaded from VRAM to RAM!");

        statGarage = 0;
        statRoad = 0;
        statEvacuated = 0;
        double totalSpeed = 0.0;

        for (const auto &car: cpuCars) {
            if (car.state == 0 || car.state == 1) {
                statRoad++;
                totalSpeed += car.speed;
            } else if (car.state == 2) {
                statEvacuated++;
            } else if (car.state == 3) {
                statGarage++;
            }
        }
        statAvgSpeed = statRoad > 0 ? static_cast<float>(totalSpeed / statRoad) : 0.0f;
        std::println("Snapshot downloaded from VRAM to RAM!");
    }

    // Converts raw GLFW window pixels into Vienna World Coordinates (Meters)
    void screenToWorld(const double screenX, const double screenY, float &outWorldX, float &outWorldY) const {
        // 1. Convert Screen Pixels to Normalized Device Coordinates (NDC: -1.0 to 1.0)
        float ndcX = (static_cast<float>(screenX) / static_cast<float>(WIDTH)) * 2.0f - 1.0f;
        // Vulkan's Y axis points down, but our world map points up, so we invert Y
        const float ndcY = -((static_cast<float>(screenY) / static_cast<float>(HEIGHT)) * 2.0f - 1.0f);

        // 2. Reverse the Aspect Ratio correction
        ndcX *= mapBounds.aspect_ratio;

        // 3. Reverse the Zoom and Extent scaling
        const float localX = (ndcX * (mapBounds.extent_width * 0.5f)) / mapBounds.zoom_level;
        const float localY = (ndcY * (mapBounds.extent_height * 0.5f)) / mapBounds.zoom_level;

        // 4. Reverse the Camera Pan
        outWorldX = localX + mapBounds.camera_x;
        outWorldY = localY + mapBounds.camera_y;
    }

    void selectClosestCar(const float worldX, const float worldY) {
        if (cpuCars.empty()) return; // Must take a snapshot first!

        float closestDistSq = std::numeric_limits<float>::max();
        int bestCarId = -1;

        for (size_t i = 0; i < cpuCars.size(); ++i) {
            const GPU_Car &car = cpuCars[i];

            // Only select cars that are actually on the road
            if (car.state >= 2 || car.current_edge_idx == -1) continue;

            const GPU_Edge &edge = cpuEdges[car.current_edge_idx];
            const GPU_Node &startNode = cpuNodes[edge.start_node_idx];
            const GPU_Node &endNode = cpuNodes[edge.end_node_idx];

            // Interpolate car's world position along the edge
            const float t = (edge.length > 0.0f) ? (car.position / edge.length) : 0.0f;
            const float carWorldX = startNode.x + t * (endNode.x - startNode.x);
            const float carWorldY = startNode.y + t * (endNode.y - startNode.y);

            // Calculate distance squared to the mouse click
            const float dx = carWorldX - worldX;
            const float dy = carWorldY - worldY;
            const float distSq = (dx * dx) + (dy * dy);

            if (distSq < closestDistSq) {
                closestDistSq = distSq;
                bestCarId = static_cast<int>(i);
            }
        }

        // If we clicked reasonably close to a car (e.g., within 20 meters), select it!
        if (bestCarId != -1 && closestDistSq < 400.0f) {
            selectedCarId = bestCarId;
            std::println("Selected Car ID: {}", selectedCarId);
        }
    }
};

auto main() -> int32_t {
    try {
        EvacuationEngine engine{};
        engine.run();
    } catch (const std::exception &e) {
        std::println(stderr, "Fatal GPU Error: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
