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
    int32_t padding1{};
    int32_t padding2{};
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
    float min_x{};
    float max_x{};
    float min_y{};
    float max_y{};
    float aspect_ratio{};
    float padding1{};
    float padding2{};
    float padding3{};
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

        mainLoop();
    }

    ~EvacuationEngine() {
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

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(static_cast<int32_t>(WIDTH), static_cast<int32_t>(HEIGHT),
                                  "Vienna Evacuation Simulator", nullptr, nullptr);

        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback); // Listen for resizes

        glfwSetKeyCallback(
            window, [](GLFWwindow *win, const int key, const int scancode, const int action, const int mods) {
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
        for (auto &e: rawEdges) e.head_car_idx = -1;
        for (auto &c: rawCars) c.next_car_idx = -1;

        totalCars = static_cast<uint32_t>(rawCars.size());
        totalEdges = static_cast<uint32_t>(rawEdges.size());

        uploadVectorToGPU(nodes, nodeBuffer, nodeMemory);
        uploadVectorToGPU(rawEdges, edgeBuffer, edgeMemory);
        uploadVectorToGPU(rawCars, carBuffer, carMemory);

        mapBounds = {100000000.0f, -100000000.0f, 100000000.0f, -100000000.0f};
        for (const auto &n: nodes) {
            if (n.x < mapBounds.min_x) mapBounds.min_x = n.x;
            if (n.x > mapBounds.max_x) mapBounds.max_x = n.x;
            if (n.y < mapBounds.min_y) mapBounds.min_y = n.y;
            if (n.y > mapBounds.max_y) mapBounds.max_y = n.y;
        }
        // Initialize Aspect Ratio
        mapBounds.aspect_ratio = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);
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
        const vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

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

            // --- NEW: INJECT DYNAMIC VIEWPORT & SCISSOR ---
            const vk::Viewport dynamicViewport{
                .x = 0.0f, .y = 0.0f, .width = static_cast<float>(swapchainExtent.width),
                .height = static_cast<float>(swapchainExtent.height), .minDepth = 0.0f, .maxDepth = 1.0f
            };
            cmd.setViewport(0, dynamicViewport);
            const vk::Rect2D dynamicScissor{.offset = {0, 0}, .extent = swapchainExtent};
            cmd.setScissor(0, dynamicScissor);

            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **graphicsPipelineLayout, 0,
                                   {*computeDescriptorSets[0]}, nullptr);
            cmd.pushConstants<GraphicsPushData>(**graphicsPipelineLayout, vk::ShaderStageFlagBits::eAllGraphics, 0,
                                                mapBounds);

            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, **streetPipeline);
            cmd.draw(2, totalEdges, 0, 0);

            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, **graphicsPipeline);
            cmd.draw(6, totalCars, 0, 0);

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

        // --- NEW: THE SHUTDOWN FIX ---
        // Force the CPU to wait until the GPU is completely finished drawing the last frame
        // BEFORE destroying the objects
        device->waitIdle();
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
