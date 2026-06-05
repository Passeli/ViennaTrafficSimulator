cmake_minimum_required(VERSION 3.28)
project(BA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Find Vulkan SDK
find_package(Vulkan REQUIRED)

# 2. Automatically download and build GLFW
include(FetchContent)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

# ==========================================
# --- AUTO-DOWNLOAD & BUILD DEAR IMGUI ---
# ==========================================
FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG docking
        SOURCE_SUBDIR "" # <--- THE FIX: Tells CMake to skip looking for a CMakeLists.txt
)
FetchContent_MakeAvailable(imgui)

# ImGui doesn't have a native CMakeLists.txt, so we build it manually using the downloaded source
add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        # Backends required for our specific Vulkan + GLFW engine
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

# Tell ImGui where to find Vulkan and GLFW headers
target_include_directories(imgui PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
        ${Vulkan_INCLUDE_DIRS}
)
target_link_libraries(imgui PUBLIC glfw)

# 3. Create the executable
add_executable(BA main.cpp)

# 4. Link the necessary libraries (Added ImGui!)
target_link_libraries(BA PRIVATE
        Vulkan::Vulkan
        glfw
        imgui
)

# 5. Enable dynamic dispatch for Vulkan-Hpp for maximum performance
target_compile_definitions(BA PRIVATE
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        VULKAN_HPP_NO_CONSTRUCTORS
)

# ==========================================
# --- AUTO-COMPILE ALL SLANG SHADERS ---
# ==========================================

set(GRAPHICS_SLANG "${CMAKE_SOURCE_DIR}/graphics.slang")
set(CLEAR_EDGES_SLANG "${CMAKE_SOURCE_DIR}/clear_edges.slang")
set(BUILD_GRID_SLANG "${CMAKE_SOURCE_DIR}/build_grid.slang")
set(PHYSICS_SLANG "${CMAKE_SOURCE_DIR}/physics.slang")
set(STREETS_SLANG "${CMAKE_SOURCE_DIR}/streets.slang")

set(VERT_SPV "${CMAKE_CURRENT_BINARY_DIR}/graphics_vert.spv")
set(FRAG_SPV "${CMAKE_CURRENT_BINARY_DIR}/graphics_frag.spv")
set(CLEAR_EDGES_SPV "${CMAKE_CURRENT_BINARY_DIR}/clear_edges.spv")
set(BUILD_GRID_SPV "${CMAKE_CURRENT_BINARY_DIR}/build_grid.spv")
set(PHYSICS_SPV "${CMAKE_CURRENT_BINARY_DIR}/physics.spv")
set(STREETS_VERT_SPV "${CMAKE_CURRENT_BINARY_DIR}/streets_vert.spv")
set(STREETS_FRAG_SPV "${CMAKE_CURRENT_BINARY_DIR}/streets_frag.spv")

add_custom_command(
        OUTPUT ${VERT_SPV}
        COMMAND slangc ${GRAPHICS_SLANG} -target spirv -entry vertexMain -stage vertex -o ${VERT_SPV}
        DEPENDS ${GRAPHICS_SLANG}
)
add_custom_command(
        OUTPUT ${FRAG_SPV}
        COMMAND slangc ${GRAPHICS_SLANG} -target spirv -entry fragmentMain -stage fragment -o ${FRAG_SPV}
        DEPENDS ${GRAPHICS_SLANG}
)
add_custom_command(
        OUTPUT ${CLEAR_EDGES_SPV}
        COMMAND slangc ${CLEAR_EDGES_SLANG} -target spirv -o ${CLEAR_EDGES_SPV}
        DEPENDS ${CLEAR_EDGES_SLANG}
)
add_custom_command(
        OUTPUT ${BUILD_GRID_SPV}
        COMMAND slangc ${BUILD_GRID_SLANG} -target spirv -o ${BUILD_GRID_SPV}
        DEPENDS ${BUILD_GRID_SLANG}
)
add_custom_command(
        OUTPUT ${PHYSICS_SPV}
        COMMAND slangc ${PHYSICS_SLANG} -target spirv -o ${PHYSICS_SPV}
        DEPENDS ${PHYSICS_SLANG}
)
add_custom_command(
        OUTPUT ${STREETS_VERT_SPV}
        COMMAND slangc ${STREETS_SLANG} -target spirv -entry vertexMain -stage vertex -o ${STREETS_VERT_SPV}
        DEPENDS ${STREETS_SLANG}
)
add_custom_command(
        OUTPUT ${STREETS_FRAG_SPV}
        COMMAND slangc ${STREETS_SLANG} -target spirv -entry fragmentMain -stage fragment -o ${STREETS_FRAG_SPV}
        DEPENDS ${STREETS_SLANG}
)

add_custom_target(CompileAllShaders DEPENDS
        ${VERT_SPV}
        ${FRAG_SPV}
        ${CLEAR_EDGES_SPV}
        ${BUILD_GRID_SPV}
        ${PHYSICS_SPV}
        ${STREETS_VERT_SPV}
        ${STREETS_FRAG_SPV}
)
add_dependencies(BA CompileAllShaders)


# ==========================================
# --- DATA ASSET PIPELINE (AUTO-COPY) ---
# ==========================================

set(NODE_BIN_SRC "${CMAKE_SOURCE_DIR}/vulkan_nodes.bin")
set(EDGE_BIN_SRC "${CMAKE_SOURCE_DIR}/vulkan_edges.bin")
set(CAR_BIN_SRC "${CMAKE_SOURCE_DIR}/vulkan_cars.bin")

set(NODE_BIN_DST "${CMAKE_CURRENT_BINARY_DIR}/vulkan_nodes.bin")
set(EDGE_BIN_DST "${CMAKE_CURRENT_BINARY_DIR}/vulkan_edges.bin")
set(CAR_BIN_DST "${CMAKE_CURRENT_BINARY_DIR}/vulkan_cars.bin")

add_custom_command(
        OUTPUT ${NODE_BIN_DST}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NODE_BIN_SRC} ${NODE_BIN_DST}
        DEPENDS ${NODE_BIN_SRC}
)
add_custom_command(
        OUTPUT ${EDGE_BIN_DST}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${EDGE_BIN_SRC} ${EDGE_BIN_DST}
        DEPENDS ${EDGE_BIN_SRC}
)
add_custom_command(
        OUTPUT ${CAR_BIN_DST}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CAR_BIN_SRC} ${CAR_BIN_DST}
        DEPENDS ${CAR_BIN_SRC}
)

add_custom_target(CopyBinaryMaps DEPENDS
        ${NODE_BIN_DST}
        ${EDGE_BIN_DST}
        ${CAR_BIN_DST}
)
add_dependencies(BA CopyBinaryMaps)