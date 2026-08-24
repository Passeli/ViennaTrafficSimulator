cmake_minimum_required(VERSION 3.28)
project(BA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

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

FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG docking
        SOURCE_SUBDIR ""
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp

        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

target_include_directories(imgui PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
        ${Vulkan_INCLUDE_DIRS}
)
target_link_libraries(imgui PUBLIC glfw)

add_executable(BA main.cpp cpu_engine.hpp)

target_link_libraries(BA PRIVATE
        Vulkan::Vulkan
        glfw
        imgui
        nlohmann_json::nlohmann_json
)

target_compile_definitions(BA PRIVATE
        VULKAN_HPP_NO_CONSTRUCTORS
)

set(SHADER_DIR "${CMAKE_SOURCE_DIR}/shaders")

set(GRAPHICS_SLANG "${SHADER_DIR}/graphics.slang")
set(CLEAR_EDGES_SLANG "${SHADER_DIR}/clear_edges.slang")
set(BUILD_GRID_SLANG "${SHADER_DIR}/build_grid.slang")
set(PHYSICS_SLANG "${SHADER_DIR}/physics.slang")
set(STREETS_SLANG "${SHADER_DIR}/streets.slang")
set(EXIT_NODES_SLANG "${SHADER_DIR}/exit_nodes.slang")

set(VERT_SPV "${CMAKE_CURRENT_BINARY_DIR}/graphics_vert.spv")
set(FRAG_SPV "${CMAKE_CURRENT_BINARY_DIR}/graphics_frag.spv")
set(CLEAR_EDGES_SPV "${CMAKE_CURRENT_BINARY_DIR}/clear_edges.spv")
set(BUILD_GRID_SPV "${CMAKE_CURRENT_BINARY_DIR}/build_grid.spv")
set(PHYSICS_SPV "${CMAKE_CURRENT_BINARY_DIR}/physics.spv")
set(STREETS_VERT_SPV "${CMAKE_CURRENT_BINARY_DIR}/streets_vert.spv")
set(STREETS_FRAG_SPV "${CMAKE_CURRENT_BINARY_DIR}/streets_frag.spv")
set(EXIT_NODES_VERT_SPV "${CMAKE_CURRENT_BINARY_DIR}/exit_nodes_vert.spv")
set(EXIT_NODES_FRAG_SPV "${CMAKE_CURRENT_BINARY_DIR}/exit_nodes_frag.spv")

set(SLANG_INCLUDE_PATH "-I${SHADER_DIR}")

add_custom_command(
        OUTPUT ${VERT_SPV}
        COMMAND slangc ${GRAPHICS_SLANG} ${SLANG_INCLUDE_PATH} -target spirv -entry vertexMain -stage vertex -o ${VERT_SPV}
        DEPENDS ${GRAPHICS_SLANG}
)
add_custom_command(
        OUTPUT ${FRAG_SPV}
        COMMAND slangc ${GRAPHICS_SLANG} ${SLANG_INCLUDE_PATH} -target spirv -entry fragmentMain -stage fragment -o ${FRAG_SPV}
        DEPENDS ${GRAPHICS_SLANG}
)
add_custom_command(
        OUTPUT ${CLEAR_EDGES_SPV}
        COMMAND slangc ${CLEAR_EDGES_SLANG} ${SLANG_INCLUDE_PATH} -target spirv -o ${CLEAR_EDGES_SPV}
        DEPENDS ${CLEAR_EDGES_SLANG}
)
add_custom_command(
        OUTPUT ${BUILD_GRID_SPV}
        COMMAND slangc ${BUILD_GRID_SLANG} ${SLANG_INCLUDE_PATH} -target spirv -o ${BUILD_GRID_SPV}
        DEPENDS ${BUILD_GRID_SLANG}
)
add_custom_command(
        OUTPUT ${PHYSICS_SPV}
        COMMAND slangc ${PHYSICS_SLANG} ${SLANG_INCLUDE_PATH} -target spirv -o ${PHYSICS_SPV}
        DEPENDS ${PHYSICS_SLANG}
)
add_custom_command(
        OUTPUT ${STREETS_VERT_SPV}
        COMMAND slangc ${STREETS_SLANG} ${SLANG_INCLUDE_PATH} -target spirv -entry vertexMain -stage vertex -o ${STREETS_VERT_SPV}
        DEPENDS ${STREETS_SLANG}
)
add_custom_command(
        OUTPUT ${STREETS_FRAG_SPV}
        COMMAND slangc ${STREETS_SLANG} ${SLANG_INCLUDE_PATH} -target spirv -entry fragmentMain -stage fragment -o ${STREETS_FRAG_SPV}
        DEPENDS ${STREETS_SLANG}
)
add_custom_command(
        OUTPUT ${EXIT_NODES_VERT_SPV}
        COMMAND slangc ${EXIT_NODES_SLANG} ${SLANG_INCLUDE_PATH} -target spirv -entry vertexMain -stage vertex -o ${EXIT_NODES_VERT_SPV}
        DEPENDS ${EXIT_NODES_SLANG}
)
add_custom_command(
        OUTPUT ${EXIT_NODES_FRAG_SPV}
        COMMAND slangc ${EXIT_NODES_SLANG} ${SLANG_INCLUDE_PATH} -target spirv -entry fragmentMain -stage fragment -o ${EXIT_NODES_FRAG_SPV}
        DEPENDS ${EXIT_NODES_SLANG}
)

add_custom_target(CompileAllShaders DEPENDS
        ${VERT_SPV}
        ${FRAG_SPV}
        ${CLEAR_EDGES_SPV}
        ${BUILD_GRID_SPV}
        ${PHYSICS_SPV}
        ${STREETS_VERT_SPV}
        ${STREETS_FRAG_SPV}
        ${EXIT_NODES_VERT_SPV}
        ${EXIT_NODES_FRAG_SPV}
)
add_dependencies(BA CompileAllShaders)

set(NODE_BIN_SRC "${CMAKE_SOURCE_DIR}/vulkan_nodes.bin")
set(EDGE_BIN_SRC "${CMAKE_SOURCE_DIR}/vulkan_edges.bin")
set(CAR_BIN_SRC "${CMAKE_SOURCE_DIR}/vulkan_cars.bin")

set(CONFIG_SRC "${CMAKE_SOURCE_DIR}/config.json")

set(NODE_BIN_DST "${CMAKE_CURRENT_BINARY_DIR}/vulkan_nodes.bin")
set(EDGE_BIN_DST "${CMAKE_CURRENT_BINARY_DIR}/vulkan_edges.bin")
set(CAR_BIN_DST "${CMAKE_CURRENT_BINARY_DIR}/vulkan_cars.bin")
set(CONFIG_DST "${CMAKE_CURRENT_BINARY_DIR}/config.json")

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
add_custom_command(
        OUTPUT ${CONFIG_DST}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CONFIG_SRC} ${CONFIG_DST}
        DEPENDS ${CONFIG_SRC}
)

add_custom_target(CopyBinaryMaps DEPENDS
        ${NODE_BIN_DST}
        ${EDGE_BIN_DST}
        ${CAR_BIN_DST}
        ${CONFIG_DST}
)
add_dependencies(BA CopyBinaryMaps)