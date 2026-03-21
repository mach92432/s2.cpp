cmake_minimum_required(VERSION 3.14)
project(s2cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

option(S2_VULKAN  "Build with Vulkan backend"  OFF)
option(S2_CUDA    "Build with CUDA backend"    OFF)
option(S2_METAL   "Build with Metal backend"   OFF)

# ---------------------------------------------------------------------------
# GGML
# ---------------------------------------------------------------------------

set(GGML_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

if(S2_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(S2_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(S2_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

set(GGML_VULKAN ON CACHE BOOL "" FORCE)

add_subdirectory(ggml)

# ---------------------------------------------------------------------------
# Crow (pour le serveur HTTP)
# ---------------------------------------------------------------------------

find_package(Crow REQUIRED)
include_directories(${CROW_INCLUDE_DIRS})

# ---------------------------------------------------------------------------
# cxxopts (pour la gestion des arguments CLI)
# ---------------------------------------------------------------------------

# Cherche d'abord une installation système
find_package(cxxopts QUIET)

if(NOT cxxopts_FOUND)
    # Si non trouvé, utilise FetchContent pour le télécharger automatiquement
    include(FetchContent)
    FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG        v3.2.1
    )
    FetchContent_MakeAvailable(cxxopts)
endif()

# ---------------------------------------------------------------------------
# s2 executable
# ---------------------------------------------------------------------------

set(S2_SOURCES
    src/s2_audio.cpp
    src/s2_tokenizer.cpp
    src/s2_sampler.cpp
    src/s2_model.cpp
    src/s2_codec.cpp
    src/s2_prompt.cpp
    src/s2_generate.cpp
    src/s2_pipeline.cpp
    src/main.cpp
)

add_executable(s2 ${S2_SOURCES})

target_include_directories(s2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src
)

target_link_libraries(s2 PRIVATE
    ggml
    ${CROW_LIBRARIES}
    cxxopts::cxxopts
)

if(S2_VULKAN)
    target_compile_definitions(s2 PRIVATE GGML_USE_VULKAN)
endif()

# Platform-specific
if(UNIX AND NOT APPLE)
    target_link_libraries(s2 PRIVATE pthread m)
endif()

# Install
install(TARGETS s2 RUNTIME DESTINATION bin)
