cmake_minimum_required(VERSION 3.24)
project(sixi_asi VERSION 0.8.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SIXI_ENABLE_CUDA "Build native sixi-asi CUDA kernels" ON)
option(SIXI_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(SIXI_BUILD_TESTS "Build sixi-asi unit, differential, and integration test suite" ON)

# Determinism-safe compiler flags (Section 4: no fast-math flags that break IEEE-754 determinism)
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
)

if(SIXI_ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
endif()

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

set(SIXI_ASI_CORE_SOURCES
    src/common/common.cpp
    src/io/gguf.cpp
    src/io/manifest.cpp
    src/io/sidecar.cpp
    src/io/io_reader.cpp
    src/memory/memory_planner.cpp
    src/memory/offload.cpp
    src/core/numeric.cpp
    src/core/core_execution.cpp
    src/speculative/speculative.cpp
    src/server/server.cpp
)

if(SIXI_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 110)
        endif()
        list(APPEND SIXI_ASI_CORE_SOURCES src/cuda/sixi_kernels.cu)
        add_compile_definitions(SIXI_HAS_CUDA=1)
    endif()
endif()

add_library(sixi_asi STATIC ${SIXI_ASI_CORE_SOURCES})

target_include_directories(sixi_asi
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_link_libraries(sixi_asi
    PUBLIC
        Threads::Threads
        OpenSSL::Crypto
        OpenSSL::SSL
        ${CMAKE_DL_LIBS}
)

if(CMAKE_CUDA_COMPILER AND SIXI_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(sixi_asi PUBLIC CUDA::cudart)
endif()

# Primary sixi-asi CLI, Server & Benchmark Executables
add_executable(sixi-cli apps/sixi_cli.cpp)
target_link_libraries(sixi-cli PRIVATE sixi_asi)

add_executable(sixi-server apps/sixi_server_main.cpp)
target_link_libraries(sixi-server PRIVATE sixi_asi)

add_executable(sixi-bench apps/sixi_bench.cpp)
target_link_libraries(sixi-bench PRIVATE sixi_asi)

# Automated Test Suite (Section 11)
if(SIXI_BUILD_TESTS)
    enable_testing()

    set(SIXI_TEST_TARGETS
        test_io_and_overflow
        test_routing_and_numerics
        test_determinism_and_speculative
        test_memory_and_offload
        test_server_and_tools
    )

    if(CMAKE_CUDA_COMPILER AND SIXI_ENABLE_CUDA)
        list(APPEND SIXI_TEST_TARGETS test_cuda_kernels)
    endif()

    foreach(test_name IN LISTS SIXI_TEST_TARGETS)
        add_executable(${test_name} tests/${test_name}.cpp)
        target_link_libraries(${test_name} PRIVATE sixi_asi)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()
