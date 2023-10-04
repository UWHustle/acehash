cmake_minimum_required(VERSION 3.4)

option(ACEHASH_ENABLE_TBB "Enable TBB" OFF)
option(ACEHASH_BUILD_TESTS "Build tests" OFF)
option(ACEHASH_BUILD_BENCHMARKS "Build benchmarks" OFF)

if (ACEHASH_ENABLE_TBB)
    list(APPEND VCPKG_MANIFEST_FEATURES "tbb")
endif ()

project(acehash)

set(CMAKE_CXX_STANDARD 20)

add_library(acehash INTERFACE)
target_include_directories(acehash INTERFACE include)

if (ACEHASH_ENABLE_TBB)
    find_package(TBB CONFIG REQUIRED)
    target_link_libraries(acehash INTERFACE TBB::tbb)
    target_compile_definitions(acehash INTERFACE ACEHASH_ENABLE_TBB)
endif ()

if (ACEHASH_BUILD_TESTS)
    add_executable(acehash_test test/test.cpp)
    target_link_libraries(acehash_test PRIVATE acehash)
endif ()

if (ACEHASH_BUILD_BENCHMARKS)
    include(FetchContent)

    FetchContent_Declare(
            abseil
            GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
            GIT_TAG 20230802.1
    )
    set(ABSL_PROPAGATE_CXX_STD ON)
    FetchContent_MakeAvailable(abseil)

    FetchContent_Declare(
            bbhash
            GIT_REPOSITORY https://github.com/rizkg/BBHash.git
            GIT_TAG 1803c23
    )
    FetchContent_MakeAvailable(bbhash)

    FetchContent_Declare(
            pthash
            GIT_REPOSITORY https://github.com/jermp/pthash.git
            GIT_TAG 4098013
    )
    FetchContent_MakeAvailable(pthash)

    find_package(Threads REQUIRED)

    add_library(acehash_bench_common INTERFACE)
    target_include_directories(acehash_bench_common INTERFACE bench)
    target_include_directories(
            acehash_bench_common
            SYSTEM INTERFACE
            ${bbhash_SOURCE_DIR}
            ${pthash_SOURCE_DIR}/include
    )
    target_link_libraries(
            acehash_bench_common
            INTERFACE
            acehash
            absl::base
            absl::flat_hash_map
            Threads::Threads
    )

    macro(acehash_add_benchmark name)
        add_executable(acehash_bench_${name} bench/bench_${name}.cpp)
        target_link_libraries(acehash_bench_${name} PRIVATE acehash_bench_common)
    endmacro()

    acehash_add_benchmark(function)
    acehash_add_benchmark(map)
endif ()
