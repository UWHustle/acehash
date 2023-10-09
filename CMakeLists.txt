cmake_minimum_required(VERSION 3.16)
cmake_policy(VERSION 3.16)

option(ACEHASH_ENABLE_TBB "Enable TBB" OFF)
option(ACEHASH_BUILD_TESTS "Build tests" OFF)
option(ACEHASH_BUILD_BENCHMARKS "Build benchmarks" OFF)

project(acehash)

set(CMAKE_CXX_STANDARD 20)

add_library(acehash INTERFACE)
target_include_directories(
        acehash
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if (ACEHASH_ENABLE_TBB)
    target_compile_definitions(acehash INTERFACE ACEHASH_ENABLE_TBB)

    include(FetchContent)

    FetchContent_Declare(
            oneTBB
            GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
            GIT_TAG v2021.10.0
    )
    FetchContent_MakeAvailable(oneTBB)

    target_link_libraries(acehash INTERFACE TBB::tbb)
endif ()

if (ACEHASH_BUILD_TESTS)
    add_subdirectory(test)
endif ()

if (ACEHASH_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif ()
