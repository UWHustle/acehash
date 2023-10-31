cmake_minimum_required(VERSION 3.16)
cmake_policy(VERSION 3.16)

option(ACEHASH_ENABLE_TBB "Enable TBB" OFF)
option(ACEHASH_BUILD_TESTS "Build tests" OFF)
option(ACEHASH_BUILD_BENCHMARKS "Build benchmarks" OFF)

set(Boost_NO_WARN_NEW_VERSIONS ON)

if (ACEHASH_ENABLE_TBB)
    list(APPEND VCPKG_MANIFEST_FEATURES "tbb")
endif ()

if (ACEHASH_BUILD_BENCHMARKS)
    list(APPEND VCPKG_MANIFEST_FEATURES "benchmarks")
endif ()

if (DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Using CMake toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
endif ()

project(acehash)
set(CMAKE_CXX_STANDARD 20)

set(VCPKG_CMAKE_CONFIGURE_OPTIONS -DCMAKE_CXX_FLAGS="-march=native")

add_library(acehash INTERFACE)
target_include_directories(
        acehash
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if (ACEHASH_ENABLE_TBB)
    target_compile_definitions(acehash INTERFACE ACEHASH_ENABLE_TBB)
    find_package(TBB CONFIG REQUIRED)
    target_link_libraries(acehash INTERFACE TBB::tbb)
endif ()

if (ACEHASH_BUILD_TESTS)
    add_subdirectory(test)
endif ()

if (ACEHASH_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif ()
