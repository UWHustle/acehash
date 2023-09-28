cmake_minimum_required(VERSION 3.4)

option(ACEHASH_ENABLE_TBB "Enable TBB" OFF)
option(ACEHASH_BUILD_TESTS "Build AceHash tests" OFF)

if (ACEHASH_ENABLE_TBB)
    list(APPEND VCPKG_MANIFEST_FEATURES "tbb")
endif ()

project(acehash)

set(CMAKE_CXX_STANDARD 20)

add_library(acehash INTERFACE include/acehash/acehash.hpp)

if (ACEHASH_ENABLE_TBB)
    find_package(TBB CONFIG REQUIRED)
    target_compile_definitions(acehash INTERFACE ACEHASH_ENABLE_TBB)
    target_link_libraries(acehash INTERFACE TBB::tbb)
endif ()

if (ACEHASH_BUILD_TESTS)
    add_executable(acehash_test test/test.cpp)
    target_include_directories(acehash_test PRIVATE include)
    target_link_libraries(acehash_test PRIVATE acehash)
endif ()
