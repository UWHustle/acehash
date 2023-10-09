find_package(Threads REQUIRED)

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

add_executable(acehash_bench_function bench_function.cpp)
target_link_libraries(acehash_bench_function PRIVATE acehash_bench_common)

add_executable(acehash_bench_map bench_map.cpp)
target_link_libraries(acehash_bench_map PRIVATE acehash_bench_common absl::base absl::flat_hash_map)

add_executable(acehash_bench_query bench_query.cpp)
target_link_libraries(acehash_bench_query PRIVATE acehash_bench_common)
