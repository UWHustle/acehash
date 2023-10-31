find_package(TBB CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(folly CONFIG REQUIRED)
find_package(Threads REQUIRED)

include(FetchContent)

FetchContent_Declare(
        bbhash
        GIT_REPOSITORY https://github.com/rizkg/BBHash.git
        GIT_TAG 1803c23
)
FetchContent_GetProperties(bbhash)
if (NOT bbhash_POPULATED)
    FetchContent_Populate(bbhash)
endif ()

FetchContent_Declare(
        pthash
        GIT_REPOSITORY https://github.com/jermp/pthash.git
        GIT_TAG 4098013
)
FetchContent_GetProperties(pthash)
if (NOT pthash_POPULATED)
    FetchContent_Populate(pthash)
endif ()

add_library(acehash_bench_common INTERFACE)
target_include_directories(acehash_bench_common INTERFACE bench)
target_include_directories(
        acehash_bench_common
        SYSTEM INTERFACE
        ${bbhash_SOURCE_DIR}
        ${pthash_SOURCE_DIR}/include
)
target_link_libraries(acehash_bench_common INTERFACE acehash TBB::tbb)

add_executable(acehash_bench_intro bench_intro.cpp)
target_link_libraries(acehash_bench_intro PRIVATE acehash_bench_common)

add_executable(acehash_bench_function bench_function.cpp)
target_link_libraries(acehash_bench_function PRIVATE acehash_bench_common)

add_executable(acehash_bench_join bench_join.cpp)
target_link_libraries(
        acehash_bench_join
        PRIVATE
        acehash_bench_common
        absl::flat_hash_map
        Folly::folly
)

add_executable(acehash_bench_aggregate bench_aggregate.cpp)
target_link_libraries(
        acehash_bench_aggregate
        PRIVATE
        acehash_bench_common
        absl::flat_hash_map
        Folly::folly
)

if (DEFINED ACEHASH_DUCKDB_INCLUDE_DIR AND DEFINED ACEHASH_DUCKDB_LIBRARY)
    message(STATUS "Using DuckDB include directory: ${ACEHASH_DUCKDB_INCLUDE_DIR}")
    message(STATUS "Using DuckDB library: ${ACEHASH_DUCKDB_LIBRARY}")
    target_compile_definitions(acehash_bench_join PRIVATE ACEHASH_ENABLE_DUCKDB)
    target_include_directories(acehash_bench_join PRIVATE ${ACEHASH_DUCKDB_INCLUDE_DIR})
    target_link_libraries(acehash_bench_join PRIVATE ${ACEHASH_DUCKDB_LIBRARY})
    target_compile_definitions(acehash_bench_aggregate PRIVATE ACEHASH_ENABLE_DUCKDB)
    target_include_directories(acehash_bench_aggregate PRIVATE ${ACEHASH_DUCKDB_INCLUDE_DIR})
    target_link_libraries(acehash_bench_aggregate PRIVATE ${ACEHASH_DUCKDB_LIBRARY})
endif ()
