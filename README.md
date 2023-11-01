# AceHash

AceHash is a performance-focused perfect hashing scheme. A perfect hash function (PHF) is a hash function that maps a set of keys to a range of positions with no collisions. When used in a hash table, a PHF reduces the overhead of collision resolution and shrinks the memory footprint, resulting in improved cache efficiency. AceHash is designed to fully exploit these benefits by producing PHFs that are fast to construct and evaluate, while maintaining size competitive with other techniques.

## Usage

AceHash is provided as a header-only library written in C++20. To use it in your project, simply include the file `acehash.hpp` found in the subdirectory `include/acehash`. You can also use CMake utilities (_e.g.,_ FetchContent) to download and include the header during configuration.

## Benchmarks

The benchmark suite uses [vcpkg](https://vcpkg.io) to manage dependencies. Some benchmarks also depend on [DuckDB](https://duckdb.org). To run the benchmarks, download vcpkg and DuckDB to a local directory using the instructions provided. Then, add the following to the appropriate triplet in `$VCPKG_PATH/triplets`, where `$VCPKG_PATH` is the path to vcpkg.
```
set(VCPKG_C_FLAGS "-march=native")
set(VCPKG_CXX_FLAGS "-march=native")
```
Then, create and navigate into a build directory. From the build directory, run the following, replacing `$DUCKDB_PATH` and `$VCPKG_PATH` with the appropriate paths.
```
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DACEHASH_BUILD_BENCHMARKS=ON \
  -DACEHASH_DUCKDB_INCLUDE_DIR=$DUCKDB_PATH \
  -DACEHASH_DUCKDB_LIBRARY=$DUCKDB_PATH/libduckdb.so \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_PATH/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_CXX_FLAGS="-march=native"
```

Finally, run the desired benchmark executables in the `bench` subdirectory. Each executable writes its results to a CSV file. You can then use the notebook `analysis.ipynb` in the subdirectory `bench/results` to produce the plots.
