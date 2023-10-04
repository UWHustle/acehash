#include "common.hpp"

std::mt19937 generator(0); // NOLINT(cert-msc51-cpp)
int num_trials = 5;

ResultsFile out("bench_map.csv",
                "Number of keys",
                "Algorithm",
                "Trial",
                "Construction time (s)",
                "Serial evaluation time (s)",
                "Parallel evaluation time (s)");

template <typename Map, typename Key, typename Value, typename... Args>
void bench_map(const std::vector<Key> &keys,
               const std::vector<Value> &values,
               Args... args) {
  for (int trial = 0; trial < num_trials; ++trial) {
    std::vector<Value> retrieved_values(keys.size());

    double t1, t2, t3;
    Map map;

    // 1. Construct the map.
    t1 = time(
        [&] { map = Map(keys.size(), keys.data(), values.data(), args...); });

    // 2. Evaluate the map (serial).
    t2 = time([&] { map(keys.size(), keys.data(), retrieved_values.data()); });

    validate(values, retrieved_values);

#ifdef ACEHASH_ENABLE_TBB
    // 3. Evaluate the map (parallel).
    t3 = time([&] {
      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, keys.size()),
          [&](const tbb::blocked_range<size_t> &r) {
            map(r.size(), &keys[r.begin()], &retrieved_values[r.begin()]);
          });
    });

    validate(values, retrieved_values);
#else
    t3 = 0.0;
#endif

    out.write(keys.size(), map.description(), trial, t1, t2, t3);
  }
}

template <typename Key, typename Value> void bench_map_integer(size_t n) {
  std::vector<Key> keys = generate_integer_keys<Key>(n, generator);
  std::vector<Value> values = generate_integer_values<Value>(n, generator);

  bench_map<StlMap<Key, Value>>(keys, values);
  bench_map<AbseilMap<Key, Value>>(keys, values);
  bench_map<AceHashMapV4<Key, Value>>(keys, values, 3.0, 0.9);

}

int main() {
  bench_map_integer<uint64_t, uint8_t>(10'000'000);

  return 0;
}
