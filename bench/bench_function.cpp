#include "common.hpp"

#include <vector>

std::mt19937 generator(0); // NOLINT(cert-msc51-cpp)
int num_trials = 5;

ResultsFile out("bench_function.csv",
                "Number of keys",
                "Algorithm",
                "Trial",
                "Number of bits",
                "Construction time (s)",
                "Serial evaluation time (s)",
                "Parallel evaluation time (s)");

template <typename Function, typename Key, typename... Args>
void bench_function(const std::vector<Key> &keys, Args... args) {
  std::vector<uint32_t> indices(keys.size());

  for (int trial = 0; trial < num_trials; ++trial) {
    double t1, t2, t3;
    Function function;

    // 1. Construct the function.
    t1 = time([&] { function = Function(keys.size(), keys.data(), args...); });

    // 2. Evaluate the function (serial).
    t2 = time([&] { function(keys.size(), keys.data(), indices.data()); });

    validate(indices);

#ifdef ACEHASH_ENABLE_TBB
    // 3. Evaluate the function (parallel).
    t3 = time([&] {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size()),
                        [&](const tbb::blocked_range<size_t> &r) {
                          function(
                              r.size(), &keys[r.begin()], &indices[r.begin()]);
                        });
    });

    validate(indices);
#else
    t3 = 0.0;
#endif

    out.write(keys.size(),
              function.description(),
              trial,
              function.num_bits(),
              t1,
              t2,
              t3);
  }
}

template <typename Key>
void bench_function_acehash(const std::vector<Key> &keys,
                            double lambda,
                            double alpha) {
  bench_function<AceHashFunctionV1<Key>>(keys, lambda, alpha);
  bench_function<AceHashFunctionV2<Key>>(keys, lambda, alpha);
  bench_function<AceHashFunctionV3<Key>>(keys, lambda, alpha);
  bench_function<AceHashFunctionV4<Key>>(keys, lambda, alpha);
#ifdef ACEHASH_ENABLE_TBB
  bench_function<AceHashFunctionV5<Key>>(keys, lambda, alpha);
#endif
}

template <bool minimal, typename Key>
void bench_function_pthash(const std::vector<Key> &keys,
                           double c,
                           double alpha) {
  bench_function<PTHashFunction<Key, pthash::partitioned_compact, minimal>>(
      keys, c, alpha);
  bench_function<PTHashFunction<Key, pthash::dictionary_dictionary, minimal>>(
      keys, c, alpha);
  bench_function<PTHashFunction<Key, pthash::elias_fano, minimal>>(
      keys, c, alpha);
}

template <typename Key> void bench_function_integer_1(size_t n) {
  std::vector<Key> keys = generate_integer_keys<Key>(n, generator);

  for (double lambda : {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5}) {
    for (double alpha : {0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0}) {
      bench_function_acehash(keys, lambda, alpha);
    }
  }

  for (double gamma : {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}) {
    for (bool parallel : {false, true}) {
      bench_function<BBHashFunction<Key>>(keys, gamma, parallel);
    }
  }

  for (double c : {4.0, 5.0, 6.0, 7.0, 8.0, 9.0}) {
    for (double alpha : {0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98}) {
      bench_function_pthash<false>(keys, c, alpha);
      bench_function_pthash<true>(keys, c, alpha);
    }
  }
}

template <typename Key> void bench_function_integer_2(size_t n) {
  std::vector<uint64_t> keys = generate_integer_keys<uint64_t>(n, generator);

  bench_function_acehash(keys, 4.0, 0.8);
  bench_function_acehash(keys, 2.5, 1.0);

  for (double gamma : {2.0, 5.0}) {
    for (bool parallel : {false, true}) {
      bench_function<BBHashFunction<Key>>(keys, gamma, parallel);
    }
  }

  bench_function<PTHashFunction<Key, pthash::dictionary_dictionary, false>>(
      keys, 5.0, 0.8);
  bench_function<PTHashFunction<Key, pthash::dictionary_dictionary, true>>(
      keys, 8.0, 0.98);
}

int main() {
  bench_function_integer_1<uint64_t>(10'000'000);

  for (size_t num_keys_e5 : {1, 2, 5, 10, 20, 50, 100, 200, 500, 1'000}) {
    bench_function_integer_2<uint64_t>(num_keys_e5 * 100'000);
  }

  return 0;
}
