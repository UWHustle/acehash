#include "common.hpp"

std::mt19937 generator(0); // NOLINT(cert-msc51-cpp)
int num_trials = 7;
int experiment;

ResultsFile out("bench_function.csv",
                "Experiment",
                "Number of keys",
                "Algorithm",
                "Parameters",
                "Trial",
                "Number of bits",
                "Construction time (s)",
                "Serial evaluation time (s)",
                "Parallel evaluation time (s)");

template <typename Function, typename Key, typename... Args>
void bench_function(const std::vector<Key> &build_keys,
                    const std::vector<Key> &probe_keys,
                    Args... args) {
  std::vector<uint32_t> indices(probe_keys.size());
  std::vector<uint32_t> validation_indices(build_keys.size());

  for (int trial = 0; trial < num_trials; ++trial) {
    double t1, t2, t3;
    Function function;

    // 1. Construct the function.
    t1 = time([&] {
      function = Function(build_keys.size(), build_keys.data(), args...);
    });

    function(build_keys.size(), build_keys.data(), validation_indices.data());
    validate(validation_indices);

    // 2. Evaluate the function (serial).
    t2 = time([&] {
      function(probe_keys.size(), probe_keys.data(), indices.data());
    });

#ifdef ACEHASH_ENABLE_TBB
    // 3. Evaluate the function (parallel).
    std::fill(indices.begin(), indices.end(), 0);
    t3 = time([&] {
      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, probe_keys.size()),
          [&](const tbb::blocked_range<size_t> &r) {
            function(r.size(), &probe_keys[r.begin()], &indices[r.begin()]);
          });
    });
#else
    t3 = 0.0;
#endif

    out.write(experiment,
              build_keys.size(),
              function.algorithm(),
              function.parameters(),
              trial,
              function.num_bits(),
              t1,
              t2,
              t3);
  }
}

template <typename Key>
void bench_function_experiment_1(const std::vector<Key> &build_keys,
                                 const std::vector<Key> &probe_keys) {
  experiment = 1;

  for (double lambda : {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5}) {
    for (double alpha : {0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0}) {
      bench_function<AceHashFunctionV4<Key>>(
          build_keys, probe_keys, lambda, alpha);
    }
  }

  //  for (double gamma : {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}) {
  //    for (bool parallel : {false, true}) {
  //      bench_function<BBHashFunction<Key>>(
  //          build_keys, probe_keys, gamma, parallel);
  //    }
  //  }

  //  for (double c : {4.0, 5.0, 6.0, 7.0, 8.0, 9.0}) {
  //    for (double alpha : {0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98}) {
  //      bench_function_pthash<false>(build_keys, probe_keys, c, alpha);
  //      bench_function_pthash<true>(build_keys, probe_keys, c, alpha);
  //    }
  //  }
}

template <typename Key>
void bench_function_experiment_2(const std::vector<Key> &build_keys,
                                 const std::vector<Key> &probe_keys) {
  experiment = 2;

  bench_function<AceHashFunctionV1<Key>>(build_keys, probe_keys, 2.5, 1.0);
  bench_function<AceHashFunctionV2<Key>>(build_keys, probe_keys, 2.5, 1.0);
  bench_function<AceHashFunctionV3<Key>>(build_keys, probe_keys, 2.5, 1.0);
  bench_function<AceHashFunctionV4<Key>>(build_keys, probe_keys, 2.5, 1.0);
#ifdef ACEHASH_ENABLE_TBB
  bench_function<AceHashFunctionV5<Key>>(build_keys, probe_keys, 2.5, 1.0);
  bench_function<AceHashFunctionV6<Key>>(build_keys, probe_keys, 2.5, 1.0);
#endif

  for (bool parallel : {false, true}) {
    bench_function<BBHashFunction<Key>>(build_keys, probe_keys, 2.0, parallel);
  }

  bench_function<PTHashFunction<Key, pthash::dictionary_dictionary, true>>(
      build_keys, probe_keys, 8.0, 0.98);
}

template <typename Key> void bench_function_integer() {
  std::vector<Key> build_keys;
  std::vector<Key> probe_keys;

  build_keys = generate_unique_integers<Key>(10'000'000, generator);
  probe_keys = sample(100'000'000, build_keys, generator);

  bench_function_experiment_1(build_keys, probe_keys);
  bench_function_experiment_2(build_keys, probe_keys);
}

int main() {
  bench_function_integer<uint64_t>();
  return 0;
}
