#include "acehash/acehash.hpp"

#include <iostream>
#include <random>
#include <unordered_set>

using namespace acehash;

template <typename Function, typename Config>
void test(const std::vector<uint64_t> &keys_vector, Config config) {
  Function function(keys_vector.begin(), keys_vector.end(), config);

  std::vector<uint32_t> indices_vector(keys_vector.size());
  std::unordered_set<uint32_t> indices_set;

  std::transform(keys_vector.begin(),
                 keys_vector.end(),
                 indices_vector.begin(),
                 [&](uint64_t key) { return function(key); });

  indices_set = {indices_vector.begin(), indices_vector.end()};
  if (indices_set.size() != keys_vector.size()) {
    throw std::runtime_error("test failed");
  }

  function(keys_vector.size(), keys_vector.data(), indices_vector.data());

  indices_set = {indices_vector.begin(), indices_vector.end()};
  if (indices_set.size() != keys_vector.size()) {
    throw std::runtime_error("test failed");
  }
}

int main() {
  uint32_t num_keys = 100'000;

  std::minstd_rand rng(0); // NOLINT(cert-msc51-cpp)
  std::uniform_int_distribution<uint64_t> dis;

  std::unordered_set<uint64_t> keys_set;
  keys_set.reserve(num_keys);

  while (keys_set.size() < num_keys) {
    keys_set.emplace(dis(rng));
  }

  std::vector<uint64_t> keys_vector(keys_set.begin(), keys_set.end());

  std::vector<std::pair<double, double>> params = {{2.0, 1.0}, {2.0, 0.99}};

  for (auto [lambda, alpha] : params) {
    test<AceHash<uint64_t, acehash::MultiplyAddHasher<uint64_t>, false, false>>(
        keys_vector, Config<>{.lambda = lambda, .alpha = alpha});
    test<AceHash<uint64_t, acehash::MultiplyAddHasher<uint64_t>, true, false>>(
        keys_vector, Config<>{.lambda = lambda, .alpha = alpha});
    test<AceHash<uint64_t, acehash::MultiplyAddHasher<uint64_t>, true, true>>(
        keys_vector, Config<>{.lambda = lambda, .alpha = alpha});

#ifdef ACEHASH_ENABLE_TBB
    test<AceHash<uint64_t>>(
        keys_vector, Config<TBBScheduler>{.lambda = lambda, .alpha = alpha});
#endif
  }

  return 0;
}
