#include "acehash/acehash.hpp"

#include <cstdint>
#include <iostream>
#include <random>
#include <unordered_set>

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

  acehash::Config config;

  auto t0 = std::chrono::high_resolution_clock::now();
  acehash::AceHash<uint64_t> m(keys_vector.begin(), keys_vector.end(), config);

  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration<double>(t1 - t0).count() << std::endl;

  std::vector<uint32_t> positions_vector(num_keys);
  auto t2 = std::chrono::high_resolution_clock::now();

  //  for (uint32_t i = 0; i < num_keys; ++i) {
  //    positions_vector[i] = m(keys_vector[i]);
  //  }

  m(num_keys, keys_vector.data(), positions_vector.data());

  auto t3 = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration<double>(t3 - t2).count() << std::endl;

  std::unordered_set<uint32_t> positions_set(positions_vector.begin(),
                                             positions_vector.end());

  std::cout << positions_set.size() << std::endl;
  std::cout << *std::max_element(positions_vector.begin(),
                                 positions_vector.end())
            << std::endl;

  return 0;
}
