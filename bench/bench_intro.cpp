#include "common.hpp"

#include <random>
#include <unordered_set>
#include <vector>

class NodeMap {
public:
  NodeMap() = default;

  NodeMap(const std::vector<uint32_t> &keys,
          const std::vector<uint32_t> &values,
          double alpha) {
    slots_.resize(std::ceil((double)keys.size() / alpha));

    for (uint32_t i = 0; i < keys.size(); ++i) {
      uint32_t key = keys[i];
      Node *node = &get_slot(key);
      while (node->next) {
        node = node->next.get();
      }

      if (node->key != 0) {
        node->next = std::make_unique<Node>();
        node = node->next.get();
      }

      node->key = key;
      node->value = values[i];
    }
  }

  uint32_t &operator[](uint32_t key) {
    Node *node = &get_slot(key);
    while (node->key != key) {
      node = node->next.get();
    }

    return node->value;
  }

private:
  struct Node {
    uint32_t key = 0;
    uint32_t value = 0;
    std::unique_ptr<Node> next;
  };

  Node &get_slot(uint32_t key) {
    return slots_[(uint64_t)hasher_(key) * slots_.size() >> 32];
  }

  acehash::MultiplyAddHasher<uint32_t> hasher_;
  std::vector<Node> slots_;
};

class VectorMapSimd {
public:
  VectorMapSimd() = default;

  VectorMapSimd(const std::vector<uint64_t> &keys,
                const std::vector<uint64_t> &values,
                double alpha) {
    size_t num_slots = std::ceil((double)keys.size() / alpha);
    size_t num_blocks = num_slots / 4 + (num_slots % 4 != 0);

    std::cout << "num_blocks: " << num_blocks << std::endl;

    blocks_.resize(num_blocks);

    for (size_t i = 0; i < keys.size(); ++i) {
      uint64_t key = keys[i];
      auto it = probe(key, 0);
      *it.first = key;
      *it.second = values[i];
    }
  }

  uint64_t &operator[](uint64_t key) { return *probe(key, key).second; }

private:
  std::pair<uint64_t *, uint64_t *> probe(uint64_t key, uint64_t target) {
    __m256i targets = _mm256_set1_epi64x((long long)target);

    long hash = ((uint64_t)hasher_(key) * blocks_.size() >> 32);
    auto it1 = blocks_.begin() + hash;

    for (auto it2 = it1; it2 != blocks_.end(); ++it2) {
      uint64_t *block = it2->data();
      __m256i keys = _mm256_load_si256((__m256i *)block);
      __m256i result = _mm256_cmpeq_epi64(keys, targets);
      uint32_t mask = _mm256_movemask_epi8(result);
      if (mask != 0) {
        int k = std::countr_zero(mask) / 8;
        return {&block[k], &block[4 + k]};
      }
    }

    for (auto it2 = blocks_.begin(); it2 != it1; ++it2) {
      uint64_t *block = it2->data();
      __m256i keys = _mm256_load_si256((__m256i *)block);
      __m256i result = _mm256_cmpeq_epi64(keys, targets);
      uint32_t mask = _mm256_movemask_epi8(result);
      if (mask != 0) {
        int k = std::countr_zero(mask) / 8;
        return {&block[k], &block[4 + k]};
      }
    }

    throw std::runtime_error("could not find appropriate block");
  }

  acehash::MultiplyAddHasher<uint64_t> hasher_;
  std::vector<std::array<uint64_t, 8>> blocks_;
};

class VectorMap {
public:
  VectorMap() = default;

  VectorMap(const std::vector<uint32_t> &keys,
            const std::vector<uint32_t> &values,
            double alpha) {
    std::mt19937 generator(0);
    hasher_ = acehash::MultiplyAddHasher<uint32_t>(generator);
    slots_.resize(std::ceil((double)keys.size() / alpha));

    for (size_t i = 0; i < keys.size(); ++i) {
      uint32_t key = keys[i];
      Slot &slot = probe(key, 0);
      slot.first = key;
      slot.second = values[i];
    }
  }

  uint32_t &operator[](uint32_t key) { return probe(key, key).second; }

private:
  using Slot = std::pair<uint32_t, uint32_t>;

  Slot &probe(uint32_t key, uint32_t target) {
    auto search = [&](Slot *begin, Slot *end, Slot *&result) {
      __m256i targets = _mm256_set1_epi32((int)target);

      for (; begin + 4 < end; begin += 4) {
        __m256i block = _mm256_loadu_si256((__m256i *)begin);
        __m256i comparison = _mm256_cmpeq_epi32(block, targets);
        uint32_t mask = _mm256_movemask_epi8(comparison);
        if (mask != 0) {
          result = &begin[std::countr_zero(mask) / 8];
          return;
        }
      }

      for (; begin < end; ++begin) {
        if (begin->first == target) {
          result = begin;
          return;
        }
      }
    };

    Slot *initial_slot = &slots_[(uint64_t)hasher_(key) * slots_.size() >> 32];
    Slot *result = nullptr;

    search(initial_slot, &*slots_.end(), result);
    if (result) {
      return *result;
    }

    search(&*slots_.begin(), initial_slot, result);
    if (result) {
      return *result;
    }

    throw std::runtime_error("could not find appropriate slot");
  }

  acehash::MultiplyAddHasher<uint32_t> hasher_;
  std::vector<Slot> slots_;
};

void bench_intro(ResultsFile &out, int num_trials = 5) {
  uint32_t num_build = 1'000'000;
  uint32_t num_probe = 10'000'000;

  std::mt19937 rng(0); // NOLINT(cert-msc51-cpp)
  std::uniform_int_distribution<uint32_t> dis;

  std::unordered_set<uint32_t> build_keys_set;
  build_keys_set.reserve(num_build);
  while (build_keys_set.size() < num_build) {
    build_keys_set.emplace(dis(rng) | 1);
  }

  std::vector<uint32_t> build_keys(build_keys_set.begin(),
                                   build_keys_set.end());

  std::vector<uint32_t> build_values(num_build);
  for (uint32_t &value : build_values) {
    value = dis(rng) & ~uint32_t(1);
  }

  std::vector<uint32_t> probe_keys = sample(num_probe, build_keys, rng);
  std::vector<uint32_t> probe_values(num_probe);

  for (int alpha_percent = 1; alpha_percent <= 99; alpha_percent += 1) {
    double alpha = (double)alpha_percent / 100;
    std::cout << alpha << std::endl;

    VectorMap vector_map(build_keys, build_values, alpha);
    AceHashMapV4<uint32_t, uint32_t> acehash_map(
        num_build, build_keys.data(), build_values.data(), 2.5, alpha);

    for (int trial = 0; trial < num_trials; ++trial) {
      uint32_t result;
      double t;

      t = time([&] {
        result = 0;
        for (uint32_t key : probe_keys) {
          result ^= vector_map[key];
        }
      });
      std::cout << result << std::endl;

      out.write(trial, alpha, "Conventional", t);

      t = time([&] {
        result = 0;
        acehash_map.retrieve(num_probe, probe_keys.data(), probe_values.data());
        for (uint32_t value : probe_values) {
          result ^= value;
        }
      });
      std::cout << result << std::endl;

      out.write(trial, alpha, "AceHash", t);
    }
  }
}

int main() {
  ResultsFile out("bench_intro.csv", "Trial", "Alpha", "Algorithm", "Time (s)");
  bench_intro(out);
  return 0;
}
