#pragma once

#include "BooPHF.h"
#include "absl/container/flat_hash_map.h"
#include "acehash/acehash.hpp"
#include "pthash.hpp"

#include <chrono>
#include <random>
#include <sstream>
#include <unordered_set>

template <typename Key,
          bool multi_round_partition,
          bool multi_slot_search,
          bool multi_key_evaluate,
          typename Scheduler = acehash::SerialScheduler>
class AceHashFunction {
public:
  AceHashFunction() = default;

  AceHashFunction(size_t n, const Key *keys, double lambda, double alpha)
      : lambda_(lambda), alpha_(alpha),
        function_(
            keys,
            keys + n,
            acehash::Config<Scheduler>{.lambda = lambda_, .alpha = alpha_}) {}

  uint32_t operator()(const Key &key) const { return function_(key); }

  void operator()(size_t n, const Key *keys, uint32_t *indices) const {
    if (multi_key_evaluate) {
      function_(n, keys, indices);
    } else {
      for (size_t i = 0; i < n; ++i) {
        indices[i] = (*this)(keys[i]);
      }
    }
  }

  [[nodiscard]] size_t num_slots() const { return function_.num_slots(); }

  [[nodiscard]] size_t num_bits() const { return function_.num_bits(); }

  [[nodiscard]] std::string description() const {
    std::ostringstream out;
    out << "AceHash;" << lambda_ << ';' << alpha_ << ';'
        << multi_round_partition << ';' << multi_slot_search << ';'
        << multi_key_evaluate << ';'
        << !std::is_same_v<Scheduler, acehash::SerialScheduler>;
    return out.str();
  }

private:
  double lambda_{};
  double alpha_{};
  acehash::AceHash<Key,
                   acehash::MultiplyAddHasher<Key>,
                   multi_round_partition,
                   multi_slot_search>
      function_;
};

template <typename Key>
using AceHashFunctionV1 =
    AceHashFunction<Key, false, false, false, acehash::SerialScheduler>;

template <typename Key>
using AceHashFunctionV2 =
    AceHashFunction<Key, true, false, false, acehash::SerialScheduler>;

template <typename Key>
using AceHashFunctionV3 =
    AceHashFunction<Key, true, true, false, acehash::SerialScheduler>;

template <typename Key>
using AceHashFunctionV4 =
    AceHashFunction<Key, true, true, true, acehash::SerialScheduler>;

#ifdef ACEHASH_ENABLE_TBB
template <typename Key>
using AceHashFunctionV5 =
    AceHashFunction<Key, true, true, true, acehash::TBBScheduler>;
#endif

template <typename Key> class BBHashFunction {
public:
  BBHashFunction() = default;

  BBHashFunction(size_t n, const Key *keys, double gamma, bool parallel)
      : gamma_(gamma), parallel_(parallel),
        function_(n,
                  Range{keys, n},
                  parallel_ ? std::thread::hardware_concurrency() : 1,
                  gamma,
                  true,
                  false,
                  0) {}

  uint32_t operator()(const Key &key) const {
    return const_cast<Function &>(function_).lookup(key);
  }

  void operator()(size_t n, const Key *keys, uint32_t *indices) const {
    for (size_t i = 0; i < n; ++i) {
      indices[i] = (*this)(keys[i]);
    }
  }

  [[nodiscard]] size_t num_bits() const {
    int old_fd, new_fd;
    fflush(stdout);
    old_fd = dup(1);
    new_fd = open("/dev/null", O_WRONLY);
    dup2(new_fd, 1);
    close(new_fd);

    size_t num_bits = const_cast<Function &>(function_).totalBitSize();

    fflush(stdout);
    dup2(old_fd, 1);
    close(old_fd);

    return num_bits;
  }

  [[nodiscard]] std::string description() const {
    std::ostringstream out;
    out << "BBHash;" << gamma_ << ';' << parallel_;
    return out.str();
  }

private:
  using Function = boomphf::mphf<Key, boomphf::SingleHashFunctor<Key>>;

  struct Range {
    const Key *begin() const { return keys; }

    const Key *end() const { return keys + n; }

    const Key *keys;
    size_t n;
  };

  double gamma_{};
  bool parallel_{};
  Function function_;
};

template <typename Key, typename Encoder, bool minimal> class PTHashFunction {
public:
  PTHashFunction() = default;

  PTHashFunction(size_t n, const Key *keys, double c, double alpha)
      : c_(c), alpha_(alpha) {
    pthash::build_configuration config;
    config.c = c_;
    config.alpha = alpha_;
    config.verbose_output = false;
    config.minimal_output = minimal;
    function_.build_in_internal_memory(keys, n, config);
  }

  uint32_t operator()(const Key &key) const { return function_(key); }

  void operator()(size_t n, const Key *keys, uint32_t *indices) const {
    for (size_t i = 0; i < n; ++i) {
      indices[i] = (*this)(keys[i]);
    }
  }

  [[nodiscard]] size_t num_bits() const { return function_.num_bits(); }

  [[nodiscard]] std::string description() const {
    std::ostringstream out;
    out << "PTHash;" << c_ << ';' << alpha_ << ';' << Encoder().name() << ';'
        << minimal;
    return out.str();
  }

private:
  double c_{};
  double alpha_{};
  pthash::single_phf<pthash::murmurhash2_64, Encoder, minimal> function_;
};

template <typename Map, typename Key, typename Value> class ConventionalMap {
public:
  ConventionalMap() = default;

  ConventionalMap(size_t n, const Key *keys, const Value *values)
      : map_(ZipIterator{keys, values}, ZipIterator{keys + n, values + n}) {}

  void operator()(size_t n, const Key *keys, Value *values) {
    for (size_t i = 0; i < n; ++i) {
      values[i] = map_.at(keys[i]);
    }
  }

private:
  struct ZipIterator {
    using difference_type = size_t;
    using value_type = std::pair<const Key &, const Value &>;
    using pointer = value_type *;
    using reference = value_type &;
    using iterator_category = std::random_access_iterator_tag;

    std::pair<const Key &, const Value &> operator*() const {
      return {*key, *value};
    }

    ZipIterator &operator++() {
      ++key;
      ++value;
      return *this;
    }

    friend bool operator!=(const ZipIterator &a, const ZipIterator &b) {
      return a.key != b.key;
    }

    friend difference_type operator-(const ZipIterator &a,
                                     const ZipIterator &b) {
      return a.key - b.key;
    }

    const Key *key;
    const Value *value;
  };

  Map map_;
};

template <typename Key, typename Value>
class StlMap
    : public ConventionalMap<std::unordered_map<Key, Value>, Key, Value> {
public:
  using ConventionalMap<std::unordered_map<Key, Value>, Key, Value>::
      ConventionalMap;

  [[nodiscard]] std::string description() const { return "std::unordered_map"; }
};

template <typename Key, typename Value>
class AbseilMap
    : public ConventionalMap<absl::flat_hash_map<Key, Value>, Key, Value> {
public:
  using ConventionalMap<absl::flat_hash_map<Key, Value>, Key, Value>::
      ConventionalMap;

  [[nodiscard]] std::string description() const {
    return "absl::flat_hash_map";
  }
};

template <typename Function, typename Key, typename Value> class PerfectMap {
public:
  PerfectMap() = default;

  template <typename... Args>
  PerfectMap(size_t n, const Key *keys, const Value *values, Args... args)
      : function_(n, keys, args...), values_(function_.num_slots()) {
    for (size_t i = 0; i < n; ++i) {
      values_[function_(keys[i])] = values[i];
    }
  }

  void operator()(size_t n, const Key *keys, Value *values) {
    for (size_t i = 0; i < n; ++i) {
      values[i] = values_[function_(keys[i])];
    }
  }

  [[nodiscard]] std::string description() const {
    return function_.description();
  }

private:
  Function function_;
  std::vector<Value> values_;
};

template <typename Key, typename Value>
using AceHashMapV1 = PerfectMap<AceHashFunctionV1<Key>, Key, Value>;

template <typename Key, typename Value>
using AceHashMapV2 = PerfectMap<AceHashFunctionV2<Key>, Key, Value>;

template <typename Key, typename Value>
using AceHashMapV3 = PerfectMap<AceHashFunctionV3<Key>, Key, Value>;

template <typename Key, typename Value>
using AceHashMapV4 = PerfectMap<AceHashFunctionV4<Key>, Key, Value>;

class ResultsFile {
public:
  template <typename... Values>
  explicit ResultsFile(const std::string &name, Values... header) : out(name) {
    write(header...);
  }

  template <typename Value> void write(const Value &value) {
    out << value << std::endl;
  }

  template <typename Value, typename... Values>
  void write(const Value &value, Values... values) {
    out << value << ',';
    write(values...);
  }

private:
  std::ofstream out;
};

template <typename F> double time(F &&f) {
  auto t0 = std::chrono::high_resolution_clock::now();
  f();
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

template <typename Key, typename Generator>
std::vector<Key> generate_integer_keys(uint32_t n, Generator &generator) {
  std::unordered_set<Key> keys;
  keys.reserve(n);

  std::uniform_int_distribution<Key> dis;

  while (n) {
    n -= keys.insert(dis(generator)).second;
  }

  return {keys.begin(), keys.end()};
}

template <typename Value, typename Generator>
std::vector<Value> generate_integer_values(size_t n, Generator &generator) {
  std::vector<Value> values(n);

  std::uniform_int_distribution<Value> dis;

  for (Value &value : values) {
    value = dis(generator);
  }

  return values;
}

void validate(const std::vector<uint32_t> &indices) {
  if (std::unordered_set<uint32_t>(indices.begin(), indices.end()).size() !=
      indices.size()) {
    throw std::runtime_error("incorrect");
  }
}

template <typename Value>
void validate(const std::vector<Value> &retrieved_values,
              const std::vector<Value> &values) {
  if (retrieved_values != values) {
    throw std::runtime_error("incorrect");
  }
}
