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

  [[nodiscard]] size_t num_slots() const { return function_.nbKeys(); }

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

  [[nodiscard]] size_t num_slots() const { return function_.table_size(); }

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

  template <typename Callback>
  void operate(size_t n, const Key *keys, const Callback &callback) {
    for (size_t i = 0; i < n; ++i) {
      callback(i, map_.at(keys[i]));
    }
  }

  void retrieve(size_t n, const Key *keys, Value *values) {
    operate(n, keys, [&](size_t i, const Value &value) { values[i] = value; });
  }

  [[nodiscard]] size_t num_bits() const { return 0; }

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

template <typename Key, typename Value> class VectorMap {
public:
  VectorMap() : alpha_(0.0) {}

  VectorMap(size_t n, const Key *keys, const Value *values, double alpha)
      : alpha_(alpha) {
    uint32_t num_slots = std::ceil((double)n / alpha);

    items_.resize(num_slots);

    std::vector<bool> occupied(num_slots);

    for (size_t i = 0; i < n; ++i) {
      const Key &key = keys[i];
      const Value &value = values[i];

      auto initial_slot = occupied.begin() + get_initial_slot_index(key);

      auto it = std::find(initial_slot, occupied.end(), false);

      if (it == occupied.end()) {
        it = std::find(occupied.begin(), initial_slot, false);
        if (it == initial_slot) {
          throw std::runtime_error("map is full");
        }
      }

      items_[std::distance(occupied.begin(), it)] = {key, value};
      *it = true;
    }

    occupied.clear();
  }

  [[nodiscard]] size_t num_bits() const {
    return 8 * sizeof(std::pair<Key, Value>) * items_.size();
  }

  template <typename Callback>
  void operate(size_t n, const Key *keys, const Callback &callback) {
    for (size_t i = 0; i < n; ++i) {
      const Key &key = keys[i];

      auto initial_slot = items_.begin() + get_initial_slot_index(key);

      auto it = std::find_if(initial_slot, items_.end(), [&](const auto &item) {
        return item.first == key;
      });

      if (it == items_.end()) {
        it = std::find_if(items_.begin(), initial_slot, [&](const auto &item) {
          return item.first == key;
        });

        if (it == initial_slot) {
          throw std::runtime_error("key not found");
        }
      }

      callback(i, it->second);
    }
  }

  void retrieve(size_t n, const Key *keys, Value *values) {
    operate(n, keys, [&](size_t i, const Value &value) { values[i] = value; });
  }

  [[nodiscard]] std::string description() const {
    std::ostringstream out;
    out << "VectorMap;" << alpha_;
    return out.str();
  }

private:
  uint32_t get_initial_slot_index(const Key &key) {
    return (uint64_t)hasher_(key) * items_.size() >> 32;
  }

  double alpha_;
  acehash::MultiplyAddHasher<Key> hasher_;
  std::vector<std::pair<Key, Value>> items_;
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

  template <typename Callback>
  void operate(size_t n, const Key *keys, const Callback &callback) {
    constexpr size_t batch_length = 128;

    size_t i = 0;

    for (; i < n / batch_length * batch_length; i += batch_length) {
      uint32_t indices[batch_length];
      function_(batch_length, &keys[i], indices);

      for (size_t j = 0; j < batch_length; ++j) {
        callback(i + j, values_[indices[j]]);
      }
    }

    for (; i < n; ++i) {
      callback(i, values_[function_(keys[i])]);
    }
  }

  void retrieve(size_t n, const Key *keys, Value *values) {
    operate(n, keys, [&](size_t i, const Value &value) { values[i] = value; });
  }

  [[nodiscard]] size_t num_bits() const {
    return function_.num_bits() + 8 * sizeof(Value) * values_.size();
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

template <typename Key, typename Value>
using BBHashMap = PerfectMap<BBHashFunction<Key>, Key, Value>;

template <typename Key, typename Value, bool minimal>
using PTHashMap =
    PerfectMap<PTHashFunction<Key, pthash::dictionary_dictionary, minimal>,
               Key,
               Value>;

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
