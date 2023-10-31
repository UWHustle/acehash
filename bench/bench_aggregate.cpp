#include "common.hpp"

#ifdef ACEHASH_ENABLE_DUCKDB
#include "duckdb.hpp"
#endif

std::mt19937 generator(0); // NOLINT(cert-msc51-cpp)
int num_trials = 5;

ResultsFile out("bench_aggregate.csv",
                "Key type",
                "Value type",
                "Key cardinality",
                "Algorithm",
                "Parameters",
                "Trial",
                "Number of bits",
                "Build time (s)",
                "Serial aggregate time (s)");

template <typename Map, typename Key, typename Value, typename... Args>
void bench_aggregate(const std::vector<Key> &build_keys,
                     const std::vector<Key> &agg_keys,
                     const std::vector<Value> &agg_values,
                     Args... args) {
  std::vector<Value> agg_values_initial(build_keys.size());
  std::vector<Value> agg_values_final(build_keys.size());

  for (int trial = 0; trial < num_trials; ++trial) {
    double t1, t2;
    Map map;

    // 1. Build.
    t1 = time([&] {
      map = Map(build_keys.size(),
                build_keys.data(),
                agg_values_initial.data(),
                args...);
    });

    // 2. Aggregate (serial).
    t2 = time([&] {
      map.operate(agg_keys.size(),
                  agg_keys.data(),
                  [&](size_t i, Value &value) { value += agg_values[i]; });

      map.retrieve(
          build_keys.size(), build_keys.data(), agg_values_final.data());
    });

    out.write(typeid(Key).name(),
              typeid(Value).name(),
              build_keys.size(),
              map.algorithm(),
              map.parameters(),
              trial,
              map.num_bits(),
              t1,
              t2);
  }
}

template <typename T> std::string get_duckdb_type() {
  if (std::is_same_v<T, uint64_t>) {
    return "UBIGINT";
  }

  if (std::is_same_v<T, uint32_t>) {
    return "UINT";
  }

  if (std::is_same_v<T, uint8_t>) {
    return "UTINYINT";
  }

  throw std::logic_error("unsupported type");
}

#ifdef ACEHASH_ENABLE_DUCKDB
template <typename Key, typename Value>
void bench_aggregate_duckdb(const std::vector<Key> &build_keys,
                            const std::vector<Key> &agg_keys,
                            const std::vector<Value> &agg_values) {
  duckdb::DuckDB db;
  duckdb::Connection con(db);

  con.Query("SET threads=1")->Print();

  std::string key_type = get_duckdb_type<Key>();
  std::string value_type = get_duckdb_type<Value>();

  con.Query("CREATE TABLE T (a " + key_type + ", b " + value_type + ")")
      ->Print();

  duckdb::Appender appender(con, "T");
  for (size_t i = 0; i < agg_keys.size(); ++i) {
    appender.AppendRow(agg_keys[i], agg_values[i]);
  }
  appender.Close();

  con.Query("SELECT a, bit_xor(b) FROM T GROUP BY a")->Fetch();

  for (int trial = 0; trial < num_trials; ++trial) {
    double t = time(
        [&] { con.Query("SELECT a, bit_xor(b) FROM T GROUP BY a")->Fetch(); });

    out.write(typeid(Key).name(),
              typeid(Value).name(),
              build_keys.size(),
              "DuckDB",
              "",
              trial,
              0,
              0,
              t);
  }
}
#endif

template <typename Key, typename Value> void bench_aggregate_integer(size_t n) {
  std::vector<Key> build_keys = generate_unique_integers<Key>(n, generator);
  std::vector<Key> agg_keys = sample(100'000'000, build_keys, generator);
  std::vector<Value> agg_values =
      generate_integers<Value>(100'000'000, generator);

  bench_aggregate<AceHashMapV4<Key, Value>>(
      build_keys, agg_keys, agg_values, 2.5, 1.0);
  bench_aggregate<StlMap<Key, Value>>(build_keys, agg_keys, agg_values);
  bench_aggregate<AbseilMap<Key, Value>>(build_keys, agg_keys, agg_values);
  bench_aggregate<FollyMap<Key, Value>>(build_keys, agg_keys, agg_values);
  bench_aggregate<BBHashMap<Key, Value>>(
      build_keys, agg_keys, agg_values, 2.0, false);
  bench_aggregate<PTHashMap<Key, Value, true>>(
      build_keys, agg_keys, agg_values, 8.0, 0.98);
#ifdef ACEHASH_ENABLE_DUCKDB
  bench_aggregate_duckdb(build_keys, agg_keys, agg_values);
#endif
}

int main() {
  for (size_t num_keys_e5 : {1, 2, 5, 10, 20, 50, 100, 200, 500, 1'000}) {
    size_t num_keys = num_keys_e5 * 100'000;
    bench_aggregate_integer<uint64_t, uint64_t>(num_keys);
    bench_aggregate_integer<uint64_t, uint8_t>(num_keys);
  }

  return 0;
}
