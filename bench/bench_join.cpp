#include "common.hpp"

#ifdef ACEHASH_ENABLE_DUCKDB
#include "duckdb.hpp"
#endif

std::mt19937 generator(0); // NOLINT(cert-msc51-cpp)
int num_trials = 7;

ResultsFile out("bench_join.csv",
                "Key type",
                "Value type",
                "Number of keys",
                "Algorithm",
                "Parameters",
                "Trial",
                "Number of bits",
                "Construction time (s)",
                "Serial evaluation time (s)",
                "Parallel evaluation time (s)");

template <typename Map, typename Key, typename Value, typename... Args>
void bench_join(const std::vector<Key> &build_keys,
                const std::vector<Value> &build_values,
                const std::vector<Key> &probe_keys,
                Args... args) {
  std::vector<Value> probe_values(probe_keys.size());

  for (int trial = 0; trial < num_trials; ++trial) {
    double t1, t2, t3;
    Map map;

    // 1. Construct the map.
    t1 = time([&] {
      map = Map(
          build_keys.size(), build_keys.data(), build_values.data(), args...);
    });

    // 2. Evaluate the map (serial).
    t2 = time([&] {
      map.retrieve(probe_keys.size(), probe_keys.data(), probe_values.data());
    });

#ifdef ACEHASH_ENABLE_TBB
    // 3. Evaluate the map (parallel).
    t3 = time([&] {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, probe_keys.size()),
                        [&](const tbb::blocked_range<size_t> &r) {
                          map.retrieve(r.size(),
                                       &probe_keys[r.begin()],
                                       &probe_values[r.begin()]);
                        });
    });
#else
    t3 = 0.0;
#endif

    out.write(typeid(Key).name(),
              typeid(Value).name(),
              build_keys.size(),
              map.algorithm(),
              map.parameters(),
              trial,
              map.num_bits(),
              t1,
              t2,
              t3);
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
void bench_join_duckdb(const std::vector<Key> &build_keys,
                       const std::vector<Value> &build_values,
                       const std::vector<Key> &probe_keys) {
  duckdb::DuckDB db;
  duckdb::Connection con(db);

  con.Query("SET threads=1")->Print();

  std::string key_type = get_duckdb_type<Key>();
  std::string value_type = get_duckdb_type<Value>();

  con.Query("CREATE TABLE R (a " + key_type + ", b " + value_type + ")")
      ->Print();
  con.Query("CREATE TABLE S (c " + key_type + ")")->Print();

  duckdb::Appender appender_r(con, "R");
  for (size_t i = 0; i < build_keys.size(); ++i) {
    appender_r.AppendRow(build_keys[i], build_values[i]);
  }
  appender_r.Close();

  duckdb::Appender appender_s(con, "S");
  for (size_t i = 0; i < probe_keys.size(); ++i) {
    appender_s.AppendRow(probe_keys[i]);
  }
  appender_s.Close();

  con.Query("SELECT bit_xor(b) FROM R, S WHERE a = c")->Print();

  for (int trial = 0; trial < num_trials; ++trial) {
    double t = time(
        [&] { con.Query("SELECT bit_xor(b) FROM R, S WHERE a = c")->Print(); });

    out.write(typeid(Key).name(),
              typeid(Value).name(),
              build_keys.size(),
              "DuckDB",
              "",
              trial,
              0,
              0,
              t,
              0);
  }
}
#endif

template <typename Key, typename Value> void bench_join_integer(size_t n) {
  std::vector<Key> build_keys = generate_unique_integers<Key>(n, generator);
  std::vector<Value> build_values = generate_integers<Value>(n, generator);
  std::vector<Key> probe_keys = sample(100'000'000, build_keys, generator);

  bench_join<AceHashMapV4<Key, Value>>(
      build_keys, build_values, probe_keys, 2.5, 1.0);
  bench_join<StlMap<Key, Value>>(build_keys, build_values, probe_keys);
  bench_join<AbseilMap<Key, Value>>(build_keys, build_values, probe_keys);
  bench_join<FollyMap<Key, Value>>(build_keys, build_values, probe_keys);
  bench_join<BBHashMap<Key, Value>>(
      build_keys, build_values, probe_keys, 2.0, false);
  bench_join<PTHashMap<Key, Value, true>>(
      build_keys, build_values, probe_keys, 8.0, 0.98);
#ifdef ACEHASH_ENABLE_DUCKDB
  bench_join_duckdb(build_keys, build_values, probe_keys);
#endif
}

int main() {
  for (size_t num_keys_e5 : {1, 2, 5, 10, 20, 50, 100, 200, 500, 1'000}) {
    size_t num_keys = num_keys_e5 * 100'000;
    bench_join_integer<uint64_t, uint64_t>(num_keys);
    bench_join_integer<uint64_t, uint8_t>(num_keys);
  }

  return 0;
}
