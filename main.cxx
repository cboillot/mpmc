// main.cxx
#include "mpmc_candidate.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <memory>
#include <atomic>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <set>
#include <cstdint>
#include <cstring>
#include <random>
#include <ctime>
#include <x86intrin.h>
#include <vector>

#if defined(__linux__)
    #include <pthread.h>
    #include <sched.h>
#endif

constexpr size_t TEST_ELEMENTS = 100'000'000;

inline uint64_t read_tsc() {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
    return __rdtsc();
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#endif
}

inline uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

template <typename T>
struct CoreState {
    T current_val;
    size_t sequence_counter;

    void init(uint64_t seed) {
        current_val = static_cast<T>(seed);
        sequence_counter = 0;
    }

    inline T next() {
        current_val = fmix64(current_val);
        return current_val;
    }
};

template <typename T>
double compute_pure_compute_roofline_ms(size_t elements) {
    CoreState<T> state;
    state.init(12345);
    uint64_t expected_xor = 0;
    uint64_t actual_xor = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < elements; ++i) {
        T val = state.next();
        expected_xor ^= static_cast<uint64_t>(val);
        actual_xor ^= static_cast<uint64_t>(val);
    }
    auto end = std::chrono::high_resolution_clock::now();

    volatile uint64_t sink = expected_xor ^ actual_xor;
    (void)sink;

    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

class TopologyManager {
public:
    struct Core {
        unsigned int physical_core_id;
        std::vector<unsigned int> logical_threads;
    };
    std::vector<Core> node0_cores;
    std::vector<Core> node1_cores;

    TopologyManager() {
#if defined(__linux__)
        std::set<unsigned int> processed;
        unsigned int total = std::thread::hardware_concurrency();
        for (unsigned int i = 0; i < total; ++i) {
            if (processed.count(i)) continue;
            unsigned int node = 0;
            std::ifstream node_file("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/node0");
            if (!node_file.good()) node = 1;
            node_file.close();

            std::ifstream sib_file("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/thread_siblings_list");
            std::vector<unsigned int> siblings;
            if (sib_file.is_open()) {
                std::string line;
                if (std::getline(sib_file, line)) {
                    std::stringstream ss(line); std::string token;
                    while (std::getline(ss, token, ',')) {
                        if (token.find('-') != std::string::npos) {
                            size_t dash = token.find('-');
                            int start = std::stoi(token.substr(0, dash));
                            int end = std::stoi(token.substr(dash + 1));
                            for (int s = start; s <= end; ++s) siblings.push_back(s);
                        } else { siblings.push_back(std::stoi(token)); }
                    }
                }
            } else { siblings.push_back(i); }

            for (auto s : siblings) processed.insert(s);
            if (node == 0) node0_cores.push_back({i, siblings});
            else node1_cores.push_back({i, siblings});
        }
#else
        unsigned int total = std::thread::hardware_concurrency();
        for (unsigned int i = 0; i < total; ++i) {
            if (i < total / 2) node0_cores.push_back({i, {i}});
            else node1_cores.push_back({i, {i}});
        }
#endif
    }

    std::vector<unsigned int> get_cpu_pool(size_t node, size_t count) {
        auto& cores = (node == 0) ? node0_cores : node1_cores;
        std::vector<unsigned int> cpus;
        for (const auto& c : cores) {
            if (!c.logical_threads.empty()) {
                cpus.push_back(c.logical_threads[0]);
                if (cpus.size() >= count) return cpus;
            }
        }
        return cpus;
    }
};

static TopologyManager g_topo;

void bind_thread(unsigned int cpu_id) {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

struct TestMetrics {
    double duration_ms;
    double cpu_time_s;
    uint64_t cycles_spent;
};

template <typename T>
TestMetrics run_test_case(size_t num_producers, size_t num_consumers) {
    MpmcChannel<T> channel(num_producers, num_consumers);

    std::vector<unsigned int> p_cpus = g_topo.get_cpu_pool(0, num_producers);
    std::vector<unsigned int> c_cpus = g_topo.get_cpu_pool(0, num_producers + num_consumers);
    if(c_cpus.size() > num_producers) {
        c_cpus.erase(c_cpus.begin(), c_cpus.begin() + num_producers);
    }

    std::atomic<uint64_t> expected_combined_xor{0};
    std::atomic<uint64_t> actual_combined_xor{0};

    std::vector<std::thread> producers, consumers;
    std::random_device rd;
    std::mt19937_64 eng(rd());

    // Calculate baseline loads statically per thread to completely avoid benchmark-side contention
    size_t base_elements_per_producer = TEST_ELEMENTS / num_producers;
    size_t remainder_elements = TEST_ELEMENTS % num_producers;

    std::clock_t start_cpu = std::clock();
    uint64_t start_cycles = read_tsc();
    auto start_wall = std::chrono::high_resolution_clock::now();

    for (size_t id = 0; id < num_consumers; ++id) {
        consumers.emplace_back([&, id]() {
            if (!c_cpus.empty()) bind_thread(c_cpus[id % c_cpus.size()]);
            
            auto cons_handle = channel.get_consumer_handle();
            uint64_t local_xor = 0;
            
            while (true) {
                // Capture reference from our local handle safety wrapper
                auto&& item = cons_handle.pop();
                if (cons_handle.exhausted()) break;
                
                local_xor ^= static_cast<uint64_t>(item);
            }
            actual_combined_xor.fetch_xor(local_xor, std::memory_order_relaxed);
        });
    }

    for (size_t id = 0; id < num_producers; ++id) {
        uint64_t thread_random_seed = eng(); 
        size_t total_assigned_elements = base_elements_per_producer + (id == num_producers - 1 ? remainder_elements : 0);

        producers.emplace_back([&, id, thread_random_seed, total_assigned_elements]() {
            if (!p_cpus.empty()) bind_thread(p_cpus[id % p_cpus.size()]);
            
            auto prod_handle = channel.get_producer_handle();
            CoreState<T> state; 
            state.init(thread_random_seed); 
            uint64_t running_producer_xor = 0;

            for (size_t i = 0; i < total_assigned_elements; ++i) {
                T val = state.next();
                prod_handle.push(val);
                running_producer_xor ^= static_cast<uint64_t>(val);
            }
            expected_combined_xor.fetch_xor(running_producer_xor, std::memory_order_relaxed);
        });
    }

    for (auto& p : producers) p.join();
    for (auto& c : consumers) c.join();

    uint64_t end_cycles = read_tsc();
    std::clock_t end_cpu = std::clock();
    auto end_wall = std::chrono::high_resolution_clock::now();

    if (expected_combined_xor.load() != actual_combined_xor.load()) {
        std::cerr << "CRITICAL ERROR: Checksum mismatch " << expected_combined_xor << " != " << actual_combined_xor << ".\n";
    }
    return {std::chrono::duration<double, std::milli>(end_wall - start_wall).count(), static_cast<double>(end_cpu - start_cpu) / CLOCKS_PER_SEC, (end_cycles > start_cycles) ? (end_cycles - start_cycles) : 0};
}

int main() {
    uint64_t global_total_messages = 0, global_total_bytes = 0;
    double global_wall_time_ms = 0.0, global_cpu_time_s = 0.0;
    
    double best_mops = 0.0, best_mops_gib = 0.0, best_mops_cyc = 0.0;
    std::string best_mops_cfg;
    double best_gib = 0.0, best_gib_mops = 0.0, best_gib_cyc = 0.0;
    std::string best_gib_cfg;

    auto run_suite = [&](const char* type_label, auto type_dummy) {
        using MsgType = decltype(type_dummy);
        
        double base_ms = compute_pure_compute_roofline_ms<MsgType>(TEST_ELEMENTS);
        double base_mops = (TEST_ELEMENTS / (base_ms / 1000.0)) / 1'000'000.0;

        std::cout << "\n--- Testing Data Payload Type: " << type_label << " (" << sizeof(MsgType) << " Bytes) ---\n";

        for (auto const& topo : {std::make_pair(1,1), {1,4}, {2,2}, {4,4}, {4,8}, {1,20}, {16,16}, {24,24}}) {
            TestMetrics m = run_test_case<MsgType>(topo.first, topo.second);
            double s = m.duration_ms / 1000.0;
            double mops = (TEST_ELEMENTS / s) / 1'000'000.0;
            double gib_s = ((TEST_ELEMENTS * sizeof(MsgType)) / (1024.0*1024.0*1024.0)) / s;
            double cyc = (double)m.cycles_spent / TEST_ELEMENTS;
            double efficiency = (mops / base_mops) * 100.0;

            global_total_messages += TEST_ELEMENTS; global_total_bytes += (TEST_ELEMENTS * sizeof(MsgType)); global_wall_time_ms += m.duration_ms; global_cpu_time_s += m.cpu_time_s;

            std::stringstream ss; ss << type_label << " (" << sizeof(MsgType) << "B), " << topo.first << " P -> " << topo.second << " C";
            if (mops > best_mops) { best_mops = mops; best_mops_gib = gib_s; best_mops_cyc = cyc; best_mops_cfg = ss.str(); }
            if (gib_s > best_gib) { best_gib = gib_s; best_gib_mops = mops; best_gib_cyc = cyc; best_gib_cfg = ss.str(); }

            std::cout << std::setw(3) << topo.first << " P -> " << std::setw(2) << topo.second << " C | "
                      << std::fixed << std::setprecision(2) << std::setw(8) << mops << " Mops/s | "
                      << std::setw(7) << gib_s << " GiB/s | "
                      << std::setw(7) << cyc << " cyc/msg | "
                      << "Efficiency: " << std::setw(6) << efficiency << "%\n";
        }
    };

    run_suite("uint8_t", uint8_t{}); run_suite("uint32_t", uint32_t{}); run_suite("uint64_t", uint64_t{});
    double total_s = global_wall_time_ms / 1000.0;
    
    std::cout << "\n=========================================================\n"
              << "AGGREGATED SUITE METRICS:\n"
              << "  -> Total Messages: " << global_total_messages << " elements\n"
              << "  -> Avg Throughput: " << std::fixed << std::setprecision(2) << (global_total_messages / 1'000'000.0) / total_s << " Mops/s\n"
              << "  -> Avg Bandwidth:  " << (global_total_bytes / (1024.0*1024.0*1024.0)) / total_s << " GiB/s\n"
              << "  -> Total Wall Time:" << total_s << " s\n"
              << "  -> Total CPU Time: " << global_cpu_time_s << " s\n"
              << "---------------------------------------------------------\n"
              << "MAX THROUGHPUT CONFIGURATION:\n"
              << "  -> Configuration:  " << best_mops_cfg << "\n"
              << "  -> Max Throughput: " << best_mops << " Mops/s\n"
              << "  -> Bandwidth:      " << best_mops_gib << " GiB/s\n"
              << "  -> Cycle Cost:     " << best_mops_cyc << " cycles/message\n"
              << "---------------------------------------------------------\n"
              << "MAX BANDWIDTH CONFIGURATION:\n"
              << "  -> Configuration:  " << best_gib_cfg << "\n"
              << "  -> Throughput:     " << best_gib_mops << " Mops/s\n"
              << "  -> Max Bandwidth:  " << best_gib << " GiB/s\n"
              << "  -> Cycle Cost:     " << best_gib_cyc << " cycles/message\n"
              << "=========================================================\n";
    return 0;
}