// mpmc_channel_test.cxx - correctness tests for mpmc_channel.hpp
#include "mpmc_channel.hpp"
#include "mpmc_shared_queue.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#define CHECK(cond) \
    do { if (!(cond)) { std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

namespace {

std::atomic<long> g_live{0};

// move-only, not default-constructible, counts live instances
struct Tracked {
    std::uint64_t value;
    explicit Tracked(std::uint64_t v) noexcept : value(v) { g_live.fetch_add(1, std::memory_order_relaxed); }
    Tracked(Tracked&& o) noexcept : value(o.value) { g_live.fetch_add(1, std::memory_order_relaxed); }
    Tracked& operator=(Tracked&& o) noexcept { value = o.value; return *this; }
    Tracked(const Tracked&) = delete;
    Tracked& operator=(const Tracked&) = delete;
    ~Tracked() { g_live.fetch_sub(1, std::memory_order_relaxed); }
};

void test_basic() {
    mpmc::channel_options o;
    o.capacity = 8;
    o.chunk = 100;
    mpmc::channel<int> ch(1, 1, o);
    CHECK(ch.capacity() == 8);
    auto p = ch.get_producer();
    auto c = ch.get_consumer();
    int v = -1;
    CHECK(!c.try_pop(v));
    for (int i = 0; i < 8; ++i) CHECK(p.try_push(i));
    CHECK(!p.try_push(99));                         // full
    for (int i = 0; i < 8; ++i) { CHECK(c.try_pop(v)); CHECK(v == i); }
    CHECK(!c.try_pop(v));
    CHECK(!c.exhausted());
    for (int lap = 0; lap < 1000; ++lap) {          // wrap around many times
        CHECK(p.try_push(lap));
        CHECK(c.try_pop(v));
        CHECK(v == lap);
    }
    p = {};                                         // last producer gone
    CHECK(!c.pop(v));
    CHECK(c.exhausted());
}

void test_limits() {
    mpmc::channel<int> ch(2, 2, 64);
    auto p1 = ch.get_producer();
    auto p2 = ch.get_producer();
    bool threw = false;
    try { auto p3 = ch.get_producer(); } catch (const std::length_error&) { threw = true; }
    CHECK(threw);
    auto c1 = ch.get_consumer();
    {
        auto c2 = ch.get_consumer();
        threw = false;
        try { auto c3 = ch.get_consumer(); } catch (const std::length_error&) { threw = true; }
        CHECK(threw);
    }
    auto c2 = ch.get_consumer();                    // the slot was released
    CHECK(static_cast<bool>(c2));
}

void test_close() {
    mpmc::channel<int> ch(8, 1, 64);                // 8 declared, only 1 ever created
    auto c = ch.get_consumer();
    int v = 0;
    {
        auto p = ch.get_producer();
        p.push(7);
        ch.close();
        bool threw = false;
        try { auto q = ch.get_producer(); } catch (const std::length_error&) { threw = true; }
        CHECK(threw);
    }
    CHECK(c.pop(v) && v == 7);
    CHECK(!c.pop(v));
    CHECK(c.exhausted());
}

void test_pop_until() {
    mpmc::channel<int> ch(1, 1, 64);
    auto p = ch.get_producer();
    auto c = ch.get_consumer();
    int v = 0;
    int polls = 0;
    CHECK(!c.pop_until(v, [&] { return ++polls > 100; }));
    CHECK(!c.exhausted());
    p.push(5);
    CHECK(c.pop_until(v, [] { return true; }) && v == 5);
}

void test_move_only_and_lifetime() {
    {
        mpmc::channel<std::unique_ptr<int>> ch(1, 1, 16);
        auto p = ch.get_producer();
        auto c = ch.get_consumer();
        p.push(std::make_unique<int>(42));
        std::unique_ptr<int> out;
        CHECK(c.try_pop(out) && *out == 42);
    }
    CHECK(g_live.load() == 0);
    {
        // 3 rings, one consumer that leaves early: items are left in a parked
        // reservation, in a stash and unreserved; all must be destroyed once.
        mpmc::channel_options o;
        o.capacity = 256;
        o.chunk = 50;
        mpmc::channel<Tracked> ch(1, 3, o);
        {
            auto p = ch.get_producer();
            for (std::uint64_t i = 0; i < 300; ++i) p.push(Tracked(i));
        }
        CHECK(g_live.load() == 300);
        {
            auto c = ch.get_consumer();
            Tracked out(0);
            for (int i = 0; i < 120; ++i) CHECK(c.pop(out));
        }
        {
            auto c = ch.get_consumer();             // picks the first ring up again
            Tracked out(0);
            for (int i = 0; i < 20; ++i) CHECK(c.pop(out));
        }
    }
    CHECK(g_live.load() == 0);
}

// a consumer that leaves in the middle of a batch hands the rest back in order
void test_handover() {
    mpmc::channel<int> ch(1, 1, 256);
    auto p = ch.get_producer();
    for (int i = 0; i < 100; ++i) p.push(i);
    int v = -1;
    {
        auto c = ch.get_consumer();
        for (int i = 0; i < 10; ++i) { CHECK(c.pop(v)); CHECK(v == i); }
    }
    auto c = ch.get_consumer();
    for (int i = 10; i < 100; ++i) { CHECK(c.try_pop(v)); CHECK(v == i); }
    CHECK(!c.try_pop(v));
}

// A consumer that owns a ring but does not pop must not make that ring
// unusable: thieves empty it and hand its slots back to the producer.
void test_stalled_consumer(bool stall_mid_batch) {
    mpmc::channel_options o;
    o.capacity = 64;
    o.chunk = 16;
    o.max_batch = 8;
    o.steal_min = 1;
    o.steal_batch = 8;
    const std::uint64_t total = 200000;
    mpmc::channel<std::uint64_t> ch(1, 2, o);
    auto stalled = ch.get_consumer();               // owns ring 0
    std::atomic<std::uint64_t> sum{0}, count{0};
    std::uint64_t held = 0;

    std::thread producer([&] {
        auto p = ch.get_producer();
        for (std::uint64_t i = 1; i <= total; ++i) p.push(i);
    });
    if (stall_mid_batch) {
        // take one item, which reserves a batch, then stop popping
        CHECK(stalled.pop(held));
        sum += held;
        ++count;
    }
    std::thread worker([&] {
        auto c = ch.get_consumer();                 // owns ring 1
        std::uint64_t v = 0;
        while (c.pop(v)) { sum += v; ++count; }
    });
    producer.join();
    worker.join();
    // whatever the stalled consumer had reserved is still its own
    std::uint64_t v = 0;
    while (stalled.pop(v)) { sum += v; ++count; }
    CHECK(count.load() == total);
    CHECK(sum.load() == total * (total + 1) / 2);
}

// every value in [0, total) must arrive exactly once
void run_stress(std::size_t producers, std::size_t rings, std::size_t consumers, std::size_t per_producer,
                mpmc::channel_options o, bool slow_first_consumer, bool use_try) {
    const std::size_t total = producers * per_producer;
    std::vector<std::atomic<std::uint8_t>> seen(total);
    for (auto& s : seen) s.store(0);
    std::atomic<std::size_t> received{0};
    {
        mpmc::channel<Tracked> ch(producers, rings, o);
        std::vector<std::thread> threads;
        for (std::size_t id = 0; id < consumers; ++id) {
            threads.emplace_back([&, id] {
                auto c = ch.get_consumer();
                Tracked out(0);
                std::size_t n = 0;
                for (;;) {
                    bool got = use_try ? c.try_pop(out) : c.pop(out);
                    if (!got) {
                        if (c.exhausted()) break;
                        std::this_thread::yield();
                        continue;
                    }
                    CHECK(out.value < total);
                    CHECK(seen[out.value].fetch_add(1, std::memory_order_relaxed) == 0);
                    ++n;
                    if (slow_first_consumer && id == 0 && (n & 63) == 0) std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                received.fetch_add(n);
            });
        }
        for (std::size_t id = 0; id < producers; ++id) {
            threads.emplace_back([&, id] {
                auto p = ch.get_producer();
                for (std::size_t i = 0; i < per_producer; ++i) {
                    Tracked item(id * per_producer + i);
                    if (use_try) {
                        while (!p.try_push(std::move(item))) std::this_thread::yield();
                    } else {
                        p.push(std::move(item));
                    }
                }
            });
        }
        for (auto& t : threads) t.join();
    }
    CHECK(received.load() == total);
    for (auto& s : seen) CHECK(s.load() == 1);
    CHECK(g_live.load() == 0);
}

void test_stress() {
    mpmc::channel_options small;
    small.capacity = 64;
    small.chunk = 16;
    small.max_batch = 8;
    small.steal_min = 2;
    small.steal_batch = 8;

    mpmc::channel_options mid;
    mid.capacity = 4096;
    mid.chunk = 1000;
    mid.steal_min = 8;

    const std::size_t n = 200000;
    run_stress(1, 1, 1, n, small, false, false);
    run_stress(4, 4, 4, n, small, false, false);
    run_stress(4, 4, 4, n, small, true, false);     // one slow consumer: the others must steal from it
    run_stress(4, 4, 1, n, small, false, false);    // three rings have no consumer at all
    run_stress(3, 8, 2, n, mid, false, false);
    run_stress(8, 2, 2, n, small, false, false);    // more producers than rings
    run_stress(4, 4, 4, n, mid, true, true);        // non-blocking API
    run_stress(2, 6, 3, n, small, true, true);
    run_stress(4, 4, 4, n, mpmc::channel_options{}, false, false);
}

// A producer that goes quiet in the middle of a chunk keeps its ring to
// itself.  The consumer of that ring must not starve: it moves to a free ring.
void test_idle_producer() {
    mpmc::channel_options o;
    o.capacity = 256;
    o.chunk = 100;
    o.max_batch = 16;
    o.steal_min = 1000000;                          // no stealing from live consumers: only migration can help
    mpmc::channel<int> ch(2, 4, o);
    auto quiet = ch.get_producer();
    auto c = ch.get_consumer();                     // ring 0
    quiet.push(-1);                                 // takes ring 0 and then never pushes again
    int v = 0;
    CHECK(c.pop(v) && v == -1);
    std::thread busy([&] {
        auto p = ch.get_producer();
        for (int i = 0; i < 100000; ++i) p.push(i);
    });
    long long sum = 0;
    for (int i = 0; i < 100000; ++i) { CHECK(c.pop(v)); sum += v; }
    busy.join();
    CHECK(sum == 100000LL * 99999 / 2);
}

// shared_queue: threads come and go, every item arrives exactly once
void test_shared_queue() {
    {
        mpmc::shared_queue<std::unique_ptr<int>> q(64, 4);
        q.push(std::make_unique<int>(3));
        std::unique_ptr<int> out;
        CHECK(q.try_pop(out) && *out == 3);
        CHECK(!q.try_pop(out));
        mpmc::shared_queue<std::unique_ptr<int>> other(64, 4);  // two queues of one type on one thread
        other.push(std::make_unique<int>(4));
        q.push(std::make_unique<int>(5));
        CHECK(other.try_pop(out) && *out == 4);
        CHECK(q.try_pop(out) && *out == 5);
    }
    {
        // more threads than max_threads at the same time
        mpmc::shared_queue<int> q(64, 1);
        q.push(1);
        bool threw = false;
        std::thread t([&] { try { q.push(2); } catch (const std::length_error&) { threw = true; } });
        t.join();
        CHECK(threw);
    }
    for (int round = 0; round < 3; ++round) {
        const std::size_t total = 300000, workers = 4, per_thread = 3000;
        std::vector<std::atomic<std::uint8_t>> seen(total);
        for (auto& s : seen) s.store(0);
        std::atomic<std::size_t> received{0};
        std::atomic<std::size_t> next{0};
        mpmc::channel_options o = mpmc::shared_queue<Tracked>::default_options(1024);
        o.chunk = 32;
        o.max_batch = 16;
        o.steal_min = 4;
        o.steal_batch = 16;
        mpmc::shared_queue<Tracked> q(o, 8);
        auto record = [&](const Tracked& item) {
            CHECK(item.value < total);
            CHECK(seen[item.value].fetch_add(1) == 0);
            received.fetch_add(1);
        };
        // Short-lived threads push a share and pop what they find, then exit:
        // slots, rings, reservations and stashes change hands all the time.
        while (next.load() < total) {
            std::vector<std::thread> threads;
            for (std::size_t w = 0; w < workers; ++w) {
                threads.emplace_back([&] {
                    Tracked out(0);
                    for (std::size_t n = 0; n < per_thread; ++n) {
                        const std::size_t i = next.fetch_add(1);
                        if (i >= total) break;
                        Tracked item(i);
                        while (!q.try_push(std::move(item))) {      // full: make room
                            if (q.try_pop(out)) record(out);
                        }
                        if ((n & 3) != 0 && q.try_pop(out)) record(out);
                    }
                });
            }
            for (auto& t : threads) t.join();
        }
        // every thread is gone; all that was not popped must still be there
        Tracked out(0);
        while (received.load() < total) {
            CHECK(q.pop_until(out, [n = 0]() mutable { return ++n > 100000; }));
            record(out);
        }
        CHECK(!q.try_pop(out));
        for (auto& s : seen) CHECK(s.load() == 1);
    }
    CHECK(g_live.load() == 0);
}

} // namespace

int main() {
    test_basic();
    test_limits();
    test_close();
    test_pop_until();
    test_handover();
    test_move_only_and_lifetime();
    test_stalled_consumer(false);
    test_stalled_consumer(true);
    test_idle_producer();
    test_stress();
    test_shared_queue();
    std::puts("mpmc_channel: all tests passed");
    return 0;
}
