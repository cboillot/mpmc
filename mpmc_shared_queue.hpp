// mpmc_shared_queue.hpp
//
// mpmc::shared_queue<T>: mpmc::channel behind a conventional queue interface.
//
//     mpmc::shared_queue<int> queue(capacity);
//     queue.push(42);                     // any thread
//     int value;
//     if (queue.try_pop(value)) { ... }   // any thread
//
// The channel wants one producer handle and one consumer handle per thread.
// This wrapper creates them the first time a thread pushes to / pops from a
// queue, keeps them in the queue and finds them again through a thread-local
// cache, so callers never see a handle.  When a thread exits, its handles go
// back to the queue: what it had not consumed stays available to the others.
//
// Costs compared with using mpmc::channel directly: one thread-local compare
// per operation, and `max_threads` rings of `capacity` slots each.
//
//   * at most max_threads threads may use one queue at the same time
//     (std::length_error otherwise)
//   * no ordering guarantee between items (see mpmc_channel.hpp)
//   * the queue must not be destroyed while another thread is inside a call

#pragma once

#include "mpmc_channel.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mpmc {

namespace detail {
    // Queues that are alive, so that a thread that exits knows whether the
    // queues it used still exist.  Only touched when a thread first uses a
    // queue, when a thread exits and when a queue is destroyed.
    struct queue_registry {
        std::mutex mutex;
        std::unordered_set<std::uint64_t> live;

        static queue_registry& instance() {
            static queue_registry* r = new queue_registry;    // never destroyed: threads may outlive statics
            return *r;
        }
        static std::uint64_t next_id() noexcept {
            static std::atomic<std::uint64_t> id{0};
            return id.fetch_add(1, std::memory_order_relaxed) + 1;  // 0 means "none"
        }
    };
}

template <typename T>
class shared_queue {
    using channel_type = channel<T>;

    struct alignas(detail::cache_line) slot {
        typename channel_type::producer producer;
        typename channel_type::consumer consumer;
        std::atomic<bool> used{false};
    };

public:
    static std::size_t default_max_threads() noexcept {
        return std::max<std::size_t>(16, 2 * std::thread::hardware_concurrency());
    }

    // Settings that suit a queue shared by threads that both push and pop.
    static channel_options default_options(std::size_t capacity) noexcept {
        channel_options o;
        o.capacity = capacity;
        o.chunk = std::max<std::size_t>(1, std::min<std::size_t>(4096, capacity / 4));
        o.max_batch = 256;
        o.steal_min = 64;
        o.steal_batch = 64;
        return o;
    }

    // capacity: slots per ring; the queue holds at least that many items.
    explicit shared_queue(std::size_t capacity, std::size_t max_threads = default_max_threads())
        : shared_queue(default_options(capacity), max_threads) {}

    explicit shared_queue(const channel_options& options, std::size_t max_threads = default_max_threads())
        : channel_(~std::size_t{0} / 2, std::max<std::size_t>(1, max_threads), options),
          slots_(new slot[std::max<std::size_t>(1, max_threads)]),
          nslots_(std::max<std::size_t>(1, max_threads)),
          id_(detail::queue_registry::next_id()) {
        auto& registry = detail::queue_registry::instance();
        std::lock_guard<std::mutex> lock(registry.mutex);
        registry.live.insert(id_);
    }

    ~shared_queue() {
        {
            // from here on, exiting threads leave our slots alone
            auto& registry = detail::queue_registry::instance();
            std::lock_guard<std::mutex> lock(registry.mutex);
            registry.live.erase(id_);
        }
        slots_.reset();                             // the handles go before the channel
    }

    shared_queue(const shared_queue&) = delete;
    shared_queue& operator=(const shared_queue&) = delete;

    // Blocks (spins, then yields) while the queue is full.
    template <typename U>
    void push(U&& value) {
        push_slot().producer.push(std::forward<U>(value));
    }

    template <typename U>
    bool try_push(U&& value) {
        return push_slot().producer.try_push(std::forward<U>(value));
    }

    bool try_pop(T& out) {
        return pop_slot().consumer.try_pop(out);
    }

    // Blocks (spins, then yields) until an item arrives.
    void pop(T& out) {
        pop_slot().consumer.pop(out);
    }

    // Like pop(), but gives up (returns false) once stop() returns true.
    template <typename Stop>
    bool pop_until(T& out, Stop&& stop) {
        return pop_slot().consumer.pop_until(out, stop);
    }

    std::size_t capacity() const noexcept { return channel_.capacity(); }
    std::size_t max_threads() const noexcept { return nslots_; }

private:
    // What a thread remembers about the queue it used last.  Trivial on
    // purpose: reading it must not go through a thread-local init guard.
    struct cache {
        std::uint64_t push_id;
        slot* push_slot;
        std::uint64_t pop_id;
        slot* pop_slot;
    };

    // Every queue of this type the thread is attached to; gives the slots
    // back when the thread exits.
    struct attachments {
        struct entry {
            std::uint64_t id;
            shared_queue* queue;
            slot* where;
            bool pops;
        };
        std::vector<entry> entries;

        ~attachments() {
            auto& registry = detail::queue_registry::instance();
            for (entry& e : entries) {
                std::lock_guard<std::mutex> lock(registry.mutex);
                if (registry.live.count(e.id)) e.queue->release(*e.where);
            }
            tls_cache_ = cache{};
        }

        entry* find(std::uint64_t id) noexcept {
            for (entry& e : entries) if (e.id == id) return &e;
            return nullptr;
        }

        // forget queues that no longer exist
        void prune() {
            auto& registry = detail::queue_registry::instance();
            std::lock_guard<std::mutex> lock(registry.mutex);
            for (std::size_t i = 0; i < entries.size();) {
                if (registry.live.count(entries[i].id)) { ++i; continue; }
                entries[i] = entries.back();
                entries.pop_back();
            }
        }
    };

    static inline thread_local cache tls_cache_{};
    static inline thread_local attachments tls_attachments_{};

    slot& push_slot() {
        cache& c = tls_cache_;
        if (MPMC_CHANNEL_UNLIKELY(c.push_id != id_)) return attach(false);
        return *c.push_slot;
    }

    slot& pop_slot() {
        cache& c = tls_cache_;
        if (MPMC_CHANNEL_UNLIKELY(c.pop_id != id_)) return attach(true);
        return *c.pop_slot;
    }

    // Slow path: first use of this queue by this thread (for pushing or for
    // popping), or the thread used another queue in between.
    MPMC_CHANNEL_NOINLINE slot& attach(bool pops) {
        attachments& mine = tls_attachments_;
        auto* e = mine.find(id_);
        if (!e) {
            mine.prune();
            slot* free_slot = nullptr;
            for (std::size_t i = 0; i < nslots_ && !free_slot; ++i) {
                slot& s = slots_[i];
                if (!s.used.load(std::memory_order_relaxed) && !s.used.exchange(true, std::memory_order_acquire)) free_slot = &s;
            }
            if (!free_slot) throw std::length_error("mpmc::shared_queue: more threads than max_threads");
            free_slot->producer = channel_.get_producer();
            mine.entries.push_back({id_, this, free_slot, false});
            e = &mine.entries.back();
        }
        if (pops && !e->pops) {
            // only a thread that pops takes a ring: a ring is where producers send items
            e->where->consumer = channel_.get_consumer();
            e->pops = true;
        }
        cache& c = tls_cache_;
        c.push_id = id_;
        c.push_slot = e->where;
        c.pop_id = e->pops ? id_ : 0;
        c.pop_slot = e->where;
        return *e->where;
    }

    // A thread is done with its slot (called with the registry locked).
    void release(slot& s) {
        if (s.consumer) {
            // items it had stolen and not popped yet go back into the queue
            s.consumer.drain_private([&s](T&& item) { s.producer.push(std::move(item)); });
            s.consumer = typename channel_type::consumer{};     // its unread reservation stays in its ring
        }
        s.producer = typename channel_type::producer{};
        s.used.store(false, std::memory_order_release);
    }

    channel_type channel_;                          // declared first: outlives the handles in the slots
    std::unique_ptr<slot[]> slots_;
    std::size_t nslots_;
    std::uint64_t id_;
};

} // namespace mpmc
