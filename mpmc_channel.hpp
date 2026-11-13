// mpmc_channel.hpp
//
// Header-only multi-producer / multi-consumer channel.
//
// Layout: one single-producer/single-consumer ring per consumer slot.
//
//   * A producer takes exclusive ownership of one ring (one atomic exchange),
//     pushes a chunk of items into it and then moves on to the next ring.
//   * A consumer owns one ring.  It reserves a batch of items with one CAS on
//     the ring's `claim` index and then pops that batch without any shared
//     read: the hot path of pop() is one compare, one move and one store.
//   * Everything a ring holds above `claim` is not reserved by anybody, so an
//     idle consumer may steal it.  The owner notices a theft for free: its
//     next reservation CAS fails and tells it where to resume.
//
// Hot paths (push with room left in the chunk, pop with items left in the
// batch) execute a single, well predicted branch.  Ownership, reservation,
// back-off, stealing and termination all live in the out-of-line slow paths.
//
// Guarantees
//   * every pushed item is delivered exactly once
//   * items pushed by one producer into one ring keep their order; there is
//     no ordering across rings or across a steal
//   * T must be nothrow-move-constructible; pop() needs T move-assignable
//
// Lifetime rules
//   * the channel must outlive every handle
//   * a handle is used by one thread at a time
//   * consumers report exhaustion once max_producers producer handles have
//     been created and destroyed (or close() was called and the existing
//     ones were destroyed) and every item has been handed out
//   * a consumer handle that is destroyed leaves its unread reservation in
//     the ring for the next consumer of that ring or for thieves; only items
//     it had already stolen and not popped are dropped (at most steal_batch)
//   * a consumer that stops popping in the middle of a batch keeps that batch
//     (at most max_batch items) and, once its ring has wrapped around to it,
//     blocks that one ring; producers then use the other rings

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    #include <immintrin.h>
    #define MPMC_CHANNEL_RELAX() _mm_pause()
#elif defined(__aarch64__)
    #define MPMC_CHANNEL_RELAX() __asm__ __volatile__("yield")
#else
    #define MPMC_CHANNEL_RELAX() ((void)0)
#endif

#if defined(__GNUC__) || defined(__clang__)
    #define MPMC_CHANNEL_UNLIKELY(x) __builtin_expect(!!(x), 0)
    #define MPMC_CHANNEL_NOINLINE __attribute__((noinline))
#else
    #define MPMC_CHANNEL_UNLIKELY(x) (x)
    #define MPMC_CHANNEL_NOINLINE
#endif

// Define MPMC_CHANNEL_STATS to count what the slow paths do (reservations,
// steals, ring switches, waits...).  The counters live in the handles and are
// added up in the channel when a handle goes away; channel::print_stats()
// prints the totals.  Nothing is counted on the hot paths, and nothing at all
// is compiled in without the macro.
#if defined(MPMC_CHANNEL_STATS)
    #include <cstdio>
    #define MPMC_CHANNEL_STAT(name) (++stats_[static_cast<std::size_t>(detail::stat::name)])
    #define MPMC_CHANNEL_STAT_ADD(name, n) (stats_[static_cast<std::size_t>(detail::stat::name)] += (n))
#else
    #define MPMC_CHANNEL_STAT(name) ((void)0)
    #define MPMC_CHANNEL_STAT_ADD(name, n) ((void)0)
#endif

namespace mpmc {

struct channel_options {
    // slots per ring (rounded up to a power of two)
    std::size_t capacity = std::size_t{1} << 20;
    // pushes a producer makes into one ring before it moves to the next one
    std::size_t chunk = 30000;
    // largest batch a consumer reserves for itself in one go; what it does not
    // reserve stays available to thieves
    std::size_t max_batch = 4096;
    // Unreserved backlog a ring must hold before an idle consumer steals from
    // it while its owner is alive.  Stealing is a safety net for consumers that
    // are stalled or cannot keep up, so the default (0 = capacity / 8) stays out
    // of the way of a healthy stream; lower it to balance load more eagerly.
    // Rings without a consumer, and any ring once the producers are done, are
    // always fair game.
    std::size_t steal_min = 0;
    // most items moved by one steal
    std::size_t steal_batch = 128;
    // A consumer that finds fewer items than this is following its producer
    // too closely: every refill would bounce the producer's cache line for a
    // handful of items.  It then pauses (up to close_spins pauses, adapting to
    // what the pause actually yields) to let a batch build up.  0 disables it.
    std::size_t close_threshold = 1024;
    std::uint32_t close_spins = 256;
    // A producer paired with a consumer (producer::pair_with) pushes into
    // that consumer's ring for as long as the ring holds fewer items than
    // this; beyond that its chunks go to the other rings.  0 = capacity / 2.
    std::size_t home_high_water = 0;
    // When not 0, a consumer that finds another consumer's ring holding this
    // many more unreserved items than its own takes a batch of them (checked
    // on one ring every few refills).  It evens out the supplies of threads
    // that re-push what they pop, which otherwise never change: such a thread
    // never runs dry, so it never steals.  0 = off.
    std::size_t balance_threshold = 0;
};

namespace detail {
    constexpr std::size_t cache_line = 64;

    inline std::size_t ceil_pow2(std::size_t v) noexcept {
        std::size_t p = 2;
        while (p < v) p <<= 1;
        return p;
    }

    // spin with a growing number of pauses, then start giving the CPU away
    inline void idle_wait(std::uint32_t round) noexcept {
        if (round <= 64) {
            for (std::uint32_t i = 0, n = std::min<std::uint32_t>(round, 16); i < n; ++i) MPMC_CHANNEL_RELAX();
        } else {
            std::this_thread::yield();
        }
    }

    struct never_stop { bool operator()() const noexcept { return false; } };

    inline int count_trailing_zeros(std::uint64_t v) noexcept {
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_ctzll(v);
#else
        int n = 0;
        while ((v & 1) == 0) { v >>= 1; ++n; }
        return n;
#endif
    }

#if defined(MPMC_CHANNEL_STATS)
    enum class stat : std::size_t {
        // consumer
        consumers,          // consumer handles that reported
        refills,            // times a consumer ran out of its batch
        own_batches,        // batches reserved from its own ring
        own_items,          // items in those batches
        own_empty,          // looks at its own ring that found nothing
        thefts_noticed,     // reservations that found part of the ring stolen
        close_pauses,       // pauses spent letting a batch build up
        steal_tries,        // attempts to steal from a ring
        steal_contended,    // attempts lost to somebody else
        steals,             // successful steals
        stolen_items,       // items moved by them
        steals_orphan,      // ... of which from a ring without consumer
        steals_patrol,      // ... of which by the periodic check, not for lack of work
        steals_balance,     // ... of which to even out the rings of two consumers
        migrations,         // moves to another ring
        idle_rounds,        // waits with nothing to pop
        yields,             // ... of which gave the cpu away
        // producer
        producers,          // producer handles that reported
        acquires,           // times a producer ran out of chunk or room
        room_refreshes,     // ... solved by re-reading the consumer's position
        chunk_ends,         // ... because the chunk was complete
        ring_full,          // ... because the ring was full
        ring_switches,      // rings taken
        home_rings_taken,   // ... of which the ring of the paired consumer
        orphan_rings_taken, // ... of which had no consumer
        busy_skips,         // rings passed over because another producer held them
        full_skips,         // rings passed over because they were full
        producer_waits,     // sweeps over all rings that found none
        count_
    };

    constexpr char const* stat_names[] = {
        "consumers", "refills", "own_batches", "own_items", "own_empty", "thefts_noticed", "close_pauses",
        "steal_tries", "steal_contended", "steals", "stolen_items", "steals_orphan", "steals_patrol", "steals_balance",
        "migrations", "idle_rounds", "yields",
        "producers", "acquires", "room_refreshes", "chunk_ends", "ring_full", "ring_switches",
        "home_rings_taken", "orphan_rings_taken", "busy_skips", "full_skips", "producer_waits"};

    using stat_array = std::uint64_t[static_cast<std::size_t>(stat::count_)];
#endif
}

template <typename T>
class channel {
    static_assert(std::is_nothrow_move_constructible<T>::value, "T must be nothrow move constructible");
    static_assert(std::is_nothrow_destructible<T>::value, "T must be nothrow destructible");

    struct alignas(detail::cache_line) ring {
        // read-only after construction
        T* data{nullptr};
        std::size_t mask{0};
        std::size_t capacity{0};

        // written by the owning producer on every push
        alignas(detail::cache_line) std::atomic<std::size_t> head{0};

        // producer ownership; kept off the head line because other producers poll it
        alignas(detail::cache_line) std::atomic<bool> producer_owned{false};
        std::size_t cached_tail{0};                 // guarded by producer_owned

        // written by the owning consumer on every pop, read by the producer when it runs out of room
        alignas(detail::cache_line) std::atomic<std::size_t> tail{0};

        // everything below claim is reserved by a consumer; written once per batch, read by thieves
        alignas(detail::cache_line) std::atomic<std::size_t> claim{0};
        std::atomic<bool> consumer_owned{false};
        // one thief at a time; also taken briefly when a consumer arrives or leaves
        std::atomic<bool> steal_lock{false};
        // reservation left behind by the last owner: [parked_pos, parked_limit)
        std::atomic<std::size_t> parked_pos{0};
        std::atomic<std::size_t> parked_limit{0};
        // stolen range whose slots have not been given back to the producer yet
        std::size_t stolen_lo{0};                   // guarded by steal_lock
        std::size_t stolen_hi{0};

        // What the owning consumer saw left over the last time it reserved a
        // batch.  Other consumers read this, not head/claim, to judge whether
        // the ring is better supplied than theirs: it changes once per batch,
        // so looking at it does not slow the owner down.
        alignas(detail::cache_line) std::atomic<std::size_t> supply{0};

        void lock_steal() noexcept {
            while (steal_lock.exchange(true, std::memory_order_seq_cst)) MPMC_CHANNEL_RELAX();
        }
        void unlock_steal() noexcept { steal_lock.store(false, std::memory_order_seq_cst); }
    };

public:
    class producer;
    class consumer;

    channel(std::size_t max_producers, std::size_t max_consumers, channel_options options = {})
        : opts_(options),
          nrings_(max_consumers ? max_consumers : 1),
          max_producers_(max_producers),
          unfinished_(max_producers),
          created_(0) {
        opts_.capacity = detail::ceil_pow2(opts_.capacity);
        opts_.chunk = std::max<std::size_t>(1, opts_.chunk);
        opts_.max_batch = std::max<std::size_t>(1, opts_.max_batch);
        opts_.steal_batch = std::max<std::size_t>(1, opts_.steal_batch);
        if (opts_.steal_min == 0) opts_.steal_min = std::max<std::size_t>(1, opts_.capacity / 8);
        opts_.close_spins = std::max<std::uint32_t>(1, opts_.close_spins);
        if (opts_.home_high_water == 0) opts_.home_high_water = std::max<std::size_t>(1, opts_.capacity / 2);

        rings_.reset(new ring[nrings_]);
        const std::size_t words = (nrings_ + 63) / 64;
        attended_.reset(new std::atomic<std::uint64_t>[words]());
        abandoned_.reset(new std::atomic<std::uint64_t>[words]());
        const std::size_t bytes = opts_.capacity * sizeof(T);
        const std::align_val_t align{std::max(alignof(T), detail::cache_line)};
        for (std::size_t i = 0; i < nrings_; ++i) {
            void* raw = ::operator new(bytes, align);
            std::memset(raw, 0, bytes);             // fault the pages in now, not on the first lap
            rings_[i].data = static_cast<T*>(raw);
            rings_[i].mask = opts_.capacity - 1;
            rings_[i].capacity = opts_.capacity;
        }
    }

    channel(std::size_t max_producers, std::size_t max_consumers, std::size_t capacity)
        : channel(max_producers, max_consumers, with_capacity(capacity)) {}

    ~channel() noexcept {
        const std::align_val_t align{std::max(alignof(T), detail::cache_line)};
        for (std::size_t i = 0; i < nrings_; ++i) {
            ring& r = rings_[i];
            if (!std::is_trivially_destructible<T>::value) {
                destroy_range(r, r.parked_pos.load(std::memory_order_relaxed), r.parked_limit.load(std::memory_order_relaxed));
                destroy_range(r, r.claim.load(std::memory_order_relaxed), r.head.load(std::memory_order_relaxed));
            }
            ::operator delete(r.data, align);
        }
    }

    channel(const channel&) = delete;
    channel& operator=(const channel&) = delete;

    // Throws std::length_error once max_producers handles have been handed out
    // (or after close()).
    producer get_producer() {
        std::size_t n = created_.load(std::memory_order_relaxed);
        do {
            if (n >= max_producers_) throw std::length_error("mpmc::channel: no producer slot left");
        } while (!created_.compare_exchange_weak(n, n + 1, std::memory_order_acq_rel, std::memory_order_relaxed));
        return producer(this, n % nrings_);
    }

    // Throws std::length_error when every ring already has a consumer.
    consumer get_consumer() {
        for (std::size_t i = 0; i < nrings_; ++i) {
            ring& r = rings_[i];
            if (!r.consumer_owned.load(std::memory_order_relaxed) &&
                !r.consumer_owned.exchange(true, std::memory_order_acquire)) {
                return consumer(this, i);
            }
        }
        throw std::length_error("mpmc::channel: no consumer slot left");
    }

    // No more producer handles will be created.  Consumers report exhaustion
    // once the existing producer handles are gone and the rings are drained.
    void close() noexcept {
        const std::size_t before = created_.exchange(max_producers_, std::memory_order_acq_rel);
        if (before < max_producers_) unfinished_.fetch_sub(max_producers_ - before, std::memory_order_release);
    }

    std::size_t capacity() const noexcept { return opts_.capacity; }
    std::size_t rings() const noexcept { return nrings_; }
    const channel_options& options() const noexcept { return opts_; }

#if defined(MPMC_CHANNEL_STATS)
    // Totals of the handles destroyed so far, plus what the rings hold now.
    void print_stats(std::FILE* out = stderr, char const* label = "mpmc") const {
        std::fprintf(out, "[%s stats]", label);
        for (std::size_t i = 0; i < static_cast<std::size_t>(detail::stat::count_); ++i) {
            std::fprintf(out, " %s=%llu", detail::stat_names[i],
                         static_cast<unsigned long long>(totals_[i].load(std::memory_order_relaxed)));
        }
        std::size_t used = 0, owned = 0, held = 0, most = 0, items = 0;
        for (std::size_t i = 0; i < nrings_; ++i) {
            const ring& r = rings_[i];
            const std::size_t fill = r.head.load(std::memory_order_relaxed) - r.tail.load(std::memory_order_relaxed);
            used += (r.head.load(std::memory_order_relaxed) != 0);
            owned += r.consumer_owned.load(std::memory_order_relaxed);
            held += r.producer_owned.load(std::memory_order_relaxed);
            most = std::max(most, fill);
            items += fill;
        }
        std::fprintf(out, " | rings=%zu ever_used=%zu with_consumer=%zu producer_held=%zu fill_total=%zu fill_max=%zu\n",
                     nrings_, used, owned, held, items, most);
    }
#endif

private:
    static channel_options with_capacity(std::size_t capacity) noexcept {
        channel_options o;
        o.capacity = capacity;
        o.chunk = std::min(o.chunk, std::max<std::size_t>(1, capacity / 2));
        return o;
    }

    static void destroy_range(ring& r, std::size_t from, std::size_t to) noexcept {
        for (; from != to; ++from) r.data[from & r.mask].~T();
    }

    // One bit per ring, in two maps.  They let a consumer go straight to the
    // rings that matter instead of visiting all of them in turn, which with
    // many more rings than threads took too long to be of any use.
    //   attended:  the ring has a consumer
    //   abandoned: the ring has no consumer and may hold items
    using ring_map = std::unique_ptr<std::atomic<std::uint64_t>[]>;

    std::size_t index_of(const ring& r) const noexcept { return static_cast<std::size_t>(&r - rings_.get()); }

    static void mark(ring_map& map, std::size_t i) noexcept {
        map[i / 64].fetch_or(std::uint64_t{1} << (i % 64), std::memory_order_seq_cst);
    }
    static void unmark(ring_map& map, std::size_t i) noexcept {
        map[i / 64].fetch_and(~(std::uint64_t{1} << (i % 64)), std::memory_order_seq_cst);
    }
    // first marked ring after `from`, wrapping around; nrings_ if there is none
    std::size_t next_marked(const ring_map& map, std::size_t from) const noexcept {
        const std::size_t words = (nrings_ + 63) / 64;
        std::size_t start = from + 1;
        if (start >= nrings_) start = 0;
        std::size_t w = start / 64;
        std::uint64_t bits = map[w].load(std::memory_order_relaxed) & (~std::uint64_t{0} << (start % 64));
        for (std::size_t n = 0; n <= words; ++n) {
            if (bits != 0) return w * 64 + static_cast<std::size_t>(detail::count_trailing_zeros(bits));
            if (++w == words) w = 0;
            bits = map[w].load(std::memory_order_relaxed);
        }
        return nrings_;
    }

    channel_options opts_;
    std::unique_ptr<ring[]> rings_;
    ring_map attended_;
    ring_map abandoned_;
    std::size_t nrings_;
    std::size_t max_producers_;
    // producers that have not finished yet, including the ones not created yet
    alignas(detail::cache_line) std::atomic<std::size_t> unfinished_;
    // producer handles handed out so far (cold: only touched by get_producer / close)
    alignas(detail::cache_line) std::atomic<std::size_t> created_;

#if defined(MPMC_CHANNEL_STATS)
    void add_stats(const detail::stat_array& local) noexcept {
        for (std::size_t i = 0; i < static_cast<std::size_t>(detail::stat::count_); ++i) {
            if (local[i]) totals_[i].fetch_add(local[i], std::memory_order_relaxed);
        }
    }
    std::atomic<std::uint64_t> totals_[static_cast<std::size_t>(detail::stat::count_)] = {};
#endif
};

//==============================================================================
template <typename T>
class alignas(detail::cache_line) channel<T>::producer {
public:
    producer() noexcept = default;
    ~producer() noexcept { reset(); }

    producer(const producer&) = delete;
    producer& operator=(const producer&) = delete;
    producer(producer&& o) noexcept { steal_state(o); }
    producer& operator=(producer&& o) noexcept {
        if (this != &o) { reset(); steal_state(o); }
        return *this;
    }

    explicit operator bool() const noexcept { return owner_ != nullptr; }

    // Blocks (spins, then yields) while every ring is full or taken.
    template <typename U>
    void push(U&& value) {
        if (MPMC_CHANNEL_UNLIKELY(pos_ == limit_)) {
            push_slow(std::forward<U>(value), true);
            return;
        }
        ::new (static_cast<void*>(data_ + (pos_ & mask_))) T(std::forward<U>(value));
        head_->store(++pos_, std::memory_order_release);
    }

    // Returns false, without waiting, when no ring can take the item right now.
    template <typename U>
    bool try_push(U&& value) {
        if (MPMC_CHANNEL_UNLIKELY(pos_ == limit_)) return push_slow(std::forward<U>(value), false);
        ::new (static_cast<void*>(data_ + (pos_ & mask_))) T(std::forward<U>(value));
        head_->store(++pos_, std::memory_order_release);
        return true;
    }

    // Gives the current ring back so other producers can use it.  Call it
    // before going quiet for a while; the next push simply takes a ring again.
    void release() noexcept {
        if (ring_) drop_ring();
    }

    // For a thread that both pushes and pops: make this producer push into
    // the ring `mine` pops from (while it holds fewer than home_high_water
    // items) instead of taking the rings in turn.
    //
    // Without it, threads that re-push what they pop feed each other's rings
    // in a fixed pattern, and the items end up piled in the ring of the
    // slowest thread while the others starve.  With it every thread keeps its
    // own supply, as in a work-stealing scheduler, and the other threads only
    // come for it when they run dry.
    //
    // `mine` must be used by the same thread as this producer and must stay
    // at the same address; pass nullptr to undo.
    void pair_with(const consumer* mine) noexcept { partner_ = mine; }

private:
    friend class channel<T>;

    producer(channel* owner, std::size_t start) noexcept : owner_(owner), idx_(start) {}

    void steal_state(producer& o) noexcept {
        pos_ = o.pos_; limit_ = o.limit_; data_ = o.data_; mask_ = o.mask_; head_ = o.head_;
        ring_ = o.ring_; chunk_end_ = o.chunk_end_; owner_ = o.owner_; idx_ = o.idx_;
        partner_ = o.partner_;
        o.owner_ = nullptr; o.ring_ = nullptr; o.pos_ = o.limit_ = 0; o.partner_ = nullptr;
#if defined(MPMC_CHANNEL_STATS)
        for (std::size_t i = 0; i < static_cast<std::size_t>(detail::stat::count_); ++i) { stats_[i] = o.stats_[i]; o.stats_[i] = 0; }
#endif
    }

    void reset() noexcept {
        if (!owner_) return;
        if (ring_) drop_ring();
#if defined(MPMC_CHANNEL_STATS)
        MPMC_CHANNEL_STAT(producers);
        owner_->add_stats(stats_);
#endif
        owner_->unfinished_.fetch_sub(1, std::memory_order_release);
        owner_ = nullptr;
    }

    void drop_ring() noexcept {
        ring_->producer_owned.store(false, std::memory_order_release);
        ring_ = nullptr;
        pos_ = limit_ = 0;
        if (++idx_ == owner_->nrings_) idx_ = 0;
    }

    void adopt(ring& r, std::size_t pos, std::size_t tail) noexcept {
        ring_ = &r;
        data_ = r.data;
        mask_ = r.mask;
        head_ = &r.head;
        pos_ = pos;
        chunk_end_ = pos + owner_->opts_.chunk;
        // the last item of a chunk goes through the slow path (see push_slow)
        limit_ = std::min(chunk_end_ - 1, tail + r.capacity);
    }

    // Slow path of push: the hot path stopped because the ring looks full, we
    // hold no ring, or this is the last item of the chunk.  That last item is
    // pushed from here so that the ring can be given back right away: a
    // producer never sits on a ring between two chunks.
    template <typename U>
    MPMC_CHANNEL_NOINLINE bool push_slow(U&& value, bool blocking) {
        if (!acquire(blocking)) return false;
        ::new (static_cast<void*>(data_ + (pos_ & mask_))) T(std::forward<U>(value));
        head_->store(++pos_, std::memory_order_release);
        if (pos_ == chunk_end_) {
            MPMC_CHANNEL_STAT(chunk_ends);
            drop_ring();
        }
        return true;
    }

    // Make sure we hold a ring with room for one more item at pos_ (and with
    // pos_ inside the chunk).
    bool acquire(bool blocking) noexcept {
        channel& ch = *owner_;
        MPMC_CHANNEL_STAT(acquires);
        if (ring_) {
            std::size_t tail = ring_->cached_tail;
            if (pos_ - tail >= ring_->capacity) {
                // out of room: the consumer may have moved on
                tail = ring_->tail.load(std::memory_order_acquire);
                ring_->cached_tail = tail;
#if defined(MPMC_CHANNEL_STATS)
                if (pos_ - tail < ring_->capacity) MPMC_CHANNEL_STAT(room_refreshes);
#endif
            }
            if (pos_ - tail < ring_->capacity) {
                limit_ = std::min(chunk_end_ - 1, tail + ring_->capacity);
                return true;
            }
            MPMC_CHANNEL_STAT(ring_full);
            drop_ring();
        }
        if (partner_ && partner_->owner_ == owner_) {
            // our own thread pops from this ring: keep our supply at home
            ring& r = *partner_->ring_;
            if (!r.producer_owned.load(std::memory_order_relaxed) &&
                !r.producer_owned.exchange(true, std::memory_order_acquire)) {
                const std::size_t pos = r.head.load(std::memory_order_relaxed);
                const std::size_t tail = r.tail.load(std::memory_order_acquire);
                r.cached_tail = tail;
                if (pos - tail < ch.opts_.home_high_water && pos - tail < r.capacity) {
                    adopt(r, pos, tail);
                    MPMC_CHANNEL_STAT(ring_switches);
                    MPMC_CHANNEL_STAT(home_rings_taken);
                    return true;
                }
                r.producer_owned.store(false, std::memory_order_release);
            }
        }
        for (std::uint32_t round = 1;; ++round) {
            // First choice: a ring somebody is popping from.  Rings without a
            // consumer are only used when there is nothing else, because what
            // lands there has to be stolen back out.
            for (std::size_t n = 0, total = 2 * ch.nrings_; n < total; ++n) {
                ring& r = ch.rings_[idx_];
                const bool wanted = n >= ch.nrings_ || r.consumer_owned.load(std::memory_order_relaxed);
#if defined(MPMC_CHANNEL_STATS)
                if (wanted && r.producer_owned.load(std::memory_order_relaxed)) MPMC_CHANNEL_STAT(busy_skips);
#endif
                if (wanted && !r.producer_owned.load(std::memory_order_relaxed) &&
                    !r.producer_owned.exchange(true, std::memory_order_acquire)) {
                    const std::size_t pos = r.head.load(std::memory_order_relaxed);
                    std::size_t tail = r.cached_tail;
                    if (pos - tail >= r.capacity) {
                        tail = r.tail.load(std::memory_order_acquire);
                        r.cached_tail = tail;
                    }
                    if (pos - tail < r.capacity) {
                        adopt(r, pos, tail);
                        MPMC_CHANNEL_STAT(ring_switches);
                        if (!r.consumer_owned.load(std::memory_order_seq_cst)) {
                            // what we push here has to be fetched by somebody: say so
                            channel::mark(ch.abandoned_, ch.index_of(r));
                            MPMC_CHANNEL_STAT(orphan_rings_taken);
                        }
                        return true;
                    }
                    MPMC_CHANNEL_STAT(full_skips);
                    r.producer_owned.store(false, std::memory_order_release);
                }
                if (++idx_ == ch.nrings_) idx_ = 0;
            }
            MPMC_CHANNEL_STAT(producer_waits);
            if (!blocking) return false;
            detail::idle_wait(round);
        }
    }

    // hot
    std::size_t pos_{0};
    std::size_t limit_{0};
    T* data_{nullptr};
    std::size_t mask_{0};
    std::atomic<std::size_t>* head_{nullptr};
    // cold
    ring* ring_{nullptr};
    std::size_t chunk_end_{0};
    channel* owner_{nullptr};
    std::size_t idx_{0};
    const consumer* partner_{nullptr};
#if defined(MPMC_CHANNEL_STATS)
    detail::stat_array stats_ = {};
#endif
};

//==============================================================================
template <typename T>
class alignas(detail::cache_line) channel<T>::consumer {
public:
    consumer() noexcept = default;
    ~consumer() noexcept { reset(); }

    consumer(const consumer&) = delete;
    consumer& operator=(const consumer&) = delete;
    consumer(consumer&& o) noexcept { steal_state(o); }
    consumer& operator=(consumer&& o) noexcept {
        if (this != &o) { reset(); steal_state(o); }
        return *this;
    }

    explicit operator bool() const noexcept { return owner_ != nullptr; }

    // Blocks (spins, then yields) until an item arrives.  Returns false only
    // when the channel is exhausted.
    bool pop(T& out) noexcept(std::is_nothrow_move_assignable<T>::value) {
        if (MPMC_CHANNEL_UNLIKELY(pos_ == limit_)) {
            detail::never_stop never;
            if (!refill(true, never)) return false;
        }
        take(out);
        return true;
    }

    // Like pop(), but also gives up (returns false) once stop() returns true.
    template <typename Stop>
    bool pop_until(T& out, Stop&& stop) {
        if (MPMC_CHANNEL_UNLIKELY(pos_ == limit_) && !refill(true, stop)) return false;
        take(out);
        return true;
    }

    // Returns false, without waiting, when nothing is available right now.
    // (When it does find items it may still pause for a few hundred cycles to
    // let a batch build up; see channel_options::close_threshold.)
    bool try_pop(T& out) noexcept(std::is_nothrow_move_assignable<T>::value) {
        if (MPMC_CHANNEL_UNLIKELY(pos_ == limit_)) {
            detail::never_stop never;
            if (!refill(false, never)) return false;
        }
        take(out);
        return true;
    }

    // True once a pop found the channel closed and completely drained.
    bool exhausted() const noexcept { return exhausted_; }

    // Hands the items this handle has already stolen into its private stash,
    // and not popped yet, to sink(T&&).  Destroying the handle drops those
    // items, so a caller that must not lose anything pushes them back first.
    template <typename Sink>
    void drain_private(Sink&& sink) {
        if (publish_ != &sink_) return;
        for (; pos_ != limit_; ++pos_) {
            sink(std::move(stash_[pos_]));
            stash_[pos_].~T();
        }
    }

private:
    friend class channel<T>;

    consumer(channel* owner, std::size_t index) noexcept : ring_(&owner->rings_[index]), owner_(owner), victim_(index) {
        attach();
    }

    // Start working on ring_, whose consumer_owned flag we already hold.
    void attach() noexcept {
        ring& r = *ring_;
        data_ = r.data;
        mask_ = r.mask;
        publish_ = &r.tail;
        r.lock_steal();                             // thieves may be emptying the parked range
        pos_ = r.parked_pos.load(std::memory_order_relaxed);    // resume what the previous owner left
        limit_ = own_limit_ = r.parked_limit.load(std::memory_order_relaxed);
        r.parked_pos.store(limit_, std::memory_order_relaxed);
        r.unlock_steal();
        starved_ = 0;
        channel::mark(owner_->attended_, owner_->index_of(r));
    }

    // Give ring_ up.  The unread part of our reservation stays parked in the
    // ring: its next consumer resumes it, or thieves empty it.
    void detach() noexcept {
        ring& r = *ring_;
        const bool own_run = (publish_ == &r.tail);
        r.lock_steal();
        r.parked_pos.store(own_run ? pos_ : own_limit_, std::memory_order_relaxed);
        r.parked_limit.store(own_run ? limit_ : own_limit_, std::memory_order_relaxed);
        r.consumer_owned.store(false, std::memory_order_seq_cst);
        r.unlock_steal();
        channel::unmark(owner_->attended_, owner_->index_of(r));
        channel::mark(owner_->abandoned_, owner_->index_of(r));     // whatever it still holds is up for grabs
    }

    void take(T& out) noexcept(std::is_nothrow_move_assignable<T>::value) {
        T* slot = data_ + (pos_ & mask_);
        out = std::move(*slot);
        slot->~T();
        publish_->store(++pos_, std::memory_order_release);
    }

    void steal_state(consumer& o) noexcept {
        pos_ = o.pos_; limit_ = o.limit_; data_ = o.data_; mask_ = o.mask_;
        publish_ = (o.publish_ == &o.sink_) ? &sink_ : o.publish_;
        ring_ = o.ring_; own_limit_ = o.own_limit_; owner_ = o.owner_; stash_ = o.stash_;
        victim_ = o.victim_; idle_ = o.idle_; steal_at_ = o.steal_at_; steal_gap_ = o.steal_gap_;
        close_spins_ = o.close_spins_; starved_ = o.starved_; refills_ = o.refills_; patrol_ = o.patrol_;
        exhausted_ = o.exhausted_;
#if defined(MPMC_CHANNEL_STATS)
        for (std::size_t i = 0; i < static_cast<std::size_t>(detail::stat::count_); ++i) { stats_[i] = o.stats_[i]; o.stats_[i] = 0; }
#endif
        o.owner_ = nullptr; o.ring_ = nullptr; o.stash_ = nullptr; o.pos_ = o.limit_ = 0;
    }

    void reset() noexcept {
        if (!owner_) return;
        // stolen items that were never popped cannot be handed back
        // (drain_private() rescues them)
        if (publish_ == &sink_) {
            for (; pos_ != limit_; ++pos_) stash_[pos_].~T();
        }
        detach();
        if (stash_) ::operator delete(stash_, std::align_val_t{std::max(alignof(T), detail::cache_line)});
        stash_ = nullptr;
#if defined(MPMC_CHANNEL_STATS)
        MPMC_CHANNEL_STAT(consumers);
        owner_->add_stats(stats_);
#endif
        owner_ = nullptr;
    }

    // Reserve the next batch of our own ring.
    bool reserve_own() noexcept {
        ring& r = *ring_;
        const channel_options& o = owner_->opts_;
        std::size_t head = r.head.load(std::memory_order_acquire);
        std::size_t base = own_limit_;
        if (head == base) {
            if (r.supply.load(std::memory_order_relaxed) != 0) r.supply.store(0, std::memory_order_relaxed);
            return false;
        }

        if (head - base < o.close_threshold) {
            // Following the producer too closely: every refill would bounce
            // its cache line for a handful of items.  Pause to let a batch
            // build up - but only while pausing pays off.  A producer that is
            // streaming adds many items per pause; one that trickles does not,
            // and then waiting would only add latency.
            const std::size_t before = head;
            MPMC_CHANNEL_STAT_ADD(close_pauses, close_spins_);
            for (std::uint32_t i = 0; i < close_spins_; ++i) MPMC_CHANNEL_RELAX();
            head = r.head.load(std::memory_order_acquire);
            if (head - before < close_spins_) {
                close_spins_ = std::max<std::uint32_t>(close_spins_ / 2, 1);
            } else if (head - base < o.close_threshold) {
                close_spins_ = std::min<std::uint32_t>(close_spins_ * 2, o.close_spins);
            }
        }

        for (;;) {
            const std::size_t avail = head - base;
            const std::size_t n = std::min(avail, o.max_batch);
            std::size_t expected = base;
            if (r.claim.compare_exchange_strong(expected, base + n, std::memory_order_seq_cst)) {
                data_ = r.data;
                mask_ = r.mask;
                publish_ = &r.tail;
                pos_ = base;
                limit_ = own_limit_ = base + n;
                r.supply.store(avail - n, std::memory_order_relaxed);
                MPMC_CHANNEL_STAT(own_batches);
                MPMC_CHANNEL_STAT_ADD(own_items, n);
                return true;
            }
            MPMC_CHANNEL_STAT(thefts_noticed);
            // Thieves took [base, expected).  A thief holds the steal lock from
            // before its claim until it has moved its items out, so once the
            // lock is seen free those slots may go back to the producer.
            while (r.steal_lock.load(std::memory_order_seq_cst)) MPMC_CHANNEL_RELAX();
            base = own_limit_ = expected;
            r.tail.store(base, std::memory_order_release);
            pos_ = limit_ = base;                   // nothing reserved yet
            if (static_cast<std::ptrdiff_t>(head - base) <= 0) {
                head = r.head.load(std::memory_order_acquire);
                if (head == base) return false;
            }
        }
    }

    void use_stash(std::size_t n) noexcept {
        data_ = stash_;
        mask_ = ~std::size_t{0};
        publish_ = &sink_;
        pos_ = 0;
        limit_ = n;
    }

    static void move_out(ring& v, std::size_t from, std::size_t n, T* to) noexcept {
        for (std::size_t i = 0; i < n; ++i) {
            T* src = v.data + ((from + i) & v.mask);
            ::new (static_cast<void*>(to + i)) T(std::move(*src));
            src->~T();
        }
    }

    // Move items nobody is going to pop out of another ring into our private
    // stash: what its departed consumer left parked, or what is not reserved.
    bool steal_from(ring& v, std::size_t min_backlog, bool& contended) {
        const channel_options& o = owner_->opts_;
        const bool orphan = !v.consumer_owned.load(std::memory_order_acquire);
        const bool parked = orphan && v.parked_pos.load(std::memory_order_relaxed) != v.parked_limit.load(std::memory_order_relaxed);
        if (!parked) {
            const std::size_t claim = v.claim.load(std::memory_order_acquire);
            const std::size_t avail = v.head.load(std::memory_order_acquire) - claim;
            if (avail == 0 || avail < min_backlog) return false;
        }
        if (!stash_) {
            stash_ = static_cast<T*>(::operator new(o.steal_batch * sizeof(T),
                                                    std::align_val_t{std::max(alignof(T), detail::cache_line)}));
        }
        MPMC_CHANNEL_STAT(steal_tries);
        if (v.steal_lock.load(std::memory_order_relaxed) || v.steal_lock.exchange(true, std::memory_order_seq_cst)) {
            contended = true;
            MPMC_CHANNEL_STAT(steal_contended);
            return false;
        }

        if (!v.consumer_owned.load(std::memory_order_seq_cst)) {
            // No consumer (one that arrives now waits for the steal lock):
            // finish the reservation the last one left behind first.
            const std::size_t from = v.parked_pos.load(std::memory_order_relaxed);
            const std::size_t left = v.parked_limit.load(std::memory_order_relaxed) - from;
            if (left != 0) {
                const std::size_t n = std::min(left, o.steal_batch);
                move_out(v, from, n, stash_);
                v.parked_pos.store(from + n, std::memory_order_relaxed);
                v.tail.store(from + n, std::memory_order_release);
                v.unlock_steal();
                use_stash(n);
                MPMC_CHANNEL_STAT(steals);
                MPMC_CHANNEL_STAT(steals_orphan);
                MPMC_CHANNEL_STAT_ADD(stolen_items, n);
                return true;
            }
        }

        std::size_t claim = v.claim.load(std::memory_order_seq_cst);
        const std::size_t avail = v.head.load(std::memory_order_acquire) - claim;
        const std::size_t share = v.consumer_owned.load(std::memory_order_relaxed) ? (avail + 1) / 2 : avail;
        const std::size_t n = std::min(share, o.steal_batch);
        if (avail == 0 || avail < min_backlog ||
            !v.claim.compare_exchange_strong(claim, claim + n, std::memory_order_seq_cst)) {
            v.unlock_steal();
            if (avail != 0 && avail >= min_backlog) {                   // lost the race for the items
                contended = true;
                MPMC_CHANNEL_STAT(steal_contended);
            }
            return false;
        }
        move_out(v, claim, n, stash_);

        // Give the stolen slots back to the producer when everything below
        // them has been consumed.  The owner does this itself when it notices
        // the theft, but an owner that is stalled, or gone, never will.
        if (claim != v.stolen_hi) v.stolen_lo = claim;
        v.stolen_hi = claim + n;
        std::size_t tail = v.tail.load(std::memory_order_acquire);
        if (tail - v.stolen_lo <= v.stolen_hi - v.stolen_lo) {
            v.tail.compare_exchange_strong(tail, v.stolen_hi, std::memory_order_seq_cst);
            v.stolen_lo = v.stolen_hi;
        }
        v.unlock_steal();
        use_stash(n);
        MPMC_CHANNEL_STAT(steals);
        MPMC_CHANNEL_STAT_ADD(stolen_items, n);
#if defined(MPMC_CHANNEL_STATS)
        if (!v.consumer_owned.load(std::memory_order_relaxed)) MPMC_CHANNEL_STAT(steals_orphan);
#endif
        return true;
    }

    // Take a batch from a ring marked as abandoned, if there is one; rings
    // that turn out to be empty or attended are unmarked on the way.
    bool collect_abandoned() {
        channel& ch = *owner_;
        const std::size_t words = (ch.nrings_ + 63) / 64;
        for (std::size_t w = 0; w < words; ++w) {
            std::uint64_t bits = ch.abandoned_[w].load(std::memory_order_relaxed);
            for (; bits != 0; bits &= bits - 1) {
                const std::size_t i = w * 64 + static_cast<std::size_t>(detail::count_trailing_zeros(bits));
                ring& v = ch.rings_[i];
                if (v.consumer_owned.load(std::memory_order_seq_cst)) {
                    // it has a consumer (maybe us): not abandoned.  Re-check
                    // afterwards in case that consumer was just leaving.
                    channel::unmark(ch.abandoned_, i);
                    if (!v.consumer_owned.load(std::memory_order_seq_cst)) channel::mark(ch.abandoned_, i);
                    continue;
                }
                bool contended = false;
                if (steal_from(v, 1, contended)) return true;
                if (contended || v.producer_owned.load(std::memory_order_seq_cst)) continue;   // more may come
                // Empty, and no producer on it.  Unmark, then look again: a
                // producer that arrived meanwhile either still holds the
                // ring or has left items behind.
                channel::unmark(ch.abandoned_, i);
                const bool parked = v.parked_pos.load(std::memory_order_seq_cst) != v.parked_limit.load(std::memory_order_seq_cst);
                const bool items = v.head.load(std::memory_order_seq_cst) != v.claim.load(std::memory_order_seq_cst);
                if (!v.consumer_owned.load(std::memory_order_seq_cst) &&
                    (parked || items || v.producer_owned.load(std::memory_order_seq_cst))) {
                    channel::mark(ch.abandoned_, i);
                }
            }
        }
        return false;
    }

    // Swap our (empty) ring for one that has no consumer.
    bool migrate() noexcept {
        channel& ch = *owner_;
        for (std::size_t i = 0; i < ch.nrings_; ++i) {
            ring& r = ch.rings_[i];
            if (&r == ring_ || r.producer_owned.load(std::memory_order_relaxed) ||
                r.consumer_owned.load(std::memory_order_relaxed) ||
                r.consumer_owned.exchange(true, std::memory_order_acquire)) {
                continue;
            }
            detach();                               // called with the current batch used up: nothing unread
            ring_ = &r;
            attach();
            return true;
        }
        return false;
    }

    // Slow path: the current batch is used up.
    template <typename Stop>
    MPMC_CHANNEL_NOINLINE bool refill(bool blocking, Stop& stop) {
        channel& ch = *owner_;
        MPMC_CHANNEL_STAT(refills);

        // Items sitting in a ring nobody pops from must not wait until we run
        // dry, which may be never: fetch a batch of them before anything else.
        if (ch.nrings_ > 1 && collect_abandoned()) {
            MPMC_CHANNEL_STAT(steals_patrol);
            return true;
        }

        // Every few refills, compare our supply with another consumer's and
        // take a batch if that one is clearly better off (balance_threshold).
        if (ch.opts_.balance_threshold != 0 && (++refills_ & 3) == 0 && ch.nrings_ > 1) {
            const std::size_t next = ch.next_marked(ch.attended_, patrol_);
            if (next != ch.nrings_) {
                patrol_ = next;
                ring& v = ch.rings_[next];
                if (&v != ring_) {
                    const std::size_t mine = ring_->head.load(std::memory_order_relaxed) - own_limit_;
                    bool contended = false;
                    if (v.supply.load(std::memory_order_relaxed) > mine + ch.opts_.balance_threshold &&
                        steal_from(v, mine + ch.opts_.balance_threshold, contended)) {
                        MPMC_CHANNEL_STAT(steals_balance);
                        return true;
                    }
                }
            }
        }

        for (;;) {
            if (reserve_own()) {
                idle_ = 0;
                starved_ = 0;
                steal_at_ = steal_gap_;
                return true;
            }
            ++idle_;
            MPMC_CHANNEL_STAT(own_empty);

            // Our ring stays empty while a producer sits on it without
            // pushing (it went quiet in the middle of a chunk, so no other
            // producer can use the ring): move to a ring nobody pops from.
            if (++starved_ >= 64 && ch.nrings_ > 1) {
                starved_ = 0;
                if (ring_->producer_owned.load(std::memory_order_relaxed) && migrate()) {
                    MPMC_CHANNEL_STAT(migrations);
                    if (pos_ != limit_) {
                        idle_ = 0;
                        return true;
                    }
                }
            }

            if (idle_ >= steal_at_ && ch.nrings_ > 1) {
                // One attempt on a ring that has a consumer; empty rings
                // without one are skipped on the way, they cost nothing.
                bool stolen = false;
                for (std::size_t n = 0; n < ch.nrings_ && !stolen; ++n) {
                    if (++victim_ >= ch.nrings_) victim_ = 0;
                    ring& v = ch.rings_[victim_];
                    if (&v == ring_) continue;
                    bool contended = false;
                    const bool orphan = !v.consumer_owned.load(std::memory_order_relaxed);
                    stolen = steal_from(v, orphan ? 1 : ch.opts_.steal_min, contended);
                    if (!orphan) break;
                }
                if (stolen) {
                    --victim_;                      // come back to the same ring first
                    idle_ = 0;
                    steal_at_ = steal_gap_ = 1;
                    return true;
                }
                steal_gap_ = std::min<std::uint32_t>(steal_gap_ * 2, 64);
                steal_at_ = idle_ + steal_gap_;
            }

            if (ch.unfinished_.load(std::memory_order_acquire) == 0) {
                // Every producer is gone, so every head is final: take
                // whatever is still unreserved anywhere, then report the end.
                if (reserve_own()) return true;
                for (bool contended = true; contended;) {
                    contended = false;
                    for (std::size_t i = 0; i < ch.nrings_; ++i) {
                        ring& v = ch.rings_[i];
                        if (&v != ring_ && steal_from(v, 1, contended)) return true;
                    }
                    if (contended) MPMC_CHANNEL_RELAX();
                }
                exhausted_ = true;
                return false;
            }

            if (!blocking || stop()) return false;
            MPMC_CHANNEL_STAT(idle_rounds);
#if defined(MPMC_CHANNEL_STATS)
            if (idle_ > 64) MPMC_CHANNEL_STAT(yields);
#endif
            detail::idle_wait(idle_);
        }
    }

    // hot
    std::size_t pos_{0};
    std::size_t limit_{0};
    T* data_{nullptr};
    std::size_t mask_{0};
    std::atomic<std::size_t>* publish_{nullptr};
    // cold
    ring* ring_{nullptr};
    std::size_t own_limit_{0};                      // end of our last reservation in our own ring
    channel* owner_{nullptr};
    T* stash_{nullptr};
    std::size_t victim_{0};
    std::uint32_t idle_{0};
    std::uint32_t steal_at_{4};
    std::uint32_t steal_gap_{4};
    std::uint32_t close_spins_{1};
    std::uint32_t starved_{0};                      // refills in a row that found our own ring empty
    std::uint32_t refills_{0};
    std::size_t patrol_{0};                         // next ring to check for abandoned items
    bool exhausted_{false};
    std::atomic<std::size_t> sink_{0};              // tail stand-in while popping from the stash
#if defined(MPMC_CHANNEL_STATS)
    detail::stat_array stats_ = {};
#endif
};

} // namespace mpmc
