// mpmc_candidate1.cxx
#include "mpmc_candidate.hpp"
#include <vector>
#include <atomic>
#include <memory>
#include <new>
#include <utility>

#if defined(__x86_64__) || defined(__i386__)
    #include <immintrin.h>
    #define CPU_PAUSE() _mm_pause()
#else
    #define CPU_PAUSE() 
#endif

constexpr size_t BUFFER_CAPACITY = 4096;
constexpr size_t CONTENTION_SAFETY_MARGIN = 32;
constexpr size_t CHUNK_SIZE_INTERNAL = 30000;
constexpr size_t CACHE_LINE = 64;

template <typename T>
struct SingleChannel {
    size_t capacity_;
    size_t mask_;
    alignas(CACHE_LINE) std::vector<T> buffer_;
    alignas(CACHE_LINE) std::atomic<size_t> head_{0};
    alignas(CACHE_LINE) std::atomic<size_t> cached_tail_{0};
    alignas(CACHE_LINE) std::atomic<size_t> tail_{0};
    alignas(CACHE_LINE) std::atomic<size_t> cached_head_{0};
    alignas(CACHE_LINE) std::atomic<bool> producer_owned{false};

    SingleChannel(size_t cap = BUFFER_CAPACITY) 
        : capacity_(cap), mask_(cap - 1), buffer_(cap) {}
};

template <typename T>
struct MpmcChannel<T>::ChannelImpl {
    size_t max_producers_;
    size_t max_consumers_;
    std::vector<std::unique_ptr<SingleChannel<T>>> channels_;
    alignas(CACHE_LINE) std::atomic<size_t> active_producers_;
    alignas(CACHE_LINE) std::atomic<size_t> consumer_assignment_counter_{0};

    ChannelImpl(size_t max_p, size_t max_c)
        : max_producers_(max_p), max_consumers_(max_c), active_producers_(max_p) {
        for (size_t i = 0; i < max_c; ++i) {
            channels_.push_back(std::make_unique<SingleChannel<T>>());
        }
    }
};

template <typename T>
struct MpmcChannel<T>::ProducerImpl {
    ChannelImpl* parent_;
    size_t current_channel_idx_{0};
    size_t items_pushed_in_chunk_{0};
    bool holds_lock_{false};

    ProducerImpl(ChannelImpl* p) : parent_(p) {}
};

template <typename T>
struct MpmcChannel<T>::ConsumerImpl {
    ChannelImpl* parent_;
    size_t channel_idx_;
    T current_item_{};       // Dedicated safe zone to back the T&& return
    bool exhausted_{false};  // Out-of-band termination status flag

    ConsumerImpl(ChannelImpl* p, size_t idx) : parent_(p), channel_idx_(idx) {}
};

// Lifecycles
template <typename T>
MpmcChannel<T>::MpmcChannel(size_t max_producers, size_t max_consumers) {
    impl_ = new ChannelImpl(max_producers, max_consumers);
}

template <typename T>
MpmcChannel<T>::~MpmcChannel() noexcept {
    delete impl_;
}

// Producer Implementation Block
template <typename T>
MpmcChannel<T>::ProducerHandle::ProducerHandle() noexcept : impl_(nullptr) {}

template <typename T>
MpmcChannel<T>::ProducerHandle::~ProducerHandle() noexcept {
    if (impl_) {
        if (impl_->holds_lock_) {
            impl_->parent_->channels_[impl_->current_channel_idx_]->producer_owned.store(false, std::memory_order_release);
        }
        impl_->parent_->active_producers_.fetch_sub(1, std::memory_order_release);
        delete impl_;
    }
}

template <typename T>
MpmcChannel<T>::ProducerHandle::ProducerHandle(ProducerHandle&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

template <typename T>
typename MpmcChannel<T>::ProducerHandle& MpmcChannel<T>::ProducerHandle::operator=(ProducerHandle&& other) noexcept {
    if (this != &other) {
        if (impl_) delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

template <typename T>
void MpmcChannel<T>::ProducerHandle::push(const T& item) noexcept {
    auto* p_impl = impl_;
    auto* parent = p_impl->parent_;

    while (true) {
        if (!p_impl->holds_lock_) {
            while (true) {
                auto& chan = parent->channels_[p_impl->current_channel_idx_];
                if (!chan->producer_owned.load(std::memory_order_relaxed)) {
                    bool expected = false;
                    if (chan->producer_owned.compare_exchange_strong(expected, true, 
                                                                    std::memory_order_acquire, 
                                                                    std::memory_order_relaxed)) {
                        p_impl->holds_lock_ = true;
                        p_impl->items_pushed_in_chunk_ = 0;
                        break;
                    }
                }
                p_impl->current_channel_idx_ = (p_impl->current_channel_idx_ + 1) % parent->max_consumers_;
            }
        }

        auto& chan = parent->channels_[p_impl->current_channel_idx_];
        size_t current_head = chan->head_.load(std::memory_order_relaxed);
        size_t current_cached_tail = chan->cached_tail_.load(std::memory_order_relaxed);
        bool risk_of_contention = false;

        if ((chan->capacity_ - (current_head - current_cached_tail)) <= CONTENTION_SAFETY_MARGIN) {
            current_cached_tail = chan->tail_.load(std::memory_order_acquire);
            chan->cached_tail_.store(current_cached_tail, std::memory_order_relaxed);
            if ((chan->capacity_ - (current_head - current_cached_tail)) <= CONTENTION_SAFETY_MARGIN) {
                risk_of_contention = true;
                if ((current_head - current_cached_tail) >= chan->capacity_) {
                    chan->producer_owned.store(false, std::memory_order_release);
                    p_impl->holds_lock_ = false;
                    p_impl->current_channel_idx_ = (p_impl->current_channel_idx_ + 1) % parent->max_consumers_;
                    continue;
                }
            }
        }

        chan->buffer_[current_head & chan->mask_] = item;
        chan->head_.store(current_head + 1, std::memory_order_release);
        
        p_impl->items_pushed_in_chunk_++;
        if (p_impl->items_pushed_in_chunk_ >= CHUNK_SIZE_INTERNAL || risk_of_contention) {
            chan->producer_owned.store(false, std::memory_order_release);
            p_impl->holds_lock_ = false;
            p_impl->current_channel_idx_ = (p_impl->current_channel_idx_ + 1) % parent->max_consumers_;
        }
        return;
    }
}

// Consumer Implementation Block
template <typename T>
MpmcChannel<T>::ConsumerHandle::ConsumerHandle() noexcept : impl_(nullptr) {}

template <typename T>
MpmcChannel<T>::ConsumerHandle::~ConsumerHandle() noexcept {
    if (impl_) delete impl_;
}

template <typename T>
MpmcChannel<T>::ConsumerHandle::ConsumerHandle(ConsumerHandle&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

template <typename T>
typename MpmcChannel<T>::ConsumerHandle& MpmcChannel<T>::ConsumerHandle::operator=(ConsumerHandle&& other) noexcept {
    if (this != &other) {
        if (impl_) delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

template <typename T>
T&& MpmcChannel<T>::ConsumerHandle::pop() noexcept {
    auto* c_impl = impl_;
    auto* parent = c_impl->parent_;

    while (true) {
        auto& chan = parent->channels_[c_impl->channel_idx_];
        size_t current_tail = chan->tail_.load(std::memory_order_relaxed);
        size_t current_cached_head = chan->cached_head_.load(std::memory_order_relaxed);

        if ((current_cached_head - current_tail) <= CONTENTION_SAFETY_MARGIN / 2) {
            for (int i = 0; i < 3; ++i) CPU_PAUSE();
        }

        if (current_tail == current_cached_head) {
            current_cached_head = chan->head_.load(std::memory_order_acquire);
            chan->cached_head_.store(current_cached_head, std::memory_order_relaxed);
            
            if (current_tail == current_cached_head) {
                for (int i = 0; i < 5; ++i) CPU_PAUSE();

                if (parent->active_producers_.load(std::memory_order_acquire) == 0) {
                    current_cached_head = chan->head_.load(std::memory_order_acquire);
                    if (current_tail == current_cached_head) {
                        c_impl->exhausted_ = true;
                        return std::move(c_impl->current_item_);
                    }
                }
                continue; 
            }
        }

        // Secure copy to local handle storage before releasing the slot back to producers
        c_impl->current_item_ = chan->buffer_[current_tail & chan->mask_];
        chan->tail_.store(current_tail + 1, std::memory_order_release);
        return std::move(c_impl->current_item_);
    }
}

template <typename T>
bool MpmcChannel<T>::ConsumerHandle::exhausted() const noexcept {
    return impl_->exhausted_;
}

// Factories
template <typename T>
typename MpmcChannel<T>::ProducerHandle MpmcChannel<T>::get_producer_handle() {
    ProducerHandle handle;
    handle.impl_ = new ProducerImpl(impl_);
    return handle;
}

template <typename T>
typename MpmcChannel<T>::ConsumerHandle MpmcChannel<T>::get_consumer_handle() {
    size_t id = impl_->consumer_assignment_counter_.fetch_add(1, std::memory_order_relaxed);
    ConsumerHandle handle;
    handle.impl_ = new ConsumerImpl(impl_, id % impl_->max_consumers_);
    return handle;
}

// Explicit Compilation Generation Targets
template class MpmcChannel<uint8_t>;
template class MpmcChannel<uint32_t>;
template class MpmcChannel<uint64_t>;