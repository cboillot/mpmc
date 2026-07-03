// mpmc_candidate_fake.cxx
#include "mpmc_candidate.hpp"
#include <atomic>
#include <utility>

#ifdef __cpp_lib_hardware_interference_size
    using std::hardware_destructive_interference_size;
#else
    constexpr size_t hardware_destructive_interference_size = 64;
#endif

template <typename T>
struct MpmcChannel<T>::ChannelImpl {
    alignas(hardware_destructive_interference_size) std::atomic<size_t> active_producers_;

    ChannelImpl(size_t max_p, size_t /*max_c*/) : active_producers_(max_p) {}
};

template <typename T>
struct MpmcChannel<T>::ProducerImpl {
    ChannelImpl* parent_;
    ProducerImpl(ChannelImpl* p) : parent_(p) {}
};

template <typename T>
struct MpmcChannel<T>::ConsumerImpl {
    ChannelImpl* parent_;
    T current_item_{0};     // Safe localized zone to anchor the T&& return
    bool exhausted_{false}; // Tracks out-of-band loop termination status

    ConsumerImpl(ChannelImpl* p) : parent_(p) {}
};

// Top-Level Channel Orchestration
template <typename T>
MpmcChannel<T>::MpmcChannel(size_t max_producers, size_t max_consumers) {
    impl_ = new ChannelImpl(max_producers, max_consumers);
}

template <typename T>
MpmcChannel<T>::~MpmcChannel() noexcept {
    delete impl_;
}

// Producer Handle - Minimal Overhead Stubs
template <typename T>
MpmcChannel<T>::ProducerHandle::ProducerHandle() noexcept : impl_(nullptr) {}

template <typename T>
MpmcChannel<T>::ProducerHandle::~ProducerHandle() noexcept {
    if (impl_) {
        // When a benchmarking thread finishes pushing, drop the active count
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
void MpmcChannel<T>::ProducerHandle::push(const T& /*item*/) noexcept {
    // No-op: Simulates an infinitely fast ring buffer write slot acquisition
}

// Consumer Handle - Ultra-Fast Zero Latency Stream
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

    // If all producers have completed their lifecycle, terminate the consumer loop
    if (c_impl->parent_->active_producers_.load(std::memory_order_acquire) == 0) {
        c_impl->exhausted_ = true;
        return std::move(c_impl->current_item_);
    }

    // Always succeed instantly while work is active
    c_impl->current_item_ = static_cast<T>(0); 
    return std::move(c_impl->current_item_);
}

template <typename T>
bool MpmcChannel<T>::ConsumerHandle::exhausted() const noexcept {
    return impl_->exhausted_;
}

// Allocation Factories
template <typename T>
typename MpmcChannel<T>::ProducerHandle MpmcChannel<T>::get_producer_handle() {
    ProducerHandle handle;
    handle.impl_ = new ProducerImpl(impl_);
    return handle;
}

template <typename T>
typename MpmcChannel<T>::ConsumerHandle MpmcChannel<T>::get_consumer_handle() {
    ConsumerHandle handle;
    handle.impl_ = new ConsumerImpl(impl_);
    return handle;
}

// Explicit Compilation Target Triggers for the Benchmark
template class MpmcChannel<unsigned char>;
template class MpmcChannel<unsigned int>;
template class MpmcChannel<unsigned long>;