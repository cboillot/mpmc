// mpmc_candidate2.cxx
//
// Plugs mpmc::channel (mpmc_channel.hpp) into the pacing suite's MpmcChannel
// interface.  All the logic lives in the header; this file only forwards.
#include "mpmc_candidate.hpp"
#include "mpmc_channel.hpp"
#include <cstdint>

template <typename T>
struct MpmcChannel<T>::ChannelImpl {
    mpmc::channel<T> channel_;
    ChannelImpl(size_t max_p, size_t max_c) : channel_(max_p, max_c) {}
};

template <typename T>
struct MpmcChannel<T>::ProducerImpl {
    typename mpmc::channel<T>::producer handle_;
    explicit ProducerImpl(mpmc::channel<T>& c) : handle_(c.get_producer()) {}
};

template <typename T>
struct MpmcChannel<T>::ConsumerImpl {
    typename mpmc::channel<T>::consumer handle_;
    explicit ConsumerImpl(mpmc::channel<T>& c) : handle_(c.get_consumer()) {}
};

template <typename T>
MpmcChannel<T>::MpmcChannel(size_t max_producers, size_t max_consumers) {
    impl_ = new ChannelImpl(max_producers, max_consumers);
}

template <typename T>
MpmcChannel<T>::~MpmcChannel() noexcept {
    delete impl_;
}

// Producer
template <typename T>
MpmcChannel<T>::ProducerHandle::ProducerHandle() noexcept : impl_(nullptr) {}

template <typename T>
MpmcChannel<T>::ProducerHandle::~ProducerHandle() noexcept {
    delete impl_;
}

template <typename T>
MpmcChannel<T>::ProducerHandle::ProducerHandle(ProducerHandle&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

template <typename T>
typename MpmcChannel<T>::ProducerHandle& MpmcChannel<T>::ProducerHandle::operator=(ProducerHandle&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

template <typename T>
void MpmcChannel<T>::ProducerHandle::push(const T& item) noexcept {
    impl_->handle_.push(item);
}

// Consumer
template <typename T>
MpmcChannel<T>::ConsumerHandle::ConsumerHandle() noexcept : impl_(nullptr) {}

template <typename T>
MpmcChannel<T>::ConsumerHandle::~ConsumerHandle() noexcept {
    delete impl_;
}

template <typename T>
MpmcChannel<T>::ConsumerHandle::ConsumerHandle(ConsumerHandle&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

template <typename T>
typename MpmcChannel<T>::ConsumerHandle& MpmcChannel<T>::ConsumerHandle::operator=(ConsumerHandle&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

template <typename T>
T MpmcChannel<T>::ConsumerHandle::pop() noexcept {
    T item{};
    impl_->handle_.pop(item);
    return item;
}

template <typename T>
bool MpmcChannel<T>::ConsumerHandle::exhausted() const noexcept {
    return impl_->handle_.exhausted();
}

// Factories
template <typename T>
typename MpmcChannel<T>::ProducerHandle MpmcChannel<T>::get_producer_handle() {
    ProducerHandle handle;
    handle.impl_ = new ProducerImpl(impl_->channel_);
    return handle;
}

template <typename T>
typename MpmcChannel<T>::ConsumerHandle MpmcChannel<T>::get_consumer_handle() {
    ConsumerHandle handle;
    handle.impl_ = new ConsumerImpl(impl_->channel_);
    return handle;
}

template class MpmcChannel<uint8_t>;
template class MpmcChannel<uint32_t>;
template class MpmcChannel<uint64_t>;
