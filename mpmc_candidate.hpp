// mpmc_candidate.hpp
#pragma once

#include <cstddef>

template <typename T>
class MpmcChannel {
public:
    struct ChannelImpl;
    struct ProducerImpl;
    struct ConsumerImpl;

    class ProducerHandle {
    public:
        ProducerHandle() noexcept;
        ~ProducerHandle() noexcept;

        ProducerHandle(const ProducerHandle&) = delete;
        ProducerHandle& operator=(const ProducerHandle&) = delete;
        ProducerHandle(ProducerHandle&& other) noexcept;
        ProducerHandle& operator=(ProducerHandle&& other) noexcept;

        void push(const T& item) noexcept;

    private:
        ProducerImpl* impl_;
        friend class MpmcChannel<T>;
    };

    class ConsumerHandle {
    public:
        ConsumerHandle() noexcept;
        ~ConsumerHandle() noexcept;

        ConsumerHandle(const ConsumerHandle&) = delete;
        ConsumerHandle& operator=(const ConsumerHandle&) = delete;
        ConsumerHandle(ConsumerHandle&& other) noexcept;
        ConsumerHandle& operator=(ConsumerHandle&& other) noexcept;

        T&& pop() noexcept;
        bool exhausted() const noexcept;

    private:
        ConsumerImpl* impl_;
        friend class MpmcChannel<T>;
    };

    MpmcChannel(size_t max_producers, size_t max_consumers);
    ~MpmcChannel() noexcept;

    MpmcChannel(const MpmcChannel&) = delete;
    MpmcChannel& operator=(const MpmcChannel&) = delete;

    ProducerHandle get_producer_handle();
    ConsumerHandle get_consumer_handle();

private:
    ChannelImpl* impl_;
};