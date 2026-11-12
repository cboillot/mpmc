# mpmc

Multi-producer / multi-consumer channel experiments and the pacing suite used to compare them.

## Build and run

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 6
./build/pacing_suite_mpmc_candidate2      # one binary per mpmc_candidate*.cxx
ctest --test-dir build                    # correctness tests of mpmc_channel.hpp
```

## Candidates

| File | What it is |
|---|---|
| `mpmc_candidate_fake.cxx` | reference that moves no data (upper bound of the suite) |
| `mpmc_candidate1.cxx` | first implementation: one SPSC ring per consumer, producers lock a ring per chunk |
| `mpmc_candidate2.cxx` | thin shim over `mpmc_channel.hpp` |

## mpmc_channel.hpp

Header-only, C++17, no dependency. Same layout as candidate 1 (one ring per consumer, a producer owns a ring
for a chunk of pushes) with:

- one branch on the hot path of `push` and `pop`; ownership, batching, back-off and stealing are in the slow paths
- `try_push`, `try_pop`, `pop_until(out, stop)` next to the blocking `push` / `pop`
- producers counted when their handle is created; extra producers or consumers are rejected (`std::length_error`);
  `close()` for when fewer producers than announced are created
- capacity, chunk and batch sizes as constructor options (`mpmc::channel_options`)
- any nothrow-movable type (no default constructor needed)
- work stealing: an idle consumer takes the items another ring's consumer has not reserved; rings without a
  consumer are drained by the others
- waiting spins with a pause, then yields
- consumers regularly collect items from rings nobody pops from, so nothing waits forever
- producers prefer rings that have a consumer; a consumer whose ring is held by a producer that went quiet
  moves to a free ring

```cpp
#include "mpmc_channel.hpp"

mpmc::channel<std::unique_ptr<Job>> channel(/*max_producers*/ 4, /*max_consumers*/ 8, /*capacity*/ 1 << 16);

// producer thread
auto producer = channel.get_producer();
producer.push(std::make_unique<Job>());
if (!producer.try_push(std::move(job))) { /* every ring is full or busy */ }

// consumer thread
auto consumer = channel.get_consumer();
std::unique_ptr<Job> job;
while (consumer.pop(job)) { /* ... */ }   // false once all producers are gone and everything was delivered
```

The rules on ordering, handle lifetime and stalled consumers are at the top of the header.

## mpmc_shared_queue.hpp

`mpmc::shared_queue<T>` is the same channel behind a conventional queue interface: one object, `push` /
`try_push` / `try_pop` / `pop` callable from any thread, no handles. It creates the per-thread handles on first
use, finds them again through a thread-local cache (one compare per call) and takes them back when a thread exits.

```cpp
#include "mpmc_shared_queue.hpp"

mpmc::shared_queue<int> queue(/*capacity*/ 1 << 14);   // optional 2nd argument: max_threads
queue.push(42);
int value;
if (queue.try_pop(value)) { /* ... */ }
```

It allocates `max_threads` rings of `capacity` slots (default: twice the hardware threads, at least 64).
`try_pop` looks at the calling thread's own ring first and only now and then at the others, so it can
return false while another ring still holds items; poll it again, or use `pop` / `pop_until`.
Other CMake projects can `add_subdirectory` / `FetchContent` this repository and link `mpmc_channel`; the
pacing suite is not built in that case.
