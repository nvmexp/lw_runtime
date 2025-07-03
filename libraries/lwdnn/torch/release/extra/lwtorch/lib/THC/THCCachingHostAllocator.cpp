#include "THCCachingHostAllocator.h"

#include <lwda_runtime_api.h>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <stdint.h>
#include <unordered_map>
#include <utility>


namespace {

typedef std::shared_ptr<THCStream> THCStreamPtr;

struct BlockSize
{
  size_t  size; // allocation size
  void*   ptr;  // host memory pointer

  BlockSize(size_t size, void* ptr=NULL) : size(size), ptr(ptr) {}
};

struct Block : public BlockSize
{
  bool  allocated;    // true if the block is lwrrently allocated
  int   event_count;  // number of outstanding lwca events
  std::set<THCStreamPtr> streams;

  Block(size_t size, void* ptr, bool allocated) :
      BlockSize(size, ptr), allocated(allocated), event_count(0), streams() {}
};

static bool BlockComparator(const BlockSize& a, const BlockSize& b)
{
  // sort by size, break ties with pointer
  if (a.size != b.size) {
    return a.size < b.size;
  }
  return (uintptr_t)a.ptr < (uintptr_t)b.ptr;
}

struct HostAllocator
{
  typedef bool (*Comparison)(const BlockSize&, const BlockSize&);

  // lock around all operations
  std::mutex mutex;

  // blocks by pointer
  std::unordered_map<void*, Block> blocks;

  // pointers that are ready to be allocated (event_count=0)
  std::set<BlockSize, Comparison> available;

  // outstanding lwca events
  std::deque<std::pair<lwdaEvent_t, void*>> lwda_events;

  HostAllocator() : available(BlockComparator) {}

  lwdaError_t malloc(void** ptr, size_t size)
  {
    std::lock_guard<std::mutex> lock(mutex);

    // process outstanding lwca events which may have oclwrred
    lwdaError_t err = processEvents();
    if (err != lwdaSuccess) {
      return err;
    }

    // search for the smallest block which can hold this allocation
    BlockSize search_key(size);
    auto it = available.lower_bound(search_key);
    if (it != available.end()) {
      Block& block = blocks.at(it->ptr);
      THAssert(!block.allocated && block.event_count == 0);
      block.allocated = true;
      *ptr = block.ptr;
      available.erase(it);
      return lwdaSuccess;
    }

    // note that lwdaHostAlloc may not touch pointer if size is 0
    *ptr = 0;

    // allocate a new block if no cached allocation is found
    err = lwdaHostAlloc(ptr, size, lwdaHostAllocDefault);
    if (err != lwdaSuccess) {
      return err;
    }

    blocks.insert({*ptr, Block(size, *ptr, true)});
    return lwdaSuccess;
  }

  lwdaError_t free(void* ptr)
  {
    std::lock_guard<std::mutex> lock(mutex);

    if (!ptr) {
      return lwdaSuccess;
    }

    // process outstanding lwca events which may have oclwrred
    lwdaError_t err = processEvents();
    if (err != lwdaSuccess) {
      return err;
    }

    auto it = blocks.find(ptr);
    THAssert(it != blocks.end());

    Block& block = it->second;
    THAssert(block.allocated);

    // free (on valid memory) shouldn't fail, so mark unallocated before
    // we process the streams.
    block.allocated = false;

    // insert LWCA events for each stream on which this block was used. This
    err = insertEvents(block);
    if (err != lwdaSuccess) {
      return err;
    }

    if (block.event_count == 0) {
      // the block can be re-used if there are no outstanding lwca events
      available.insert(block);
    }
    return lwdaSuccess;
  }

  lwdaError_t recordEvent(void* ptr, THCStream *stream)
  {
    std::lock_guard<std::mutex> lock(mutex);
    lwdaError_t err;

    auto it = blocks.find(ptr);
    if (it == blocks.end()) {
      // ignore events for untracked pointers
      return lwdaSuccess;
    }

    Block& block = it->second;
    THAssert(block.allocated);

    THCStreamPtr stream_ptr(stream, &THCStream_free);
    THCStream_retain(stream);

    block.streams.insert(std::move(stream_ptr));
    return lwdaSuccess;
  }

  lwdaError_t processEvents()
  {
    // Process outstanding lwdaEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    while (!lwda_events.empty()) {
      auto& e = lwda_events.front();
      lwdaEvent_t event = e.first;

      lwdaError_t err = lwdaEventQuery(event);
      if (err == lwdaErrorNotReady) {
        break;
      } else if (err != lwdaSuccess) {
        return err;
      }
      err = lwdaEventDestroy(event);
      if (err != lwdaSuccess) {
        return err;
      }

      Block& block = blocks.at(e.second);
      block.event_count--;
      if (block.event_count == 0 && !block.allocated) {
        available.insert(block);
      }
      lwda_events.pop_front();
    }
    return lwdaSuccess;
  }

  void emptyCache()
  {
    std::lock_guard<std::mutex> lock(mutex);

    // remove events for freed blocks
    for (auto it = lwda_events.begin(); it != lwda_events.end(); ++it) {
      lwdaEvent_t event = it->first;
      Block& block = blocks.at(it->second);
      if (!block.allocated) {
        THLwdaCheckWarn(lwdaEventDestroy(event));
        block.event_count--;
      }
    }

    // all lwda_events have been processed
    lwda_events.clear();

    // clear list of available blocks
    available.clear();

    // free and erase non-allocated blocks
    for (auto it = blocks.begin(); it != blocks.end();) {
      Block& block = it->second;
      if (!block.allocated) {
        THLwdaCheckWarn(lwdaFreeHost(block.ptr));
        it = blocks.erase(it);
      } else {
        ++it;
      }
    }
  }

  lwdaError_t insertEvents(Block& block)
  {
    lwdaError_t err;

    int prev_device;
    err = lwdaGetDevice(&prev_device);
    if (err != lwdaSuccess) return err;

    std::set<THCStreamPtr> streams(std::move(block.streams));
    for (auto it = streams.begin(); it != streams.end(); ++it) {
      auto& stream = *it;

      err = lwdaSetDevice(stream->device);
      if (err != lwdaSuccess) break;

      lwdaEvent_t event;
      err = lwdaEventCreateWithFlags(&event, lwdaEventDisableTiming);
      if (err != lwdaSuccess) break;

      err = lwdaEventRecord(event, stream->stream);
      if (err != lwdaSuccess) break;

      block.event_count++;
      lwda_events.emplace_back(event, block.ptr);
    }

    lwdaSetDevice(prev_device);
    return err;
  }
};

}  // namespace

static HostAllocator allocator;

static void* THCCachingHostAllocator_malloc(void* ctx, ptrdiff_t size)
{
  THAssert(size >= 0);
  void *ptr;
  THLwdaCheck(allocator.malloc(&ptr, size));
  return ptr;
}

static void THCCachingHostAllocator_free(void* ctx, void* ptr)
{
  allocator.free(ptr);
}

lwdaError_t THCCachingHostAllocator_recordEvent(void *ptr, THCStream *stream)
{
  return allocator.recordEvent(ptr, stream);
}

void THCCachingHostAllocator_emptyCache()
{
  allocator.emptyCache();
}

THAllocator THCCachingHostAllocator = {
  &THCCachingHostAllocator_malloc,
  NULL,
  &THCCachingHostAllocator_free,
};
