#include "THCCachingAllocator.h"

#include <lwda_runtime_api.h>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>

//
// Yet another caching allocator for LWCA device allocations.
//
// - Allocations are associated with a stream. Once freed, blocks can be
//   re-allocated on the same stream, but not on any other stream.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to lwdaMalloc.
// - If the lwdaMalloc fails, the allocator will free all cached blocks that
//   are not split and retry the allocation.
// - Large (>1MB) and small allocation requests are handled separately. Large
//   allocation requests can be filled by a lwdaMalloc call of the exact size.
//   Small requests will allocate and split a 1MB buffer, if necessary.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with streams, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple streams.
//
// The library provides a recordStream() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the block is not reused before each recorded stream completes
// work.
//


namespace {

typedef std::shared_ptr<THCStream> THCStreamPtr;
typedef std::set<THCStreamPtr> stream_set;

const size_t kRoundSmall = 512;     // round up small allocs to 512 bytes
const size_t kRoundLarge = 131072;  // round up large allocs to 128 KiB
const size_t kSmallAlloc = 1048576; // largest "small" allocation is 1 MiB

struct Block {
  int           device;      // gpu
  lwdaStream_t  stream;      // allocation stream
  stream_set    stream_uses; // streams on which the block was used
  size_t        size;        // block size in bytes
  char*         ptr;         // memory address
  bool          allocated;   // in-use flag
  Block*        prev;        // prev block if split from a larger allocation
  Block*        next;        // next block if split from a larger allocation
  int           event_count; // number of outstanding LWCA events

  Block(int device, lwdaStream_t stream, size_t size, char* ptr=NULL) :
      device(device), stream(stream), stream_uses(), size(size), ptr(ptr),
      allocated(0), prev(NULL), next(NULL), event_count(0) { }
};

static bool BlockComparator(const Block* a, const Block* b)
{
  if (a->device != b->device) {
    return a->device < b->device;
  }
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

} // namespace

struct THCCachingAllocator
{
  typedef bool (*Comparison)(const Block*, const Block*);
  typedef std::set<Block*, Comparison> FreeBlocks;

  // lock around all operations
  std::mutex mutex;

  // lock around calls to lwdaFree (to prevent deadlocks with LWCL)
  std::mutex lwda_free_mutex;

  // cached blocks larger than 1 MB
  FreeBlocks large_blocks;

  // cached blocks 1 MB or smaller
  FreeBlocks small_blocks;

  // allocated blocks by device pointer
  std::unordered_map<void*, Block*> allocated_blocks;

  // outstanding lwca events
  std::deque<std::pair<lwdaEvent_t, Block*>> lwda_events;

  THCCachingAllocator() :
      large_blocks(BlockComparator),
      small_blocks(BlockComparator) {}

  /** allocates a block which is safe to use from the provided stream */
  lwdaError_t malloc(void** devPtr, size_t size, lwdaStream_t stream)
  {
    std::lock_guard<std::mutex> lock(mutex);

    int device;
    lwdaError_t err = lwdaGetDevice(&device);
    if (err != lwdaSuccess) {
      return err;
    }

    err = process_events();
    if (err != lwdaSuccess) {
      return err;
    }

    size = round_size(size);
    bool small = size <= kSmallAlloc;

    Block search_key(device, stream, size);
    auto& free_blocks = small ? large_blocks : small_blocks;

    Block* block = NULL;
    Block* remaining = NULL;

    auto it = free_blocks.lower_bound(&search_key);
    if (it != free_blocks.end() && (*it)->device == device && (*it)->stream == stream) {
      block = *it;
      free_blocks.erase(it);
    } else {
      void* ptr;
      size_t alloc_size = small ? kSmallAlloc : size;
      err = lwda_malloc_retry(device, &ptr, alloc_size);
      if (err != lwdaSuccess) {
        return err;
      }
      block = new Block(device, stream, alloc_size, (char*)ptr);
    }

    if (block->size - size >= (small ? kRoundSmall : kSmallAlloc + 1)) {
      remaining = block;

      block = new Block(device, stream, size, block->ptr);
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr += size;
      remaining->size -= size;
      free_blocks.insert(remaining);
    }

    block->allocated = true;
    allocated_blocks[block->ptr] = block;

    *devPtr = (void*)block->ptr;
    return lwdaSuccess;
  }

  lwdaError_t free(void* ptr)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!ptr) {
      return lwdaSuccess;
    }

    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return lwdaErrorIlwalidDevicePointer;
    }

    Block* block = it->second;
    allocated_blocks.erase(it);
    block->allocated = false;

    if (!block->stream_uses.empty()) {
      return insert_events(block);
    }

    free_block(block);
    return lwdaSuccess;
  }

  /** returns cached blocks to the system allocator */
  lwdaError_t emptyCache()
  {
    std::lock_guard<std::mutex> lock(mutex);
    lwdaError_t err = free_blocks(large_blocks, large_blocks.begin(), large_blocks.end());
    if (err != lwdaSuccess) {
      return err;
    }
    err = free_blocks(small_blocks, small_blocks.begin(), small_blocks.end());
    if (err != lwdaSuccess) {
      return err;
    }
    return lwdaSuccess;
  }

  void* getBaseAllocation(void* ptr, size_t* outSize)
  {
    std::lock_guard<std::mutex> lock(mutex);
    Block* block = find_allocated_block(ptr);
    if (!block) {
      THError("invalid device pointer: %p", ptr);
    }
    while (block->prev) {
      block = block->prev;
    }
    void *basePtr = block->ptr;
    if (outSize) {
      size_t size = 0;
      while (block) {
        size += block->size;
        block = block->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  // Aclwmulates sizes of all memory blocks for given device in given free list
  void cacheInfoAux(FreeBlocks& blocks, int dev_id, size_t* total, size_t* largest)
  {
    Block search_key(dev_id, 0, 0);
    auto it = blocks.lower_bound(&search_key);
    for (;it != blocks.end() && *it && (*it)->device == dev_id; ++it) {
      size_t blocksize = (*it)->size;
      *total += blocksize;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }

  void cacheInfo(int dev_id, size_t* total, size_t* largest)
  {
    std::lock_guard<std::mutex> lock(mutex);
    cacheInfoAux(large_blocks, dev_id, total, largest);
    cacheInfoAux(small_blocks, dev_id, total, largest);
  }

  void recordStream(void* ptr, THCStream* stream)
  {
    std::lock_guard<std::mutex> lock(mutex);
    Block* block = find_allocated_block(ptr);
    if (!block) {
      THError("invalid device pointer: %p", ptr);
    }
    if (stream->stream == block->stream) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    THCStream_retain(stream);
    block->stream_uses.insert(THCStreamPtr(stream, &THCStream_free));
  }

  /** moves a block into the free block list */
  void free_block(Block* block)
  {
    THAssert(!block->allocated && block->event_count == 0);
    bool small = block->size <= kSmallAlloc;
    auto& free_blocks = small ? large_blocks : small_blocks;
    try_merge_blocks(block, block->prev, free_blocks);
    try_merge_blocks(block, block->next, free_blocks);
    free_blocks.insert(block);
  }

  /** combine previously split blocks */
  void try_merge_blocks(Block* dst, Block* src, FreeBlocks& free_blocks)
  {
    if (!src || src->allocated || src->event_count > 0) {
      return;
    }
    if (dst->prev == src) {
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else {
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }
    dst->size += src->size;
    free_blocks.erase(src);
    delete src;
  }

  size_t round_size(size_t size)
  {
    if (size < kRoundSmall) {
      size = kRoundSmall;
    } else if (size < kSmallAlloc) {
      size += kRoundSmall - 1 - (size - 1) % kRoundSmall;
    } else {
      size += kRoundLarge - 1 - (size - 1) % kRoundLarge;
    }
    return size;
  }

  lwdaError_t lwda_malloc_retry(int device, void** devPtr, size_t size)
  {
    // Try lwdaMalloc. If lwdaMalloc fails, frees all non-split cached blocks
    // and retries.
    lwdaError_t err = lwdaMalloc(devPtr, size);
    if (err != lwdaSuccess) {
      lwdaGetLastError();
      err = free_cached_blocks(device);
      if (err != lwdaSuccess) {
        return err;
      }
      err = lwdaMalloc(devPtr, size);
      if (err != lwdaSuccess) {
        return err;
      }
    }
    return lwdaSuccess;
  }

  lwdaError_t free_cached_blocks(int device)
  {
    // Free all non-split cached blocks on device
    Block lower_bound(device, NULL, 0);
    Block upper_bound(device + 1, NULL, 0);

    lwdaError_t err = free_blocks(
        large_blocks,
        large_blocks.lower_bound(&lower_bound),
        large_blocks.lower_bound(&upper_bound));
    if (err != lwdaSuccess) {
      return err;
    }
    err = free_blocks(
        small_blocks,
        small_blocks.lower_bound(&lower_bound),
        small_blocks.lower_bound(&upper_bound));
    return err;
  }

  lwdaError_t free_blocks(FreeBlocks& blocks, FreeBlocks::iterator it, FreeBlocks::iterator end)
  {
    // Frees all non-split blocks between `it` and `end`
    std::lock_guard<std::mutex> lock(lwda_free_mutex);
    while (it != end) {
      Block* block = *it;
      if (!block->prev && !block->next) {
        lwdaError_t err = lwdaFree((void*)block->ptr);
        if (err != lwdaSuccess) {
          return err;
        }
        auto lwr = it;
        ++it;
        blocks.erase(lwr);
        delete block;
      } else {
        ++it;
      }
    }
    return lwdaSuccess;
  }

  Block* find_allocated_block(void *ptr) {
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return NULL;
    }
    return it->second;
  }

  lwdaError_t insert_events(Block* block)
  {
    lwdaError_t err;

    int prev_device;
    err = lwdaGetDevice(&prev_device);
    if (err != lwdaSuccess) return err;

    std::set<THCStreamPtr> streams(std::move(block->stream_uses));
    THAssert(block->stream_uses.empty());
    for (auto it = streams.begin(); it != streams.end(); ++it) {
      auto& stream = *it;

      err = lwdaSetDevice(stream->device);
      if (err != lwdaSuccess) break;

      lwdaEvent_t event;
      err = lwdaEventCreateWithFlags(&event, lwdaEventDisableTiming);
      if (err != lwdaSuccess) break;

      err = lwdaEventRecord(event, stream->stream);
      if (err != lwdaSuccess) break;

      block->event_count++;
      lwda_events.emplace_back(event, block);
    }

    lwdaSetDevice(prev_device);
    return err;
  }

  lwdaError_t process_events()
  {
    // Process outstanding lwdaEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    while (!lwda_events.empty()) {
      auto& e = lwda_events.front();
      lwdaEvent_t event = e.first;
      Block* block = e.second;

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

      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
      lwda_events.pop_front();
    }
    return lwdaSuccess;
  }
};

static lwdaError_t THCCachingAllocator_malloc(void* ctx, void** ptr, size_t size, lwdaStream_t stream)
{
  THCCachingAllocator* a = (THCCachingAllocator*) ctx;
  return a->malloc(ptr, size, stream);
}

static lwdaError_t THCCachingAllocator_free(void* ctx, void* ptr)
{
  THCCachingAllocator* a = (THCCachingAllocator*) ctx;
  return a->free(ptr);
}

static lwdaError_t THCCachingAllocator_emptyCache(void* ctx)
{
  THCCachingAllocator* a = (THCCachingAllocator*) ctx;
  return a->emptyCache();
}

static lwdaError_t THCCachingAllocator_cacheInfo(void* ctx, int dev_id, size_t* cachedAndFree, size_t* largestBlock)
{
  THCCachingAllocator* a = (THCCachingAllocator*) ctx;
  a->cacheInfo(dev_id, cachedAndFree, largestBlock);
  return lwdaSuccess;
}

static THCCachingAllocator caching_allocator;
static THCDeviceAllocator device_allocator = {
  &THCCachingAllocator_malloc,
  NULL,
  &THCCachingAllocator_free,
  &THCCachingAllocator_emptyCache,
  &THCCachingAllocator_cacheInfo,
  &caching_allocator
};

THC_API THCDeviceAllocator* THCCachingAllocator_get(void)
{
  return &device_allocator;
}

THC_API void* THCCachingAllocator_getBaseAllocation(void *ptr, size_t *size)
{
  return caching_allocator.getBaseAllocation(ptr, size);
}

THC_API void THCCachingAllocator_recordStream(void *ptr, THCStream* stream)
{
  caching_allocator.recordStream(ptr, stream);
}

THC_API std::mutex* THCCachingAllocator_getLwdaFreeMutex()
{
  return &caching_allocator.lwda_free_mutex;
}
