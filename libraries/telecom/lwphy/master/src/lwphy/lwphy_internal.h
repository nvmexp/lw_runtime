/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined LWPHY_INTERNAL_H_INCLUDED_
#define LWPHY_INTERNAL_H_INCLUDED_

#include <lwda_runtime.h>
#include <stdio.h>
#include <memory>

#ifdef __LWDACC__
#define LWDA_BOTH __host__ __device__
#define LWDA_BOTH_INLINE __forceinline__ __host__ __device__
#define LWDA_INLINE __forceinline__ __device__
#else
#define LWDA_BOTH
#define LWDA_INLINE
#ifdef WINDOWS
#define LWDA_BOTH_INLINE __inline
#else
#define LWDA_BOTH_INLINE __inline__
#endif
#endif

#define LWDA_CHECK(result)                        \
    if((lwdaError_t)result != lwdaSuccess)        \
    {                                             \
        fprintf(stderr,                           \
                "LWCA Runtime Error: %s:%i:%s\n", \
                __FILE__,                         \
                __LINE__,                         \
                lwdaGetErrorString(result));      \
    }

#if LWPHY_DEBUG
  #define DEBUG_PRINTF(...) do { printf(__VA_ARGS__); } while(0)
  #define DEBUG_PRINT_FUNC_ATTRIBUTES(func) do                                              \
          {                                                                                 \
              lwdaFuncAttributes fAttr;                                                     \
              LWDA_CHECK(lwdaFuncGetAttributes(&fAttr, func));                              \
              printf(#func ":\n");                                                          \
              printf("\tbinaryVersion:             %i\n",  fAttr.binaryVersion);            \
              printf("\tcacheModeCA:               %i\n",  fAttr.cacheModeCA);              \
              printf("\tconstSizeBytes:            %lu\n", fAttr.constSizeBytes);           \
              printf("\tlocalSizeBytes:            %lu\n", fAttr.localSizeBytes);           \
              printf("\tmaxDynamicSharedSizeBytes: %i\n", fAttr.maxDynamicSharedSizeBytes); \
              printf("\tmaxThreadsPerBlock:        %i\n", fAttr.maxThreadsPerBlock);        \
              printf("\tnumRegs:                   %i\n", fAttr.numRegs);                   \
              printf("\tpreferredShmemCarveout:    %i\n", fAttr.preferredShmemCarveout);    \
              printf("\tptxVersion:                %i\n", fAttr.ptxVersion);                \
              printf("\tsharedSizeBytes:           %lu\n", fAttr.sharedSizeBytes);          \
          } while(0)
  #define DEBUG_PRINT_FUNC_MAX_BLOCKS(func, blkDim, dynShMem) do                            \
          {                                                                                 \
              int maxBlocks = 0;                                                            \
              LWDA_CHECK(lwdaOclwpancyMaxActiveBlocksPerMultiprocessor(&maxBlocks,          \
                         func,                                                              \
                         blkDim.x * blkDim.y * blkDim.z,                                    \
                         dynShMem));                                                        \
              printf(#func " max blocks per SM: %i\n", maxBlocks);                          \
          } while(0)
#else
  #define DEBUG_PRINTF(...)
  #define DEBUG_PRINT_FUNC_ATTRIBUTES(func)
  #define DEBUG_PRINT_FUNC_MAX_BLOCKS(func, blkSize, dynShMem)
#endif

#define TIME_KERNEL(kernel_call, ITER_COUNT, strm)                              \
    do                                                                          \
    {                                                                           \
        lwdaEvent_t eStart, eFinish;                                            \
        LWDA_CHECK(lwdaEventCreate(&eStart));                                   \
        LWDA_CHECK(lwdaEventCreate(&eFinish));                                  \
        lwdaEventRecord(eStart, strm);                                          \
        for(size_t i = 0; i < ITER_COUNT; ++i)                                  \
        {                                                                       \
            kernel_call;                                                        \
        }                                                                       \
        lwdaEventRecord(eFinish, strm);                                         \
        lwdaEventSynchronize(eFinish);                                          \
        lwdaError_t e = lwdaGetLastError();                                     \
        if(lwdaSuccess != e)                                                    \
        {                                                                       \
            fprintf(stderr, "LWCA ERROR: (%s:%i) %s\n", __FILE__, __LINE__,     \
                    lwdaGetErrorString(e));                                     \
        }                                                                       \
        float elapsed_ms = 0.0f;                                                \
        lwdaEventElapsedTime(&elapsed_ms, eStart, eFinish);                     \
        printf("Average (%i iterations) elapsed time in usec = %.0f\n",         \
               ITER_COUNT, elapsed_ms * 1000 / ITER_COUNT);                     \
    lwdaEventDestroy(eStart);                                                   \
    lwdaEventDestroy(eFinish);                                                  \
  } while (0)

namespace
{
LWDA_BOTH_INLINE bool is_set(unsigned int flag, unsigned int mask)
{
    return (0 != (flag & mask));
}
LWDA_BOTH_INLINE unsigned int bit(unsigned int pos) { return (1UL << pos); }
LWDA_BOTH_INLINE unsigned int bit_set(unsigned int pos, unsigned int mask)
{
    return (mask |= bit(pos));
}

template <typename T>
LWDA_BOTH_INLINE T round_up_to_next(T val, T increment)
{
    return ((val + (increment - 1)) / increment) * increment;
}
template <typename T>
LWDA_BOTH_INLINE T div_round_up(T val, T divide_by)
{
    return ((val + (divide_by - 1)) / divide_by);
}

} // namespace

namespace lwphy_i // lwphy internal
{
////////////////////////////////////////////////////////////////////////
// lwda_exception
// Exception class for errors from LWCA
class lwda_exception : public std::exception //
{
public:
    lwda_exception(lwdaError_t s) :
        status_(s) {}
    virtual ~lwda_exception() = default;
    virtual const char* what() const noexcept
    {
        return lwdaGetErrorString(status_);
    }
    lwdaError_t status() const { return status_; }

private:
    lwdaError_t status_;
};

template <class T>
struct device_deleter
{
    // typedef typename std::remove_all_extents<T>::type ptr_t;
    typedef T ptr_t;
    void      operator()(ptr_t* p) const
    {
        //printf("Freeing device bytes at 0x%p\n", p);
        lwdaFree(p);
    }
};
template <class T>
struct pinned_deleter
{
    // typedef typename std::remove_all_extents<T>::type ptr_t;
    typedef T ptr_t;
    void      operator()(ptr_t* p) const { lwdaFreeHost(p); }
};

template <typename T>
using unique_device_ptr = std::unique_ptr<T, device_deleter<T>>;

template <typename T>
unique_device_ptr<T> make_unique_device(size_t count = 1)
{
    typedef typename unique_device_ptr<T>::pointer pointer_t;
    pointer_t                                      p;
    lwdaError_t                                    res = lwdaMalloc(&p, count * sizeof(T));
    if(lwdaSuccess != res)
    {
        throw lwda_exception(res);
    }
    //printf("Allocated %lu device bytes at 0x%p\n", count * sizeof(T), p);
    return unique_device_ptr<T>(p);
}

} // namespace lwphy_i

#endif // !defined(LWPHY_INTERNAL_H_INCLUDED_)
