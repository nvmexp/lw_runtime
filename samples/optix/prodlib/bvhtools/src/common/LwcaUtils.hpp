// Copyright LWPU Corporation 2013
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once
#include <lwda_runtime.h>
#include <string>
#include <vector>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// Utility macro for calling kernel launchers defined in *.lw files.

#define LAUNCH(lwca, kernel, warpsPerBlock, numThreads, ...)                                              \
  do {                                                                                                    \
    dim3 blockDim(32, (warpsPerBlock));                                                                   \
    dim3 gridDim = (lwca).calcGridDim((numThreads), blockDim);                                            \
    (lwca).beginTimer("launch" #kernel);                                                                  \
    lwdaError_t err = lwdaErrorUnknown;                                                                   \
    if (!launch ## kernel(gridDim, blockDim, (lwca).getDefaultStream(), __VA_ARGS__) ||                   \
        (err = lwdaGetLastError()) != lwdaSuccess )                                                       \
      throw LwdaRuntimeError(RT_EXCEPTION_INFO, "launch" #kernel "()", err);               \
    (lwca).endTimer();                                                                                    \
  } while (0)

//------------------------------------------------------------------------

class LwdaUtils
{
public:
                                LwdaUtils           (void);
                                ~LwdaUtils          (void);

    // Streams, events, and synchronization.
    // Note: If silent = true, the sync methods will not use timers or check for errors.

    void                        setDefaultStream    (lwdaStream_t stream) { m_defaultStream = stream; }
    lwdaStream_t                getDefaultStream    (void) const { return m_defaultStream; }

    lwdaEvent_t                 createEvent         (void);
    void                        destroyEvent        (lwdaEvent_t ev);
    void                        recordEvent         (lwdaEvent_t ev, lwdaStream_t stream);
    void                        recordEvent         (lwdaEvent_t ev) { recordEvent(ev, getDefaultStream()); }

    void                        deviceSynchronize   (bool silent = false);
    void                        streamSynchronize   (lwdaStream_t stream, bool silent = false);
    void                        streamSynchronize   (bool silent = false) { streamSynchronize(getDefaultStream(), silent); }
    void                        eventSynchronize    (lwdaEvent_t ev, bool silent = false);

    // Memory management.

    char*                       deviceAlloc         (size_t size);
    void                        deviceFree          (void* ptr);
    char*                       hostAlloc           (size_t size);
    void                        hostFree            (void* ptr);

    void                        clearDeviceBuffer   (void* ptr, unsigned char value, size_t size, lwdaStream_t stream);
    void                        clearDeviceBuffer   (void* ptr, unsigned char value, size_t size) { clearDeviceBuffer(ptr, value, size, getDefaultStream()); }

    void                        setInt              (int* dst, int value, lwdaStream_t stream);
    void                        setInt              (int* dst, int value) { setInt(dst, value, getDefaultStream()); }

    void                        memcpyHtoDAsync     (void* dst, const void* src, size_t size, lwdaStream_t stream);
    void                        memcpyDtoHAsync     (void* dst, const void* src, size_t size, lwdaStream_t stream);
    void                        memcpyDtoDAsync     (void* dst, const void* src, size_t size, lwdaStream_t stream);
    void                        memcpyHtoDAsync     (void* dst, const void* src, size_t size) { memcpyHtoDAsync(dst, src, size, getDefaultStream()); }
    void                        memcpyDtoHAsync     (void* dst, const void* src, size_t size) { memcpyDtoHAsync(dst, src, size, getDefaultStream()); }
    void                        memcpyDtoDAsync     (void* dst, const void* src, size_t size) { memcpyDtoDAsync(dst, src, size, getDefaultStream()); }

    // Device properties.

    int                         getSMArch           (void) { initDeviceProps(); return m_deviceProp.major * 10 + m_deviceProp.minor; } // e.g. 20 for Fermi
    int                         getMaxGridWidth     (void) { initDeviceProps(); return m_deviceProp.maxGridSize[0]; }
    int                         getTextureAlign     (void) { initDeviceProps(); return (int)m_deviceProp.textureAlignment; }
    int                         getNumSMs           (void) { initDeviceProps(); return m_deviceProp.multiProcessorCount; }
    int                         getMaxThreadsPerSM  (void) { initDeviceProps(); return m_deviceProp.maxThreadsPerMultiProcessor; }
    int                         getMaxThreads       (void) { initDeviceProps(); return m_deviceProp.multiProcessorCount * m_deviceProp.maxThreadsPerMultiProcessor; }
    int                         getMaxTextureSize1D (void) { initDeviceProps(); return m_deviceProp.maxTexture1DLinear; }
    void                        getMemInfo          (size_t& free, size_t& total);

    dim3                        calcGridDim         (int numThreads, dim3 blockDim);

    // Timing.

    void                        resetTiming         (bool enable);  // Clear all timing info collected so far. If enable = false, the rest of the timing methods will return immediately.
    void                        repeatTiming        (void);         // Perform another round of timing for the same set of operations. Useful for obtaining more accurate results.
    void                        printTiming         (int numItems = 0, FILE* out = stdout, const char* linePrefix = ""); // Reports the timing info collected so far, taking the minimum over all rounds.

    void                        beginTimer          (const char* name, lwdaStream_t stream); // Start a new timer and push it to the stack.
    void                        beginTimer          (const char* name) { beginTimer(name, getDefaultStream()); }

    void                        endTimer            (lwdaStream_t stream); // Stop the innermost timer and pop it off the stack.
    void                        endTimer            (void) { endTimer(getDefaultStream()); }

private:
    void                        clearTimerEvents    (void);
    void                        updateTimerTotals   (void);
    void                        initDeviceProps     (void);
    
private:
                                LwdaUtils           (const LwdaUtils&); // forbidden
    LwdaUtils&                  operator=           (const LwdaUtils&); // forbidden

private:
    lwdaStream_t                m_defaultStream;
    std::vector<lwdaEvent_t>    m_eventPool;        // Pool of unused event objects.

    bool                        m_devicePropsInited;
    int                         m_deviceId;
    lwdaDeviceProp              m_deviceProp;

    // Timing.

    bool                        m_enableTiming;     // Collect timing info?
    std::string                 m_lwrrTimerName;    // Full hierarchical name of the innermost running timer. Empty string if no timers are running.

    std::vector<lwdaEvent_t>    m_timerEvents;      // Chronological list of timer events recorded in the current round.
    std::vector<std::string>    m_timerEventNames;  // Full hierarchical names corresponding to the timer events.

    struct TimerTotal
    {
        std::string             name;               // Full hierarchical name of the timer.
        int                     num;                // Number of times the timer has been started.
        double                  sec;                // Number of seconds the timer has been running.
    };
    std::vector<TimerTotal>     m_timerTotals;      // Minimum over all rounds finished so far.
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
