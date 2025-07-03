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

#include "LwdaUtils.hpp"
#include "SetIntKernel.hpp"
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/IlwalidDevice.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/system/Knobs.h>
#include <corelib/misc/String.h>
#include <memory.h>
#include <set>
#include <map>
#include <algorithm>

//------------------------------------------------------------------------

static const char* const    TIMER_GROUP_SEPARATOR   = "\x1";
static const char* const    TIMER_GROUP_INDENT      = "  ";

//------------------------------------------------------------------------

namespace
{
Knob<bool>   k_memScribbleOn(          RT_DSTRING("bvhtools.memScribble"),            false, RT_DSTRING("Turn on memory scribbling") );
Knob<int>    k_memScribbleValueDevice( RT_DSTRING("bvhtools.memScribbleValueDevice"), 0xab,  RT_DSTRING("Byte value to scribble into device allocations") );
Knob<int>    k_memScribbleValueHost(   RT_DSTRING("bvhtools.memScribbleValueHost"),   0xcd,  RT_DSTRING("Byte value to scribbled into host allocations") );
Knob<size_t> k_maximumDeviceMemory(    RT_DSTRING("lwca.maximumDeviceMemory"),        0,     RT_DSTRING("Set a limit on the visible device memory. Default is 0, which means use what the driver reports.") );
}

//------------------------------------------------------------------------

#define CHK_LWDA( call )                                                            \
    do                                                                              \
    {                                                                               \
        lwdaError err__ = call;                                                     \
        if (err__ != lwdaSuccess)                                                   \
            throw prodlib::LwdaRuntimeError(RT_EXCEPTION_INFO, #call, err__); \
    }                                                                               \
    while (0)

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

LwdaUtils::LwdaUtils(void)
:   m_defaultStream     (0),
    m_devicePropsInited (false),
    m_deviceId          (-1),
    m_enableTiming      (false)
{
    // Note: It must be possible to create a LwdaUtils even if LWCA is not available.
    // Hence, there can be no LWCA calls in the constructor.
}

//------------------------------------------------------------------------

LwdaUtils::~LwdaUtils(void)
{
    clearTimerEvents();
    for (int i = 0; i < (int)m_eventPool.size(); i++)
        lwdaEventDestroy(m_eventPool[i]);
}

//------------------------------------------------------------------------

lwdaEvent_t LwdaUtils::createEvent(void)
{
    lwdaEvent_t ev;
    if (!m_eventPool.size())
        CHK_LWDA(lwdaEventCreate(&ev, 0));
    else
    {
        ev = m_eventPool.back();
        m_eventPool.pop_back();
    }
    return ev;
}

//------------------------------------------------------------------------

void LwdaUtils::destroyEvent(lwdaEvent_t ev)
{
    if (ev)
        m_eventPool.push_back(ev);
}

//------------------------------------------------------------------------

void LwdaUtils::recordEvent(lwdaEvent_t ev, lwdaStream_t stream)
{
    CHK_LWDA(lwdaEventRecord(ev, stream));
}

//------------------------------------------------------------------------

void LwdaUtils::deviceSynchronize(bool silent)
{
    if (silent)
        lwdaDeviceSynchronize();
    else
    {
        beginTimer("lwdaDeviceSynchronize");
        CHK_LWDA(lwdaDeviceSynchronize());
        endTimer();
    }
}

//------------------------------------------------------------------------

void LwdaUtils::streamSynchronize(lwdaStream_t stream, bool silent)
{
    if (silent)
        lwdaStreamSynchronize(stream);
    else
    {
        beginTimer("lwdaStreamSynchronize");
        CHK_LWDA(lwdaStreamSynchronize(stream));
        endTimer();
    }
}

//------------------------------------------------------------------------

void LwdaUtils::eventSynchronize(lwdaEvent_t ev, bool silent)
{
    if (silent)
        lwdaEventSynchronize(ev);
    else
    {
        beginTimer("lwdaEventSynchronize");
        CHK_LWDA(lwdaEventSynchronize(ev));
        endTimer();
    }
}

//------------------------------------------------------------------------

char* LwdaUtils::deviceAlloc(size_t size)
{
    void *ptr;
    beginTimer("lwdaMalloc");

    CHK_LWDA(lwdaMalloc(&ptr, size));
    if (k_memScribbleOn.get())
        CHK_LWDA(lwdaMemset(ptr, k_memScribbleValueDevice.get(), size));
        
    endTimer();
    return (char*)ptr;
}

//------------------------------------------------------------------------

void LwdaUtils::deviceFree(void* ptr)
{
    if (ptr)
        lwdaFree(ptr); // Do not use CHK_LWDA() or timers here; it would mess with exception handling.
}

//------------------------------------------------------------------------

char* LwdaUtils::hostAlloc(size_t size)
{
    void* ptr;
    beginTimer("lwdaMallocHost");

    CHK_LWDA(lwdaMallocHost(&ptr, size));
    if (k_memScribbleOn.get())
        memset(ptr, k_memScribbleValueHost.get(), size);

    endTimer();
    return (char*)ptr;
}

//------------------------------------------------------------------------

void LwdaUtils::hostFree(void* ptr)
{
    if (ptr)
        lwdaFreeHost(ptr); // Do not put CHK_LWDA() here; it would mess with exception handling.
}

//------------------------------------------------------------------------

void LwdaUtils::clearDeviceBuffer(void* ptr, unsigned char value, size_t size, lwdaStream_t stream)
{
    if (!size)
        return;

    RT_ASSERT(ptr);
    CHK_LWDA(lwdaMemsetAsync(ptr, value, size, stream));
}

//------------------------------------------------------------------------

void LwdaUtils::setInt(int* dst, int value, lwdaStream_t stream)
{
  prodlib::bvhtools::setInt(stream, dst, value);
}

//------------------------------------------------------------------------

void LwdaUtils::memcpyHtoDAsync(void* dst, const void* src, size_t size, lwdaStream_t stream)
{
    if (!size)
        return;

    RT_ASSERT(dst && src);
    CHK_LWDA(lwdaMemcpyAsync(dst, src, size, lwdaMemcpyHostToDevice, stream));
}

//------------------------------------------------------------------------

void LwdaUtils::memcpyDtoHAsync(void* dst, const void* src, size_t size, lwdaStream_t stream)
{
    if (!size)
        return;

    RT_ASSERT(dst && src);
    CHK_LWDA(lwdaMemcpyAsync(dst, src, size, lwdaMemcpyDeviceToHost, stream));
}

//------------------------------------------------------------------------

void LwdaUtils::memcpyDtoDAsync(void* dst, const void* src, size_t size, lwdaStream_t stream)
{
    if (!size)
        return;

    RT_ASSERT(dst && src);
    CHK_LWDA(lwdaMemcpyAsync(dst, src, size, lwdaMemcpyDeviceToDevice, stream));
}

//------------------------------------------------------------------------

void LwdaUtils::initDeviceProps( void )
{
    if (m_devicePropsInited)
        return;

    m_devicePropsInited = true;
    CHK_LWDA(lwdaGetDevice(&m_deviceId));
    CHK_LWDA(lwdaDeviceGetAttribute(&m_deviceProp.major, lwdaDevAttrComputeCapabilityMajor, m_deviceId));
    CHK_LWDA(lwdaDeviceGetAttribute(&m_deviceProp.minor, lwdaDevAttrComputeCapabilityMinor, m_deviceId));
    CHK_LWDA(lwdaDeviceGetAttribute(&m_deviceProp.warpSize, lwdaDevAttrWarpSize, m_deviceId));   
    CHK_LWDA(lwdaDeviceGetAttribute(&m_deviceProp.maxGridSize[0], lwdaDevAttrMaxGridDimX, m_deviceId));
    int textureAlignment = 0;
    CHK_LWDA(lwdaDeviceGetAttribute(&textureAlignment, lwdaDevAttrTextureAlignment, m_deviceId));
    m_deviceProp.textureAlignment = (size_t)textureAlignment;
    CHK_LWDA(lwdaDeviceGetAttribute(&m_deviceProp.multiProcessorCount, lwdaDevAttrMultiProcessorCount, m_deviceId));
    CHK_LWDA(lwdaDeviceGetAttribute(&m_deviceProp.maxThreadsPerMultiProcessor, lwdaDevAttrMaxThreadsPerMultiProcessor, m_deviceId));
    CHK_LWDA(lwdaDeviceGetAttribute(&m_deviceProp.maxTexture1DLinear, lwdaDevAttrMaxTexture1DLinearWidth, m_deviceId));
}

//------------------------------------------------------------------------

void LwdaUtils::getMemInfo( size_t& free, size_t& total )
{
    const size_t memLimit = k_maximumDeviceMemory.get();
    CHK_LWDA( lwdaMemGetInfo( &free, &total ) );
    total = ( memLimit > 0 && memLimit < total ) ? memLimit : total;
}

//------------------------------------------------------------------------

dim3 LwdaUtils::calcGridDim(int numThreads, dim3 blockDim)
{
    int gridWidth = (numThreads <= 0) ? 1 : ((numThreads - 1) / (blockDim.x * blockDim.y * blockDim.z) + 1);
    int gridHeight = 1;
    while (gridWidth > getMaxGridWidth())
    {
        gridWidth = (gridWidth + 1) >> 1;
        gridHeight <<= 1;
    }
    return dim3(gridWidth, gridHeight);
}

//------------------------------------------------------------------------

void LwdaUtils::resetTiming(bool enable)
{
    // Clear timing info.

    clearTimerEvents();
    m_timerTotals.clear();

    // Set enabled status.

    m_enableTiming = enable;
    m_lwrrTimerName = "";

    // Wait for any previous GPU work to finish.

    if (m_enableTiming)
        deviceSynchronize(true);
}

//------------------------------------------------------------------------

void LwdaUtils::repeatTiming(void)
{
    // Timing disabled => skip.

    if (!m_enableTiming)
        return;

    // There are timers still running => end them.

    while (m_lwrrTimerName.length())
        endTimer();

    // Update totals for the current round.

    deviceSynchronize(true);
    updateTimerTotals();

    // Start a new round.

    clearTimerEvents();
    deviceSynchronize(true);
}

//------------------------------------------------------------------------

void LwdaUtils::printTiming(int numItems, FILE* out, const char* linePrefix)
{
    RT_ASSERT(numItems >= 0);
    RT_ASSERT(out);
    RT_ASSERT(linePrefix);

    // Timing disabled => skip.

    if (!m_enableTiming)
        return;

    // Finish up the current round.

    repeatTiming();

    // Choose the timers to report.

    std::vector<TimerTotal> timers;
    for (int i = 0; i < (int)m_timerTotals.size(); i++)
        if (m_timerTotals[i].num)
            timers.push_back(m_timerTotals[i]);

    // Collect the names of timer groups.

    std::set<std::string> groupNames;
    for (int i = 0; i < (int)timers.size(); i++)
    {
        std::string groupName = timers[i].name.substr(0, timers[i].name.rfind(TIMER_GROUP_SEPARATOR));
        if (!groupNames.count(groupName))
            groupNames.insert(groupName);
    }

    // Generate table rows by traversing the timers in hierarchical order.

    std::vector<TimerTotal> rows;
    std::vector<TimerTotal> groupStack;
    std::string indent = "";

    for (;;)
    {
        // Find timers that belong in the current group.

        for (int i = 0; i < (int)timers.size(); i++)
        {
            // Already processed => skip.

            TimerTotal timer = timers[i];
            if (!timer.name.length())
                continue;

            // Does not belong in the current group => skip.

            std::string groupName = timer.name.substr(0, timer.name.rfind(TIMER_GROUP_SEPARATOR));
            std::string activeGroupName = (groupStack.size()) ? groupStack.back().name : "";
            if (groupName != activeGroupName)
                continue;

            // Mark as processed.

            timers[i].name = "";

            // Add table row.

            rows.push_back(timer);
            rows.back().name = indent + timer.name.substr(groupName.length() + strlen(TIMER_GROUP_SEPARATOR));

            // Group => hide totals and process relwrsively.

            if (groupNames.count(timer.name))
            {
                rows.back().num = -1;
                rows.back().sec = -1.0;
                groupStack.push_back(timer);
                indent += TIMER_GROUP_INDENT;
                i = 0;
            }
        }

        // No more top-level groups => done.

        if (!groupStack.size())
            break;

        // Add overhead row for the current group.

        rows.push_back(groupStack.back());
        rows.back().name = indent + "overhead";
        rows.back().num = -1;

        // Continue with the parent group.

        groupStack.pop_back();
        indent = indent.substr(strlen(TIMER_GROUP_INDENT));
    }

    // Choose name column width and callwlate grand total.

    const char* nameTitle = "Timer name";
    int nameWidth = (int)strlen(nameTitle);

    TimerTotal grandTotal;
    grandTotal.name = "Total";
    grandTotal.num = -1;
    grandTotal.sec = 0.0;

    for (int i = 0; i < (int)rows.size(); i++)
    {
        nameWidth = std::max(nameWidth, (int)rows[i].name.length());
        if (rows[i].sec >= 0.0)
            grandTotal.sec += rows[i].sec;
    }

    rows.push_back(grandTotal);

    // Print title row.

    fprintf(out, "%s%-*s%5s%12s%11s%7s\n", linePrefix, nameWidth, nameTitle, "num", "ms", "ns/item", "%");

    // Print data rows.

    for (int i = 0; i < (int)rows.size(); i++)
    {
        const TimerTotal& row = rows[i];

        // Line prefix and title.

        fprintf(out, "%s%-*s", linePrefix, nameWidth, row.name.c_str());

        // Count.

        if (row.num >= 0)
            fprintf(out, "%5d", row.num);
        else
            fprintf(out, "%5s", "-");

        // Milliseconds.

        if (row.sec >= 0.0)
            fprintf(out, "%12.3f", row.sec * 1.0e3);
        else
            fprintf(out, "%12s", "-");

        // Nanoseconds per item.

        if (row.sec >= 0.0 && numItems > 0)
            fprintf(out, "%11.2f", row.sec / (double)numItems * 1.0e9);
        else
            fprintf(out, "%11s", "-");

        // Percent of total.

        double percent = row.sec / grandTotal.sec * 100.0;
        if (percent >= 0.05)
            fprintf(out, "%7.1f", percent);
        else
            fprintf(out, "%7s", "-");

        // Newline.

        fprintf(out, "\n");
    }
}

//------------------------------------------------------------------------

void LwdaUtils::beginTimer(const char* name, lwdaStream_t stream)
{
    // Timing disabled => skip.

    if (!m_enableTiming)
        return;

    // Postfix the name string to indicate a child of the previous timer.

    RT_ASSERT(name);
    RT_ASSERT(strlen(name));
    RT_ASSERT(std::string(name).find(TIMER_GROUP_SEPARATOR) == std::string::npos);
    m_lwrrTimerName += TIMER_GROUP_SEPARATOR;
    m_lwrrTimerName += name;

    // Record start event for the child timer.

    lwdaEvent_t ev = createEvent();
    recordEvent(ev, stream);
    m_timerEvents.push_back(ev);
    m_timerEventNames.push_back(m_lwrrTimerName);
}

//------------------------------------------------------------------------

void LwdaUtils::endTimer(lwdaStream_t stream)
{
    // Timing disabled => skip.

    if (!m_enableTiming)
        return;

    // Strip the name string to indicate the parent of the previous timer.

    RT_ASSERT(m_lwrrTimerName.rfind(TIMER_GROUP_SEPARATOR) != std::string::npos);
    m_lwrrTimerName = m_lwrrTimerName.substr(0, m_lwrrTimerName.rfind(TIMER_GROUP_SEPARATOR));

    // Record new start event for the parent timer.

    lwdaEvent_t ev = createEvent();
    recordEvent(ev, stream);
    m_timerEvents.push_back(ev);
    m_timerEventNames.push_back(m_lwrrTimerName);
}

//------------------------------------------------------------------------

void LwdaUtils::clearTimerEvents(void)
{
    for (int i = 0; i < (int)m_timerEvents.size(); i++)
        destroyEvent(m_timerEvents[i]);
    m_timerEvents.clear();
    m_timerEventNames.clear();
}

//------------------------------------------------------------------------

void LwdaUtils::updateTimerTotals(void)
{
    // No timer events recorded on this round => skip.

    if (!m_timerEvents.size())
        return;

    // Assign a running index for each unique timer name encountered so far.

    int numNames = 0;
    std::map<std::string, int> nameToIdx;

    for (int i = 0; i < (int)m_timerTotals.size(); i++)
        if (!nameToIdx.count(m_timerTotals[i].name))
            nameToIdx[m_timerTotals[i].name] = numNames++;

    for (int i = 0; i < (int)m_timerEventNames.size(); i++)
        if (!nameToIdx.count(m_timerEventNames[i]))
            nameToIdx[m_timerEventNames[i]] = numNames++;

    // Setup the array of TimerTotals.

    std::vector<TimerTotal> totals(numNames);
    for (auto i = nameToIdx.begin(); i != nameToIdx.end(); i++)
    {
        TimerTotal& t = totals[i->second];
        t.name = i->first;
        t.num = 0;
        t.sec = 0.0;
    }

    // Query events and accumulate totals for the current round.

    for (int i = 0; i < (int)m_timerEvents.size() - 1; i++)
    {
        TimerTotal& t = totals[nameToIdx[m_timerEventNames[i]]];
        t.num++;

        float millis;
        CHK_LWDA(lwdaEventElapsedTime(&millis, m_timerEvents[i], m_timerEvents[i + 1]));
        t.sec += millis * 1.0e-3f;
    }

    // Merge results from the previous rounds by taking the minimum.

    for (int i = 0; i < (int)m_timerTotals.size(); i++)
    {
        const TimerTotal& told = m_timerTotals[i];
        TimerTotal& tnew = totals[nameToIdx[told.name]];
        tnew.num = std::min(tnew.num, told.num);
        tnew.sec = std::min(tnew.sec, told.sec);
    }

    // Store the new totals.

    m_timerTotals = totals;
}

//------------------------------------------------------------------------
