/*
 * Copyright (c) 2017 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include "lwntest_cpp.h"
#include "lwn_utils.h"

#include <time.h>

//////////////////////////////////////////////////////////////////////////

using namespace lwn;

#define _ALIGN_UP(v, gran)   (((v) + ((gran) - 1)) & ~((gran)-1))

#ifdef LW_WINDOWS
#include <windows.h>
#define milliSecondSleep(x)    Sleep(x)
#else
#include <sys/time.h>
#include <unistd.h>
#define milliSecondSleep(x)    usleep(1000 * (x))
#endif

class LWNeventTest
{
public:
    LWNeventTest()
    {}

    LWNTEST_CppMethods();
private:
};

int LWNeventTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(53, 204);
}

lwString LWNeventTest::getDescription() const
{
    return "Test events API: test uses CPU/GPU signal/wait to verify\n"
           "that events work as expected. It creates 3 runs to test\n"
           "GPU_UNCACHED|CPU_UNCACHED, GPU_UNCACHED|CPU_CACHED and\n"
           "GPU_CACHED|CPU_NO_ACCESS pools.\n"
           "In these pools it creates a number of events.\n"
           "It then uses the API to test the basic functionality with different\n"
           "signal and wait pairs and clears to GREEN on pass, RED on fail\n";
}

void LWNeventTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    int supportsReductionOps;
    device->GetInteger(lwn::DeviceInfo::EVENTS_SUPPORT_REDUCTION_OPERATIONS, &supportsReductionOps);

    const int numEvents = 32;
    const int eventSize = sizeof(uint32_t);
    EventBuilder eventBuilder;

    const MemoryPoolFlags poolTypes[] =
    {
        MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_UNCACHED,
        MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED,
#if defined(LW_TEGRA)
        // memory pools on the reference platform should be used
        // with care. This test can't support this case since it
        // has not secondary queue to flush/ilwalidate mapped ranges
        // which uses the GPU to copy between internal pitch and
        // CPU memory of the pool.
        MemoryPoolFlags::CPU_CACHED| MemoryPoolFlags::GPU_UNCACHED
#endif
    };

    EventSignalLocation locations[] =
    {
        EventSignalLocation::TOP,
        EventSignalLocation::VERTEX_PIPE,
        EventSignalLocation::BOTTOM
    };

    EventWaitMode waitModes[] =
    {
        EventWaitMode::EQUAL,
        EventWaitMode::GEQUAL_WRAP,
    };

    bool pass = true;
    Event* events[numEvents];
    for (size_t pt = 0; pt < __GL_ARRAYSIZE(poolTypes); pt++) {
        int alignedSize = _ALIGN_UP(numEvents * eventSize, LWN_MEMORY_POOL_STORAGE_GRANULARITY);
        MemoryPool *eventPool = device->CreateMemoryPoolWithFlags(NULL, alignedSize, poolTypes[pt]);

        uint32_t payload = 0x0ABCA2000;
        uint32_t payloadAdd = 0x10;
        for (int i = 0; i < numEvents; i++) {
            eventBuilder.SetDefaults().
                         SetStorage(eventPool, i * eventSize);
            events[i] = eventBuilder.CreateEvent();

            // Check sanity of GetMemoryPool & GetMemoryOffset
            if (eventBuilder.GetMemoryPool() != eventPool) {
                pass = false;
            } else if(eventBuilder.GetMemoryOffset() != i * eventSize) {
                pass = false;
            } else if(events[i]->GetMemoryPool() != eventPool) {
                pass = false;
            } else if (events[i]->GetMemoryOffset() != i * eventSize) {
                pass = false;
            }
        }

        for (int i = 0; i < numEvents; i++) {
            int timeout;
            if (!(poolTypes[pt] & MemoryPoolFlags::CPU_NO_ACCESS)) {
                // signal with CPU
                events[i]->Signal( EventSignalMode::WRITE, payload);
                if ((poolTypes[pt] & MemoryPoolFlags::CPU_CACHED)) {
                    eventPool->FlushMappedRange(i * eventSize, eventSize);
                }
                if (payload != events[i]->GetValue()) {
                    pass = false;
                    break;
                }

                // wait on GPU
                queueCB.WaitEvent(events[i], EventWaitMode::EQUAL, payload);
                queueCB.submit();
                queue->Finish();

                if (supportsReductionOps) {
                    events[i]->Signal( EventSignalMode::ADD, payloadAdd);
                    eventPool->FlushMappedRange(i * eventSize, eventSize);
                    if ((payload + payloadAdd) != events[i]->GetValue()) {
                        pass = false;
                        break;
                    }

                    // wait on GPU
                    queueCB.WaitEvent(events[i], waitModes[i % __GL_ARRAYSIZE(waitModes)], payload + payloadAdd);
                    queueCB.submit();
                    queue->Finish();
                }
                payload += payloadAdd + 1;

                // first flush GPU wait, then signal on CPU
                // wait on GPU
                queueCB.WaitEvent(events[i], waitModes[i % __GL_ARRAYSIZE(waitModes)], payload);
                queueCB.submit();
                queue->Flush();

                events[i]->Signal( EventSignalMode::WRITE, payload);
                if ((poolTypes[pt] & MemoryPoolFlags::CPU_CACHED)) {
                    eventPool->FlushMappedRange(i * eventSize, eventSize);
                }

                payload += payloadAdd + 1;

                // signal on GPU
                queueCB.SignalEvent(events[i], EventSignalMode::WRITE,
                                    locations[i % __GL_ARRAYSIZE(locations)], 0, payload);
                queueCB.submit();
                queue->Flush();

                // poll wait on CPU
                timeout = 0;
                do {
                    if ((poolTypes[pt] & MemoryPoolFlags::CPU_CACHED)) {
                        eventPool->IlwalidateMappedRange(i * eventSize, eventSize);
                    }
                    if (payload == events[i]->GetValue()) {
                        break;
                    }
                    milliSecondSleep(1);
                } while(timeout++ < 10);
                if (timeout >= 10) {
                    pass = false;
                }

                if (supportsReductionOps) {

                    // signal on GPU with ADD mode
                    queueCB.SignalEvent(events[i], EventSignalMode::ADD,
                                        locations[i % __GL_ARRAYSIZE(locations)], 0, payloadAdd);
                    queueCB.submit();
                    queue->Flush();

                    // poll wait on CPU
                    timeout = 0;
                    do {
                        if ((poolTypes[pt] & MemoryPoolFlags::CPU_CACHED)) {
                            eventPool->IlwalidateMappedRange(i * eventSize, eventSize);
                        }
                        if ((payload + payloadAdd) == events[i]->GetValue()) {
                            break;
                        }
                        milliSecondSleep(1);
                    } while(timeout++ < 10);
                    if (timeout >= 10) {
                        pass = false;
                    }
                }

                payload += payloadAdd + 1;
            }

            // signal on GPU
            queueCB.SignalEvent(events[i], EventSignalMode::WRITE,
                                locations[i % __GL_ARRAYSIZE(locations)], 0, payload);
            // wait on GPU
            queueCB.WaitEvent(events[i], waitModes[i % __GL_ARRAYSIZE(waitModes)], payload);

            if (supportsReductionOps) {
                // signal on GPU with ADD
                queueCB.SignalEvent(events[i], EventSignalMode::ADD,
                                    locations[i % __GL_ARRAYSIZE(locations)], 0, payloadAdd);
                // wait on GPU
                queueCB.WaitEvent(events[i], waitModes[i % __GL_ARRAYSIZE(waitModes)], payload + payloadAdd);
            }

            queueCB.submit();
            queue->Finish();

            payload += payloadAdd + 1;
        }

        for (int i = 0; i < numEvents; i++) {
            events[i]->Free();
        }

        eventPool->Free();
    }

    if (pass) {
        queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0, LWN_CLEAR_COLOR_MASK_RGBA);
    } else {
        queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0, LWN_CLEAR_COLOR_MASK_RGBA);
    }

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNeventTest, lwn_event,);
