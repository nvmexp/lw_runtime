/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

using namespace lwn;

class LWNClearBuffer
{
public:
    LWNTEST_CppMethods();
};

lwString LWNClearBuffer::getDescription() const
{
    lwStringBuf sb;
    sb << "Test the ClearBuffer API in LWN. Tests clearing GPU buffers in the following way:\n\n"
        " * Create a buffer\n"
        " * Fill it first using the CPU, then by repeated ClearBuffer calls\n"
        " * Test that the expected values got written into the buffer\n\n"
        "The test checks that the buffer was cleared using the expected values and outputs\n"
        "a series of green or red squares.  Green means pass and red means failure.\n";
    return sb.str();
}

int LWNClearBuffer::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 23);
}

static uint32_t clearPattern(int idx)
{
    return idx * 0x04030201;
}

static bool patternTest()
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Issue N ClearBuffer calls at increasing sizes, leaving one dword untouched
    // between each clear.
    //
    // The contents of the buffer will look something like this:
    //
    // idx|0123456789ABCDEF...
    // ---+--------------------
    // val|X1X22X333X4444X5...X
    //
    // Each 'X slot is cleared to value UNCLEARED on the CPU and it is not
    // expected to change as a result of any of the ClearBuffer calls.  The
    // slots not marked with 'X' are cleared by ClearBuffer to value
    // 'clearPattern(index_of_the_clear)'.
    const uint32_t UNCLEARED = 0x88888888;
    const int N = 16;

    // Callwlate the space required for our destination buffer:
    //
    // - one dword to pad the last dword of the buffer
    // - one dword prefix per each ClearBuffer subregion
    // - n_i dwords per a ClearBuffer subregion
    //
    // So: 1 + N + sum([1..N])
    int numDwords = 1 + N + N*(N+1)/2;
    const int bufStorageSize = numDwords * 4;

    MemoryPoolAllocator bufAllocator(device, NULL, bufStorageSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *dstBuf = bufAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, bufStorageSize);
    assert(dstBuf);

    uint32_t *dstAddr = (uint32_t *)dstBuf->Map();
    for (int i = 0; i < numDwords; i++) {
        dstAddr[i] = UNCLEARED;
    }

    // Issue N command buffer clears.  Increase the clear length by one dword
    // in each iteration.
    //
    // Note: Start clears at length 0 to test that a ClearBuffer call with
    // zero doesn't issue a debug validation error or other adverse effects.
    for (int n = 0, dstOffset = 0; n <= N; n++) {
        queueCB.ClearBuffer(dstBuf->GetAddress() + dstOffset * 4, n * 4, clearPattern(n));
        dstOffset += n + 1;
    }
    queueCB.submit();
    queue->Finish();

    const uint32_t *src = dstAddr;
    bool passed = true;
    for (int n = 1; n <= N; n++) {
        if (*src != UNCLEARED) {
            passed = false;
            goto fail;
        }

        src++;
        uint32_t pattern = clearPattern(n);
        for (int i = 0; i < n; i++, src++) {
            if (*src != pattern) {
                passed = false;
                goto fail;
            }
        }
    }
    if (*src != UNCLEARED) {
        passed = false;
    }
    src++;

    assert(int(src - dstAddr) == numDwords);

fail:
    bufAllocator.freeBuffer(dstBuf);
    return passed;
}

// Fill in a large (> 64K) buffer and test that it got correctly cleared. This
// is mainly to test that we don't bump against some max 16-bit width limits
// or such.
static bool largeFillTest()
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const uint32_t UNCLEARED = 0x88888888;
    const int NUM_DWORDS = 1024*1024;
    const int bufStorageSize = NUM_DWORDS * 4;

    MemoryPoolAllocator bufAllocator(device, NULL, bufStorageSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *dstBuf = bufAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, bufStorageSize);
    assert(dstBuf);

    uint32_t *dstAddr = (uint32_t *)dstBuf->Map();
    for (int i = 0; i < NUM_DWORDS; i++) {
        dstAddr[i] = UNCLEARED;
    }

    const uint32_t MAGIC = 0xdeadbeef;
    queueCB.ClearBuffer(dstBuf->GetAddress() + 4, (NUM_DWORDS - 2) * 4, MAGIC);

    queueCB.submit();
    queue->Finish();

    const uint32_t *src = dstAddr;
    bool passed = true;

    passed = *src == UNCLEARED;
    src++;

    for (int i = 0; i < NUM_DWORDS - 2; i++, src++) {
        if (*src != MAGIC) {
            passed = false;
            break;
        }
    }
    if (*src != UNCLEARED) {
        passed = false;
    }

    bufAllocator.freeBuffer(dstBuf);
    return passed;
}

void LWNClearBuffer::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const int cellWidth = 20;
    const int cellHeight = 20;
    int cellNum = 0;
    cellTestInit(cellWidth, cellHeight);

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    bool ok = patternTest();
    SetCellViewportScissorPadded(queueCB, cellNum % cellWidth, cellNum / cellWidth, 1);
    queueCB.ClearColor(0, ok ? 0.0 : 1.f, ok ? 1.f : 0.0, 0.0);
    cellNum++;

    ok = largeFillTest();
    SetCellViewportScissorPadded(queueCB, cellNum % cellWidth, cellNum / cellWidth, 1);
    queueCB.ClearColor(0, ok ? 0.0 : 1.f, ok ? 1.f : 0.0, 0.0);
    cellNum++;

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNClearBuffer, lwn_clear_buffer, );
