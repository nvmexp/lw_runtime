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

#ifdef LW_WINDOWS
#include <windows.h>
#define milliSecondSleep(x)    Sleep(x)
#else
#include <sys/time.h>
#include <unistd.h>
#define milliSecondSleep(x)    usleep(1000 * (x))
#endif

#include <numeric>

#define VERBOSE 0

// This test is sensitive to scheduler jitter, so it is split in multiple rounds
// to make random failures unlikely.
//
// We run NUM_ITERATIONS of length ITERATION_IN_MS each, and sample GPU and CPU
// timestamp at start and end. We compare the deltas on CPU and GPU to each
// other, and the test succeeds if all of the following conditions are met:
//
// - the average sampled delta on CPU and GPU doesn't differ by more than
//   MAX_AVERAGE_ERROR %
// - the delta with the lowest difference has a difference of no more than
//   MAX_MINIMUM_ERROR %
//
// We sample the lowest difference, so we can use a tighter bound on CPU/GPU
// differences than the average permits.  Using the minimum instead of the
// maximum also potentially filters out abnormally large differences caused by
// scheduling issues (assuming they don't blow out the average too much).

#define ITERATION_IN_MS 500
#define NUM_ITERATIONS 5
#define MAX_AVERAGE_ERROR 10
#define MAX_MINIMUM_ERROR 5

#if VERBOSE
#define DP(fmt, ...) printf(fmt "\n", __VA_ARGS__)
#else
#define DP(fmt, ...)
#endif

using namespace lwn;

class LWNTimestampColwersionTest
{
public:
    LWNTEST_CppMethods();

private:
    struct TimeStampBundle
    {
        uint64_t cpuTs;
        uint64_t gpuTs;
        uint64_t gpuTsRaw;
    };

    void initialize() const;
    void reportResult(bool result) const;
    void finish() const;

    void getBundle(TimeStampBundle & result, bool timestampFromTop) const;
    LWNuint sampleDeltaError(bool timestampFromTop) const;
    std::vector<LWNuint> sampleAllDeltaErrors(bool timestampFromTop) const;
    bool validateAverageError(std::vector<LWNuint> & errors) const;
    bool validateMinimumError(std::vector<LWNuint> & errors) const;
    bool testDeltaError(bool timestampFromTop) const;

    static MemoryPoolAllocator * m_allocator;
};

MemoryPoolAllocator * LWNTimestampColwersionTest::m_allocator;

lwString LWNTimestampColwersionTest::getDescription() const
{
    return "Test checking whether the timestamps returned by the counter "
        "objects have the correct colwersions applied. Those are necessary on "
        "TX1 due to a clock configuration inconsistency, and should be a NOP "
        "everywhere else.";
}

int LWNTimestampColwersionTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(40, 13);
}

void LWNTimestampColwersionTest::initialize() const
{
    QueueCommandBuffer &commandBuffer = *g_lwnQueueCB;
    Device *device = DeviceState::GetActive()->getDevice();

    commandBuffer.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    commandBuffer.submit();

    m_allocator = new MemoryPoolAllocator(device, NULL, 0x1000,
        LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
}

void LWNTimestampColwersionTest::reportResult(bool result) const
{
    QueueCommandBuffer &commandBuffer = *g_lwnQueueCB;

    DP("result: %d", result);

    if (result)
        commandBuffer.ClearColor(0, 0.0, 1.0, 0.0, 0.0);
    else
        commandBuffer.ClearColor(0, 1.0, 0.0, 0.0, 0.0);

    commandBuffer.submit();
}

void LWNTimestampColwersionTest::finish() const
{
    Queue *queue = DeviceState::GetActive()->getQueue();
    QueueCommandBuffer &queueCommandBuffer = DeviceState::GetActive()->getQueueCB();

    queueCommandBuffer.submit();
    queue->Finish();

    delete m_allocator;
}

void LWNTimestampColwersionTest::getBundle(
    LWNTimestampColwersionTest::TimeStampBundle & result,
    bool timestampFromTop) const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCommandBuffer = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();
    BufferBuilder bufferBuilder;
    bufferBuilder.SetDevice(device).SetDefaults();

    Buffer *timestampBuffer = m_allocator->allocBuffer(&bufferBuilder,
        BUFFER_ALIGN_COUNTER_BIT, sizeof(CounterData));
    BufferAddress counterAddress = timestampBuffer->GetAddress();
    CounterData * counters = static_cast<CounterData *>(timestampBuffer->Map());

    CounterType counterType = timestampFromTop ?
        CounterType::TIMESTAMP_TOP :
        CounterType::TIMESTAMP;

    queueCommandBuffer.ReportCounter(counterType, counterAddress);
    queueCommandBuffer.submit();
    queue->Finish();

    // The CPU timer is read as a value/frequency pair, which we try to
    // colwert to nanoseconds without overflowing or losing too much
    // precision.
    uint64_t cpuTicks = lwogGetTimerValue();
    uint64_t cpuFrequency = lwogGetTimerFrequency();
    uint64_t cpuSeconds = cpuTicks / cpuFrequency;
    uint64_t cpuSubSeconds = cpuTicks % cpuFrequency;
    uint64_t cpuNanoseconds = (cpuSeconds * 1000000000ULL +
                               uint64_t(double(cpuSubSeconds) / double(cpuFrequency) * 1E9));

    result.cpuTs = cpuNanoseconds;
    result.gpuTsRaw = counters[0].timestamp;
    result.gpuTs = device->GetTimestampInNanoseconds(&counters[0]);
    DP("Raw timer: CPU %llu GPURAW %llu GPU %llu FACTOR %f DIFF %llu",
        result.cpuTs, result.gpuTsRaw, result.gpuTs,
        (float) result.gpuTs / (float) result.gpuTsRaw,
        result.cpuTs - result.gpuTs);
}

LWNuint LWNTimestampColwersionTest::sampleDeltaError(bool timestampFromTop) const
{
    TimeStampBundle pairBefore, pairAfter;
    uint64_t deltaCpu, deltaGpu, error;

    getBundle(pairBefore, timestampFromTop);

    milliSecondSleep(ITERATION_IN_MS);

    getBundle(pairAfter, timestampFromTop);

    deltaCpu = pairAfter.cpuTs - pairBefore.cpuTs;
    deltaGpu = pairAfter.gpuTs - pairBefore.gpuTs;
    error = labs(int64_t(deltaGpu / (deltaCpu / 100) - 100));

    DP("Delta: GPU %llu, CPU %llu, error %llu%%, top %d", deltaGpu, deltaCpu, error,
        timestampFromTop);

    return error;
}

std::vector<LWNuint> LWNTimestampColwersionTest::sampleAllDeltaErrors(
    bool timestampFromTop) const
{
    std::vector<LWNuint> deltas(NUM_ITERATIONS);

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        deltas[i] = sampleDeltaError(timestampFromTop);
    }

    return deltas;
}

bool LWNTimestampColwersionTest::validateAverageError(
    std::vector<LWNuint> & errors) const
{
    return std::accumulate(errors.begin(), errors.end(), 0) / errors.size()
        <= MAX_AVERAGE_ERROR;
}

bool LWNTimestampColwersionTest::validateMinimumError(
    std::vector<LWNuint> & errors) const
{
    return *std::min_element(errors.begin(), errors.end()) <= MAX_MINIMUM_ERROR;
}

bool LWNTimestampColwersionTest::testDeltaError(bool timestampFromTop) const
{
    bool result = true;
    std::vector<LWNuint> deltaErrors;

    deltaErrors = sampleAllDeltaErrors(timestampFromTop);

    result &= validateAverageError(deltaErrors);
    result &= validateMinimumError(deltaErrors);

    return result;
}

void LWNTimestampColwersionTest::doGraphics() const
{
    bool result = true;

    DP("Initial CPU timer %llu, frequency %llu", lwogGetTimerValue(), lwogGetTimerFrequency());

    initialize();

    result &= testDeltaError(true);
    result &= testDeltaError(false);

    reportResult(result);

    finish();
}

OGTEST_CppTest(LWNTimestampColwersionTest, lwn_timestamp_colwersion,);
