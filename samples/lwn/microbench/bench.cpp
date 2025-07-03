/*
 * Copyright (c) 2015-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "bench.hpp"
#include "options.hpp"
#include <assert.h>

#include <stdio.h>
#include <math.h>

#include <vector>
#include <string>

// Benchmark cases
#include "cases/warp_lwlling.hpp"
#include "cases/cbf.hpp"
#include "cases/constantbind.hpp"
#include "cases/clear_buffer.hpp"
#include "cases/cpu_overhead.hpp"
#include "cases/fillrate.hpp"
#include "cases/fillrate_compute.hpp"
#include "cases/gpfifo.hpp"
#include "cases/kickoff.hpp"
#include "cases/shaderbind.hpp"
#include "cases/shaderperf.hpp"
#include "cases/tex.hpp"
#include "cases/trirate.hpp"
#include "cases/pool_flush.hpp"
#include "cases/tex_init.hpp"
#include "cases/malloc_perf.hpp"
#include "cases/multibind.hpp"
#include "cases/tiled_cache.hpp"
#include "cases/gpu_time.hpp"
#include "cases/drawtest.hpp"

using namespace LwnUtil;

struct GpuCounterDescr
{
    const char*    name;
    LWNcounterType type;
};

struct GpuCounter
{
    uint64_t value;
    uint64_t timestamp;
};

#define DEF_COUNTER(n) { #n, LWN_COUNTER_TYPE_ ##n }

static const GpuCounterDescr s_counters[] = {
    DEF_COUNTER(SAMPLES_PASSED),
    DEF_COUNTER(INPUT_VERTICES),
    DEF_COUNTER(INPUT_PRIMITIVES),
    DEF_COUNTER(VERTEX_SHADER_ILWOCATIONS),
    DEF_COUNTER(FRAGMENT_SHADER_ILWOCATIONS)
};

static const int NUM_GPU_COUNTERS = sizeof(s_counters)/sizeof(s_counters[0]);

class TestResult
{
private:
    std::string m_name;
    std::string m_units;
    double      m_value;
public:
    TestResult(const char* name, const char* units) :
        m_name(name),
        m_units(units)
    {
    }

    void setValue(double v)
    {
        m_value = v;
    }

    const char* name() const  { return m_name.c_str(); }
    const char* units() const { return m_units.c_str(); }
    double value() const      { return m_value; }
};

ResultPrinter::ResultPrinter()
{
}

ResultPrinter::~ResultPrinter()
{
}

ResultPrinterStdout::ResultPrinterStdout()
{
}
ResultPrinterStdout::~ResultPrinterStdout()
{
}

void ResultPrinterStdout::print(const char* test, double value, const char* units)
{
#if defined(LW_HOS)
    // Note: NN_LOG doesn't seem to support %f in DDK v0.5
    if (g_options.flags() & Options::REST_OUTPUT_BIT) {
        NN_LOG("[REST:, test_case = %s, numeric = %lld.%d, disposition = pass]\n", test, (long long)value, (int)((value - floor(value)) * 1000.0));
    } else {
        NN_LOG("Test '%s' result %lld.%d %s\n", test, (long long)value, (int)((value - floor(value)) * 1000.0), units);
    }
#else
    if (g_options.flags() & Options::REST_OUTPUT_BIT) {
        printf("[REST:, test_case = %s, numeric = %f, disposition = pass]\n", test, value);
    }
    else {
        printf("Test '%s' result %f %s\n", test, value, units);
        fflush(stdout);
    }
#endif
}

// Note: the implementation of SampleArray could just as well be
// inside ResultCollector.  However, it lwrrently uses STL and I
// didn't want to leak STL includes into the bench.hpp header file.
class SampleArray
{
private:
    std::vector<TestResult> m_tests;
public:
    void add(const TestResult& res)
    {
        m_tests.push_back(res);
    }

    TestResult& current()
    {
        return m_tests[m_tests.size()-1];
    }

    const std::vector<TestResult>& tests() const
    {
        return m_tests;
    }
};

ResultCollector::ResultCollector() :
    m_samples(new SampleArray())
{
}

ResultCollector::~ResultCollector()
{
    delete m_samples;
}

void ResultCollector::begin(const char* name, const char* units)
{
    m_samples->add(TestResult(name, units));
}

void ResultCollector::end(double value)
{
    m_samples->current().setValue(value);
}

void ResultCollector::print(ResultPrinter& printer)
{
    const std::vector<TestResult>& tests = m_samples->tests();

    for (std::vector<TestResult>::const_iterator t = tests.begin(); t != tests.end(); ++t) {
        printer.print(t->name(), t->value(), t->units());
    }
}

uint32_t BenchmarkCase::testProps() const
{
    return 0;
}

// Setup a command buffer that sets various pieces of state to default
// values
void BenchmarkContextLWN::setupDefaultState()
{
    m_defaultState = new LwnUtil::CompiledCmdBuf(m_device, m_contextPools->coherent(), 4096);

    LWNcommandBuffer* cmd = m_defaultState->cmd();
    m_defaultState->begin();

    lwnCommandBufferSetTiledCacheAction(cmd, LWN_TILED_CACHE_ACTION_DISABLE);
    lwnCommandBufferSetScissor(cmd, 0, 0, m_width, m_height);
    lwnCommandBufferSetViewport(cmd, 0, 0, m_width, m_height);

    m_descriptorPool->setPools(cmd);

    uint32_t colorDepthMask = RenderTarget::DEST_WRITE_DEPTH_BIT | RenderTarget::DEST_WRITE_COLOR_BIT;
    LwnUtil::RenderTarget::setColorDepthMode(cmd, colorDepthMask, false);

    LWNmultisampleState msaa;
    lwnMultisampleStateSetDefaults(&msaa);
    lwnCommandBufferBindMultisampleState(cmd, &msaa);

    LWNcolorState colorState;
    lwnColorStateSetDefaults(&colorState);
    lwnCommandBufferBindColorState(cmd, &colorState);

    LWNblendState blend;
    lwnBlendStateSetDefaults(&blend);
    lwnCommandBufferBindBlendState(cmd, &blend);

    m_defaultState->end();
}

void BenchmarkContextLWN::initDescriptorPool()
{
    // Use one shared descriptor pool across tests.  This is not reset
    // after each init/deinit of a benchmark case, so make sure you
    // have a big enough table to hold all allocated texture
    // descriptors.
    const int NUM_DESCRIPTORS = 256;

    int samplerSize = 0;
    int textureSize = 0;
    int numReservedSamplers = 0;
    int numReservedTextures = 0;
    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_SAMPLER_DESCRIPTOR_SIZE, &samplerSize);
    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_TEXTURE_DESCRIPTOR_SIZE, &textureSize);
    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_RESERVED_SAMPLER_DESCRIPTORS, &numReservedSamplers);
    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_RESERVED_TEXTURE_DESCRIPTORS, &numReservedTextures);

    size_t size = (NUM_DESCRIPTORS + numReservedTextures) * textureSize;
    size += (NUM_DESCRIPTORS + numReservedSamplers) * samplerSize;

    m_descriptorMemPool = new LwnUtil::DescBufferPool(m_device, NULL, size);
    m_descriptorPool = new LwnUtil::DescriptorPool(m_device, m_descriptorMemPool, NUM_DESCRIPTORS);
}

BenchmarkContextLWN::BenchmarkContextLWN(LWNdevice* dev,
                                         LWNnativeWindow nativeWindow,
                                         int w, int h) :
    m_device(dev),
    m_defaultState(nullptr),
    m_width(w),
    m_height(h),
    m_rtBufferIdx(0)
{
    m_queue = new LWNqueue;
    LWNqueueBuilder qb;
    lwnQueueBuilderSetDevice(&qb, m_device);
    lwnQueueBuilderSetDefaults(&qb);
    lwnQueueBuilderSetControlMemorySize(&qb, 256 << 10);
    lwnQueueInitialize(m_queue, &qb);

#if defined(LW_HOS)
    // Increase LWN queue timeout
    // This is necessary on HOS since the default timeout there is too short
    // which causes some subtests to fail by timeout.
    PFNLWNQUEUESETTIMEOUTLWX lwnQueueSetTimeoutLWX =
        (PFNLWNQUEUESETTIMEOUTLWX)lwnDeviceGetProcAddress(m_device, "lwnQueueSetTimeoutLWX");
    assert(lwnQueueSetTimeoutLWX);
    lwnQueueSetTimeoutLWX(m_queue, 5000000000ULL);
#endif

    m_timer = Timer::instance();

    const int CPU_CACHED_POOL_SIZE = 16*1024*1024;

    initDescriptorPool();

    m_contextPools   = new LwnUtil::Pools(m_device, m_descriptorPool, 32*1024*1024, 1024*1024, 0);
    m_testcasePools  = new LwnUtil::Pools(m_device, m_descriptorPool, 128*1024*1024, 96*1024*1024, CPU_CACHED_POOL_SIZE);

    const int counterBufSize = NUM_GPU_COUNTERS * sizeof(GpuCounter);
    m_counterBuf = new LwnUtil::Buffer(m_device, m_contextPools->coherent(), nullptr, counterBufSize,
                                       BUFFER_ALIGN_COUNTER_BIT);

    m_counterCmds0 = new LwnUtil::CompiledCmdBuf(m_device, m_contextPools->coherent(), 2048);
    m_counterCmds1 = new LwnUtil::CompiledCmdBuf(m_device, m_contextPools->coherent(), 2048);

    m_counterCmds0->begin();
    for (int i = 0; i < NUM_GPU_COUNTERS; i++) {
        lwnCommandBufferResetCounter(m_counterCmds0->cmd(), s_counters[i].type);
    }
    m_counterCmds0->end();

    m_counterCmds1->begin();
    for (int i = 0; i < NUM_GPU_COUNTERS; i++) {
        lwnCommandBufferReportCounter(m_counterCmds1->cmd(),
                                      s_counters[i].type,
                                      m_counterBuf->address() + i * sizeof(GpuCounter));
    }
    m_counterCmds1->end();

    setupDefaultState();

    m_renderTarget = new LwnUtil::RenderTarget(m_device,
                                               m_contextPools,
                                               m_width, m_height,
                                               0);

    // Create a window from color textures, connect to native window.
    LWNwindowBuilder windowBuilder;
    lwnWindowBuilderSetDevice(&windowBuilder, m_device);
    lwnWindowBuilderSetDefaults(&windowBuilder);
    lwnWindowBuilderSetTextures(&windowBuilder, 2, m_renderTarget->colorBuffers());
    lwnWindowBuilderSetNativeWindow(&windowBuilder, nativeWindow);
    lwnWindowInitialize(&m_window, &windowBuilder);

    lwnSyncInitialize(&m_textureAvailableSync, m_device);
}

BenchmarkContextLWN::~BenchmarkContextLWN()
{
    lwnQueueFinish(m_queue);

    delete m_defaultState;
    delete m_counterCmds0;
    delete m_counterCmds1;
    delete m_counterBuf;

    // Must finalize the LWNwindow before deleting its render target
    // textures.
    lwnWindowFinalize(&m_window);
    lwnSyncFinalize(&m_textureAvailableSync);

    delete m_renderTarget;

    delete m_descriptorMemPool;
    delete m_testcasePools;
    delete m_descriptorPool;
    delete m_contextPools;

    lwnQueueFinalize(m_queue);
    delete m_queue;
}

void BenchmarkContextLWN::runAll(ResultCollector* results)
{
    {
        BenchmarkTrirateLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkCBFLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkWarpLwllingLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkFillrateLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkFillrateComputeLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkConstantBindLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkClearBufferLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkShaderBindLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkCpuOverheadLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkKickoffLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkTextureLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkPoolFlushLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkGpfifoLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkTexInitLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkMallocPerfLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkMultiBindLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkTiledCacheLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkGpuTimeLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkDrawTestLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
    {
        BenchmarkShaderPerfLWN test(m_device, m_queue, m_testcasePools, m_width, m_height);
        run(results, &test);
    }
}

bool BenchmarkContextLWN::testRunnable(const BenchmarkCaseLWN *bench, int subtestIdx) const
{
    auto descr = bench->description(subtestIdx);

    if (!g_options.matchTestName(descr.name))
        return false;

#if !defined(LW_HOS)
    // Check if the test is marked to run by default only on HOS
    // (unless overridden on the command line).
    if (bench->testProps() & (uint32_t)BenchmarkCase::TEST_PROP_HOS_ONLY_BIT) {
        return g_options.flags() & Options::NO_PLATFORM_SKIP_BIT ? true : false;
    }
#endif
    return true;
}

void BenchmarkContextLWN::run(ResultCollector* collector, BenchmarkCaseLWN* bench)
{
    int       nLoops    = g_options.numLoops();
    int       nSubtests = bench->numSubtests();
    LWNqueue* queue     = m_queue;

    for (int loop = 0; loop < nLoops; loop++) {
        for (int subtestIdx = 0; subtestIdx < nSubtests; subtestIdx++) {

            if (!testRunnable(bench, subtestIdx)) {
                continue;
            }

            auto descr = bench->description(subtestIdx);
            PRINTF("Initialize benchmark case %s (%d/%d)\n", descr.name, subtestIdx+1, nSubtests);
            FLUSH_STDOUT();

            // Free everything in all pools
            m_testcasePools->freeAll();

            collector->begin(descr.name, descr.units);

            // Set a render target even in case we never flip to display.
            // This is what all tests will use as their default render
            // target unless they want to setup their own RT themselves.
            m_renderTarget->setTargets(m_queue, 0);

            m_defaultState->submit(queue);
            bench->init(subtestIdx);
            lwnQueueFinish(queue);

            uint64_t  startTime = m_timer->getTicks();
            const int numFrames = g_options.numFrames();
            uint32_t  drawFlags = 0;

            if (g_options.flags() & Options::FLIP_BIT) {
                drawFlags |= BenchmarkCase::DISPLAY_PRESENT_BIT;
            }

            if (g_options.flags() & Options::COLLECT_GPU_COUNTERS_BIT) {
                m_counterCmds0->submit(queue);
            }

            if (!(bench->testProps() & (uint32_t)BenchmarkCase::TEST_PROP_BYPASS_FRAME_LOOP_BIT)) {
                for (int i = 0; i < numFrames; i++) {
                    if (drawFlags & BenchmarkCase::DISPLAY_PRESENT_BIT) {
                        lwnWindowAcquireTexture(&m_window, &m_textureAvailableSync, &m_rtBufferIdx);
                        lwnQueueWaitSync(m_queue, &m_textureAvailableSync);

                        assert((uint32_t)m_rtBufferIdx < 2);
                        m_renderTarget->setTargets(m_queue, m_rtBufferIdx);
                    }

                    BenchmarkCaseLWN::DrawParamsLWN params;
                    params.flags    = drawFlags;
                    params.dstColor = m_renderTarget->colorBuffers()[m_rtBufferIdx];
                    params.dstDepth = m_renderTarget->depthBuffer();

                    bench->draw(&params);

                    if (drawFlags & BenchmarkCase::DISPLAY_PRESENT_BIT) {
                        lwnQueuePresentTexture(m_queue, &m_window, m_rtBufferIdx);

                        // flip() call is needed for Windows which still
                        // must call SwapBuffers().
                        flip();
                    }
                }
            } else {
                BenchmarkCaseLWN::DrawParamsLWN params;
                params.window       = &m_window;
                params.renderTarget = m_renderTarget;
                bench->draw(&params);
                lwnQueueFlush(m_queue);
            }

            if (g_options.flags() & Options::COLLECT_GPU_COUNTERS_BIT) {
                m_counterCmds1->submit(queue);
            }

            // This is required for timing but also to ensure counter writes have happened
            lwnQueueFinish(queue);

            uint64_t endTime = m_timer->getTicks();
            double   value   = bench->measuredValue(subtestIdx, m_timer->ticksToSecs(endTime - startTime));

            collector->end(value);

            if (g_options.flags() & Options::COLLECT_GPU_COUNTERS_BIT) {
                const GpuCounter* p = (const GpuCounter*)m_counterBuf->ptr();
                for (int i = 0; i < NUM_GPU_COUNTERS; i++) {
                PRINTF("GPU counter %30s: %f M / frame\n",
                       s_counters[i].name,
                       (double)(p[i].value / numFrames) / 1000000.0);
                }
            }

            bench->deinit(subtestIdx);
        }
    }
}

BenchmarkCase::BenchmarkCase(int w, int h) : m_width(w), m_height(h)
{
}

BenchmarkCase::~BenchmarkCase()
{
}

BenchmarkCaseLWN::BenchmarkCaseLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCase(w, h),
    m_device(dev),
    m_queue(q),
    m_pools(pools)
{
}

BenchmarkCaseLWN::~BenchmarkCaseLWN()
{
}
