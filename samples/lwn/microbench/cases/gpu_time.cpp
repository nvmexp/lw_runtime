/*
 * Copyright (c) 2016-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// GPU time test
//
// Measures how much GPU work we can do between frame start and vsync
// to reliably hit 60 fps.  Ie., what is the maximum amount of GPU
// work we can perform to not start missing frames.
//
// SOL for this test should be 1/fps, ie. 16.6667 ms at 60 fps.  On
// 2016-06-17 in HR14 HOS, we can render roughly 16.02 ms to stay
// within vsync limit.

#include "gpu_time.hpp"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "lwnUtil/lwnUtil_TiledCacheState.h"

struct GpuCounters
{
    LWNcounterData gpuTime0;
    LWNcounterData gpuTime1;
};

static const BenchmarkGpuTimeLWN::TestDescr subtests[] = {
    { 0 }
};

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::Vec3i;
using LwnUtil::Vec4f;
using LwnUtil::RenderTarget;
using LwnUtil::Timer;

const int GRID_X = 1024;
const int GRID_Y = 512;

static const char *VS_STRING =
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "layout(location = 0) in vec3 position;\n"
    "out IO { vec4 vtxcol; };\n"
    "void main() {\n"
    "  gl_Position = vec4(position, 1.0);\n"
    "}\n";

// Version of the fragment shader using bound textures.
static const char *FS_STRING =
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "layout(location = 0) out vec4 color;\n"
    "void main() {\n"
    "  color = vec4(0.3, 0.4, 0.5, 1.);\n"
    "}\n";

BenchmarkGpuTimeLWN::BenchmarkGpuTimeLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h)
{
}

int BenchmarkGpuTimeLWN::numSubtests() const
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

uint32_t BenchmarkGpuTimeLWN::testProps() const
{
    return (uint32_t)TEST_PROP_BYPASS_FRAME_LOOP_BIT | (uint32_t)TEST_PROP_HOS_ONLY_BIT;
}

BenchmarkCase::Description BenchmarkGpuTimeLWN::description(int subtest) const
{
    static char testName[256];

    sprintf(testName, "gpu_time");

    Description d;
    d.name  = testName;
    d.units = "ms";
    return d;
}

void BenchmarkGpuTimeLWN::init(int subtest)
{
    m_res = new Resources();

    const TestDescr& testDescr = subtests[subtest];
    m_testDescr = &testDescr;

    m_numInstancesRendered = 0;

    m_res->cmdBuf.reset(new LwnUtil::CmdBuf(device(), queue(), coherentPool(), 32, 65536*4, 16384));

    const int counterBufSize = sizeof(GpuCounters);
    m_res->counters.reset(new LwnUtil::Buffer(device(), coherentPool(), nullptr, counterBufSize,
                                       BUFFER_ALIGN_COUNTER_BIT));


    LWNcommandBuffer* cmd = m_res->cmdBuf->cmd();
    lwnCommandBufferBeginRecording(cmd);

    // Create programs from the device, provide them shader code and compile/link them
    lwnProgramInitialize(&m_pgm, device());

    LWNshaderStage stages[2] = { LWN_SHADER_STAGE_VERTEX, LWN_SHADER_STAGE_FRAGMENT };
    const char *sources[2]   = { VS_STRING, FS_STRING };
    int32_t nSources = 2;

    if (!LwnUtil::compileAndSetShaders(&m_pgm, stages, nSources, sources))
    {
        assert(0); // TBD
    }

    m_res->vertex.reset(new LwnUtil::VertexState);
    m_res->vertex->setAttribute(0, LWN_FORMAT_RGB32F, 0, 0);
    m_res->vertex->setStream(0, 12);

    m_res->mesh.reset(LwnUtil::Mesh::createGrid(device(), coherentPool(), GRID_X, GRID_Y, Vec2f(-0.5f, -0.5f), Vec2f(1.5f, 1.5f), 1.f));

    float clearColor[] = { 0, 0, 0, 1 };
    lwnCommandBufferClearColor(cmd, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    lwnCommandBufferClearDepthStencil(cmd, 1.0, LWN_TRUE, 0, 0);
    lwnCommandBufferBindProgram(cmd, &m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);
    m_res->vertex->bind(cmd);
    lwnCommandBufferBindVertexBuffer(cmd, 0, m_res->mesh->vboAddress(), m_res->mesh->numVertices()*sizeof(Vec3f));

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(cmd);
    m_res->cmdBuf->submit(1, &cmdHandle);

    lwnSyncInitialize(&m_textureAvailableSync, device());
}


void BenchmarkGpuTimeLWN::renderAndFlip(const DrawParamsLWN* drawParams, int* renderTargetIdx, int numTriangles)
{
    // Acquire & bind render target
    lwnWindowAcquireTexture(drawParams->window, &m_textureAvailableSync, renderTargetIdx);
    lwnQueueWaitSync(queue(), &m_textureAvailableSync);

    // *** TIME THIS *** should be ~0.5ms
    lwnQueueFinish(queue());

    // GPU idle, vsync just happened
    drawParams->renderTarget->setTargets(queue(), *renderTargetIdx);

    // Draw
    LWNcommandBuffer* cmd = m_res->cmdBuf->cmd();
    lwnCommandBufferBeginRecording(cmd);

    lwnCommandBufferReportCounter(cmd, LWN_COUNTER_TYPE_TIMESTAMP, m_res->counters->address() + offsetof(GpuCounters, gpuTime0));

    LWNbufferAddress iboAddress    = m_res->mesh->iboAddress();

    int numDraws      = numTriangles / m_res->mesh->numTriangles();
    int triRemainder  = numTriangles % m_res->mesh->numTriangles();

    for (int drawIdx = 0; drawIdx < numDraws; drawIdx++) {
        lwnCommandBufferDrawElements(cmd, LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT,
                                     m_res->mesh->numTriangles()*3, iboAddress);
    }
    lwnCommandBufferDrawElements(cmd, LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT,
                                 triRemainder*3, iboAddress);

    lwnCommandBufferReportCounter(cmd, LWN_COUNTER_TYPE_TIMESTAMP, m_res->counters->address() + offsetof(GpuCounters, gpuTime1));

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(cmd);
    m_res->cmdBuf->submit(1, &cmdHandle);

    // Flip
    lwnQueuePresentTexture(queue(), drawParams->window, *renderTargetIdx);

    // TODO if this is removed, must do something to double buffer the
    // report counter update/reads.
    lwnQueueFinish(queue());
}

void BenchmarkGpuTimeLWN::timedRender(const BenchmarkCaseLWN::DrawParamsLWN *params, int *renderTargetIdx, int numTriangles, float *cpuTime, float *gpuTime)
{
    Timer *timer = Timer::instance();

    const GpuCounters *counterVA = (const GpuCounters *)m_res->counters->ptr();
    uint64_t startTime = timer->getTicks();
    renderAndFlip(params, renderTargetIdx, numTriangles);
    uint64_t endTime = timer->getTicks();

    uint64_t gpuTimeNs0 = lwnDeviceGetTimestampInNanoseconds(device(), &counterVA->gpuTime0);
    uint64_t gpuTimeNs1 = lwnDeviceGetTimestampInNanoseconds(device(), &counterVA->gpuTime1);
    uint64_t gpuTimeNs  = gpuTimeNs1 - gpuTimeNs0;

    *cpuTime = (float)(timer->ticksToSecs(endTime - startTime));
    *gpuTime = gpuTimeNs / 1000000000.f;
}

static float average(const float *arr, int size)
{
    float sum = 0.f;

    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum / (float)size;
}

void BenchmarkGpuTimeLWN::draw(const DrawParams* drawParams)
{
    BenchmarkCaseLWN::DrawParamsLWN* params = (BenchmarkCaseLWN::DrawParamsLWN*)drawParams;

    m_numInstancesRendered++;

    int renderTargetIdx = 0;

    const int maxTriangles = m_res->mesh->numTriangles() * 8;
    int stepSize = maxTriangles / 128;

    bool  firstMissedVsync = false;
    float minAvgGpuTime = 0.f;
    float maxAvgGpuTime = 0.f;

    const int warmUpFrames = 10;
    for (int i = 0; i < warmUpFrames; i++) {
        float cpuTime, gpuTime;
        timedRender(params, &renderTargetIdx, m_res->mesh->numTriangles(), &cpuTime, &gpuTime);
    }

    for (int numTriangles = m_res->mesh->numTriangles()/2; numTriangles < maxTriangles; numTriangles += stepSize) {
        const int NUM_SAMPLES = 8;

        float cpuTime[NUM_SAMPLES];
        float gpuTime[NUM_SAMPLES];

        int missedVsync = 0;

        for (int sidx = 0; sidx < NUM_SAMPLES; sidx++) {
            timedRender(params, &renderTargetIdx, numTriangles, &cpuTime[sidx], &gpuTime[sidx]);

            // Less than vsync?
            if (cpuTime[sidx] >= 17.5f / 1000.f) {
                missedVsync++;
            }
        }

        if (missedVsync >= NUM_SAMPLES/4) {
            firstMissedVsync = true;
        }

        if (missedVsync == 0 && !firstMissedVsync) {
            minAvgGpuTime = average(gpuTime, NUM_SAMPLES);
        }

        // All frames missed vsync, we're done
        if (missedVsync == NUM_SAMPLES) {
            maxAvgGpuTime = average(gpuTime, NUM_SAMPLES);
            break;
        }

        int intFrameTime = (int)(gpuTime[0] * 1000.f);
        // Slow down step size after 13 ms/frame
        int cutoff = 13;
        if (intFrameTime >= cutoff) {
            stepSize = maxTriangles / (128 << (intFrameTime - cutoff + 2));
        }
    }

    m_minGpuTime = minAvgGpuTime;

    printf("** GPU time that always hits vsync budget: %f\n", minAvgGpuTime * 1000.f);
    printf("** GPU time is just above vsync budget:    %f\n", maxAvgGpuTime * 1000.f);
}

double BenchmarkGpuTimeLWN::measuredValue(int subtest, double elapsedTime)
{
    // Unfortunately we don't have a good way of returning multiple
    // benchmark results from one subtest of a benchmark case.  So we
    // only display the "minimum GPU to before we start missing vsync".
    return m_minGpuTime * 1000.f;
}

void BenchmarkGpuTimeLWN::deinit(int subtest)
{
    delete m_res;
    lwnProgramFinalize(&m_pgm);
    lwnSyncFinalize(&m_textureAvailableSync);
}

BenchmarkGpuTimeLWN::~BenchmarkGpuTimeLWN()
{
}
