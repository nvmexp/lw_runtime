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

// Debug output control; set to 1 to print debug spew for test configurations
// with errors or 2 to spew globally.
#define LWN_TIMESTAMP_LOG_OUTPUT      0

#if LWN_TIMESTAMP_LOG_OUTPUT >= 2
#define SHOULD_LOG(_result)     true
#define LOG(x)                  printf x
#define LOG_INFO(x)             printf x
#elif LWN_TIMESTAMP_LOG_OUTPUT >= 1
#define SHOULD_LOG(_result)     (!(_result))
#define LOG(x)                  printf x
#define LOG_INFO(x)
#else
#define SHOULD_LOG(_result)     false
#define LOG(x)
#define LOG_INFO(x)
#endif

// Set to something nonzero to enable strict mode, where we don't tolerate any
// misbehavior. If you do this, it might be helpful to disable the system GPU
// timeslice on Odin devices with:
// TargetTools\NX-NXFP2-a64\DevMenuCommand\Release\DevMenuCommand.nsp -- debug disable-gpu-resource-limitation
// so you don't get spurious preemptions.
// See bug 200311823 for details.
#define LWN_TIMESTAMP_STRICT 0

using namespace lwn;

static uint64_t ticks2ns(uint64_t ticks, uint64_t freq)
{
    uint64_t seconds = ticks / freq;
    uint64_t sub_second = ticks % freq;
    return seconds * 1000000000ULL +
           uint64_t(double(sub_second) / double(freq) * 1E9);
}

class LWNTimestampTest
{
    static const int nInstances = 400;
    static const int nLoops = 40;
public:
    LWNTEST_CppMethods();
};

lwString LWNTimestampTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "This is a basic test for the TIMESTAMP and TIMESTAMP_TOP counter types.\n"
        "\n"
        "The test loops N = "<<nLoops<<" times, each time drawing a number of identical instances\n"
        "of a single triangle.\n"
        "\n"
        "For each loop, we report the TIMESTAMP_TOP counter before the draws and\n"
        "the TIMESTAMP counter after the draws.\n"
        "\n"
        "On the left edge of the window, we have four columns of cells drawn in green,\n"
        "red or blue, where each row is for a separate set of counters and draw calls,\n"
        "so we have N rows for N loops. A blue cell means that the test is not supported\n"
        "on the current platform.\n"
        "\n"
        "This also tests lwnDeviceGetLwrrentTimestampInNanoseconds in a fifth column.\n"
        "\n"
        "The four columns are drawn in red or green based on the following results:\n"
        "\n"
        "* Column 0: The end-of-loop timestamp is greater than the beginning-of-loop\n"
        "  timestamp.\n"
        "\n"
        "* Column 1: The beginning-of-loop timestamp is greater than the\n"
        "  beginning-of-loop timestamp of the previous loop.\n"
        "\n"
        "* Column 2: The end-of-loop timestamp is greater than the end-of-loop\n"
        "  timestamp of the previous loop.\n"
        "\n"
        "* Column 3: The beginning-of-loop timestamp is LESS than the end-of-loop\n"
        "  timestamp of the previous loop.  This verifies that TIMESTAMP_TOP is\n"
        "  reported without waiting on previous draws or bottom-of-pipe timestamps.\n"
        "  We only show failures in this column if > 10% of the iterations have no\n"
        "  overlap. Since preemptions (e.g. for the system timeslice) can oolwr at\n"
        "  any time, it's possible that for some small percentage of the loops\n"
        "  TIMESTAMP will be lower than TIMESTAMP_TOP. See bug 200311823 for details.\n"
        "\n"
        "* Column 4: It has only 4 cells corresponding to three checks on a\n"
        "  current-timestamp requested just after flushing the queue to the GPU:\n"
        "\n"
        "  - Cell 0: The call was a success, i.e. current-timestamp is strictly positive.\n"
        "\n"
        "  - Cell 1: current-timestamp is younger than the last TIMESTAMP.\n"
        "\n"
        "  - Cell 2: current-timestamp is older than the first TIMESTAMP_TOP.\n"
        "\n"
        "  - Cell 3: The call did not block.\n";
#if !defined(LW_TEGRA)
    sb <<
        "    The test is disabled on non-CheetAh platforms and renders in blue because the timing\n"
        "    of normal GPU operations might not meet the expectations of the logic determining\n"
        "    if the call blocked.\n";
#endif
    return sb.str();
}

int LWNTimestampTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(40, 5);
}

void LWNTimestampTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    static const double tTolerance = 1e-2;

    static const float red[]   = { 1.0, 0.0, 0.0, 1.0 };
    static const float green[] = { 0.0, 1.0, 0.0, 1.0 };

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 color;\n"
        "out vec3 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ocolor = color;\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(ocolor, 1.0);\n"
        "}\n";

    Program *pgm = device->CreateProgram();

    // Compile and call lwnProgramSetShaders.
    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        LWNFailTest();
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-0.6, -0.8, -0.8), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(+0.6, -0.8, -0.4), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(-0.6, +0.8, +0.8), dt::vec3(1.0, 0.0, 0.0) },
    };

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, 64*1024, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // Create a buffer for the vertex data.
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 3, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // One sync object to rule our time-reading activities on gpu/cpu.
    Sync *sync_top = device->CreateSync();

    // Create a buffer for the GPU-recorded counter data, with two CounterData
    // structures per loop.
    BufferBuilder cbb;
    cbb.SetDevice(device).SetDefaults();
    Buffer *counterBuffer = allocator.allocBuffer(&cbb, BUFFER_ALIGN_COUNTER_BIT, (nLoops * 2 + 1) * sizeof(CounterData));
    BufferAddress counterAddr = counterBuffer->GetAddress();
    CounterData *counterPtr = (CounterData *) counterBuffer->Map();

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.ClearDepthStencil(1.0, LWN_TRUE, 0, 0);
    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    // Render all the loops in the right side of the screen, reporting
    // timestamps before and after each iteration.
    queueCB.SetViewportScissor(lwrrentWindowWidth/4, 0, 3*lwrrentWindowWidth/4, lwrrentWindowHeight);
    for (int i = 0; i < nLoops; i++) {
        queueCB.ReportCounter(CounterType::TIMESTAMP_TOP, counterAddr + (2 * i + 0) * sizeof(CounterData));
        if (i == 0) {
            // This is the first loop, we signal to the CPU we have read the first time_top.
            queueCB.FenceSync(sync_top, SyncCondition::GRAPHICS_WORLD_SPACE_COMPLETE, 0 /*!FLUSH_FOR_CPU*/ );
        }

        queueCB.DrawArraysInstanced(DrawPrimitive::TRIANGLES, 0, 3, 0, nInstances);
        queueCB.ReportCounter(CounterType::TIMESTAMP, counterAddr + (2 * i + 1) * sizeof(CounterData));
    }

    // Send to GPU.
    queueCB.submit();
    queue->Flush();

    // Make sure the queue is in GPU. Note that we Flush()ed it.
    sync_top->Wait(LWN_WAIT_TIMEOUT_MAXIMUM);

    const uint64_t cpuTicks0 = lwogGetTimerValue();
    const uint64_t gpuNs = device->GetLwrrentTimestampInNanoseconds();
    const uint64_t cpuTicks1 = lwogGetTimerValue();

    // The GPU can read its last timestamp.
    queueCB.ReportCounter(CounterType::TIMESTAMP, counterAddr + (2 * nLoops) * sizeof(CounterData));
    queueCB.submit();
    queue->Flush();

    // Wait for everything to finish so we can check counters with the CPU.
    queue->Finish();

    // Are we forcing the loop overlap test to pass?
    bool loopOverlapMasterPass = false;

    // We check the loopOverlap counter before everything else so that we only show
    // loopOverlap failures when > 10% of the iterations fail.
    int loopOverlapFails = 0;
    for (int i = 1; i < nLoops; ++i) {
        bool loopOverlap = counterPtr[2 * i - 1].timestamp >= counterPtr[2 * i + 0].timestamp;
        if (!loopOverlap) {
            ++loopOverlapFails;
        }
    }
    if (SHOULD_LOG(loopOverlapFails > 0)) {
        LOG(("Loop overlap fails: %d\n", loopOverlapFails));
    }
    // As long as (loopOverlapFails / nLoops) < (1 / 10) we don't have a problem
    // with the occasional failure.
    loopOverlapMasterPass = (loopOverlapFails * 10 < nLoops);
#if LWN_TIMESTAMP_STRICT != 0
    loopOverlapMasterPass = false;
#endif

    int cellSize = LW_MIN(lwrrentWindowWidth/16, lwrrentWindowHeight/nLoops);
    for (int i = 0; i < nLoops; i++) {
        bool loopOrdered = true;
        bool timestampTopOrdered = true;
        bool timestampBottomOrdered = true;
        bool loopOverlap = true;

        // Compare timestamps within the iteration (<loopOrdered>) and
        // relative to the previous one (if available).  For <loopOverlap>, we
        // are expecting that the old loop's end timestamp will be newer than
        // the new loop's start timestamp, since TIMESTAMP_TOP shouldn't wait
        // for the completion of the previous loop.
        //
        // NOTE:  We now use ">=" tests here because some test systems
        // (GM107s) seem to intermittently show two loops with the same
        // bottom-of-the-pipe timestamp.  This is technically possible (the
        // timer isn't super-high resolution), but I had expected that drawing
        // sufficiently many instances of the triangle wouldn't have this
        // happen.
        loopOrdered = counterPtr[2*i+1].timestamp >= counterPtr[2*i+0].timestamp;
        if (i > 0) {
            timestampTopOrdered = counterPtr[2 * i + 0].timestamp >= counterPtr[2 * i - 2].timestamp;
            timestampBottomOrdered = counterPtr[2 * i + 1].timestamp >= counterPtr[2 * i - 1].timestamp;
            // If loopOverlapMasterPass is set, we don't alert on individual failures
            loopOverlap = loopOverlapMasterPass ||
                counterPtr[2 * i - 1].timestamp >= counterPtr[2 * i + 0].timestamp;
        }

        queueCB.SetViewportScissor(0 * cellSize + 1, i * cellSize + 1, cellSize - 2, cellSize - 2);
        queueCB.ClearColor(0, loopOrdered ? green : red, ClearColorMask::RGBA);
        queueCB.SetViewportScissor(1 * cellSize + 1, i * cellSize + 1, cellSize - 2, cellSize - 2);
        queueCB.ClearColor(0, timestampTopOrdered ? green : red, ClearColorMask::RGBA);
        queueCB.SetViewportScissor(2 * cellSize + 1, i * cellSize + 1, cellSize - 2, cellSize - 2);
        queueCB.ClearColor(0, timestampBottomOrdered ? green : red, ClearColorMask::RGBA);
        queueCB.SetViewportScissor(3 * cellSize + 1, i * cellSize + 1, cellSize - 2, cellSize - 2);
        queueCB.ClearColor(0, loopOverlap ? green : red, ClearColorMask::RGBA);

        if (SHOULD_LOG(true)) {
            LOG(("%d: ", i));
            for (int j = 0; j < 2; j++) {
                CounterData *data = counterPtr + 2 * i + j;
                uint64_t delta = data->timestamp - counterPtr[0].timestamp;
                LOG_INFO((" %08x%08x", LWNuint(delta >> 32), LWNuint(delta & 0xFFFFFFFFULL)));
                (void)delta;  // hide unused variable
            }
            LOG(("\n"));
        }
    }

    const uint64_t cpuFrequency = lwogGetTimerFrequency();

    // From here on, we test the quality of device->GetLwrrentTimestampInNanoseconds
    uint64_t cpuNs0 = ticks2ns(cpuTicks0, cpuFrequency);
    uint64_t cpuNs1 = ticks2ns(cpuTicks1, cpuFrequency);

    // Callwlate the time the GPU took to process the commands
    uint64_t gpu_time = device->GetTimestampInNanoseconds(&counterPtr[2 * nLoops - 1]) - device->GetTimestampInNanoseconds(&counterPtr[0]);

    int lwrrentColumn = 4 * cellSize + 1;
    int lwrrentRow = 1;

    // The call was a success, i.e. the return value is non-zero.
    bool callSuccess = gpuNs > 0;

    queueCB.SetViewportScissor(lwrrentColumn, lwrrentRow, cellSize - 2, cellSize - 2);
    lwrrentRow += cellSize;
    queueCB.ClearColor(0, callSuccess ? green : red, ClearColorMask::RGBA);

    // Make sure the call went straight to the HW bypassing the queue.
    // This bottom timestamp comes from a separate queue that we flushed
    // after our call to LwrrentTimestamp.
    int lastBotIdx = 2 * nLoops;
    uint64_t lastBottomNs = device->GetTimestampInNanoseconds(&counterPtr[lastBotIdx]);
    bool gotLwrrentBeforeLastBottom = gpuNs <= lastBottomNs;

    queueCB.SetViewportScissor(lwrrentColumn, lwrrentRow, cellSize - 2, cellSize - 2);
    lwrrentRow += cellSize;
    queueCB.ClearColor(0, gotLwrrentBeforeLastBottom ? green : red, ClearColorMask::RGBA);

    // The first timestamp was dished out before our call.
    uint64_t firstTopNs = device->GetTimestampInNanoseconds(&counterPtr[0]);
    bool gotLwrrentAfterFirstTop = gpuNs >= firstTopNs;

    queueCB.SetViewportScissor(lwrrentColumn, lwrrentRow, cellSize - 2, cellSize - 2);
    lwrrentRow += cellSize;
    queueCB.ClearColor(0, gotLwrrentAfterFirstTop ? green : red, ClearColorMask::RGBA);

    // Make sure the call to the timer was fast.
    // I.e. that it did not block for the duration of the queue,
    // but got quickly inserted and exelwted.
    double relativeDuration = double(cpuNs1 - cpuNs0) / double(gpu_time);
    bool callDurationWithinTolerances = relativeDuration <= tTolerance;

    queueCB.SetViewportScissor(lwrrentColumn, lwrrentRow, cellSize - 2, cellSize - 2);

#if defined(LW_TEGRA)
    // Testing if the call to GetLwrrentTimestampInNanoseconds is blocked by
    // previous submits is only done on HOS. See http://lwbugs/2470007
    // It is expected that the time needed after GetLwrrentTimestampInNanoseconds
    // is small compared to the time needed by the GPU to process the commands.
    // On Windows the test might run on faster GPUs which process the commands
    // much faster which increases the risk to fail the test. Therefore
    // limiting the test to HOS where is will only run on Maxwell.
    queueCB.ClearColor(0, callDurationWithinTolerances ? green : red, ClearColorMask::RGBA);
#else
    (void)callDurationWithinTolerances;
    queueCB.ClearColor(0, 0.0f, 0.0f, 1.0f, ClearColorMask::RGBA);
#endif

    queueCB.submit();
    queue->Finish();

    sync_top->Free();
}

OGTEST_CppTest(LWNTimestampTest, lwn_timestamp, );
