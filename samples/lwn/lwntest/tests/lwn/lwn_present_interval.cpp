/*
* Copyright(c) 2016 LWPU Corporation.All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#include <math.h>

#include "../../elw/cmdline.h"

#define DEBUG_MODE 0
#if DEBUG_MODE
    #define DEBUG_PRINT(x) do { \
        printf x; \
        fflush(stdout); \
    } while (0)
#else
    #define DEBUG_PRINT(x)
#endif

using namespace lwn;

class LWNPresentInterval
{
public:
    LWNPresentInterval(bool fullRange) : m_fullRange(fullRange) {}
    LWNTEST_CppMethods();

private:
    const bool m_fullRange;
};

#define ALMOST_EQUAL(a, b, epsilon) (fabs(a - b) < epsilon)

lwString LWNPresentInterval::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Test to verify the SetPresentInterval function. The test renders multiple frames using "
        "different present intervals and measures the frame time. If no texture is available the "
        "AcquireTexture function will block the application until a texture gets freed for reuse. "
        "If a present interval greater than 0 is used this will happen after the requested number "
        "of VBlanks oclwrred. ";
        if (m_fullRange) {
            sb <<
                "All present intervals in the range of [0,MAX_PRESENT_INTERVAL] are tested. "
                "The expected results are:\n"
                "* 0: Free run. The frame time is expected to be very small.\n"
                "* n: The frame time is expected to be n times the time between two VBlanks.\n";
        } else {
            sb <<
                "The following present intervals are tested:\n"
                "* 0: Free run. The frame time is expected to be very small.\n"
                "* 1: The frame time is expected to be equal to the time between two VBlanks.\n"
                "* 2: The frame time is expected to be twice the time between two VBlanks.\n"
                "* 3: The frame time is expected to be three times the time between two VBlanks.\n"
                "\n"
                "If the frame times are as expected a green quad is drawn, if not a red quad is drawn.";
        }

    return sb.str();
}

int LWNPresentInterval::isSupported() const
{
#if !defined(LW_WINDOWS)
    return lwogCheckLWNAPIVersion(48, 3);
#else
    return (!useGLPresent && lwogCheckLWNAPIVersion(53, 6));
#endif
}

void LWNPresentInterval::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(440);
    vs <<
        "layout(location = 0) in vec3 position;\n"
        "layout(location = 1) in vec3 color;\n"
        "out vec4 col;\n"
        "void main() {\n"
        "  gl_Position = vec4(position.x, position.y, position.z, 1.0f);\n"
        "  col = vec4(color.r, color.g, color.b, 1.0);\n"
        "}\n";

    FragmentShader fs(440);
    fs <<
        "in vec4 col;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = col;\n"
        "}\n";

    Program *program = device->CreateProgram();

    if (!g_glslcHelper->CompileAndSetShaders(program, vs, fs)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));

        LWNFailTest();
        return;
    }

    const LWNuint vertexCount = 4;

    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };

    const Vertex vertices[] = {
        { dt::vec3(-1.0f, -1.0f, 0.0f), dt::vec3(0.0f, 0.0f, 1.0f) },
        { dt::vec3(1.0f, -1.0f, 0.0f), dt::vec3(0.0f, 0.0f, 1.0f) },
        { dt::vec3(-1.0f, 1.0f, 0.0f), dt::vec3(0.0f, 0.0f, 1.0f) },
        { dt::vec3(1.0f, 1.0f, 0.0f), dt::vec3(0.0f, 0.0f, 1.0f) }
    };

    MemoryPoolAllocator bufferAllocator(device, NULL, vertexCount * sizeof(vertices), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream vertexStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, color);
    VertexArrayState vertexState = vertexStream.CreateVertexArrayState();
    Buffer *vbo = vertexStream.AllocateVertexBuffer(device, vertexCount, bufferAllocator, vertices);
    BufferAddress vboAddr = vbo->GetAddress();

    int originalPresentInterval = g_lwnWindowFramebuffer.getPresentInterval();

    CommandBuffer cmdBuf;
    cmdBuf.Initialize(device);
    g_lwnCommandMem.populateCommandBuffer(&cmdBuf, CommandBufferMemoryManager::Coherent);

    cmdBuf.BeginRecording();

    cmdBuf.SetViewport(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    cmdBuf.SetScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    cmdBuf.ClearColor(0, 1.0f, 1.0f, 1.0f, 1.0f);

    cmdBuf.BindVertexArrayState(vertexState);
    cmdBuf.BindVertexBuffer(0, vboAddr, sizeof(vertices));

    cmdBuf.BindProgram(program, ShaderStageBits::ALL_GRAPHICS_BITS);

    cmdBuf.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, vertexCount);

    CommandHandle drawCommand = cmdBuf.EndRecording();

    int maxPresentInterval = 0;
    device->GetInteger(DeviceInfo::MAX_PRESENT_INTERVAL, &maxPresentInterval);

    // In LWN version 53.2 MAX_PRESENT_INTERVAL was increased from 2 to 3
    int expectedMaxPresentInterval = lwogCheckLWNAPIVersion(53, 2)  ? 3 : 2;
    // In LWN version 53.10 MAX_PRESENT_INTERVAL was increased from 3 to 5
    expectedMaxPresentInterval     = lwogCheckLWNAPIVersion(53, 10) ? 5 : expectedMaxPresentInterval;
    bool success = (maxPresentInterval == expectedMaxPresentInterval);

    // To limit the duration of the test we test by default only the range [0,3]
    if (!m_fullRange) {
        maxPresentInterval = LW_MIN(3, maxPresentInterval);
    }

    const int numPresentIntervals = maxPresentInterval + 1;
    const int numFrames = 100;
    const float timerFrequency = (float)lwogGetTimerFrequency();

    uint64_t timeStart;
    // The array is value-initialized to 0.
    float *timeInterval = new float[numPresentIntervals]();

    for (int n = 0; n < numPresentIntervals && success; ++n) {

#if defined(LW_HOS)
        // Disable testing of a zero swap interval due to bug 1785354.
        // Production LWN elwironments won't support an interval of zero due to tearing.
        if (n == 0) {
            continue;
        }
#endif

        g_lwnWindowFramebuffer.setPresentInterval(n);
        success = success && ((int)n == g_lwnWindowFramebuffer.getPresentInterval());

        timeStart = lwogGetTimerValue();

        for (int i = 0; i < numFrames && success; ++i) {

            g_lwnWindowFramebuffer.bind();

            queueCB.CallCommands(1, &drawCommand);
            queueCB.submit();
            queue->Flush();

            g_lwnWindowFramebuffer.present();

            lwogSwapBuffers();
        }

        timeInterval[n] = (float)(lwogGetTimerValue() - timeStart) / timerFrequency;
    }

    // Expected time to render numFrames with the current display's refresh rate. 
    const uint32_t refreshRate = lwogGetRefreshRate();
    if (refreshRate == 0) {
        DEBUG_PRINT(("Could not query display refresh rate, which is required for this test!\n"));

        LWNFailTest();
    }

    const float expectedTime = (float)(numFrames) / (float)(refreshRate);

    // Tolerance used when checking if the actual time for numFrames matches the expected
    // time. We use a high tolerance of 4ms per frame to avoid failing the test due to
    // scheduling overhead, but if the refresh rate is very high (for example 144 Hz) we instead
    // use half the refresh time.
    const float epsMs = LW_MIN(4.0f, 1000.0f/(refreshRate*2.0f));
    const float eps = (float)numFrames * (epsMs/ 1000.0f);

    // Check if the time used to render with PresentInterval = 0 is significantly smaller
    // than rendering with PresentIntervals = 1.
    success = success && (timeInterval[0] < (timeInterval[1] / 2.0f));
    // Check if the times to render with different present intervals match the expected times.
    for (int i = 1; i < numPresentIntervals && success; ++i) {
        success = success && ALMOST_EQUAL(expectedTime, timeInterval[i] / (float)(i), eps);
    }

    delete[] timeInterval;

    g_lwnWindowFramebuffer.bind();

    queueCB.SetViewport(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);

    if (success) {
        queueCB.ClearColor(0, 0.0f, 1.0f, 0.0f, 1.0f);
    } else {
        queueCB.ClearColor(0, 1.0f, 0.0f, 0.0f, 1.0f);
    }

    queueCB.submit();
    queue->Flush();

    queue->Finish();

    cmdBuf.Finalize();

    // restore default present interval
    g_lwnWindowFramebuffer.setPresentInterval(originalPresentInterval);
}

OGTEST_CppTest(LWNPresentInterval, lwn_present_interval,      (false));
OGTEST_CppTest(LWNPresentInterval, lwn_present_interval_full, (true));
