/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// lwn_conditional_render.cpp
//
// Test conditional rendering using an occlusion-lwlling-like algorithm.
//
#include "lwntest_cpp.h"
#include "lwn_utils.h"

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
using namespace lwn::dt;

struct Vertex {
    vec3 position;
    vec3 color;
};

static void drawQuad(QueueCommandBuffer &queueCB, Queue *queue, Buffer *vbo, const vec3& color, float depth)
{
    // Ensure that any previous rendering has finished before we mess with the VBO.
    queueCB.submit();
    queue->Finish();

    Vertex* vboMem = static_cast<Vertex*>(vbo->Map());

    for (int i = 0; i < 4; ++i) {
        vboMem[i].position[2] = depth;
        vboMem[i].color = color;
    }

    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
}

class LWNConditionalRenderTest
{
public:
    LWNTEST_CppMethods();
    static void drawImage(CommandBuffer *queueCB, int level, int layer,
                          TextureBuilder *builder, MemoryPool *pool);
};

lwString LWNConditionalRenderTest::getDescription() const
{
    return "Test conditional rendering using its most common use case--occlusion lwlling. "
           "For each cell, draw two opaque depth-tested quads at depth 0.5 and depth 1.0, not "
           "necessarily in that order, querying SAMPLES_PASSED before and after the second "
           "quad. Call SetRenderEnabledConditional with RENDER_IF_EQUAL or "
           "RENDER_IF_NOT_EQUAL, then draw a third quad at depth 0.0. Should produce only "
           "green quads in the final output.  For the conditional render phase, we insert "
           "a FenceSync after reporting sampler counts and a WaitSync before rendering "
           "conditionally, to make sure the results are available.  We test all four "
           "combinations of Fences and Waits in queues and command buffers.";
}

int LWNConditionalRenderTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(27, 0);
}

void LWNConditionalRenderTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 aPosition;\n"
        "layout(location=1) in vec3 aColor;\n"
        "out vec3 vColor;\n"
        "void main() {\n"
        "    gl_Position = vec4(aPosition, 1.0);\n"
        "    vColor = aColor;\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec3 vColor;\n"
        "out vec4 fColor;\n"
        "void main() {\n"
        "    fColor = vec4(vColor, 1.0);\n"
        "}\n";
    Program *program = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(program, vs, fs)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        LWNFailTest();
        return;
    }

    VertexStream vertexStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, color);
    VertexArrayState vertexState = vertexStream.CreateVertexArrayState();

    static const Vertex vertexData[] = {
        { vec3(-1, -1, 0), vec3(0, 0, 0) },
        { vec3( 1, -1, 0), vec3(0, 0, 0) },
        { vec3( 1,  1, 0), vec3(0, 0, 0) },
        { vec3(-1,  1, 0), vec3(0, 0, 0) },
    };

    MemoryPoolAllocator allocator(device, NULL, sizeof(vertexData) + sizeof(CounterData) * 2,
                                  LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    Buffer *vbo = vertexStream.AllocateVertexBuffer(device, 4, allocator, vertexData);

    BufferBuilder bufferBuilder;
    bufferBuilder.SetDevice(device).SetDefaults();
    Buffer *counterBuffer = allocator.allocBuffer(&bufferBuilder, BUFFER_ALIGN_COUNTER_BIT,
                                                  sizeof(CounterData) * 2);
    BufferAddress counterAddress = counterBuffer->GetAddress();

    DepthStencilState depthState;
    depthState.SetDefaults()
        .SetDepthTestEnable(LWN_TRUE)
        .SetDepthWriteEnable(LWN_TRUE);
    queueCB.BindDepthStencilState(&depthState);

    queueCB.ClearColor(0, 0.0, 0.0, 0.1, 1.0);

    queueCB.BindVertexArrayState(vertexState);
    queueCB.BindVertexBuffer(0, vbo->GetAddress(), sizeof(vertexData));
    queueCB.BindProgram(program, ShaderStageBits::ALL_GRAPHICS_BITS);
    vec3 red(1.0f, 0.0f, 0.0f);
    vec3 green(0.0f, 1.0f, 0.0f);
    Sync *sync = device->CreateSync();

    // Each test case draws three quads covering the same pixels:
    // 1: Drawn unconditionally and unoccluded
    // 2: Drawn with an occlusion query. Might or might not be occluded by quad 1.
    // 3: Drawn unoccluded but conditionally, based on quad 2's occlusion query.
    struct {
        bool occludeQuad2;
        ConditionalRenderMode quad3mode;
        int whichVisible; // After drawing all three quads, which one is visible?
    } testCases[] = {
        { true, ConditionalRenderMode::RENDER_IF_NOT_EQUAL, 1 },
        { false, ConditionalRenderMode::RENDER_IF_NOT_EQUAL, 3 },
        { true, ConditionalRenderMode::RENDER_IF_EQUAL, 3 },
        { false, ConditionalRenderMode::RENDER_IF_EQUAL, 2 },
    };
    size_t numTests = __GL_ARRAYSIZE(testCases);
    size_t totalTests = 4 * numTests;

    for (size_t i = 0; i < totalTests; ++i) {

        size_t which = i % numTests;
        size_t variant = i / numTests;
        bool fenceInQueue = (0 == (variant & 1));
        bool waitInQueue = (0 == (variant & 2));

        queueCB.SetViewport(LWNint(i * lwrrentWindowWidth / totalTests + 1), 1,
                            LWNuint(lwrrentWindowWidth / totalTests - 2), lwrrentWindowHeight - 2);

        // Quad 1
        depthState.SetDepthFunc(DepthFunc::ALWAYS);
        queueCB.BindDepthStencilState(&depthState);
        drawQuad(queueCB, queue, vbo, (testCases[which].whichVisible == 1) ? green : red,
                    testCases[which].occludeQuad2 ? 0.5f : 1.0f);

        // Quad 2
        depthState.SetDepthFunc(DepthFunc::LESS);
        queueCB.BindDepthStencilState(&depthState);
        queueCB.ReportCounter(CounterType::SAMPLES_PASSED, counterAddress);
        drawQuad(queueCB, queue, vbo, (testCases[which].whichVisible == 2) ? green : red,
                    testCases[which].occludeQuad2 ? 1.0f : 0.5f);
        queueCB.ReportCounter(CounterType::SAMPLES_PASSED, counterAddress + sizeof(CounterData));

        // Generate a FenceSync to check that the previous counter has landed.
        if (fenceInQueue) {
            queueCB.submit();
            queue->FenceSync(sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        } else {
            queueCB.FenceSync(sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        }

        // Insert a WaitSync in the queue or command buffer wait on the
        // previous fence.
        if (waitInQueue) {
            if (!fenceInQueue) {
                queueCB.submit();
            }
            queue->WaitSync(sync);
        } else {
            queueCB.WaitSync(sync);
        }

        // Quad 3
        queueCB.SetRenderEnableConditional(testCases[which].quad3mode, counterAddress);
        drawQuad(queueCB, queue, vbo, (testCases[which].whichVisible == 3) ? green : red,
                    0.0f);
        queueCB.SetRenderEnable(LWN_TRUE);
    }

    queueCB.submit();
    queue->Finish();
    sync->Free();
}

OGTEST_CppTest(LWNConditionalRenderTest, lwn_conditional_render, );

