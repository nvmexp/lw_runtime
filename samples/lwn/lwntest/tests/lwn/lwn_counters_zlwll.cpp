/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
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

class LWNCounterZlwllTest
{
private:
    static const int cellSize = 16;
    static const int cellMargin = 2;
    static const int cellsX = 640 / cellSize;
    static const int cellsY = 480 / cellSize;

    static const int fboSize = 256;

    // Layout of GPU report in memory.  See docs for LWNcounterType &
    // LWN_COUNTER_TYPE_ZLWLL_STATS for individual field meaning.
    struct ZlwllCounterMem {
        uint32_t zlwll0;
        uint32_t zlwll1;
        uint32_t zlwll2;
        uint32_t zlwll3;
    };

    bool m_noZlwll;

    void drawCell(CommandBuffer *queueCB, int cx, int cy, bool result) const;
    void runTest(DeviceState *deviceState, std::vector<bool> &results) const;
public:
    LWNTEST_CppMethods();

    LWNCounterZlwllTest(bool disableZlwll) : m_noZlwll(disableZlwll)
    {
    }
};

lwString LWNCounterZlwllTest::getDescription() const
{
    lwStringBuf sb;
    sb << "Simple test exercising zlwll stats queries in LWN.\n\n"
          "This tests that zlwll is working and that we can query\n"
          "internal zlwll counters using the ReportCounter API. \n";

    if (m_noZlwll) {
        sb << "This test variant tests that ZLwll is not in effect when the LWNqueue used \n"
              "for rendering has been created with LWN_QUEUE_FLAGS_NO_ZLWLL_BIT.\n";
    }

    sb << "You should see only green cells on test pass, red cells otherwise.";
    return sb.str();
}

int LWNCounterZlwllTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 27);
}

// Display the results of a subtest in cell (cx,cy) in red/green based on
// <result>.
void LWNCounterZlwllTest::drawCell(CommandBuffer *queueCB, int cx, int cy, bool result) const
{
    queueCB->SetScissor(cx * cellSize + cellMargin, cy * cellSize + cellMargin,
                        cellSize - 2 * cellMargin, cellSize - 2* cellMargin);
    queueCB->ClearColor(0, result ? 0.0 : 1.0, result ? 1.0 : 0.0, 0.0, 1.0);
}

void LWNCounterZlwllTest::runTest(DeviceState *deviceState, std::vector<bool> &results) const
{
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();

    // allocator will create pool at first allocation
    // 64k is more than for this test but anything below 64k always allocates one page in the sparse pool
    const LWNsizeiptr poolSize = 0x10000UL;
    MemoryPoolAllocator allocator(device, NULL, poolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(binding = 0) uniform segmentBlock {\n"
        "  vec4 scale;\n"
        "  vec4 offset;\n"
        "  vec4 color;\n"
        "};\n"
        "out vec4 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position*scale.xyz, 1.0) + offset;\n"
        "  ocolor = color;\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec4 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = ocolor;\n"
        "}\n";

    Program *pgm = device->CreateProgram();
    // Compile and call lwnProgramSetShaders.
    g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
    };

    struct Uniforms {
        dt::vec4 scale;
        dt::vec4 offset;
        dt::vec4 color;
    };

    static const Vertex vertexData[] = {
        { dt::vec3(-0.5, -0.5, 0.5) },
        { dt::vec3(-0.5, +0.5, 0.5) },
        { dt::vec3(+0.5, -0.5, 0.5) },
    };

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 3, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // Create a vertex buffer and fill it with data
    BufferBuilder bufferBuilder;
    bufferBuilder.SetDevice(device).SetDefaults();
    Buffer *ubo = allocator.allocBuffer(&bufferBuilder, BUFFER_ALIGN_UNIFORM_BIT, sizeof(Uniforms));
    // Get a handle to be used for setting the buffer as a uniform buffer
    BufferAddress uboAddr = ubo->GetAddress();
    void *uboMem = (void *)ubo->Map();
    memset(uboMem, 0, sizeof(Uniforms));
    Uniforms *uboCpuVa = (Uniforms *)uboMem;

    // Set up a framebuffer for offline rendering.
    Framebuffer fbo(fboSize, fboSize);
    fbo.setColorFormat(0, Format::RGBA8);
    fbo.setDepthStencilFormat(Format::DEPTH24_STENCIL8);
    fbo.alloc(device);
    fbo.bind(queueCB);
    queueCB.SetViewport(0, 0, fboSize, fboSize);
    queueCB.SetScissor(0, 0, fboSize, fboSize);

    queueCB.ClearColor(0, 0, 0, 0, 0);
    queueCB.ClearDepthStencil(1.0f, LWN_TRUE, 0x00, 0xff);

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboAddr, sizeof(Uniforms));

    DepthStencilState depthState;
    depthState.SetDefaults()
        .SetDepthTestEnable(LWN_TRUE)
        .SetDepthWriteEnable(LWN_TRUE)
        .SetDepthFunc(DepthFunc::LEQUAL);
    queueCB.BindDepthStencilState(&depthState);

    Buffer *queryBuffer = allocator.allocBuffer(&bb, BUFFER_ALIGN_COUNTER_BIT, sizeof(ZlwllCounterMem));
    BufferAddress queryAddr = queryBuffer->GetAddress();
    ZlwllCounterMem *countersCpuVa = (ZlwllCounterMem *)queryBuffer->Map();
    memset(countersCpuVa, 0x11, sizeof(ZlwllCounterMem));

    bool result0 = (countersCpuVa->zlwll0 != 0 &&
                    countersCpuVa->zlwll1 != 0 &&
                    countersCpuVa->zlwll2 != 0 &&
                    countersCpuVa->zlwll3 != 0);

    // Clear zlwll stats to zero & check that reporting them writes
    // just zeros.
    queueCB.ResetCounter(CounterType::ZLWLL_STATS);
    queueCB.ReportCounter(CounterType::ZLWLL_STATS, queryAddr);
    queueCB.submit();
    queue->Finish();
    bool result1 = (countersCpuVa->zlwll0 == 0 &&
                    countersCpuVa->zlwll1 == 0 &&
                    countersCpuVa->zlwll2 == 0 &&
                    countersCpuVa->zlwll3 == 0);

    // Render 2 triangles into an offscreen buffer.  The second
    // green triangle will be partially occluded by the first red
    // triangle.
    uboCpuVa->scale  = dt::vec4(1, 1, 1, 1);
    uboCpuVa->offset = dt::vec4(0, 0, 0, 0);
    uboCpuVa->color  = dt::vec4(1, 0, 0, 1);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
    queueCB.ReportCounter(CounterType::ZLWLL_STATS, queryAddr);
    queueCB.submit();
    queue->Finish();
    ZlwllCounterMem queryResult0 = *countersCpuVa;

    uboCpuVa->scale  = dt::vec4(1.5, 1.5, 1.0, 1.0);
    uboCpuVa->offset = dt::vec4(0.1, 0.1, 0.1f, 0);
    uboCpuVa->color  = dt::vec4(0, 1, 0, 1);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
    queueCB.ReportCounter(CounterType::ZLWLL_STATS, queryAddr);
    queueCB.submit();
    queue->Finish();
    ZlwllCounterMem queryResult1 = *countersCpuVa;

    results.push_back(result0);
    results.push_back(result1);
    // queryResult0 should contain tiles flowing through zlwll (dword 0)
    // but no lwlled tiles.
    results.push_back(queryResult0.zlwll0 != 0 &&
                      queryResult0.zlwll1 == 0 &&
                      queryResult0.zlwll2 == 0 &&
                      queryResult0.zlwll3 == 0);
    // More tiles should've flown through zlwll now (stats0).  Also some tiles
    // should've been lwlled (unless we're running with ZLwll disabled).
    results.push_back(queryResult1.zlwll0 > queryResult0.zlwll0 &&
                      (m_noZlwll ? queryResult1.zlwll1 == 0 : queryResult1.zlwll1 != 0) &&
                      queryResult1.zlwll2 == 0 &&
                      queryResult1.zlwll3 == 0);
    fbo.destroy();

    pgm->Free();
}

void LWNCounterZlwllTest::doGraphics() const
{
    DisableLWNObjectTracking();

    std::vector<bool> results;

    {
        DeviceState deviceStateNoZlwll(LWNdeviceFlagBits(0),
                                       LWN_WINDOW_ORIGIN_MODE_LOWER_LEFT,
                                       LWN_DEPTH_MODE_NEAR_IS_MINUS_W,
                                       LWNqueueFlags(m_noZlwll ? LWN_QUEUE_FLAGS_NO_ZLWLL_BIT : 0));

        deviceStateNoZlwll.SetActive();
        runTest(&deviceStateNoZlwll, results);
    }

    DeviceState::SetDefaultActive();
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    for (int i = 0; i < (int)results.size(); i++) {
        drawCell(queueCB, i, 0, results[i]);
    }

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNCounterZlwllTest, lwn_counters_zlwll, (false));
OGTEST_CppTest(LWNCounterZlwllTest, lwn_counters_zlwll_no_zlwll, (true));
