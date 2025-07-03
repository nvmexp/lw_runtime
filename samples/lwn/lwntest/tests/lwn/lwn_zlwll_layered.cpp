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

class LWNZLwllLayeredTest
{
public:
    struct Variant
    {
        bool bindFbo;
        bool useSaveRestore;
        int  firstTriangleDstLayer;
        int  secondTriangleDstLayer;
    };

    LWNTEST_CppMethods();

    LWNZLwllLayeredTest(Format depthFormat, bool useSubregions) :
        m_depthFormat(depthFormat),
        m_useSubregions(useSubregions)
    { }

private:
    static const int cellSize = 16;
    static const int cellMargin = 2;
    static const int cellsX = 640 / cellSize;
    static const int cellsY = 480 / cellSize;

    static const int fboWidth  = 128;
    static const int fboHeight = 64;

    Format m_depthFormat;
    bool m_useSubregions;

    // Layout of GPU report in memory.  See docs for LWNcounterType &
    // LWN_COUNTER_TYPE_ZLWLL_STATS for individual field meaning.
    struct ZLwllCounterMem
    {
        uint32_t zlwll0;
        uint32_t zlwll1;
        uint32_t zlwll2;
        uint32_t zlwll3;
    };

    void drawCell(QueueCommandBuffer &queueCB, int cx, int cy, bool result) const;
    void drawResultCells(QueueCommandBuffer &queueCB, const std::vector<bool>& results) const;
    void showFbo(QueueCommandBuffer &queueCB, Texture *tex, int srcW, int srcH, int srcLayerIdx, int dstIdx) const;
};

static const char* depthFormatName(Format fmt)
{
    switch (fmt) {
    case Format::DEPTH16:           return "DEPTH16";
    case Format::DEPTH24:           return "DEPTH24";
    case Format::DEPTH24_STENCIL8:  return "DEPTH24_STENCIL8";
    case Format::DEPTH32F:          return "DEPTH32F";
    case Format::DEPTH32F_STENCIL8: return "DEPTH32F_STENCIL8";
    default: assert(false);
    }
    return NULL;
}

lwString LWNZLwllLayeredTest::getDescription() const
{
    lwStringBuf sb;

    sb <<
        "Test ZLwll layered rendering support.\n"
        "\n"
        "The test renders 2 triangles into an offscreen buffer fbo_0.  The second\n"
        "larger green triangle will be partially occluded by the first\n"
        "red triangle.  There may be an RT switch to fbo_1 and back to fbo_0\n"
        "between the first and second triangle, depending on which test variant runs.\n"
        "\n"
        "The output of this test should contain multiple tiles of either overlapping two\n"
        "triangles or a single red triangle.\n"
        "\n"
        "On success, you should also see a row of GREEN cells at the bottom of the image.\n"
        "Test failures should show as RED cells.\n"
        "\n"
        "The test uses ZLWLL_STATS counters to verify that a) pixels\n"
        "flowed through the HW ZLwll unit and b) that hidden pixels\n"
        "got lwlled by ZLwll."
        "\n"
        "There are several variants of how the two triangles are rendered:\n"
        "\n"
        "  bindFbo == false: Only render to fbo_0, never switch render targets.\n"
        "  bindFbo == true:  Switch to fbo_0, render red triangle.\n"
        "                    Switch to fbo_1, render to fbo_1.\n"
        "                    Switch to fbo_0, render green triangle.\n"
        "\n"
        "  useSaveRestore == false: Do not attempt to preserve ZLwll contents on\n"
        "                    RT changes.\n"
        "  useSaveRestore == true: Save ZLwll contents prior to switching to fbo_1,\n"
        "                    restore ZLwll contents after switching back to fbo_0.\n"
        "\n"
        "When we switch between two RTs, ZLwll is not expected to be\n"
        "effective unless we Save & Restore ZLwll contents across render targets.\n"
        "We verify that ZLwll was effective across RT binds in the\n"
        "(bindFbo=true AND useSaveRestore=true) case.\n"
        "\n"
        "Testing for layered rendering support: Render the above two triangles\n"
        "into either the same output layer (expecting ZLwll to work) or into two\n"
        "separate layers (expecting ZLwll not to lwll).\n";

    sb << "\nUse " << depthFormatName(m_depthFormat) << " format for the depth texture.\n";

    if (m_useSubregions) {
        sb << "\nEnables ADAPTIVE_ZLWLL for the depth texture.\n";
    }

    return sb.str();
}

int LWNZLwllLayeredTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(38, 10);
}

// Display the results of a subtest in cell (cx,cy) in red/green based on
// <result>.
void LWNZLwllLayeredTest::drawCell(QueueCommandBuffer &queueCB, int cx, int cy, bool result) const
{
    queueCB.SetScissor(cx * cellSize + cellMargin, cy * cellSize + cellMargin,
                       cellSize - 2 * cellMargin, cellSize - 2* cellMargin);
    queueCB.ClearColor(0, result ? 0.0 : 1.0, result ? 1.0 : 0.0, 0.0, 1.0);
}

void LWNZLwllLayeredTest::drawResultCells(QueueCommandBuffer &queueCB, const std::vector<bool> &results) const
{
    int x = 0;
    for (std::vector<bool>::const_iterator it = results.begin(); it != results.end(); ++it, x++) {
        drawCell(queueCB, x, 0, *it);
    }
}

void LWNZLwllLayeredTest::showFbo(QueueCommandBuffer &queueCB, Texture *tex, int srcW, int srcH, int srcLayerIdx,
                                  int dstIdx) const
{
    CopyRegion srcRegion = { 0, 0, srcLayerIdx, srcW, srcH, 1 };

    assert(dstIdx < 16);

    int dstX = (dstIdx & 3) * lwrrentWindowWidth / 4;
    int dstY = ((dstIdx>>2) & 3) * lwrrentWindowHeight / 4;
    CopyRegion dstRegion = { dstX, dstY, 0, lwrrentWindowWidth / 4, lwrrentWindowHeight / 4, 1 };

    queueCB.CopyTextureToTexture(tex, NULL, &srcRegion,
                                 g_lwnWindowFramebuffer.getAcquiredTexture(), NULL, &dstRegion,
                                 CopyFlags::NONE);
}

void LWNZLwllLayeredTest::doGraphics() const
{
    const int32_t zlwllDebugWarningID = 1262;  // LWN_DEBUG_MESSAGE_DEPTH_STENCIL_ZLWLL_ILWALID

    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();

    // This test intentionally renders into a depth buffer without validating
    // ZLwll by clearing or SaveZLwllData/RestoreZLwllData'ing after a
    // SetRenderTargets call.  So disable this debug layer warning for the
    // duration of this test.
    DebugWarningIgnore(zlwllDebugWarningID);

    // allocator will create pool at first allocation
    // 64k is more than for this test but anything below 64k always allocates one page in the sparse pool
    const LWNsizeiptr poolSize = 0x10000UL;
    MemoryPoolAllocator allocator(device, NULL, poolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(binding = 0) uniform segmentBlock {\n"
        "  vec4 scale;\n"
        "  vec4 offset;\n"
        "  vec4 color;\n"
        "  int  outputLayer;\n"
        "};\n"
        "out vec4 vcolor;\n"
        "out int vlayer;\n"
        "void main() {\n"
        "  gl_Position = vec4(position*scale.xyz, 1.0) + offset;\n"
        "  vcolor = color;\n"
        "  vlayer = outputLayer;\n"
        "}\n";

    GeometryShader gs(450);
    gs.setGSParameters(GL_TRIANGLES, GL_TRIANGLE_STRIP, 3, 1);
    gs <<
        "in int vlayer[3];\n"
        "in vec4 vcolor[3];\n"
        "out vec4 ocolor;\n"
        "void main() {\n"
        "  for (int i = 0; i < 3; i++) {\n"
        "     gl_Position = gl_in[i].gl_Position;\n"
        "     gl_Layer = vlayer[i];\n"
        "     ocolor = vcolor[i];\n"
        "     EmitVertex();\n"
        "  }\n"
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
    g_glslcHelper->CompileAndSetShaders(pgm, gs, vs, fs);

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
    };

    struct Uniforms {
        dt::vec4 scale;
        dt::vec4 offset;
        dt::vec4 color;
        int outputLayer;
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

    Framebuffer fbos[] = {
        Framebuffer(fboWidth, fboHeight),
        Framebuffer(fboWidth, fboHeight)
    };
    const int NUM_FBOS = sizeof(fbos)/sizeof(fbos[0]);

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboAddr, sizeof(Uniforms));

    const int numLayers = 3;

    for (int i = 0; i < NUM_FBOS; i++) {
        fbos[i].setDepth(numLayers);
        fbos[i].setColorFormat(0, Format::RGBA8);
        fbos[i].setDepthStencilFormat(m_depthFormat);

        if (m_useSubregions) {
            fbos[i].setFlags(TextureFlags::ADAPTIVE_ZLWLL);
        }
        fbos[i].alloc(device);
    }

    LWNint zlwllBufAlignment = 0;
    device->GetInteger(DeviceInfo::ZLWLL_SAVE_RESTORE_ALIGNMENT, &zlwllBufAlignment);
    Texture *depthTex = fbos[0].getDepthTexture();
    const size_t zlwllBufferSize = depthTex->GetZLwllStorageSize();

    Buffer *zlwllBuffer = allocator.allocBuffer(&bb, BUFFER_ALIGN_ZLWLL_SAVE_BIT, zlwllBufferSize + zlwllBufAlignment);
    memset(zlwllBuffer->Map(), 0, zlwllBufferSize);

    Buffer *queryBuffer = allocator.allocBuffer(&bb, BUFFER_ALIGN_COUNTER_BIT, sizeof(ZLwllCounterMem));
    BufferAddress queryAddr = queryBuffer->GetAddress();
    ZLwllCounterMem *countersCpuVa = (ZLwllCounterMem *)queryBuffer->Map();

    std::vector<bool> testResults;

    static const Variant variants[] = {
        { false, false, 0, 0 }, // ZLwll should work
        { true,  false, 0, 0 }, // no ZLwll
        { true,  true,  0, 0 }, // ZLwll should work

        { false, false, 1, 0 }, // no ZLwll
        { false, false, 1, 1 }, // ZLwll should work
        { false, false, 1, 2 }, // no ZLwll

        { false, false, 2, 1 }, // no ZLwll
        { false, false, 2, 0 }, // no ZLwll
        { false, false, 2, 2 }, // ZLwll should work

        { true,  true,  1, 1 }, // ZLwll should work
        { true,  true,  2, 2 }, // ZLwll should work
    };

    for (int vi = 0; vi < (int)(sizeof(variants)/sizeof(variants[0])); vi++) {
        const Variant &variant = variants[vi];

        fbos[0].bind(queueCB);

        queueCB.SetViewport(0, 0, fboWidth, fboHeight);
        queueCB.SetScissor(0, 0, fboWidth, fboHeight);
        queueCB.ClearColor(0, 0.1f, 0.2f, 0.4f, 0);
        queueCB.ClearDepthStencil(1.0f, LWN_TRUE, 0x00, 0xff);

        DepthStencilState depthState;
        depthState.SetDefaults()
            .SetDepthTestEnable(LWN_TRUE)
            .SetDepthWriteEnable(LWN_TRUE)
            .SetDepthFunc(DepthFunc::LEQUAL);
        queueCB.BindDepthStencilState(&depthState);

        // Render 2 triangles into an offscreen buffer.  The second,
        // larger green triangle will be partially occluded by the first
        // red triangle.
        //
        // The first triangle is rendered into layer
        // variant.firstTriangleDstLayer and the second into
        // variant.secondTriangleDstLayer.  If the triangles are
        // rendered into the same layer, the latter triangle is
        // expected to be partially occluded by the first, and we
        // should see this from ZLWLL_STATS.

        queueCB.ResetCounter(CounterType::ZLWLL_STATS);
        uboCpuVa->scale  = dt::vec4(1, 1, 1, 1);
        uboCpuVa->offset = dt::vec4(0, 0, 0, 0);
        uboCpuVa->color  = dt::vec4(1, 0, 0, 1);
        uboCpuVa->outputLayer = variant.firstTriangleDstLayer;

        queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
        queueCB.ReportCounter(CounterType::ZLWLL_STATS, queryAddr);
        queueCB.submit();
        queue->Finish();
        ZLwllCounterMem queryResult0 = *countersCpuVa;

        // queryResult0 should contain tiles flowing through ZLwll (dword 0)
        // but no lwlled tiles.
        bool result0 = (queryResult0.zlwll0 != 0 &&
                        queryResult0.zlwll1 == 0 &&
                        queryResult0.zlwll2 == 0 &&
                        queryResult0.zlwll3 == 0);
        testResults.push_back(result0);

        if (variant.bindFbo) {
            // Bind another FBO here and clear its contents, render a
            // triangle into it and go back to the first FBO.
            if (variant.useSaveRestore) {
                queueCB.SaveZLwllData(zlwllBuffer->GetAddress(), zlwllBufferSize);
            }

            fbos[1].bind(queueCB);

            queueCB.ClearColor(0, 0, 0, 0, 0);
            queueCB.ClearDepthStencil(1.0f, LWN_TRUE, 0x00, 0xff);

            // Render a triangle into each layer
            for (int i = 0; i < numLayers; i++) {
                uboCpuVa->outputLayer = 0;
                queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
            }
            fbos[0].bind(queueCB);

            if (variant.useSaveRestore) {
                queueCB.RestoreZLwllData(zlwllBuffer->GetAddress(), zlwllBufferSize);
            }
        }

        queueCB.ResetCounter(CounterType::ZLWLL_STATS);
        uboCpuVa->scale  = dt::vec4(1.5, 1.5, 1.0, 1.0);
        uboCpuVa->offset = dt::vec4(0.1, 0.1, 0.1f, 0);
        uboCpuVa->color  = dt::vec4(0, 1, 0, 1);
        uboCpuVa->outputLayer = variant.secondTriangleDstLayer;
        queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
        queueCB.ReportCounter(CounterType::ZLWLL_STATS, queryAddr);
        queueCB.submit();
        queue->Finish();
        ZLwllCounterMem queryResult1 = *countersCpuVa;

        bool expectZLwlling = false;
        if (!variant.bindFbo) {
            // If we didn't jump between render targets, we shouldn't have
            // trouble zlwlling.
            expectZLwlling = true;
        } else {
            // If we switched between render targets, we expect zlwll
            // to lwll only when we save/restored ZLwll state.
            expectZLwlling = variant.useSaveRestore;
        }

        // If we layered the first triangle into another layer than
        // the second triangle, no zlwlling should've happened.
        if (variant.firstTriangleDstLayer != variant.secondTriangleDstLayer) {
            expectZLwlling = false;
        }

        // Note: The second triangle covers a larger screen area than
        // the first one.  We sanity check this by also checking that
        // more tiles flowed through zlwll when we rendered the second
        // triangle.  Hence the check for
        // "result1.zlwll0 > result0.zlwll0".
        bool result1 = (queryResult1.zlwll0 > queryResult0.zlwll0 &&
                        (expectZLwlling ? queryResult1.zlwll1 != 0 : queryResult1.zlwll1 == 0) &&
                        queryResult1.zlwll2 == 0 &&
                        queryResult1.zlwll3 == 0);
        testResults.push_back(result1);

        showFbo(queueCB, fbos[0].getColorTexture(0), fboWidth, fboHeight,
                variant.firstTriangleDstLayer, vi);
    }

    g_lwnWindowFramebuffer.bind();
    drawResultCells(queueCB, testResults);

    queueCB.submit();
    queue->Finish();

    for (int i = 0; i < NUM_FBOS; i++) {
        fbos[i].destroy();
    }

    DebugWarningAllow(zlwllDebugWarningID);
}

OGTEST_CppTest(LWNZLwllLayeredTest, lwn_zlwll_layered_d16,           (Format::DEPTH16, false));
OGTEST_CppTest(LWNZLwllLayeredTest, lwn_zlwll_layered_d24,           (Format::DEPTH24, false));
OGTEST_CppTest(LWNZLwllLayeredTest, lwn_zlwll_layered_d24s8,         (Format::DEPTH24_STENCIL8, false));
OGTEST_CppTest(LWNZLwllLayeredTest, lwn_zlwll_layered_d32f,          (Format::DEPTH32F, false));
OGTEST_CppTest(LWNZLwllLayeredTest, lwn_zlwll_layered_d32fs8,        (Format::DEPTH32F_STENCIL8, false));

OGTEST_CppTest(LWNZLwllLayeredTest, lwn_zlwll_layered_d16_subreg,    (Format::DEPTH16, true));
OGTEST_CppTest(LWNZLwllLayeredTest, lwn_zlwll_layered_d24_subreg,    (Format::DEPTH24, true));
OGTEST_CppTest(LWNZLwllLayeredTest, lwn_zlwll_layered_d24s8_subreg,  (Format::DEPTH24_STENCIL8, true));
OGTEST_CppTest(LWNZLwllLayeredTest, lwn_zlwll_layered_d32f_subreg,   (Format::DEPTH32F, true));
OGTEST_CppTest(LWNZLwllLayeredTest, lwn_zlwll_layered_d32fs8_subreg, (Format::DEPTH32F_STENCIL8, true));
