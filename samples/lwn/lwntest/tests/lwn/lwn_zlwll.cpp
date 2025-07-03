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

class LWNZLwllTest
{
public:
    enum FramebufferVariant {
        FramebufferVariantNone,                     // don't perform any special framebuffer binds
        FramebufferVariantColorRTChange,            // do a color-only bind that shouldn't mess up ZLwll state
        FramebufferVariantAliasedDepthRTChange,     // do an aliased depth buffer bind (bug 3363342)
        FramebufferVariantDuplicateDepthRTChange,   // do a duplicate depth buffer bind that shouldn't mess up ZLwll state
    };
    struct Variant
    {
        bool bindFbo;
        bool useSaveRestore;
    };

    LWNTEST_CppMethods();

    LWNZLwllTest(FramebufferVariant framebufferVariant, int numSamples, bool useSubregions) :
        m_framebufferVariant(framebufferVariant),
        m_numSamples(numSamples),
        m_useSubregions(useSubregions)
    {
    }
private:
    static const int cellSize = 16;
    static const int cellMargin = 2;
    static const int cellsX = 640 / cellSize;
    static const int cellsY = 480 / cellSize;

    static const int fboWidth  = 128;
    static const int fboHeight = 64;

    FramebufferVariant m_framebufferVariant;
    int m_numSamples;
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
    void showFbo(QueueCommandBuffer &queueCB, Texture *tex, int srcW, int srcH, int dstIdx) const;
};

lwString LWNZLwllTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Test to exercise driver ZLwll implementation and ZLwll APIs.\n"
        "\n"
        "The test renders 2 triangles into an offscreen buffer fbo_0.  The second\n"
        "larger green triangle will be partially occluded by the first\n"
        "red triangle.  There may be an RT switch to fbo_1 and back to fbo_0\n"
        "between the first and second triangle, depending on which test variant runs.\n"
        "\n"
        "The output of this test should contain multiple tiles of two overlapping triangles.\n"
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
        "(bindFbo=true AND useSaveRestore=true) case.\n";

    switch (m_framebufferVariant) {
    case FramebufferVariantNone:
        break;
    case FramebufferVariantColorRTChange:
        sb <<
            "\nAlso, in the test variants using bindFbo, bind only color the color "
            "render target of fbo_1, which should leave ZLwll state unmodified.";
        break;
    case FramebufferVariantAliasedDepthRTChange:
        sb <<
            "\nAlso, in the test variants not using bindFbo, bind a depth buffer "
            "using the same memory as the main FBO's depth buffer, bind the "
            "main FBO's depth buffer again, and then draw a triangle that should "
            "fail the depth buffer.  If bug 3363342 is fixed, the sequence will "
            "ilwalidate ZLwll state, but we should rebuild occluders from the extra "
            "triangle.";
        break;
    case FramebufferVariantDuplicateDepthRTChange:
        sb <<
            "\nAlso, in the test variants not using bindFbo, do a redundant bind "
            "of the primary depth buffer, which should leave ZLwll state unmodified.";
        break;
    default:
        assert(0);
        break;
    }

    if (m_useSubregions) {
        sb << "\nThis test variant enables LWN_TEXTURE_FLAGS_ADAPTIVE_ZLWLL for the depth target.\n";
    }
    if (m_numSamples) {
        sb << "\nThis test uses MSAA with " << m_numSamples << " samples.\n";
    }

    return sb.str();
}

int LWNZLwllTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(38, 10);
}

// Display the results of a subtest in cell (cx,cy) in red/green based on
// <result>.
void LWNZLwllTest::drawCell(QueueCommandBuffer &queueCB, int cx, int cy, bool result) const
{
    queueCB.SetScissor(cx * cellSize + cellMargin, cy * cellSize + cellMargin,
                       cellSize - 2 * cellMargin, cellSize - 2* cellMargin);
    queueCB.ClearColor(0, result ? 0.0 : 1.0, result ? 1.0 : 0.0, 0.0, 1.0);
}

void LWNZLwllTest::drawResultCells(QueueCommandBuffer &queueCB, const std::vector<bool> &results) const
{
    int x = 0;
    for (std::vector<bool>::const_iterator it = results.begin(); it != results.end(); ++it, x++) {
        drawCell(queueCB, x, 0, *it);
    }
}

void LWNZLwllTest::showFbo(QueueCommandBuffer &queueCB, Texture *tex, int srcW, int srcH, int dstIdx) const
{
    CopyRegion srcRegion = { 0, 0, 0, srcW, srcH, 1 };

    assert(dstIdx < 4);

    int dstX = (dstIdx & 1) * lwrrentWindowWidth / 2;
    int dstY = ((dstIdx>>1) & 1) * lwrrentWindowHeight / 2;
    CopyRegion dstRegion = { dstX, dstY, 0, lwrrentWindowWidth / 2, lwrrentWindowHeight / 2, 1 };

    queueCB.CopyTextureToTexture(tex, NULL, &srcRegion,
                                 g_lwnWindowFramebuffer.getAcquiredTexture(), NULL, &dstRegion,
                                 CopyFlags::NONE);
}

void LWNZLwllTest::doGraphics() const
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
    queueCB.submit();

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

    Framebuffer fbos[] = {
        Framebuffer(fboWidth, fboHeight),
        Framebuffer(fboWidth, fboHeight)
    };
    const int NUM_FBOS = sizeof(fbos)/sizeof(fbos[0]);

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboAddr, sizeof(Uniforms));

    for (int i = 0; i < NUM_FBOS; i++) {
        fbos[i].setColorFormat(0, Format::RGBA8);
        fbos[i].setDepthStencilFormat(Format::DEPTH24);

        if (m_useSubregions) {
            fbos[i].setFlags(TextureFlags::COMPRESSIBLE | TextureFlags::ADAPTIVE_ZLWLL);
        }

        if (m_numSamples) {
            fbos[i].setSamples(m_numSamples);
        }

        fbos[i].alloc(device);

        fbos[i].bind(queueCB);
        queueCB.ClearDepthStencil(1.0f, LWN_TRUE, 0x00, 0xff);
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

    // Create a separate "aliased" depth texture with the same base address as
    // our regular FBO depth texture with a different size.  This is used to
    // trigger the conditions related to bug 3363342, where temporarily binding
    // this aliased texture will ilwalidate ZLwll regions but fail to ilwoke
    // ClearZlwllRegion to allow us to start re-building occluders.
    TextureBuilder aliasedDepthTexBuilder;
    Texture aliasedDepthTex;
    aliasedDepthTexBuilder.SetDefaults().SetDevice(device).SetLevels(1).
        SetFormat(Format::DEPTH24).SetFlags(TextureFlags::COMPRESSIBLE);
    aliasedDepthTexBuilder.SetSize2D(fboWidth / 2, fboHeight / 2);
    aliasedDepthTexBuilder.SetStorage(depthTex->GetMemoryPool(), depthTex->GetMemoryOffset());
    if (m_numSamples > 0) {
        aliasedDepthTexBuilder.SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE).SetSamples(m_numSamples);
    }
    aliasedDepthTex.Initialize(&aliasedDepthTexBuilder);

    std::vector<bool> testResults;

    static const Variant variants[] = {
        { false, false},
        { true, false},
        { true, true },
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

        if (m_numSamples) {
            MultisampleState msaa;
            msaa.SetDefaults();
            msaa.SetSamples(m_numSamples);
            msaa.SetMultisampleEnable(LWN_TRUE);
            queueCB.BindMultisampleState(&msaa);
        }

        // Render a first red triangle into the offscreen buffer.  This will be
        // used as an occluder when rendering a second larger green triangle
        // bind this red triangle.
        queueCB.ResetCounter(CounterType::ZLWLL_STATS);
        uboCpuVa->scale  = dt::vec4(1, 1, 1, 1);
        uboCpuVa->offset = dt::vec4(0, 0, 0, 0);
        uboCpuVa->color  = dt::vec4(1, 0, 0, 1);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
        queueCB.ReportCounter(CounterType::ZLWLL_STATS, queryAddr);
        queueCB.submit();
        queue->Finish();
        ZLwllCounterMem queryResult0 = *countersCpuVa;

        // queryResult0 should contain tiles flowing through zlwll (dword 0)
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

            if (m_framebufferVariant == FramebufferVariantColorRTChange) {
                // Bind color only
                Texture *color = fbos[1].getColorTexture(0);
                queueCB.SetRenderTargets(1, &color, NULL, NULL, NULL);
            } else {
                // Color & depth RT change
                fbos[1].bind(queueCB);
            }

            queueCB.ClearColor(0, 0, 0, 0, 0);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
            queueCB.submit();

            // Bind back to original color & depth RTs
            fbos[0].bind(queueCB);

            if (variant.useSaveRestore) {
                queueCB.RestoreZLwllData(zlwllBuffer->GetAddress(), zlwllBufferSize);
            }
        } else if (m_framebufferVariant == FramebufferVariantDuplicateDepthRTChange) {
            // Do a redundant bind of the main FBO, which should leave the ZLwll
            // state unmodified.
            fbos[0].bind(queueCB);
        } else if (m_framebufferVariant == FramebufferVariantAliasedDepthRTChange) {
            // Exercise bug 3363342, where we temporarily bind and unbind a
            // depth texture using the same storage as the depth texture in the
            // main FBO but with different parameters (size).  This pointless
            // bind will ilwalidate ZLwll regions.  To recover the ZLwll state,
            // we will render an extra blue polygon with Z values greater than
            // the red one.  That will fail the depth test, but should cause
            // ZLwll to rebuild some occluders based on the results of those
            // tests.  Without the driver fix to bug 3363342, we will fail to
            // send the ClearZlwllRegion method and the region will remain
            // invalid.
            Texture *colorTextures[] = { fbos[0].getColorTexture(0) };
            queueCB.SetRenderTargets(1, colorTextures, NULL, &aliasedDepthTex, NULL);
            queueCB.SetRenderTargets(1, colorTextures, NULL, fbos[0].getDepthTexture(), NULL);
            uboCpuVa->scale  = dt::vec4(1, 1, 1, 1);
            uboCpuVa->offset = dt::vec4(0, 0, +0.1, 0);
            uboCpuVa->color  = dt::vec4(0, 0, 1, 1);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
            queueCB.submit();
            queue->Finish();
        }

        // Render a second green triangle into the offscreen buffer while
        // collecting ZLwll statistics.  This will be partially occluded by the
        // previous red triangle.
        queueCB.ResetCounter(CounterType::ZLWLL_STATS);
        uboCpuVa->scale  = dt::vec4(1.5, 1.5, 1.0, 1.0);
        uboCpuVa->offset = dt::vec4(0.1, 0.1, 0.1f, 0);
        uboCpuVa->color  = dt::vec4(0, 1, 0, 1);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
        queueCB.ReportCounter(CounterType::ZLWLL_STATS, queryAddr);
        queueCB.submit();
        queue->Finish();
        ZLwllCounterMem queryResult1 = *countersCpuVa;

        bool expectZLwlling = false;
        if (!variant.bindFbo) {

            switch (m_framebufferVariant) {
            case FramebufferVariantAliasedDepthRTChange:
            case FramebufferVariantDuplicateDepthRTChange:
                // If we did aliased or duplicate depth binds, the ZLwll state
                // should not be disturbed (duplicate) or should have at least
                // been rebuilt (aliased) in normal testing.  However, the
                // adaptive ZLwll (subregion) binds do not use the same
                // filtering and will not preserve/recover state on these binds.
                expectZLwlling = !m_useSubregions;
                break;

            default:
                // If we didn't jump between render targets, we shouldn't have
                // trouble zlwlling.
                expectZLwlling = true;
                break;
            }
        } else {
            // When switching between render targets, we expect ZLwll
            // to lwll only when:
            //
            // 1. We explicitly used SaveZLwllData/RestoreZLwllData to preserve
            //    ZLwll contents, OR
            // 2. We switched to a color-only RT and then switched back to the
            //    original color & depth.  There's no need to clear ZLwll
            //    in this case as the depth buffer didn't actually change.
            //
            // The clear filtering path only works when ADAPTIVE_ZLWLL
            // is disabled.
            expectZLwlling = variant.useSaveRestore || (m_framebufferVariant == FramebufferVariantColorRTChange && !m_useSubregions);
        }

        bool result1 = (queryResult1.zlwll0 > queryResult0.zlwll0 &&
                        (expectZLwlling ? queryResult1.zlwll1 != 0 : queryResult1.zlwll1 == 0) &&
                        queryResult1.zlwll2 == 0 &&
                        queryResult1.zlwll3 == 0);
        testResults.push_back(result1);

        if (m_numSamples) {
            fbos[0].downsample(queueCB);
        }
        showFbo(queueCB, fbos[0].getColorTexture(0), fboWidth, fboHeight, vi);
    }

    g_lwnWindowFramebuffer.bind();
    drawResultCells(queueCB, testResults);

    queueCB.submit();
    queue->Finish();

    aliasedDepthTex.Finalize();
    for (int i = 0; i < NUM_FBOS; i++) {
        fbos[i].destroy();
    }

    DebugWarningAllow(zlwllDebugWarningID);
}

OGTEST_CppTest(LWNZLwllTest, lwn_zlwll, (LWNZLwllTest::FramebufferVariantNone, 0, false));
OGTEST_CppTest(LWNZLwllTest, lwn_zlwll_msaa_4, (LWNZLwllTest::FramebufferVariantNone, 4, false));
OGTEST_CppTest(LWNZLwllTest, lwn_zlwll_msaa_8, (LWNZLwllTest::FramebufferVariantNone, 8, false));
OGTEST_CppTest(LWNZLwllTest, lwn_zlwll_rt_change, (LWNZLwllTest::FramebufferVariantColorRTChange, 0, false));
OGTEST_CppTest(LWNZLwllTest, lwn_zlwll_zalias_change, (LWNZLwllTest::FramebufferVariantAliasedDepthRTChange, 0, false));
OGTEST_CppTest(LWNZLwllTest, lwn_zlwll_zdup_change, (LWNZLwllTest::FramebufferVariantDuplicateDepthRTChange, 0, false));

OGTEST_CppTest(LWNZLwllTest, lwn_zlwll_subreg, (LWNZLwllTest::FramebufferVariantNone, 0, true));
OGTEST_CppTest(LWNZLwllTest, lwn_zlwll_msaa_4_subreg, (LWNZLwllTest::FramebufferVariantNone, 4, true));
OGTEST_CppTest(LWNZLwllTest, lwn_zlwll_msaa_8_subreg, (LWNZLwllTest::FramebufferVariantNone, 8, true));
OGTEST_CppTest(LWNZLwllTest, lwn_zlwll_rt_change_subreg, (LWNZLwllTest::FramebufferVariantColorRTChange, 0, true));
OGTEST_CppTest(LWNZLwllTest, lwn_zlwll_zalias_change_subreg, (LWNZLwllTest::FramebufferVariantAliasedDepthRTChange, 0, true));
OGTEST_CppTest(LWNZLwllTest, lwn_zlwll_zdup_change_subreg, (LWNZLwllTest::FramebufferVariantDuplicateDepthRTChange, 0, true));
