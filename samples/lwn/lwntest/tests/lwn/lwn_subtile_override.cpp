/*
 * Copyright (c) 2020 LWPU Corporation.  All rights reserved.
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

class LWNSubtileOverrideTest
{
    static const int mrtCount = 4;          // number of textures for MRT sub-tests
    static const int texWidth = 32;         // size of off-screen textures
    static const int texHeight = 32;
    static const int cellWidth = 40;        // size of cells used to display off-screen textures
    static const int cellHeight = 40;
    static const int cellMarginX = (cellWidth - texWidth) / 2;
    static const int cellMarginY = (cellHeight - texHeight) / 2;

    // When testing, we iterate over a number of different subtile override
    // settings stored in a static array.
    static const int overrides[];
    static const size_t nOverrides;
    static const int OVERRIDE_DISABLED = -1;            // disable the override
    static const int OVERRIDE_TO_SHADER_VALUE = 2;      // set the override to a value queried from the shader

    // Record a comment to override the subtile size based on <override>, using
    // <program> for queries on OVERRIDE_TO_SHADER_VALUE.
    void setOverride(QueueCommandBuffer &queueCB, const Program *program, int override) const;

public:
    LWNTEST_CppMethods();
};

const int LWNSubtileOverrideTest::overrides[] = {
    OVERRIDE_DISABLED,
    OVERRIDE_TO_SHADER_VALUE,
    0x01,
    0x10,
    0x20,
    0x40,
    0x80        // maximum subtile size
};
const size_t LWNSubtileOverrideTest::nOverrides = __GL_ARRAYSIZE(LWNSubtileOverrideTest::overrides);

lwString LWNSubtileOverrideTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Functional test for shader subtile size override APIs.  This test renders simple shaded "
        "quads with different subtile size overrides.  Each pair of columns (except the last one) "
        "tests a different combination of override value and program type (separable or combined). "
        "Each column has two sets of five rows, where the bottom set overrides after binding "
        "programs and the top set overrides before.  In each set of five rows, the bottom row "
        "uses a regular single-output shader, while the top four rows use an MRT shader with four "
        "outputs.  The rightmost row displays a set of red/green results depending on whether "
        "the results of subtile size queries for each program object are plausible.";
    return sb.str();
}

int LWNSubtileOverrideTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(54, 1);
}

void LWNSubtileOverrideTest::setOverride(QueueCommandBuffer &queueCB, const Program *program, int override) const
{
    switch (override) {
    case OVERRIDE_DISABLED:
        queueCB.OverrideSubtileSize(LWN_FALSE, 0);
        break;
    case OVERRIDE_TO_SHADER_VALUE:
        queueCB.OverrideSubtileSize(LWN_TRUE, program->GetSubtileSize());
        break;
    default:
        assert(override > 0 && override <= 0x80);
        queueCB.OverrideSubtileSize(LWN_TRUE, override);
        break;
    }
}

void LWNSubtileOverrideTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

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

    FragmentShader fsMRT(440);
    fsMRT <<
        "in vec3 ocolor;\n"
        "layout(location=0) out vec4 fcolor0;\n"
        "layout(location=1) out vec4 fcolor1;\n"
        "layout(location=2) out vec4 fcolor2;\n"
        "layout(location=3) out vec4 fcolor3;\n"
        "void main() {\n"
        "  fcolor0 = vec4(ocolor, 1.0);\n"
        "  fcolor1 = vec4(ocolor.r, ocolor.g, 0.0,      1.0);\n"
        "  fcolor2 = vec4(ocolor.r, 0.0,      ocolor.b, 1.0);\n"
        "  fcolor3 = vec4(0.0,      ocolor.g, ocolor.b, 1.0);\n"
        "}\n";

    Program *pgm = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);
    Program *pgmMRT = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(pgmMRT, vs, fsMRT);
    Program *pgmVSOnly = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(pgmVSOnly, vs);
    Program *pgmFSOnly = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(pgmFSOnly, fs);
    Program *pgmMRTFSOnly = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(pgmMRTFSOnly, fsMRT);

    // Set up some off-screen textures that we can render to -- one for a single
    // render target test and multiple for an MRT test.
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults().
        SetSize2D(texWidth, texHeight).
        SetTarget(TextureTarget::TARGET_2D).
        SetFormat(Format::RGBA8);
    size_t texSize = tb.GetStorageSize();
    MemoryPoolAllocator *texAllocator = new MemoryPoolAllocator(device, NULL, (mrtCount + 1) * texSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    Texture *renderTex = texAllocator->allocTexture(&tb);
    Texture *mrtTex[mrtCount];
    for (int i = 0; i < mrtCount; i++) {
        mrtTex[i] = texAllocator->allocTexture(&tb);
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-0.8, -0.8, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-0.8, +0.8, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+0.8, +0.8, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+0.8, -0.8, 0.0), dt::vec3(1.0, 0.0, 0.0) },
    };

    MemoryPoolAllocator allocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    int cellX = 0;
    int cellY = 0;
    Texture *winTex = g_lwnWindowFramebuffer.getAcquiredTexture();
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    // Iterate over override settings, separable/non-separable programs, and
    // applying to override before/after binding programs.
    queueCB.SetViewportScissor(0, 0, texWidth, texHeight);
    for (uint32_t oidx = 0; oidx < nOverrides; oidx++) {
        int overrideValue = overrides[oidx];
        for (uint32_t separable = 0; separable < 2; separable++) {
            for (uint32_t overrideFirst = 0; overrideFirst < 2; overrideFirst++) {

                if (overrideFirst) {
                    setOverride(queueCB, separable ? pgmFSOnly : pgm, overrideValue);
                }

                // Render to a single off-screen render target, using a basic
                // LWN shader.
                if (separable) {
                    queueCB.BindProgram(NULL, ShaderStageBits::ALL_GRAPHICS_BITS);
                    queueCB.BindProgram(pgmVSOnly, ShaderStageBits::VERTEX);
                    queueCB.BindProgram(pgmFSOnly, ShaderStageBits::FRAGMENT);
                } else {
                    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
                }
                if (!overrideFirst) {
                    setOverride(queueCB, separable ? pgmFSOnly : pgm, overrideValue);
                }
                queueCB.SetRenderTargets(1, &renderTex, NULL, NULL, NULL);
                queueCB.ClearColor(0, 0.0, 0.2, 0.0, 0.0);
                queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);

                // In we're using "override to shader value" mode in
                // override-first mode, switch to using an override derived from
                // the MRT shader we will use for the next test.
                if (overrideFirst && overrideValue == OVERRIDE_TO_SHADER_VALUE) {
                    setOverride(queueCB, separable ? pgmMRTFSOnly : pgmMRT, overrideValue);
                }

                // Render to multiple off-screen render targets, using the MRT
                // variant of our basic LWN shader.
                if (separable) {
                    queueCB.BindProgram(pgmMRTFSOnly, ShaderStageBits::FRAGMENT);
                } else {
                    queueCB.BindProgram(pgmMRT, ShaderStageBits::ALL_GRAPHICS_BITS);
                }
                if (!overrideFirst) {
                    setOverride(queueCB, separable ? pgmMRTFSOnly : pgmMRT, overrideValue);
                }
                queueCB.SetRenderTargets(mrtCount, mrtTex, NULL, NULL, NULL);
                for (int i = 0; i < mrtCount; i++) {
                    queueCB.ClearColor(i, 0.2, 0.0, 0.0, 0.0);
                }
                queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);

                // Copy the off-screen rendered textures (both regular and MRT)
                // to cells on the screen.
                CopyRegion srcRegion = { 0, 0, 0, texWidth, texHeight, 1 };
                CopyRegion dstRegion = { 0, 0, 0, texWidth, texHeight, 1 };
                dstRegion.xoffset = cellX * cellWidth + cellMarginX;
                dstRegion.yoffset = cellY * cellHeight + cellMarginY;
                queueCB.CopyTextureToTexture(renderTex, NULL, &srcRegion, winTex, NULL, &dstRegion, CopyFlags::NONE);
                for (int i = 0; i < mrtCount; i++) {
                    dstRegion.yoffset += cellHeight;
                    queueCB.CopyTextureToTexture(mrtTex[i], NULL, &srcRegion, winTex, NULL, &dstRegion, CopyFlags::NONE);
                }

                // Bump the Y cell number to cover the single and MRT cells and
                // leave an additional blank cell before the next iteration.
                cellY += (mrtCount + 2);
            }
            cellX++;
            cellY = 0;
        }
    }

    g_lwnWindowFramebuffer.bind();

    // Add one more column at the end that queries the subtile size, verifies
    // that the returned values are plausible, and displays the results as
    // red/green.
    struct SubtileQueryExpectations {
        const Program *program;
        bool shouldBeZero;
    } subtileQueryExpectations[] = {
        { pgm, false },
        { pgmVSOnly, true },
        { pgmFSOnly, false },
        { pgmMRT, false },
        { pgmMRTFSOnly, false }
    };
    for (uint32_t pgmidx = 0; pgmidx < __GL_ARRAYSIZE(subtileQueryExpectations); pgmidx++) {
        bool valid;
        int subtileSize = subtileQueryExpectations[pgmidx].program->GetSubtileSize();
        if (subtileQueryExpectations[pgmidx].shouldBeZero) {
            valid = (subtileSize == 0);
        } else {
            valid = (subtileSize > 0 && subtileSize <= 0x80);
        }
        queueCB.SetViewportScissor(cellX * cellWidth + cellMarginX, cellY * cellHeight + cellMarginY, texWidth, texHeight);
        if (valid) {
            queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
        } else {
            queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
        }
        cellY++;
    }
    cellX++;
    cellY = 0;

    // Reset the subtile override to the default (disabled).
    queueCB.OverrideSubtileSize(LWN_FALSE, 0);

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNSubtileOverrideTest, lwn_subtile_override, );
