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

using namespace lwn;

class LWNBlendAdvancedTest
{
    // Render 70x70 quads inside an 8x6 grid of 80x80 cells, each for a
    // different blend mode.  This size gives the closest match to equivalent
    // OpenGL tests without trying to completely replicate their wacky
    // rendering algorithm.
    static const int cellSize = 80;
    static const int cellMargin = 5;
    static const int cellsX = 8;
    static const int cellsY = 6;

    static const BlendAdvancedMode blendModes[];
    static const BlendAdvancedOverlap overlapModes[];

    int m_variant;
    BlendAdvancedOverlap m_overlap;
    bool m_premultiplied;

public:
    LWNBlendAdvancedTest(int variant, BlendAdvancedOverlap overlap, bool premultiplied) :
        m_variant(variant), m_overlap(overlap), m_premultiplied(premultiplied) {}
    LWNTEST_CppMethods();
};

// Enumerate all the blend modes supported in a table, ordering to closely
// match a similar OpenGL test.
const BlendAdvancedMode LWNBlendAdvancedTest::blendModes[] = {
    BlendAdvancedMode::BLEND_ZERO, // 0
    BlendAdvancedMode::BLEND_SRC, // 1
    BlendAdvancedMode::BLEND_DST, // 2
    BlendAdvancedMode::BLEND_SRC_OVER, // 3
    BlendAdvancedMode::BLEND_DST_OVER, // 4
    BlendAdvancedMode::BLEND_SRC_IN, // 5
    BlendAdvancedMode::BLEND_DST_IN, // 6
    BlendAdvancedMode::BLEND_SRC_OUT, // 7
    BlendAdvancedMode::BLEND_DST_OUT, // 8
    BlendAdvancedMode::BLEND_SRC_ATOP, // 9
    BlendAdvancedMode::BLEND_DST_ATOP, // 10
    BlendAdvancedMode::BLEND_XOR, // 11
    BlendAdvancedMode::BLEND_PLUS, // 12
    BlendAdvancedMode::BLEND_PLUS_DARKER, // 13
    BlendAdvancedMode::BLEND_PLUS_CLAMPED, // 14
    BlendAdvancedMode::BLEND_MULTIPLY, // 15
    BlendAdvancedMode::BLEND_SCREEN, // 16
    BlendAdvancedMode::BLEND_OVERLAY, // 17
    BlendAdvancedMode::BLEND_DARKEN, // 18
    BlendAdvancedMode::BLEND_LIGHTEN, // 19
    BlendAdvancedMode::BLEND_COLORDODGE, // 20
    BlendAdvancedMode::BLEND_COLORBURN, // 21
    BlendAdvancedMode::BLEND_HARDLIGHT, // 22
    BlendAdvancedMode::BLEND_SOFTLIGHT, // 23
    BlendAdvancedMode::BLEND_SOFTLIGHT, // 24 // SOFTLIGHT_SVG was removed, replace this with SOFTLIGHT as a placeholder
    BlendAdvancedMode::BLEND_DIFFERENCE, // 25
    BlendAdvancedMode::BLEND_MINUS_CLAMPED, // 26
    BlendAdvancedMode::BLEND_EXCLUSION, // 27
    BlendAdvancedMode::BLEND_CONTRAST, // 28
    BlendAdvancedMode::BLEND_ILWERT, // 29
    BlendAdvancedMode::BLEND_ILWERT_RGB, // 30
    BlendAdvancedMode::BLEND_LINEARDODGE, // 31
    BlendAdvancedMode::BLEND_LINEARBURN, // 32
    BlendAdvancedMode::BLEND_VIVIDLIGHT, // 33
    BlendAdvancedMode::BLEND_LINEARLIGHT, // 34
    BlendAdvancedMode::BLEND_PINLIGHT, // 35
    BlendAdvancedMode::BLEND_HARDMIX, // 36
    BlendAdvancedMode::BLEND_RED, // 37
    BlendAdvancedMode::BLEND_GREEN, // 38
    BlendAdvancedMode::BLEND_BLUE, // 39
    BlendAdvancedMode::BLEND_HSL_HUE, // 40
    BlendAdvancedMode::BLEND_HSL_SATURATION, // 41
    BlendAdvancedMode::BLEND_HSL_COLOR, // 42
    BlendAdvancedMode::BLEND_HSL_LUMINOSITY, // 43
    BlendAdvancedMode::BLEND_PLUS_CLAMPED_ALPHA, // 44
    BlendAdvancedMode::BLEND_MINUS, // 45
    BlendAdvancedMode::BLEND_ILWERT_OVG, // 46
};

lwString LWNBlendAdvancedTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Test exercising advanced blending modes.  This test renders an array of cells, "
        "each of which uses one of the modes.  The test renders a gradient into the "
        "framebuffer as the background color, and then renders a second gradient on "
        "top of the background with blending enabled."
        "\n\n"
        "The background has a gradient between two colors going from bottom to top; the "
        "foreground has a gradient between two colors going from left to right.  ";
    switch (m_variant) {
    case 0:
        sb <<
            "Both gradients go between opaque black to opaque white, to test "
            "the f() functions used to combine two colors.";
        break;
    case 1:
        sb <<
            "Source goes from opaque black to opaque white; destination is "
            "transparent, to test the Y*Cs term.";
        break;
    case 2:
        sb <<
            "Destination goes from opaque black to opaque white; source is "
            "transparent, to test the Z*Cd term.";
        break;
    case 3:
        sb <<
            "Source and destination are both white, going from fully "
            "transparent to fully opaque, to test the p0, p1, and p2 terms.";
        break;
    case 4:
        sb <<
            "Both gradients go between 50% opaque black to 50% opaque white, to test "
            "the f() functions used to combine two colors with less saturation.";
        break;
    case 6:
        sb <<
            "Source and destination both go from an opaque reddish color to "
            "a transparent blue-green.";
        break;
    case 7:
        sb <<
            "Source and destination are both white, going from fully transparent "
            "to fully opaque.  The test displays the alpha result of the blend.";
        break;
    case 8:
        sb <<
            "Source and destination are both white, going from fully transparent "
            "to 70% opaque.  The test displays the alpha result of the blend.";
        break;
    default:
        assert(0);
        break;
    }
    return sb.str();    
}

int LWNBlendAdvancedTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(40, 15) && g_lwnDeviceCaps.supportsAdvancedBlendModes;
}

void LWNBlendAdvancedTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Set up a program that passes through position and color.
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec4 color;\n"
        "out vec4 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
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
    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        LWNFailTest();
        return;
    }

    // Set up colors to be used for the test.  Then extrapolate them a bit so
    // we will test for proper interpolation/clamping at the edges.
    dt::vec4 srcleft, srcright, dstbot, dsttop;
    switch (m_variant) {
    case 0:
        srcleft  = dt::vec4(0,0,0,1);
        srcright = dt::vec4(1,1,1,1);
        dstbot   = dt::vec4(0,0,0,1);
        dsttop   = dt::vec4(1,1,1,1);
        break;
    case 1:
        srcleft  = dt::vec4(0,0,0,1);
        srcright = dt::vec4(1,1,1,1);
        dstbot   = dt::vec4(0,0,0,0);
        dsttop   = dt::vec4(1,1,1,0);
        break;
    case 2:
        srcleft  = dt::vec4(0,0,0,0);
        srcright = dt::vec4(1,1,1,0);
        dstbot   = dt::vec4(0,0,0,1);
        dsttop   = dt::vec4(1,1,1,1);
        break;
    case 3:
        srcleft  = dt::vec4(1,1,1,0);
        srcright = dt::vec4(1,1,1,1);
        dstbot   = dt::vec4(1,1,1,0);
        dsttop   = dt::vec4(1,1,1,1);
        break;
    case 4:
        srcleft  = dt::vec4(0.0,0.0,0.0,0.5);
        srcright = dt::vec4(0.5,0.5,0.5,0.5);
        dstbot   = dt::vec4(0.0,0.0,0.0,0.5);
        dsttop   = dt::vec4(0.5,0.5,0.5,0.5);
        break;
    case 6:
        srcleft  = dt::vec4(1.0,0.0,0.3,1.0);
        srcright = dt::vec4(0.0,1.0,0.8,0.0);
        dstbot   = dt::vec4(1.0,0.0,0.3,1.0);
        dsttop   = dt::vec4(0.0,1.0,0.8,0.0);
        break;
    case 7:
        srcleft  = dt::vec4(1,1,1,0);
        srcright = dt::vec4(1,1,1,1);
        dstbot   = dt::vec4(1,1,1,0);
        dsttop   = dt::vec4(1,1,1,1);
        break;
    case 8:
        srcleft  = dt::vec4(1,1,1,0);
        srcright = dt::vec4(1,1,1,0.7);
        dstbot   = dt::vec4(1,1,1,0);
        dsttop   = dt::vec4(1,1,1,0.7);
        break;
    }

    // All colors above are non-premultiplied.  Multiply destination colors by
    // alpha and source colors by alpha if premultipled source color should be
    // used.  Changing <premultComponents> to 4 (squaring alpha) is wrong, but
    // more closely matches some of the equivalent OpenGL tests.
    int premultComponents = 3;
    if (m_premultiplied) {
        for (int i = 0; i < premultComponents; i++) {
            srcleft[i] = srcleft[i] * srcleft[3];
            srcright[i] = srcright[i] * srcright[3];
        }
    }
    for (int i = 0; i < premultComponents; i++) {
        dsttop[i] = dsttop[i] * dsttop[3];
        dstbot[i] = dstbot[i] * dstbot[3];
    }

    // Adjust the colors to extrapolate slightly outside the range of values,
    // which exercises clamping.
    float adjust = 1.f / 32;
    dt::vec4 color0 = dstbot - (dsttop - dstbot) * adjust;
    dt::vec4 color1 = dsttop + (dsttop - dstbot) * adjust;
    dt::vec4 color2 = srcleft  - (srcright - srcleft) * adjust;
    dt::vec4 color3 = srcright + (srcright - srcleft) * adjust;

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec4 color;
    };
    Vertex vertexData[] = {
        // Background goes from color0 to color1, bottom to top.
        { dt::vec3(-1.0, -1.0, 0.0), color0 },
        { dt::vec3(-1.0, +1.0, 0.0), color1 },
        { dt::vec3(+1.0, -1.0, 0.0), color0 },
        { dt::vec3(+1.0, +1.0, 0.0), color1 },

        // Foreground goes from color2 to color3, left to right.
        { dt::vec3(-1.0, -1.0, 0.0), color2 },
        { dt::vec3(-1.0, +1.0, 0.0), color2 },
        { dt::vec3(+1.0, -1.0, 0.0), color3 },
        { dt::vec3(+1.0, +1.0, 0.0), color3 },

        // "Show alpha" geometry is solid white.
        { dt::vec3(-1.0, -1.0, 0.0), dt::vec4(1,1,1,1) },
        { dt::vec3(-1.0, +1.0, 0.0), dt::vec4(1,1,1,1) },
        { dt::vec3(+1.0, -1.0, 0.0), dt::vec4(1,1,1,1) },
        { dt::vec3(+1.0, +1.0, 0.0), dt::vec4(1,1,1,1) },
    };
    MemoryPoolAllocator allocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 12, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // Set up blend and color state objects for the test loop.
    BlendState bs;
    ColorState cs;
    bs.SetDefaults();
    cs.SetDefaults();

    // Set up blend state to show alpha.
    BlendState bsa;
    bsa.SetDefaults();
    bsa.SetBlendFunc(BlendFunc::DST_ALPHA, BlendFunc::ZERO, BlendFunc::ZERO, BlendFunc::ONE);

    queueCB.ClearColor(0, 0.25, 0.0, 0.0, 0.0);
    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    // Set up non-mode advanced blending state appropriate for the test.
    bs.SetAdvancedOverlap(m_overlap);
    bs.SetAdvancedPremultipliedSrc(m_premultiplied);
    bs.SetAdvancedNormalizedDst(LWN_TRUE);

    // Loop over all the blend modes, combining the two gradients in each.
    CellIterator2D cell(cellsX, cellsY);
    for (LWNuint i = 0; i < __GL_ARRAYSIZE(blendModes); i++) {
        bs.SetAdvancedMode(blendModes[i]);
        queueCB.SetViewportScissor(cell.x() * cellSize + cellMargin,
                                    cell.y() * cellSize + cellMargin,
                                    cellSize - 2 * cellMargin,
                                    cellSize - 2 * cellMargin);
        queueCB.BindBlendState(&bs);
        cs.SetBlendEnable(0, LWN_FALSE);
        queueCB.BindColorState(&cs);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
        cs.SetBlendEnable(0, LWN_TRUE);
        queueCB.BindColorState(&cs);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 4, 4);

        // Tests 7 and 8 want to display the alpha result of the blend, so we
        // blend in a white polygon with a color blend func of (DST_ALPHA,
        // ZERO).
        if (m_variant == 7 || m_variant == 8) {
            queueCB.BindBlendState(&bsa);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 8, 4);
        }

        cell++;
    }

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queueCB.submit();
    queue->Finish();
}

#define MKTEST(suff, num, overlap, premult)                         \
  OGTEST_CppTest(LWNBlendAdvancedTest, lwn_blendadv_##num##suff,    \
                 (num, overlap, premult))

#define MKTESTP(suff, num, overlap)     \
  MKTEST(suff, num, overlap, false)     \
  MKTEST(suff##p, num, overlap, true)

#define MKTESTO(num)                                    \
  MKTESTP(c, num, BlendAdvancedOverlap::CONJOINT)       \
  MKTESTP(d, num, BlendAdvancedOverlap::DISJOINT)       \
  MKTESTP(u, num, BlendAdvancedOverlap::UNCORRELATED)

MKTESTO(0)
MKTESTO(1)
MKTESTO(2)
MKTESTO(3)
MKTESTO(4)
// Skip test set #5.
MKTESTO(6)
MKTESTO(7)
MKTESTO(8)
