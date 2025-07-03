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

#define M_PI 3.14159265358979323846

using namespace lwn;

class LWNGSMultiResTest
{
    // We pick an effective texture size that fits in half of a 640x480
    // window.  The multi-resolution texture is 3/4 the size of a regular
    // texture of the effective size.
    static const int effectiveTexSize = 300;
    static const int multiResTexSize = 3 * effectiveTexSize / 4;
public:
    enum GeometryShaderType { Single, Instanced, Passthrough };
    enum TessellationType { NoTessellation, FixedTessellation };
private:
    GeometryShaderType m_gsType;
    TessellationType m_tessType;
public:
    LWNGSMultiResTest(GeometryShaderType gsType, TessellationType tessType) : 
        m_gsType(gsType), m_tessType(tessType)
    {}
    LWNTEST_CppMethods();
};

lwString LWNGSMultiResTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Basic multi-resolution geometry shader test for LWN.  This test draws "
        "a colorful fan of white, green, and blue wedges to a multi-resolution "
        "texture.  Geometry shaders are used to send the wedges to multiple "
        "viewports arranged in a tic-tac-toe board pattern.  The inner square "
        "is rendered at full resolution; the outer portions are rendered at "
        "half-resolution in one or both directions.  The left side of the screen "
        "displays the multi-resolution texture in its original form with "
        "varying resolutions.  It also draws a white grid showing the viewport "
        "joints.  The right side of the screen shows the multi-resolution texture "
        "reprojected to the original resolution."
        "\n\n";
    switch (m_gsType) {
    case Instanced:
        sb <<
            "This test uses an instanced geometry shader that spawns nine "
            "geometry shader ilwocations per triangle, each of which may send "
            "its triangle to one of the nine viewports.";
        break;
    case Passthrough:
        sb <<
            "This test uses an passthrough geometry shader with multicast that "
            "sends each triangle to multiple viewports using the viewport mask.";
        break;
    case Single:
        sb <<
            "This test uses a regular geometry shader that emits up to nine "
            "triangles, each going to a separate viewport.";
        break;
    default:
        assert(0);
        break;
    }
    switch (m_tessType) {
    case NoTessellation:
        break;
    case FixedTessellation:
        sb <<
            "  This test also enables a tessellation evaluation shader.";
        break;
    default:
        assert(0);
        break;
    }
    return sb.str();    
}

int LWNGSMultiResTest::isSupported() const
{
    if (m_gsType == Passthrough && !g_lwnDeviceCaps.supportsPassthroughGeometryShaders) {
        return 0;
    }
    return lwogCheckLWNAPIVersion(40, 8);
}

void LWNGSMultiResTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Utility vertex shader that passes through position, color, and a
    // texture coordinate.
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 color;\n"
        "layout(location=2) in vec2 texcoord;\n"
        "out Attrs {\n"
        "  vec3 ocolor;\n"
        "  vec2 otc;\n"
        "};\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ocolor = color;\n"
        "  otc = texcoord;\n"
        "}\n";

    // Tessellation shader that passes through a tessellated primitive.  We
    // flat-shade color to match the flat-shaded color from non-tessellated
    // primitives.
    TessEvaluationShader tesBasic(440);
    tesBasic.setTESMode(GL_TRIANGLES);
    tesBasic <<
        "in Attrs {\n"
        "  vec3 ocolor;\n"
        "  vec2 otc;\n"
        "} attr_in[];\n"
        "out Attrs {\n"
        "  vec3 ocolor;\n"
        "  vec2 otc;\n"
        "} attr_out;\n"
        "void main() {\n"
        "  gl_Position = (gl_in[0].gl_Position * gl_TessCoord.x +\n"
        "                 gl_in[1].gl_Position * gl_TessCoord.y +\n"
        "                 gl_in[2].gl_Position * gl_TessCoord.z);\n"
        "  attr_out.ocolor = attr_in[2].ocolor;\n"
        "  attr_out.otc = (attr_in[0].otc * gl_TessCoord.x +\n"
        "                  attr_in[1].otc * gl_TessCoord.y +\n"
        "                  attr_in[2].otc * gl_TessCoord.z);\n"
        "}\n";

    // Geometry shader to do the multi-viewport transformation.  We pass through
    // attributes, and also write out a "computed" per-primitive color that is
    // just a flat-shaded version of the original color.
    GeometryShader gs(440);
    const char *optPassthrough = (m_gsType == Passthrough) ? "layout(passthrough) " : "";
    gs.setGSInputType(GL_TRIANGLES);
    if (m_gsType != Passthrough) {
        gs.setGSOutputType(GL_TRIANGLE_STRIP);
    }
    gs <<
        "in gl_PerVertex {\n"
        "  " << optPassthrough << "vec4 gl_Position;\n"
        "} gl_in[];\n" <<
        optPassthrough << "in Attrs {\n"
        "  vec3 ocolor;\n"
        "  vec2 otc;\n"
        "} attr_in[];\n"
        "out vec3 primcolor;\n";

    // Passthrough geometry shaders don't copy attributes explicitly.  Instead,
    // they use the "passthrough" layout qualifier to do the copy.
    if (m_gsType != Passthrough) {
        gs <<
            "out Attrs {\n"
            "  vec3 ocolor;\n"
            "  vec2 otc;\n"
            "} attr_out;\n";
    }

    switch (m_gsType) {
    case Passthrough:
        gs.addExtension(lwShaderExtension::LW_geometry_shader_passthrough);
        gs.addExtension(lwShaderExtension::LW_viewport_array2);
        gs << 
            "void main() {\n"
            "  gl_ViewportMask[0] = 0x1FF;\n"
            "  primcolor = attr_in[2].ocolor;\n"
            "}\n";
        break;
    case Instanced:
        gs.setGSVerticesOut(3);
        gs.setGSIlwocations(9);
        gs <<
            "void main() {\n"
            "  for (int j = 0; j < 3; j++) {\n"
            "    gl_ViewportIndex = gl_IlwocationID;\n"
            "    gl_Position = gl_in[j].gl_Position;\n"
            "    attr_out.ocolor = attr_in[j].ocolor;\n"
            "    primcolor = attr_in[2].ocolor;\n"
            "    EmitVertex();\n"
            "  }\n"
            "}\n";
        break;
    case Single:
        gs.setGSVerticesOut(27);
        gs <<
            "void main() {\n"
            "  for (int i = 0; i < 9; i++) {\n"
            "    for (int j = 0; j < 3; j++) {\n"
            "      gl_ViewportIndex = i;\n"
            "      gl_Position = gl_in[j].gl_Position;\n"
            "      attr_out.ocolor = attr_in[j].ocolor;\n"
            "      primcolor = attr_in[2].ocolor;\n"
            "      EmitVertex();\n"
            "    }\n"
            "    EndPrimitive();\n"
            "  }\n"
            "}\n";
        break;
    default:
        assert(0);
        break;
    }

    // Fragment shader to simply display a passed-through color value.
    FragmentShader fsColor(440);
    fsColor <<
        "in Attrs {\n"
        "  flat vec3 ocolor;\n"
        "  vec2 otc;\n"
        "};\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(ocolor, 1.0);\n"
        "}\n";

    // Variant of <fsColor> to work with geometry shaders that write out a
    // per-primitive color <primcolor>.  We replace red per-primitive colors
    // with white.  This tests correct attribute passing of computed
    // passthrough geometry shader outputs to the fragment shader -- the
    // original wedges are red, green, and blue, and we should instead end up
    // with white, green, and blue.
    FragmentShader fsColorRedWithWhite(440);
    fsColorRedWithWhite <<
        "in Attrs {\n"
        "  flat vec3 ocolor;\n"
        "  vec2 otc;\n"
        "};\n"
        "flat in vec3 primcolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  if (primcolor.r == 1.0) {\n"
        "    fcolor = vec4(1.0);\n"
        "  } else {\n"
        "    fcolor = vec4(ocolor, 1.0);\n"
        "  }\n"
        "}\n";

    // This fragment shader displays the multi-resolution texture directly
    // with texture coordinates from [-1,+1] mapped to [0,1].  Does not
    // account for the distortion from multiple viewports.
    FragmentShader fsBasicDraw(440);
    fsBasicDraw <<
        "layout(binding = 0) uniform sampler2D multiResTex;\n"
        "in Attrs {\n"
        "  flat vec3 ocolor;\n"
        "  vec2 otc;\n"
        "};\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  vec2 tc = 0.5 * otc + 0.5;\n"
        "  fcolor = texture(multiResTex, tc);\n"
        "}\n";

    // Fragment shader used to do a corrected lookup in the multi-resolution
    // texture, where the viewport (in NDCs) is mapped as:
    //
    // x in [-1.0, -0.5]:  Half resolution, takes 1/6 of the multi-res texture.
    // x in [-0.5, +0.5]:  Full resolution, takes 2/3 of the multi-res texture.
    // x in [+0.5, +1.0]:  Half resolution, takes 1/6 of the multi-res texture.
    //
    // We transform the <x> texture coordinate from [-1,+1] to [0,1] using:
    //
    //   if (x < -0.5)       x' = (x + 1) / 3
    //   else if (x < +0.5)  x' = (4x + 3) / 6
    //   else                x' = (x + 2) / 3
    //
    FragmentShader fsMultiResDraw(440);
    fsMultiResDraw <<
        "layout(binding = 0) uniform sampler2D multiResTex;\n"
        "in Attrs {\n"
        "  flat vec3 ocolor;\n"
        "  vec2 otc;\n"
        "};\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  vec2 tc;\n"
        "  if (otc.x < -0.5) {\n"
        "    tc.x = (otc.x + 1.0) / 3.0;\n"
        "  } else if (otc.x < +0.5) {\n"
        "    tc.x = (otc.x * 4.0 + 3.0) / 6.0;\n"
        "  } else {\n"
        "    tc.x = (otc.x + 2.0) / 3.0;\n"
        "  }\n"
        "  if (otc.y < -0.5) {\n"
        "    tc.y = (otc.y + 1.0) / 3.0;\n"
        "  } else if (otc.y < +0.5) {\n"
        "    tc.y = (otc.y * 4.0 + 3.0) / 6.0;\n"
        "  } else {\n"
        "    tc.y = (otc.y + 2.0) / 3.0;\n"
        "  }\n"
        "  fcolor = texture(multiResTex, tc);\n"
        "}\n";

    // Draw colored outputs using a GS to the multi-resolution texture.
    Program *drawProgram = device->CreateProgram();
    Shader shaders[4];
    int nShaders = 3;
    shaders[0] = vs;
    shaders[1] = gs;
    shaders[2] = fsColorRedWithWhite;
    if (m_tessType == FixedTessellation) {
        // For the tessellation variant, plug in the tessellation shader and
        // enable the combined tessellation / passthrough geometry shader flag
        // in the compiler.
        shaders[3] = tesBasic;
        nShaders = 4;
        g_glslcHelper->EnableTessellationAndPassthroughGS(LWN_TRUE);
    }
    LWNboolean compiled = g_glslcHelper->CompileAndSetShaders(drawProgram, nShaders, shaders);
    if (m_tessType == FixedTessellation) {
        g_glslcHelper->EnableTessellationAndPassthroughGS(LWN_FALSE);
    }
    if (!compiled) {
        LWNFailTest();
        return;
    }

    // Draw colored outputs at regular resolution (used for grid lines).
    Program *drawProgramNoGS = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(drawProgramNoGS, vs, fsColor)) {
        LWNFailTest();
        return;
    }

    // Draw the multi-resolution texture without correction for resolution.
    Program *multiResDrawNoCorrectionProgram = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(multiResDrawNoCorrectionProgram, vs, fsBasicDraw)) {
        LWNFailTest();
        return;
    }

    // Draw the multi-resolution texture with correction for resolution.
    Program *multiResDrawCorrectionProgram = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(multiResDrawCorrectionProgram, vs, fsMultiResDraw)) {
        LWNFailTest();
        return;
    }

    MemoryPoolAllocator vboAllocator(device, NULL, 64 * 1024, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    MemoryPoolAllocator texAllocator(device, NULL, 1024 * 1024, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    // Set up the vertex format for this test.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
        dt::vec2 texcoord;
    };
    Vertex *v;
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, texcoord);
    VertexArrayState vertex = stream.CreateVertexArrayState();

    // Set up a VBO to draw the wedges into the multi-resolution texture.
    static const int nWedges = 20;
    static const int nWedgeVertices = nWedges + 2;
    Buffer *drawVBO = stream.AllocateVertexBuffer(device, nWedgeVertices, vboAllocator, NULL);
    BufferAddress drawVBOAddr = drawVBO->GetAddress();
    Vertex *drawVBOVertices = (Vertex *) drawVBO->Map();
    v = drawVBOVertices;
    v->position = dt::vec3(0.0);
    v->color = dt::vec3(0.0);
    v->texcoord = dt::vec2(0.5, 0.0);
    v++;
    for (int i = 0; i <= nWedges; i++) {
        v->position = dt::vec3(1.0 * cos((i * 2.0 * M_PI) / nWedges), 1.0 * sin((i * 2 * M_PI) / nWedges), 0.0);
        v->color = dt::vec3(0.0);
        v->color[i % 3] = 1.0;
        v->texcoord = dt::vec2(i, 1.0);
        v++;
    }

    // Set up an index buffer to draw the wedges.  We use indexed draws so we
    // can use the same indices for separate triangles or triangular
    // tessellation patches (which can't be strips).
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *drawIBO = vboAllocator.allocBuffer(&bb, BUFFER_ALIGN_INDEX_BIT, nWedges * 3 * sizeof(uint16_t));
    BufferAddress drawIBOAddr = drawIBO->GetAddress();
    uint16_t *drawIBOIndices = (uint16_t *) drawIBO->Map();
    for (int i = 0; i < nWedges; i++) {
        drawIBOIndices[3 * i + 0] = 0;
        drawIBOIndices[3 * i + 1] = i + 1;
        drawIBOIndices[3 * i + 2] = i + 2;
    }

    // Set up a VBO to draw a full-screen quad for displaying the texture (4
    // vertices) and four grid lines to show the internal joints in the
    // multi-resolution texture (8 vertices).
    Buffer *displayVBO = stream.AllocateVertexBuffer(device, 12, vboAllocator, NULL);
    BufferAddress displayVBOAddr = displayVBO->GetAddress();
    Vertex *displayVBOVertices = (Vertex *) displayVBO->Map();
    v = displayVBOVertices;

    // The first four vertices here are for a full-screen quad.
    for (int i = 0; i < 4; i++) {
        v->position = dt::vec3(((i & 2) ? +1.0 : -1.0), ((i & 1) ? +1.0 : -1.0), 0.0);
        v->color = dt::vec3(1.0);
        v->texcoord = dt::vec2(v->position[0], v->position[1]);
        v++;
    }

    // The next eight vertices are for grid lines at the joints of the
    // multi-line texture.
    for (int i = 0; i < 8; i++) {
        dt::vec3 position = dt::vec3(-2.0 / 3.0, -1.0, 0);
        if (i & 1) position[1] = +1.0;
        if (i & 2) position[0] = 2.0 / 3.0;
        if (i & 4) position = dt::vec3(position[1], position[0], 0.0);
        v->position = position;
        v->color = dt::vec3(1.0);
        v->texcoord = dt::vec2(v->position[0], v->position[1]);
        v++;
    }

    // Set up the viewport transformations and scissors for the
    // multi-resolution textures.  We send the same primitives into nine
    // viewports arranged in a 3x3 grid.  For columns, we have:
    //
    // x in [-1.0, -0.5]:  Half resolution, takes 1/6 of the texture.
    // x in [-0.5, +0.5]:  Full resolution, takes 4/6 of the texture.
    // x in [+0.5, +1.0]:  Half resolution, takes 1/6 of the texture.
    //
    // Each viewport transformation should map its own range of the input
    // coordinates into the corresponding range of the multi-res texture.  The
    // logic for mapping the original [-1,+1] range to a normalized [0,1]
    // range on screen is the same as that used to re-project texture
    // coordinates for multi-resolution:
    //
    //   if (x < -0.5)       f(x) = (x + 1) / 3
    //   else if (x < +0.5)  f(x) = (4x + 3) / 6
    //   else                f(x) = (x + 2) / 3
    //
    // The viewports would then be programmed as:
    //
    //   left edge = multiResTexSize * f(-1.0)
    //   width     = multiResTexSize * (f(+1.0) - f(-1.0))
    //
    // Scissors are applied to each viewport so we only draw the portion of
    // the primitive that hits the viewport.
    LWNfloat viewports[36];
    LWNint   scissors[36];
    static const float viewportEdges[3]  = { 0.0 / 3.0, -1.0 / 6.0, +1.0 / 3.0 };
    static const float viewportWidths[3] = { 2.0 / 3.0,  4.0 / 3.0,  2.0 / 3.0 };
    static const float scissorEdges[3]   = { 0.0 / 3.0,  1.0 / 6.0,  5.0 / 6.0 };
    static const float scissorWidths[3]  = { 1.0 / 6.0,  4.0 / 6.0,  1.0 / 6.0 };
    for (int i = 0; i < 9; i++) {
        int col = (i % 3);
        int row = (i / 3);
        LWNfloat *viewport = viewports + 4 * i;
        LWNint *scissor = scissors + 4 * i;
        viewport[0] = multiResTexSize * viewportEdges[col];
        viewport[1] = multiResTexSize * viewportEdges[row];
        viewport[2] = multiResTexSize * viewportWidths[col];
        viewport[3] = multiResTexSize * viewportWidths[row];
        scissor[0] = LWNint(multiResTexSize * scissorEdges[col] + 0.5);
        scissor[1] = LWNint(multiResTexSize * scissorEdges[row] + 0.5);
        scissor[2] = LWNint(multiResTexSize * scissorWidths[col] + 0.5);
        scissor[3] = LWNint(multiResTexSize * scissorWidths[row] + 0.5);
    }

    // Set up the multi-resolution texture, a simple sampler, and a texture
    // handle so the texture can be displayed.
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetSize2D(multiResTexSize, multiResTexSize);
    tb.SetFlags(TextureFlags::COMPRESSIBLE);
    tb.SetFormat(Format::RGBA8);
    tb.SetLevels(1);
    Texture *multiResTex = texAllocator.allocTexture(&tb);

    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetWrapMode(WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE);
    sb.SetMinMagFilter(MinFilter::LINEAR, MagFilter::LINEAR);
    Sampler *smp = sb.CreateSampler();

    TextureHandle multiResTexHandle = device->GetTextureHandle(multiResTex->GetRegisteredTextureID(), smp->GetRegisteredID());

    // Set up tessellation levels and primitive types for the tessellation tests.
    DrawPrimitive drawPrimitive = DrawPrimitive::TRIANGLES;
    if (m_tessType == FixedTessellation) {
        const float tessLevels[4] = { 3, 3, 3, 3 };
        queueCB.SetPatchSize(3);
        queueCB.SetInnerTessellationLevels(tessLevels);
        queueCB.SetOuterTessellationLevels(tessLevels);
        drawPrimitive = DrawPrimitive::PATCHES;
    }

    // All primitives use the same vertex attribute/stream state.
    queueCB.BindVertexArrayState(vertex);

    // Render the first pass to the multi-resolution texture using the draw
    // program.
    queueCB.SetRenderTargets(1, &multiResTex, NULL, NULL, NULL);
    queueCB.SetViewportScissor(0, 0, multiResTexSize, multiResTexSize);
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.BindProgram(drawProgram, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexBuffer(0, drawVBOAddr, nWedgeVertices * sizeof(Vertex));
    queueCB.SetViewports(0, 9, viewports);
    queueCB.SetScissors(0, 9, scissors);
    queueCB.DrawElements(drawPrimitive, IndexType::UNSIGNED_SHORT, 3 * nWedges, drawIBOAddr);

    // Insert a render-to-texture barrier.
    queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS | BarrierBits::ILWALIDATE_TEXTURE);

    // Clear the window to dark red.
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);

    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, multiResTexHandle);
    queueCB.BindVertexBuffer(0, displayVBOAddr, 12 * sizeof(Vertex));

    // Display the multi-resolution texture without correction on the left
    // side and then display grid lines on top of the multi-resolution
    // texture.
    queueCB.BindProgram(multiResDrawNoCorrectionProgram, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.SetViewportScissor((effectiveTexSize - multiResTexSize) / 2,
                               (lwrrentWindowHeight - multiResTexSize) / 2,
                               multiResTexSize, multiResTexSize);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
    queueCB.BindProgram(drawProgramNoGS, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.DrawArrays(DrawPrimitive::LINES, 4, 8);

    // Display the multi-resolution texture on the right side with correction.
    queueCB.BindProgram(multiResDrawCorrectionProgram, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.SetViewportScissor(lwrrentWindowWidth - effectiveTexSize, 
                               (lwrrentWindowHeight - effectiveTexSize) / 2,
                               effectiveTexSize, effectiveTexSize);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNGSMultiResTest, lwn_gs_multires_inst, (LWNGSMultiResTest::Instanced, LWNGSMultiResTest::NoTessellation));
OGTEST_CppTest(LWNGSMultiResTest, lwn_gs_multires_pass, (LWNGSMultiResTest::Passthrough, LWNGSMultiResTest::NoTessellation));
OGTEST_CppTest(LWNGSMultiResTest, lwn_gs_multires_single, (LWNGSMultiResTest::Single, LWNGSMultiResTest::NoTessellation));

OGTEST_CppTest(LWNGSMultiResTest, lwn_gs_multires_inst_tess, (LWNGSMultiResTest::Instanced, LWNGSMultiResTest::FixedTessellation));
OGTEST_CppTest(LWNGSMultiResTest, lwn_gs_multires_pass_tess, (LWNGSMultiResTest::Passthrough, LWNGSMultiResTest::FixedTessellation));
OGTEST_CppTest(LWNGSMultiResTest, lwn_gs_multires_single_tess, (LWNGSMultiResTest::Single, LWNGSMultiResTest::FixedTessellation));

//////////////////////////////////////////////////////////////////////////

class LWNGSLwbeMapTest
{
    // Size of each face in our lwbe map.
    static const int lwbeTexSize = 200;

    // We render 100x100 cells for different projections of the lwbe map.
    static const int cellMargin = 4;
    static const int cellSize = 100;
    static const int cellsX = 6;
    static const int cellsY = 4;

public:
    enum GeometryShaderType { Single, Instanced, Passthrough };
private:
    GeometryShaderType m_gsType;

    // Maximum sizes of vertex and index buffers.
    static const int maxVertices = 8192;
    static const int maxIndices = 8192;

    // Maximum vertex count for showing a portion of the lwbe in each cell.
    static const int maxLwbeVertices = cellsX * cellsY * 4;

    // Vertex format used for rendering the scene.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 normal;
        dt::vec3 material;
        dt::vec3 tesslevels;    // per-primitived fixed tessellation level (not implemented)
    };

    // Utility class for aclwmulating primitives into vertex/index buffers.
    class DrawData
    {
        Vertex          *m_vboPtr;
        uint16_t        *m_iboPtr;
        BufferAddress   m_iboAddr;
        int             m_vboCount;
        int             m_iboCount;
        dt::vec3        m_material;     // current material color, added to each vertex
    public:
        DrawData(Vertex *vbo, uint16_t *ibo, BufferAddress iboAddr) :
            m_vboPtr(vbo), m_iboPtr(ibo), m_iboAddr(iboAddr), m_vboCount(0),
            m_iboCount(0), m_material(dt::vec3(0.8)) {}

        void setMaterial(float r, float g, float b)
        {
            m_material = dt::vec3(r,g,b);
        }

        void addGround(int tesslvl, float x1, float x2, float z1, float z2, float y);
        void addStrip(int tesslvl,
                      double cx, double cz, double y1, double y2, double r1, double r2,
                      double snxz1, double ny1, double snxz2, double ny2, int slices);
        void addTree(int tesslvl, float x, float y, float z, float htrunk, float rtrunk,
                     float hleaves, float rleaves, int slices);
        void addSphere(int tesslvl, float x, float y, float z, float size, int slices, int stacks);
        void draw(QueueCommandBuffer &queueCB, DrawPrimitive drawPrimitive);

        void checkOverflow(int maxVertices, int maxIndices)
        {
            assert(m_vboCount <= maxVertices && m_iboCount <= maxIndices);
        }
    };

    // Utility class for 3x3 matrix operations.  Note that we use "double" and curious
    // formulation of math expressions in an attempt to coerce the Microsoft compiler
    // for x86 and x64 to generate the same code.
    class dmat3 {
    public:
        double f[3][3];
    public:
        dmat3() {}
        dmat3(float v) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    f[i][j] = v;
                }
            }
        }
        static dmat3 identity()
        {
            dmat3 r;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    r.f[i][j] = (i==j) ? 1.0 : 0.0;
                }
            }
            return r;
        }

        static dmat3 rotation(float angle, float x, float y, float z)
        {
            dmat3 r;
            angle *= (M_PI * 2.0 / 360.0);
            double cosine = cos(angle);
            double sine = sin(angle);
            dt::vec3 axis(x, y, z);
            double len = sqrt(x*x + y*y + z*z);
            axis = axis / len;
            double ab = axis[0] * axis[1] * (1 - cosine);
            double bc = axis[1] * axis[2] * (1 - cosine);
            double ca = axis[2] * axis[0] * (1 - cosine);
            double tx = axis[0] * axis[0];
            double ty = axis[1] * axis[1];
            double tz = axis[2] * axis[2];

            r.f[0][0] = tx + cosine * (1 - tx);
            r.f[0][1] = ab + axis[2] * sine;
            r.f[0][2] = ca - axis[1] * sine;
            r.f[1][0] = ab - axis[2] * sine;
            r.f[1][1] = ty + cosine * (1 - ty);
            r.f[1][2] = bc + axis[0] * sine;
            r.f[2][0] = ca + axis[1] * sine;
            r.f[2][1] = bc - axis[0] * sine;
            r.f[2][2] = tz + cosine * (1 - tz);
            return r;
        }

        dmat3 operator *(const dmat3 &m) const
        {
            dmat3 r(0);
            for (int row = 0; row < 3; row++) {
                for (int col = 0; col < 3; col++) {
                    double p0 = f[row][0] * m.f[0][col];
                    double p1 = f[row][1] * m.f[1][col];
                    double p2 = f[row][2] * m.f[2][col];
                    p0 = p0 + p1;
                    p0 = p0 + p2;
                    r.f[row][col] = p0;
                }
            }
            return r;
        }

        dt::vec3 operator *(const dt::vec3 &v) const
        {
            dt::vec3 r;
            for (int row = 0; row < 3; row++) {
                double p0 = f[row][0] * v[0];
                double p1 = f[row][1] * v[1];
                double p2 = f[row][2] * v[2];
                p0 = p0 + p1;
                p0 = p0 + p2;
                r[row] = p0;
            }
            return r;
        }
    };

public:
    LWNGSLwbeMapTest(GeometryShaderType gsType) : m_gsType(gsType) {}
    LWNTEST_CppMethods();
};

lwString LWNGSLwbeMapTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Basic single-pass geometry shader lwbemap rendering test for LWN."
        "\n\n";
    switch (m_gsType) {
    case Instanced:
        sb <<
            "This test uses an instanced geometry shader that spawns six "
            "geometry shader ilwocations per triangle, each projecting the "
            "primitive to a different lwbe face (layer).";
        break;
    case Passthrough:
        sb <<
            "This test uses an passthrough geometry shader with multicast that "
            "sends each triangle to multiple viewports using the viewport mask.  "
            "The viewport swizzle is used to re-project the triangle for each "
            "viewport onto a lwbe face.";
        break;
    case Single:
        sb <<
            "This test uses a regular geometry shader that emits up to six "
            "triangles, each projecting to a different lwbe face/layer.";
        break;
    default:
        assert(0);
    }
    return sb.str();
}

int LWNGSLwbeMapTest::isSupported() const
{
    if (m_gsType == Passthrough && !g_lwnDeviceCaps.supportsPassthroughGeometryShaders) {
        return 0;
    }
    if (m_gsType == Passthrough && !g_lwnDeviceCaps.supportsViewportSwizzle) {
        return 0;
    }
    return lwogCheckLWNAPIVersion(40, 8);
}

// Add some ground to the scene from (x1,z1) to (x2,z2) with constant y.
void LWNGSLwbeMapTest::DrawData::addGround(int tesslvl, float x1, float x2, float z1, float z2, float y)
{
    Vertex v;
    v.material = m_material;
    v.normal = dt::vec3(0.0, 1.0, 0.0);
    v.tesslevels = dt::vec3(tesslvl);

    Vertex *vbo = m_vboPtr + m_vboCount;
    uint16_t *ibo = m_iboPtr + m_iboCount;
    v.position = dt::vec3(x1, y, z1);
    vbo[0] = v;
    v.position = dt::vec3(x1, y, z2);
    vbo[1] = v;
    v.position = dt::vec3(x2, y, z1);
    vbo[2] = v;
    v.position = dt::vec3(x2, y, z2);
    vbo[3] = v;

    ibo[0] = m_vboCount;
    ibo[1] = m_vboCount + 1;
    ibo[2] = m_vboCount + 3;
    ibo[3] = m_vboCount;
    ibo[4] = m_vboCount + 3;
    ibo[5] = m_vboCount + 2;

    m_vboCount += 4;
    m_iboCount += 6;
}

// Draw a quad strip to display one cirlwlar layer of a tessellated sphere,
// cone, or other quadric.  This draws a vertical stack with Y going from <y1>
// to <y2> centered at (<cx>,*,<cz>).  The radius of the bottom/top are <r1>
// and <r2>.  The Y components of the normals at bottom/top are given by <ny1>
// and <ny2>.  The X/Z components of the normals computed relative to the
// circle on their edge and are then scaled by <snxz1> and <snxz2>.  <slices>
// indicates how finely the bottom/top circles are tessellated.
void LWNGSLwbeMapTest::DrawData::addStrip
    (int tesslvl,
     double cx, double cz, double y1, double y2, double r1, double r2,
     double snxz1, double ny1, double snxz2, double ny2, int slices)
{
    Vertex v;
    int slice;
    double dx, dz;
    double lodmin, lodmax, rval;

    v.material = m_material;
    for (slice = 0; slice < slices; slice++) {
        double dx2, dz2;
        if (slice == 0) {
            dx = 1.0;
            dz = 0.0;
        } else {
            dx = cos(2 * M_PI * slice / slices);
            dz = sin(2 * M_PI * slice / slices);
        }
        if (slice == (slices - 1)) {
            dx2 = 1.0;
            dz2 = 0.0;
        } else {
            dx2 = cos(2 * M_PI * (slice + 1) / slices);
            dz2 = sin(2 * M_PI * (slice + 1) / slices);
        }

        // For random tessellation (tesslvl == -1), set up our tessellation
        // LODs for the two triangles of each quad.
        if (tesslvl < 0) {
            rval = lwFloatRand(0, 1);
            if (rval < 0.1) {
                // 10% chance of picking random large LODs
                lodmin = 6.0;
                lodmax = 12.0;
            } else if (rval < 0.4) {
                // 30% chance of picking LOD 1
                lodmin = 1.0;
                lodmax = 1.0;
            } else if (rval < 0.7) {
                // 30% chance of picking random LODs in [1,3]
                lodmin = 1.0;
                lodmax = 3.0;
            } else {
                // 30% chance of picking random LODs in [1,6]
                lodmin = 1.0;
                lodmax = 6.0;
            }
            v.tesslevels = dt::vec3(lwFloatRand(lodmin, lodmax), lwFloatRand(lodmin, lodmax), lwFloatRand(lodmin, lodmax));
        }

        // B+-----+D   For the old test using QUAD_STRIP,
        //  |    /|    we sent A / B for each slice and
        //  |   / |    hardware draws ABD and ADC.  
        //  |  /  |
        //  | /   |    Do something similar for independent
        //  |/    |    triangles
        // A+-----+C
        Vertex *vbo = m_vboPtr + m_vboCount;
        uint16_t *ibo = m_iboPtr + m_iboCount;
        v.normal = dt::vec3(dx * snxz1, ny1, dz * snxz1);
        v.position = dt::vec3(cx + r1*dx, y1, cz + r1*dz);
        vbo[0] = v;
        v.normal = dt::vec3(dx * snxz2, ny2, dz * snxz2);
        v.position = dt::vec3(cx + r2*dx, y2, cz + r2*dz);
        vbo[1] = v;
        v.normal = dt::vec3(dx2 * snxz1, ny1, dz2 * snxz1);
        v.position = dt::vec3(cx + r1*dx2, y1, cz + r1*dz2);
        vbo[2] = v;
        v.normal = dt::vec3(dx2 * snxz2, ny2, dz2 * snxz2);
        v.position = dt::vec3(cx + r2*dx2, y2, cz + r2*dz2);
        vbo[3] = v;

        ibo[0] = m_vboCount;
        ibo[1] = m_vboCount + 1;
        ibo[2] = m_vboCount + 3;
        ibo[3] = m_vboCount;
        ibo[4] = m_vboCount + 3;
        ibo[5] = m_vboCount + 2;

        m_vboCount += 4;
        m_iboCount += 6;
    }
}

// Draw a tree with a base centered at (x,y,z).  The trunk's height and radius
// are <htrunk> and <rtrunk>.  The "leaves" are a cone with a height and base
// radius of <hleaves> and <rleaves>.  <slices> indicates how finely the trunk
// and "leaves" are tessellated.
void LWNGSLwbeMapTest::DrawData::addTree
    (int tesslvl, float x, float y, float z, float htrunk, float rtrunk,
     float hleaves, float rleaves, int slices)
{
    // Compute the slope of the cone, used to scale normals.
    float coneNY = rleaves / sqrt(hleaves*hleaves + rleaves*rleaves);
    float coneNXZ = hleaves / sqrt(hleaves*hleaves + rleaves*rleaves);

    // Draw the trunk.
    setMaterial(0.6, 0.3, 0.1);
    addStrip(tesslvl, x, z, y, y + htrunk, rtrunk, rtrunk, 1, 0, 1, 0, slices);

    // Draw the "leaves".  There are two parts - the bottom of the cone
    // (normals pointing down) and the cone itself.
    setMaterial(0.2, 0.8, 0.2);
    addStrip(tesslvl, x, z, y + htrunk, y + htrunk, rtrunk, rleaves, 0, -1, 0, -1, slices);
    addStrip(tesslvl, x, z, y + htrunk, y + htrunk + hleaves, rleaves, 0, coneNXZ, coneNY, coneNXZ, coneNY, slices);
}

// Draw a sphere centered at (x,y,z) of radius <size>.  <stacks> gives the
// number of vertical slices of the sphere, and <slices> gives how finely
// subdivided each slice is.
void LWNGSLwbeMapTest::DrawData::addSphere(int tesslvl, float x, float y, float z, float size, int slices, int stacks)
{
    double sy1, sxz1, sy2, sxz2;
    int stack;
    sy2 = -1.0;
    sxz2 = 0.0;
    for (stack = 1; stack <= stacks; stack++) {
        sy1 = sy2;
        sxz1 = sxz2;
        sy2 = -cos(stack * M_PI / stacks);
        sxz2 = sin(stack * M_PI / stacks);
        if (stack == stacks) {
            sy2 = +1;
            sxz2 = 0;
        }
        addStrip(tesslvl, x, z, y + size*sy1, y + size*sy2, size*sxz1, size*sxz2,
                 sxz1, sy1, sxz2, sy2, slices);
    }
}

// Render the scene using the embedded vertex and index buffers.
void LWNGSLwbeMapTest::DrawData::draw(QueueCommandBuffer &queueCB, DrawPrimitive drawPrimitive)
{
    queueCB.DrawElements(drawPrimitive, IndexType::UNSIGNED_SHORT, m_iboCount, m_iboAddr);
}

void LWNGSLwbeMapTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Utility vertex shader that passes through position, color, and a
    // texture coordinate.
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 normal;\n"
        "layout(location=2) in vec3 material;\n"
        "layout(location=3) in vec3 tesslevels;\n"
        "out Attrs {\n"
        "  vec3 color;\n"
        "  vec3 tesslevels;\n"
        "} ov;\n"
        "void main() {\n"
        
        // Scale positions by 256 to avoid clipping with 1/W buffering.  By
        // default, our projection uses Z=1 and W=+/-(x,y,z) depending on the
        // face.  Default clipping will still clip post-projection Z/W to
        // [-1,+1].  If we use Z=1 and W=+x in our projection, this clipping
        // will accept only 1/x <= 1, or x >= 1.  Scaling by 256 puts the near
        // plane at 1/256 instead.
        "  gl_Position = vec4(position * 256.0, 1.0);\n"

        // Stupid diffuse lighting callwlation.
        "  float f = clamp(0.2 + 1.2 * dot(normal, normalize(vec3(0.1, 0.3, +0.9))), 0, 1);\n"

        "  ov.color = f * material;\n"
        "  ov.tesslevels = tesslevels;\n"
        "}\n";

    // Build a geometry shader to project onto all faces.
    const char *optPassthrough = (m_gsType == Passthrough) ? "layout(passthrough) " : "";
    GeometryShader lwbeGS(440);
    lwbeGS.setGSInputType(GL_TRIANGLES);
    if (m_gsType != Passthrough) {
        lwbeGS.setGSOutputType(GL_TRIANGLE_STRIP);
    }
    lwbeGS <<
        "in gl_PerVertex {\n"
        "  " << optPassthrough << "vec4 gl_Position;\n"
        "} gl_in[];\n" <<
        "in Attrs {\n"
        "  " << optPassthrough << "vec3 color;\n"
        "  vec3 tesslevels;\n"
        "} iv[];\n";
    switch (m_gsType) {
    case Passthrough:
        lwbeGS.addExtension(lwShaderExtension::LW_geometry_shader_passthrough);
        lwbeGS.addExtension(lwShaderExtension::LW_viewport_array2);
        lwbeGS <<
            // Passthrough shaders just use viewport mask to broadcast to six
            // viewports and rely on viewport swizzle to re-project.  The
            // "viewport_relative" layout qualifier here says that viewport N
            // will also go to layer (lwbe face) N.
            "layout(viewport_relative) out int gl_Layer;\n"
            "void main() {\n"
            "  gl_Layer = 0;\n"
            "  gl_ViewportMask[0] = 0x3F;\n"
            "}\n";
        break;
    case Instanced:
    case Single:
        // Non-passthrough shaders make one (instanced) or six (single) copies
        // of the triangle, each projected onto a different face.
        if (m_gsType == Instanced) {
            lwbeGS.setGSVerticesOut(3);
            lwbeGS.setGSIlwocations(6);
        } else {
            lwbeGS.setGSVerticesOut(18);
        }
        lwbeGS <<
            "out Attrs {\n"
            "  vec3 color;\n"
            "} ov;\n"
            "void main() {\n";
        if (m_gsType == Single) {
            lwbeGS << " for (int iid = 0; iid < 6; iid++) {\n";
        } else {
            lwbeGS << "  int iid = gl_IlwocationID;\n";
        }
        lwbeGS <<
            "  for (int j = 0; j < 3; j++) {\n"
            "    vec4 ipos = gl_in[j].gl_Position;\n"
            // Re-orient position according to the selected lwbe face.
            "    switch (iid) {\n"
            "    case 0:  gl_Position = vec4(-ipos.z, -ipos.y, ipos.w, +ipos.x); break;\n"
            "    case 1:  gl_Position = vec4(+ipos.z, -ipos.y, ipos.w, -ipos.x); break;\n"
            "    case 2:  gl_Position = vec4(+ipos.x, +ipos.z, ipos.w, +ipos.y); break;\n"
            "    case 3:  gl_Position = vec4(+ipos.x, -ipos.z, ipos.w, -ipos.y); break;\n"
            "    case 4:  gl_Position = vec4(+ipos.x, -ipos.y, ipos.w, +ipos.z); break;\n"
            "    case 5:  gl_Position = vec4(-ipos.x, -ipos.y, ipos.w, -ipos.z); break;\n"
            "    }\n"
            "    gl_Layer = iid;\n"
            "    ov.color= iv[j].color;\n"
            "    EmitVertex();\n"
            "  }\n";
        if (m_gsType == Single) {
            lwbeGS <<
                "  EndPrimitive();\n"
                " }\n";
        }
        lwbeGS << "}\n";
        break;
    default:
        assert(0);
        break;
    }

    // Fragment shader to simply display a passed-through color value.
    FragmentShader fsColor(440);
    fsColor <<
        "in Attrs {\n"
        "  vec3 color;\n"
        "} f;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(f.color, 1.0);\n"
        "}\n";

    // Program to draw the scene into the lwbe map.
    Program *drawProgram = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(drawProgram, vs, lwbeGS, fsColor)) {
        printf("compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
        LWNFailTest();
        return;
    }

    VertexShader lwbeDrawVS(440);
    lwbeDrawVS <<
        "layout(location = 0) in vec3 position;\n"
        "layout(location = 1) in vec3 lwbeCoord;\n"
        "out Attrs {\n"
        "  vec3 lwbeCoord;\n"
        "} ov;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ov.lwbeCoord = lwbeCoord;\n"
        "}\n";

    FragmentShader lwbeDrawFS(440);
    lwbeDrawFS <<
        "layout(binding = 0) uniform samplerLwbe lwbeMap;\n"
        "in Attrs {\n"
        "  vec3 lwbeCoord;\n"
        "} f;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = texture(lwbeMap, f.lwbeCoord);\n"
        "}\n";

    // Program to display from the lwbe map texture.
    Program *lwbeDrawProgram = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(lwbeDrawProgram, lwbeDrawVS, lwbeDrawFS)) {
        printf("compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
        LWNFailTest();
        return;
    }

    MemoryPoolAllocator vboAllocator(device, NULL, 1024 * 1024, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    MemoryPoolAllocator texAllocator(device, NULL, 1024 * 1024, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    // Set up the vertex format for this test.
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, normal);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, material);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, tesslevels);
    VertexArrayState vertex = stream.CreateVertexArrayState();

    // Allocate vertex and index buffers to hold the scene.
    Buffer *drawVBO = stream.AllocateVertexBuffer(device, maxVertices, vboAllocator, NULL);
    BufferAddress drawVBOAddr = drawVBO->GetAddress();
    Vertex *drawVBOPtr = (Vertex *) drawVBO->Map();

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *drawIBO = vboAllocator.allocBuffer(&bb, BUFFER_ALIGN_INDEX_BIT, maxIndices * sizeof(uint16_t));
    BufferAddress drawIBOAddr = drawIBO->GetAddress();
    uint16_t *drawIBOPtr = (uint16_t *) drawIBO->Map();

    DrawData drawData(drawVBOPtr, drawIBOPtr, drawIBOAddr);

    int tesslvl = 4;

    // Draw the ground, covering the y = -0.4 plane.
    drawData.setMaterial(0.0, 0.5, 0.0);
    drawData.addGround(8, -100, +100, -100, +100, -0.4);

    // Draw some grid lines just above the ground at y = -0.3.
    drawData.setMaterial(1.0, 1.0, 0.0);
    for (int i = -9; i < +9; i++) {
        float v1 = i * 0.5 - 0.01;
        float v2 = i * 0.5 + 0.01;
        drawData.addGround(8, v1, v2, -100, +100, -0.3);
    }

    // Draw some red spheres fairly close to the viewer at the center of the
    // +/-X and +/-Z faces.
    drawData.setMaterial(0.7, 0.0, 0.0);
    drawData.addSphere(tesslvl, +1.0, 0.1, 0.0, 0.2, 10, 10);
    drawData.addSphere(tesslvl, -1.0, 0.1, 0.0, 0.2, 10, 10);
    drawData.addSphere(tesslvl, 0.0, 0.1, +1.0, 0.2, 10, 10);
    drawData.addSphere(tesslvl, 0.0, 0.1, -1.0, 0.2, 10, 10);

    // Draw a few trees.
    drawData.addTree(tesslvl, +0.3, -0.5, -7.2, 3.2, 0.4, 5.0, 2.0, 10);  // far down -Z axis
    drawData.addTree(tesslvl, +2.9, -0.5, -5.2, 3.2, 0.4, 5.0, 2.0, 10);  // on -Z off to the right
    drawData.addTree(tesslvl, +2.6, -0.5, +0.0, 3.2, 0.4, 5.0, 2.0, 10);  // on +X, but tall enough to be on +Y, too
    drawData.addTree(tesslvl, +0.0, -0.5, +8.0, 3.2, 0.4, 5.0, 2.0, 10);  // straight behind on +Z

    drawData.checkOverflow(maxVertices, maxIndices);

    // Set up the lwbe map texture, a simple sampler, and a texture handle so
    // the texture can be displayed.
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetTarget(TextureTarget::TARGET_LWBEMAP);
    tb.SetSize3D(lwbeTexSize, lwbeTexSize, 6);
    tb.SetFlags(TextureFlags::COMPRESSIBLE);
    tb.SetFormat(Format::RGBA8);
    tb.SetLevels(1);
    Texture *lwbeColorTex = texAllocator.allocTexture(&tb);
    tb.SetFormat(Format::DEPTH24_STENCIL8);
    Texture *lwbeDepthTex = texAllocator.allocTexture(&tb);

    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetWrapMode(WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE);
    sb.SetMinMagFilter(MinFilter::LINEAR, MagFilter::LINEAR);
    Sampler *smp = sb.CreateSampler();

    TextureHandle lwbeColorHandle = device->GetTextureHandle(lwbeColorTex->GetRegisteredTextureID(), smp->GetRegisteredID());

    // Set up six separate (but identical) viewports for the test; needed for
    // the passthrough (multicast) case.
    LWNfloat viewport[] = { 0, 0, lwbeTexSize, lwbeTexSize };
    LWNint scissor[] = { 0, 0, lwbeTexSize, lwbeTexSize };
    for (int i = 0; i < 6; i++) {
        queueCB.SetViewports(i, 1, viewport);
        queueCB.SetScissors(i, 1, scissor);
    }

    // Set up the depth test for 1/W buffering, where we clear to zero (1/W==0
    // implies W==infinity) and use a depth test of GREATER, where a depth
    // value of 1 is the near plane.
    DepthStencilState dss;
    dss.SetDefaults();
    dss.SetDepthFunc(DepthFunc::GREATER);
    dss.SetDepthTestEnable(LWN_TRUE);
    dss.SetDepthWriteEnable(LWN_TRUE);

    // Bind the entire lwbe map for rendering and clear with a shade of blue
    // for the sky.
    queueCB.SetRenderTargets(1, &lwbeColorTex, NULL, lwbeDepthTex, NULL);
    queueCB.ClearColor(0, 0.2, 0.5, 0.8, 1.0, ClearColorMask::RGBA);
    queueCB.ClearDepthStencil(0.0, LWN_TRUE, 0, 0);
    queueCB.BindDepthStencilState(&dss);

    // For passthrough geometry shaders, use the viewport swizzle to do a
    // per-viewport reprojection.
    if (m_gsType == Passthrough) {
        const ViewportSwizzle lwbeSwizzles[] = {
            // case 0:  gl_Position = vec4(-ipos.z, -ipos.y, ipos.w, +ipos.x);
            ViewportSwizzle::NEGATIVE_Z,
            ViewportSwizzle::NEGATIVE_Y,
            ViewportSwizzle::POSITIVE_W,
            ViewportSwizzle::POSITIVE_X,
            // case 1:  gl_Position = vec4(+ipos.z, -ipos.y, ipos.w, -ipos.x);
            ViewportSwizzle::POSITIVE_Z,
            ViewportSwizzle::NEGATIVE_Y,
            ViewportSwizzle::POSITIVE_W,
            ViewportSwizzle::NEGATIVE_X,
            // case 2:  gl_Position = vec4(+ipos.x, +ipos.z, ipos.w, +ipos.y);
            ViewportSwizzle::POSITIVE_X,
            ViewportSwizzle::POSITIVE_Z,
            ViewportSwizzle::POSITIVE_W,
            ViewportSwizzle::POSITIVE_Y,
            // case 3:  gl_Position = vec4(+ipos.x, -ipos.z, ipos.w, -ipos.y);
            ViewportSwizzle::POSITIVE_X,
            ViewportSwizzle::NEGATIVE_Z,
            ViewportSwizzle::POSITIVE_W,
            ViewportSwizzle::NEGATIVE_Y,
            // case 4:  gl_Position = vec4(+ipos.x, -ipos.y, ipos.w, +ipos.z);
            ViewportSwizzle::POSITIVE_X,
            ViewportSwizzle::NEGATIVE_Y,
            ViewportSwizzle::POSITIVE_W,
            ViewportSwizzle::POSITIVE_Z,
            // case 5:  gl_Position = vec4(-ipos.x, -ipos.y, ipos.w, -ipos.z);
            ViewportSwizzle::NEGATIVE_X,
            ViewportSwizzle::NEGATIVE_Y,
            ViewportSwizzle::POSITIVE_W,
            ViewportSwizzle::NEGATIVE_Z,
        };
        queueCB.SetViewportSwizzles(0, 6, lwbeSwizzles);
    }

    // Draw the scene.
    queueCB.BindProgram(drawProgram, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, drawVBOAddr, maxVertices * sizeof(Vertex));
    drawData.draw(queueCB, DrawPrimitive::TRIANGLES);

    // Reset the viewports swizzle state.
    if (m_gsType == Passthrough) {
        ViewportSwizzle defaultSwizzles[] =
        {
            ViewportSwizzle::POSITIVE_X,
            ViewportSwizzle::POSITIVE_Y,
            ViewportSwizzle::POSITIVE_Z,
            ViewportSwizzle::POSITIVE_W,
        };
        for (int i = 0; i < 6; i++) {
            queueCB.SetViewportSwizzles(i, 1, defaultSwizzles);
        }
    }

    // Insert a render-to-texture barrier.
    queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS | BarrierBits::ILWALIDATE_TEXTURE);

    // Set up the vertex format for displaying a portion of the lwbe map,
    // texturing using coordinates of <lwbeCoord>.
    struct LwbeVertex {
        dt::vec3 position;
        dt::vec3 lwbeCoord;
        LwbeVertex() {}
        LwbeVertex(dt::vec3 pos, dt::vec3 cc) : position(pos), lwbeCoord(cc) {}
    };
    VertexStream lwbeStream(sizeof(LwbeVertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(lwbeStream, LwbeVertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(lwbeStream, LwbeVertex, lwbeCoord);
    VertexArrayState lwbeVertex = lwbeStream.CreateVertexArrayState();

    Buffer *lwbeVBO = lwbeStream.AllocateVertexBuffer(device, maxLwbeVertices, vboAllocator, NULL);
    BufferAddress lwbeVBOAddr = lwbeVBO->GetAddress();
    LwbeVertex *lwbeVBOPtr = (LwbeVertex *) lwbeVBO->Map();

    // Turn off depth testing.
    dss.SetDepthTestEnable(LWN_FALSE);
    dss.SetDepthWriteEnable(LWN_FALSE);
    queueCB.BindDepthStencilState(&dss);

    // Now set up to draw to the window using the lwbe map texture.
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.ClearColor(0, 0, 0, 0, 0, ClearColorMask::RGBA);
    queueCB.BindProgram(lwbeDrawProgram, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(lwbeVertex);
    queueCB.BindVertexBuffer(0, lwbeVBOAddr, maxLwbeVertices * sizeof(LwbeVertex));
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, lwbeColorHandle);

    // Array of views for the cells displaying the lwbe map.  Each set of four
    // values specify rotations around the Y, X, and Z axes and a distance in
    // Z to focus on.
    float views[24][4] = {

        // A row of six views to show the six faces of the lwbe map directly.
        { 0, 0, 0, 1 },
        { 90, 0, 0, 1 },
        { 180, 0, 0, 1 },
        { 270, 0, 0, 1 },
        { 0, -90, 0, 1 },
        { 0, +90, 0, 1 },

        // A row of six slightly rotated views relative the -Z axis.
        { +30, +15, 0, 1 },
        { +20, +10, 0, 1 },
        { +10, +5, 0, 1 },
        { +0, +0, 0, 1 },
        { -10, -5, 0, 1 },
        { -20, -10, 0, 1 },

        // A set of six views going mostly down the -Z axis with different
        // zoom factors (from a fish-eye type view to closeups).
        { -20, -5, +5, 0.3 },
        { -20, -5, +5, 0.6 },
        { -20, -5, +5, 0.9 },
        { -20, -5, +5, 1.4 },
        { -20, -5, +5, 2.0 },
        { -20, -5, +5, 3.0 },

        // A set of six views spinning around the -Z axis like a barrel roll.
        { -10, +0, -30, 1 },
        { -10, +0, -20, 1 },
        { -10, +0, -10, 1 },
        { -10, +0, +0, 1 },
        { -10, +0, +10, 1 },
        { -10, +0, +20, 1 },
    };

    for (int i = 0; i < 24; i++) {

        // Set up a rotation matrix.
        dmat3 m = (dmat3::rotation(views[i][2], 0, 0, 1) *
                   dmat3::rotation(views[i][1], 1, 0, 0) *
                   dmat3::rotation(views[i][0], 0, 1, 0));

        // Start with a square centered on the Z axis (at a view-dependent
        // distance), and then rotate per the matrix.
        for (int j = 0; j < 4; j++) {
            dt::vec3 pos = dt::vec3((j & 2) ? +1.0 : -1.0, (j & 1) ? +1.0 : -1.0, 0);
            dt::vec3 tc = dt::vec3(pos[0], pos[1], -views[i][3]);
            tc = m * tc;
            lwbeVBOPtr[4*i+j] = LwbeVertex(pos, tc);
        }

        // Set up a cell-dependent viewport and display the view of the lwbe.
        queueCB.SetViewportScissor((cellSize + 2 * cellMargin) * (i % 6) + cellMargin,
                                   (cellSize + 2 * cellMargin) * (i / 6) + cellMargin,
                                   cellSize, cellSize);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 4*i, 4);
    }

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNGSLwbeMapTest, lwn_gs_lwbe_inst, (LWNGSLwbeMapTest::Instanced));
OGTEST_CppTest(LWNGSLwbeMapTest, lwn_gs_lwbe_pass, (LWNGSLwbeMapTest::Passthrough));
OGTEST_CppTest(LWNGSLwbeMapTest, lwn_gs_lwbe_single, (LWNGSLwbeMapTest::Single));
