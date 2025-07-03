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

class LWNTessellationTest
{
public:
    enum Variant { TessDefault, TessEven, TessOdd };
private:
    enum TessMode {
        TRIANGLES,
        QUADS,
        ISOLINES,
        NTESSMODE
    };
    Variant m_variant;
public:
    LWNTessellationTest(Variant variant) : m_variant(variant) {}
    LWNTEST_CppMethods();
};

// Canned shaders for tessellation testing, including variants of tessellation
// control and evaluation shaders for the different tessellation types.
static const char *vsstring =
    "layout(location = 0) in vec4 position;\n"
    "void main() {\n"
    "  gl_Position = position;\n"
    "}\n";

static const char *fsstring =
    "layout(location = 0) out vec4 color;\n"
    "void main() {\n"
    "  color = vec4(1.0);\n"
    "}\n";

static const char *tcsstring_tris =
    "#define iid gl_IlwocationID\n"
    "layout(vertices=3) out;\n"
    "const float lod = 12.0;\n"
    "void main() {\n"
    "  gl_out[iid].gl_Position = gl_in[iid].gl_Position.yxzw;\n"
    "  gl_TessLevelOuter[0] = lod;\n"
    "  gl_TessLevelOuter[1] = lod;\n"
    "  gl_TessLevelOuter[2] = lod;\n"
    "  gl_TessLevelInner[0] = lod;\n"
    "}\n";

static const char *tcsstring_quads =
    "#define iid gl_IlwocationID\n"
    "layout(vertices=4) out;\n"
    "const float lod = 12.0;\n"
    "void main() {\n"
    "  gl_out[iid].gl_Position = gl_in[iid].gl_Position.yxzw;\n"
    "  gl_TessLevelOuter[0] = lod;\n"
    "  gl_TessLevelOuter[1] = lod;\n"
    "  gl_TessLevelOuter[2] = lod;\n"
    "  gl_TessLevelOuter[3] = lod;\n"
    "  gl_TessLevelInner[0] = lod;\n"
    "  gl_TessLevelInner[1] = lod;\n"
    "}\n";

static const char *tesstring_tris =
    "void main() {\n"
    "  gl_Position = (gl_in[0].gl_Position * gl_TessCoord.x + \n"
    "                 gl_in[1].gl_Position * gl_TessCoord.y + \n"
    "                 gl_in[2].gl_Position * gl_TessCoord.z);\n"
    // This only has a meaning when rendering with point_mode in TES.
    "  gl_PointSize = 3.0;\n"
    "}\n";

static const char *tesstring_quads =
    "vec4 lerp4(vec4 a, vec4 b, vec4 c, vec4 d, vec3 tessCoord)\n"
    "{\n"
    "  vec4 r0 = mix(a, b, tessCoord.x);\n"
    "  vec4 r1 = mix(c, d, tessCoord.x);\n"
    "  return mix(r0, r1, tessCoord.y);\n"
    "}\n"
    "\n"
    "void main() {\n"
    "  gl_Position = lerp4(gl_in[0].gl_Position, gl_in[1].gl_Position,\n"
    "                      gl_in[2].gl_Position, gl_in[3].gl_Position,\n"
    "                      gl_TessCoord);\n"
    // This only has a meaning when rendering with point_mode in TES.
    "  gl_PointSize = 3.0;\n"
    "}\n";

static const char *tesstring_isolines =
    "vec4 lerp4(vec4 a, vec4 b, vec4 c, vec4 d, vec3 tessCoord)\n"
    "{\n"
    "  vec4 r0 = mix(a, b, tessCoord.x);\n"
    "  vec4 r1 = mix(c, d, tessCoord.x);\n"
    "  return mix(r0, r1, tessCoord.y);\n"
    "}\n"
    "\n"
    "void main() {\n"
    "  gl_Position = lerp4(gl_in[0].gl_Position, gl_in[1].gl_Position,\n"
    "                      gl_in[2].gl_Position, gl_in[3].gl_Position,\n"
    "                      gl_TessCoord);\n"
    // This only has a meaning when rendering with point_mode in TES.
    "  gl_PointSize = 3.0;\n"
    "}\n";

static const char *tris_layout_default =
    "layout(triangles) in;\n";
static const char *quads_layout_default =
    "layout(quads) in;\n";
static const char *isolines_layout_default =
    "layout(isolines) in;\n";

static const char *tris_layout_odd =
    "layout(triangles, fractional_odd_spacing, point_mode) in;\n";
static const char *quads_layout_odd =
    "layout(quads, fractional_odd_spacing, point_mode) in;\n";
static const char *isolines_layout_odd =
    "layout(isolines, fractional_odd_spacing, point_mode) in;\n";

static const char *tris_layout_even =
    "layout(triangles, fractional_even_spacing, point_mode) in;\n";
static const char *quads_layout_even =
    "layout(quads, fractional_even_spacing, point_mode) in;\n";
static const char *isolines_layout_even =
    "layout(isolines, fractional_even_spacing, point_mode) in;\n";

static const char *tcsStrings[] = { tcsstring_tris, tcsstring_quads, tcsstring_quads };
static const char *tesStrings[] = { tesstring_tris, tesstring_quads, tesstring_isolines };

static const char *defaultLayout[] = { tris_layout_default, quads_layout_default, isolines_layout_default };
static const char *oddLayout[]     = { tris_layout_odd,     quads_layout_odd,     isolines_layout_odd };
static const char *evenLayout[]    = { tris_layout_even,    quads_layout_even,    isolines_layout_even };

lwString LWNTessellationTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Basic test of tessellation showing different topology types. The three "
        "columns of the bottom row show triangular, quad, and isoline tessellation "
        "with no TCS and a fixed level-of-detail of 8.\n";
    if (m_variant == TessDefault) {
        sb <<
            "The three columns on the top row use a TCS, are rotated 90 degrees, "
            "and have a level-of-detail of 12.";
    } else if (m_variant == TessEven) {
        sb <<
            "This test uses a tessellation evaluation shader with even spacing "
            "and point mode layout to verify that the driver works properly. "
            "The three columns on the top row are drawn as points with no TCS and "
            "have an even (8) level-of-detail.";
    } else {
        // m_variant == TessOdd
        sb <<
            "This test uses a tessellation evaluation shader with odd spacing "
            "and point mode layout to verify that the driver works properly. "
            "The three columns on the top row are drawn as points with no TCS and "
            "have an odd (9) level-of-detail.";
    }
    return sb.str();
}

int LWNTessellationTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(5, 0);
}

void LWNTessellationTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, 0x100UL, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // Clear the framebuffer.
    float clearColor[] = { 0, 0, 0, 1 };
    queueCB.ClearColor(0, clearColor, ClearColorMask::RGBA);

    // Build different program objects for each configuration.
    VertexShader vs(440);
    FragmentShader fs(440);
    vs << vsstring;
    fs << fsstring;
    LWNboolean compiled = true;
    Program *programs[NTESSMODE][2];
    for (int i = 0; i < NTESSMODE; i++) {
        TessControlShader tcs(440);
        TessEvaluationShader tes1(440);
        TessEvaluationShader tes2(440);

        tcs << tcsStrings[i];
        tes1 << defaultLayout[i];
        tes1 << tesStrings[i];

        programs[i][0] = device->CreateProgram();
        compiled = g_glslcHelper->CompileAndSetShaders(programs[i][0], vs, fs, tes1);

        programs[i][1] = device->CreateProgram();
        switch (m_variant) {
        case TessEven:
            tes2 << evenLayout[i];
            tes2 << tesStrings[i];
            compiled = g_glslcHelper->CompileAndSetShaders(programs[i][1], vs, fs, tes2);
            break;
        case TessOdd:
            tes2 << oddLayout[i];
            tes2 << tesStrings[i];
            compiled = g_glslcHelper->CompileAndSetShaders(programs[i][1], vs, fs, tes2);
            break;
        default:
            tes2 << defaultLayout[i];
            tes2 << tesStrings[i];
            compiled = g_glslcHelper->CompileAndSetShaders(programs[i][1], vs, fs, tcs, tes2);
            break;
        }

    }

    // If any compilation fails, clear the screen to red and bail.
    if (!compiled) {
        float clearColor[] = { 1, 0, 0, 1 };
        queueCB.ClearColor(0, clearColor, ClearColorMask::RGBA);
        return;
    }

    // Set up the vertex state and buffer objects for the test, with one
    // single-triangle patch followed by one single-quad patch.
    struct Vertex {
        dt::vec3 position;
    };
    static const Vertex vertexData[] = {
        // Single triangle patch.
        { dt::vec3(-0.9f, -0.9f, 0.5f) },
        { dt::vec3(+0.9f, -0.9f, 0.5f) },
        { dt::vec3(+0.0f, +0.9f, 0.5f) },
        // Single quad patch.
        { dt::vec3(-0.9f, -0.9f, 0.5f) },
        { dt::vec3(+0.9f, -0.9f, 0.5f) },
        { dt::vec3(-0.9f, +0.9f, 0.5f) },
        { dt::vec3(+0.9f, +0.9f, 0.5f) },
    };
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertexData), allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    PolygonState polygon;
    polygon.SetDefaults();
    polygon.SetPolygonMode(PolygonMode::LINE);

    // Set up pipeline objects for each configuration.
    queueCB.BindPolygonState(&polygon);
    queueCB.BindVertexArrayState(vertex);

    LWNfloat levels[] = { 8, 8, 8, 8 };
    queueCB.SetOuterTessellationLevels(levels);
    queueCB.SetInnerTessellationLevels(levels);

    for (int i = 0; i < NTESSMODE; i++) {
        for (int j = 0; j < 2; j++) {
            int patchSize = (i == TRIANGLES) ? 3 : 4;
            int firstIndex = (i == TRIANGLES) ? 0 : 3;
            queueCB.SetViewport(i * (lwrrentWindowWidth / NTESSMODE), j * lwrrentWindowHeight / 2,
                                lwrrentWindowWidth / NTESSMODE, lwrrentWindowHeight / 2);
            queueCB.SetScissor(i * (lwrrentWindowWidth / NTESSMODE), j * lwrrentWindowHeight / 2,
                               lwrrentWindowWidth / NTESSMODE, lwrrentWindowHeight / 2);
            queueCB.SetPatchSize(patchSize);
            queueCB.BindProgram(programs[i][j], ShaderStageBits::ALL_GRAPHICS_BITS);
            queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
            queueCB.DrawArrays(DrawPrimitive::PATCHES, firstIndex, patchSize);
        }
    }

    queueCB.submit();

    queue->Finish();
}

OGTEST_CppTest(LWNTessellationTest, lwn_tess_basic, (LWNTessellationTest::TessDefault));
OGTEST_CppTest(LWNTessellationTest, lwn_tess_even,  (LWNTessellationTest::TessEven));
OGTEST_CppTest(LWNTessellationTest, lwn_tess_odd,   (LWNTessellationTest::TessOdd));
