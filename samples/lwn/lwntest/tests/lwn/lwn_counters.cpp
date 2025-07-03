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

// Debug output control; set to 1 to print debug spew for test configurations
// with errors or 2 to spew globally.
#define LWN_COUNTERS_LOG_OUTPUT      0

#if LWN_COUNTERS_LOG_OUTPUT >= 2
#define SHOULD_LOG(_result)     true
#define LOG(x)                  printf x
#define LOG_INFO(x)             printf x
#elif LWN_COUNTERS_LOG_OUTPUT >= 1
#define SHOULD_LOG(_result)     (!(_result))
#define LOG(x)                  printf x
#define LOG_INFO(x)
#else
#define SHOULD_LOG(_result)     false
#define LOG(x)
#define LOG_INFO(x)
#endif

using namespace lwn;

class LWNCounterTest
{
private:
    static const int cellSize = 16;
    static const int cellMargin = 2;
    static const int cellsX = 640 / cellSize;
    static const int cellsY = 480 / cellSize;

    static const int fboSize = 32;      // use a 32 x 32 FBO for some rendering
    static const int vboQuads = 12;     // generate a buffer with 12 quads (3 sets of 4)

    enum QueryType {
        SamplesPassed,
        InputVertices,
        InputPrimitives,
        VertexShaderIlwocations,
        TessControlShaderIlwocations,
        TessEvaluationShaderIlwocations,
        GeometryShaderIlwocations,
        FragmentShaderIlwocations,
        TessEvaluationShaderPrimitives,
        GeometryShaderPrimitives,
        ClipperInputPrimitives,
        ClipperOutputPrimitives,
        Timestamp,
        TimestampTop,
        Report17,
        Report98,
        QueryTypeCount,
    };

    struct CounterMem {                 // layout of a GPU report in memory
        LWNuint64       counter;
        LWNuint64       timestamp;
    };

    struct CounterGroup {               // group of counter queries in memory
        CounterMem      counters[QueryTypeCount];
    };

    static const CounterType countersToQuery[];

    void queryCounters(CommandBuffer *queueCB, int groupNumber, BufferAddress queryDstAddr) const;
    void checkCounters(int groupNumber, const char *groupDescription, int row,
                       const CounterGroup *allGroups, const CounterGroup *expected) const;
    void drawCell(CommandBuffer *queueCB, int cx, int cy, bool result) const;

public:
    LWNTEST_CppMethods();

};

const CounterType LWNCounterTest::countersToQuery[] =
{
    CounterType::SAMPLES_PASSED,
    CounterType::INPUT_VERTICES,
    CounterType::INPUT_PRIMITIVES,
    CounterType::VERTEX_SHADER_ILWOCATIONS,
    CounterType::TESS_CONTROL_SHADER_ILWOCATIONS,
    CounterType::TESS_EVALUATION_SHADER_ILWOCATIONS,
    CounterType::GEOMETRY_SHADER_ILWOCATIONS,
    CounterType::FRAGMENT_SHADER_ILWOCATIONS,
    CounterType::TESS_EVALUATION_SHADER_PRIMITIVES,
    CounterType::GEOMETRY_SHADER_PRIMITIVES,
    CounterType::CLIPPER_INPUT_PRIMITIVES,
    CounterType::CLIPPER_OUTPUT_PRIMITIVES,
    CounterType::TIMESTAMP,
    CounterType::TIMESTAMP_TOP,
    CounterType::TIMESTAMP,     // placeholder, this is for ReportValue(17, ...)
    CounterType::TIMESTAMP,     // placeholder, this is for ReportValue(98, ...)
};

lwString LWNCounterTest::getDescription() const
{
    return 
        "Simple test exercising various counter queries in LWN.  Each row renders "
        "a scene to an off-screen framebuffer and then checks various counter "
        "values, each of which is assigned a column.  Results matching expectations "
        "are displayed as green cells; mismatches are red.\n"
        "\n"
        "NOTE:  This test is expecting exact values for some counters that could "
        "have implementation-dependent behavior (e.g., number of shader ilwocations).  "
        "If test variance oclwrs, it might be necessary to modify to check ranges "
        "of values.";
}

int LWNCounterTest::isSupported() const
{
    // Counters were added in version 8.1, but we added TIMESTAMP_TOP in 40.5
    // and extended the test to match.  We bumped the minimum version to 40.5
    // so we don't have to check everywhere we use TIMESTAMP_TOP.  The same
    // process was repeated for lwnCounterReportValue in API version 48.2.
    return lwogCheckLWNAPIVersion(48, 2);
}

// Query a group of counters at an offset of <groupNumber> into the buffer
// given by <queryDstHandle>.
void LWNCounterTest::queryCounters(CommandBuffer *queueCB, int groupNumber, BufferAddress queryDstAddr) const
{
    ct_assert(QueryTypeCount == __GL_ARRAYSIZE(countersToQuery));
    int offset = groupNumber * sizeof(CounterGroup);
    for (int i = 0; i < QueryTypeCount; i++) {
        switch (i) {
        case Report17:
            queueCB->ReportValue(17, queryDstAddr + offset);
            break;
        case Report98:
            queueCB->ReportValue(98, queryDstAddr + offset);
            break;
        default:
            queueCB->ReportCounter(countersToQuery[i], queryDstAddr + offset);
            break;
        }
        offset += sizeof(CounterMem);
    }
}

// Check the counters for group <groupNumber> in the counter output buffer
// pointed to by <allGroups>.  <expected> gives the expected delta between
// before and after counts.  Displays the results in row <row> on-screen.
void LWNCounterTest::checkCounters(int groupNumber, const char *groupDescription, int row,
                                   const CounterGroup *allGroups, const CounterGroup *expected) const
{
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;
    bool allTestsPassed = true;
    bool result;

    // The first group in <allGroups> is filled with zeroes; group <N> is
    // written at an offset of <N>+1.
    const CounterGroup *previous = &allGroups[groupNumber];
    const CounterGroup *current  = &allGroups[groupNumber + 1];

    for (int j = 0; j < QueryTypeCount; j++) {
        if (j == Timestamp || j == TimestampTop) {
            // Timestamps have no expected delta; they should just land in
            // wall-clock order with enough resolution to not sample the same
            // clock twice.
            result = (previous->counters[j].timestamp < current->counters[j].timestamp);
        } else {
            result = (current->counters[j].counter ==
                      (previous->counters[j].counter + expected->counters[j].counter));
        }
        if (!result) {
            allTestsPassed = false;
        }
        drawCell(queueCB, j, row, result);
    }

    // Print raw counter values if debug output is desired.
    (void)allTestsPassed;
    if (SHOULD_LOG(allTestsPassed)) {
        LOG(("Row %d (%s)\n", row, groupDescription));
        for (int j = 0; j < QueryTypeCount; j++) {
            LOG(("%2d 0x%08x%08x 0x%08x%08x 0x%08x%08x 0x%08x%08x\n", j,
                 LWNuint(expected->counters[j].counter >> 32),
                 LWNuint(expected->counters[j].counter & 0xFFFFFFFF),
                 LWNuint(previous->counters[j].counter >> 32),
                 LWNuint(previous->counters[j].counter & 0xFFFFFFFF),
                 LWNuint(current->counters[j].counter >> 32),
                 LWNuint(current->counters[j].counter & 0xFFFFFFFF),
                 LWNuint(current->counters[j].timestamp >> 32),
                 LWNuint(current->counters[j].timestamp & 0xFFFFFFFF)));
        }
    }
}

// Display the results of a subtest in cell (cx,cy) in red/green based on
// <result>.
void LWNCounterTest::drawCell(CommandBuffer *queueCB, int cx, int cy, bool result) const
{
    queueCB->SetScissor(cx * cellSize + cellMargin, cy * cellSize + cellMargin,
                       cellSize - 2 * cellMargin, cellSize - 2* cellMargin);
    queueCB->ClearColor(0, result ? 0.0 : 1.0, result ? 1.0 : 0.0, 0.0, 1.0);
}

void LWNCounterTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();

    // allocator will create pool at first allocation
    // 64k is more than for this test but anything below 64k always allocates one page in the sparse pool
    const LWNsizeiptr poolSize = 0x10000UL; 
    MemoryPoolAllocator allocator(device, NULL, poolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.submit();

    if (lwogCheckLWNAPIVersion(12, 4)) {
        g_glslcHelper->SetSeparable(LWN_TRUE); 
    }

    // We use a bunch of super-simple separable programs.
    VertexShader vs(440);
    vs <<
        "out gl_PerVertex {\n"
        "  vec4 gl_Position;\n"
        "};\n"
        "layout(location=0) in vec3 position;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "}\n";
    Program *vp = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(vp, vs);

    FragmentShader fs(440);
    fs <<
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(1.0);\n"
        "}\n";
    Program *fp = device->CreateProgram();

    g_glslcHelper->CompileAndSetShaders(fp, fs);

    TessControlShader tcs(440);
    tcs << 
        "in gl_PerVertex {\n"
        "  vec4 gl_Position;\n"
        "} gl_in[];\n"
        "out gl_PerVertex {\n"
        "  vec4 gl_Position;\n"
        "} gl_out[];\n"
        "void main() {\n"
        "  gl_out[gl_IlwocationID].gl_Position = gl_in[gl_IlwocationID].gl_Position;\n"
        "  barrier();\n"
        "  if (gl_IlwocationID == 0) {\n"
        "    for (int i = 0; i < 4; i++) { gl_TessLevelOuter[i] = 2.0; }\n"
        "    for (int i = 0; i < 2; i++) { gl_TessLevelInner[i] = 2.0; }\n"
        "  }\n"
        "}\n";
    tcs.setTCSParameters(4);
    Program *tcp = device->CreateProgram();

    g_glslcHelper->CompileAndSetShaders(tcp, tcs);

    TessEvaluationShader tes(440);
    tes <<
        "in gl_PerVertex {\n"
        "  vec4 gl_Position;\n"
        "} gl_in[];\n"
        "out gl_PerVertex {\n"
        "  vec4 gl_Position;\n"
        "};\n"
        "void main() {\n"
        "  gl_Position = mix(mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x),\n"
        "                    mix(gl_in[3].gl_Position, gl_in[2].gl_Position, gl_TessCoord.x),\n"
        "                    gl_TessCoord.y);\n"
        "}\n";
    // xxx whoa, what?  What this do
    tes.setTESParameters(GL_QUADS, GL_EQUAL, GL_CCW);
    Program *tep = device->CreateProgram();

    g_glslcHelper->CompileAndSetShaders(tep, tes);

    GeometryShader gs(440);
    gs << 
        "in gl_PerVertex {\n"
        "  vec4 gl_Position;\n"
        "} gl_in[];\n"
        "out gl_PerVertex {\n"
        "  vec4 gl_Position;\n"
        "};\n"
        "void main() {\n"
        "  for (int i = 0; i < 3; i++) {\n"
        "    gl_Position = gl_in[i].gl_Position;\n"
        "    EmitVertex();\n"
        "  }\n"
        "}\n";
    gs.setGSParameters(GL_TRIANGLES, GL_TRIANGLE_STRIP, 3);
    Program *gp = device->CreateProgram();

    g_glslcHelper->CompileAndSetShaders(gp, gs);

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
    };
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, vboQuads * 4, allocator, NULL);
    BufferAddress vboAddr = vbo->GetAddress();
    Vertex *vertices = (Vertex *) vbo->Map();

    // We have three batches of four full-sized quads.  In each batch, Z is
    // arranged in increasing order (0.1, 0.3, 0.5, 0.7).  The second batch
    // translates two of the four quads to be lwlled; the third batch causes
    // all to be lwlled.
    for (int i = 0; i < vboQuads * 4; i++) {
        int qvert = i % 4;
        int quad = (i / 4) %  4;
        int batch = (i / 16) % 4;
        float x = ((qvert + 1) & 2) ? 1.0 : -1.0;
        float y = (qvert & 2) ? 1.0 : -1.0;
        float z = 0.1 + 0.2 * quad;
        switch (batch) {
        case 0:     break;
        case 1:     if (quad & 1) { x += 2.5; } break;
        case 2:     x += 2.5; break;
        default:    assert(0); break;
        }
        vertices[i].position = dt::vec3(x, y, z);
    }

    // Bind vertex buffer and state
    queueCB.BindVertexBuffer(0, vboAddr, vboQuads * 4 * sizeof(Vertex));
    queueCB.BindVertexArrayState(vertex);

    // Set up default tessellation parameters for 3x3 tessellation of quad
    // patches with no TCS.
    LWNfloat tessLevels[] = { 3.0, 3.0, 3.0, 3.0 };
    queueCB.SetPatchSize(4);
    queueCB.SetInnerTessellationLevels(tessLevels);
    queueCB.SetOuterTessellationLevels(tessLevels);

    // Create the depth state vector we'll need later in this test
    DepthStencilState depth;
    depth.SetDefaults();

    // Set up a framebuffer for offline rendering.
    Framebuffer fbo(fboSize, fboSize);
    fbo.setColorFormat(0, Format::RGBA8);
    fbo.setDepthStencilFormat(Format::DEPTH24_STENCIL8);
    fbo.alloc(device);

    // Allocate a scratch buffer to hold counter group values; one for each
    // row, plus an extra one at the beginning filled with zeroes.  We do two
    // separate iterations, first writing counters to a buffer object that
    // isn't mapped (could be vidmem) then to one that can be mapped directly.
    const int MaxGroups = (cellsY / 2) + 1;
    CounterGroup *receivedCounts = new CounterGroup[MaxGroups+1];
    int row = 0;
    for (int useMappableBuffer = 0; useMappableBuffer < 2; useMappableBuffer++) {

        // Reset all the counters to zero at the beginning.
        for (int i = 0; i < QueryTypeCount; i++) {
            if (countersToQuery[i] != CounterType::TIMESTAMP && countersToQuery[i] != CounterType::TIMESTAMP_TOP) {
                queueCB.ResetCounter(countersToQuery[i]);
            }
        }

        // Set up a query buffer to receive the query groups.
        bb.SetDefaults();
        Buffer *queryBuffer = allocator.allocBuffer(&bb, BUFFER_ALIGN_COUNTER_BIT, MaxGroups * sizeof(CounterGroup));
        BufferAddress queryAddr = queryBuffer->GetAddress();

        // We run two passes through our various test.  The first one renders
        // and updates counters; the second one checks results.
        for (int pass = 0; pass < 2; pass++) {

            CounterGroup expected;
            for (int i = 0; i < QueryTypeCount; i++) {
                expected.counters[i].counter = 0;
                expected.counters[i].timestamp = 0;
            }

            bool render = (pass == 0);

            if (render) {
                fbo.bind(queueCB);
                queueCB.SetViewport(0, 0, fboSize, fboSize);
                queueCB.SetScissor(0, 0, fboSize, fboSize);
            } else {
                g_lwnWindowFramebuffer.bind();
                // no scissor/viewport; we will scissor/clear each cell
            }

            int group = 0;

            // Render with the default settings (4 full-size quads).
            if (render) {
                queueCB.BindProgram(vp, ShaderStageBits::VERTEX);
                queueCB.BindProgram(fp, ShaderStageBits::FRAGMENT);
                queueCB.DrawArrays(DrawPrimitive::QUADS, 0, 16);
                queryCounters(queueCB, group, queryAddr);
            } else {
                expected.counters[InputVertices].counter = 16;
                expected.counters[VertexShaderIlwocations].counter = 16;
                expected.counters[InputPrimitives].counter = 8;            // 4 quads = 8 triangles
                expected.counters[ClipperInputPrimitives].counter = 8;
                expected.counters[ClipperOutputPrimitives].counter = 8;
                expected.counters[SamplesPassed].counter = fboSize * fboSize * 4;
                expected.counters[FragmentShaderIlwocations].counter = fboSize * fboSize * 4;

                // Reports of "17" and "98" always report those values;
                // pretend like their "counters" incremented from zero on the
                // first report.
                expected.counters[Report17].counter = 17;
                expected.counters[Report98].counter = 98;

                checkCounters(group, "Default 4 quads", row, receivedCounts, &expected);

                // And now, we don't expect 17 and 98 to "increment" any more.
                expected.counters[Report17].counter = 0;
                expected.counters[Report98].counter = 0;

                row++;
            }
            group++;

            // Render with a depth test of GREATER and a starting Z of 0.0.
            // Quads have increasing Z, so all pass.
            if (render) {
                depth.SetDepthFunc(DepthFunc::GREATER);
                depth.SetDepthTestEnable(LWN_TRUE);
                queueCB.BindDepthStencilState(&depth);
                queueCB.ClearDepthStencil(0.0, LWN_TRUE, 0, 0);
                queueCB.DrawArrays(DrawPrimitive::QUADS, 0, 16);
                queryCounters(queueCB, group, queryAddr);
            } else {
                checkCounters(group, "4 quads pass Z", row, receivedCounts, &expected);
                row++;
            }
            group++;

            // Render with a depth test of LESS and a starting Z of 1.0.
            // Quads have increasing Z, so only the first one passes.
            if (render) {
                depth.SetDepthFunc(DepthFunc::LESS);
                queueCB.BindDepthStencilState(&depth);
                queueCB.ClearDepthStencil(1.0, LWN_TRUE, 0, 0);
                queueCB.DrawArrays(DrawPrimitive::QUADS, 0, 16);
                queryCounters(queueCB, group, queryAddr);
                depth.SetDepthTestEnable(LWN_FALSE);
            } else {
                expected.counters[FragmentShaderIlwocations].counter = fboSize * fboSize * 1;
                expected.counters[SamplesPassed].counter = fboSize * fboSize * 1;
                checkCounters(group, "1 quad passes Z", row, receivedCounts, &expected);
                row++;
            }
            group++;

            // Render with a depth test of LESS and a starting Z of 0.0.  This
            // will render nothing.
            if (render) {
                queueCB.ClearDepthStencil(0.0, LWN_TRUE, 0, 0);
                queueCB.DrawArrays(DrawPrimitive::QUADS, 0, 16);
                queryCounters(queueCB, group, queryAddr);
            } else {
                expected.counters[FragmentShaderIlwocations].counter = 0;
                expected.counters[SamplesPassed].counter = 0;
                checkCounters(group, "0 quads pass Z", row, receivedCounts, &expected);
                row++;
            }
            group++;

            // Render with the default pipeline, but with SetRenderEnable set
            // to false.  It should kill all draw calls.
            if (render) {
                queueCB.SetRenderEnable(LWN_FALSE);
                queueCB.BindDepthStencilState(&depth);
                queueCB.DrawArrays(DrawPrimitive::QUADS, 0, 16);
                queryCounters(queueCB, group, queryAddr);
                queueCB.SetRenderEnable(LWN_TRUE);
            } else {
                expected.counters[InputVertices].counter = 0;
                expected.counters[VertexShaderIlwocations].counter = 0;
                expected.counters[InputPrimitives].counter = 0;
                expected.counters[ClipperInputPrimitives].counter = 0;
                expected.counters[ClipperOutputPrimitives].counter = 0;
                expected.counters[SamplesPassed].counter = 0;
                expected.counters[FragmentShaderIlwocations].counter = 0;
                checkCounters(group, "Render enable off", row, receivedCounts, &expected);
                row++;
            }
            group++;

            // Render with rasterizer discard enabled.  It should kill all
            // primitives before clipping and rasterization.
            if (render) {
                queueCB.SetRasterizerDiscard(LWN_TRUE);
                queueCB.DrawArrays(DrawPrimitive::QUADS, 0, 16);
                queryCounters(queueCB, group, queryAddr);
                queueCB.SetRasterizerDiscard(LWN_FALSE);
            } else {
                expected.counters[InputVertices].counter = 16;
                expected.counters[VertexShaderIlwocations].counter = 16;
                expected.counters[InputPrimitives].counter = 8;            // 4 quads = 8 triangles
                checkCounters(group, "Rasterizer discard", row, receivedCounts, &expected);
                row++;
            }
            group++;

            // Render the second batch of quads (vertices 16..31) that are
            // half on-screen and half off-screen.
            if (render) {
                queueCB.DrawArrays(DrawPrimitive::QUADS, 16, 16);
                queryCounters(queueCB, group, queryAddr);
            } else {
                expected.counters[ClipperInputPrimitives].counter = 8;
                expected.counters[ClipperOutputPrimitives].counter = 4;
                expected.counters[SamplesPassed].counter = fboSize * fboSize * 2;
                expected.counters[FragmentShaderIlwocations].counter = fboSize * fboSize * 2;
                checkCounters(group, "Clipped Prims", row, receivedCounts, &expected);
                row++;
            }
            group++;

            // Render the third batch of quads (vertices 32..47) that are all
            // off-screen.
            if (render) {
                queueCB.DrawArrays(DrawPrimitive::QUADS, 32, 16);
                queryCounters(queueCB, group, queryAddr);
            } else {
                expected.counters[ClipperInputPrimitives].counter = 8;
                expected.counters[ClipperOutputPrimitives].counter = 0;
                expected.counters[SamplesPassed].counter = 0;
                expected.counters[FragmentShaderIlwocations].counter = 0;
                checkCounters(group, "Lwlled Prims", row, receivedCounts, &expected);
                row++;
            }
            group++;

            // Render with a pass-through geometry shader.
            if (render) {
                queueCB.BindProgram(gp, ShaderStageBits::GEOMETRY);
                queueCB.DrawArrays(DrawPrimitive::QUADS, 0, 16);
                queryCounters(queueCB, group, queryAddr);
                queueCB.BindProgram(NULL, ShaderStageBits::GEOMETRY);
            } else {
                expected.counters[ClipperInputPrimitives].counter = 8;
                expected.counters[ClipperOutputPrimitives].counter = 8;
                expected.counters[SamplesPassed].counter = fboSize * fboSize * 4;
                expected.counters[FragmentShaderIlwocations].counter = fboSize * fboSize * 4;
                expected.counters[GeometryShaderIlwocations].counter = 8;
                expected.counters[GeometryShaderPrimitives].counter = 8;
                checkCounters(group, "Geometry shader", row, receivedCounts, &expected);
                expected.counters[GeometryShaderIlwocations].counter = 0;
                expected.counters[GeometryShaderPrimitives].counter = 0;
                row++;
            }
            group++;

            // Render with a tessellation evaluation shader with no control
            // shader.  This will tessellate quad patches into a 3x3 mesh.
            if (render) {
                queueCB.BindProgram(tep, ShaderStageBits::TESS_EVALUATION);
                queueCB.DrawArrays(DrawPrimitive::PATCHES, 0, 16);
                queryCounters(queueCB, group, queryAddr);
            } else {
                expected.counters[InputPrimitives].counter = 4;
                expected.counters[TessEvaluationShaderIlwocations].counter = 4 * 4 * 4; // 4 patches, 3x3 grid = 4x4 vertices
                expected.counters[ClipperInputPrimitives].counter = 4 * 3 * 3 * 2;      // 4 patches, 3x3 quads, 2 tri/quad
                expected.counters[ClipperOutputPrimitives].counter = 4 * 3 * 3 * 2;
                expected.counters[TessEvaluationShaderPrimitives].counter = 4 * 3 * 3 * 2;
                checkCounters(group, "Tess evaluation shader", row, receivedCounts, &expected);
                row++;
            }
            group++;

            // Render with tessellation control and evaluation shaders.  This
            // will tessellate quad patches into a 2x2 mesh.
            if (render) {
                queueCB.BindProgram(tcp, ShaderStageBits::TESS_CONTROL);
                queueCB.DrawArrays(DrawPrimitive::PATCHES, 0, 16);
                queryCounters(queueCB, group, queryAddr);
                queueCB.BindProgram(NULL, ShaderStageBits::TESS_CONTROL |
                                          ShaderStageBits::TESS_EVALUATION);
            } else {
                expected.counters[InputPrimitives].counter = 4;
                expected.counters[TessControlShaderIlwocations].counter = 4;
                expected.counters[TessEvaluationShaderIlwocations].counter = 4 * 3 * 3; // 4 patches, 2x2 grid = 3x3 vertices
                expected.counters[ClipperInputPrimitives].counter = 4 * 2 * 2 * 2;  // 4 patches, 2x2 quads, 2 tri/quad
                expected.counters[ClipperOutputPrimitives].counter = 4 * 2 * 2 * 2;
                expected.counters[TessEvaluationShaderPrimitives].counter = 4 * 2 * 2 * 2;
                checkCounters(group, "Tess control and evaluation shader", row, receivedCounts, &expected);
                row++;
            }
            group++;

            // After completing all our rendering, set up a buffer mapping to
            // read back the query results, and then copy back into
            // <receivedCounts>.
            if (render) {
                queueCB.submit();
                Buffer *mapBuffer;
                if (useMappableBuffer) {
                    mapBuffer = queryBuffer;
                } else {
                    // For non-mappable buffers, schedule a buffer-to-buffer
                    // copy into a mappable buffer to get the results out.
                    bb.SetDefaults();

                    mapBuffer = allocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, group * sizeof(CounterGroup));
                    Sync *sync = device->CreateSync();
                    queue->FenceSync(sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, SyncFlagBits::FLUSH_FOR_CPU);
                    queue->WaitSync(sync);
                    sync->Free();
                    queueCB.CopyBufferToBuffer(queryAddr, mapBuffer->GetAddress(), group*sizeof(CounterGroup), CopyFlags::NONE);
                    queueCB.submit();
                }
                queue->Finish();
                CounterGroup *mem = (CounterGroup *) mapBuffer->Map();
                memset(receivedCounts, 0, sizeof(CounterGroup));                // first group holds zeroes
                memcpy(receivedCounts + 1, mem, group*sizeof(CounterGroup));
            }
        }

    }

    queueCB.submit();
    queue->Finish();
    fbo.destroy();

    delete[] receivedCounts;
}

OGTEST_CppTest(LWNCounterTest, lwn_counters, );
