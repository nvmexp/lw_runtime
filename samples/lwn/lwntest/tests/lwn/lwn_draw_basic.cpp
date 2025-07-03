/*
 * Copyright (c) 2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include "string.h"

// For the SPIR-V versions of the test, the hex of the SPIR-V
// binaries are embedded in separate files.
#include "resources/lwn_draw_basic/lwn_draw_basic_spv_instanced_hex_vert.txt"
#include "resources/lwn_draw_basic/lwn_draw_basic_spv_instanced_hex_frag.txt"
#include "resources/lwn_draw_basic/lwn_draw_basic_spv_non_instanced_hex_vert.txt"
#include "resources/lwn_draw_basic/lwn_draw_basic_spv_non_instanced_hex_geom.txt"
#include "resources/lwn_draw_basic/lwn_draw_basic_spv_non_instanced_hex_frag.txt"

#define LWN_BASIC_DO_PRINTF     0

#if LWN_BASIC_DO_PRINTF
#define log_output printf
#else
static void log_output(const char *fmt, ...)
{
}
#endif

// For instanced tests this defines the instance count and is used when sizing
// one of the sanity-checked UBO arrays.
//
// NOTE: For the *spirv* testing, this value is hardcoded in the SPIR-V binaries.  If this
// value needs to change, please regenerate the SPIR-V binaries with updated values for
// the expected instance ID array in the UBO.
#define INSTANCE_COUNT 4

using namespace lwn;

class LwnDrawBasicTest {
public:
    LwnDrawBasicTest(LWNboolean isSpirv);

    LWNTEST_CppMethods();

private:
    LWNboolean m_isSpirv;
};

LwnDrawBasicTest::LwnDrawBasicTest(LWNboolean isSpirv)
{
    m_isSpirv = isSpirv;
}

int LwnDrawBasicTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(11,2);
}

lwString LwnDrawBasicTest::getDescription() const
{
    return "This is a basic touch test for each CommandBuffer Draw call. It draws 7 columns as follows:\n"
           " * DrawArrays, 1 right-angle triangle\n"
           " * DrawElements, 2 isosceles triangles\n"
           " * DrawElementsBaseVertex, 2 isosceles triangles\n"
           " * DrawArraysInstanced, 4 right-angle triangles\n"
           " * DrawElementsInstanced, 4 isosceles triangles\n"
           " * DrawArraysIndirect, 4 right-angle triangles\n"
           " * DrawElementsIndirect, 4 isosceles triangles\n"
           "The bottom two rows of triangles should be green and the top two rows should be blue,\n"
           "where applicable.\n"
           "The built-ins gl_BaseVertexARB and gl_BaseInstanceARB are also tested against the\n"
           "expected values, and a failure will cause the triangle to be colored yellowish-brown.\n"
           "We test that gl_DrawIDARB is always 0 since it won't be used in any of these tests.";
}

void LwnDrawBasicTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &cmd = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // allocator will create pool at first allocation
    // 64k is more than for this test but anything below 64k always allocates one page in the sparse pool
    const LWNsizeiptr poolSize = 0x10000UL;
    MemoryPoolAllocator allocator(device, NULL, poolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    cmd.SetPrimitiveRestart(true, 0xFFFF);

    LWNfloat clearColor[] = {0.25, 0, 0, 1};
    cmd.ClearColor(0, clearColor, ClearColorMask::RGBA);

    struct UboData {
        LWNfloat xform[4];
        LWNint   expectedInstanceID[INSTANCE_COUNT];   // Expected instance ID for gl_InstanceIndex and gl_InstanceID lookup.
        LWNint   baseInstanceExpected; // Expected value from gl_BaseInstanceARB
        LWNint   baseVertexExpected;   // Expected value from gl_BaseVertexARB
    } uboData = { {0.2f, 0.2f, -0.875f, -0.9f}, {0, 1, 2, 3}, 0, 0};

    for (int i = 0; i < INSTANCE_COUNT; ++i) {
        uboData.expectedInstanceID[i] = i;
    }

    // Data types
    struct VertexPos {
        dt::vec4 position;
    };
    struct VertexColor {
        dt::vec4 color;
    };
    Buffer *ubo;

    // Non-instanced test cases
    {
        // Create programs from the device, provide them shader code and compile/link them
        Program *pgm = device->CreateProgram();

        LWNboolean compiled = false;

        if (m_isSpirv) {
            LWNshaderStage spvStage[3];
            spvStage[0] = LWN_SHADER_STAGE_VERTEX;
            spvStage[1] = LWN_SHADER_STAGE_GEOMETRY;
            spvStage[2] = LWN_SHADER_STAGE_FRAGMENT;
            unsigned char * spvShaders[3] = { lwn_draw_basic_non_instanced_vert_spv,
                                              lwn_draw_basic_non_instanced_geom_spv,
                                              lwn_draw_basic_non_instanced_frag_spv };

            SpirvParams spvParams;
            spvParams.sizes[0] = lwn_draw_basic_non_instanced_vert_spv_len;
            spvParams.sizes[1] = lwn_draw_basic_non_instanced_geom_spv_len;
            spvParams.sizes[2] = lwn_draw_basic_non_instanced_frag_spv_len;

            compiled =
                g_glslcHelper->lwnUtil::GLSLCHelper::CompileAndSetShaders(reinterpret_cast<LWNprogram*> (pgm),
                                                                          spvStage, 3, (const char **)spvShaders,
                                                                          &spvParams);

            if (!compiled) {
                log_output("Spirv shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog());
                LWNFailTest();
                return;
            }
        } else {
            VertexShader vs(440);
            vs <<
                "#extension GL_ARB_shader_draw_parameters : enable\n"

                "layout(location = 0) in vec4 position;\n"
                "layout(location = 1) out vec4 fcol;\n"
                "layout(binding = 0) uniform Block {\n"
                "    vec4 transform;\n"
                "    ivec4 instanceIdExpected;\n"
                "    int  baseInstanceExpected;\n"
                "    int  baseVertexExpected;\n"
                "};\n"
                "void main() {\n"
                "  gl_Position = position;\n"
                "  if ((gl_BaseInstanceARB != baseInstanceExpected) ||\n"
                "      (gl_BaseVertexARB   != baseVertexExpected)   ||\n"
                "      (gl_DrawIDARB       != 0) ||\n"
                "      (gl_InstanceID      != 0))\n"
                "  {\n"
                "    fcol = vec4(0.5, 0.0, 0.4, 1.0);\n"
                "  } else {\n"
                "    fcol = vec4(0, 1, 0, 1);\n"
                "  }\n"
                "}\n";

            GeometryShader gs(440);
            gs <<
                "layout(triangles, ilwocations = 1) in;\n"
                "layout(triangle_strip, max_vertices = 3) out;\n"
                "layout (location = 1) in vec4 iColor[];\n"

                "out vec4 fcol;\n"
                "layout(binding = 0) uniform Block {\n"
                "    vec4 transform;\n"
                "    ivec4 instanceIdExpected;\n"
                "    int  baseInstanceExpected;\n"
                "    int  baseVertexExpected;\n"
                "};\n"
                "void main() {\n"
                "  for (int i=0; i<3; i++) {\n"
                "    vec4 pos = gl_in[i].gl_Position;\n"
                "    pos.y += float(gl_PrimitiveIDIn)*1.2;\n"
                "    gl_Position = pos*vec4(transform.xy, 1, 1) + vec4(transform.zw, 0, 0);\n"
                "    fcol = iColor[i];\n"
                "    EmitVertex();\n"
                "  }\n"
                "  EndPrimitive();\n"
                "}\n";

            FragmentShader fs(440);
            fs <<
                "precision highp float;\n"
                "in vec4 fcol;\n"
                "layout(location = 0) out vec4 color;\n"
                "void main() {\n"
                "  color = fcol;\n"
                "}\n";

            compiled = g_glslcHelper->CompileAndSetShaders(pgm, vs, fs, gs);
            if (!compiled) {
                log_output("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog());
                LWNFailTest();
                return;
            }
        }

        // Set up the vertex format and buffer for non-instanced cases
        static const VertexPos vertexPosData[] = {
            { dt::vec4(0, 0, 0, 0) }, // vertex at infinity
            { dt::vec4(0, 0, 0, 0) }, // vertex at infinity
            { dt::vec4(0, 0, 0, 1) },
            { dt::vec4(1, 0, 0, 1) },
            { dt::vec4(0, 1, 0, 1) },
            { dt::vec4(1, 1, 0, 1) },
            { dt::vec4(0.5, 0, 0, 1) },
        };
        VertexStream posStream(sizeof(VertexPos));
        LWN_VERTEX_STREAM_ADD_MEMBER(posStream, VertexPos, position);
        VertexStreamSet streamSet(posStream);
        VertexArrayState vertex = streamSet.CreateVertexArrayState();

        Buffer *posVbo = posStream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertexPosData), allocator, vertexPosData);

        BufferAddress posAddr = posVbo->GetAddress();

        // Index buffer for non-instanced cases
        static const uint16_t indexData[9] = {0, 1, 0xFFFF, 3, 4, 5, 4, 5, 3};
        Buffer *ibo = AllocAndFillBuffer(device, queue, cmd, allocator,  indexData, sizeof(indexData), BUFFER_ALIGN_INDEX_BIT, false); BufferAddress iboAddr = ibo->GetAddress();

        cmd.BindProgram(pgm,ShaderStageBits::ALL_GRAPHICS_BITS);
        cmd.BindVertexArrayState(vertex);

        // ARB_shader_draw_parameters says that both these built-in values
        // should be "0" if the corresponding API doesn't have inputs for these
        // particular variables.
        uboData.baseInstanceExpected = 0;
        uboData.baseVertexExpected = 0;

        // DrawArrays
        cmd.BindVertexBuffer(0, posAddr, sizeof(vertexPosData));
        ubo = AllocAndFillBuffer(device, queue, cmd, allocator,  &uboData, sizeof(uboData), BUFFER_ALIGN_UNIFORM_BIT, false);
        cmd.BindUniformBuffer(ShaderStage::GEOMETRY, 0, ubo->GetAddress(), sizeof(uboData));
        cmd.BindUniformBuffer(ShaderStage::VERTEX, 0, ubo->GetAddress(), sizeof(uboData));
        cmd.DrawArrays(DrawPrimitive::TRIANGLES, 2, 3);
        uboData.xform[2] += 0.25f;

        // DrawElements - has no baseVertex, so we bind the vertex buffer at
        // offset of vertexPosData[1] to emulate baseVertex=1
        cmd.BindVertexBuffer(0, posAddr + sizeof(VertexPos), sizeof(vertexPosData));
        ubo = AllocAndFillBuffer(device, queue, cmd, allocator,  &uboData, sizeof(uboData), BUFFER_ALIGN_UNIFORM_BIT, false);
        cmd.BindUniformBuffer(ShaderStage::GEOMETRY, 0, ubo->GetAddress(), sizeof(uboData));
        cmd.BindUniformBuffer(ShaderStage::VERTEX, 0, ubo->GetAddress(), sizeof(uboData));
        cmd.DrawElements(DrawPrimitive::TRIANGLES, IndexType::UNSIGNED_SHORT, 7, iboAddr + 4);
        uboData.xform[2] += 0.25f;

        // DrawElementsBaseVertex
        uboData.baseVertexExpected = 1;
        cmd.BindVertexBuffer(0, posAddr, sizeof(vertexPosData));
        ubo = AllocAndFillBuffer(device, queue, cmd, allocator, &uboData, sizeof(uboData), BUFFER_ALIGN_UNIFORM_BIT, false);
        cmd.BindUniformBuffer(ShaderStage::GEOMETRY, 0, ubo->GetAddress(), sizeof(uboData));
        cmd.BindUniformBuffer(ShaderStage::VERTEX, 0, ubo->GetAddress(), sizeof(uboData));
        cmd.DrawElementsBaseVertex(DrawPrimitive::TRIANGLES, IndexType::UNSIGNED_SHORT, 7, iboAddr + 4, 1);
        uboData.xform[2] += 0.25f;

        cmd.submit();
        queue->Finish();
    }

    // Instanced test cases
    {
        // Create programs from the device, provide them shader code and compile/link them
        Program *pgm = device->CreateProgram();

        LWNboolean compiled = false;

        if (m_isSpirv) {
            LWNshaderStage spvStages[2];
            spvStages[0] = LWN_SHADER_STAGE_VERTEX;
            spvStages[1] = LWN_SHADER_STAGE_FRAGMENT;
            unsigned char * spvShaders[2] = { lwn_draw_basic_instanced_vert_spv,
                                              lwn_draw_basic_instanced_frag_spv };
            SpirvParams spvParams;
            spvParams.sizes[0] = lwn_draw_basic_instanced_vert_spv_len;
            spvParams.sizes[1] = lwn_draw_basic_instanced_frag_spv_len;

            compiled =
                g_glslcHelper->lwnUtil::GLSLCHelper::CompileAndSetShaders(reinterpret_cast<LWNprogram*> (pgm),
                    spvStages, 2, (const char **)spvShaders,
                    &spvParams);

            if (!compiled) {
                log_output("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog());
                LWNFailTest();
                return;
            }
        } else {
            VertexShader vs(440);
            vs <<
                "#extension GL_ARB_shader_draw_parameters : enable\n"
                "layout(location = 0) in vec4 position;\n"
                "layout(location = 1) in vec4 color;\n"
                "layout(binding = 0) uniform Block {\n"
                "    vec4 transform;\n"
                "    ivec4 instanceIdExpected;\n"
                "    int  baseInstanceExpected;\n"
                "    int  baseVertexExpected;\n"
                "};\n"

                "out vec4 fcol;\n"

                "void main() {\n"
                "  vec4 pos = position;\n"
                "  pos.y += float(gl_InstanceID)*1.2;\n"
                "  gl_Position = pos*vec4(transform.xy, 1, 1) + vec4(transform.zw, 0, 0);\n"
                "  if ((gl_BaseInstanceARB != baseInstanceExpected) ||\n"
                "      (gl_BaseVertexARB   != baseVertexExpected)   ||\n"
                "      (gl_DrawIDARB       != 0) ||\n"
                "      (instanceIdExpected[gl_InstanceID] != gl_InstanceID)) {\n"
                "    fcol = vec4(0.5, 0.4, 0.0, 1.0);\n"
                "  } else {\n"
                "    fcol = color;\n"
                "  }\n"
                "}\n";

            FragmentShader fs(440);
            fs << "precision highp float;\n"
                  "layout(location = 0) out vec4 color;\n"
                  "in vec4 fcol;\n"
                  "void main() {\n"
                  "  color = fcol;\n"
                  "}\n";

            compiled = g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);
            if (!compiled) {
                log_output("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog());
                LWNFailTest();
                return;
            }
        }

        // Set up the vertex format for the instanced cases.
        static const VertexPos vertexPosData[] = {
            { dt::vec4(0, 0, 0, 0) }, // vertex at infinity
            { dt::vec4(0, 0, 0, 0) }, // vertex at infinity
            { dt::vec4(0, 0, 0, 1) },
            { dt::vec4(1, 0, 0, 1) },
            { dt::vec4(0, 1, 0, 1) },
            { dt::vec4(1, 1, 0, 1) },
            { dt::vec4(0.5, 0, 0, 1) },
        };
        static const VertexColor vertexColorData[] = {
            { dt::vec4(1, 0, 0, 1) },
            { dt::vec4(0, 1, 0, 1) },
            { dt::vec4(0, 0, 1, 1) },
            { dt::vec4(0, 1, 1, 1) },
        };
        VertexStream posStream(sizeof(VertexPos));
        VertexStream colorStream(sizeof(VertexColor), 2);
        LWN_VERTEX_STREAM_ADD_MEMBER(posStream, VertexPos, position);
        LWN_VERTEX_STREAM_ADD_MEMBER(colorStream, VertexColor, color);
        VertexStreamSet streamSet(posStream, colorStream);
        VertexArrayState vertex = streamSet.CreateVertexArrayState();

        Buffer *posVbo = posStream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertexPosData), allocator, vertexPosData);
        Buffer *colorVbo = colorStream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertexColorData), allocator, vertexColorData);

        BufferAddress posAddr = posVbo->GetAddress();
        BufferAddress colorAddr = colorVbo->GetAddress();

        // Index buffer for instanced cases
        static const uint16_t indexData[6] = {0, 1, 0xFFFF, 3, 4, 5};
        Buffer *ibo = AllocAndFillBuffer(device, queue, cmd, allocator,  indexData, sizeof(indexData), BUFFER_ALIGN_INDEX_BIT, false);
        BufferAddress iboAddr = ibo->GetAddress();

        unsigned int indirect[20] = {0};
        DrawArraysIndirectData *dai = (DrawArraysIndirectData *)&indirect[2];
        DrawElementsIndirectData *dae = (DrawElementsIndirectData *)&indirect[10];
        dai->count = 3;
        dai->instanceCount = 4;
        dai->first = 2;
        dai->baseInstance = 1;

        dae->count = 4;
        dae->instanceCount = 4;
        dae->firstIndex = 2;
        dae->baseVertex = 1;
        dae->baseInstance = 1;

        Buffer *indirectVbo = AllocAndFillBuffer(device, queue, cmd, allocator,  indirect, sizeof(indirect), BUFFER_ALIGN_INDIRECT_BIT, false);
        BufferAddress indirectAddr = indirectVbo->GetAddress();

        cmd.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
        cmd.BindVertexArrayState(vertex);
        cmd.BindVertexBuffer(0, posAddr, sizeof(vertexPosData));
        cmd.BindVertexBuffer(1, colorAddr, sizeof(vertexColorData));

        // All these draw commands will use a baseInstance of 1.  Only baseVertex changes
        uboData.baseInstanceExpected = 1;
        uboData.baseVertexExpected = 0;

        // DrawArraysInstanced
        ubo = AllocAndFillBuffer(device, queue, cmd, allocator,  &uboData, sizeof(uboData), BUFFER_ALIGN_UNIFORM_BIT, false);
        cmd.BindUniformBuffer(ShaderStage::VERTEX, 0, ubo->GetAddress(), sizeof(uboData));
        cmd.DrawArraysInstanced(DrawPrimitive::TRIANGLES, 2, 3, 1, INSTANCE_COUNT);
        uboData.xform[2] += 0.25f;

        // DrawElementsInstanced
        uboData.baseVertexExpected = 1;
        ubo = AllocAndFillBuffer(device, queue, cmd, allocator, &uboData, sizeof(uboData), BUFFER_ALIGN_UNIFORM_BIT, false);
        cmd.BindUniformBuffer(ShaderStage::VERTEX, 0, ubo->GetAddress(), sizeof(uboData));
        cmd.DrawElementsInstanced(DrawPrimitive::TRIANGLES, IndexType::UNSIGNED_SHORT, 4, iboAddr + 4, 1, 1, INSTANCE_COUNT);
        uboData.xform[2] += 0.25f;

        // DrawArraysIndirect
        uboData.baseVertexExpected = 0;
        ubo = AllocAndFillBuffer(device, queue, cmd, allocator, &uboData, sizeof(uboData), BUFFER_ALIGN_UNIFORM_BIT, false);
        cmd.BindUniformBuffer(ShaderStage::VERTEX, 0, ubo->GetAddress(), sizeof(uboData));
        cmd.DrawArraysIndirect(DrawPrimitive::TRIANGLES, indirectAddr + 2*sizeof(unsigned int));
        uboData.xform[2] += 0.25f;

        // DrawElementsIndirect
        uboData.baseVertexExpected = 1;
        ubo = AllocAndFillBuffer(device, queue, cmd, allocator, &uboData, sizeof(uboData), BUFFER_ALIGN_UNIFORM_BIT, false);
        cmd.BindUniformBuffer(ShaderStage::VERTEX, 0, ubo->GetAddress(), sizeof(uboData));
        cmd.DrawElementsIndirect(DrawPrimitive::TRIANGLES, IndexType::UNSIGNED_SHORT, iboAddr, indirectAddr + 10*sizeof(unsigned int));

        cmd.submit();
        queue->Finish();
    }
}

OGTEST_CppTest(LwnDrawBasicTest, lwn_draw_basic, (false));
OGTEST_CppTest(LwnDrawBasicTest, lwn_draw_basic_spirv, (true));
