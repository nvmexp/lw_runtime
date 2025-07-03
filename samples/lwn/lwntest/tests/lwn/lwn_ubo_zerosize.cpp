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

class LWNZeroSizeUBOTest
{
public:
    LWNTEST_CppMethods();
};

lwString LWNZeroSizeUBOTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Basic functional test to exercise zero-sized uniform buffer bindings, "
        "where LWN treats affected binding points as unpopulated.  When hardware "
        "attempts to access memory for an unpopulated binding point, no memory "
        "access will be performed and zero will be returned.\n"
        "\n"
        "This test draws four cells, each of which should be green if zero-size "
        "bindings are supported correctly and red otherwise.  The bottom rows "
        "uses a fragment shader; the top rows uses a compute shader.  The "
        "left column uses the single-bind API (BindUniformBuffer); the right "
        "column uses the multi-bind API (BindUniformBuffers).";
    return sb.str();
}

int LWNZeroSizeUBOTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(5, 0);
}

void LWNZeroSizeUBOTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const char *uboDecls =
        // We declare four UBOs.  UBONormal is backed by a regular buffer that
        // holds 1.0 values.  UBOUnbound is backed by a regular buffer address
        // with a zero size (should return 0.0).  UBOUnbound2 is backed by a
        // binding using a zero address and size (should return 0.0).
        // UBOIndices holds an identify transformation and is used to be sure
        // we test both direct and indexed UBO accesses in hardware, even when
        // the compiler unrolls loops.
        "layout(binding=0) uniform UBONormal {\n"
        "  vec4 bound[16];\n"
        "};\n"
        "layout(binding=1) uniform UBOUnbound {\n"
        "  vec4 unbound[16];\n"
        "};\n"
        "layout(binding=2) uniform UBOUnbound2 {\n"
        "  vec4 unbound2[16];\n"
        "};\n"
        "layout(binding=3) uniform UBOIndices {\n"
        "  ivec4 indices;\n"
        "};\n";

    const char *uboTest =
        // When testing we read from the first four vectors of each of the
        // UBO binding points and make sure we get expected values (1.0 for
        // bound, 0.0 for unbound).  We test with direct accesses, indexed
        // accesses that will be turned into direct accesses with compiler
        // loop unrolling, and true indirect address (using indices taken
        // from our index UBO).
        "  bool ok = true;\n"
        "  ok = ok && all(equal(bound[0], vec4(1.0)));\n"
        "  ok = ok && all(equal(bound[1], vec4(1.0)));\n"
        "  ok = ok && all(equal(bound[2], vec4(1.0)));\n"
        "  ok = ok && all(equal(bound[3], vec4(1.0)));\n"
        "  for (int i = 0; i < 4; i++) {\n"
        "    ok = ok && all(equal(bound[indices[i]], vec4(1.0)));\n"
        "    ok = ok && all(equal(unbound[i], vec4(0.0)));\n"
        "    ok = ok && all(equal(unbound[indices[i]], vec4(0.0)));\n"
        "    ok = ok && all(equal(unbound2[i], vec4(0.0)));\n"
        "    ok = ok && all(equal(unbound2[indices[i]], vec4(0.0)));\n"
        "  }\n";

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        uboDecls <<
        "out vec4 fcolor;\n"
        "void main() {\n" <<
        uboTest <<
        "  if (ok) {\n"
        "    fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "  } else {\n"
        "    fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "  }\n"
        "}\n";

    Program *gfxProgram = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(gfxProgram, vs, fs)) {
        LWNFailTest();
        return;
    }

    ComputeShader cs(440);
    cs.setCSGroupSize(1);
    cs <<
        uboDecls <<
        "layout(binding=0) buffer SSBO {\n"
        "  vec4 fcolor;\n"
        "};\n"
        "void main() {\n" <<
        uboTest <<
        "  if (ok) {\n"
        "    fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "  } else {\n"
        "    fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "  }\n"
        "}\n";

    Program *computeProgram = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(computeProgram, cs)) {
        LWNFailTest();
        return;
    }

    // Set up the vertex format and buffer for graphics tests.
    struct Vertex {
        dt::vec3 position;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0) },
        { dt::vec3(-1.0, +1.0, 0.0) },
        { dt::vec3(+1.0, -1.0, 0.0) },
        { dt::vec3(+1.0, +1.0, 0.0) },
    };
    MemoryPoolAllocator vboAllocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, vboAllocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // Build three UBOs:  (1) a regular data buffer filled with 1.0 values,
    // (2) An "index" buffer filled with incrementing values to test indexing,
    // and (3) a dummy UBO filled with garbage (11.0) that is bound at the end
    // of each subtest to make sure new bindings work OK.  Also build an SSBO
    // that we will zero-initialize and we will use for compute shader output.
    size_t uboSize = 256;
    size_t ssboSize = 256;
    MemoryPoolAllocator uboSsboAllocator(device, NULL, 3 * uboSize + ssboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();

    Buffer *ubo = uboSsboAllocator.allocBuffer(&bb, BUFFER_ALIGN_UNIFORM_BIT, uboSize);
    BufferAddress uboAddr = ubo->GetAddress();
    float *uboMem = (float *) ubo->Map();
    for (unsigned int i = 0; i < uboSize / sizeof(float); i++) {
        uboMem[i] = 1.0;
    }

    Buffer *ubo2 = uboSsboAllocator.allocBuffer(&bb, BUFFER_ALIGN_UNIFORM_BIT, uboSize);
    BufferAddress ubo2Addr = ubo2->GetAddress();
    int *ubo2Mem = (int *) ubo2->Map();
    for (unsigned int i = 0; i < uboSize / sizeof(float); i++) {
        ubo2Mem[i] = i;
    }

    Buffer *ubo3 = uboSsboAllocator.allocBuffer(&bb, BUFFER_ALIGN_UNIFORM_BIT, uboSize);
    BufferAddress ubo3Addr = ubo3->GetAddress();
    float *ubo3Mem = (float *) ubo3->Map();
    for (unsigned int i = 0; i < uboSize / sizeof(float); i++) {
        ubo3Mem[i] = 11.0;
    }

    Buffer *ssbo = uboSsboAllocator.allocBuffer(&bb, BUFFER_ALIGN_SHADER_STORAGE_BIT, uboSize);
    BufferAddress ssboAddr = ssbo->GetAddress();
    float *ssboMem = (float *) ssbo->Map();
    for (unsigned int i = 0; i < ssboSize / sizeof(float); i++) {
        ssboMem[i] = 0.0;
    }

    // Set up ranges for four UBO bindings used by the test, two of which use
    // a size of zero.  See the comment for <uboDecls> for more info.
    BufferRange uboRanges[4] = {
        { uboAddr, uboSize },
        { uboAddr, 0 },
        { 0, 0 },
        { ubo2Addr, uboSize },
    };

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    queueCB.BindStorageBuffer(ShaderStage::COMPUTE, 0, ssboAddr, ssboSize);

    // Test compute and graphics shaders, as well as single- and multi-bind
    // APIs.
    for (int compute = 0; compute < 2; compute++) {
        for (int multibind = 0; multibind < 2; multibind++) {

            ShaderStage stage = compute ? ShaderStage::COMPUTE : ShaderStage::FRAGMENT;
            ShaderStageBits programBindMask = compute ? ShaderStageBits::COMPUTE : ShaderStageBits::ALL_GRAPHICS_BITS;
            Program *pgm = compute ? computeProgram : gfxProgram;

            queueCB.SetViewportScissor(compute * 320 + 32, multibind * 240 + 24, 256, 192);

            // Bind an appropriate program and uniform buffer set.
            queueCB.BindProgram(pgm, programBindMask);
            if (multibind) {
                queueCB.BindUniformBuffers(stage, 0, 4, uboRanges);
            } else {
                for (int i = 0; i < 4; i++) {
                    queueCB.BindUniformBuffer(stage, i, uboRanges[i].address, uboRanges[i].size);
                }
            }

            if (compute) {
                // When testing compute, initialize the vector in the SSBO to
                // red and then run the compute shader.  It should write green
                // to the SSBO memory.  Wait for completion and then clear the
                // cell using the values in the SSBO.
                for (int i = 0; i < 4; i++) {
                    ssboMem[i] = (i == 0 || i == 3) ? 1.0 : 0.0;
                }
                queueCB.DispatchCompute(1, 1, 1);
                queueCB.submit();
                queue->Finish();
                queueCB.ClearColor(0, ssboMem, ClearColorMask::RGBA);
            } else {
                // When testing graphics, just draw a full-screen quad, which
                // will be colored by the fragment shader.
                queueCB.DrawArrays(DrawPrimitive::QUAD_STRIP, 0, 4);
            }

            // Bind our dummy UBO to make sure that subsequent tests don't get
            // "lucky" and pass because of previous bindings.
            for (int i = 0; i < 4; i++) {
                queueCB.BindUniformBuffer(stage, i, ubo3Addr, uboSize);
            }
        }
    }

    queueCB.BindStorageBuffer(ShaderStage::COMPUTE, 0, 0, 0);
    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNZeroSizeUBOTest, lwn_ubo_zerosize, );
