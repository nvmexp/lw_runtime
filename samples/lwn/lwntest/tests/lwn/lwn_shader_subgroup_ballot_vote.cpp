/*
 * Copyright (c) 2016-2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

//
// lwn/lwn_shader_subgroup_ballot.cpp
//
// Touch test for KHR_shader_subgroup_vote and KHR_shader_subgroup_ballot
//
// TODO - more exhaustive testing (all stages, etc)
// see https://gitlab.khronos.org/spirv/spirv-extensions/blob/master/SPV_KHR_shader_ballot_tests.txt
// for some additional ideas
//

#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include <limits>

using namespace lwn;

#define CELLS_X 2
#define CELLS_Y 1

#define CELL_COUNT (CELLS_X*CELLS_Y)

enum TEST_TYPE {
    TEST_TYPE_BALLOT,
    TEST_TYPE_VOTE
};

class LwnShaderSubgroupBallotVote {
public:
    LwnShaderSubgroupBallotVote(TEST_TYPE testType) : m_testType(testType) {}

    LWNTEST_CppMethods();

private:

    LWNboolean CompileBallotShaders(Program * compiledPrograms[CELL_COUNT]) const;
    LWNboolean CompileVoteShaders(Program * compiledPrograms[CELL_COUNT]) const;

    TEST_TYPE m_testType;

    LWNint GetSubgroupSize() const;

    struct Vertex {
        dt::vec2 position;
    };
};

lwString LwnShaderSubgroupBallotVote::getDescription() const
{
    lwStringBuf sb;

    sb << "Touch test for " << (m_testType == TEST_TYPE_BALLOT ? "KHR_shader_subgroup_ballot" : "KHR_shader_subgroup_vote") <<
          ". Green means all the checks in the shader passed, red means "
          "at least one of the checks failed. ";
    if (m_testType == TEST_TYPE_BALLOT) {
        sb << "First triangle uses a fragment shader that checks that various "
              "subgroupBroadcast* functions work as expected.  The second triangle checks that various subgroup mask "
              "builtins work as expected as well as using subgroupBroadcast.";
    }
    else if (m_testType == TEST_TYPE_VOTE) {
        sb << "First triangle tests various subgroup voting operations in both "
              "the vertex and fragment shaders. The vertex shader uses the "
              "gl_vertexID to determine the thread in the subgroup to test the "
              "various voting operations, while the fragment simply tests the "
              "subgroup vote operations return as expected when operating on "
              "the subgroup globally.  The second triangle uses the same vertex "
              "shader, but the fragment shader uses the gl_SubgroupInstanceID "
              "to perform more complex subgroup voting operations.";
    }

    return sb.str();
}

int LwnShaderSubgroupBallotVote::isSupported(void) const
{
    return g_lwnDeviceCaps.supportsShaderSubgroup;
}

LWNint LwnShaderSubgroupBallotVote::GetSubgroupSize() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();

    LWNint subgroupSize = 0;
    device->GetInteger(DeviceInfo::Enum::SHADER_SUBGROUP_SIZE, &subgroupSize);

    return subgroupSize;
}

// Compiles for the KHR_shader_subgroup_ballot tests
LWNboolean LwnShaderSubgroupBallotVote::CompileBallotShaders(Program * compiledPrograms[CELL_COUNT]) const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();

    // Shaders
    VertexShader vss(450);
    vss.addExtension(lwShaderExtension::KHR_shader_subgroup_basic);
    vss.addExtension(lwShaderExtension::KHR_shader_subgroup_ballot);
    vss.addExtension(lwShaderExtension::ARB_gpu_shader_int64);
    vss <<
        "\n"
        "layout(location = 0) in vec2 aVertex;\n"
        "layout(location = 0) out vec4 vColor;\n"
        "\n"
        "void main()\n"
        "{\n"
        "  float testArray[2];\n"
        "  testArray[0] = 0.0; testArray[1] = 1e-20;\n"
        "  uint lwrThread = gl_SubgroupIlwocationID;\n"
        "  vColor = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "  gl_Position = vec4(aVertex, testArray[int(lwrThread)&1], 1);\n"
        "}\n"
    ;

    FragmentShader fss1(450);
    fss1.addExtension(lwShaderExtension::KHR_shader_subgroup_basic);
    fss1.addExtension(lwShaderExtension::KHR_shader_subgroup_ballot);
    fss1.addExtension(lwShaderExtension::ARB_gpu_shader_int64);
    fss1 <<
        "\n"
        "layout(location = 0) in vec4 vColor;\n"
        "layout(location = 0) out vec4 oFrag;\n"
        "void main()\n"
        "{\n"
        "  uint lwrThread = gl_SubgroupIlwocationID;\n"
        "  uint firstThread = subgroupBroadcastFirst(lwrThread);\n"
        "  if (lwrThread != firstThread) {\n"
        "    uint newFirstThread = subgroupBroadcastFirst(lwrThread);\n"
        "    if (newFirstThread > firstThread) {\n"
        "      oFrag = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "    } else {\n"
        "      oFrag = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "    }\n"
        "  } else {\n"
        "    oFrag = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "  }\n"
        "}\n"
        ;

    LWNint subgroupSize = GetSubgroupSize();

    if (subgroupSize != 32) {
        printf("The subgroup size retrieved by LWN is %d which is mismatching the expected 32\n", subgroupSize);
        LWNFailTest();
    }

    FragmentShader fss2(450);
    fss2.addExtension(lwShaderExtension::KHR_shader_subgroup_basic);
    fss2.addExtension(lwShaderExtension::KHR_shader_subgroup_ballot);
    fss2.addExtension(lwShaderExtension::ARB_gpu_shader_int64);
    fss2 <<
        "\n"
        "layout(location = 0) in vec4 vColor;\n"
        "layout(location = 0) out vec4 oFrag;\n"
        "void main()\n"
        "{\n"
        "  bool success = true;\n"
        "  uint lwrThread = gl_SubgroupIlwocationID;\n"
        "  uint testMaskEq = uint(1) << lwrThread;\n"
        "  success = success && (testMaskEq == uint(gl_SubgroupEqMask));\n"
        "  const uint subGroupSize = gl_SubgroupSize;\n"
        "  success = success && (subGroupSize == 32);\n"
        "  success = success && lwrThread < 32;\n"
        "  success = success && (subgroupBallot(false) == 0);\n"
        "  success = success && ((uint(gl_SubgroupGeMask) | uint(gl_SubgroupLtMask) | \n"
        "                         uint(gl_SubgroupGtMask) | uint(gl_SubgroupLeMask) | \n"
        "                         uint(gl_SubgroupEqMask)) != 0);\n"
        "  uint testMaskActive = uint(subgroupBallot(true));\n"
        "  uint testMaskAll = 0;\n"
        "  for (int i = 0; i < subGroupSize; i++) {\n"
        "    testMaskAll |= 1 << i;\n"
        "  }\n"
        "  success = success && (testMaskActive <= testMaskAll);\n"
        "  uint lastThread = findMSB(uint(testMaskActive));\n"
        "  uint lastThreadValue = subgroupBroadcast(uint(lwrThread), lastThread);\n"
        "  success = success && (lastThreadValue == lastThread);\n"
        "  if (success) {\n"
        "    oFrag = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "  } else {\n"
        "    oFrag = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "  }\n"
        "}\n";

    // Two programs (one for each cell).
    FragmentShader * fshaders[CELL_COUNT] = {
        &fss1,
        &fss2
    }; // If the count ever changes then will have to do something different here; but compiler should warn.

    // Compile pass
    for (int i = 0; i < CELL_COUNT; ++i) {
        compiledPrograms[i] = device->CreateProgram();
        LWNboolean compiled = g_glslcHelper->CompileAndSetShaders(compiledPrograms[i], vss, *fshaders[i]);
        if (!compiled) {
            printf("Error compiling program %d:\n%s\n", i, g_glslcHelper->GetInfoLog());
            LWNFailTest();
            return LWN_FALSE;
        }
    }

    return LWN_TRUE;
}

// Compiles the sahders for KHR_shader_subgroup_vote tests
LWNboolean LwnShaderSubgroupBallotVote::CompileVoteShaders(Program * compiledPrograms[CELL_COUNT]) const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();

    // Shaders
    VertexShader vss(450);
    vss.addExtension(lwShaderExtension::KHR_shader_subgroup_basic);
    vss.addExtension(lwShaderExtension::KHR_shader_subgroup_vote);
    vss <<
        "layout(location = 0) in vec2 aVertex;\n"
        "layout(location = 0) out vec4 vColor;\n"
        "\n"
        "void main()\n"
        "{\n"
        "  const bool odd = (gl_VertexID & 1) != 0;\n"
        "  const bool even = !odd;\n"
        "  bool A, B, C, D;\n"
        "  A = false;\n"
        "  B = true;\n"
        "  C = false;\n"
        "  D = false;\n"
        "\n"
        "  if (even) {\n"
        "    A = subgroupAll(even);\n"
        "  } else {\n"
        "    A = subgroupAll(odd);\n"
        "  }\n"
        "\n"
        "  if (even) {\n"
        "    B = subgroupAny(odd);\n"
        "  } else {\n"
        "    B = subgroupAny(even);\n"
        "  }\n"
        "\n"
        "  if (even) {\n"
        "    C = subgroupAllEqual(odd);\n"
        "    D = subgroupAllEqual(even);\n"
        "  } else {\n"
        "    C = subgroupAllEqual(even);\n"
        "    D = subgroupAllEqual(odd);\n"
        "  }\n"
        "\n"
        "  // Verify: A == true, B == false, C == true, D == true;\n"
        "  if (A && !B && C && D) {\n"
        "    vColor = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "  } else {\n"
        "    vColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "  }\n"
        "  gl_Position = vec4(aVertex, 0, 1);\n"
        "}\n"
    ;

    FragmentShader fss1(450);
    fss1.addExtension(lwShaderExtension::KHR_shader_subgroup_basic);
    fss1.addExtension(lwShaderExtension::KHR_shader_subgroup_vote);
    fss1 <<
        "layout(location = 0) in vec4 vColor;\n"
        "layout(location = 0) out vec4 oFrag;\n"
        "void main()\n"
        "{\n"
        "  bool success = true;\n"
        "\n"
        "  // Test OpSubgroupAllKHR:\n"
        "  //   Set result = OpSubgroupAllKHR(true) \n"
        "  //   Assert result == true\n"
        "  //   Set result = OpSubgroupAllKHR(false)\n"
        "  //   Assert result == false\n"
        "  success = success && subgroupAll(true);\n"
        "  success = success && !subgroupAll(false);\n"
        "\n"
        "  // Test OpSubgroupAnyKHR:\n"
        "  //    Set result = OpSubgroupAnyKHR(false)\n"
        "  //    Assert result == false\n"
        "  //    Set result = OpSubgroupAnyKHR(true)\n"
        "  //    Assert result == true\n"
        "  success = success && !subgroupAny(false);\n"
        "  success = success && subgroupAny(true);\n"
        "\n"
        "  // Test OpSubgroupAllEqualKHR:\n"
        "  //    Set result = OpSubgroupAllEqualKHR(false)\n"
        "  //    Assert result == true\n"
        "  success = success && subgroupAllEqual(false);\n"
        "  success = success && subgroupAllEqual(true);\n"
        "\n"
        "\n"
        "  if (success) {\n"
        "    oFrag = vColor;\n"
        "  } else {\n"
        "    oFrag = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "  }\n"
        "}\n"
        ;

    FragmentShader fss2(450);
    fss2.addExtension(lwShaderExtension::KHR_shader_subgroup_basic);
    fss2.addExtension(lwShaderExtension::KHR_shader_subgroup_ballot);
    fss2.addExtension(lwShaderExtension::KHR_shader_subgroup_vote);
    fss2 <<
        "layout(location = 0) in vec4 vColor;\n"
        "layout(location = 0) out vec4 oFrag;\n"
        "void main()\n"
        "{\n"
        "  const bool odd = (gl_SubgroupIlwocationID & 1) != 0;\n"
        "  const bool even = !odd;\n"
        "  bool A, B, C, D;\n"
        "  A = false;\n"
        "  B = true;\n"
        "  C = false;\n"
        "  D = false;\n"
        "\n"
        "  if (even) {\n"
        "    A = subgroupAll(even);\n"
        "  } else {\n"
        "    A = subgroupAll(odd);\n"
        "  }\n"
        "\n"
        "  if (even) {\n"
        "    B = subgroupAny(odd);\n"
        "  } else {\n"
        "    B = subgroupAny(even);\n"
        "  }\n"
        "\n"
        "  if (even) {\n"
        "    C = subgroupAllEqual(odd);\n"
        "    D = subgroupAllEqual(even);\n"
        "  } else {\n"
        "    C = subgroupAllEqual(even);\n"
        "    D = subgroupAllEqual(odd);\n"
        "  }\n"
        "\n"
        "  // Verify: A == true, B == false, C == true, D == true;\n"
        "  if (A && !B && C && D) {\n"
        "    oFrag = vColor;\n"
        "  } else {\n"
        "    oFrag = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "  }\n"
        "}\n"
        ;

    // Two programs (one for each cell).
    FragmentShader * fshaders[CELL_COUNT] = {
        &fss1,
        &fss2
    }; // If the count ever changes then will have to do something different here; but compiler should warn.

    // Compile pass
    for (int i = 0; i < CELL_COUNT; ++i) {
        compiledPrograms[i] = device->CreateProgram();
        LWNboolean compiled = g_glslcHelper->CompileAndSetShaders(compiledPrograms[i], vss, *fshaders[i]);
        if (!compiled) {
            printf("Error compiling program %d:\n%s\n", i, g_glslcHelper->GetInfoLog());
            LWNFailTest();
            return LWN_FALSE;
        }
    }

    return LWN_TRUE;
}

// This function is rather simple - since the shaders are self-validating (green/blue with no special inputs),
// we just set up vertex attributes / shaders and just draw a single triangle (for each cell).
void LwnShaderSubgroupBallotVote::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Just a simple triangle
    static Vertex const vertices[] = {
        { { -0.5f, -0.5f }, },
        { {  0.5f, -0.5f }, },
        { {  0.5f,  0.5f }, },
    };

    // allocator will create pool at first allocation
    // 64k is more than for this test but anything below 64k always allocates one page in the sparse pool
    const LWNsizeiptr poolSize = 0x10000UL;
    MemoryPoolAllocator allocator(device, NULL, poolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream posStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(posStream, Vertex, position);
    VertexStreamSet streamSet(posStream);
    VertexArrayState vertex = streamSet.CreateVertexArrayState();
    Buffer *posVbo = posStream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertices), allocator, vertices);
    BufferAddress posAddr = posVbo->GetAddress();

    // Compile one of the two shaders depending on the test type
    Program * programs[CELL_COUNT];
    switch (m_testType) {
    case TEST_TYPE_BALLOT:
        if (!CompileBallotShaders(programs)) {
            return;
        }
        break;
    case TEST_TYPE_VOTE:
        if (!CompileVoteShaders(programs)) {
            return;
        }
        break;
    default:
        printf("Invalid test type !\n");
        return;
        break;
    }

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);

    // Draw pass
    for (int cell = 0; cell < CELL_COUNT; ++cell) {
        int cellX = cell % CELLS_X;
        int cellY = cell / CELLS_X;

        queueCB.SetViewportScissor(cellX*(lwrrentWindowWidth / CELLS_X), cellY*(lwrrentWindowHeight / CELLS_Y),
                                   (lwrrentWindowWidth / CELLS_X), (lwrrentWindowHeight / CELLS_Y));
        queueCB.BindProgram(programs[cell], ShaderStageBits::ALL_GRAPHICS_BITS);
        queueCB.BindVertexArrayState(vertex);
        queueCB.BindVertexBuffer(0, posAddr, sizeof(vertices));
        queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
    }

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LwnShaderSubgroupBallotVote, lwn_shader_subgroup_ballot, (TEST_TYPE_BALLOT));
OGTEST_CppTest(LwnShaderSubgroupBallotVote, lwn_shader_subgroup_vote, (TEST_TYPE_VOTE));
