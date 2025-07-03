/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-LwidiaProprietary
 *
 * LWPU CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from LWPU CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <array>
#include <vector>

#include "lwntest_cpp.h"
#include "lwn_utils.h"

namespace {

using namespace lwn;

class ProvokingVertexTest {
public:
    ProvokingVertexTest();
    LWNTEST_CppMethods();

private:
    static constexpr const LWNsizeiptr programPoolSize = 0x100000UL; // 1MB pool size

    struct UniformBlock {
        uint32_t primitiveType;
        uint32_t vertexCount;
        uint32_t provokingMode;
        uint32_t padding;
    };

    // Color value is not interpolated.
    static constexpr const char *vsstring = 
        "layout(location = 0) in vec3 position;\n"
        "layout(location = 1) in uint vertexIndex;\n"
        "layout(location = 0) out flat uint outVertexIndex;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1);\n"
        "  outVertexIndex = vertexIndex;\n"
        "}\n";

    // Color value is not interpolated.
    static constexpr const char *fsstring = 
        "layout(location = 0) in flat uint vertexIndex;\n"
        "layout(location = 0) out vec4 color;\n"
        "\n"
        "layout(binding = 0, std140) uniform Block {\n"
        "  uvec4 setup;\n"
        "};\n"
        "\n"
        "void main() {\n"
        "  uint primitiveType = setup.x;\n"
        "  uint vertexCount   = setup.y;\n"
        "  bool provokingLast = (setup.z == ProvokingVertexMode_Last);\n"
        "\n"
        "  bool result = false;\n"
        "  if (primitiveType == DrawPrimitive_POINTS) {\n"
        "    result = vertexIndex == gl_PrimitiveID;\n"
        "  } else if (primitiveType == DrawPrimitive_LINES) {\n"
        "    result = vertexIndex == gl_PrimitiveID * 2 + (provokingLast ? 1 : 0);\n"
        "  } else if (primitiveType == DrawPrimitive_LINE_STRIP) {\n"
        "    result = vertexIndex == gl_PrimitiveID + (provokingLast ? 1 : 0);\n"
        "  } else if (primitiveType == DrawPrimitive_LINE_LOOP) {\n"
        "    result = vertexIndex == (gl_PrimitiveID + (provokingLast ? 1 : 0)) % vertexCount;\n"
        "  } else if (primitiveType == DrawPrimitive_TRIANGLES) {\n"
        "    result = vertexIndex == gl_PrimitiveID * 3 + (provokingLast ? 2 : 0);\n"
        "  } else if (primitiveType == DrawPrimitive_TRIANGLE_STRIP) {\n"
        "    result = vertexIndex == gl_PrimitiveID + (provokingLast ? 2 : 0);\n"
        "  } else if (primitiveType == DrawPrimitive_TRIANGLE_FAN) {\n"
        "    result = vertexIndex == gl_PrimitiveID + (provokingLast ? 2 : 1);\n"
        "  } else if (primitiveType == DrawPrimitive_QUADS) {\n"
        "    result = vertexIndex == gl_PrimitiveID * 4 + (provokingLast ? 3 : 0);\n"
        "  } else if (primitiveType == DrawPrimitive_QUAD_STRIP) {\n"
        "    result = vertexIndex == gl_PrimitiveID * 2 + (provokingLast ? 3 : 0);\n"
        "  } else if (primitiveType == DrawPrimitive_LINES_ADJACENCY) {\n"
        "    result = vertexIndex == gl_PrimitiveID * 4 + (provokingLast ? 2 : 1);\n"
        "  } else if (primitiveType == DrawPrimitive_LINE_STRIP_ADJACENCY) {\n"
        "    result = vertexIndex == gl_PrimitiveID + (provokingLast ? 2 : 1);\n"
        "  } else if (primitiveType == DrawPrimitive_TRIANGLES_ADJACENCY) {\n"
        "    result = vertexIndex == gl_PrimitiveID * 6 + (provokingLast ? 4 : 0);\n"
        "  } else if (primitiveType == DrawPrimitive_TRIANGLE_STRIP_ADJACENCY) {\n"
        "    result = vertexIndex == (gl_PrimitiveID + (provokingLast ? 2 : 0)) * 2;\n"
        "  }\n"
        "\n"
        "  if (result) {\n"
        "    color = vec4(0.0, 1.0, 0.0, 1.0); // pass\n"
        "  } else {\n"
        "    color = vec4(1.0, 0.0, 0.0, 1.0); // fail\n"
        "  }\n"
        "}\n";

    struct Subtest {
        DrawPrimitive primitiveType;
        std::vector<size_t> indices;
    };

    std::vector<Subtest> subtests;
    std::vector<std::array<float,2>> xy;
};

ProvokingVertexTest::ProvokingVertexTest() {
    xy = {
        {{ -0.9f,  0.9f }},
        {{  0.0f,  0.9f }},
        {{  0.9f,  0.9f }},
        {{ -0.45f, 0.0f }},
        {{  0.45f, 0.0f }},
        {{ -0.9f, -0.9f }},
        {{  0.0f, -0.9f }},
        {{  0.9f, -0.9f }},
    };

    subtests = {
        { DrawPrimitive::POINTS, { 0, 1, 2, 3, 4, 5, 6, 7 } },
        { DrawPrimitive::LINES, { 0, 1,  2, 3,  4, 5,  6, 7 } },
        { DrawPrimitive::LINE_STRIP, { 0, 1, 2, 3, 4, 5, 6, 7 } },
        { DrawPrimitive::LINE_LOOP, { 0, 1, 2, 3, 4, 5, 6, 7 } },
        { DrawPrimitive::TRIANGLES, { 2, 1, 4,  4, 1, 3,  4, 3, 7,  7, 3, 6 } },
        { DrawPrimitive::TRIANGLE_STRIP, { 2, 1, 4,  3,  7,  6 } },
        { DrawPrimitive::TRIANGLE_FAN, { 4, 2, 1,  3,  6,  7 } },
        { DrawPrimitive::QUADS, { 2, 1, 3, 4,  4, 3, 6, 7 } },
        { DrawPrimitive::QUAD_STRIP, { 2, 1, 4, 3,  7, 6 } },
        { DrawPrimitive::LINES_ADJACENCY, { 0, 1, 2, 3,  4, 5, 6, 7 } },
        { DrawPrimitive::LINE_STRIP_ADJACENCY, { 0, 1,  2, 3,  4, 5,  6, 7 } },
        { DrawPrimitive::TRIANGLES_ADJACENCY, { 1, 0, 3, 6, 4, 2,  6, 7, 4, 1, 3, 5 } },
        { DrawPrimitive::TRIANGLE_STRIP_ADJACENCY, { 1, 0,  3, 2,  4, 5,  6, 7 } },
    };
}


int ProvokingVertexTest::isSupported() const
{
    return 1;
}

lwString ProvokingVertexTest::getDescription() const
{
    lwStringBuf sb;
    sb << "Test for provoking vertex modes.  "
          "This test renders different primitives in different provoking vertex modes "
          "with a same set of vertices.  The bottom row exercises ProvokingVertex:LAST, "
          "and the top row exercises ProvokingVertex::FIRST.  "
          "Each column draws ";

    const size_t subtestCount = subtests.size();
    for (size_t subtestIndex = 0; subtestIndex < subtestCount; subtestIndex++) {
        switch (subtests[subtestIndex].primitiveType) {
        case DrawPrimitive::POINTS:                    sb << "point"; break;
        case DrawPrimitive::LINES:                     sb << "independent line"; break;
        case DrawPrimitive::LINE_STRIP:                sb << "line strip"; break;
        case DrawPrimitive::LINE_LOOP:                 sb << "line loop"; break;
        case DrawPrimitive::TRIANGLES:                 sb << "independent triangle"; break;
        case DrawPrimitive::TRIANGLE_STRIP:            sb << "triangle strip"; break;
        case DrawPrimitive::TRIANGLE_FAN:              sb << "triangle fan"; break;
        case DrawPrimitive::QUADS:                     sb << "independent quad"; break;
        case DrawPrimitive::QUAD_STRIP:                sb << "quad strip"; break;
        case DrawPrimitive::LINES_ADJACENCY:           sb << "independent adjacency line"; break;
        case DrawPrimitive::LINE_STRIP_ADJACENCY:      sb << "adjacency line strip"; break;
        case DrawPrimitive::TRIANGLES_ADJACENCY:       sb << "independent adjacency triangle"; break;
        case DrawPrimitive::TRIANGLE_STRIP_ADJACENCY:  sb << "adjacency triangle strip"; break;
        default:
            assert(!"Not supported draw primitive type.");
            break;
        }

        const bool isLast = (subtestIndex == subtestCount - 1);
        if (!isLast) {
            sb << ", ";
        }
    }

    sb << " primitives in the order.  "
          "Fragment will be drawn in green when the test is passing; "
          "otherwise it will be drawn in red.";

    return sb.str();
}

void ProvokingVertexTest::doGraphics() const
{
    DeviceState* deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer& queueCB = deviceState->getQueueCB();
    Queue* queue = deviceState->getQueue();

    const size_t subtestCount = subtests.size();

    cellTestInit(static_cast<uint32_t>(subtestCount), 2);
    queueCB.ClearColor(0.0f, 0.1f, 0.0f, 0.0f, 1.0f);
    queueCB.SetPointSize(3.0);
    queueCB.SetLineWidth(2.0);

    size_t totalVertexCount = 0;
    std::vector<int> vertexOffsets;
    vertexOffsets.reserve(subtestCount);

    for (const Subtest &subtest : subtests) {
        vertexOffsets.emplace_back(static_cast<int>(totalVertexCount));
        totalVertexCount += subtest.indices.size();
    }

    constexpr const size_t vertexStride = sizeof(float) * 3 + sizeof(uint32_t); // x, y, z, vertexIndex
    const size_t vboSize = vertexStride * totalVertexCount;

    MemoryPoolAllocator bufferAllocator(device, NULL, vboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // Set up a persistently mapped VBO where we can write in our vertex attribute values.
    BufferBuilder vboBuilder;
    vboBuilder.SetDefaults()
              .SetDevice(device);
    Buffer *vbo = bufferAllocator.allocBuffer(&vboBuilder, BUFFER_ALIGN_VERTEX_BIT, vboSize);
    uint8_t *vboMem = (uint8_t*)vbo->Map();
    assert(vboMem);

    void *vboNext = vboMem;
    for (const Subtest &subtest : subtests) {
        uint32_t vertexIndex = 0;
        for (size_t index : subtest.indices) {
            const auto &vertex = xy[index];

            for (float xyz : {std::get<0>(vertex), std::get<1>(vertex), 0.f}) {
                *static_cast<float*>(vboNext) = xyz;
                vboNext = static_cast<float*>(vboNext) + 1;
            }

            *static_cast<uint32_t*>(vboNext) = vertexIndex;
            vboNext = static_cast<uint32_t*>(vboNext) + 1;
            vertexIndex++;
        }
    }
    assert(vboNext == vboMem + vboSize);
    queueCB.BindVertexBuffer(0, vbo->GetAddress(), vboSize);

    VertexStreamState streamState;
    streamState.SetDefaults().SetStride(vertexStride);
    queueCB.BindVertexStreamState(1, &streamState);

    VertexAttribState attribStates[2];
    attribStates[0].SetDefaults().SetFormat(Format::RGB32F, 0); // xyz
    attribStates[1].SetDefaults().SetFormat(Format::R32UI, 3 * sizeof(float)); // vertexIndex
    queueCB.BindVertexAttribState(2, attribStates);

    VertexShader vs(430);
    vs << vsstring;

    FragmentShader fs(430);
    fs << "const uint ProvokingVertexMode_Last = "               << ProvokingVertexMode::LAST << ";\n";
    fs << "const uint DrawPrimitive_POINTS = "                   << DrawPrimitive::POINTS << ";\n";
    fs << "const uint DrawPrimitive_LINES = "                    << DrawPrimitive::LINES << ";\n";
    fs << "const uint DrawPrimitive_LINE_STRIP = "               << DrawPrimitive::LINE_STRIP << ";\n";
    fs << "const uint DrawPrimitive_LINE_LOOP = "                << DrawPrimitive::LINE_LOOP << ";\n";
    fs << "const uint DrawPrimitive_TRIANGLES = "                << DrawPrimitive::TRIANGLES << ";\n";
    fs << "const uint DrawPrimitive_TRIANGLE_STRIP = "           << DrawPrimitive::TRIANGLE_STRIP << ";\n";
    fs << "const uint DrawPrimitive_TRIANGLE_FAN = "             << DrawPrimitive::TRIANGLE_FAN << ";\n";
    fs << "const uint DrawPrimitive_QUADS = "                    << DrawPrimitive::QUADS << ";\n";
    fs << "const uint DrawPrimitive_QUAD_STRIP = "               << DrawPrimitive::QUAD_STRIP << ";\n";
    fs << "const uint DrawPrimitive_LINES_ADJACENCY = "          << DrawPrimitive::LINES_ADJACENCY << ";\n";
    fs << "const uint DrawPrimitive_LINE_STRIP_ADJACENCY = "     << DrawPrimitive::LINE_STRIP_ADJACENCY << ";\n";
    fs << "const uint DrawPrimitive_TRIANGLES_ADJACENCY = "      << DrawPrimitive::TRIANGLES_ADJACENCY << ";\n";
    fs << "const uint DrawPrimitive_TRIANGLE_STRIP_ADJACENCY = " << DrawPrimitive::TRIANGLE_STRIP_ADJACENCY << ";\n";
    fs << fsstring;

    lwnTest::GLSLCHelper glslcHelper(device, programPoolSize, g_glslcLibraryHelper, g_glslcHelperCache);
    MemoryPool *scratchMemoryPool = device->CreateMemoryPool(NULL, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE, MemoryPoolType::GPU_ONLY);
    glslcHelper.SetShaderScratchMemory(scratchMemoryPool, 0, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE, NULL);

    Program *pgm = device->CreateProgram();
    if (!glslcHelper.CompileAndSetShaders(pgm, vs, fs)) {
        LWNFailTest();
        return;
    }

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    for (int col = 0; col < static_cast<int>(subtestCount); ++col) {
        const Subtest &subtest = subtests[col];
        const DrawPrimitive primType = subtest.primitiveType;
        const int vertexOffset = vertexOffsets[col];
        const int vertexCount = static_cast<int>(subtest.indices.size());

        for (int row = 0; row < 2; ++row) {
            if (!cellAllowed(col, row)) {
                continue;
            }

            SetCellViewportScissorPadded(queueCB, col, row, 0);

            ProvokingVertexMode provokingMode;
            if (row == 0) {
                // bottom
                provokingMode = ProvokingVertexMode::LAST;
            } else {
                // top
                provokingMode = ProvokingVertexMode::FIRST;
            }

            UniformBlock uboData;
            uboData.primitiveType = subtest.primitiveType;
            uboData.vertexCount = static_cast<uint32_t>(subtest.indices.size());
            uboData.provokingMode = provokingMode;

            Buffer *ubo = AllocAndFillBuffer(device, queue, queueCB, bufferAllocator, &uboData, sizeof(uboData),
                            BUFFER_ALIGN_UNIFORM_BIT, false);

            queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, ubo->GetAddress(), sizeof(uboData));
            queueCB.SetProvokingVertexMode(provokingMode);
            queueCB.DrawArrays(primType, vertexOffset, vertexCount);
        }
    }

    queueCB.SetProvokingVertexMode(ProvokingVertexMode::LAST);
    queueCB.submit();
    queue->Finish();
}

} // unnamed namespace

OGTEST_CppTest(ProvokingVertexTest, lwn_provoking_vertex, /* no parameters */);
