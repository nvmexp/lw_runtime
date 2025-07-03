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
#include "cppogtest.h"
#include "lwn_utils.h"

#define _USE_MATH_DEFINES 1
#include <math.h>

using namespace lwn;

class LWNUniformBufferUpdateTest
{
public:
    struct Matrix {
        dt::vec4    m[4];
    };

    LWNTEST_CppMethods();
};

static void buildMatrix(LWNUniformBufferUpdateTest::Matrix &M, float angle, dt::vec3 const &scale, dt::vec4 const &t)
{
    M.m[0] = dt::vec4(scale.x() * cosf(angle), -scale.y() * sinf(angle),      0.0f, 0.0f);
    M.m[1] = dt::vec4(scale.x() * sinf(angle),  scale.y() * cosf(angle),      0.0f, 0.0f);
    M.m[2] = dt::vec4(                   0.0f,                     0.0f, scale.z(), 0.0f);
    M.m[3] = t;
}

lwString LWNUniformBufferUpdateTest::getDescription() const
{
    lwStringBuf sb;
    sb << "This test updates UBOs using memcpy and inband updates. Each test draws a number "
          "of triangles which have different position and orientation. The transformation matrix "
          "for each triangle is stored in an UBO. The first 4 rows are drawn using a UBO in "
          "system memory which is updated using memcpy, where each row has its own memory. "
          "The next 4 rows are drawn using an UBO which is allocated in GPU memory and which is "
          "repeatedly updated using inband updates."
          "\n\n"
          "The test output are 8 rows of triangles, each row shows 8 triangles where each one is "
          "rotated slightly around z compared top its predecessor\n";
    return sb.str();
}

int LWNUniformBufferUpdateTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(40, 4);
}

void LWNUniformBufferUpdateTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const unsigned int numTriangles = 16;
    const unsigned int numRowsPerTest = 4;
    const unsigned int numRows = 2 * numRowsPerTest;
    const unsigned int uboRowSize = sizeof(Matrix) * numTriangles;
    const unsigned int uboSize = uboRowSize * numRowsPerTest;

    LWNint uboAlignment;

    device->GetInteger(DeviceInfo::UNIFORM_BUFFER_ALIGNMENT, &uboAlignment);

    assert(uboRowSize % uboAlignment == 0);

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec4 position;\n"
        "layout(location=1) in vec3 color;\n"
        "layout(binding=0, std140) uniform Block {\n"
        "  mat4 M[" << numTriangles << "];\n"
        "};\n"
        "out vec3 ocolor;\n"
        "void main() {\n"
        "  gl_Position = M[gl_InstanceID] * position;\n"
        "  ocolor = color;\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(ocolor, 1.0);\n"
        "}\n";

    Program *pgm = device->CreateProgram();

    // Compile and call lwnProgramSetShaders.
    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        printf("Compile error:\n%s\n", g_glslcHelper->GetInfoLog());
        LWNFailTest();
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3( 1.0, -1.0, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3( 0.0,  1.0, 0.0), dt::vec3(1.0, 0.0, 0.0) },
    };

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, 3 * sizeof(vertexData) + uboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 3, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // Create an UBO in system memory which will be used by the fist tests
    BufferBuilder uboBuilder;
    uboBuilder.SetDefaults();
    uboBuilder.SetDevice(device);

    Buffer *uboSysMem = allocator.allocBuffer(&uboBuilder, BUFFER_ALIGN_UNIFORM_BIT, uboSize);
    char *ptr = (char*)uboSysMem->Map();
    BufferAddress uboSysMemAddr = uboSysMem->GetAddress();

    // Create UBo in GPU mem which will be used for inband updates.  We only
    // allocate storage for one row and repeatedly overwrite.
    MemoryPool *uboGpuMem = device->CreateMemoryPool(NULL, uboRowSize, MemoryPoolType::GPU_ONLY);
    BufferAddress uboGpuMemAddr = uboGpuMem->GetBufferAddress();

    queueCB.ClearColor(0, 0.0, 0.0, 0.4, 0.0);
    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    for (unsigned int n = 0; n < numRows; ++n) {
        Matrix M[numTriangles];

        for (unsigned int i = 0; i < numTriangles; ++i) {
            const dt::vec3 scale(1.0f / static_cast<float>(numTriangles + 0.2f), 1.0f / static_cast<float>(numRows + 0.2f), 1.0f);

            buildMatrix(M[i], i * M_PI_2 / numTriangles, scale, dt::vec4(-1.0f + (1.0f / numTriangles) + i * (2.0f / numTriangles), 1.0 - (1.0f / numRows) - n * (2.0f / numRows), 0.0f, 1.0f));
        }

        if (n < numRowsPerTest) {
            memcpy(ptr + n * uboRowSize, M, uboRowSize);
            queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboSysMemAddr + n * uboRowSize, uboRowSize);
        }
        else
        {
            queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboGpuMemAddr, uboRowSize);

            // Throw in a little variety in updates.  Either update the full
            // buffer or in pieces.  Also throw in a zero-sized update for fun.
            if (n & 1) {
                queueCB.UpdateUniformBuffer(uboGpuMemAddr, uboRowSize, 0, uboRowSize, M);
            } else {
                queueCB.UpdateUniformBuffer(uboGpuMemAddr, uboRowSize, sizeof(Matrix), uboRowSize - sizeof(Matrix), M + 1);
                queueCB.UpdateUniformBuffer(uboGpuMemAddr, uboRowSize, 0, sizeof(Matrix), M);
                queueCB.UpdateUniformBuffer(uboGpuMemAddr, uboRowSize, 0, 0, M);
            }
        }

        queueCB.DrawArraysInstanced(DrawPrimitive::TRIANGLES, 0, 3, 0, numTriangles);
    }

    queueCB.submit();
    queue->Finish();

    uboGpuMem->Free();
}

OGTEST_CppTest(LWNUniformBufferUpdateTest, lwn_uniformbuffer, );
