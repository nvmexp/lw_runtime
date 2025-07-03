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

#define LWN_HLSL_LOG_OUTPUT     0

#if LWN_HLSL_LOG_OUTPUT >= 1
#define LOG(x) printf x; fflush(stdout);
#else
#define LOG(x)
#endif

using namespace lwn;

class LWNHlslTest
{
private:
    static const int cellSize = 64;
    static const int cellMargin = 2;
    static const int cellsX = 640 / (cellSize + cellMargin);
    static const int cellsY = 480 / (cellSize + cellMargin);

public:
    LWNHlslTest();
    ~LWNHlslTest();
    void setCellRect(CommandBuffer *queueCB, int testId) const;
    LWNTEST_CppMethods();
};

LWNHlslTest::LWNHlslTest()
{
}

LWNHlslTest::~LWNHlslTest()
{
}

void LWNHlslTest::setCellRect(CommandBuffer *queueCB, int testId) const
{
    int cx = testId % cellsX;
    int cy = testId / cellsX;

    queueCB->SetViewport(cx * (cellSize + cellMargin), cy * (cellSize + cellMargin),
        cellSize, cellSize);
    queueCB->SetScissor(cx * (cellSize + cellMargin), cy * (cellSize + cellMargin),
        cellSize, cellSize);
}

lwString LWNHlslTest::getDescription() const
{
    return
        "Simple test exercising HLSL shaders in LWN. Each cell is a subtest of HLSL. "
        "All subtests require the DXC library to be available at runtime, otherwise the "
        "test will not be supported. \n"
        "The first cell displays a triangle using the HLSL vertex shader and pixel shader. ";
}

int LWNHlslTest::isSupported() const
{
    return g_dxcLibraryHelper->IsLoaded();
}

static void hlslBasic()
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const char* vs_string =
        "struct PSInput {\n"
        "  float4 position : SV_Position;\n"
        "  float4 color : COLOR;\n"
        "};\n"
        "PSInput main(float4 position : POSITION, float4 color : COLOR) {\n"
        "  PSInput result;\n"
        "  result.position = position;\n"
        "  result.color = color;\n"
        "  return result;\n"
        "}\n";

    const char* ps_string =
        "float4 main(float4 input : COLOR) : SV_Target\n"
        "{\n"
        "  return input;\n"
        "}\n";

    ShaderStage stages[] =
    {
        ShaderStage::VERTEX,
        ShaderStage::FRAGMENT
    };
    const char * shaders[] = {
        vs_string,
        ps_string
    };
    Program *pgm = device->CreateProgram();

    if (!g_glslcHelper->CompileAndSetShadersHLSL(pgm, stages, __GL_ARRAYSIZE(stages), shaders)) {
        LOG(("HLSL Shader compile error.\n"));
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-0.375, -0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-0.375, +0.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+0.375, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
    };

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, 3 * sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 3, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);

    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);

    queueCB.submit();
    queue->Finish();
}

typedef void (*TEST_FUNCTION_TYPE)(void);

// Add tests here!!
// This array contains function pointers for each test.  The tests
// need to draw the results in their cells.
// To add a test, create the function with the interface TEST_FUNCTION_TYPE,
// and add the entry to the blow array.
static TEST_FUNCTION_TYPE testFunctions[] =
{
    hlslBasic,
};

void LWNHlslTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);

    unsigned int numTests = __GL_ARRAYSIZE(testFunctions);
    assert(numTests <= cellsX * cellsY);
    for (unsigned int i = 0; i < numTests; i++) {
        setCellRect(queueCB, i);

        // Call the test function.
        testFunctions[i]();
    }
}

OGTEST_CppTest(LWNHlslTest, lwn_hlsl, );
