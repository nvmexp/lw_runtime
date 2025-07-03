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

class LWNCmdAddMemoryTest
{
public:
    LWNTEST_CppMethods();
};

lwString LWNCmdAddMemoryTest::getDescription() const
{
    lwStringBuf sb;
    sb << "This test is based on lwn_01tri_cpp as of commit 755f2824.\n"
          "We want to test Add*Memory functions outside of OOM callbacks,\n"
          "while recording a CommandBuffer.";
    return sb.str();
}

int LWNCmdAddMemoryTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(5, 0);
}

void LWNCmdAddMemoryTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();

    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    CommandBuffer *cmdBuf = device->CreateCommandBuffer();

    int command_size = 0;
    int control_size = 0;
    device->GetInteger(DeviceInfo::COMMAND_BUFFER_MIN_COMMAND_SIZE, &command_size);
    device->GetInteger(DeviceInfo::COMMAND_BUFFER_MIN_CONTROL_SIZE, &control_size);

    command_size *= 2;
    control_size *= 2;

    // CPU_COHERENT means CPU_UNCACHED | GPU_CACHED | COMPRESSIBLE
    MemoryPool *mp = device->CreateMemoryPool(NULL, 2 * command_size, MemoryPoolType::CPU_COHERENT);

    int control_allocated = 2 * control_size;
    char *control_space = new char[control_allocated];
    memset(control_space, 0, control_allocated);

    ptrdiff_t last_ctrl_offset = 0;
    ptrdiff_t last_cmd_offset = 0;

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 color;\n"
        "out vec3 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
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
    g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-0.375, -0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-0.375, +0.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+0.375, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        // small triangle below
        { dt::vec3(+0.375,  0.0, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(+0.375, +0.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+0.0  ,  0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
    };

    // Allocator will create pool at first allocation.
    MemoryPoolAllocator allocator(device, NULL, 3 * sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();

    Buffer *vbo = stream.AllocateVertexBuffer(device, 6, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // This is the 'classic' use: outside of a recording.
    cmdBuf->AddCommandMemory(mp, last_cmd_offset, command_size);
    last_cmd_offset += command_size;
    cmdBuf->AddControlMemory(control_space + last_ctrl_offset, control_size);
    last_ctrl_offset += control_size;

    cmdBuf->BeginRecording();
    {
        cmdBuf->SetViewport(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
        cmdBuf->SetScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);

        cmdBuf->ClearColor(0, 0.4, 0.0, 0.0, 0.0);

        cmdBuf->BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
        cmdBuf->BindVertexArrayState(vertex);

        // Have a draw before switching the command/control memory.
        cmdBuf->BindVertexBuffer(0, vboAddr, sizeof(vertexData));
        cmdBuf->DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);

        // This is the actual test: switch memory, while recording.
        cmdBuf->AddCommandMemory(mp, last_cmd_offset, command_size);
        last_cmd_offset += command_size;
        cmdBuf->AddControlMemory(control_space + last_ctrl_offset, control_size);
        last_ctrl_offset += control_size;
        // If we haven't crashed here, we are ok.

        // And a draw after switching.
        cmdBuf->DrawArrays(DrawPrimitive::TRIANGLES, 3, 6);
    }
    CommandHandle handle = cmdBuf->EndRecording();

    // Doing first the queueCB.submit sets up the elw.
    queueCB.submit();
    // Here-after we inherit a state in queue that
    // will properly display our triangles.
    queue->SubmitCommands(1, &handle);
    queue->Finish();

    pgm->Free();
    cmdBuf->Free();
    delete[] control_space;
    mp->Free();
}

OGTEST_CppTest(LWNCmdAddMemoryTest, lwn_cmdbuf_add_memory, );
