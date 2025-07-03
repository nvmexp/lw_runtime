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

class LWNDebugLabel
{
public:
    LWNTEST_CppMethods();
};

lwString LWNDebugLabel::getDescription() const
{
    lwStringBuf sb;
    sb << "Devtools debug label touch-test.\n"
          "Test only uses functions and checks that nothing's broken,";
    return sb.str();
}

int LWNDebugLabel::isSupported() const
{
    return lwogCheckLWNAPIVersion(53, 13);
}

void LWNDebugLabel::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &cmd = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    {
        LWNdebugDomainId firstDomain = device->GenerateDebugDomainId("unused");
        LWNdebugDomainId secondDomain = device->GenerateDebugDomainId("unused");
        if (firstDomain == secondDomain) {
            lwnTest::fail("Debug domains should be unique.");
            return;
        }
    }

    device->SetDebugLabel("myDevice");
    cmd.SetDebugLabel("myCmd");
    queue->SetDebugLabel("myQueue");
    TexturePool* texPool = (TexturePool*)g_lwnTexIDPool->GetTexturePool();
    SamplerPool* smpPool = (SamplerPool*)g_lwnTexIDPool->GetSamplerPool();
    texPool->SetDebugLabel("myTexturePool");
    smpPool->SetDebugLabel("mySamplerPool");
    LWNwindow* win = g_lwnWindowFramebuffer.getWindow();
    lwnWindowSetDebugLabel(win, "myWindow");

    cmd.PushDebugGroup("Creating sampler");
    Sampler* smp;
    SamplerBuilder sb;
    sb.SetDevice(device);
    sb.SetDefaults();
    sb.SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    smp = sb.CreateSampler();
    smp->SetDebugLabel("mySampler");
    cmd.PopDebugGroup();

    cmd.PushDebugGroupStatic(1, "Creating texture");
    TextureBuilder tb;
    tb.SetDevice(device);
    tb.SetDefaults();
    tb.SetTarget(TextureTarget::TARGET_1D);
    tb.SetSize1D(3);
    tb.SetFormat(Format::RGBA8);
    LWNsizeiptr tbSize = tb.GetStorageSize();
    MemoryPool* texGpuMemPool = device->CreateMemoryPool(NULL, tbSize, MemoryPoolType::GPU_ONLY);
    texGpuMemPool->SetDebugLabel("myTexGpuPool");
    Texture* tex = tb.CreateTextureFromPool(texGpuMemPool, 0);
    tex->SetDebugLabel("myTexture");
    cmd.PushDebugGroupDynamic(2, "Filling texture");
    MemoryPool* texCpuMemPool = device->CreateMemoryPool(NULL, tbSize, MemoryPoolType::CPU_COHERENT);
    texCpuMemPool->SetDebugLabel("myTexCpuPool");
    BufferBuilder bb;
    bb.SetDevice(device);
    bb.SetDefaults();
    Buffer* texBuffer = bb.CreateBufferFromPool(texCpuMemPool, 0, tbSize);
    dt::u8lwec4* texBuffPtr = static_cast<dt::u8lwec4*>(texBuffer->Map());
    texBuffPtr[0] = dt::u8lwec4(1.0, 0.0, 0.0, 1.0);
    texBuffPtr[1] = dt::u8lwec4(0.0, 1.0, 0.0, 1.0);
    texBuffPtr[2] = dt::u8lwec4(0.0, 0.0, 1.0, 1.0);
    CopyRegion cr = {0,0,0,3,1,1};
    cmd.CopyBufferToTexture(texBuffer->GetAddress(), tex, 0, &cr, CopyFlags::NONE);
    Sync* sync = device->CreateSync();
    sync->SetDebugLabel("mySync");
    queue->FenceSync(sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, SyncFlagBits::FLUSH_FOR_CPU);
    queue->WaitSync(sync);
    texBuffer->Free();
    cmd.PopDebugGroupId(2);
    TextureHandle texHandle = device->GetTextureHandle(tex->GetRegisteredTextureID(), smp->GetRegisteredID());
    cmd.PopDebugGroupId(1);

    cmd.PushDebugGroup("Creating shader program");
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout (binding=0) uniform sampler1D tex;\n"
        "out vec3 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ocolor = texture(tex, (float(gl_VertexID)+0.5)/3.0).rgb;\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(ocolor, 1.0);\n"
        "}\n";

    Program *pgm = device->CreateProgram();
    pgm->SetDebugLabel("myProgram");

    // Compile and call lwnProgramSetShaders.
    g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);
    cmd.PopDebugGroup();

    // Set up the vertex format and buffer.
    cmd.PushDebugGroup("Setting attributes");
    struct Vertex {
        dt::vec3 position;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-0.375, -0.5, 0.0) },
        { dt::vec3(-0.375, +0.5, 0.0) },
        { dt::vec3(+0.375, -0.5, 0.0) },
    };
    MemoryPoolAllocator allocator(device, NULL, 3 * sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 3, allocator, vertexData);
    vbo->SetDebugLabel("myBuffer");
    BufferAddress vboAddr = vbo->GetAddress();
    cmd.PopDebugGroup();

    cmd.PushDebugGroup("Render");
    cmd.InsertDebugMarker("Clear color");
    cmd.ClearColor(0, 0.4, 0.0, 0.0, 0.0);

    cmd.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    cmd.BindTexture(ShaderStage::VERTEX, 0, texHandle);
    cmd.BindVertexArrayState(vertex);
    cmd.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    cmd.InsertDebugMarkerStatic(1, "Start draw");
    cmd.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
    cmd.PopDebugGroup();

    cmd.InsertDebugMarkerDynamic(1, "String Size Test");
    cmd.InsertDebugMarkerDynamic(1, "");
    cmd.InsertDebugMarkerDynamic(1, "1");
    cmd.InsertDebugMarkerDynamic(1, "12");
    cmd.InsertDebugMarkerDynamic(1, "123");
    cmd.InsertDebugMarkerDynamic(1, "1234");
    cmd.InsertDebugMarkerDynamic(~0L, "12345");

    cmd.submit();
    queue->Finish();

    pgm->Free();
    tex->Free();
    smp->Free();
    texGpuMemPool->Free();
    texCpuMemPool->Free();

    // test command buffer finalization
    CommandBuffer cb;
    CommandHandle cbHandle;
    cb.Initialize(device);
    MemoryPool* cbMemPool = device->CreateMemoryPool(NULL, 512, MemoryPoolType::CPU_COHERENT);
    cb.AddCommandMemory(cbMemPool, 0, 512);
    char ctrlMem[512];
    cb.AddControlMemory((void*)ctrlMem, 512);
    cb.BeginRecording();
    cb.ClearColor(0);
    cbHandle = cb.EndRecording();
    lwnDeviceFinalizeCommandHandle((LWNdevice*)device, cbHandle);
    cbMemPool->Free();
    cb.Finalize();
}

OGTEST_CppTest(LWNDebugLabel, lwn_debuglabel, );
