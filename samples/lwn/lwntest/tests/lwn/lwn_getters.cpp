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
#include "lwn_basic.h"
#include "lwn/lwn_Cpp.h"

using namespace lwn;

class LWNGettersTest
{
public:
    LWNTEST_CppMethods();
};

lwString LWNGettersTest::getDescription() const
{
    return
            "Test checks whether the values from the getters are as expected.\n"
            "Test creates multisampled framebuffer and draws four different traingles\n"
            "and various getters are tested throughout (states, builders, build objects).\n";
}

int LWNGettersTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(53, 209);
}

struct Vertex
{
    dt::vec3 position;
    dt::vec4 color;
};

#define TESTRES_MAX_CNT 200
static bool testResults[TESTRES_MAX_CNT];
static int testCount;
static void compareTest(bool val)
{
    assert(testCount<TESTRES_MAX_CNT);
    testResults[testCount] = val;
    testCount++;
}

static void drawTriangle(const Vertex verts[], TextureHandle handle = 0)
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(440);
    FragmentShader fs(440);
    Program* pgm;

    if (handle) {
        // if textured, then color attrib is treated as UV coords
        vs <<
          "layout(location=0) in vec3 position;\n"
          "layout(location=1) in vec4 color;\n"
          "out vec2 uv;\n"
          "void main() {\n"
          "  gl_Position = vec4(position, 1.0);\n"
          "  uv = color.xy;\n"
          "}\n";
        fs <<
          "in vec2 uv;\n"
          "layout (binding=0) uniform sampler2D tex;\n"
          "out vec4 fcolor;\n"
          "void main() {\n"
          "  fcolor = texture(tex,uv);\n"
          "}\n";
    }
    else {
        vs <<
            "layout(location=0) in vec3 position;\n"
            "layout(location=1) in vec4 color;\n"
            "out vec4 ocolor;\n"
            "void main() {\n"
            "  gl_Position = vec4(position, 1.0);\n"
            "  ocolor = color;\n"
            "}\n";
        fs <<
            "in vec4 ocolor;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"
            "  fcolor = ocolor;\n"
            "}\n";
    }
    pgm = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

    MemoryPoolAllocator allocator(device, NULL, 3 * sizeof(Vertex), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 3, allocator, verts);
    BufferAddress vboAddr = vbo->GetAddress();
    {
        VertexStreamState vss;
        vss.SetDivisor(1);
        vss.SetStride(sizeof(Vertex));
        compareTest(1==vss.GetDivisor());
        compareTest(sizeof(Vertex)==vss.GetStride());

        VertexAttribState vas;
        vas.SetDefaults();
        compareTest(0==vas.GetStreamIndex());
        Format vfmt;
        ptrdiff_t voff;
        vas.GetFormat(&vfmt, &voff);
        compareTest(Format::NONE==vfmt && 0==voff);

        compareTest(3*sizeof(Vertex)==vbo->GetSize());

        BufferBuilder bb;
        bb.SetDevice(device).SetDefaults().SetStorage(0,0,5);
        compareTest(5==bb.GetSize());
        compareTest(bb.GetDevice() == device);
    }

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, handle);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(Vertex)*3);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);

    queueCB.submit();
    queue->Finish();

    pgm->Free();
}

static void drawTexturedFSQuad(LWNuint texID)
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // shaders and program
    VertexShader vs(440);
    FragmentShader fs(440);
    Program *prog;
    vs <<
          "out vec2 uv;\n"
          "void main() {\n"

          "  vec2 position; "
          "  if (gl_VertexID == 0) position = vec2(-1.0, -1.0);\n"
          "  if (gl_VertexID == 1) position = vec2(1.0, -1.0);\n"
          "  if (gl_VertexID == 2) position = vec2(1.0, 1.0);\n"
          "  if (gl_VertexID == 3) position = vec2(-1.0, 1.0);\n"

          "  gl_Position = vec4(position, 0.0, 1.0);\n"
          "  uv = position*0.5 + vec2(0.5,0.5);\n"
          "}\n";
    fs <<
        "in vec2 uv;\n"
        "layout (binding=0) uniform sampler2D tex;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = texture(tex,uv);\n"
        "}\n";
    prog = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(prog, vs, fs);


    // sampler
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults().SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    Sampler* smp = sb.CreateSampler();
    LWNuint samplerID = smp->GetRegisteredID();
    TextureHandle handle = device->GetTextureHandle(texID, samplerID);
    {
        MinFilter minf;
        MagFilter magf;
        WrapMode wm[3];
        float lodclamp[2];
        CompareMode cm;
        CompareFunc cf;
        float bdcol[4];
        int bdcoli[4];
        unsigned bdcolui[4];

        compareTest(sb.GetDevice() == device);
        sb.GetMinMagFilter(&minf,&magf);
        compareTest(MinFilter::NEAREST==minf && MagFilter::NEAREST==magf);
        sb.GetWrapMode(wm, wm+1, wm+2);
        compareTest(WrapMode::REPEAT==wm[0] && WrapMode::REPEAT==wm[1] && WrapMode::REPEAT==wm[2]);
        sb.GetLodClamp(lodclamp, lodclamp+1);
        compareTest(0.0==lodclamp[0] && 1000.0==lodclamp[1]);
        compareTest(0.0==sb.GetLodBias());
        sb.GetCompare(&cm, &cf);
        compareTest(CompareMode::NONE==cm && CompareFunc::LESS==cf);
        sb.GetBorderColor(bdcol);
        compareTest(0.0==bdcol[0] && 0.0==bdcol[1] && 0.0==bdcol[2] && 0.0==bdcol[3]);
        sb.GetBorderColori(bdcoli);
        compareTest(0==bdcoli[0] && 0==bdcoli[1] && 0==bdcoli[2] && 0==bdcoli[3]);
        sb.GetBorderColorui(bdcolui);
        compareTest(0==bdcolui[0] && 0==bdcolui[1] && 0==bdcolui[2] && 0==bdcolui[3]);
        compareTest(1.0==sb.GetMaxAnisotropy());
        compareTest(SamplerReduction::AVERAGE==sb.GetReductionFilter());

        smp->GetMinMagFilter(&minf,&magf);
        compareTest(MinFilter::NEAREST==minf && MagFilter::NEAREST==magf);
        smp->GetWrapMode(wm, wm+1, wm+2);
        compareTest(WrapMode::REPEAT==wm[0] && WrapMode::REPEAT==wm[1] && WrapMode::REPEAT==wm[2]);
        smp->GetLodClamp(lodclamp, lodclamp+1);
        compareTest(0.0==lodclamp[0] && 1000.0==lodclamp[1]);
        compareTest(0.0==smp->GetLodBias());
        smp->GetCompare(&cm, &cf);
        compareTest(CompareMode::NONE==cm && CompareFunc::LESS==cf);
        smp->GetBorderColor(bdcol);
        compareTest(0.0==bdcol[0] && 0.0==bdcol[1] && 0.0==bdcol[2] && 0.0==bdcol[3]);
        smp->GetBorderColori(bdcoli);
        compareTest(0==bdcoli[0] && 0==bdcoli[1] && 0==bdcoli[2] && 0==bdcoli[3]);
        smp->GetBorderColorui(bdcolui);
        compareTest(0==bdcolui[0] && 0==bdcolui[1] && 0==bdcolui[2] && 0==bdcolui[3]);
        compareTest(1.0==smp->GetMaxAnisotropy());
        compareTest(SamplerReduction::AVERAGE==smp->GetReductionFilter());
    }

    // bind and draw
    queueCB.BindProgram(prog, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, handle);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
    queueCB.submit();
    queue->Finish();

    smp->Free();
    prog->Free();
}

void LWNGettersTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    testCount = 0;

    {
        compareTest(DepthMode::NEAR_IS_MINUS_W==device->GetDepthMode());
        compareTest(WindowOriginMode::LOWER_LEFT==device->GetWindowOriginMode());

        CommandBuffer* cmd = device->CreateCommandBuffer();
        MemoryPool* mpool;
        mpool = device->CreateMemoryPool(0, 8192, MemoryPoolType::CPU_COHERENT);
        cmd->SetMemoryCallbackData(NULL);
        cmd->AddCommandMemory(mpool, 4096, 4096);
        cmd->AddControlMemory(mpool, 4096);

        compareTest(NULL==cmd->GetMemoryCallback());
        compareTest(NULL==cmd->GetMemoryCallbackData());
        compareTest(4096==cmd->GetCommandMemorySize());
        compareTest(4096==cmd->GetControlMemorySize());
        compareTest(4096==cmd->GetControlMemoryFree());

        mpool->Free();
        cmd->Free();
    }

    // triangle #1 (textured)
    static const Vertex tri1[] = {
        { dt::vec3(-0.8, -0.8, 0.0), dt::vec4(0.0, 0.0, 0.0, 0.0) },
        { dt::vec3(+0.8, -0.8, 0.0), dt::vec4(0.0, 8.0, 0.0, 0.0) },
        { dt::vec3(+0.8, +0.8, 0.0), dt::vec4(8.0, 8.0, 0.0, 0.0) }
    };
    // triangle #2 (lwlled, white)
    static const Vertex tri2[] = {
        { dt::vec3(-0.8, +0.8, 0.0), dt::vec4(1.0, 1.0, 1.0, 1.0) },
        { dt::vec3(+0.8, +0.8, 0.0), dt::vec4(1.0, 1.0, 1.0, 1.0) },
        { dt::vec3(-0.8, -0.8, 0.0), dt::vec4(1.0, 1.0, 1.0, 1.0) }
    };
    // triangle #3 (line, blue)
    static const Vertex tri3[] = {
        { dt::vec3(+0.0, +1.0, -1.0), dt::vec4(0.0, 0.0, 1.0, 1.0) },
        { dt::vec3(-1.0, +0.0, -1.0), dt::vec4(0.0, 0.0, 1.0, 1.0) },
        { dt::vec3(+1.0, -1.0, +1.0), dt::vec4(0.0, 0.0, 1.0, 1.0) }
    };
    // triangle #4 (transparent, rgb)
    static const Vertex tri4[] = {
        { dt::vec3(-0.7, -0.4, +0.3), dt::vec4(1.0, 0.0, 0.0, 0.8) },
        { dt::vec3(+0.4, -0.4, -0.3), dt::vec4(0.0, 0.0, 1.0, 0.8) },
        { dt::vec3(+0.4, +0.7, +0.7), dt::vec4(0.0, 1.0, 0.0, 0.8) }
    };

    // texture
    MemoryPool *cpuBufferMemPool = NULL;
    MemoryPool* gpuTexMemPool = NULL;
    Texture* tex = NULL;
    Buffer *textureBuffer = NULL;
    Sampler* smp = NULL;

    size_t poolSize = PoolStorageSize(1);
    cpuBufferMemPool = device->CreateMemoryPool(NULL, poolSize, MemoryPoolType::CPU_COHERENT);
    gpuTexMemPool = device->CreateMemoryPool(NULL, poolSize, MemoryPoolType::GPU_ONLY);
    {
        compareTest(poolSize==cpuBufferMemPool->GetSize());
        compareTest((MemoryPoolFlags::CPU_UNCACHED|MemoryPoolFlags::GPU_CACHED|MemoryPoolFlags::COMPRESSIBLE)==cpuBufferMemPool->GetFlags());

        MemoryPoolBuilder mpb;
        mpb.SetDevice(device).SetDefaults();
        compareTest(mpb.GetDevice() == device);
        compareTest((MemoryPoolFlags::CPU_NO_ACCESS|MemoryPoolFlags::GPU_CACHED)==mpb.GetFlags());
        compareTest(0==mpb.GetSize());
        compareTest(NULL==mpb.GetMemory());
    }

    TextureBuilder tb;
    tb.SetDevice(device)
            .SetDefaults()
            .SetFormat(Format::RGBA8)
            .SetSize2D(2,2)
            .SetTarget(TextureTarget::TARGET_2D);
    {
        TextureSwizzle r,g,b,a;

        compareTest(tb.GetDevice() == device);
        compareTest(0==tb.GetFlags());
        compareTest(TextureTarget::TARGET_2D==tb.GetTarget());
        compareTest(2==tb.GetWidth() && 2==tb.GetHeight() && 1==tb.GetDepth());
        compareTest(1==tb.GetLevels());
        compareTest(Format::RGBA8==tb.GetFormat());
        compareTest(0==tb.GetSamples());
        tb.GetSwizzle(&r,&g,&b,&a);
        compareTest(TextureSwizzle::R==r && TextureSwizzle::G==g && TextureSwizzle::B==b && TextureSwizzle::A==a);
        compareTest(TextureDepthStencilMode::DEPTH==tb.GetDepthStencilMode());
        compareTest(NULL==tb.GetPackagedTextureData());
        compareTest(0==tb.GetStride());

        PackagedTextureLayout ptl, ptl_result;
        memset(&ptl, 0, sizeof(PackagedTextureLayout));
        memset(&ptl_result, 0, sizeof(PackagedTextureLayout));
        ptl.layout[0] = 0x13;   // log2GobsPerBlockY = 3, log2GobsPerBlockZ = 1

        if (g_lwnDeviceCaps.supportsMaxwellSparsePackagedTextures) {
            ptl.layout[1] = 0x05;   // log2GobsPerTileX = 5
        } else {
            ptl.layout[1] = 0x01;   // log2GobsPerTileX = 1
        }

        ptl.layout[4] = 0x08;   // Texture packager version
        tb.SetPackagedTextureLayout(&ptl);
        tb.GetPackagedTextureLayout(&ptl_result);
        compareTest(*(reinterpret_cast<uint32_t*>(&ptl.layout)) == *(reinterpret_cast<uint32_t*>(&ptl_result.layout)));
        // Reset PackagedTextureLayout to default
        memset(&ptl, 0, sizeof(PackagedTextureLayout));
        ptl.layout[4] = 0x08;  // Set version to avoid debug layer warnings
        tb.SetPackagedTextureLayout(&ptl);

        TextureView tv;
        int bl, nl;
        Format fmt;
        TextureTarget tt;
        TextureDepthStencilMode tdsm;
        tv.SetDefaults().SetSwizzle(TextureSwizzle::R,TextureSwizzle::G,TextureSwizzle::B,TextureSwizzle::A);
        tv.GetLevels(&bl,&nl);
        compareTest(0==bl && 0==nl);
        tv.GetLayers(&bl,&nl);
        compareTest(0==bl && 0==nl);
        tv.GetFormat(&fmt);
        compareTest(Format::NONE==fmt);
        tv.GetSwizzle(&r,&g,&b,&a);
        compareTest(TextureSwizzle::R==r && TextureSwizzle::G==g && TextureSwizzle::B==b && TextureSwizzle::A==a);
        tv.GetTarget(&tt);
        compareTest(TextureTarget::TARGET_1D==tt);
        tv.GetDepthStencilMode(&tdsm);
        compareTest(TextureDepthStencilMode::DEPTH==tdsm);
    }
    tex = tb.CreateTextureFromPool(gpuTexMemPool, 0);
    {
        compareTest(0==tex->GetFlags());
        compareTest(TextureTarget::TARGET_2D==tex->GetTarget());
        compareTest(2==tex->GetWidth() && 2==tex->GetHeight() && 1==tex->GetDepth());
        compareTest(1==tex->GetLevels());
        compareTest(Format::RGBA8==tex->GetFormat());
        compareTest(0==tex->GetSamples());
        TextureSwizzle r,g,b,a;
        tex->GetSwizzle(&r,&g,&b,&a);
        compareTest(TextureSwizzle::R==r && TextureSwizzle::G==g && TextureSwizzle::B==b && TextureSwizzle::A==a);
        compareTest(TextureDepthStencilMode::DEPTH==tex->GetDepthStencilMode());
        compareTest(0==tex->GetStride());
        // Texture::GetStorageSize not tested
        // see lwn_texture_storagesize (https://git-p4swmirror.lwpu.com/r/#/c/17691/)
    }
    size_t bufSz = tb.GetStorageSize(); //2*2*1024;

    // buffer (fill with red, green, blue, white)
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    textureBuffer = bb.CreateBufferFromPool(cpuBufferMemPool, 0, bufSz);
    dt::u8lwec4* ptr = static_cast<dt::u8lwec4*>(textureBuffer->Map());
    ptr[0] = dt::u8lwec4(1.0,0.0,0.0,1.0);
    ptr[1] = dt::u8lwec4(0.0,1.0,0.0,1.0);
    ptr[2] = dt::u8lwec4(0.4,0.4,1.0,1.0);
    ptr[3] = dt::u8lwec4(1.0,1.0,1.0,1.0);

    // fill texture with data
    CopyRegion cr = {0,0,0,2,2,1};
    queueCB.CopyBufferToTexture(textureBuffer->GetAddress(), tex, 0, &cr, CopyFlags::NONE);
    queueCB.submit();
    queue->Finish();

    // sampler
    SamplerBuilder sb;
    sb.SetDevice(device)
            .SetDefaults()
            .SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST)
            .SetWrapMode(WrapMode::REPEAT, WrapMode::REPEAT, WrapMode::REPEAT);
    smp = sb.CreateSampler();
    LWNuint samplerID = smp->GetRegisteredID();

    // bind texture
    LWNuint texID = g_lwnTexIDPool->Register(tex, 0);
    TextureHandle handle = device->GetTextureHandle(texID, samplerID);
    {
        int spsize;
        device->GetInteger(lwn::DeviceInfo::RESERVED_SAMPLER_DESCRIPTORS, &spsize);
        spsize += g_lwnTexIDPool->NUM_PUBLIC_SAMPLERS;
        compareTest(spsize==g_lwnTexIDPool->GetSamplerPool()->GetSize());

        int tpsize;
        device->GetInteger(lwn::DeviceInfo::RESERVED_TEXTURE_DESCRIPTORS, &tpsize);
        tpsize += g_lwnTexIDPool->NUM_PUBLIC_TEXTURES;
        compareTest(tpsize==g_lwnTexIDPool->GetTexturePool()->GetSize());
    }

    // states
    MultisampleState ms_state;
    PolygonState pl_state;
    DepthStencilState ds_state;
    ChannelMaskState cm_state;
    BlendState bl_state;
    ColorState cl_state;

    // setup multisampled rendertarget/fbo
    int mssc = 4;   // multisample samples count
    int ww = lwrrentWindowWidth / 8;
    int wh = lwrrentWindowHeight / 8;
    Framebuffer fbResult(ww,wh);
    fbResult.setFlags(TextureFlags::COMPRESSIBLE);
    fbResult.setColorFormat(0, Format::RGBA8);
    fbResult.setDepthSamples(mssc);
    fbResult.setDepthStencilFormat(Format::DEPTH24);
    fbResult.setSamples(mssc);
    fbResult.alloc(device);
    fbResult.bind(queueCB);
    queueCB.SetViewportScissor(0, 0, ww, wh);
    ms_state.SetDefaults()
            .SetSamples(mssc);
    queueCB.BindMultisampleState(&ms_state);

    // set depth test
    ds_state.SetDefaults()
            .SetDepthTestEnable(true)
            .SetDepthFunc(DepthFunc::LEQUAL);
    queueCB.BindDepthStencilState(&ds_state);

    // set face lwll
    pl_state.SetDefaults()
            .SetLwllFace(Face::BACK)
            .SetFrontFace(FrontFace::CCW);
    queueCB.BindPolygonState(&pl_state);

    // clear buffers
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);
    queueCB.ClearDepthStencil(1.0, true, 0, 0);

    // triangle #1 (textured triangle, color masked)
    cm_state.SetDefaults().SetChannelMask(0, true, true, false, true);
    queueCB.BindChannelMaskState(&cm_state);
    drawTriangle(tri1, handle);
    {
        LWNboolean r,g,b,a;
        cm_state.GetChannelMask(0, &r, &g, &b, &a);
        compareTest(r && g && !b && a);

        compareTest(!!ds_state.GetDepthTestEnable());
        compareTest(!!ds_state.GetDepthWriteEnable());
        compareTest(DepthFunc::LEQUAL==ds_state.GetDepthFunc());
        compareTest(!ds_state.GetStencilTestEnable());
        compareTest(StencilFunc::ALWAYS==ds_state.GetStencilFunc(Face::FRONT) && StencilFunc::ALWAYS==ds_state.GetStencilFunc(Face::BACK));
        StencilOp sop[3];
        ds_state.GetStencilOp(Face::FRONT,sop,sop+1,sop+2);
        compareTest(StencilOp::KEEP==sop[0] && StencilOp::KEEP==sop[1] && StencilOp::KEEP==sop[2]);
        ds_state.GetStencilOp(Face::BACK,sop,sop+1,sop+2);
        compareTest(StencilOp::KEEP==sop[0] && StencilOp::KEEP==sop[1] && StencilOp::KEEP==sop[2]);

        compareTest(!!ms_state.GetMultisampleEnable());
        compareTest(!ms_state.GetAlphaToCoverageEnable());
        compareTest(!!ms_state.GetAlphaToCoverageDither());
        compareTest(!ms_state.GetCoverageToColorEnable());
        compareTest(0==ms_state.GetCoverageToColorOutput());
        compareTest(!ms_state.GetSampleLocationsEnable());
        compareTest(!ms_state.GetSampleLocationsGridEnable());
    }
    cm_state.SetChannelMask(0, true, true, true, true);
    queueCB.BindChannelMaskState(&cm_state);

    // triangle #2 (white, lwlled)
    drawTriangle(tri2);

    // triangle #3 (red lines)
    pl_state.SetPolygonMode(PolygonMode::LINE);
    queueCB.BindPolygonState(&pl_state);
    drawTriangle(tri3);
    {
        compareTest(PolygonMode::LINE==pl_state.GetPolygonMode());
        compareTest(Face::BACK==pl_state.GetLwllFace());
        compareTest(FrontFace::CCW==pl_state.GetFrontFace());
    }
    pl_state.SetPolygonMode(PolygonMode::FILL);
    queueCB.BindPolygonState(&pl_state);

    // triangle #4 (transparent rgb)
    ds_state.SetDepthTestEnable(false);
    queueCB.BindDepthStencilState(&ds_state);
    bl_state.SetDefaults()
            .SetBlendFunc(BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA, BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA);
    queueCB.BindBlendState(&bl_state);
    cl_state.SetDefaults().SetBlendEnable(0,true);
    queueCB.BindColorState(&cl_state);
    drawTriangle(tri4);
    {
        compareTest(0==bl_state.GetBlendTarget());
        BlendFunc blfnsrgb, blfndrgb, blfnsa, blfnda;
        bl_state.GetBlendFunc(&blfnsrgb, &blfndrgb, &blfnsa, &blfnda);
        compareTest(BlendFunc::SRC_ALPHA==blfnsrgb && BlendFunc::ONE_MINUS_SRC_ALPHA==blfndrgb);
        compareTest(BlendFunc::SRC_ALPHA==blfnsa && BlendFunc::ONE_MINUS_SRC_ALPHA==blfnda);
        BlendEquation eqrgb, eqa;
        bl_state.GetBlendEquation(&eqrgb,&eqa);
        compareTest(BlendEquation::ADD==eqrgb && BlendEquation::ADD==eqa);
        compareTest(BlendAdvancedMode::BLEND_NONE==bl_state.GetAdvancedMode());
        compareTest(BlendAdvancedOverlap::UNCORRELATED==bl_state.GetAdvancedOverlap());
        compareTest(!!bl_state.GetAdvancedPremultipliedSrc());
        compareTest(!!bl_state.GetAdvancedNormalizedDst());

        compareTest(!!cl_state.GetBlendEnable(0));
        compareTest(LogicOp::COPY==cl_state.GetLogicOp());
        compareTest(AlphaFunc::ALWAYS==cl_state.GetAlphaTest());
    }

    // resolve multisampled rendertarget/fbo and display it
    fbResult.downsample(queueCB);
    LWNuint texMS_ID = g_lwnTexIDPool->Register(fbResult.getColorTexture(0), 0);
    g_lwnWindowFramebuffer.bind();
    queueCB.SetViewportScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    queueCB.ClearColor(0, 1.0, 0.0, 1.0, 1.0);
    ms_state.SetSamples(0);
    queueCB.BindMultisampleState(&ms_state);
    drawTexturedFSQuad(texMS_ID);
    fbResult.destroy();

    // Test Get-er of QueueBuilder
    {
        size_t  memSize = 32 * LWN_MEMORY_POOL_STORAGE_GRANULARITY;
        void*   queueMem = lwnUtil::AlignedStorageAlloc(memSize, LWN_MEMORY_POOL_STORAGE_GRANULARITY);

        int cmdMemSize  = 0;
        int compMemSize = 0;
        int ctrlMemSize = 0;
        int flushThreshold = 0;

        device->GetInteger(DeviceInfo::QUEUE_COMMAND_MEMORY_MIN_SIZE, &cmdMemSize);
        device->GetInteger(DeviceInfo::QUEUE_COMPUTE_MEMORY_MIN_SIZE, &compMemSize);
        device->GetInteger(DeviceInfo::QUEUE_COMPUTE_MEMORY_MIN_SIZE, &ctrlMemSize);
        device->GetInteger(DeviceInfo::QUEUE_COMMAND_MEMORY_MIN_FLUSH_THRESHOLD, &flushThreshold);

        QueueBuilder qb;

        qb.SetDefaults().SetDevice(device);

        int size = 0;
        compareTest(!qb.GetCommandMemorySize(&size) &&
                    !qb.GetComputeMemorySize(&size) &&
                    !qb.GetControlMemorySize(&size));

        qb.SetQueueMemory(queueMem, memSize)
          .SetCommandMemorySize(cmdMemSize)
          .SetComputeMemorySize(compMemSize)
          .SetControlMemorySize(ctrlMemSize)
          .SetQueuePriority(QueuePriority::LOW)
          .SetCommandFlushThreshold(flushThreshold);

        int returnedFlushThreshold = -1;
        compareTest(qb.GetDevice() == device);
        compareTest(qb.GetFlags() == LWNqueueFlags::LWN_QUEUE_FLAGS_NONE);
        compareTest(qb.GetCommandMemorySize(&size) && (size == cmdMemSize));
        compareTest(qb.GetComputeMemorySize(&size) && (size == compMemSize));
        compareTest(qb.GetControlMemorySize(&size) && (size == ctrlMemSize));
        compareTest(qb.GetQueuePriority() == QueuePriority::LOW);
        compareTest(qb.GetCommandFlushThreshold(&returnedFlushThreshold) && (returnedFlushThreshold == flushThreshold));
        compareTest(qb.GetMemorySize() == int(memSize));
        compareTest(qb.GetMemory() == queueMem);

        lwnUtil::AlignedStorageFree(queueMem);
    }

    // Test Get-er of EventBuilder
    {
        MemoryPool* memPool;
        memPool = device->CreateMemoryPool(0, 1024, MemoryPoolType::GPU_ONLY);

        const int64_t offset = 128;

        EventBuilder eb;
        eb.SetDefaults()
          .SetStorage(memPool, offset);

        int64_t returnedOffset = 0;
        compareTest(eb.GetStorage(&returnedOffset) == memPool);
        compareTest(returnedOffset == offset);

        memPool->Free();
    }

    // Test Get-er of DeviceBuilder
    {
        DeviceBuilder db;
        db.SetDefaults();
        db.SetFlags(DeviceFlagBits::DEBUG_ENABLE_LEVEL_4);

        compareTest(db.GetFlags() == DeviceFlagBits::DEBUG_ENABLE_LEVEL_4);
    }

    // clean and display results
    textureBuffer->Free();
    smp->Free();
    tex->Free();
    cpuBufferMemPool->Free();
    gpuTexMemPool->Free();
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);

    int padding = 4;
    int sqdim = 16;
    int perRow = lwrrentWindowWidth / (padding+sqdim);

    for (int c=0; c<testCount; c++)
    {
        int i = c / perRow;
        int j = c % perRow;
        bool val = testResults[c];
        int x = padding + j*(sqdim+padding);
        int y = padding + i*(sqdim+padding);
        queueCB.SetViewportScissor(x, y, sqdim, sqdim);
        queueCB.ClearColor(0, !val?1.0:0.0, val?1.0:0.0, 0.0);
        queueCB.submit();
        queue->Finish();
    }
}

#undef TESTRES_MAX_CNT

OGTEST_CppTest(LWNGettersTest, lwn_getters, );
