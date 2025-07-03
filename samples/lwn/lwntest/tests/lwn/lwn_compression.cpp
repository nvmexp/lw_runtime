#include "lwntest_cpp.h"
#include "lwn_utils.h"

#include <time.h>

//////////////////////////////////////////////////////////////////////////

using namespace lwn;

class LWNCompression
{
public:
    enum LWNcompressiolwariant {
        CLEAR,
        GRADIENT
    };

    enum LWNcompressionState {
        COMPRESSED,
        NON_COMPRESSED
    };
    
    LWNCompression(LWNcompressiolwariant variant) 
        : _variant(variant)
    {}

    LWNTEST_CppMethods();
private:
    LWNCompression();

    LWNcompressiolwariant _variant;
};

int LWNCompression::isSupported() const
{
#if defined(LW_TEGRA)
    return lwogCheckLWNAPIVersion(51, 1);
#else
    return 0;
#endif
}

lwString LWNCompression::getDescription() const
{
    return "Test memory pool with compression enabled and disabled. "
           "Create GPU only pools with and without compression and clear/render to a texture. Copy "
           "the whole pool and create textures from the copy and display them. "
           "The render target is initialized with purple color(255, 0, 255), and "
           "thus, purple signifies unwritten framebuffer memory in the bottom-left "
           "quadrant.";
}

void LWNCompression::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    LWNfloat grayish[] = { 0.2, 0.2, 0.2, 1.0 };
    queueCB.ClearColor(0, grayish, LWN_CLEAR_COLOR_MASK_RGBA);

    VertexShader vs(440);
    vs <<
        "layout(location = 0) in vec4 position;\n"
        "layout(location = 1) in vec2 tc;\n"
        "out vec2 texcoord;\n"
        "void main() {\n"
        "  gl_Position = position;\n"
        "  texcoord = vec2(1.0, 1.0) + vec2(0.5, 0.5)*vec2(position);\n"
        "}\n";

    FragmentShader fs_gradient(440);
    fs_gradient << "precision highp float;\n"
        "layout(location = 0) out vec4 color;\n"
        "in vec2 texcoord;\n"
        "void main() {\n"
        "  color = vec4(1.0-texcoord.x, 0.2 + 0.8 * (1.0-texcoord.x), 1.0-texcoord.x, 1.0);\n"
        "}\n";

    FragmentShader fs_texture(440);
    fs_texture << "precision highp float;\n"
        "layout(location = 0) out vec4 color;\n"
        "layout (binding=0) uniform sampler2D tex;\n"
        "in vec2 texcoord;\n"
        "void main() {\n"
        "  color = vec4(texture(tex, texcoord));\n"
        "}\n";

    // shader program
    Program *pgm = device->CreateProgram();
    LWNboolean compiled = g_glslcHelper->CompileAndSetShaders(pgm, vs, fs_gradient);
    if (!compiled) {
        printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
        return;
    }

    Program *pgmTexture = device->CreateProgram();
    compiled = g_glslcHelper->CompileAndSetShaders(pgmTexture, vs, fs_texture);
    if (!compiled) {
        printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec4 position;
    };
    static const Vertex vertexData[] = {
        { dt::vec4(-1.0, -1.0, 0.0, 1.0) },
        { dt::vec4(-1.0, +1.0, 0.0, 1.0) },
        { dt::vec4(+1.0, -1.0, 0.0, 1.0) },
        { dt::vec4(+1.0, +1.0, 0.0, 1.0) },
    };

    MemoryPoolAllocator vertex_allocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexStreamSet streamSet(stream);
    VertexArrayState vertex = streamSet.CreateVertexArrayState();

    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, vertex_allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults().SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    Sampler *sampler = sb.CreateSampler();

    const int texWidth = 32, texHeight = 32;

    BufferBuilder bufferBuilder;
    bufferBuilder.SetDevice(device).SetDefaults();

    // setup basic render target texture builder
    TextureBuilder textureBuilder;
    textureBuilder.SetDevice(device).SetDefaults().
        SetTarget(TextureTarget::TARGET_2D).
        SetSize2D(texWidth, texHeight).
        SetFormat(Format::RGBA8);
    textureBuilder.SetFlags(TextureFlags::COMPRESSIBLE);
    const LWNuintptr poolSizeCompressed = textureBuilder.GetStorageSize();

    textureBuilder.SetFlags(TextureFlags(0));
    const LWNuintptr poolSizeNonCompressed = textureBuilder.GetStorageSize();

    const LWNuintptr poolSize = (poolSizeCompressed > poolSizeNonCompressed) ? poolSizeCompressed : poolSizeNonCompressed;

    MemoryPoolAllocator coherent_allocator(device, NULL, 0, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    LWNuint data[texWidth][texHeight];

    // pink texture image
    for (int y = 0; y < texHeight; y++) {
        for (int x = 0; x < texWidth; x++) {
            data[x][y] = 0xFFFF00FF;
        }
    }

    Buffer *initbo = AllocAndFillBuffer(device, queue, queueCB, coherent_allocator, data,
                                        sizeof(LWNuint)*texHeight*texWidth, BUFFER_ALIGN_COPY_READ_BIT, false);

    const LWNmemoryPoolFlags flagsNonCompressible = LWNmemoryPoolFlags(LWN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS_BIT | LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT);
    const LWNmemoryPoolFlags flagsCompressible = LWN_MEMORY_POOL_TYPE_GPU_ONLY;
    for (int state = 0; state < 2; state++) {
        const bool bCompressed = (state == 0);
        MemoryPoolAllocator compressible_allocator(device, NULL, poolSize, (bCompressed ? flagsCompressible : flagsNonCompressible));

        Texture *tex;

        textureBuilder.SetDefaults().
            SetTarget(TextureTarget::TARGET_2D).
            SetSize2D(texWidth, texHeight).
            SetFormat(Format::RGBA8);
        textureBuilder.SetFlags(bCompressed ? TextureFlags::COMPRESSIBLE : TextureFlags(0));

        // allocate a texture either with compression or not
        tex = compressible_allocator.allocTexture(&textureBuilder);

        // init texture with pink, this color will shine through in places
        // that do not get written in the texture during render step
        CopyRegion copyRegion = { 0, 0, 0, texWidth, texHeight, 1 };
        queueCB.CopyBufferToTexture(initbo->GetAddress(), tex, NULL, &copyRegion, CopyFlags::NONE);

        // render to full texture
        queueCB.SetViewportScissor(0, 0, texWidth, texHeight);

        // bind compressible texture as render target
        queueCB.SetRenderTargets(1, &tex, NULL, NULL, NULL);

        // render something to the texture
        // this will leave compressed data in the texture that
        // we will display 
        LWNfloat fcolor_clear[] = { 0.0, 0.0, 0.0, 1.0 };
        switch (_variant) {
        case CLEAR:
            queueCB.ClearColor(0, fcolor_clear, LWN_CLEAR_COLOR_MASK_RGBA);
            break;
        case GRADIENT:
            queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
            break;
        }

        queueCB.submit();
        queue->Finish();

        bufferBuilder.SetDefaults();
        Buffer *source_bo = bufferBuilder.CreateBufferFromPool(compressible_allocator.pool(tex), 0, poolSize);

        bufferBuilder.SetDefaults();

        // make a uncompressible copy of the original pool and....
        MemoryPoolAllocator gpu_allocator(device, NULL, poolSize, LWNmemoryPoolFlags(LWN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS_BIT | LWN_MEMORY_POOL_FLAGS_GPU_UNCACHED_BIT));
        Buffer *readback_bo = gpu_allocator.allocBuffer(&bufferBuilder, BufferAlignBits(BUFFER_ALIGN_COPY_WRITE_BIT | BUFFER_ALIGN_COPY_READ_BIT), poolSize);

        queueCB.CopyBufferToBuffer(source_bo->GetAddress(), readback_bo->GetAddress(), poolSize, CopyFlags::NONE);

        queueCB.submit();
        queue->Finish();

        textureBuilder.SetDefaults().
            SetTarget(TextureTarget::TARGET_2D).
            SetSize2D(texWidth, texHeight).
            SetFormat(Format::RGBA8).
            SetFlags(TextureFlags(0));

        // ... create a texture from the pool copy, interpret the raw data already in it as texture
        LWNuint offs = compressible_allocator.offset(tex);
        Texture *texDisplay = textureBuilder.CreateTextureFromPool(gpu_allocator.pool(readback_bo), offs);
        LWNuint texDisplayHandle = device->GetTextureHandle(texDisplay->GetRegisteredTextureID(), sampler->GetRegisteredID());

        // rebind default framebuffer
        g_lwnWindowFramebuffer.bind();

        queueCB.BindProgram(pgmTexture, ShaderStageBits::ALL_GRAPHICS_BITS);

        queueCB.SetViewportScissor(state * lwrrentWindowWidth / 2, state * lwrrentWindowHeight / 2, lwrrentWindowWidth / 2, lwrrentWindowHeight / 2);
        queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texDisplayHandle);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

        queueCB.submit();
        queue->Finish();

        compressible_allocator.freeTexture(tex);
        source_bo->Free();
        gpu_allocator.freeBuffer(readback_bo);
        texDisplay->Free();
    }
}

OGTEST_CppTest(LWNCompression, lwn_compression_zbc, (LWNCompression::CLEAR));
OGTEST_CppTest(LWNCompression, lwn_compression_gradient, (LWNCompression::GRADIENT));
