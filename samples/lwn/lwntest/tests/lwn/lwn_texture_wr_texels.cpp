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

#define DEBUG_MODE 0
#if DEBUG_MODE
    #define DEBUG_PRINT(x) do { \
        printf x; \
        fflush(stdout); \
    } while (0)
#else
    #define DEBUG_PRINT(x)
#endif

#define MEMPOOL_SZ 0x1600000
#define MAX_SUBTESTS 32

using namespace lwn;
using namespace lwn::dt;

#ifndef ROUND_UP
#define ROUND_UP(N, S) ((((N) + (S) - 1) / (S)) * (S))
#endif

static int GetFormatBPP(Format fmt)
{
    switch (fmt) {
        case Format::RGBA8: return 4;
        case Format::RGBX8: return 4;
        case Format::RGBA16: return 8;
        case Format::RGB10A2: return 4;
        case Format::RGBA32F: return 16;
        default:
            assert(!"Unsupported format.");
            return 0;
    }
}

const static int BLOCK_DIM = 4;
const static int BLOCK_BYTES = 8;

static int TexelsToBlocks(int numTexels) {
    return (numTexels + BLOCK_DIM - 1) / BLOCK_DIM;
}

const static int smileyRows = 23;
const static char* smiley[] = {
    "_____________________________________________________________",
    "_..........................................................._",
    "_.....................################......................_",
    "_..................######################..................._",
    "_................##########################................._",
    "_.............################################.............._",
    "_...........####################################............_",
    "_..........##########___############___##########..........._",
    "_.........##########_____##########_____##########.........._",
    "_........##########!!!___#########!!!___###########........._",
    "_........##########!!!___#########!!!___###########........._",
    "_........###########_____##########_____###########........._",
    "_........############___############___############........._",
    "_........##########################################........._",
    "_........##########################################........._",
    "_........##########################################........._",
    "_........###########____________________###########........._",
    "_.........############________________############.........._",
    "_..........###############_________##############..........._",
    "_.............###############__##############..............._",
    "_.................########################.................._",
    "_.......................############........................_",
    "_____________________________________________________________",
};

// ------------------------------- LWNTextureReadWriteTexels --------------------------------------

enum TestFillMode {
    FILLMODE_NONE,
    FILLMODE_CHECKERBOARD,
    FILLMODE_SMILEY,
    FILLMODE_DXT1_GREEN,
    FILLMODE_DXT1_BLUE,
};

enum TestFlags {
    TFLAG_NONE = 0x0,
    TFLAG_MIPMAP_ENABLED = 0x1,
    TFLAG_LWBEMAP_MULTIFACE = 0x2,
    TFLAG_STRIDE = 0x4,
};

// Used for the Strided variants of the tests.
enum StrideType {
    // Dense packing
    S_0,       // Dense pack; pass zero as stride
    S_MIN,     // Dense pack; pass actual stride value

    // Aligned strides (for fast-path copies)
    S_16,      // 16-byte granularity (sub-segment width)
    S_64,      // 64-byte granularity (GOB width)
    S_512,     // 512-byte granularity (GOB size)
    S_8192,    // 8192-byte granularity (16 GOBs per block)

    // Adverse
    S_UNALIGN, // Specifically unaligned copies
    S_DOUBLE,  // Twice as large as dense pack
};

static int CalcStride(StrideType type, int minStride)
{
    switch (type) {
    case S_0:
    case S_MIN:
        return minStride;
    case S_16:
        return ROUND_UP(minStride, 16);
    case S_64:
        return ROUND_UP(minStride, 64);
    case S_512:
        return ROUND_UP(minStride, 512);
    case S_8192:
        return ROUND_UP(minStride, 8192);
    case S_UNALIGN:
        return ROUND_UP(minStride, 16) + 1;
    case S_DOUBLE:
        return minStride * 2;
    }

    assert(!"Unrecognized stride type");
    return 0;
}

class LWNTextureReadWriteTexels {
    // Rest of the stuff we need to draw the texture.
    Program *m_program2D;
    Program *m_program2DArray;
    Program *m_programLwbemap;
    Program *m_programLwbemapArray;
    Program *m_program3D;
    VertexArrayState m_vertexState;
    Buffer *m_vbo;
    MemoryPoolAllocator* m_pool;
    MemoryPoolAllocator* m_bufpool;
    Sampler *m_sampler;
    LWNuint m_vertexDataSize;

    struct Vertex {
        vec3 position;
        vec3 uv;
    };

    struct UBOBlock
    {
        // This will be a vec4,
        // testInfo[0] is the layer of the texture will be tested,
        // testInfo[1] is the level of the texture will be tested.
        float testInfo[4];
    };
    MemoryPool* m_uboMem;
    size_t m_uboSize;
    BufferAddress m_uboAddress;
    UBOBlock* m_uboPtr;

    bool m_mipmap;
    bool m_lwbemapMultiface;
    bool m_stride;

public:
    struct TexelBuffer {
        unsigned char* ptr;
        int size;
        int rowStride;
        int imageStride;
        TexelBuffer() : ptr(NULL), size(0) {}
        ~TexelBuffer() {
            free(ptr);
            ptr = NULL;
        }
    };

    explicit LWNTextureReadWriteTexels(int flags);
    ~LWNTextureReadWriteTexels();

    bool init(void);
    bool createAndFillTexture(TextureTarget target, int w, int h, int d, Format fmt,
                              Texture *&texture, TextureView *textureView, TestFillMode fillMode,
                              bool writeBack, int testLayer, int testLevel,
                              StrideType rowStrideType, StrideType imageStrideType);
    void fillTexelData(TextureTarget target, int w, int h, int d, Format fmt,
                       TestFillMode fillMode, TexelBuffer& buffer,
                       StrideType rowStrideType, StrideType imageStrideType);
    void draw(TextureTarget target, Texture *texture, int testLayer, int testLevel);
};

LWNTextureReadWriteTexels::LWNTextureReadWriteTexels(int flags)
    : m_pool(NULL), m_bufpool(NULL), m_mipmap(!!(flags & TFLAG_MIPMAP_ENABLED)),
      m_lwbemapMultiface(!!(flags & TFLAG_LWBEMAP_MULTIFACE)),
      m_stride(!!(flags & TFLAG_STRIDE))
{
}

LWNTextureReadWriteTexels::~LWNTextureReadWriteTexels() {
    m_program2D->Free();
    m_program2DArray->Free();
    m_programLwbemap->Free();
    m_programLwbemapArray->Free();
    m_program3D->Free();
    m_uboMem->Free();
    delete m_bufpool;
    delete m_pool;
}

bool LWNTextureReadWriteTexels::init() {
    DEBUG_PRINT(("LWNTextureReadWriteTexels:: Creating test assets...\n"));
    Device *device = DeviceState::GetActive()->getDevice();
    m_bufpool = new MemoryPoolAllocator(device, NULL, 0x10000, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // Compile shaders.
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 uv;\n"
        "out gl_PerVertex {\n"
        "    vec4 gl_Position;\n"
        "};\n"
        "out vec2 ouv;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ouv = uv.xy;\n"
        "}\n";

    const char *commonFS =
        "in vec2 ouv;\n"
        "out vec4 fcolor;\n"
        "layout(binding=0, std140) uniform Block {\n"
        "  vec4 testInfo;\n"
        "};\n";
    FragmentShader fs2D(440);
    fs2D <<
        commonFS <<
        "layout (binding=0) uniform sampler2D tex;\n"
        "void main() {\n"
        "  fcolor = textureLod(tex, ouv, testInfo.y);\n"
        "}\n";
    FragmentShader fs2DArray(440);
    fs2DArray <<
        commonFS <<
        "layout (binding=0) uniform sampler2DArray tex;\n"
        "void main() {\n"
        "  fcolor = textureLod(tex, vec3(ouv,testInfo.x), testInfo.y);\n"
        "}\n";
    FragmentShader fsLwbemap(440);
    fsLwbemap <<
        commonFS <<
        "layout (binding=0) uniform samplerLwbe tex;\n"
        "void main() {\n"
        "  fcolor = textureLod(tex, vec3(ouv * 2.0 - 1.0, 0.35), testInfo.y);\n"
        "}\n";
    FragmentShader fsLwbemapArray(440);
    fsLwbemapArray <<
        commonFS <<
        "layout (binding=0) uniform samplerLwbeArray tex;\n"
        "void main() {\n"
        "  fcolor = textureLod(tex, vec4(ouv * 2.0 - 1.0, 0.35, testInfo.x), testInfo.y);\n"
        "}\n";
    FragmentShader fs3D(440);
    fs3D <<
        commonFS <<
        "layout (binding=0) uniform sampler3D tex;\n"
        "void main() {\n"
        "  fcolor = textureLod(tex, vec3(ouv, ouv.x), testInfo.y);\n"
        "}\n";

    // 2d and 2d_mipmap
    m_program2D = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_program2D, vs, fs2D)) {
        return false;
    }
    // 3d and 3d_mipmap
    m_program3D = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_program3D, vs, fs3D)) {
        return false;
    }
    // 2d_array and 2d_array_mipmap
    m_program2DArray = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_program2DArray, vs, fs2DArray)) {
        return false;
    }
    // lwbemap and lwbemap_mipmap
    m_programLwbemap = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_programLwbemap, vs, fsLwbemap)) {
        return false;
    }
    // lwbemap_array and lwbemap_array_mipmap
    m_programLwbemapArray = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_programLwbemapArray, vs, fsLwbemapArray)) {
        return false;
    }

    // Create vertex data.
    const int vertexCount = 4;
    static const Vertex vertexData[] = {
        { vec3(-1, -1, 0.0), vec3(0.0, 1.0, 0.0) },
        { vec3(+1, -1, 0.0), vec3(1.0, 1.0, 0.0) },
        { vec3(+1, +1, 0.0), vec3(1.0, 0.0, 0.0) },
        { vec3(-1, +1, 0.0), vec3(0.0, 0.0, 0.0) }
    };
    m_vertexDataSize = sizeof(vertexData);
    VertexStream vertexStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, uv);
    m_vertexState = vertexStream.CreateVertexArrayState();
    m_vbo = vertexStream.AllocateVertexBuffer(device, vertexCount, *m_bufpool, vertexData);

    SamplerBuilder sb;
    if (m_mipmap) {
        sb.SetDevice(device).SetDefaults()
            .SetMinMagFilter(MinFilter::LINEAR_MIPMAP_LINEAR, MagFilter::LINEAR);
    } else {
        sb.SetDevice(device).SetDefaults()
            .SetMinMagFilter(MinFilter::LINEAR, MagFilter::LINEAR);
    }
    m_sampler = sb.CreateSampler();
    m_pool = new MemoryPoolAllocator(device, NULL, MEMPOOL_SZ, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);

    LWNint uboAlignment;
    device->GetInteger(DeviceInfo::UNIFORM_BUFFER_ALIGNMENT, &uboAlignment);
    m_uboSize = ROUND_UP(sizeof(UBOBlock), uboAlignment);
    m_uboMem = device->CreateMemoryPool(NULL, m_uboSize, MemoryPoolType::CPU_COHERENT);
    m_uboAddress = m_uboMem->GetBufferAddress();
    m_uboPtr = static_cast<UBOBlock*>(m_uboMem->Map());

    return true;
}

template <typename T>
static void FillTexel(void* data, const T& src)
{
    memcpy(data, &src, sizeof(src));
}

static void FillTexelFmt(void* data, Format fmt, float r = 1.0f, float g = 0.5f, float b = 0.5f, float a = 1.0f)
{
    switch (fmt) {
    case Format::RGBA8:
    case Format::RGBX8:
        FillTexel(data, u8vec4(r * 255.0, g * 255.0, b * 255.0, a * 255.0));
        break;
    case Format::RGBA16:
        FillTexel(data, u16vec4(r * 65535.0, g * 65535.0, b * 65535.0, a * 65535.0));
        break;
    case Format::RGB10A2:
        FillTexel(data, vec4_rgb10a2(r, g, b, a));
        break;
    case Format::RGBA32F:
        FillTexel(data, vec4(r, g, b, a));
        break;
    default:
        assert(!"Unknown format.");
    }
}

void LWNTextureReadWriteTexels::fillTexelData(TextureTarget target, int w, int h, int d, Format fmt,
                                              TestFillMode fillMode, TexelBuffer& buffer,
                                              StrideType rowStrideType, StrideType imageStrideType)
{
    if (buffer.ptr) {
        free(buffer.ptr);
    }

    int minRowStride = (fmt == Format::RGB_DXT1) ? (TexelsToBlocks(w) * BLOCK_BYTES) : (w * GetFormatBPP(fmt));
    buffer.rowStride = CalcStride(rowStrideType, minRowStride);

    int minImageStride = buffer.rowStride * ((fmt == Format::RGB_DXT1) ? TexelsToBlocks(h) : h);
    buffer.imageStride = CalcStride(imageStrideType, minImageStride);

    DEBUG_PRINT(("LWNTextureReadWriteTexels:: Filling %d x %d x %d texture %s buffer, stride %d x %d ...\n", w, h, d,
                (fmt==Format::RGB_DXT1) ? "RGB_DXT1" : "uncompressed", buffer.rowStride, buffer.imageStride));

    buffer.size = buffer.imageStride * d;
    buffer.ptr = (unsigned char*) malloc(buffer.size);
    memset(buffer.ptr, 0xFF, buffer.size);
    assert(buffer.ptr);

    if (fmt == Format::RGB_DXT1) {
        int reinterpretWidth = TexelsToBlocks(w);
        int reinterpretHeight = TexelsToBlocks(h);

        uint64_t color = 0;
        ct_assert(sizeof(color) == BLOCK_BYTES);

        switch (fillMode) {
            case FILLMODE_DXT1_GREEN:
            {
                // Fills texture with green.
                // RGB_DXT1: lookup table, green, green
                color = 0x1111111107E007E0;
                break;
            }
            case FILLMODE_DXT1_BLUE:
            {
                // Fills texture with blue
                // RGB_DXT1: lookup table, blue, blue
                color = 0x11111111008F008F;
                break;
            }
            case FILLMODE_NONE:
            case FILLMODE_CHECKERBOARD:
            case FILLMODE_SMILEY:
                assert(!"Fill not not supported for this format.");
                return;
            // No default here so compiler complains if we missed a case.
        }

        for (int z = 0; z < d; z++) {
            for (int y = 0; y < reinterpretHeight; y++) {
                for (int x = 0; x < reinterpretWidth; x++) {
                    int offset = z * buffer.imageStride + y * buffer.rowStride + x * BLOCK_BYTES;
                    memcpy(buffer.ptr + offset, &color, BLOCK_BYTES);
                }
            }
        }
    } else {
        // Uncompressed format
        int bpp = GetFormatBPP(fmt);
        for (int z = 0; z < d; z++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int offset = z * buffer.imageStride + y * buffer.rowStride + x * bpp;
                    void* texel = buffer.ptr + offset;
                    switch (fillMode) {
                        case FILLMODE_CHECKERBOARD:
                        {
                            // Fill with a blue / yellow checkerboard.
                            bool checker = !!( ((x  / 12) % 2) ^ ((y  / 12) % 2) );
                            if (checker) {
                                FillTexelFmt(texel, fmt, 0.051f, 0.302f, 0.302f);
                            } else {
                                FillTexelFmt(texel, fmt, 0.667f, 0.424f, 0.224f);
                            }
                            break;
                        }
                        case FILLMODE_SMILEY:
                        {
                            double x_ = ((double) x / (double) (w - 1)) * strlen(smiley[0]);
                            double y_ = ((double) y / (double) (h - 1)) * smileyRows;
                            int idxX = (int) x_, idxY = (int)y_;
                            if (idxX > (int) strlen(smiley[0]) - 1) idxX = (int) strlen(smiley[0]) - 1;
                            if (idxY > smileyRows - 1) idxY = smileyRows - 1;
                            char c = smiley[idxY][idxX];
                            if (c == '.') {
                                FillTexelFmt(texel, fmt, 0.6677f, 0.224f, 0.224f);
                            } else if (c == '!') {
                                FillTexelFmt(texel, fmt, 0.0f, 0.0f, 0.0f);
                            } else if (c == '_') {
                                FillTexelFmt(texel, fmt, 1.0f, 1.0f, 1.0f);
                            } else if (c == '#') {
                                FillTexelFmt(texel, fmt, 0.176f, 0.533f, 0.176f);
                            } else {
                                assert (!"Invalid smiley character");
                            }
                            break;
                        }
                        case FILLMODE_NONE:
                        case FILLMODE_DXT1_GREEN:
                        case FILLMODE_DXT1_BLUE:
                            assert(!"Fill not not supported for this format.");
                            return;
                        // No default here so compiler complains if we missed a case.
                    }
                }
            }
        }
    }
}

bool LWNTextureReadWriteTexels::createAndFillTexture(TextureTarget target, int w, int h, int d, Format fmt,
        Texture *&texture, TextureView *textureView, TestFillMode fillMode, bool writeBack, int testLayer, int testLevel,
        StrideType rowStrideType, StrideType imageStrideType)
{
    DEBUG_PRINT(("LWNTextureReadWriteTexels:: Creating %d x %d x %d texture ...\n", w, h, d));
    assert(m_stride || (rowStrideType == S_0 && imageStrideType == S_0));

    Device *device = DeviceState::GetActive()->getDevice();

    // Allocate new test texture.
    TextureBuilder textureBuilder;
    textureBuilder.SetDevice(device)
                  .SetDefaults()
                  .SetTarget(target)
                  .SetFormat(fmt)
                  .SetSize3D(w, h, d);
    if (this->m_mipmap) {
        // Make the texture numLevels always be 4.
        textureBuilder.SetLevels(4);
    }
    texture = m_pool->allocTexture(&textureBuilder);

    if (fillMode != FILLMODE_NONE) {
        // Fill and write to the entire texture.
        TexelBuffer buffer;
        int bufferW = w >> testLevel;
        int bufferH = h >> testLevel;
        int bufferD = 1;
        if (target == TextureTarget::TARGET_3D) {
            bufferD = d >> testLevel;
        }
        if ((target == TextureTarget::TARGET_LWBEMAP || target == TextureTarget::TARGET_LWBEMAP_ARRAY) && m_lwbemapMultiface) {
            bufferD = d;
        }
        fillTexelData(target, bufferW, bufferH, bufferD, fmt, fillMode, buffer, rowStrideType, imageStrideType);

        // CPU blit the textures and swizzle into block-linear.
        DEBUG_PRINT(("LWNTextureReadWriteTexels:: Writing to %d x %d x %d texture buffer at (layer %d, level %d)...\n",
                     w, h, d, testLayer, testLevel));
        CopyRegion region;
        if (target == TextureTarget::TARGET_3D) {
            region = { 0, 0, 0, bufferW, bufferH, bufferD };
        } else {
            // Region depth should be 1 for non-3D texture
            region = { 0, 0, testLayer, bufferW, bufferH, 1 };
        }

        // The stride we pass to LWN might be zero if we want LWN to figure out the actual stride
        // on its own.
        int apiRowStride = (rowStrideType == S_0) ? 0 : buffer.rowStride;
        int apiImageStride = (imageStrideType == S_0) ? 0 : buffer.imageStride;

        // Fills all the textures for the lwbemap and lwbemap_array.
        if (target == TextureTarget::TARGET_LWBEMAP || target == TextureTarget::TARGET_LWBEMAP_ARRAY) {
            if (m_lwbemapMultiface) {
                // Perform a single multi-face lwbemap write.
                region.zoffset = 0;
                region.depth = d;
                if (m_stride) {
                    texture->WriteTexelsStrided(textureView, &region, buffer.ptr, apiRowStride,
                                                apiImageStride);
                } else {
                    texture->WriteTexels(textureView, &region, buffer.ptr);
                }
                texture->FlushTexels(textureView, &region);
            } else {
                // Write layer by layer.
                for (int i = 0; i < d; i++) {
                    region.zoffset = i;
                    if (m_stride) {
                        texture->WriteTexelsStrided(textureView, &region, buffer.ptr, apiRowStride,
                                                    apiImageStride);
                    } else {
                        texture->WriteTexels(textureView, &region, buffer.ptr);
                    }
                    texture->FlushTexels(textureView, &region);
                }
                // Read back one texture layer.
                region.zoffset = testLayer;
                region.depth = 1;
            }
        } else {
            if (m_stride) {
                texture->WriteTexelsStrided(textureView, &region, buffer.ptr, apiRowStride,
                                            apiImageStride);
            } else {
                texture->WriteTexels(textureView, &region, buffer.ptr);
            }
            texture->FlushTexels(textureView, &region);
        }

        // Read the same information back and expect the same thing.
        unsigned char* readbuffer = (unsigned char*)malloc(buffer.size);
        memset(readbuffer, 0, buffer.size);
        assert(readbuffer);
        texture->IlwalidateTexels(textureView, &region);
        if (m_stride) {
            texture->ReadTexelsStrided(textureView, &region, readbuffer, apiRowStride, apiImageStride);
        } else {
            texture->ReadTexels(textureView, &region, readbuffer);
        }

        int numValidRows = (fmt == Format::RGB_DXT1) ? ((h + 3) / 4) : h;
        int validBytesPerRow = (fmt == Format::RGB_DXT1) ? (8 * (w + 3) / 4) : (w * GetFormatBPP(fmt));

        for (int z = 0; z < bufferD; ++z) {
            for (int y = 0; y < numValidRows; ++y) {
                int offset = z * buffer.imageStride + y * buffer.rowStride;
#if DEBUG_MODE
                for (int x = 0; x < validBytesPerRow; ++x) {
                    if (readbuffer[offset + x] != buffer.ptr[offset + x]) {
                        DEBUG_PRINT(("LWNTextureReadWriteTexels:: byte (%d, %d, %d) different: %d vs %d!\n",
                                     x, y, z, readbuffer[offset + x], buffer.ptr[offset + x]));
                        break;
                    }
                }
#endif // DEBUG_MODE
                if (!writeBack) {
                    if (memcmp(readbuffer + offset, buffer.ptr + offset, validBytesPerRow) != 0) {
                        DEBUG_PRINT(("LWNTextureReadWriteTexels:: READ ERROR!\n"));
                        free(readbuffer);
                        return false;
                    }
                }
            }
        }

        if (writeBack) {
            if (m_stride) {
                texture->WriteTexelsStrided(textureView, &region, readbuffer, apiRowStride, apiImageStride);
            } else {
                texture->WriteTexels(textureView, &region, readbuffer);
            }
            texture->FlushTexels(textureView, &region);
        }

        free(readbuffer);
    }
    return true;
}

void LWNTextureReadWriteTexels::draw(TextureTarget target, Texture *texture, int testLayer, int testLevel) {
    Device *device = DeviceState::GetActive()->getDevice();
    QueueCommandBuffer &queueCB = DeviceState::GetActive()->getQueueCB();

    TextureHandle texHandle = device->GetTextureHandle(texture->GetRegisteredTextureID(), m_sampler->GetRegisteredID());
    assert(texHandle);

    Program *program = NULL;
    switch (target) {
        case TextureTarget::TARGET_2D:
            program = m_program2D;
            break;
        case TextureTarget::TARGET_2D_ARRAY:
            program = m_program2DArray;
            break;
        case TextureTarget::TARGET_LWBEMAP:
            program = m_programLwbemap;
            break;
        case TextureTarget::TARGET_LWBEMAP_ARRAY:
            program = m_programLwbemapArray;
            break;
        case TextureTarget::TARGET_3D:
            program = m_program3D;
            break;
        default:
            assert(!"unsupported target.");
            return;
    }

    queueCB.BindProgram(program, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(m_vertexState);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    queueCB.BindVertexBuffer(0, m_vbo->GetAddress(), m_vertexDataSize);

    m_uboPtr->testInfo[0] = testLayer;
    m_uboPtr->testInfo[1] = testLevel;
    queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, m_uboAddress, m_uboSize);

    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
}

// ----------------------------- LWNTextureReadWriteTexelsTest ------------------------------------

class LWNTextureReadWriteTexelsTest {
    const static int cellSize = 100;
    const static int cellMargin = 1;
    TextureTarget target;

    struct TestCase {
        int bgWidth;
        int bgHeight;
        int bgDepth;
        Format bgFormat;
        TestFillMode bgFillMode;

        int fgWidth;
        int fgHeight;
        int fgDepth;
        int fgOffsetX;
        int fgOffsetY;
        int fgOffsetZ;
        Format fgFormat;
        TestFillMode fgFillMode;
        bool writeBack;

        // From 0 to layerNums-1
        int testLayer;
        int testLevel;

        // Stride types for pitch-linear data. Only used for _stride test variants.
        StrideType rowStrideType;
        StrideType imageStrideType;
    };

    const TestCase *testCases;
    int numTestCases;
    int flags;
    bool useMipmap;

public:
    LWNTEST_CppMethods();
    LWNTextureReadWriteTexelsTest(TextureTarget testTarget, int flags);
};

LWNTextureReadWriteTexelsTest::LWNTextureReadWriteTexelsTest(TextureTarget testTarget, int flags)
        : target(testTarget), flags(flags), useMipmap(!!(flags & TFLAG_MIPMAP_ENABLED))
{
    const static TestCase testCases2D[] = {
        // sizeX, sizeY, sizeZ, format, bgFillMode, fgSizeX, fgSizeY, fgSizeZ, copyOffsetX, copyOffsetY, copyOffsetZ, fgFormat, fgFillMode, testReadByWritingBack, testLayer, testLevel

        // Medium sized checkerboard, with smiley image blitted across various cases of BL tile alignment.
        {128, 128,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_0,      S_0},
        {128, 128,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,  15,  31, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_MIN,    S_0},
        {128, 128,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,  31,  15, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_64,     S_MIN},
        {128, 128,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,  20,  21, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_UNALIGN,S_0},
        // Larger sized checkerboard, with smiley image blitted across various cases of BL tile alignment, along with some NPOT image sizes.
        {512, 512,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1,  20,  21, 0, Format::RGBA8,   FILLMODE_SMILEY, false, 0, 0, S_0,      S_0},
        {512, 512,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1,  20,  21, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_MIN,    S_0},
        {512, 513,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1, 120, 231, 0, Format::RGBA8,   FILLMODE_SMILEY, false, 0, 0, S_0,      S_0},
        {512, 513,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1, 120, 231, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_MIN,    S_0},
        {513, 512,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1, 311,   1, 0, Format::RGBA8,   FILLMODE_SMILEY, false, 0, 0, S_0,      S_0},
        {513, 512,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1, 311,   1, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_16,     S_0},
        {128, 128,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD,  32,  32, 1,  70,  70, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_DOUBLE, S_0},
        {253, 123,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD, 100,  32, 1,   1,   1, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_64,     S_0},
        // Tiny images testing that misaligned write/readTexels within a single block works.
        { 18,  17,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD,  10,  10, 1,   5,   5, 0, Format::RGBA8,   FILLMODE_SMILEY, false, 0, 0, S_0,      S_0},
        { 18,  17,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD,  10,  10, 1,   5,   5, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_64,     S_0},
        // Test that different image formats work.
        {432, 432,   1, Format::RGBA16,   FILLMODE_CHECKERBOARD, 234, 231, 1, 142,  99, 0, Format::RGBA16,  FILLMODE_SMILEY,  true, 0, 0, S_0,      S_0},
        {432, 432,   1, Format::RGBA16,   FILLMODE_CHECKERBOARD, 234, 231, 1,  86, 111, 0, Format::RGBA16,  FILLMODE_SMILEY, false, 0, 0, S_64,     S_0},
        {432, 432,   1, Format::RGBA16,   FILLMODE_CHECKERBOARD, 234, 231, 1,  86, 111, 0, Format::RGBA16,  FILLMODE_SMILEY,  true, 0, 0, S_UNALIGN,S_0},
        {432, 432,   1, Format::RGB10A2,  FILLMODE_CHECKERBOARD, 234, 231, 1,  86, 111, 0, Format::RGB10A2, FILLMODE_SMILEY, false, 0, 0, S_0,      S_0},
        {432, 432,   1, Format::RGB10A2,  FILLMODE_CHECKERBOARD, 234, 231, 1,  86, 111, 0, Format::RGB10A2, FILLMODE_SMILEY,  true, 0, 0, S_UNALIGN,S_0},
        {432, 432,   1, Format::RGBA32F,  FILLMODE_CHECKERBOARD, 234, 231, 1,  11,  23, 0, Format::RGBA32F, FILLMODE_SMILEY, false, 0, 0, S_UNALIGN,S_0},
        // Test compressed texture.
        { 16,  16,   1, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,     4,   4, 1,   0,   0, 0, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 0, 0,S_0,      S_0},
        { 18,  17,   1, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,     8,   8, 1,   4,   4, 0, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 0, 0,S_0,      S_0},
        {128, 128,   1, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,    76,  76, 1,   0,   0, 0, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 0, 0,S_MIN,    S_0},
        {253, 253,   1, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,    76,  76, 1,  64,  64, 0, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 0, 0,S_UNALIGN,S_0},
    };
    const static TestCase testCases2DMipmap[] = {
        { 81,  83,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_0,      S_0},
        {256, 256,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_MIN,    S_0},
        {277, 291,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_16,     S_0},
        {256, 256,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 1, S_UNALIGN,S_0},
        {277, 291,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 1, S_64,     S_MIN},
        // Test compressed texture.
        { 32,  32,   1, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,     4,   4, 1,   0,   0, 0, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 0, 0,S_0,      S_0},
        { 32,  32,   1, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,     4,   4, 1,   0,   0, 0, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 0, 1,S_MIN,    S_0},
        { 33,  33,   1, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,     4,   4, 1,   4,   4, 0, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 0, 1,S_UNALIGN,S_0},
        { 81,  83,   1, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,    16,  16, 1,   0,   0, 0, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 0, 1,S_16,     S_0},
        {277, 291,   1, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,    76,  76, 1,  28,  24, 0, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 0, 1,S_64,     S_0},
    };
    const static TestCase testCases2DArray[] = {
        // Medium sized checkerboard, with smiley image blitted across various cases of BL tile alignment.
        {128, 128,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_0,      S_0},
        {128, 128,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,  15,  31, 1, Format::RGBA8,   FILLMODE_SMILEY,  true, 1, 0, S_MIN,    S_0},
        {128, 128,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,  31,  15, 2, Format::RGBA8,   FILLMODE_SMILEY,  true, 2, 0, S_0,      S_MIN},
        {128, 128,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,  20,  21, 3, Format::RGBA8,   FILLMODE_SMILEY,  true, 3, 0, S_UNALIGN,S_UNALIGN},
        // Larger sized checkerboard, with smiley image blitted across various cases of BL tile alignment, along with some NPOT image sizes.
        {512, 512,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1,  20,  21, 0, Format::RGBA8,   FILLMODE_SMILEY, false, 0, 0, S_0,      S_0},
        {512, 512,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1,  20,  21, 1, Format::RGBA8,   FILLMODE_SMILEY,  true, 1, 0, S_MIN,    S_0},
        {512, 513,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1, 120, 231, 2, Format::RGBA8,   FILLMODE_SMILEY, false, 2, 0, S_0,      S_MIN},
        {512, 513,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1, 120, 231, 3, Format::RGBA8,   FILLMODE_SMILEY,  true, 3, 0, S_MIN,    S_512},
        {513, 512,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1, 311,   1, 0, Format::RGBA8,   FILLMODE_SMILEY, false, 0, 0, S_64,     S_8192},
        {513, 512,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1, 311,   1, 1, Format::RGBA8,   FILLMODE_SMILEY,  true, 1, 0, S_64,     S_0},
        {128, 128,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD,  32,  32, 1,  70,  70, 2, Format::RGBA8,   FILLMODE_SMILEY,  true, 2, 0, S_UNALIGN,S_UNALIGN},
        {253, 123,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD, 100,  32, 1,   1,   1, 3, Format::RGBA8,   FILLMODE_SMILEY,  true, 3, 0, S_64,     S_UNALIGN},
        // Tiny images testing that misaligned write/readTexels within a single block works.
        { 18,  17,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD,  10,  10, 1,   5,   5, 0, Format::RGBA8,   FILLMODE_SMILEY, false, 0, 0, S_0,      S_0},
        { 18,  17,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD,  10,  10, 1,   5,   5, 2, Format::RGBA8,   FILLMODE_SMILEY,  true, 2, 0, S_UNALIGN,S_UNALIGN},
        // Test that different image formats work.
        {432, 432,   4, Format::RGBA16,   FILLMODE_CHECKERBOARD, 234, 231, 1, 142,  99, 0, Format::RGBA16,  FILLMODE_SMILEY,  true, 0, 0, S_0,      S_0},
        {432, 432,   4, Format::RGBA16,   FILLMODE_CHECKERBOARD, 234, 231, 1,  86, 111, 1, Format::RGBA16,  FILLMODE_SMILEY, false, 1, 0, S_MIN,    S_MIN},
        {432, 432,   4, Format::RGBA16,   FILLMODE_CHECKERBOARD, 234, 231, 1,  86, 111, 2, Format::RGBA16,  FILLMODE_SMILEY,  true, 2, 0, S_UNALIGN,S_UNALIGN},
        {432, 432,   4, Format::RGB10A2,  FILLMODE_CHECKERBOARD, 234, 231, 1,  86, 111, 3, Format::RGB10A2, FILLMODE_SMILEY, false, 3, 0, S_0,      S_0},
        {432, 432,   4, Format::RGB10A2,  FILLMODE_CHECKERBOARD, 234, 231, 1,  86, 111, 1, Format::RGB10A2, FILLMODE_SMILEY,  true, 1, 0, S_UNALIGN,S_UNALIGN},
        {432, 432,   4, Format::RGBA32F,  FILLMODE_CHECKERBOARD, 234, 231, 1,  11,  23, 3, Format::RGBA32F, FILLMODE_SMILEY, false, 3, 0, S_UNALIGN,S_UNALIGN},
        // Test compressed texture.
        { 18,  17,   4, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,     4,   4, 1,   0,   0, 2, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 2, 0,S_0,      S_0},
        {128, 128,   4, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,    76,  76, 1,   0,   0, 0, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 0, 0,S_UNALIGN,S_UNALIGN},
    };
    const static TestCase testCases2DArrayMipmap[] = {
        // Medium sized checkerboard, with smiley image blitted across various cases of BL tile alignment.
        {256, 256,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_0,      S_0},
        {256, 256,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 1, S_0,      S_0},
        {256, 256,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 2, Format::RGBA8,   FILLMODE_SMILEY,  true, 2, 0, S_0,      S_0},
        {256, 256,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 3, Format::RGBA8,   FILLMODE_SMILEY,  true, 3, 1, S_UNALIGN,S_0},
        {277, 291,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 3, Format::RGBA8,   FILLMODE_SMILEY,  true, 3, 1, S_UNALIGN,S_UNALIGN},
        // Test compressed texture.
        { 18,  17,   4, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,     4,   4, 1,   0,   0, 0, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 0, 1,S_0,      S_UNALIGN},
        { 33,  33,   4, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,     4,   4, 1,   0,   0, 2, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 2, 1,S_64,     S_0},
        {277, 291,   4, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,    76,  76, 1,   0,   0, 3, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 3, 1,S_64,     S_8192},
    };
    const static TestCase testCasesLwbemap[] = {
        // face   layer    position
        // +x       0       right
        // -x       1       left
        // +y       2       bottom
        // -y       3       top
        // +z       4       center
        // -z       5       does not show
        // Medium sized checkerboard, with smiley image blitted across various cases of BL tile alignment.
        {128, 128,   6, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_0,      S_0},
        {128, 128,   6, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,  33,  31, 1, Format::RGBA8,   FILLMODE_SMILEY,  true, 1, 0, S_MIN,    S_0},
        {128, 128,   6, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,  31,  33, 2, Format::RGBA8,   FILLMODE_SMILEY,  true, 2, 0, S_UNALIGN,S_MIN},
        {129, 129,   6, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,  20,  21, 3, Format::RGBA8,   FILLMODE_SMILEY,  true, 3, 0, S_UNALIGN,S_UNALIGN},
        // Larger sized checkerboard, with smiley image blitted across various cases of BL tile alignment, along with some NPOT image sizes.
        {512, 512,   6, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1, 331, 331, 1, Format::RGBA8,   FILLMODE_SMILEY,  true, 1, 0, S_64,     S_0},
        {513, 513,   6, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1, 120, 331, 2, Format::RGBA8,   FILLMODE_SMILEY, false, 2, 0, S_64,     S_8192},
        // Test that different image formats work.
        { 64,  64,   6, Format::RGBA8,    FILLMODE_CHECKERBOARD,  32,  32, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_0,      S_0},
        {128, 128,   6, Format::RGBA16,   FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 3, Format::RGBA16,  FILLMODE_SMILEY,  true, 3, 0, S_UNALIGN,S_UNALIGN},
        // Test compressed texture.
        { 18,  18,   6, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,     4,   4, 1,   0,   0, 3, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 3, 0,S_0,      S_0},
        {128, 128,   6, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,    76,  76, 1,   0,   0, 0, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 0, 0,S_UNALIGN,S_UNALIGN},
    };
    const static TestCase testCasesLwbemapMipmap[] = {
        // Medium sized checkerboard, with smiley image blitted across various cases of BL tile alignment.
        {256, 256,   6, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_0,      S_0},
        {256, 256,   6, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 1, S_MIN,    S_0},
        {256, 256,   6, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,  33,  33, 4, Format::RGBA8,   FILLMODE_SMILEY,  true, 4, 0, S_UNALIGN,S_0},
        {256, 256,   6, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,  15,  14, 3, Format::RGBA8,   FILLMODE_SMILEY,  true, 3, 1, S_UNALIGN,S_UNALIGN},
        {277, 277,   6, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 3, Format::RGBA8,   FILLMODE_SMILEY,  true, 3, 1, S_64,     S_512},
        // Test compressed texture.
        { 17,  17,   6, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,     4,   4, 1,   0,   0, 0, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 0, 1,S_0,      S_0},
        { 33,  33,   6, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,     4,   4, 1,   4,   4, 4, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 4, 1,S_16,     S_512},
        {291, 291,   6, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,    76,  76, 1,  12,  12, 3, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 3, 1,S_UNALIGN,S_UNALIGN},
    };
    const static TestCase testCasesLwbemapArray[] = {
        // The testLayer is the lwbeID
        // Medium sized checkerboard, with smiley image blitted across various cases of BL tile alignment.
        {128, 128,  12, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_0,      S_0},
        {128, 128,  12, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,  32,  32, 7, Format::RGBA8,   FILLMODE_SMILEY,  true, 1, 0, S_UNALIGN,S_UNALIGN},
        // Larger sized checkerboard, with smiley image blitted across various cases of BL tile alignment, along with some NPOT image sizes.
        {512, 512,  12, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1, 331, 331, 7, Format::RGBA8,   FILLMODE_SMILEY,  true, 1, 0, S_0,      S_0},
        {513, 513,  12, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1, 120, 331, 8, Format::RGBA8,   FILLMODE_SMILEY, false, 1, 0, S_64,     S_8192},
        // Test that different image formats work.
        { 64,  64,  12, Format::RGBA8,    FILLMODE_CHECKERBOARD,  32,  32, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_UNALIGN,S_UNALIGN},
        {128, 128,  12, Format::RGBA16,   FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 9, Format::RGBA16,  FILLMODE_SMILEY,  true, 1, 0, S_UNALIGN,S_UNALIGN},
        // Test compressed texture.
        { 18,  18,  12, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,     4,   4, 1,   0,   0, 9, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 1, 0,S_0,      S_0},
        {128, 128,  12, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,    76,  76, 1,   0,   0, 0, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 0, 0,S_UNALIGN,S_UNALIGN},
    };
    const static TestCase testCasesLwbemapArrayMipmap[] = {
        // Medium sized checkerboard, with smiley image blitted across various cases of BL tile alignment.
        {256, 256,  12, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_0,      S_0},
        {256, 256,  12, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 1, S_MIN,    S_MIN},
        {256, 256,  12, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,  33,  35,10, Format::RGBA8,   FILLMODE_SMILEY,  true, 1, 0, S_UNALIGN,S_MIN},
        {256, 256,  12, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,  15,  14, 9, Format::RGBA8,   FILLMODE_SMILEY,  true, 1, 1, S_UNALIGN,S_UNALIGN},
        {277, 277,  12, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,   0,   0, 9, Format::RGBA8,   FILLMODE_SMILEY,  true, 1, 1, S_64,     S_8192},
        // Test compressed texture.
        { 17,  17,  12, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,     4,   4, 1,   0,   0, 0, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 0, 1,S_0,      S_0},
        { 33,  33,  12, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,     4,   4, 1,   4,   4,10, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 1, 1,S_16,     S_512},
        {291, 291,  12, Format::RGB_DXT1, FILLMODE_DXT1_GREEN,    76,  76, 1,  12,  12, 9, Format::RGB_DXT1,FILLMODE_DXT1_BLUE,true, 1, 1,S_UNALIGN,S_UNALIGN},
    };
    const static TestCase testCases3D[] = {
        // Medium sized checkerboard, with smiley image blitted across various cases of BL tile alignment.
        {128, 128,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 2,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_0,      S_0},
        {128, 128,   5, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 1,  15,  31, 1, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_MIN,    S_MIN},
        {128, 128,   6, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 2,  31,  15, 2, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_UNALIGN,S_0},
        {128, 128,   7, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 2,  20,  21, 3, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_UNALIGN,S_UNALIGN},
        // Larger sized checkerboard, with smiley image blitted across various cases of BL tile alignment, along with some NPOT image sizes.
        {512, 512,   8, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 3,  20,  21, 0, Format::RGBA8,   FILLMODE_SMILEY, false, 0, 0, S_0,      S_UNALIGN},
        {512, 512,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1,  20,  21, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_MIN,    S_0},
        {512, 513,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1, 120, 231, 0, Format::RGBA8,   FILLMODE_SMILEY, false, 0, 0, S_64,     S_512},
        {512, 513,   2, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 2, 120, 231, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_64,     S_8192},
        {513, 512,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1, 311,   1, 0, Format::RGBA8,   FILLMODE_SMILEY, false, 0, 0, S_0,      S_0},
        {513, 512,   2, Format::RGBA8,    FILLMODE_CHECKERBOARD, 123, 123, 1, 311,   1, 1, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_64,     S_8192},
        {128, 128,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD,  32,  32, 1,  70,  70, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_0,      S_UNALIGN},
        {253, 123,   1, Format::RGBA8,    FILLMODE_CHECKERBOARD, 100,  32, 1,   1,   1, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_64,     S_8192},
        // Tiny images testing that misaligned write/readTexels within a single block works.
        { 18,  17,  13, Format::RGBA8,    FILLMODE_CHECKERBOARD,  10,  10, 6,   5,   5, 0, Format::RGBA8,   FILLMODE_SMILEY, false, 0, 0, S_0,      S_0},
        { 18,  17,  13, Format::RGBA8,    FILLMODE_CHECKERBOARD,  10,  10, 6,   5,   5, 5, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_UNALIGN,S_UNALIGN},
        // Test that different image formats work.
        {432, 432,   4, Format::RGBA16,   FILLMODE_CHECKERBOARD, 234, 231, 2, 142,  99, 0, Format::RGBA16,  FILLMODE_SMILEY,  true, 0, 0, S_0,      S_0},
        {432, 432,   3, Format::RGBA16,   FILLMODE_CHECKERBOARD, 234, 231, 2,  86, 111, 1, Format::RGBA16,  FILLMODE_SMILEY, false, 0, 0, S_MIN,    S_MIN},
        {432, 432,   6, Format::RGBA16,   FILLMODE_CHECKERBOARD, 234, 231, 6,  86, 111, 0, Format::RGBA16,  FILLMODE_SMILEY,  true, 0, 0, S_UNALIGN,S_UNALIGN},
        {432, 432,   1, Format::RGB10A2,  FILLMODE_CHECKERBOARD, 234, 231, 1,  86, 111, 0, Format::RGB10A2, FILLMODE_SMILEY, false, 0, 0, S_UNALIGN,S_UNALIGN},
        {432, 432,   6, Format::RGB10A2,  FILLMODE_CHECKERBOARD, 234, 231, 6,  86, 111, 0, Format::RGB10A2, FILLMODE_SMILEY,  true, 0, 0, S_0,      S_0},
        {432, 432,   1, Format::RGBA32F,  FILLMODE_CHECKERBOARD, 234, 231, 1,  11,  23, 0, Format::RGBA32F, FILLMODE_SMILEY, false, 0, 0, S_64,     S_8192},
    };
    const static TestCase testCases3DMipmap[] = {
        {128, 128,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 2,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 0, S_0,      S_0},
        {256, 256,   4, Format::RGBA8,    FILLMODE_CHECKERBOARD,  75,  75, 2,   0,   0, 0, Format::RGBA8,   FILLMODE_SMILEY,  true, 0, 1, S_UNALIGN,S_UNALIGN},
    };

    if (target == TextureTarget::TARGET_2D && !useMipmap) {
        testCases = &testCases2D[0];
        numTestCases = __GL_ARRAYSIZE(testCases2D);
    }
    if (target == TextureTarget::TARGET_2D && useMipmap) {
        testCases = &testCases2DMipmap[0];
        numTestCases = __GL_ARRAYSIZE(testCases2DMipmap);
    }

    if (target == TextureTarget::TARGET_2D_ARRAY && !useMipmap) {
        testCases = &testCases2DArray[0];
        numTestCases = __GL_ARRAYSIZE(testCases2DArray);
    }
    if (target == TextureTarget::TARGET_2D_ARRAY && useMipmap) {
        testCases = &testCases2DArrayMipmap[0];
        numTestCases = __GL_ARRAYSIZE(testCases2DArrayMipmap);
    }

    if (target == TextureTarget::TARGET_LWBEMAP && !useMipmap) {
        testCases = &testCasesLwbemap[0];
        numTestCases = __GL_ARRAYSIZE(testCasesLwbemap);
    }
    if (target == TextureTarget::TARGET_LWBEMAP && useMipmap) {
        testCases = &testCasesLwbemapMipmap[0];
        numTestCases = __GL_ARRAYSIZE(testCasesLwbemapMipmap);
    }

    if (target == TextureTarget::TARGET_LWBEMAP_ARRAY && !useMipmap) {
        testCases = &testCasesLwbemapArray[0];
        numTestCases = __GL_ARRAYSIZE(testCasesLwbemapArray);
    }
    if (target == TextureTarget::TARGET_LWBEMAP_ARRAY && useMipmap) {
        testCases = &testCasesLwbemapArrayMipmap[0];
        numTestCases = __GL_ARRAYSIZE(testCasesLwbemapArrayMipmap);
    }

    if (target == TextureTarget::TARGET_3D && !useMipmap) {
        testCases = &testCases3D[0];
        numTestCases = __GL_ARRAYSIZE(testCases3D);
    }
    if (target == TextureTarget::TARGET_3D && useMipmap) {
        testCases = &testCases3DMipmap[0];
        numTestCases = __GL_ARRAYSIZE(testCases3DMipmap);
    }
}

lwString LWNTextureReadWriteTexelsTest::getDescription() const
{
    lwStringBuf sb;
    sb << "Simple test for lwnTextureRead/Write texels. The test writes a checkerboard "
          "background and writes a smaller foreground smiley image onto a subregion of the background.  "
          "To test read, the background is read back and then the read data is written again.  "
          "Passing test should look like squares with smiley images with a white border on a checkerboard pattern "
          "background.  RGBA8, RGBA16, RGB10A2, RGBA32F formats are tested with various sub-region."
          "copy parameters to test various cases crossing image block borders."
          "Display green as background and blue subregion as foreground when testing the compressed format."
          "Base level and non-base level mipmap are tested for all texture targets.";
    if (target== TextureTarget::TARGET_2D_ARRAY) {
        sb << "  Test for the base and non-base layer of the texture.";
    }
    if (target== TextureTarget::TARGET_LWBEMAP || target== TextureTarget::TARGET_LWBEMAP_ARRAY) {
        sb << "  The test fills all the faces with checkerboard background and then fills a subresion of one face "
              "with smiley image. Display 5 faces on the render target (except the -z face).";
    }
    if (target== TextureTarget::TARGET_3D) {
        sb << "  Each square iterates through all layers, and since the smiley images are blitted "
              "to a subset of those layers the smileys will appear fade to the background checkerboard pattern.";
    }
    if (this->flags & TFLAG_STRIDE) {
        sb << "  This test uses the Strided variant of lwnTextureRead/WriteTexels. The background "
              "image has a variety of specified row and image strides.";
    }
    return sb.str();
}

int LWNTextureReadWriteTexelsTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(41, 2);
}

void LWNTextureReadWriteTexelsTest::doGraphics() const
{
    static const int cellsX = lwrrentWindowWidth / cellSize;
    static const int cellsY = lwrrentWindowHeight / cellSize;
    cellTestInit(cellsX, cellsY);

    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.ClearColor(0, 0.15, 0.15, 0.15, 1.0);

    LWNTextureReadWriteTexels textureRWTexelsTest(this->flags);
    textureRWTexelsTest.init();
    Texture *BGtextures[MAX_SUBTESTS];

    bool useStride = (this->flags & TFLAG_STRIDE) != 0;

    for (int k = 0; k < numTestCases; k++) {
        SetCellViewportScissorPadded(queueCB, k % cellsX, k / cellsX, cellMargin);
        queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0, 1.0);
        const TestCase& tc = testCases[k];

        // Create the BG texture and fill with info.
        TextureView textureView;
        textureView.SetDefaults().SetLevels(tc.testLevel, 1);

        bool ret = textureRWTexelsTest.createAndFillTexture(
            target, tc.bgWidth, tc.bgHeight, tc.bgDepth,
            tc.bgFormat, BGtextures[k], &textureView, tc.bgFillMode,
            tc.writeBack, tc.testLayer, tc.testLevel,
            useStride ? tc.rowStrideType : S_0, useStride ? tc.imageStrideType : S_0
        );
        if (!ret) {
            // Error oclwred while reading texture.
            continue;
        }

        // Create the FG texture and fill with info.
        LWNTextureReadWriteTexels::TexelBuffer FGTexelsBuffer;
        textureRWTexelsTest.fillTexelData(
            target, tc.fgWidth, tc.fgHeight, tc.fgDepth,
            tc.fgFormat, tc.fgFillMode, FGTexelsBuffer,
            useStride ? tc.rowStrideType : S_0, useStride ? tc.imageStrideType : S_0
        );

        // Blit the texture from FG to BG.
        CopyRegion region;
        region.xoffset = tc.fgOffsetX;
        region.yoffset = tc.fgOffsetY;
        region.zoffset = tc.fgOffsetZ;
        region.width = tc.fgWidth;
        region.height = tc.fgHeight;
        region.depth = tc.fgDepth;

        DEBUG_PRINT(("LWNTexture:: WriteTexels: blitting from fg to bg layer...\n"));
        if (useStride) {
            BGtextures[k]->WriteTexelsStrided(&textureView, &region, FGTexelsBuffer.ptr,
                                              (tc.rowStrideType == S_0) ? 0 : FGTexelsBuffer.rowStride,
                                              (tc.imageStrideType == S_0) ? 0 : FGTexelsBuffer.imageStride);
        } else {
            BGtextures[k]->WriteTexels(&textureView, &region, FGTexelsBuffer.ptr);
        }
        BGtextures[k]->FlushTexels(&textureView, &region);

        // Draw the texture.
        textureRWTexelsTest.draw(target, BGtextures[k], tc.testLayer, tc.testLevel);

        queueCB.submit();
        queue->Finish();
        assert(k < MAX_SUBTESTS);
    }
}

#define MKTEST(name, target, flags) \
    OGTEST_CppTest(LWNTextureReadWriteTexelsTest, lwn_texture_wr_texels_ ## name, (TextureTarget::TARGET_ ## target, flags))

MKTEST(2d,                                  2D,             TFLAG_NONE);
MKTEST(2d_mipmap,                           2D,             TFLAG_MIPMAP_ENABLED);
MKTEST(2d_array,                            2D_ARRAY,       TFLAG_NONE);
MKTEST(2d_array_mipmap,                     2D_ARRAY,       TFLAG_MIPMAP_ENABLED);
MKTEST(lwbemap,                             LWBEMAP,        TFLAG_NONE);
MKTEST(lwbemap_mipmap,                      LWBEMAP,        TFLAG_MIPMAP_ENABLED);
MKTEST(lwbemap_multi,                       LWBEMAP,        TFLAG_LWBEMAP_MULTIFACE);
MKTEST(lwbemap_mipmap_multi,                LWBEMAP,        TFLAG_MIPMAP_ENABLED | TFLAG_LWBEMAP_MULTIFACE);
MKTEST(lwbemap_array,                       LWBEMAP_ARRAY,  TFLAG_NONE);
MKTEST(lwbemap_array_mipmap,                LWBEMAP_ARRAY,  TFLAG_MIPMAP_ENABLED);
MKTEST(lwbemap_array_multi,                 LWBEMAP_ARRAY,  TFLAG_LWBEMAP_MULTIFACE);
MKTEST(lwbemap_array_mipmap_multi,          LWBEMAP_ARRAY,  TFLAG_MIPMAP_ENABLED | TFLAG_LWBEMAP_MULTIFACE);
MKTEST(3d,                                  3D,             TFLAG_NONE);
MKTEST(3d_mipmap,                           3D,             TFLAG_MIPMAP_ENABLED);
MKTEST(2d_stride,                           2D,             TFLAG_STRIDE);
MKTEST(2d_mipmap_stride,                    2D,             TFLAG_MIPMAP_ENABLED | TFLAG_STRIDE);
MKTEST(2d_array_stride,                     2D_ARRAY,       TFLAG_STRIDE);
MKTEST(2d_array_mipmap_stride,              2D_ARRAY,       TFLAG_MIPMAP_ENABLED | TFLAG_STRIDE);
MKTEST(lwbemap_stride,                      LWBEMAP,        TFLAG_STRIDE);
MKTEST(lwbemap_mipmap_stride,               LWBEMAP,        TFLAG_MIPMAP_ENABLED | TFLAG_STRIDE);
MKTEST(lwbemap_multi_stride,                LWBEMAP,        TFLAG_LWBEMAP_MULTIFACE | TFLAG_STRIDE);
MKTEST(lwbemap_mipmap_multi_stride,         LWBEMAP,        TFLAG_MIPMAP_ENABLED | TFLAG_LWBEMAP_MULTIFACE | TFLAG_STRIDE);
MKTEST(lwbemap_array_stride,                LWBEMAP_ARRAY,  TFLAG_STRIDE);
MKTEST(lwbemap_array_mipmap_stride,         LWBEMAP_ARRAY,  TFLAG_MIPMAP_ENABLED | TFLAG_STRIDE);
MKTEST(lwbemap_array_multi_stride,          LWBEMAP_ARRAY,  TFLAG_LWBEMAP_MULTIFACE | TFLAG_STRIDE);
MKTEST(lwbemap_array_mipmap_multi_stride,   LWBEMAP_ARRAY,  TFLAG_MIPMAP_ENABLED | TFLAG_LWBEMAP_MULTIFACE | TFLAG_STRIDE);
MKTEST(3d_stride,                           3D,             TFLAG_STRIDE);
MKTEST(3d_mipmap_stride,                    3D,             TFLAG_MIPMAP_ENABLED | TFLAG_STRIDE);


class TextureViewTest {
public:
    LWNTEST_CppMethods();

    private:
    // Compression block stats for BPTC encoding
    static const int BLOCK_WIDTH = 4;
    static const int BLOCK_HEIGHT = 4;
    static const int BLOCK_BYTES = 16;

    static const int TEX_WIDTH = 1024;
    static const int TEX_HEIGHT = 1024;
    static const int TEX_BYTES = (TEX_WIDTH / BLOCK_WIDTH) * (TEX_HEIGHT / BLOCK_HEIGHT) * BLOCK_BYTES;
};

lwString TextureViewTest::getDescription() const
{
    return
        "Regression test for Bug 2048368. Populating a BPTC texture using\n"
        "Texture::WriteTexels and redundantly specifying the format inside a TextureView\n"
        "passed to Texture::WriteTexels would produce incorrect texture contents.\n"
        "\n"
        "Populate a texture in this way, read back the results using"
        "CommandBuffer::CopyTextureToBuffer, and compare the buffer contents with the\n"
        "source data. Clear to green if the contents are identical; red otherwise.";
}

int TextureViewTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(53, 304);
}

void TextureViewTest::doGraphics() const
{
    // Use 16-bit ints, because lwRandNumber returns 16-bit values.
    std::vector<uint16_t> srcData(TEX_BYTES / sizeof(uint16_t));
    for (size_t i = 0; i < TEX_BYTES / sizeof(uint16_t); ++i) {
        srcData[i] = uint16_t(lwRandNumber());
    }

    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    MemoryPoolAllocator allocator(device, nullptr, TEX_BYTES * 2,
                                  MemoryPoolFlags::CPU_CACHED | MemoryPoolFlags::GPU_CACHED);

    TextureBuilder textureBuilder;
    textureBuilder.SetDevice(device)
                  .SetDefaults()
                  .SetTarget(TextureTarget::TARGET_2D)
                  .SetSize2D(TEX_WIDTH, TEX_HEIGHT)
                  .SetFormat(Format::BPTC_UNORM);
    Texture* texture = allocator.allocTexture(&textureBuilder);

    BufferBuilder bufferBuilder;
    bufferBuilder.SetDevice(device).SetDefaults();
    Buffer* destBuffer = allocator.allocBuffer(&bufferBuilder, BUFFER_ALIGN_COPY_READ_BIT, TEX_BYTES);

    // Redundantly setting the texture format inside a view caused the original bug.
    TextureView textureView;
    textureView.SetDefaults()
               .SetFormat(Format::BPTC_UNORM);

    CopyRegion region = { 0, 0, 0, TEX_WIDTH, TEX_HEIGHT, 1 };

    texture->WriteTexels(&textureView, &region, srcData.data());
    texture->FlushTexels(&textureView, &region);

    queueCB.CopyTextureToBuffer(texture, nullptr, &region, destBuffer->GetAddress(), CopyFlags::NONE);
    queueCB.submit();
    queue->Finish();
    destBuffer->IlwalidateMappedRange(0, TEX_BYTES);

    void* srcPtr = srcData.data();
    void* destPtr = destBuffer->Map();

    if (memcmp(srcPtr, destPtr, TEX_BYTES) == 0) {
        queueCB.ClearColor(0, 0.0f, 1.0f, 0.0f, 1.0f);
    } else {
        queueCB.ClearColor(0, 1.0f, 0.0f, 0.0f, 1.0f);
    }

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(TextureViewTest, lwn_texture_wr_texels_texture_view, );
