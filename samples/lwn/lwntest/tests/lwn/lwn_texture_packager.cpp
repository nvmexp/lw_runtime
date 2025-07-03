/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

//
// lwn_texture_packager.cpp
//
// Test for various offline texture packager packed textures.
//

#include <vector>

#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include "lwnTool/lwnTool_TexpkgConsumer.h"
#include "g_lwn_texpkg_data.h"
#include "g_lwn_texpkg_astc_data.h"

#define DEBUG_MODE 0
#if DEBUG_MODE
    #define DEBUG_PRINT(x) do { \
        printf x; \
        fflush(stdout); \
    } while (0)
#else
    #define DEBUG_PRINT(x)
#endif

using namespace lwn;
using namespace lwn::dt;
using namespace lwnTool;

// --------------------------------- LWNTexturePackagerHarness ----------------------------------------

#define MAX_XTX_ARCHIVES 5

enum TexpkgSamplerType {
    SAMPLERTYPE_2D,
    SAMPLERTYPE_LWBE,
    SAMPLERTYPE_3D,
    SAMPLERTYPE_2D_ARRAY,
    SAMPLERTYPE_MAX
};

struct XTXTexture {
    Texture *texture;
    bool textureLoaded;

    texpkg::LWNHWTextureHeader *textureHeader;
    uint64_t offset;
    uint64_t size;
};

class LWNTexturePackagerHarness {
    MemoryPool *m_pool[MAX_XTX_ARCHIVES];
    std::vector<XTXTexture*> m_textures;

    MemoryPoolAllocator *m_bufpool;
    Program *m_program[SAMPLERTYPE_MAX];
    Sampler *m_sampler;
    VertexArrayState m_vertexState;
    Buffer *m_vbo;
    LWNuint m_vertexDataSize;
    const bool m_srgb;

    struct Vertex {
        vec3 position;
        vec3 uv;
    };

public:
    explicit LWNTexturePackagerHarness(bool srgb);
    ~LWNTexturePackagerHarness();

    bool initXTXArchive(unsigned char* data, unsigned int dataSz, int archiveIdx);
    bool init(void);
    void draw(int idx, int miplvl);

    int numTextures(void);
    int numMipmapLevels(int idx);
};

LWNTexturePackagerHarness::LWNTexturePackagerHarness(bool srgb)
        : m_bufpool(NULL), m_sampler(NULL), m_vbo(NULL),
          m_vertexDataSize(0), m_srgb(srgb)
{
}

LWNTexturePackagerHarness::~LWNTexturePackagerHarness()
{
    for (int i = 0; i < (int) m_textures.size(); i++) {
        delete m_textures[i];
        m_textures[i] = NULL;
    }
    delete m_bufpool;
}

bool LWNTexturePackagerHarness::initXTXArchive(unsigned char* data, unsigned int dataSz, int archiveIdx)
{
    assert(archiveIdx >= 0 && archiveIdx < MAX_XTX_ARCHIVES);
    assert(data && dataSz);
    Device *device = DeviceState::GetActive()->getDevice();

    // Verify the archive header.
    texpkg::ArchiveHeader* archiveHeader = reinterpret_cast<texpkg::ArchiveHeader*>(&data[0]);
    if (archiveHeader->magic != LWNFD_ARCHIVE_MAGIC) {
        DEBUG_PRINT(("Invalid XTX file header\n"));
        return false;
    }

    // Build a pool. Hold a pool party. Fun for the whole family!.
    m_pool[archiveIdx] = device->CreateMemoryPool(data, dataSz, MemoryPoolType::GPU_ONLY);

    // Loop through archive and build the list of texture headers.
    uint64_t offset = archiveHeader->headerSize;

    while (true) {

        // Load the block header.
        texpkg::BlockHeader* blockHeader = reinterpret_cast<texpkg::BlockHeader*>(&data[offset]);
        if (blockHeader->magic != LWNFD_BLOCKHEADER_MAGIC) {
            DEBUG_PRINT(("Invalid XTX block header\n"));
            return false;
        }

        if (blockHeader->blockType == texpkg::LWNFD_BLOCK_END) {
            break;
        }

        // Load header information.
        if (blockHeader->blockType == texpkg::LWNFD_BLOCK_TEXTURE_HEADER) {
            assert(blockHeader->blockSize >= sizeof(texpkg::LWNHWTextureHeader));
            texpkg::LWNHWTextureHeader* textureHeader =
                    reinterpret_cast<texpkg::LWNHWTextureHeader *>(&data[offset + blockHeader->offset]);

            DEBUG_PRINT(("Texture %d size: %u x %u x %u mipmaps %d format 0x%x ",
                    blockHeader->typeIndex,
                    textureHeader->width,
                    textureHeader->height,
                    textureHeader->depth,
                    textureHeader->mipmapLevels,
                    textureHeader->format));

            m_textures.push_back(new XTXTexture);
            size_t idx = m_textures.size() - 1;
            m_textures[idx]->textureHeader = textureHeader;
            m_textures[idx]->textureLoaded = false;
        }

        // Load and build the texture data.
        if (blockHeader->blockType == texpkg::LWNFD_BLOCK_TEXTURE) {
            size_t idx = m_textures.size() - 1;
            assert(m_textures.size() >= blockHeader->typeIndex + 1);
            assert(m_textures[idx]);
            m_textures[idx]->offset = offset + blockHeader->offset;
            m_textures[idx]->size = blockHeader->blockSize;
            DEBUG_PRINT(("offset 0x%x sz %u\n",
                    (int) m_textures[idx]->offset,
                    (int)m_textures[idx]->size));
            assert(m_textures[idx]->textureHeader);
        }

        // Advance onto next block.
        offset += blockHeader->offset + blockHeader->blockSize;
    }

    // Loop through all the texture headers and create textures out of them.
    TextureBuilder textureBuilder;
    textureBuilder.SetDevice(device);
    for (int i = 0; i < (int) m_textures.size(); i++) {
        if (m_textures[i]->textureLoaded) {
            continue;
        }

        // Don't set up the texture if it was packaged using the sparse option
        // and the underlying GPU doesn't support it.  The debug layer will
        // complain when SetPackagedTextureLayout is called.
        if (m_textures[i]->textureHeader->sparse && !g_lwnDeviceCaps.supportsMaxwellSparsePackagedTextures) {
            continue;
        }

        // Set texture builder.
        assert(m_textures[i] && m_textures[i]->textureHeader);
        texpkg::LWNHWTextureHeader* textureHeader = m_textures[i]->textureHeader;
        DEBUG_PRINT(("Creating LWN %stexture %d...\n", textureHeader->sparse ? "sparse " : "", i));

        Format fmt = Format::Enum(textureHeader->format);

        if (m_srgb) {
            assert((uint32_t)fmt >= (uint32_t)Format::RGBA_ASTC_4x4 && (uint32_t)fmt <= (uint32_t)Format::RGBA_ASTC_12x12);

            static const uint32_t numAstcRgbFormats = (uint32_t)Format::RGBA_ASTC_12x12 - (uint32_t)Format::RGBA_ASTC_4x4 + 1;
            // Chose the corresponding SRGB format
            fmt = Format::Enum(textureHeader->format + numAstcRgbFormats);
        }

        textureBuilder.SetDefaults()
                      .SetFlags(textureHeader->sparse ? LWN_TEXTURE_FLAGS_SPARSE_BIT : (LWNtextureFlags) 0)
                      .SetTarget(TextureTarget::Enum(textureHeader->target))
                      .SetFormat(fmt)
                      .SetLevels(textureHeader->mipmapLevels)
                      .SetSize3D(textureHeader->width, textureHeader->height, textureHeader->depth)
                      .SetPackagedTextureLayout((const PackagedTextureLayout*) &textureHeader->layout)
                      .SetPackagedTextureData(data + m_textures[i]->offset);

        // Check alignment.
        if (m_textures[i]->offset % textureBuilder.GetStorageAlignment() != 0) {
            return false;
        }

        // Create texture.
        m_textures[i]->texture = textureBuilder.CreateTextureFromPool(
                m_pool[archiveIdx], m_textures[i]->offset);
        if (m_textures[i]->texture == NULL) {
            return false;
        }
        m_textures[i]->textureLoaded = true;
    }

    return true;
}

bool LWNTexturePackagerHarness::init(void)
{
    Device *device = DeviceState::GetActive()->getDevice();
    m_bufpool = new MemoryPoolAllocator(device, NULL, 0x10000, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // Compile shaders.
    // The frag shader shows alpha using a small checkerboard pattern.
    for (int i = 0 ; i < (int) SAMPLERTYPE_MAX; i++) {

        static const char* samplerShaderStr[] = {
            "sampler2D",
            "samplerLwbe",
            "sampler3D",
            "sampler2DArray"
        };
        static const char* samplerShaderFetchStr[] = {
            "textureLod(tex, ouv, 0.0)",
            "textureLod(tex, vec3(ouv * 2.0 - 1.0, 0.35), 0.0)",
            "textureLod(tex, vec3(fract(ouv * 2.0), (floor(ouv.x * 2.0) + floor(ouv.y * 2.0) * 2.0) / 4.0 + 0.125), 0.0)",
            "textureLod(tex, vec3(fract(ouv * 2.0), floor(ouv.x * 2.0) + floor(ouv.y * 2.0) * 2.0 ), 0.0)"
        };

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
        FragmentShader fs(440);
        fs <<
            "in vec2 ouv;\n"
            "out vec4 fcolor;\n"
            "layout (binding=0) uniform " << samplerShaderStr[i] << " tex;\n";

            if (m_srgb) {
                fs <<
                    "vec3 colwertToSrgb(vec3 cl) {\n"
                    "    bvec3 limit = lessThan(cl, vec3(0.0031308f));\n"
                    "    vec3 l = vec3(12.92f) * cl;\n"
                    "    vec3 h = vec3(1.055f) * pow(cl, vec3(0.41666)) - vec3(0.055);\n"
                    "    return mix(h, l, limit);\n"
                    "}\n";
            }
        fs <<
            "void main() {\n"
            "  fcolor = " << samplerShaderFetchStr[i] << ";\n";

        if (m_srgb) {
            // g_lwn_texpkg_data.h does not contain any SRGB data. To test SRGB, we interpret the regular
            // RGB value as SRGB and revert the colwersion from SRGB to RGB here in the fragment shader by
            // colwerting RGB back to SRGB.
            fs <<
                "  fcolor = vec4(colwertToSrgb(fcolor.rgb), fcolor.a);\n";
        }
        fs <<
            "}\n";

        m_program[i] = device->CreateProgram();
        if (!g_glslcHelper->CompileAndSetShaders(m_program[i], vs, fs)) {
            DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
            return false;
        }
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

    return true;
}


void LWNTexturePackagerHarness::draw(int idx, int miplvl)
{
    Device *device = DeviceState::GetActive()->getDevice();
    QueueCommandBuffer &queueCB = DeviceState::GetActive()->getQueueCB();

    assert(idx >= 0 && idx < (int) m_textures.size());
    assert(miplvl >= 0 && miplvl < numMipmapLevels(idx));

    if (m_textures[idx]->textureHeader->sparse && !g_lwnDeviceCaps.supportsMaxwellSparsePackagedTextures) {
        // Sparse texture not supported on pre-Maxwell devices.
        queueCB.ClearColor(0, 0.2, 0.2, 1.0, 1.0);
        return;
    }

    // Set sampler to LOD clamp to our desired mipmap level.
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults()
      .SetLodClamp(miplvl, miplvl)
      .SetMinMagFilter(MinFilter::LINEAR_MIPMAP_LINEAR, MagFilter::LINEAR);
    m_sampler = sb.CreateSampler();
    TextureHandle texHandle = device->GetTextureHandle(
        m_textures[idx]->texture->GetRegisteredTextureID(), m_sampler->GetRegisteredID());
    assert(texHandle);

    // Figure out the the program of the right sampler type to bind.
    switch ((LWNtextureTarget) m_textures[idx]->textureHeader->target) {
        case LWN_TEXTURE_TARGET_2D:
            queueCB.BindProgram(m_program[SAMPLERTYPE_2D], ShaderStageBits::ALL_GRAPHICS_BITS);
            break;
        case LWN_TEXTURE_TARGET_LWBEMAP:
            queueCB.BindProgram(m_program[SAMPLERTYPE_LWBE], ShaderStageBits::ALL_GRAPHICS_BITS);
            break;
        case LWN_TEXTURE_TARGET_3D:
            queueCB.BindProgram(m_program[SAMPLERTYPE_3D], ShaderStageBits::ALL_GRAPHICS_BITS);
            break;
        case LWN_TEXTURE_TARGET_2D_ARRAY:
            queueCB.BindProgram(m_program[SAMPLERTYPE_2D_ARRAY], ShaderStageBits::ALL_GRAPHICS_BITS);
            break;
        default:
            DEBUG_PRINT(("Unknown texture target. Defaulting to 2D.\n"));
            queueCB.BindProgram(m_program[SAMPLERTYPE_2D], ShaderStageBits::ALL_GRAPHICS_BITS);
            break;
    }

    queueCB.BindVertexArrayState(m_vertexState);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    queueCB.BindVertexBuffer(0, m_vbo->GetAddress(), m_vertexDataSize);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
}

int LWNTexturePackagerHarness::numTextures(void)
{
    return (int) m_textures.size();
}

int LWNTexturePackagerHarness::numMipmapLevels(int idx)
{
    assert(idx >= 0 && idx < (int) m_textures.size());
    assert(m_textures[idx]);
    assert(m_textures[idx]->textureHeader);
    return (int) m_textures[idx]->textureHeader->mipmapLevels;
}


// --------------------------------- LWNTexturePackagerTest ----------------------------------------

enum TestMode {
    TEST_TEXPKG,
    TEST_ASTC,
    TEST_ASTC_SRGB
};

class LWNTexturePackagerTest {
    static const int cellSize = 54;
    static const int cellSizeAstc = 84;
    static const int cellMargin = 1;
    TestMode m_mode;

public:
    LWNTexturePackagerTest(TestMode mode);
    LWNTEST_CppMethods();
};

LWNTexturePackagerTest::LWNTexturePackagerTest(TestMode mode)
    : m_mode(mode)
{
}

lwString LWNTexturePackagerTest::getDescription() const
{
    lwStringBuf sb;
    if (m_mode == TEST_TEXPKG) {
        sb <<
            "Simple LWN texture packager pre-built TX1 archive pre-test.\n"
            "Goes through every texture and draws each mipmap of it, one tile each.\n"
            "The pre-build archive contains textures using a variety of formats and\n"
            "texture targets.\n"
            "Correct output should look something like, from bottom left:\n"
            "    * 6 Grid pattern tiles with letters R ,G, B, A, 6 mipmap lvls 1 tile each.\n"
            "    * 23 Green face tiles, the RGBA8 & DXT1 textures have mipmaps.\n"
            "    * 4 Lwbemap texture tiles, 1 unmipped + 3 mipmap lvls 1 tile each.\n"
            "    * 14 ASTC textures (blue if unsupported) feature mini LW logos.\n"
            "    * 6 RGB10A2 volume texture tiles with colored numbers, 5 mipmaps 1 tile each.\n"
            "    * 6 RGBA8 volume texture tiles with colored numbers, 5 mipmaps 1 tile each.\n"
            "    * 4 array tex tiles  with colored numbers, 4 mipmaps 1 tile each.\n";
    } else if (m_mode == TEST_ASTC) {
        sb <<
            "Simple LWN texture packager pre-built TX1 archive pre-test for ASTC RGB textures.\n"
            "Goes through every texture and draws each mipmap of it, one tile each.\n"
            "The pre-build archive contains ASTC textures using a variety of formats and\n"
            "texture targets.\n"
            "Correct output should look something like a space photo at different mipmap levels.\n";
    } else if (m_mode == TEST_ASTC_SRGB) {
        sb <<
            "Simple LWN texture packager pre-built TX1 archive pre-test for ASTC sRGB textures.\n"
            "The pre-built archive contains non-sRGB ASTC textures that are interpreted as though\n"
            "they were sRGB. The results of the lookup undergo an linear-to-sRGB colwersion so\n"
            "images look like the equivalent non-sRGB test.\n"
            "Correct output should look something like a space photo at different mipmap levels.\n";
    } else {
        assert(!"Unknown test mode.");
    }
    return sb.str(); 
}

int LWNTexturePackagerTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(21, 6);
}

void LWNTexturePackagerTest::doGraphics() const
{
    int cellsX = lwrrentWindowWidth / ((m_mode == TEST_TEXPKG) ? cellSize : cellSizeAstc);
    int cellsY = lwrrentWindowHeight / ((m_mode == TEST_TEXPKG) ? cellSize : cellSizeAstc);;
    cellTestInit(cellsX, cellsY);

    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.ClearColor(0, 0.15, 0.15, 0.15, 1.0);

    // Initialize test harness class.
    LWNTexturePackagerHarness texpkgTest(m_mode == TEST_ASTC_SRGB);
    std::vector<unsigned char*> alignedAllocations;

    if (m_mode == TEST_TEXPKG) {
        LWNsizeiptr out_xtx_size_aligned = PoolStorageSize(out_xtx_size);
        unsigned char* out_xtx_aligned = static_cast<unsigned char*>(PoolStorageAlloc(out_xtx_size_aligned));
        memcpy(out_xtx_aligned, out_xtx, out_xtx_size); // From g_lwn_texpkg_data.h
        if (!texpkgTest.initXTXArchive(out_xtx_aligned, out_xtx_size_aligned, 0)) {
            DEBUG_PRINT(("Loading of out_xtx failed\n"));
            LWNFailTest();
            return;
        }
        alignedAllocations.push_back(out_xtx_aligned);

        LWNsizeiptr array_xtx_size_aligned = PoolStorageSize(array_xtx_size);
        unsigned char* array_xtx_aligned = static_cast<unsigned char*>(PoolStorageAlloc(array_xtx_size_aligned));
        memcpy(array_xtx_aligned, array_xtx, array_xtx_size); // From g_lwn_texpkg_data.h
        if (!texpkgTest.initXTXArchive(array_xtx_aligned, array_xtx_size_aligned, 1)) {
            DEBUG_PRINT(("Loading of array_xtx failed\n"));
            LWNFailTest();
            return;
        }
        alignedAllocations.push_back(array_xtx_aligned);


        LWNsizeiptr sparse_xtx_size_aligned = PoolStorageSize(sparse_xtx_size);
        unsigned char* sparse_xtx_aligned = static_cast<unsigned char*>(PoolStorageAlloc(sparse_xtx_size_aligned));
        memcpy(sparse_xtx_aligned, sparse_xtx, sparse_xtx_size); // From g_lwn_texpkg_data.h
        if (!texpkgTest.initXTXArchive(sparse_xtx_aligned, sparse_xtx_size_aligned, 2)) {
            DEBUG_PRINT(("Loading of sparse_xtx failed\n"));
            LWNFailTest();
            return;
        }
        alignedAllocations.push_back(sparse_xtx_aligned);

        LWNsizeiptr minimal_layout_xtx_size_aligned = PoolStorageSize(minimal_layout_xtx_size);
        unsigned char* minimal_layout_xtx_aligned = static_cast<unsigned char*>(PoolStorageAlloc(minimal_layout_xtx_size_aligned));
        memcpy(minimal_layout_xtx_aligned, minimal_layout_xtx, minimal_layout_xtx_size); // From g_lwn_texpkg_data.h
        if (!texpkgTest.initXTXArchive(minimal_layout_xtx_aligned, minimal_layout_xtx_size_aligned, 3)) {
            DEBUG_PRINT(("Loading of minimal_layout_xtx failed\n"));
            LWNFailTest();
            return;
        }
        alignedAllocations.push_back(minimal_layout_xtx_aligned);

        LWNsizeiptr small_xtx_size_aligned = PoolStorageSize(small_xtx_size);
        unsigned char* small_xtx_aligned = static_cast<unsigned char*>(PoolStorageAlloc(small_xtx_size_aligned));
        memcpy(small_xtx_aligned, small_xtx, small_xtx_size); // From g_lwn_texpkg_data.h
        if (!texpkgTest.initXTXArchive(small_xtx_aligned, small_xtx_size_aligned, 4)) {
            DEBUG_PRINT(("Loading of small_xtx failed\n"));
            LWNFailTest();
            return;
        }
        alignedAllocations.push_back(small_xtx_aligned);
    } else {
        LWNsizeiptr astc0_xtx_size_aligned = PoolStorageSize(astc0_xtx_size);
        unsigned char* astc0_xtx_aligned = static_cast<unsigned char*>(PoolStorageAlloc(astc0_xtx_size_aligned));
        memcpy(astc0_xtx_aligned, astc0_xtx, astc0_xtx_size); // From g_lwn_texpkg_astc_data.h
        if (!texpkgTest.initXTXArchive(astc0_xtx_aligned, astc0_xtx_size_aligned, 0)) {
            DEBUG_PRINT(("Loading of astc0_xtx failed\n"));
            LWNFailTest();
            return;
        }
        alignedAllocations.push_back(astc0_xtx_aligned);

        LWNsizeiptr astc1_xtx_size_aligned = PoolStorageSize(astc1_xtx_size);
        unsigned char* astc1_xtx_aligned = static_cast<unsigned char*>(PoolStorageAlloc(astc1_xtx_size_aligned));
        memcpy(astc1_xtx_aligned, astc1_xtx, astc1_xtx_size); // From g_lwn_texpkg_astc_data.h
        if (!texpkgTest.initXTXArchive(astc1_xtx_aligned, astc1_xtx_size_aligned, 0)) {
            DEBUG_PRINT(("Loading of astc1_xtx failed\n"));
            LWNFailTest();
            return;
        }
        alignedAllocations.push_back(astc1_xtx_aligned);

        LWNsizeiptr astc2_xtx_size_aligned = PoolStorageSize(astc2_xtx_size);
        unsigned char* astc2_xtx_aligned = static_cast<unsigned char*>(PoolStorageAlloc(astc2_xtx_size_aligned));
        memcpy(astc2_xtx_aligned, astc2_xtx, astc2_xtx_size); // From g_lwn_texpkg_astc_data.h
        if (!texpkgTest.initXTXArchive(astc2_xtx_aligned, astc2_xtx_size_aligned, 1)) {
            DEBUG_PRINT(("Loading of astc2_xtx failed\n"));
            LWNFailTest();
            return;
        }
        alignedAllocations.push_back(astc2_xtx_aligned);
    }

    if (!texpkgTest.init()) {
        DEBUG_PRINT(("Loading of test assets failed\n"));
        LWNFailTest();
        return;
    }

    // Loop through and display every mipmap level of every texture.
    int numTextures = texpkgTest.numTextures();
    for (int i = 0, k = 0; i < numTextures; i++) {
        int numMipmaps = texpkgTest.numMipmapLevels(i);
        for (int j = 0; j < numMipmaps; j++, k++) {
            DEBUG_PRINT(("    Displaying texture %d mip %d\n", i, j));
            SetCellViewportScissorPadded(queueCB, k % cellsX, k / cellsX, cellMargin);
            texpkgTest.draw(i, j);
        }
    }

    queueCB.submit();
    queue->Finish();

    for (int i = 0; i < (int) alignedAllocations.size(); i++) {
        PoolStorageFree(alignedAllocations[i]);
    }
    alignedAllocations.clear();
}

OGTEST_CppTest(LWNTexturePackagerTest, lwn_texture_packager,           (TEST_TEXPKG));
OGTEST_CppTest(LWNTexturePackagerTest, lwn_texture_packager_astc,      (TEST_ASTC));
OGTEST_CppTest(LWNTexturePackagerTest, lwn_texture_packager_astc_srgb, (TEST_ASTC_SRGB));

