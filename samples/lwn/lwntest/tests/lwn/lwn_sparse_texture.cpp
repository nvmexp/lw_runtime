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

// Test for linear textures using sparse storage.
class LWNSparseLinearTest
{
    // We are using a 512x512 RGBA8 texture, which requires 1MB or 16 64KB
    // memory pages.  We've chosen a width so each group of 32 rows fits on
    // a 64KB page.
    static const int bytesPerTexel = 4;
    static const int texSize = 512;
    static const int texStride = texSize * bytesPerTexel;
    static const int texMemorySize = texSize * texSize * bytesPerTexel;

    // Cell sizes used for displaying the textures.
    static const int cellWidth = 320;
    static const int cellHeight = 240;
    static const int cellMargin = 4;

public:
    LWNTEST_CppMethods();
};

lwString LWNSparseLinearTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Basic test for sparse linear textures and render targets.  This test "
        "allocates 2D, rectangle, and buffer textures out of a virtual memory "
        "pool and populates every other page of the pool with memory.  It then "
        "renders a color gradient into the sparse texture and then displays "
        "the results on screen.  The resulting image should have three cells "
        "each displaying horizonal bands of the color gradient.  The cells "
        "are displaying the 2D (lower left), rectangle (lower right), and "
        "buffer (upper left) textures."
        "\n\n"
        "The test also performs a number of queries and will display a bright "
        "red image if any query returns unexpected values.";
    return sb.str();
}

int LWNSparseLinearTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 8);
}

void LWNSparseLinearTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Query properties of the device (page size for sparse and required
    // linear texture alignments).
    LWNint pageSize = -1;
    LWNint linearTexAlignment = -1;
    LWNint linearRTAlignment = -1;
    device->GetInteger(DeviceInfo::MEMORY_POOL_PAGE_SIZE, &pageSize);
    if (pageSize & (pageSize - 1)) {
        LWNFailTest();
        return;
    }
    device->GetInteger(DeviceInfo::LINEAR_TEXTURE_STRIDE_ALIGNMENT, &linearTexAlignment);
    if (linearTexAlignment & (linearTexAlignment - 1)) {
        LWNFailTest();
        return;
    }
    device->GetInteger(DeviceInfo::LINEAR_RENDER_TARGET_STRIDE_ALIGNMENT, &linearRTAlignment);
    if (linearRTAlignment & (linearRTAlignment - 1)) {
        LWNFailTest();
        return;
    }

    // Passthrough vertex shader.
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 color;\n"
        "layout(location=2) in vec2 texcoord;\n"
        "out vec3 ocolor;\n"
        "out vec2 otc;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ocolor = color;\n"
        "  otc = texcoord;\n"
        "}\n";

    // Shader to display a color gradient.
    FragmentShader fsColor(440);
    fsColor <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(ocolor, 1.0);\n"
        "}\n";

    // Shader to display the contents of a 2D texture with a simple texture
    // lookup.
    FragmentShader fsTex2D(440);
    fsTex2D <<
        "layout(binding = 0) uniform sampler2D tex;\n"
        "in vec2 otc;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  vec2 tc = otc;\n"
        "  fcolor = texture(tex, tc);\n"
        "}\n";

    // Shader to display the contents of a rectangle texture with a simple
    // texture lookup.
    FragmentShader fsTexRect(440);
    fsTexRect <<
        "layout(binding = 0) uniform sampler2DRect tex;\n"
        "in vec2 otc;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  vec2 tc = otc * vec2(" << texSize << ");\n"
        "  fcolor = texture(tex, tc);\n"
        "}\n";

    // Shader to display the contents of a buffer texture that holds a
    // <texSize> x <texSize> array of texels, where we map a 2D texture
    // coordinate to a texel.  We do two lookups for adjacent vertical rows to
    // simulate linear filtering.  With the chosen sizes, interpolation
    // appears to produce a razor's edge condition where point sampling will
    // result in some rows near a page boundary partially rounding up and
    // partially rounding down.
    FragmentShader fsTexBuffer(440);
    fsTexBuffer <<
        "layout(binding = 0) uniform samplerBuffer tex;\n"
        "in vec2 otc;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  int index;\n"
        "  vec2 tc = otc * vec2(" << texSize << ");\n"
        "  index = int(tc.y) * " << texSize << " + int(tc.x);\n"
        "  vec4 texel1 = texelFetch(tex, index);\n"
        "  index = (int(tc.y) + 1) * " << texSize << " + int(tc.x);\n"
        "  vec4 texel2 = texelFetch(tex, index);\n"
        "  fcolor = mix(texel1, texel2, fract(tc.y));\n"
        "}\n";

    // Set up all of our programs and bail if any fail to compile.
    LWNboolean status = LWN_TRUE;
    Program *pgmColor = device->CreateProgram();
    status = status && g_glslcHelper->CompileAndSetShaders(pgmColor, vs, fsColor);
    Program *pgmTex2D = device->CreateProgram();
    status = status && g_glslcHelper->CompileAndSetShaders(pgmTex2D, vs, fsTex2D);
    Program *pgmTexRect = device->CreateProgram();
    status = status && g_glslcHelper->CompileAndSetShaders(pgmTexRect, vs, fsTexRect);
    Program *pgmTexBuffer = device->CreateProgram();
    status = status && g_glslcHelper->CompileAndSetShaders(pgmTexBuffer, vs, fsTexBuffer);
    if (!status) {
        LWNFailTest();
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
        dt::vec2 texcoord;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0), dt::vec3(0.0, 0.0, 1.0), dt::vec2(0.0, 0.0) },
        { dt::vec3(-1.0, +1.0, 0.0), dt::vec3(0.0, 1.0, 0.5), dt::vec2(0.0, 1.0) },
        { dt::vec3(+1.0, -1.0, 0.0), dt::vec3(1.0, 0.0, 0.5), dt::vec2(1.0, 0.0) },
        { dt::vec3(+1.0, +1.0, 0.0), dt::vec3(1.0, 1.0, 0.0), dt::vec2(1.0, 1.0) },
    };

    MemoryPoolAllocator allocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, texcoord);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // Compute the sizes needed to hold the texture in the sparse pool.  We
    // allocate a physical pool with memory for half of the texture pages.
    // We allocate a virtual pool that can hold two textures -- the "real"
    // texture, and a fully unpopulated "dummy" texture that can be used to
    // clear out the dummy page mapped into unpopulated portions of virtual
    // pools.
    uint64_t paddedTexSize = pageSize * ((texMemorySize + pageSize - 1) / pageSize);
    uint64_t physicalPoolSize = paddedTexSize / 2;
    uint64_t virtualPoolSize = paddedTexSize * 2;

    // Create physical and virtual memory pools.
    MemoryPool *physicalPool = device->CreateMemoryPoolWithFlags(NULL, physicalPoolSize,
                                                                 (MemoryPoolFlags::CPU_NO_ACCESS |
                                                                  MemoryPoolFlags::GPU_NO_ACCESS |
                                                                  MemoryPoolFlags::PHYSICAL |
                                                                  MemoryPoolFlags::COMPRESSIBLE));
    MemoryPool *virtualPool = device->CreateMemoryPoolWithFlags(NULL, virtualPoolSize,
                                                                (MemoryPoolFlags::CPU_NO_ACCESS |
                                                                 MemoryPoolFlags::GPU_CACHED |
                                                                 MemoryPoolFlags::VIRTUAL));

    // Map storage for every other physical page into the virtual pool.
    LWNuint physicalPoolPages = physicalPoolSize / pageSize;
    MappingRequest *mappings = new MappingRequest[physicalPoolPages];
    for (LWNuint i = 0; i < physicalPoolPages; i++) {
        mappings[i].physicalPool = physicalPool;
        mappings[i].physicalOffset = i * pageSize;
        mappings[i].virtualOffset = 2 * i * pageSize;
        mappings[i].size = pageSize;
        mappings[i].storageClass = LWN_STORAGE_CLASS_BUFFER;
    }
    virtualPool->MapVirtual(physicalPoolPages, mappings);
    delete [] mappings;

    // Set up a basic sampler with default state.
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    Sampler *smp = sb.CreateSampler();

    // Set up a 2D texture from the virtual pool.
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetFormat(Format::RGBA8);
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetSize2D(texSize, texSize);
    tb.SetLevels(1);
    tb.SetFlags(TextureFlags::LINEAR | TextureFlags::LINEAR_RENDER_TARGET);
    tb.SetStride(texStride);
    Texture *tex2D = tb.CreateTextureFromPool(virtualPool, 0);
    TextureHandle tex2DHandle = device->GetTextureHandle(tex2D->GetRegisteredTextureID(), smp->GetRegisteredID());

    // Query the texture builder and texture state to make sure the returned
    // sizes, alignments, and storage classes are correct.
    if (tb.GetStorageSize() != texMemorySize ||
        tb.GetStorageAlignment() != size_t(linearRTAlignment) ||
        tb.GetStorageClass() != LWN_STORAGE_CLASS_BUFFER ||
        tex2D->GetStorageClass() != LWN_STORAGE_CLASS_BUFFER)
    {
        LWNFailTest();
        return;
    }

    // Also check for proper queries for linear textures without the render
    // target bit set.
    tb.SetFlags(TextureFlags::LINEAR);
    if (tb.GetStorageSize() != texMemorySize ||
        tb.GetStorageAlignment() != size_t(linearTexAlignment) ||
        tb.GetStorageClass() != LWN_STORAGE_CLASS_BUFFER)
    {
        LWNFailTest();
        return;
    }
    tb.SetFlags(TextureFlags::LINEAR | TextureFlags::LINEAR_RENDER_TARGET);

    // Set up a second dummy texture in the upper half of the virtual pool
    // that can be used for clearing the dummy page.
    Texture *tex2DDummy = tb.CreateTextureFromPool(virtualPool, paddedTexSize);

    // Set up a rectangle texture from the same memory, and repeat the
    // queries.
    tb.SetTarget(TextureTarget::TARGET_RECTANGLE);
    Texture *texRect = tb.CreateTextureFromPool(virtualPool, 0);
    TextureHandle texRectHandle = device->GetTextureHandle(texRect->GetRegisteredTextureID(), smp->GetRegisteredID());
    if (tb.GetStorageSize() != texMemorySize ||
        tb.GetStorageAlignment() != size_t(linearRTAlignment) ||
        tb.GetStorageClass() != LWN_STORAGE_CLASS_BUFFER ||
        texRect->GetStorageClass() != LWN_STORAGE_CLASS_BUFFER)
    {
        LWNFailTest();
        return;
    }

    // Set up a buffer texture from the same memory and repeat the queries.
    tb.SetTarget(TextureTarget::TARGET_BUFFER);
    tb.SetSize1D(texSize * texSize);
    tb.SetFlags(TextureFlags(0));
    Texture *texBuffer = tb.CreateTextureFromPool(virtualPool, 0);
    TextureHandle texBufferHandle = device->GetTexelFetchHandle(texBuffer->GetRegisteredTextureID());
    if (tb.GetStorageSize() != texMemorySize ||
        tb.GetStorageAlignment() != bytesPerTexel ||
        tb.GetStorageClass() != LWN_STORAGE_CLASS_BUFFER ||
        texBuffer->GetStorageClass() != LWN_STORAGE_CLASS_BUFFER)
    {
        LWNFailTest();
        return;
    }

    // Now render to the 2D texture using our color gradient shader.  This
    // will only fill the populated portion of the texture; the remainder
    // should be black.
    queueCB.SetRenderTargets(1, &tex2D,  NULL, NULL, NULL);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    queueCB.BindProgram(pgmColor, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.SetViewportScissor(0, 0, texSize, texSize);
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

    // On hardware not supporting zero reads from unpopulated pages, we just
    // scribbled on the dummy page.  Clear the dummy texture to black to
    // scribble on it and set it back to black.
    if (!g_lwnDeviceCaps.supportsZeroFromUndefinedMappedPoolPages) {
        queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS);
        queueCB.SetRenderTargets(1, &tex2DDummy, NULL, NULL, NULL);
        queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    }

    // Wait for the rendering before displaying the texture.
    queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS | BarrierBits::ILWALIDATE_TEXTURE);

    // Switch back to the main render target, and display our three textures
    // (all sharing the same memory we just rendered to).
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);

    queueCB.SetViewportScissor(cellMargin, cellMargin,
                               cellWidth - 2 * cellMargin, cellHeight - 2 * cellMargin);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, tex2DHandle);
    queueCB.BindProgram(pgmTex2D, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

    queueCB.SetViewportScissor(cellWidth + cellMargin, cellMargin,
                               cellWidth - 2 * cellMargin, cellHeight - 2 * cellMargin);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texRectHandle);
    queueCB.BindProgram(pgmTexRect, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

    queueCB.SetViewportScissor(cellMargin, cellHeight + cellMargin,
                               cellWidth - 2 * cellMargin, cellHeight - 2 * cellMargin);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texBufferHandle);
    queueCB.BindProgram(pgmTexBuffer, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queueCB.submit();
    queue->Finish();

    // Cleanup resources before unmapping and finalizing
    // the virtual pool.
    tex2D->Free();
    tex2DDummy->Free();
    texRect->Free();
    texBuffer->Free();
    virtualPool->Free();
}

OGTEST_CppTest(LWNSparseLinearTest, lwn_sparse_texture_linear, );


//////////////////////////////////////////////////////////////////////////

// Test for regular block linear textures using sparse storage and "tiled"
// layouts.
class LWNSparseTextureTest
{
    // Allocate enough virtual memory for 16 layers of 1024x1024 RGBA32
    // textures with mipmapping.  That's 16MB per layer plus another ~1/3 for
    // mipmaps.  We allocate 24MB per layer x 16 layers.  Since these textures
    // will be only sparsely populated, allocate less physical memory to avoid
    // fragmentation (bug 1751322).
    static const int vboPoolSize = 1 * 1024 * 1024;
    static const int physicalPoolSize = 16 * 1024 * 1024;
    static const int virtualPoolSize = 384 * 1024 * 1024;

    // 2D textures in this test should be no bigger than 1024x1024, and arrays
    // can have up to 16 layers.  3D textures should be no bigger than
    // 256x256x256.
    static const int tex2DSize = 1024;
    static const int tex2DLevels = 11;
    static const int tex2DLayers = 16;
    static const int tex3DSize = 256;
    static const int tex3DLevels = 9;

    // To emulate "sparse reads zero" on older textures, we set up a dummy
    // texture and use it to clear the unpopulated "dummy" page to zero after
    // we may have scribbled on it.
    static const int dummyTexSize = 256;

    // We render in 70x70 cells (64x64 plus margin), to fit 9 columns in a
    // 640x480 image.  That tests three different formats, each folwsing on
    // one of three layers (near, far, middle) of multi-layer textures.
    static const int cellSize = 70;
    static const int cellMargin = 3;
    static const int nTestFormats = 3;
    static const int maxTestFolwsLayers = 3;

    // Only the three lowest layers get a full-resolution cell; other smaller
    // layers are displayed at half-resolution to fit on-screen.
    static const int nFullResCells = 3;

    // Pad a memory allocation to the physical page size.
    static size_t padToPageSize(size_t size, LWNuint pageSize) {
        return pageSize * ((size + pageSize - 1) / pageSize);
    }

    // For 3D textures, the number of layers shrinks when going to smaller
    // mipmaps.  This function picks an "equivalent" mipmap layer for mipmap
    // level <level>, that either selects near/far/middle for that layer.
    static LWNuint remap3DTexLayer(LWNuint level, LWNuint inLayer, LWNuint texDepth)
    {
        LWNuint mipDepth = texDepth >> level;
        if (mipDepth <= 1) {
            return 0;
        }
        if (inLayer >= texDepth - 1) {
            return mipDepth - 1;
        } else if (inLayer >= texDepth / 2) {
            return mipDepth / 2;
        } else {
            return 0;
        }
    }

    TextureTarget m_target;
    int m_formatSet;
    bool m_compressible;

public:
    LWNSparseTextureTest(TextureTarget target, int formatSet, bool compressible) :
        m_target(target), m_formatSet(formatSet), m_compressible(compressible) {}
    LWNTEST_CppMethods();
};

lwString LWNSparseTextureTest::getDescription() const
{
    const char *target = "bogus";
    switch (m_target) {
    case TextureTarget::TARGET_2D:          target = "2D"; break;
    case TextureTarget::TARGET_2D_ARRAY:    target = "2D array"; break;
    case TextureTarget::TARGET_3D:          target = "3D"; break;
    default:                                assert(0); break;
    }

    lwStringBuf sb;
    sb << "This test exercises partially populated sparse " << target << " textures.  ";
    if (m_target == TextureTarget::TARGET_2D) {
        sb <<  "Each column displays a different format.";
    } else {
        sb << 
            "Each set of three columns displays a different format, with each "
            "column in the set folwsing on a different layer (near, middle, far).";
    }
    sb <<
        "  Each column displays a set of mipmap layers stacked vertically.  We render "
        "gradients into mipmap layers with red increasing from left to right, "
        "green from bottom to top, and blue from near layer to far layer.  All "
        "colors darken when moving from the high-resolution mip (bottom) to the "
        "lowest-resolution mip (top).";
    if (!g_lwnDeviceCaps.supportsZeroFromUndefinedMappedPoolPages &&
        m_target == TextureTarget::TARGET_2D_ARRAY) {
        sb <<
            "  This device doesn't support full sparse tiling of array textures, "
            "so the texture is truncated to eliminate small mipmap levels.";
    }
    sb <<
        "\n\n"
        "The cells for set of mipmaps that can be decomposed into tiles (not in the "
        "'tail' are highlighted with a green margin.  For those cells, we populate "
        "only portions of the mipmap in a diagonal pattern; unpopulated portions are "
        "in black.  The set of tiles populated is shifted based on the layer being "
        "displayed.  The lowest-resolution non-tail mipmap will appear solid since "
        "it will have only one tile vertically and/or horizontally.  All mipmaps in "
        "the tail will be solid gradients."
        "\n\n"
        "The formats supported by this test (left to right) are:  ";
    switch (m_formatSet) {
    case 0:     sb << "R8, RG8, and RGBA8"; break;
    case 1:     sb << "RGB10A2, RGBA16F, and RGBA32F"; break;
    default:    assert(0); break;
    }
    sb <<
        ".  The texture will be allocated " << (m_compressible ? "with" : "without") <<
        "support for framebuffer compression.";

    return sb.str();    
}

int LWNSparseTextureTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(40, 9);
}

void LWNSparseTextureTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Set up the set of formats to exercise in this test.
    Format testTexFormats[nTestFormats];
    switch (m_formatSet) {
    case 0:
        testTexFormats[0] = Format::R8;
        testTexFormats[1] = Format::RG8;
        testTexFormats[2] = Format::RGBA8;
        break;
    case 1:
        testTexFormats[0] = Format::RGB10A2;
        testTexFormats[1] = Format::RGBA16F;
        testTexFormats[2] = Format::RGBA32F;
        break;
    default:
        assert(0);
    };

    // Query the GPU page size to use for subsequent tests.
    LWNint pageSize;
    device->GetInteger(DeviceInfo::MEMORY_POOL_PAGE_SIZE, &pageSize);

    // Set up the texture size, level count, and the number of layers to focus
    // on based on the target.  GPUs that don't support full sparse texturing
    // have different rules for mipmap tails, and layered (array textures)
    // don't work if any mipmap level is smaller than the tile size.  So for
    // those devices, clamp the level count to 3 (256x256) for the array
    // texture tests.
    dt::ivec3 testTexSize(0, 0, 0);
    int testTexLevelCount = 1;
    LWNuint testTexFolwsLayers[3] = { 0, 0, 0 };
    LWNuint testTexFolwsLayerCount = 1;
    switch (m_target) {
    case TextureTarget::TARGET_2D:
        testTexSize = dt::ivec3(tex2DSize, tex2DSize, 1);
        testTexLevelCount = tex2DLevels;
        break;
    case TextureTarget::TARGET_2D_ARRAY:
        testTexSize = dt::ivec3(tex2DSize, tex2DSize, tex2DLayers);
        if (g_lwnDeviceCaps.supportsZeroFromUndefinedMappedPoolPages) {
            testTexLevelCount = tex2DLevels;
        } else {
            testTexLevelCount = 3;
        }
        testTexFolwsLayers[0] = 0;
        testTexFolwsLayers[1] = tex2DLayers / 2;
        testTexFolwsLayers[2] = tex2DLayers - 1;
        testTexFolwsLayerCount = 3;
        break;
    case TextureTarget::TARGET_3D:
        testTexSize = dt::ivec3(tex3DSize, tex3DSize, tex3DSize);
        testTexLevelCount = tex3DLevels;
        testTexFolwsLayers[0] = 0;
        testTexFolwsLayers[1] = tex3DSize / 2;
        testTexFolwsLayers[2] = tex3DSize - 1;
        testTexFolwsLayerCount = 3;
        break;
    default:
        assert(0);
    }

    // Set up a vertex shader that passes through position, color, and a
    // texture coordinate.
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 color;\n"
        "layout(location=2) in vec4 texcoord;\n"
        "out Attributes {\n"
        "  vec3 color;\n"
        "  vec4 texcoord;\n"
        "} ov;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ov.color = color;\n"
        "  ov.texcoord = texcoord;\n"
        "}\n";

    // Set up a fragment shader that passes through color from the vertex
    // shader (used to set up sparse textures).
    FragmentShader fsColor(440);
    fsColor <<
        "in Attributes {\n"
        "  vec3 color;\n"
        "  vec4 texcoord;\n"
        "} f;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(f.color, 1.0);\n"
        "}\n";

    // Set up a fragment shader to read from a texture of the appropriate
    // target using passed-through coordinates, with the level of detail
    // specified in the "w" texture coordinate.
    static const char *samplerType = "bogus";
    switch (m_target) {
    case TextureTarget::TARGET_2D:
        samplerType = "sampler2D";
        break;
    case TextureTarget::TARGET_2D_ARRAY:
        samplerType = "sampler2DArray";
        break;
    case TextureTarget::TARGET_3D:
        samplerType = "sampler3D";
        break;
    default:
        assert(0);
        break;
    }
    FragmentShader fsTex(440);
    fsTex <<
        "layout(binding = 0) uniform " << samplerType << " s;\n"
        "in Attributes {\n"
        "  vec3 color;\n"
        "  vec4 texcoord;\n"
        "} f;\n"
        "out vec4 fcolor;\n"
        "void main() {\n";
    switch (m_target) {
    case TextureTarget::TARGET_2D:
        fsTex << "  vec2 tc = f.texcoord.xy;\n";
        break;
    case TextureTarget::TARGET_2D_ARRAY:
    case TextureTarget::TARGET_3D:
        fsTex << "  vec3 tc = f.texcoord.xyz;\n";
        break;
    default:
        assert(0);
        break;
    }
    fsTex <<
        "  fcolor = textureLod(s, tc, f.texcoord.w);\n"
        "}\n";

    // Set up program objects to do color and texture rendering.
    Program *colorPgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(colorPgm, vs, fsColor)) {
        printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
        LWNFailTest();
        return;
    }

    Program *texPgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(texPgm, vs, fsTex)) {
        printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
        LWNFailTest();
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
        dt::vec4 texcoord;
    };
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, texcoord);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    MemoryPool *vboPool = device->CreateMemoryPoolWithFlags(NULL, vboPoolSize,
                                                            (MemoryPoolFlags::CPU_UNCACHED |
                                                             MemoryPoolFlags::GPU_CACHED));
    BufferAddress vboAddr = vboPool->GetBufferAddress();
    Vertex *vboPtr = (Vertex *) vboPool->Map();
    Vertex *v = vboPtr;

    // Set up VBO data for displaying the mipmaps and folwsed layers.  We
    // generate a quad for each layer/mipmap level combination tested.
    BufferAddress drawTexVBO = vboAddr;
    for (LWNuint flayer = 0; flayer < testTexFolwsLayerCount; flayer++) {
        for (int mipmap = 0; mipmap < testTexLevelCount; mipmap++) {
            for (int i = 0; i < 4; i++) {
                float layer = testTexFolwsLayers[flayer];
                if (m_target == TextureTarget::TARGET_3D) {
                    layer = remap3DTexLayer(mipmap, layer, testTexSize[2]);
                    layer = (layer + 0.5) / (testTexSize[2] >> mipmap);
                }
                v->position = dt::vec3((i & 2) ? +1.0 : -1.0, (i & 1) ? +1.0 : -1.0, 0.0);
                v->texcoord = dt::vec4((i & 2) ? 1.0 : 0.0, (i & 1) ? 1.0 : 0.0, layer, mipmap);
                v->color = dt::vec3(0);
                v++;
            }
        }
    }
    LWNuint drawTexVBOSize = testTexFolwsLayerCount * testTexLevelCount * 4 * sizeof(Vertex);
    vboAddr += drawTexVBOSize;

    // Set up VBO data for displaying gradients in the texture.  Each
    // layer/mipmap combination gets a different set of colors.  Red varies
    // from left to right; green from bottom to top; blue from near to far.
    // All color components get darker as we move to smaller mipmaps.
    BufferAddress drawColorVBO = vboAddr;
    for (LWNuint flayer = 0; flayer < testTexFolwsLayerCount; flayer++) {
        for (int mipmap = 0; mipmap < testTexLevelCount; mipmap++) {
            float cscale = 0.3 + 0.7 * (float(testTexLevelCount - mipmap) / testTexLevelCount);
            for (int i = 0; i < 4; i++) {
                v->position = dt::vec3((i & 2) ? +1.0 : -1.0, (i & 1) ? +1.0 : -1.0, 0.0);
                v->texcoord = dt::vec4(0.0);
                v->color = dt::vec3(cscale * ((i & 2) ? 0.8 : 0.2),
                                    cscale * ((i & 1) ? 0.7 : 0.2),
                                    cscale * 0.2 + 0.8 * testTexFolwsLayers[flayer] / testTexSize[2]);
                v++;
            }
        }
    }
    LWNuint drawColorVBOSize = testTexFolwsLayerCount * testTexLevelCount * 4 * sizeof(Vertex);
    vboAddr += drawColorVBOSize;

    // Set up memory pools for the sparse texture.
    MemoryPool *physicalPool = device->CreateMemoryPoolWithFlags(NULL, physicalPoolSize,
                                                                 (MemoryPoolFlags::CPU_NO_ACCESS |
                                                                  MemoryPoolFlags::GPU_NO_ACCESS |
                                                                  MemoryPoolFlags::PHYSICAL |
                                                                  MemoryPoolFlags::COMPRESSIBLE));
    MemoryPool *virtualPool = device->CreateMemoryPoolWithFlags(NULL, virtualPoolSize,
                                                                (MemoryPoolFlags::CPU_NO_ACCESS |
                                                                 MemoryPoolFlags::GPU_CACHED |
                                                                 MemoryPoolFlags::VIRTUAL));

    // Set up a sampler to use for rendering.
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetMinMagFilter(MinFilter::LINEAR_MIPMAP_LINEAR, MagFilter::LINEAR);
    sb.SetWrapMode(WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE);
    Sampler *smp = sb.CreateSampler();

    // Set up a basic texture builder we'll used for subsequent queries and
    // texture creation.
    TextureBuilder texBuilder;
    texBuilder.SetDevice(device);
    texBuilder.SetDefaults();
    texBuilder.SetTarget(m_target);
    texBuilder.SetSize3D(testTexSize[0], testTexSize[1], testTexSize[2]);
    texBuilder.SetLevels(testTexLevelCount);
    if (m_compressible) {
        texBuilder.SetFlags(TextureFlags::SPARSE | TextureFlags::COMPRESSIBLE);
    } else {
        texBuilder.SetFlags(TextureFlags::SPARSE);
    }

    // Create a second texture builder for a dummy texture to work around
    // pre-Maxwell sparse dummy page issues.
    TextureBuilder tbDummy;
    tbDummy.SetDevice(device);
    tbDummy.SetDefaults();
    tbDummy.SetTarget(TextureTarget::TARGET_2D);
    tbDummy.SetSize2D(dummyTexSize, dummyTexSize);
    tbDummy.SetLevels(1);

    // Set up a basic texture view for selecting levels.
    TextureView texView;
    TextureView *texViews[] = { &texView };
    texView.SetDefaults();

    queueCB.ClearColor(0, 0, 0, 0, 0);
    queueCB.BindVertexArrayState(vertex);

    int col = 0;

    // Loop over formats being exercised by this test.
    for (int fidx = 0; fidx < nTestFormats; fidx++)  {

        // Set up the builder for the appropriate format.
        Format fmt = testTexFormats[fidx];
        texBuilder.SetFormat(fmt);

        // Query the amount of storage required by the texture and pad to the page size.
        size_t totalTexSize = padToPageSize(texBuilder.GetStorageSize(), pageSize);
        assert(totalTexSize < virtualPoolSize);

        // Query the sparse tile layout for the texture.
        TextureSparseTileLayout layout;
        texBuilder.GetSparseTileLayout(&layout);

        // Create the real texture at the beginning of the sparse pool and set
        // up a handle.
        Texture *tex = texBuilder.CreateTextureFromPool(virtualPool, 0);
        TextureHandle texHandle = device->GetTextureHandle(tex->GetRegisteredTextureID(), smp->GetRegisteredID());

        // Set up the sparse mapping for the texture.
        LWNstorageClass storageClass = tex->GetStorageClass();
        MappingRequest mappings[1024];
        MappingRequest *mapping = mappings;
        LWNuint nMappings = 0;

        // Figure out the number of bytes in each layer to compute virtual
        // addresses to be used when setting up a given layer.
        size_t layerSize;
        if (m_target == TextureTarget::TARGET_2D_ARRAY) {
            texView.SetDefaults();
            texView.SetLayers(1, 1);
            layerSize = tex->GetViewOffset(&texView);
        } else {
            layerSize = totalTexSize;
        }

        // Loop over the layers that we want to exercise in the test.  For
        // each tested layer, we will map in the entire tail plus a diagonal
        // pattern of tiles in the non-tail portions.  The rest of the texture
        // will be unpopulated.
        size_t physicalOffset = 0;
        for (LWNuint flayer = 0; flayer < testTexFolwsLayerCount; flayer++) {
            LWNuint layer = testTexFolwsLayers[flayer];

            // Start by mapping the entire tail, if there are any tail levels.
            // For 3D textures, we map the entire tail once regardless of the
            // number of layers to focus on.
            if (layout.numTiledLevels < testTexLevelCount) {
                if (flayer == 0 || m_target != TextureTarget::TARGET_3D) {

                    // Query the offset of the first level in the tail (for
                    // the selected layer).
                    texView.SetDefaults();
                    texView.SetLevels(layout.numTiledLevels, 0);
                    if (m_target == TextureTarget::TARGET_2D_ARRAY) {
                        texView.SetLayers(layer, 1);
                    }
                    LWNuint tailOffset = tex->GetViewOffset(&texView);
                    LWNuint alignedTailOffset = pageSize * (tailOffset / pageSize);
                    assert(layout.numTiledLevels == 0 || tailOffset == alignedTailOffset);
                    size_t alignedTailEnd = padToPageSize((layer+1) * layerSize, pageSize);

                    // Set up a mapping structure to map in the entire tail
                    // range.
                    mapping->physicalPool = physicalPool;
                    mapping->physicalOffset = physicalOffset;
                    mapping->virtualOffset = alignedTailOffset;
                    mapping->size = alignedTailEnd - alignedTailOffset;
                    mapping->storageClass = storageClass;
                    physicalOffset += mapping->size;
                    mapping++;
                    nMappings++;
                }
            }

            // Loop over all non-tail levels and set up a diagonal pattern of
            // single pages in each in each.
            for (int level = 0; level < layout.numTiledLevels; level++) {

                // Figure out the number of tiles in X and Y plus the larger
                // of the two values.
                LWNuint tilesX = (testTexSize[0] >> level) / layout.tileWidth;
                LWNuint tilesY = (testTexSize[1] >> level) / layout.tileHeight;
                LWNuint maxXYTiles = (tilesX > tilesY) ? tilesX : tilesY;

                // For 3D textures, we need to figure out a "brick row" for
                // the tile.  If tiles are 32x32x32, layers 0..31 are all in
                // the same brick (tz==0), while layers 32..63 are in the next
                // one (tz==1).
                LWNuint tz = 0;
                if (m_target == TextureTarget::TARGET_3D) {
                    int tileLayer = remap3DTexLayer(level, layer, testTexSize[2]);
                    tz = tileLayer / layout.tileDepth;
                }

                // Find the starting offset of the mipmap level.
                texView.SetDefaults();
                texView.SetLevels(level, 1);
                if (m_target == TextureTarget::TARGET_2D_ARRAY) {
                    texView.SetLayers(layer, 1);
                }
                LWNuint mipOffset = tex->GetViewOffset(&texView);

                // Loop over tiles in the larger of the X and/or Y dimensions,
                // setting up a diagonal pattern.  The diagonal pattern start
                // is shifted by the layer number.
                for (LWNuint j = 0; j < maxXYTiles; j++) {
                    int zshift = (m_target == TextureTarget::TARGET_3D) ? tz : flayer;
                    int tx = j % tilesX;
                    int ty = (j + zshift) % tilesY;
                    LWNuint tileOffset = mipOffset + pageSize * ((tz * tilesY + ty) * tilesX + tx);
                    mapping->physicalPool = physicalPool;
                    mapping->physicalOffset = physicalOffset;
                    mapping->virtualOffset = tileOffset;
                    mapping->size = pageSize;
                    mapping->storageClass = storageClass;
                    physicalOffset += mapping->size;
                    mapping++;
                    nMappings++;
                }
            }
        }
        assert(physicalOffset <= physicalPoolSize);
        virtualPool->MapVirtual(nMappings, mappings);

        // Create a dummy texture of the same format from the virtual pool and
        // program it to be completely unpopulated.  This will be used for
        // clearing the "dummy page" on devices not supporting full zero-ing
        // behavior.
        tbDummy.SetFormat(fmt);
        ptrdiff_t dummyTexOffset = totalTexSize;
        size_t dummyTexMemSize = padToPageSize(tbDummy.GetStorageSize(), pageSize);
        assert(dummyTexOffset + dummyTexMemSize <= virtualPoolSize);
        Texture *dummyTex = tbDummy.CreateTextureFromPool(virtualPool, dummyTexOffset);
        MappingRequest dummyMapping;
        dummyMapping.physicalPool = NULL;
        dummyMapping.physicalOffset = 0;
        dummyMapping.virtualOffset = dummyTexOffset;
        dummyMapping.size = dummyTexMemSize;
        dummyMapping.storageClass = storageClass;
        virtualPool->MapVirtual(1, &dummyMapping);

        // Render each of the texture levels and layers with a different
        // shade.  First clear the whole texture to red (should be
        // overwritten) and then render our gradients.
        queueCB.BindProgram(colorPgm, ShaderStageBits::ALL_GRAPHICS_BITS);
        queueCB.BindVertexBuffer(0, drawColorVBO, drawColorVBOSize);
        for (int level = 0; level < testTexLevelCount; level++) {
            queueCB.SetViewportScissor(0, 0, testTexSize[0] >> level, testTexSize[1] >> level);
            texView.SetDefaults();
            texView.SetLevels(level, 1);
            queueCB.SetRenderTargets(1, &tex, texViews, NULL, NULL);
            queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
            for (LWNuint flayer = 0; flayer < testTexFolwsLayerCount; flayer++) {
                LWNuint layer = testTexFolwsLayers[flayer];
                if (m_target == TextureTarget::TARGET_3D) {
                    layer = remap3DTexLayer(level, layer, testTexSize[2]);
                }
                texView.SetDefaults();
                texView.SetLevels(level, 1);
                texView.SetLayers(layer, 1);
                queueCB.SetRenderTargets(1, &tex, texViews, NULL, NULL);
                queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, (flayer * testTexLevelCount + level) * 4, 4);
            }
        }

        // On pre-Maxwell2 GPUs, clear the (completely unmapped) dummy
        // texture, which will zero out the dummy page used for unpopulated
        // portions of the real texture.
        if (!g_lwnDeviceCaps.supportsZeroFromUndefinedMappedPoolPages) {
            queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS);
            queueCB.SetRenderTargets(1, &dummyTex, NULL, NULL, NULL);
            queueCB.SetViewportScissor(0, 0, dummyTexSize, dummyTexSize);
            queueCB.ClearColor(0, 0, 0, 0, 0);
        }

        // Insert a render-to-texture barrier.
        queueCB.Barrier(BarrierBits::ILWALIDATE_TEXTURE | BarrierBits::ORDER_FRAGMENTS);

        // Now display the tested layers and mipmap levels of the texture.
        g_lwnWindowFramebuffer.bind();
        queueCB.BindVertexBuffer(0, drawTexVBO, drawTexVBOSize);
        queueCB.BindProgram(texPgm, ShaderStageBits::ALL_GRAPHICS_BITS);
        queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
        for (LWNuint flayer = 0; flayer < testTexFolwsLayerCount; flayer++) {
            for (int level = 0; level < testTexLevelCount; level++) {

                // For the biggest levels, we draw full-size cells stacked
                // vertically.  For smaller levels, we continue to stack
                // vertically but draw a half-size cells so we can see all the
                // mipmap levels.  (The 1x1 level on hangs slightly off the
                // top.)
                int cx, cy, cs, cmw, cmh;
                if (level < nFullResCells) {
                    cx = col * cellSize + cellMargin;
                    cy = level * cellSize + cellMargin;
                    cs = cellSize - 2 * cellMargin;
                    cmw = cellMargin;
                    cmh = cellMargin;
                } else {
                    cx = col * cellSize + cellMargin + cellSize / 4;
                    cy = nFullResCells * cellSize + (level - nFullResCells) * cellSize / 2 + cellMargin / 2;
                    cs = cellSize / 2 - cellMargin;
                    cmw = cellMargin;
                    cmh = cellMargin / 2;
                }

                // For levels not in the tail, render a green background
                // slightly larger than the cell to make a "halo".
                if (level < layout.numTiledLevels) {
                    queueCB.SetViewportScissor(cx - cmw, cy - cmw, cs + 2 * cmw, cs + 2 * cmh);
                    queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
                }

                queueCB.SetViewportScissor(cx, cy, cs, cs);
                queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, (flayer * testTexLevelCount + level) * 4, 4);
            }
            col++;
        }
    
        // Wait for rendering to complete before messing with the page mappings.
        queueCB.submit();
        queue->Finish();

        // Erase all of the virtual mappings by redoing all the previous
        // mappings using a NULL physical pool.
        for (LWNuint i = 0; i < nMappings; i++) {
            mappings[i].physicalPool = NULL;
            mappings[i].physicalOffset = 0;
        }
        virtualPool->MapVirtual(nMappings, mappings);

    }
}

OGTEST_CppTest(LWNSparseTextureTest, lwn_sparse_texture_2d_f0, (TextureTarget::TARGET_2D, 0, false));
OGTEST_CppTest(LWNSparseTextureTest, lwn_sparse_texture_2d_f1, (TextureTarget::TARGET_2D, 1, false));
OGTEST_CppTest(LWNSparseTextureTest, lwn_sparse_texture_2da_f0, (TextureTarget::TARGET_2D_ARRAY, 0, false));
OGTEST_CppTest(LWNSparseTextureTest, lwn_sparse_texture_2da_f1, (TextureTarget::TARGET_2D_ARRAY, 1, false));
OGTEST_CppTest(LWNSparseTextureTest, lwn_sparse_texture_3d_f0, (TextureTarget::TARGET_3D, 0, false));
OGTEST_CppTest(LWNSparseTextureTest, lwn_sparse_texture_3d_f1, (TextureTarget::TARGET_3D, 1, false));

OGTEST_CppTest(LWNSparseTextureTest, lwn_sparse_texture_compr_2d_f0, (TextureTarget::TARGET_2D, 0, true));
OGTEST_CppTest(LWNSparseTextureTest, lwn_sparse_texture_compr_2d_f1, (TextureTarget::TARGET_2D, 1, true));
OGTEST_CppTest(LWNSparseTextureTest, lwn_sparse_texture_compr_2da_f0, (TextureTarget::TARGET_2D_ARRAY, 0, true));
OGTEST_CppTest(LWNSparseTextureTest, lwn_sparse_texture_compr_2da_f1, (TextureTarget::TARGET_2D_ARRAY, 1, true));
OGTEST_CppTest(LWNSparseTextureTest, lwn_sparse_texture_compr_3d_f0, (TextureTarget::TARGET_3D, 0, true));
OGTEST_CppTest(LWNSparseTextureTest, lwn_sparse_texture_compr_3d_f1, (TextureTarget::TARGET_3D, 1, true));

//////////////////////////////////////////////////////////////////////////

class LWNSparseTextureTileSizeTest
{
    static const int virtualPoolSize = 256 * 1024 * 1024;
    static const int tex2DSize = 1024;
    static const int tex3DSize = 256;
    static const int nSampleCounts = 4;

    static bool checkLayout(const TextureSparseTileLayout &layout, const dt::ivec2 &expected)
    {
        return
            (layout.tileWidth  == expected[0] &&
             layout.tileHeight == expected[1] &&
             layout.tileDepth  == 1);
    }

    static bool checkLayout(const TextureSparseTileLayout &layout, const dt::ivec3 &expected)
    {
        return
            (layout.tileWidth  == expected[0] &&
             layout.tileHeight == expected[1] &&
             layout.tileDepth  == expected[2]);
    }

public:
    LWNTEST_CppMethods();
};

lwString LWNSparseTextureTileSizeTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Verifies that that tile sizes for sparse textures of various formats "
        "match the LWN programming guide.  Clears the screen to green if we get "
        "all matches, and red if any configuration mismatches.";
    return sb.str();    
}

int LWNSparseTextureTileSizeTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(44, 0);
}

void LWNSparseTextureTileSizeTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Create a virtual pool that we can use for creating textures.  We never
    // map any pages or render with the textures.
    MemoryPool *virtualPool = 
        device->CreateMemoryPoolWithFlags(NULL, virtualPoolSize,
                                          (MemoryPoolFlags::CPU_NO_ACCESS |
                                           MemoryPoolFlags::GPU_CACHED |
                                           MemoryPoolFlags::VIRTUAL));

    // For 2D textures, we loop over supported sample counts.
    int sampleCounts[] = { 0, 2, 4, 8 };
    ct_assert(__GL_ARRAYSIZE(sampleCounts) == nSampleCounts);

    // We loop over all bit counts and create 3D and 2D textures with the
    // representative formats.  This table holds all the expected tile sizes.
    struct ExpectedTileSize {
        Format      format;
        dt::ivec3   size3D;
        dt::ivec2   size2D[nSampleCounts];
    } tileSizeTests[] = {
        {
            Format::R8, dt::ivec3(64, 32, 32),
            { dt::ivec2(256, 256), dt::ivec2(128, 256), dt::ivec2(128, 128), dt::ivec2(64, 128) }
        },
        {
            Format::RG8, dt::ivec3(32, 32, 32),
            { dt::ivec2(256, 128), dt::ivec2(128, 128), dt::ivec2(128, 64), dt::ivec2(64, 64) }
        },
        {
            Format::RGBA8, dt::ivec3(32, 32, 16),
            { dt::ivec2(128, 128), dt::ivec2(64, 128), dt::ivec2(64, 64), dt::ivec2(32, 64) }
        },
        {
            Format::RGBA16F, dt::ivec3(32, 16, 16),
            { dt::ivec2(128, 64), dt::ivec2(64, 64), dt::ivec2(64, 32), dt::ivec2(32, 32) }
        },
        {
            Format::RGBA32F, dt::ivec3(16, 16, 16),
            { dt::ivec2(64, 64), dt::ivec2(32, 64), dt::ivec2(32, 32), dt::ivec2(16, 32) }
        },
    };

    bool result = true;

    for (uint32_t i = 0; i < __GL_ARRAYSIZE(tileSizeTests); i++) {
        TextureBuilder tb;
        TextureSparseTileLayout layout;
        Texture *tex;

        // Set up the texture builder for a 3D texture, create the texture,
        // and query tile sizes using both the builder and the texture itself.
        tb.SetDefaults();
        tb.SetDevice(device);
        tb.SetTarget(TextureTarget::TARGET_3D);
        tb.SetFormat(tileSizeTests[i].format);
        tb.SetFlags(TextureFlags::SPARSE);
        tb.SetSize3D(tex3DSize, tex3DSize, tex3DSize);
        tb.GetSparseTileLayout(&layout);
        if (!checkLayout(layout, tileSizeTests[i].size3D)) {
            result = false;
        }

        tex = tb.CreateTextureFromPool(virtualPool, 0);
        tex->GetSparseTileLayout(&layout);
        if (!checkLayout(layout, tileSizeTests[i].size3D)) {
            result = false;
        }
        tex->Free();

        // Set up the texture builder for a 2D texture with differnent sample
        // counts, create the texture, and query tile sizes using both the
        // builder and the texture itself.
        for (int j = 0; j < nSampleCounts; j++) {
            if (sampleCounts[j]) {
                tb.SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE);
            } else {
                tb.SetTarget(TextureTarget::TARGET_2D);
            }
            tb.SetSize2D(tex2DSize, tex2DSize);
            tb.SetSamples(sampleCounts[j]);
            tb.GetSparseTileLayout(&layout);
            if (!checkLayout(layout, tileSizeTests[i].size2D[j])) {
                result = false;
            }

            tex = tb.CreateTextureFromPool(virtualPool, 0);
            tex->GetSparseTileLayout(&layout);
            if (!checkLayout(layout, tileSizeTests[i].size2D[j])) {
                result = false;
            }
            tex->Free();
        }
    }

    // Clear to green or red depending on whether we had any mismatches.
    if (result) {
        queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
    } else {
        queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
    }
    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();

    virtualPool->Free();
}

OGTEST_CppTest(LWNSparseTextureTileSizeTest, lwn_sparse_tile_size, );

//////////////////////////////////////////////////////////////////////////

class LWNSparseTextureCompressedTest
{
    static const int numConfigs = 3;
    bool mCompressiblePools;
public:
    LWNSparseTextureCompressedTest(bool compressiblePools)
        : mCompressiblePools(compressiblePools)
    {}
    LWNTEST_CppMethods();
};

lwString LWNSparseTextureCompressedTest::getDescription() const
{
    lwStringBuf sb;
    sb << "Test creates sparse textures of 2x2 tiles, with block-compressed format.\n"
          "Two diagonal tiles are mapped to the same memory, and each format\n"
          "is displayed. Output is " << numConfigs*2 << " solid green\n"
          "rectangles in zig-zag pattern.\n";
    return sb.str();
}

int LWNSparseTextureCompressedTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 19);
}

void LWNSparseTextureCompressedTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    Format testConfigs[numConfigs] = {
        Format::RGB_DXT1,
        Format::RGBA_DXT3,
        Format::RGBA_DXT5,
    };

    int cellW = lwrrentWindowWidth / numConfigs;
    int cellH = lwrrentWindowHeight;
    int pageSize = 0;
    device->GetInteger(DeviceInfo::MEMORY_POOL_PAGE_SIZE, &pageSize);

    // program
    Program* pgmDisplay;
    VertexShader vs(430);
    FragmentShader fs(430);
    vs <<
        "out vec2 uv;\n"
        "void main() {\n"
        "  vec2 pos = vec2(0.0);\n"
        "  if (gl_VertexID == 0) pos = vec2(-1.0, -1.0);\n"
        "  if (gl_VertexID == 1) pos = vec2(1.0, -1.0);\n"
        "  if (gl_VertexID == 2) pos = vec2(1.0, 1.0);\n"
        "  if (gl_VertexID == 3) pos = vec2(-1.0, 1.0);\n"
        "  gl_Position = vec4(pos, 0.0, 1.0);\n"
        "  uv = pos * 0.5 + 0.5;\n"
        "}\n";
    fs <<
        "layout(binding=0) uniform sampler2D tex;\n"
        "in vec2 uv;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = texture(tex, uv);\n"
        "}\n";
    pgmDisplay = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(pgmDisplay, vs, fs);

    // nearest sampler
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    Sampler* nearestSampler = sb.CreateSampler();

    // setup and clear
    BlendState bs;
    bs.SetDefaults();
    bs.SetBlendFunc(BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA, BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA);
    queueCB.BindBlendState(&bs);
    ColorState cs;
    cs.SetDefaults();
    cs.SetBlendEnable(0, 1);
    queueCB.BindColorState(&cs);
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);
    queueCB.submit();

    // texture
    for (int cfgIdx=0; cfgIdx<numConfigs; cfgIdx++) {
        Texture* tex;
        TextureHandle texHandle;
        Buffer* texBuffer;
        MemoryPool* virtualPool;
        MemoryPool* texGpuPool;
        MemoryPool* texCpuPool;

        TextureBuilder tb;
        tb.SetDevice(device).SetDefaults();
        tb.SetTarget(TextureTarget::TARGET_2D);
        tb.SetFormat(testConfigs[cfgIdx]);
        tb.SetFlags(TextureFlags::SPARSE);
        // get tile size
        tb.SetSize2D(1024, 1024);
        TextureSparseTileLayout tstl;
        tb.GetSparseTileLayout(&tstl);
        // make texture with total of 4 tiles
        int texW = tstl.tileWidth * 2;
        int texH = tstl.tileHeight * 2;
        int halfTexW = texW/2;
        int halfTexH = texH/2;
        int blocksHorz = halfTexW/4;
        int blocksVert = halfTexH/4;
        tb.SetSize2D(texW, texH);

        // create pools
        MemoryPoolFlags cf = mCompressiblePools ? MemoryPoolFlags::COMPRESSIBLE : 0;
        size_t tbStorageSize = tb.GetStorageSize();
        texCpuPool = device->CreateMemoryPoolWithFlags(NULL, tbStorageSize, (MemoryPoolFlags::CPU_UNCACHED |
                                                                             MemoryPoolFlags::GPU_CACHED |
                                                                             cf));
        texGpuPool = device->CreateMemoryPoolWithFlags(NULL, tbStorageSize, (MemoryPoolFlags::CPU_NO_ACCESS |
                                                                             MemoryPoolFlags::GPU_NO_ACCESS |
                                                                             MemoryPoolFlags::PHYSICAL |
                                                                             cf));
        virtualPool = device->CreateMemoryPoolWithFlags(NULL, tbStorageSize,
                                                        (MemoryPoolFlags::CPU_NO_ACCESS |
                                                         MemoryPoolFlags::GPU_CACHED |
                                                         MemoryPoolFlags::VIRTUAL));
        MappingRequest mrs[2];
        for (int mrIdx = 0; mrIdx<2; mrIdx++) {
            MappingRequest& mr  = mrs[mrIdx];
            mr.physicalOffset   = 0;
            mr.virtualOffset    = pageSize * mrIdx * 3;
            mr.physicalPool     = texGpuPool;
            mr.size             = pageSize;
            mr.storageClass     = tb.GetStorageClass();
        }
        virtualPool->MapVirtual(2, mrs);

        // create and fill texture
        tex = tb.CreateTextureFromPool(virtualPool, 0);
        BufferBuilder bb;
        bb.SetDevice(device).SetDefaults();
        texBuffer = bb.CreateBufferFromPool(texCpuPool, 0, tbStorageSize);
        uint64_t* texDataPtr = static_cast<uint64_t*>(texBuffer->Map());

        switch (tb.GetFormat()) {
        case Format::RGB_DXT1:
            for (int i=0; i<blocksVert; i++) {
                for (int j=0; j<blocksHorz; j++) {
                    // 64b per 4x4 block
                    // 64b: color indices (4x4 x 2b = 32b) + boundary colors (R5G6B5 x2 = 32b)
                    texDataPtr[j+i*blocksHorz] = 0x1111111107E007E0;  // green
                }
            }
            break;
        case Format::RGBA_DXT3:
            for (int i=0; i<blocksVert; i++) {
                for (int j=0; j<blocksHorz; j++) {
                    // 128b per 4x4 block
                    // 64b: alpha mask (4x4 x 4b)
                    // 64b: color indices (4x4 x 2b = 32b) + boundary colors (R5G6B5 x2 = 32b)
                    texDataPtr[(j+i*blocksHorz)*2] =   0xFFFFFFFFFFFFFFFF;
                    texDataPtr[(j+i*blocksHorz)*2+1] = 0x1111111107E007E0;
                }
            }
            break;
        case Format::RGBA_DXT5:
            for (int i=0; i<blocksVert; i++) {
                for (int j=0; j<blocksHorz; j++) {
                    // 128b per 4x4 block
                    // 64b: alpha indices (4x4 x 3b = 48b) + alpha boundary values (8b x2 = 16b)
                    // 64b: 4x4 color indices (32bits) + R5G6B5 x2 boundary colors (32bits)
                    texDataPtr[(j+i*blocksHorz)*2] =   0x111111111111FFFF;
                    texDataPtr[(j+i*blocksHorz)*2+1] = 0x1111111107E007E0;
                }
            }
            break;
        default:
            printf("\nUnsupported format.\n");
            break;
        }
        CopyRegion cr = {0, 0, 0, halfTexW, halfTexH, 1};
        queueCB.CopyBufferToTexture(texBuffer->GetAddress(), tex, 0, &cr, CopyFlags::NONE);
        Sync* sync = device->CreateSync();
        queue->FenceSync(sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, SyncFlagBits::FLUSH_FOR_CPU);
        queue->WaitSync(sync);
        sync->Free();
        texHandle = device->GetTextureHandle(tex->GetRegisteredTextureID(), nearestSampler->GetRegisteredID());
        texBuffer->Free();

        // draw
        queueCB.SetViewportScissor(cellW * cfgIdx, 0, cellW, cellH);
        queueCB.BindProgram(pgmDisplay, ShaderStageBits::ALL_GRAPHICS_BITS);
        queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
        queueCB.submit();
        queue->Finish();

        // cleanup
        tex->Free();
        virtualPool->Free();
        texGpuPool->Free();
        texCpuPool->Free();
    }

    // cleanup
    pgmDisplay->Free();
    nearestSampler->Free();
}

OGTEST_CppTest(LWNSparseTextureCompressedTest, lwn_sparse_compressed, (false));
OGTEST_CppTest(LWNSparseTextureCompressedTest, lwn_sparse_compressed_compressible_pool, (true));
