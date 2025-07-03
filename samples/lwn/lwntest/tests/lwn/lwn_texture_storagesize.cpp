/*
* Copyright(c) 2016 LWPU Corporation.All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include "lwn_basic.h"
#include "lwn/lwn_Cpp.h"

#define DEBUG_MODE 0
#if DEBUG_MODE
#define DEBUG_PRINT(x) printf x
#else
#define DEBUG_PRINT(x)
#endif

// Padding bytes will be filled with this magic byte value.
#define PAD_MAGIC 0x3A

// We pad with this amount of bytes before and after the texture.
#define PAD_SIZE 0x10000

// These texture sizes are looped over to test GetStorageSize() for various 1D, 2D, 3D and
// lwbe texture sizes.
// We want to ensure most interesting width / height /depth combinations are covered,
// but looping through every possible one would take forever to run.
// Therefore, the values chosen here are chosen to compromise between coverage and runtime.

// 1D test texture sizes.
#define MAX_SIZE_1D 4036
#define STEP_1D 231

// 2D test texture sizes.
#define MAX_WIDTH_2D 1236
#define WIDTH_STEP_2D 271
#define MAX_HEIGHT_2D 136
#define HEIGHT_STEP_2D 31

// 3D test texture sizes.
#define MAX_WIDTH_3D 434
#define WIDTH_STEP_3D 161
#define MAX_HEIGHT_3D 136
#define HEIGHT_STEP_3D 41
#define MAX_DEPTH_3D 26
#define DEPTH_STEP_3D 8

// Lwbemap test texture sizes.
#define MAX_DIM_LWBE 436
#define DIM_STEP_LWBE 41

// We randomly shuffle the number of mipmap levels per-iteration.
#define MIPMAP_LEVEL_SHUFFLE 0x13F

/* This is (a)/(b) rounded up instead of down */
#ifndef CEIL
#define CEIL(a,b)        (((a)+(b)-1)/(b))
#endif

#ifndef ROUND_UP
#define ROUND_UP(N, S) (CEIL((N),(S)) * (S))
#endif

using namespace lwn;

template<class T>
class LwnScopedObject
{
public:
    LwnScopedObject(T *lwnObj) : m_obj(lwnObj)
    {
        assert(m_obj != nullptr);
    }

    ~LwnScopedObject()
    {
        if (m_obj) {
            m_obj->Free();
        }
    }

    operator bool() const { return (m_obj != nullptr); }

    operator T*() const { return m_obj; }

    T* operator->() const { return m_obj; }

private:
    T  *m_obj;
};

class LWNTexStorageSize
{
    struct TexConfig {
        TextureTarget target;
        Format format;
        int width;
        int height;
        int depth;
        int levels;

        int texDataSize;
        unsigned char* paddedTexMemPoolData;
    };

    bool testIteration(TexConfig& cfg) const;

public:
    LWNTEST_CppMethods();
};

static int TargetMipmapDim(TextureTarget target)
{
    switch (target) {
    case TextureTarget::TARGET_1D_ARRAY:
        // For 1D arrays, only the width is mipmapped; height is the array size.
        return 1;
    case TextureTarget::TARGET_2D_ARRAY:
    case TextureTarget::TARGET_2D_MULTISAMPLE_ARRAY:
    case TextureTarget::TARGET_LWBEMAP_ARRAY:
        // For these targets, only the width and height are mipmapped; depth is the array size.
        return 2;
    default:
        return 3;
    }
}

static int NextMipMapSize(int x)
{
    return (x > 1) ? (x >> 1) : 1;
}

static int NextTargetMipMapSize(TextureTarget target, int dim, int x)
{
    return dim >= TargetMipmapDim(target) ? x : NextMipMapSize(x);
}

static bool TargetSupportsMipmapping(TextureTarget target)
{
    switch (target) {
    case TextureTarget::TARGET_RECTANGLE:
    case TextureTarget::TARGET_BUFFER:
    case TextureTarget::TARGET_2D_MULTISAMPLE_ARRAY:
    case TextureTarget::TARGET_2D_MULTISAMPLE:
        return false;
    default:
        return true;
    }
}

static int GetMaxMipmapLevel(TextureTarget target, int width, int height, int depth)
{
    if (!TargetSupportsMipmapping(target)) {
        return 1;
    }
    int mipLevels = 1;
    int tdim = TargetMipmapDim(target);
    while ((tdim > 0 && width > 1) || (tdim > 1 && height > 1) || (tdim > 2 && depth > 1)) {
        mipLevels++;

        width = NextTargetMipMapSize(target, 0, width);
        height = NextTargetMipMapSize(target, 1, height);
        depth = NextTargetMipMapSize(target, 2, depth);
    }
    return mipLevels;
}

int LWNTexStorageSize::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 9);
}

lwString LWNTexStorageSize::getDescription() const
{
    return
        "Test LWN texture builder GetStorageSize().\n"
        "To test this we loop through various combinations of TextureBuilder parameters,\n"
        "then allocate a padded paddedTexMemPool with the reported storage size. We then blit into the\n"
        "texture, before reading the paddedTexMemPool to verify that the bytes written correspond with the\n"
        "reported storage size.\n";
}

bool LWNTexStorageSize::testIteration(TexConfig& cfg) const
{
    DEBUG_PRINT(("testIteration %d x %d x %d levels %d format 0x%x target 0x%x\n",
        cfg.width, cfg.height, cfg.depth, cfg.levels,
        (int)cfg.format, (int)cfg.target));

    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Create texture texBuilder.
    TextureBuilder texBuilder;
    texBuilder.SetDevice(device)
        .SetDefaults()
        .SetTarget(cfg.target)
        .SetFormat(cfg.format)
        .SetLevels(cfg.levels)
        .SetSize3D(cfg.width, cfg.height, cfg.depth);
    if (FormatIsDepthStencil((LWNformat)(lwn::Format::Enum) cfg.format)) {
        texBuilder.SetFlags(TextureFlags::COMPRESSIBLE);
    }

    // Create buffer builder
    BufferBuilder bufferBuilder;
    bufferBuilder.SetDevice(device);
    bufferBuilder.SetDefaults();

    LWNint blockLinearPageSize;
    size_t texSize;
    size_t paddedTexMemPoolSize;

    device->GetInteger(DeviceInfo::MEMPOOL_TEXTURE_OBJECT_PAGE_ALIGNMENT, &blockLinearPageSize);
    texSize = texBuilder.GetStorageSize();
    if (!texSize) {
        DEBUG_PRINT(("Failed to get texture storage size.\n"));
        return false;
    }
    texSize = ROUND_UP(texSize, blockLinearPageSize);

    // We pad with 64K of bogus data before and after the texture.
    // The reason we pad here is to catch GetStorageSize() under-reporting the texture size;
    // We do this by verifying that the padding bytes aren't modified when we blit to the texture.
    paddedTexMemPoolSize = texSize + 2 * PAD_SIZE;

    // 1. Create the texture.

    if (PAD_SIZE % texBuilder.GetStorageAlignment() != 0) {
        DEBUG_PRINT(("Invalid pad alignment. Please update test.\n"));
        DEBUG_PRINT(("PAD_SIZE 0x%x GetStorageAlignment 0x%x\n", PAD_SIZE, (int)texBuilder.GetStorageAlignment()));
        return false;
    }

    // Create a memory pool to allocate the texture from.
    LwnScopedObject<MemoryPool> paddedTexMemPool(device->CreateMemoryPool(NULL, paddedTexMemPoolSize, MemoryPoolType::CPU_NON_COHERENT));
    if (!paddedTexMemPool) {
        DEBUG_PRINT(("Failed to create paddedTexMemPool.\n"));
        return false;
    }

    // Fill memory pool with magic values
    memset(paddedTexMemPool->Map(), PAD_MAGIC, paddedTexMemPoolSize);
    paddedTexMemPool->FlushMappedRange(0, paddedTexMemPoolSize);

    // Allocate texture from memory pool.
    LwnScopedObject<Texture> texture(texBuilder.CreateTextureFromPool(paddedTexMemPool, PAD_SIZE));
    if (!texture) {
        DEBUG_PRINT(("Failed to create texture from paddedTexMemPool.\n"));
        return false;
    }
    if (texBuilder.GetStorageSize() != (size_t)texture->GetStorageSize()) {
        DEBUG_PRINT(("Size mismatch between texture and texture builder.\n"));
        return false;
    }

    // 2. Fill the texture with white.

    // Figure out pitch-linear texture data size.
    int elementSize = -1;
    int blockWidth = 1;
    int blockHeight = 1;
    switch (cfg.format) {
    case Format::RGBA8:
    case Format::R32I:
        elementSize = 4;
        break;
    case Format::RGBA4:
        elementSize = 2;
        break;
    case Format::DEPTH24_STENCIL8:
        elementSize = 4;
        break;
    case Format::RGB5:
    case Format::RGB5A1:
    case Format::RGB565:
        elementSize = 2;
        break;
    case Format::RGB32F:
        elementSize = 12;
        break;
    case Format::RGB_DXT1:
        elementSize = 8;
        blockWidth = 4;
        blockHeight = 4;
        break;
    default:
        break;
    }
    if (TargetMipmapDim(cfg.target) == 1) {
        // 1D compressed array textures aren't compressed height-wise.
        blockHeight = 1;
    }
    if (elementSize < 0) {
        DEBUG_PRINT(("Unknown format.\n"));
        return false;
    }

    // Create a buffer large enough to store texture level 0.
    // The mipmap levels will be blitted from the same buffer.
    int textureDataSizeLevel0 = CEIL(cfg.width, blockWidth) *
        CEIL(cfg.height, blockHeight) *
        cfg.depth * elementSize *2;
    assert(textureDataSizeLevel0);

    // Create a memory pool to store texture data buffer.
    LwnScopedObject<MemoryPool> tempPool(device->CreateMemoryPool(NULL, textureDataSizeLevel0, MemoryPoolType::CPU_COHERENT));
    if (!tempPool) {
        DEBUG_PRINT(("Could not create memory pool.\n"));
        return false;
    }

    // Create the texture data buffer.
    LwnScopedObject<Buffer> textureDataBuffer(bufferBuilder.CreateBufferFromPool(tempPool, 0, textureDataSizeLevel0));
    void* textureDataBufferPtr = textureDataBuffer->Map();
    if (!textureDataBufferPtr) {
        DEBUG_PRINT(("Could not map texture data buffer\n"));
        return false;
    }

    // Fill the texture data with white pixels.
    memset(textureDataBufferPtr, 0xFF, textureDataSizeLevel0);

    // Copy buffer containing white pixels into the given texture.
    int mipmapLevelWidth = cfg.width;
    int mipmapLevelHeight = cfg.height;
    int mipmapLevelDepth = cfg.depth;
    cfg.texDataSize = 0;

    TextureView view;
    for (int i = 0; i < cfg.levels; i++) {
        CopyRegion region = { 0, 0, 0, mipmapLevelWidth, mipmapLevelHeight, mipmapLevelDepth };
        view.SetDefaults().SetLevels(i, 1);
        queueCB.CopyBufferToTexture(textureDataBuffer->GetAddress(), texture, &view,
            &region, LWN_COPY_FLAGS_NONE);

        cfg.texDataSize += CEIL(mipmapLevelWidth, blockWidth) *
            CEIL(mipmapLevelHeight, blockHeight) *
            mipmapLevelDepth * elementSize;

        mipmapLevelWidth = NextTargetMipMapSize(cfg.target, 0, mipmapLevelWidth);
        mipmapLevelHeight = NextTargetMipMapSize(cfg.target, 1, mipmapLevelHeight);
        mipmapLevelDepth = NextTargetMipMapSize(cfg.target, 2, mipmapLevelDepth);
    }

    queueCB.submit();
    queue->Finish();

    // 3. Verify the contents of the pool.

    // Create a CPU visible pool and blit old pool into cpu-visible pool.
    LwnScopedObject<MemoryPool> cpuMemPool(device->CreateMemoryPool(NULL, paddedTexMemPoolSize, MemoryPoolType::CPU_COHERENT));
    LwnScopedObject<Buffer> buffer(bufferBuilder.CreateBufferFromPool(cpuMemPool, 0, paddedTexMemPoolSize));
    LwnScopedObject<Buffer> bufferSrc(bufferBuilder.CreateBufferFromPool(paddedTexMemPool, 0, paddedTexMemPoolSize));
    queueCB.CopyBufferToBuffer(bufferSrc->GetAddress(), buffer->GetAddress(), paddedTexMemPoolSize, LWN_COPY_FLAGS_NONE);
    queueCB.submit();
    queue->Finish();

    unsigned char* poolData = static_cast<unsigned char*>(buffer->Map());
    if (!poolData) {
        return false;
    }

    buffer->IlwalidateMappedRange(0, paddedTexMemPoolSize);

    // Verify that the first N padded bytes aren't touched.
    for (int i = 0; i < PAD_SIZE; i++) {
        if (poolData[i] != PAD_MAGIC)  {
            DEBUG_PRINT(("Padding byte %u was touched by texture copy.\n", i));
            DEBUG_PRINT(("    Data: %u expected: %u.\n", poolData[i], PAD_MAGIC));
            return false;
        }
    }

    // Verify that the last N padded bytes aren't touched.
    for (int i = 0; i < PAD_SIZE; i++) {
        if (poolData[(paddedTexMemPoolSize - 1) - i] != PAD_MAGIC)  {
            DEBUG_PRINT(("Padding byte %u was touched by texture copy.\n", i));
            DEBUG_PRINT(("    Data: %u expected: %u.\n", poolData[(paddedTexMemPoolSize - 1) - i], PAD_MAGIC));
            DEBUG_PRINT(("    Texture blit wrote to texture data offset %u.\n", ((paddedTexMemPoolSize - 1) - i)));
            DEBUG_PRINT(("    Last byte written should've been %u.\n", (paddedTexMemPoolSize - 1) - PAD_SIZE));
            DEBUG_PRINT(("    Texsize is %u.\n", texSize));
            return false;
        }
    }

    // Verify that the contents of the texture are what we wrote.
    mipmapLevelWidth = cfg.width;
    mipmapLevelHeight = cfg.height;
    mipmapLevelDepth = cfg.depth;

    for (int i = 0; i < cfg.levels; i++) {
        memset(textureDataBufferPtr, 0x0, textureDataSizeLevel0);

        CopyRegion region = { 0, 0, 0, mipmapLevelWidth, mipmapLevelHeight, mipmapLevelDepth };
        view.SetDefaults().SetLevels(i, 1);
        queueCB.CopyTextureToBuffer(texture, &view, &region, textureDataBuffer->GetAddress(),
                                    CopyFlags::NONE);
        queueCB.submit();
        queue->Finish();

        buffer->IlwalidateMappedRange(0, paddedTexMemPoolSize);
        int levelBytes = CEIL(mipmapLevelWidth, blockWidth) *
                         CEIL(mipmapLevelHeight, blockHeight) *
                         mipmapLevelDepth * elementSize;
        for (int j = 0; j < levelBytes; ++j) {
            int texelByte = static_cast<const unsigned char*>(textureDataBufferPtr)[j];
            
            if (texelByte != 0xFF) {
                DEBUG_PRINT(("Texel at level %u, byte offset %u is corrupt.\n", i, j)) 
                DEBUG_PRINT(("    Data: %u expected: %u.\n", 0xFF, texelByte));
                return false;
            }
        }
        mipmapLevelWidth = NextTargetMipMapSize(cfg.target, 0, mipmapLevelWidth);

        mipmapLevelHeight = NextTargetMipMapSize(cfg.target, 1, mipmapLevelHeight);
        mipmapLevelDepth = NextTargetMipMapSize(cfg.target, 2, mipmapLevelDepth);
    }

    return true;
}

void LWNTexStorageSize::doGraphics(void) const
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    int cellSize = 32;
    int cellMargin = 4;
    static const int cellsX = lwrrentWindowWidth / cellSize;
    static const int cellsY = lwrrentWindowHeight / cellSize;
    cellTestInit(cellsX, cellsY);
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.ClearColor(0, 0.45, 0.45, 0.45, 1.0);

    const int numFormats = 8;
    const int numFormatNonCompressed = 7;

    // Few formats to be tested.
    // We cannot afford to test all the combinations of targets, formats and
    // sizes, it just takes too long. So we shuffle just some combos.
    Format formats[numFormats] = {
        Format::RGBA8,
        Format::RGBA4,
        Format::DEPTH24_STENCIL8,
        Format::RGB5A1,
        Format::RGB565,
        Format::R32I,
        Format::RGB32F,
        Format::RGB_DXT1
    };

    unsigned int testCount = 0;
    unsigned int itr = 0;

    // Test 1D texture targets.
    TextureTarget targets1D[] = {
        TextureTarget::TARGET_1D,
        TextureTarget::TARGET_BUFFER
    };
    const int numTargets1D = __GL_ARRAYSIZE(targets1D);

    itr = 0;
    for (int x = 1; x < MAX_SIZE_1D; x += STEP_1D) {
        TexConfig cfg;
        cfg.target = targets1D[itr % numTargets1D];
        cfg.format = formats[itr % numFormatNonCompressed]; // 1D textures can't be compressed.
        cfg.width = x;
        cfg.height = 1;
        cfg.depth = 1;
        cfg.levels = 1 + ((itr * MIPMAP_LEVEL_SHUFFLE) %
            GetMaxMipmapLevel(cfg.target, cfg.width, cfg.height, cfg.depth));

        SetCellViewportScissorPadded(queueCB, testCount % cellsX, testCount / cellsX, cellMargin);
        bool success = testIteration(cfg);
        if (success) {
            queueCB.ClearColor(0, 0.0, 1.0, 0.0, 0.0);
        }
        else {
            DEBUG_PRINT(("Failed: %d x 1 levels %d\n", x, cfg.levels));
            queueCB.ClearColor(0, 1.0, 0.0, 0.0, 0.0);
        }

        itr++;
        testCount++;
    }

    // Test 2D texture targets.
    TextureTarget targets2D[] = {
        TextureTarget::TARGET_2D,
        TextureTarget::TARGET_1D_ARRAY,
        TextureTarget::TARGET_RECTANGLE
    };
    const int numTargets2D = __GL_ARRAYSIZE(targets2D);

    itr = 0;
    for (int x = 1; x < MAX_WIDTH_2D; x += WIDTH_STEP_2D) {
        for (int y = 1; y < MAX_HEIGHT_2D; y += HEIGHT_STEP_2D) {
            TexConfig cfg;
            cfg.target = targets2D[itr % numTargets2D];
            cfg.format = formats[itr % numFormats];
            cfg.width = x;
            cfg.height = y;
            cfg.depth = 1;
            cfg.levels = 1 + ((itr * MIPMAP_LEVEL_SHUFFLE) %
                GetMaxMipmapLevel(cfg.target, cfg.width, cfg.height, cfg.depth));

            if (cfg.target == TextureTarget::TARGET_1D_ARRAY) {
                cfg.format = formats[itr % numFormatNonCompressed]; // 1D textures can't be compressed.
            }

            SetCellViewportScissorPadded(queueCB, testCount % cellsX, testCount / cellsX, cellMargin);
            bool success = testIteration(cfg);
            if (success) {
                queueCB.ClearColor(0, 0.0, 1.0, 0.0, 0.0);
            }
            else {
                DEBUG_PRINT(("Failed: %d x %d levels %d\n", x, y, cfg.levels));
                queueCB.ClearColor(0, 1.0, 0.0, 0.0, 0.0);
            }

            itr++;
            testCount++;
        }
    }

    // Test 3D texture targets.
    TextureTarget targets3D[] = {
        TextureTarget::TARGET_3D,
        TextureTarget::TARGET_2D_ARRAY
    };
    const int numTargets3D = __GL_ARRAYSIZE(targets3D);

    itr = 0;
    for (int x = 1; x < MAX_WIDTH_3D; x += WIDTH_STEP_3D) {
        for (int y = 1; y < MAX_HEIGHT_3D; y += HEIGHT_STEP_3D) {
            for (int z = 1; z < MAX_DEPTH_3D; z += DEPTH_STEP_3D) {
                TexConfig cfg;
                cfg.target = targets3D[(itr+x) % numTargets3D];
                cfg.format = formats[itr % numFormats];
                cfg.width = x;
                cfg.height = y;
                cfg.depth = z;
                cfg.levels = 1 + ((itr * MIPMAP_LEVEL_SHUFFLE) %
                    GetMaxMipmapLevel(cfg.target, cfg.width, cfg.height, cfg.depth));

                SetCellViewportScissorPadded(queueCB, testCount % cellsX, testCount / cellsX, cellMargin);
                bool success = testIteration(cfg);
                if (success) {
                    queueCB.ClearColor(0, 0.0, 1.0, 0.0, 0.0);
                }
                else {
                    DEBUG_PRINT(("Failed: %d x %d x %d levels %d\n", x, y, z, cfg.levels));
                    queueCB.ClearColor(0, 1.0, 0.0, 0.0, 0.0);
                }

                itr++;
                testCount++;
            }
        }
    }

    // Test lwbe texture targets.
    TextureTarget targetsLwbe[] = {
        TextureTarget::TARGET_LWBEMAP,
        TextureTarget::TARGET_LWBEMAP_ARRAY
    };
    const int numTargetsLwbe = __GL_ARRAYSIZE(targetsLwbe);

    itr = 0;
    for (int d = 1; d < MAX_DIM_LWBE; d += DIM_STEP_LWBE) {
        TexConfig cfg;
        cfg.target = targetsLwbe[itr % numTargetsLwbe];
        cfg.format = formats[itr % numFormats];
        cfg.width = d;
        cfg.height = d;
        cfg.depth = 6;
        cfg.levels = 1 + ((itr * MIPMAP_LEVEL_SHUFFLE) %
            GetMaxMipmapLevel(cfg.target, cfg.width, cfg.height, cfg.depth));

        SetCellViewportScissorPadded(queueCB, testCount % cellsX, testCount / cellsX, cellMargin);
        bool success = testIteration(cfg);
        if (success) {
            queueCB.ClearColor(0, 0.0, 1.0, 0.0, 0.0);
        }
        else {
            DEBUG_PRINT(("Failed: %d x %d x %d levels %d\n", d, d, z, cfg.levels));
            queueCB.ClearColor(0, 1.0, 0.0, 0.0, 0.0);
        }

        itr++;
        testCount++;
    }

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNTexStorageSize, lwn_texture_storagesize, );
