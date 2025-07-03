/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_c.h"
#include "lwn_utils.h"
#include "../../elw/cmdline.h"

#include <time.h>

//////////////////////////////////////////////////////////////////////////

static int offscreenWidth = 100, offscreenHeight = 100;

class LWNMemoryPoolTest
{
    static const int cellSize = 20;
    static const int cellMargin = 1;
    static const int cellsX = 10;
    static const int cellsY = 10;

public:
    enum MemPoolTestVariant 
    {
        TEST_VARIANT_RT,
        TEST_VARIANT_RT_Z,
        TEST_VARIANT_TEX_FROM_POOL,
        TEST_VARIANT_BUF_FROM_POOL,
        TEST_VARIANT_BUFFER_COPY,
        TEST_VARIANT_PRE_BAKED,
        TEST_VARIANT_COHERENT,
        TEST_VARIANT_NON_COHERENT,
        TEST_VARIANT_GPU_ONLY,
        TEST_VARIANT_POOL_SIZE,
        TEST_VARIANT_LAST
    };
    enum TestType
    {
        TT_TEXTURE,
        TT_BUFFER
    };

    LWNTEST_CppMethods();

private:
    void drawResult(CellIterator2D& cell, int result) const;

    LWNtexture *createRenderTarget(LWNmemoryPool **pool) const;

    // sub-tests
    int simpleColorRenderTargetFromPool(bool useZ) const;            // create and clear a color/Z rendertarget in pool
    int simpleFromPool(enum LWNMemoryPoolTest::TestType type) const; // test texture or buffer from pool
    int simpleBufferToBufferFromPool() const;                        // test buffer to buffer copies in pool
    int simplePrebakedAsset() const;                                 // create asset and create a pool from it
    int simpleCoherentPool() const;                                  // test coherent pool
    int simpleNonCoherentPool() const;                               // test non-coherent pool
    int simpleGpuOnlyPool() const;                                   // test GPU only pool
    int simplePoolSizes() const;                                     // test all pool types allocation with varying sizes
    int simplePoolSizesCpp() const;
};

int LWNMemoryPoolTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(11, 0);
}

lwString LWNMemoryPoolTest::getDescription() const
{
    return "Simple tests for lwnMemoryPool.\n"
           "Each subtest's result is rendered as a green/red square.\n"
           "Subtests:\n"
           "* simpleColorRenderTargetFromPool*:create a render target (w/ and W/o Z) and clear to color from different pool types (* called once each for non-Z and Z)\n"
           "* simpleBufferToBufferFromPool:    buffer copies in different pool types\n"
           "* simpleFromPool*:                 create and use buffer or texture from different pool types (* called once each for buffers or textures)\n"
           "* simplePrebakedAsset:             create a prebaked asset and create another pool from a pointer to it\n"
           "* simpleCoherentPool:              test coherent pool mapping/flushing/ilwalidating/texture capability\n"
           "* simpleNonCoherentPool:           test non-coherent pool mapping/flushing/ilwalidating/texture capability\n"
           "* simpleGpuOnlyPool:               test GPU only pool mapping/flushing/ilwalidating/texture capability\n"
           "* simplePoolSizes:                 create differently sized pool with different pool types\n";
}

// ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

// show result as red/green square
void LWNMemoryPoolTest::drawResult(CellIterator2D& cell, int result) const
{
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;

    g_lwnWindowFramebuffer.bind();

    LWNfloat color[] = { 0.0, 0.0, 0.0, 1.0 };
    if (result != 0) {
        // shade of red
        color[0] = 1.0 + (0.1 * result);
    } else {
        // green
        color[1] = 1.0;
    }
    // scissor
    lwnCommandBufferSetScissor(queueCB, cell.x() * cellSize + cellMargin, cell.y() * cellSize + cellMargin,
                      cellSize - 2*cellMargin, cellSize - 2*cellMargin);

    // clear
    lwnCommandBufferClearColor(queueCB, 0, color, LWN_CLEAR_COLOR_MASK_RGBA);

    queueCB.submit();

    // advance in grid
    cell++;
}


// ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

LWNtexture *LWNMemoryPoolTest::createRenderTarget(LWNmemoryPool **pool) const
{
    LWNdevice *device = g_lwnDevice;

    LWNtextureBuilder *textureBuilder = lwnDeviceCreateTextureBuilder(device);

    lwnTextureBuilderSetDefaults(textureBuilder);
    lwnTextureBuilderSetFlags(textureBuilder, LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT);
    lwnTextureBuilderSetSize2D(textureBuilder, offscreenWidth, offscreenHeight);
    lwnTextureBuilderSetTarget(textureBuilder, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetSamples(textureBuilder, 0);
    lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_RGBA8);

    *pool = lwnDeviceCreateMemoryPool(device, NULL, lwnTextureBuilderGetStorageSize(textureBuilder), LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    LWNtexture *rtTex = lwnTextureBuilderCreateTextureFromPool(textureBuilder, *pool, /*offset*/0);

    lwnTextureBuilderFree(textureBuilder);

    return rtTex;
}

// ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

// simpleColorRenderTargetFromPool: create a render target and clear to color from different pool types
int LWNMemoryPoolTest::simpleColorRenderTargetFromPool(bool useZ) const
{
    LWNdevice *device = g_lwnDevice;
    LWNqueue *queue = g_lwnQueue;
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;
    struct _poolTypes 
    {
        LWNmemoryPoolFlags  poolFlags;
        bool                mappable;
    }poolTypes[] = 
    { 
#if defined (LW_TEGRA)
        { LWN_MEMORY_POOL_TYPE_CPU_COHERENT,        true}, // coherent pool allows no textures on OS other than HOS
#endif
        { LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT,    true},
        { LWN_MEMORY_POOL_TYPE_GPU_ONLY,            false}
    };

    int result = 0;
    for (size_t poolType = 0; poolType < __GL_ARRAYSIZE(poolTypes); poolType++) {
        LWNtextureBuilder *textureBuilder = lwnDeviceCreateTextureBuilder(device);

        /// create a GPU pool for textures
        //
        LWNmemoryPool *memPool = lwnDeviceCreateMemoryPool(device, NULL, 0x100000, poolTypes[poolType].poolFlags);

        LWNtexture *rtTex = NULL;

        lwnTextureBuilderSetDefaults(textureBuilder);
        lwnTextureBuilderSetFlags(textureBuilder, LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT);
        lwnTextureBuilderSetSize2D(textureBuilder, offscreenWidth, offscreenHeight);
        lwnTextureBuilderSetTarget(textureBuilder, LWN_TEXTURE_TARGET_2D);
        lwnTextureBuilderSetSamples(textureBuilder, 0);
        lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_RGBA8);

        // offset needs to be manually adjusted if you change any of the other allocations from this pool
        // (overlap etc.)
        ptrdiff_t offset = 0;
        rtTex = lwnTextureBuilderCreateTextureFromPool(textureBuilder, memPool, offset);
        offset += lwnTextureBuilderGetStorageSize(textureBuilder);

        LWNtexture *depthTex = NULL;

        if (useZ) {
            lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_DEPTH32F);
            offset = AlignSize(offset, lwnTextureBuilderGetStorageAlignment(textureBuilder));
            depthTex = lwnTextureBuilderCreateTextureFromPool(textureBuilder, memPool, offset);
            offset += lwnTextureBuilderGetStorageSize(textureBuilder);
        }

        lwnCommandBufferSetRenderTargets(queueCB, 1, &rtTex, NULL, depthTex, NULL);

        // scissor
        lwnCommandBufferSetScissor(queueCB, 0, 0, offscreenWidth, offscreenHeight);

        // clear
        LWNfloat clearColor[] = {0, 1, 0, 1};
        lwnCommandBufferClearColor(queueCB, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
        if (useZ) {
            lwnCommandBufferClearDepthStencil(queueCB, 1, LWN_TRUE, 0, 0);
        }
        queueCB.submit();

        LWNbufferBuilder *bufferBuilder = lwnDeviceCreateBufferBuilder(device);
        lwnBufferBuilderSetDefaults(bufferBuilder);

        LWNcopyRegion offscreenRegion = { 0, 0, 0, offscreenWidth, offscreenHeight, 1 };

        LWNmemoryPool *coherent_mempool = lwnDeviceCreateMemoryPool(device, NULL,  offscreenWidth*offscreenHeight*4, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
        LWNbuffer *readbo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, coherent_mempool, /*offset*/0, offscreenWidth*offscreenHeight*4);
        LWNbufferAddress readboAddr = lwnBufferGetAddress(readbo);
        lwnCommandBufferCopyTextureToBuffer(queueCB, rtTex, NULL, &offscreenRegion, readboAddr, LWN_COPY_FLAGS_NONE);
        queueCB.submit();
        lwnQueueFinish(queue);  
        unsigned int* texdata = (unsigned int*)lwnBufferMap(readbo);

        result -= (texdata[0] == 0xFF00FF00) ? 0 : -1;

        if (useZ) {
            LWNbuffer *readZbo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, coherent_mempool, /*offset*/0, offscreenWidth*offscreenHeight*sizeof(float));
            LWNbufferAddress readZboAddr = lwnBufferGetAddress(readZbo);
            lwnCommandBufferCopyTextureToBuffer(queueCB, depthTex, NULL, &offscreenRegion, readZboAddr, LWN_COPY_FLAGS_NONE);
            queueCB.submit();
            lwnQueueFinish(queue);  
            float* depthdata = (float*)lwnBufferMap(readZbo);

            result -= (depthdata[0] == 1.0f) ? 0 : -1;

            lwnBufferFree(readZbo);
            lwnTextureFree(depthTex);
        }

        lwnTextureBuilderFree(textureBuilder);
        lwnTextureFree(rtTex);
        lwnBufferFree(readbo);
        lwnBufferBuilderFree(bufferBuilder);
        lwnMemoryPoolFree(coherent_mempool);
        lwnMemoryPoolFree(memPool);
    }

    return result;
}

// ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

// simpleFromPool: render buffer/texture from different pool types
int LWNMemoryPoolTest::simpleFromPool(enum LWNMemoryPoolTest::TestType type) const
{
    static const char *vsstring =
        "#version 440 core\n"
        "layout(location = 0) in vec4 position;\n"
        "out vec4 tcout;\n"
        "void main() {\n"
        "  gl_Position = position;\n"
        "  tcout = (position + vec4(1.0, 1.0, 0.0, 0.0)) / vec4(2.0, 2.0, 1.0, 1.0);\n"
        "}\n";

    static const char *fsstring = 
        "#version 440 core\n"
        "layout(binding = 0) uniform sampler2D tex;\n"
        "layout(location = 0) out vec4 outColor;\n"
        "out vec4 tcin;\n"
        "void main() {\n"
        "  outColor = texture(tex, tcin.xy);\n"
        "}\n";

    LWNdevice *device = g_lwnDevice;
    LWNqueue *queue = g_lwnQueue;
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;

    // Create programs from the device, provide them shader code and compile/link them
    LWNprogram *pgm = lwnDeviceCreateProgram(device);

    VertexShader vs(440);
    FragmentShader fs(440);
    vs << vsstring;
    fs << fsstring;

    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        return -1;
    }


    struct _poolTypes 
    {
        LWNmemoryPoolFlags  poolFlags;
        bool                mappable;
        bool                texture;
    }poolTypes[] = 
    { 
#if defined(LW_TEGRA)
        { LWN_MEMORY_POOL_TYPE_CPU_COHERENT,        true,   true}, // coherent pool allowed on HOS
#else
        { LWN_MEMORY_POOL_TYPE_CPU_COHERENT,        true,   false}, // coherent pool allows no textures on non-HOS platforms
#endif
        { LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT,    true,   true },
        { LWN_MEMORY_POOL_TYPE_GPU_ONLY,            false,  true }
    };

    int result = 0;
    for (size_t poolType = 0; poolType < __GL_ARRAYSIZE(poolTypes); poolType++) {
        // skip certain pools for texture test
        if ( (type == TT_TEXTURE) && !poolTypes[poolType].texture) {
            continue;
        }

        LWNtextureBuilder *textureBuilder = lwnDeviceCreateTextureBuilder(device);

        // create a GPU pool for textures
        // arbitrary size (=1MB) here
        const int memSize = 0x100000;
        LWNmemoryPool *memPool = lwnDeviceCreateMemoryPool(device, NULL, memSize, poolTypes[poolType].poolFlags);

        LWNmemoryPool *rendertarget_pool;
        LWNtexture *rtTex = createRenderTarget(&rendertarget_pool);

        lwnCommandBufferSetRenderTargets(queueCB, 1, &rtTex, NULL, NULL, NULL);

        const int texWidth = 4, texHeight = 4;
        lwnTextureBuilderSetDefaults(textureBuilder);
        lwnTextureBuilderSetTarget(textureBuilder, LWN_TEXTURE_TARGET_2D);
        lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_RGBA8);
        lwnTextureBuilderSetSize2D(textureBuilder, texWidth, texHeight);

        // attach texture, initialise with green color (or such) and draw with it
        LWNmemoryPool *texture_mempool = NULL;
        if (type != TT_TEXTURE) {
            texture_mempool = lwnDeviceCreateMemoryPool(device, NULL, lwnTextureBuilderGetStorageSize(textureBuilder), LWN_MEMORY_POOL_TYPE_GPU_ONLY);
        }
        LWNtexture *tex = lwnTextureBuilderCreateTextureFromPool(textureBuilder, (type == TT_TEXTURE) ? memPool : texture_mempool, /*offset*/0);
        LWNuint textureID = lwnTextureGetRegisteredTextureID(tex);

        LWNbufferBuilder *bufferBuilder = lwnDeviceCreateBufferBuilder(device);

        lwnBufferBuilderSetDefaults(bufferBuilder);

        LWNmemoryPool *coherent_mempool = lwnDeviceCreateMemoryPool(device, NULL, texWidth*texHeight*4, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
        LWNmemoryPool *copy_mempool = NULL;
        LWNbuffer *pbo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, (type == TT_BUFFER) ? memPool : coherent_mempool, /*offset*/0, texWidth*texHeight*4);
        LWNbufferAddress pboAddr = lwnBufferGetAddress(pbo);
        unsigned char *texdata;
        LWNbuffer *copybo = NULL;
        LWNbufferAddress copyboAddr  = 0;
        if (poolTypes[poolType].mappable) {
            texdata = (unsigned char *)lwnBufferMap(pbo);
        } else {
            copy_mempool = lwnDeviceCreateMemoryPool(device, NULL, texWidth*texHeight*4, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
            copybo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, copy_mempool, /*offset*/0, texWidth*texHeight*4);
            copyboAddr = lwnBufferGetAddress(copybo);
            texdata = (unsigned char *)lwnBufferMap(copybo);
        }
        // fill with texture data
        for (int j = 0; j < texWidth; ++j) {
            for (int i = 0; i < texHeight; ++i) {
                texdata[4*(j*texWidth+i)+0] = 0x00;
                texdata[4*(j*texWidth+i)+1] = 0xFF;
                texdata[4*(j*texWidth+i)+2] = 0x00;
                texdata[4*(j*texWidth+i)+3] = 0xFF;
            }
        }
        if (!poolTypes[poolType].mappable) {
            lwnCommandBufferCopyBufferToBuffer(queueCB, copyboAddr, pboAddr, texWidth*texHeight * 4, LWN_COPY_FLAGS_NONE);
        }

        // we've possibly touched memory in a non-coherent buffer
        lwnMemoryPoolFlushMappedRange(memPool, 0, memSize);

        // Download the texture data
        LWNcopyRegion downloadRegion = { 0, 0, 0, texWidth, texHeight, 1 };
        lwnCommandBufferCopyBufferToTexture(queueCB, pboAddr, tex, NULL, &downloadRegion, LWN_COPY_FLAGS_NONE);

        LWLwertexAttribState attribState;
        lwlwertexAttribStateSetDefaults(&attribState);
        lwlwertexAttribStateSetFormat(&attribState, LWN_FORMAT_RG32F, 0);
        lwlwertexAttribStateSetStreamIndex(&attribState, 0);

        LWLwertexStreamState streamState;
        lwlwertexStreamStateSetDefaults(&streamState);
        lwlwertexStreamStateSetStride(&streamState, 8);

        // a quad
        static LWNfloat vertexData[] = {-1.0f, -1.0f,  
                                        -1.0f,  1.0f,  
                                         1.0f,  1.0f,  
                                         1.0f, -1.0f};

        lwnBufferBuilderSetDefaults(bufferBuilder);
        LWNmemoryPool *vbo_mempool = lwnDeviceCreateMemoryPool(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
        LWNbuffer *vbo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, vbo_mempool, /*offset*/0, sizeof(vertexData));
        // create persistent mapping
        void *ptr = lwnBufferMap(vbo);
        // fill ptr with vertex data followed by color data
        memcpy(ptr, vertexData, sizeof(vertexData));

        // Get a handle to be used for setting the buffer as a vertex buffer
        LWNbufferAddress vboAddr = lwnBufferGetAddress(vbo);

        // sampler
        LWNsamplerBuilder *samplerBuilder = lwnDeviceCreateSamplerBuilder(device);
        LWNsampler *sampler = lwnSamplerBuilderCreateSampler(samplerBuilder);
        LWNuint samplerID = lwnSamplerGetRegisteredID(sampler);

        // combined texture/sampler handle
        LWNtextureHandle texHandle = lwnDeviceGetTextureHandle(device, textureID, samplerID);

        // scissor
        lwnCommandBufferSetScissor(queueCB, 0, 0, offscreenWidth, offscreenHeight);

        // clear
        LWNfloat clearColor[] = {0, 1, 0, 1};
        lwnCommandBufferClearColor(queueCB, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);

        // bind vertex state, texture and sampler
        lwnCommandBufferBindVertexAttribState(queueCB, 1, &attribState);
        lwnCommandBufferBindVertexStreamState(queueCB, 1, &streamState);
        lwnCommandBufferBindTexture(queueCB, LWN_SHADER_STAGE_FRAGMENT, 0, texHandle);
        lwnCommandBufferBindProgram(queueCB, pgm, LWN_SHADER_STAGE_ALL_GRAPHICS_BITS);
        lwnCommandBufferBindVertexBuffer(queueCB, 0, vboAddr, sizeof(vertexData));
        lwnCommandBufferDrawArrays(queueCB, LWN_DRAW_PRIMITIVE_QUADS, 0, 4);
        queueCB.submit();

        lwnBufferBuilderSetDefaults(bufferBuilder);
        LWNmemoryPool *read_mempool = lwnDeviceCreateMemoryPool(device, NULL, offscreenWidth*offscreenHeight*4, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
        LWNbuffer *readbo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, read_mempool, /*offset*/0, offscreenWidth*offscreenHeight*4);
        LWNbufferAddress readboAddr = lwnBufferGetAddress(readbo);
        LWNcopyRegion copyRegion = { 0, 0, 0, offscreenWidth, offscreenHeight, 1 };
        lwnCommandBufferCopyTextureToBuffer(queueCB, rtTex, NULL, &copyRegion, readboAddr, LWN_COPY_FLAGS_NONE);
        queueCB.submit();
        lwnQueueFinish(queue);  
        unsigned int* resdata = (unsigned int *)lwnBufferMap(readbo);

        result -= (resdata[0] == 0xFF00FF00) ? 0 : -1;

        // cleanup
        lwnBufferBuilderFree(bufferBuilder);
        lwnBufferFree(vbo);
        lwnBufferFree(pbo);
        lwnBufferFree(readbo);
        if (copybo) {
            lwnBufferFree(copybo);
        }
        lwnTextureBuilderFree(textureBuilder);
        lwnTextureFree(tex);
        lwnTextureFree(rtTex);
        lwnSamplerBuilderFree(samplerBuilder);
        lwnSamplerFree(sampler);

        lwnMemoryPoolFree(read_mempool);
        lwnMemoryPoolFree(vbo_mempool);
        lwnMemoryPoolFree(memPool);
        if(copy_mempool) {
            lwnMemoryPoolFree(copy_mempool);
        }
        if (texture_mempool) {
            lwnMemoryPoolFree(texture_mempool);
        }
        lwnMemoryPoolFree(rendertarget_pool);
    }
    lwnProgramFree(pgm);

    return (result != 0) ? result - 1 : 0;
}

// ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

// simpleBufferToBufferFromPool: copy buffers around in memory pools of different type
int LWNMemoryPoolTest::simpleBufferToBufferFromPool() const
{
    LWNdevice *device = g_lwnDevice;
    LWNqueue *queue = g_lwnQueue;
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;

    // these values are arbitrarily chosen since we don't have 
    // APIs that allow us to query constraints related to the values like
    // alignment, max. sizes etc.)
    const int memPages = 16;
    const int pageSize = (64*1024);
    const int memSize =  pageSize * memPages; 

    struct _poolTypes 
    {
        LWNmemoryPoolFlags  poolFlags;
        bool                mappable;
        bool                texture;
    }poolTypes[] = 
    { 
#if defined(LW_TEGRA)
        { LWN_MEMORY_POOL_TYPE_CPU_COHERENT,        true,   true}, // coherent pool allowed on HOS
#else
        { LWN_MEMORY_POOL_TYPE_CPU_COHERENT,        true,   false}, // coherent pool allows no textures on non-HOS platforms
#endif
        { LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT,    true,   true },
        { LWN_MEMORY_POOL_TYPE_GPU_ONLY,            false,  true }
    };

    LWNbufferBuilder *bufferBuilder = lwnDeviceCreateBufferBuilder(device);


    const int bufferW = 16;

    LWNmemoryPool *coherent_mempool = lwnDeviceCreateMemoryPool(device, NULL, 2*bufferW + 32, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // copy from a mappable buffer to a memory pool buffer to a mappable buffer and check values
    lwnBufferBuilderSetDefaults(bufferBuilder);
    LWNbuffer *src_bo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, coherent_mempool, /*offset*/0, bufferW);
    LWNbufferAddress src_bo_addr = lwnBufferGetAddress(src_bo);

    lwnBufferBuilderSetDefaults(bufferBuilder);
    LWNbuffer *final_bo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, coherent_mempool, /*offset*/0, bufferW);
    LWNbufferAddress final_bo_addr = lwnBufferGetAddress(final_bo);

    int result = 0;
    for (size_t poolType = 0; poolType < __GL_ARRAYSIZE(poolTypes); poolType++) {

        LWNmemoryPool *memPool = lwnDeviceCreateMemoryPool(device, NULL, memSize, poolTypes[poolType].poolFlags);
        
        LWNbuffer *dst_bo[memPages];
        LWNbufferAddress dst_bo_addr[memPages];
        lwnBufferBuilderSetDefaults(bufferBuilder);
        for (int i = 0 ; i < memPages ; i++) {
            // offset multiple of 64k to make sure we touch another page in the pool
            dst_bo[i]= lwnBufferBuilderCreateBufferFromPool(bufferBuilder, memPool, pageSize * i, bufferW);
            dst_bo_addr[i] = lwnBufferGetAddress(dst_bo[i]);
        }

        unsigned char *bufdata = (unsigned char *)lwnBufferMap(src_bo);
        // fill with data
        for (int i = 0; i < bufferW; ++i) {
            bufdata[i] = (unsigned char)(i*(256.f/bufferW));
        }

        lwnCommandBufferCopyBufferToBuffer(queueCB, src_bo_addr, dst_bo_addr[0], bufferW, LWN_COPY_FLAGS_NONE);
        for (int i = 0; i < memPages - 1 ; i++) {
            lwnCommandBufferCopyBufferToBuffer(queueCB, dst_bo_addr[i], dst_bo_addr[i+1], bufferW, LWN_COPY_FLAGS_NONE);
        }

        lwnCommandBufferCopyBufferToBuffer(queueCB, dst_bo_addr[memPages - 1], final_bo_addr, bufferW, LWN_COPY_FLAGS_NONE);
        queueCB.submit();
        lwnQueueFinish(queue);  

        unsigned char *results = (unsigned char *)lwnBufferMap(final_bo);
        for (int i = 0; i < bufferW; ++i) {
            if (results[i] != i*(256.f/bufferW) ) {
                result -= 1;
                break;
            }
        }

        for (int i = 0 ; i < memPages ; i++) {
            lwnBufferFree(dst_bo[i]);
        }

        lwnMemoryPoolFree(memPool);
    }
    lwnBufferBuilderFree(bufferBuilder);
    lwnBufferFree(src_bo);
    lwnBufferFree(final_bo);
    lwnMemoryPoolFree(coherent_mempool);
    
    return result;
}

// ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
// simplePrebakedAsset: create an asset and initialise different types
// of pools with it
int LWNMemoryPoolTest::simplePrebakedAsset() const
{
    LWNdevice *device = g_lwnDevice;
    LWNqueue *queue = g_lwnQueue;
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;

    struct _poolTypes 
    {
        LWNmemoryPoolFlags  poolFlags;
        bool                mappable;
        bool                texture;
    }poolTypes[] = 
    { 
#if defined(LW_TEGRA)
        { LWN_MEMORY_POOL_TYPE_CPU_COHERENT,        true,   true}, // coherent pool allowed on HOS
#else
        { LWN_MEMORY_POOL_TYPE_CPU_COHERENT,        true,   false}, // coherent pool allows no textures on non-HOS platforms
#endif
        { LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT,    true,   true },
        { LWN_MEMORY_POOL_TYPE_GPU_ONLY,            false,  true }
    };


    LWNbufferBuilder *bufferBuilder = lwnDeviceCreateBufferBuilder(device);

    lwnBufferBuilderSetDefaults(bufferBuilder);

    int result = 0;

    // these values are arbitrarily chosen since we don't have 
    // APIs that allow us to query constraints related to the values like
    // alignment, max. sizes etc.)
    const int bufferW = 256;
    const int memPages = 16;
    const int pageSize = (64*1024);
    const int memSize =  pageSize * memPages; 
    const int res_offset = memSize/2;

    // create a pool
    LWNmemoryPool *memPool = lwnDeviceCreateMemoryPool(device, NULL, memSize, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);

    // create some buffer in it (add textures at a later time)
    LWNbuffer *asset_bo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, memPool, res_offset, bufferW);
    unsigned char *asset = (unsigned char *)lwnBufferMap(asset_bo);
    // fill with texture data
    for (int i = 0; i < bufferW; ++i) {
        asset[i] = (unsigned char)(0xFF - i*(256.f/bufferW));
    }

    // flush from sysmem pool to sparse pool
    lwnMemoryPoolFlushMappedRange(memPool, 0, memSize);

    // what type of barrier/flush here for coherency?
    lwnQueueFinish(queue);  

    // create a buffer from the whole pool
    lwnBufferBuilderSetDefaults(bufferBuilder);
    LWNbuffer *copy_bo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, memPool, 0, memSize);
    
    // make a copy using map
    unsigned char *bufdata = (unsigned char *)lwnBufferMap(copy_bo);
    unsigned char *prebaked = (unsigned char*)PoolStorageAlloc(memSize);
    memcpy(prebaked, bufdata, memSize);

    lwnBufferFree(asset_bo);
    lwnBufferFree(copy_bo);

    // checkwith texture data
    for (int i = 0; i < bufferW; ++i) {
        if (prebaked[res_offset + i] != (0xFF - i*(256.f/bufferW))) {
            result -= 1;
            break;
        }
    }

    lwnMemoryPoolFree(memPool);

    if (result) {
        return result;
    }

    for (size_t poolType = 0; poolType < __GL_ARRAYSIZE(poolTypes); poolType++) {

    // then create a new pool and copy asset into it
        LWNmemoryPool *anotherMemPool = lwnDeviceCreateMemoryPool(device, prebaked, memSize, poolTypes[poolType].poolFlags);

        lwnBufferBuilderSetDefaults(bufferBuilder);
        LWNbuffer *real_asset_bo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, anotherMemPool, res_offset, bufferW);
        LWNbufferAddress real_asset_bo_addr = lwnBufferGetAddress(real_asset_bo);

        // flush to sysmem pool from sparse pool
        lwnMemoryPoolFlushMappedRange(anotherMemPool, 0, memSize);

        lwnQueueFinish(queue);  

        LWNmemoryPool *map_mempool = NULL;
        LWNbuffer *mapbo = NULL;
        LWNbufferAddress mapbo_addr = 0;
        if (poolTypes[poolType].mappable) {
            bufdata = (unsigned char *)lwnBufferMap(real_asset_bo);
        } else {
            map_mempool = lwnDeviceCreateMemoryPool(device, NULL, bufferW, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
            lwnBufferBuilderSetDefaults(bufferBuilder);
            mapbo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, map_mempool, 0, bufferW);
            mapbo_addr = lwnBufferGetAddress(mapbo);
            lwnCommandBufferCopyBufferToBuffer(queueCB, real_asset_bo_addr, mapbo_addr, bufferW, LWN_COPY_FLAGS_NONE);
            queueCB.submit();
            lwnQueueFinish(queue);  
            bufdata = (unsigned char *)lwnBufferMap(mapbo);
        }

        // checkwith texture data
        for (int i = 0; i < bufferW; ++i) {
            if (bufdata[i] != (0xFF - i*(256.f/bufferW))) {
                result -= 1;
                break;
            }
        }

        lwnBufferFree(real_asset_bo);
        if (!poolTypes[poolType].mappable) {
            lwnBufferFree(mapbo);
            lwnMemoryPoolFree(map_mempool);
        }

        lwnMemoryPoolFree(anotherMemPool);
    }
    lwnBufferBuilderFree(bufferBuilder);

    PoolStorageFree(prebaked);
    prebaked = NULL;

    return result ? result - 1 : 0;
}

// ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

// simpleCoherentPool: verify coherent pool functionality
int LWNMemoryPoolTest::simpleCoherentPool() const
{
    LWNdevice *device = g_lwnDevice;
    LWNqueue *queue = g_lwnQueue;
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;

    int result = 0;

    // these values are arbitrarily chosen since we don't have 
    // APIs that allow us to query constraints related to the values like
    // alignment, max. sizes etc.)
    const int bufferW = 256;
    const int memPages = 16;
    const int pageSize = (64*1024);
    const int memSize =  pageSize * memPages; 
    const int res_offset = memSize/2;

    LWNbufferBuilder *bufferBuilder = lwnDeviceCreateBufferBuilder(device);
    lwnBufferBuilderSetDefaults(bufferBuilder);

    // create a pool
    LWNmemoryPool *memPool = lwnDeviceCreateMemoryPool(device, NULL, memSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    void* poolMap = lwnMemoryPoolMap(memPool);
    if (memPool && !poolMap) {
        return -1;
    }

    // create some buffers and textures in it
    LWNbuffer *bo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, memPool, res_offset, bufferW);
    LWNbufferAddress bo_addr = lwnBufferGetAddress(bo);
    unsigned char *data = (unsigned char *)lwnBufferMap(bo);

    if (!data) {
        result -= 1;
    } else {
        for (int i = 0; i < bufferW; ++i) {
            data[i] = (unsigned char)(0xFF - i*(256.f/bufferW));
        }
    }

    // cause potential overwrite when implementation copies back from sparse to sysmem
    //
    // lwnMemoryPoolIlwalidateMappedRange would copy from sparse video memory to a sysmem
    // pitch linear copy on WDDM drivers. WDDM have the downside that they can't give us
    // CPU maps of video memory resources and therefore the implementation needs to
    // be a little more sophisticated on these platforms. 
    // If lwnMemoryPoolIlwalidateMappedRange were faulty it would overwrite what
    // we just wrote with the CPU from the GPU copy.
    lwnMemoryPoolIlwalidateMappedRange(memPool, 0, memSize);
    lwnQueueFinish(queue);  

    // verify data
    for (int i = 0; i < bufferW; ++i) {
        if (data[i] != (0xFF - i*(256.f/bufferW))) {
            result -= 1;
            break;
        }
    }

    // create a second buffer and copy first to second, we should be able
    // see the data w/o ilwalidates
    LWNbuffer *final_bo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, memPool, /* offset */ 0, bufferW);
    LWNbufferAddress final_bo_addr = lwnBufferGetAddress(final_bo);

    lwnCommandBufferCopyBufferToBuffer(queueCB, bo_addr, final_bo_addr, bufferW, LWN_COPY_FLAGS_NONE);
    queueCB.submit();
    // what type of barrier/flush here for CPU coherency?
    lwnQueueFinish(queue);  

    lwnBufferFree(bo);

    data = (unsigned char *)lwnBufferMap(final_bo);

    // verify data
    for (int i = 0; i < bufferW; ++i) {
        if (data[i] != (0xFF - i*(256.f/bufferW))) {
            result -= 1;
            break;
        }
    }

    lwnBufferFree(final_bo);

    lwnBufferBuilderFree(bufferBuilder);
    lwnMemoryPoolFree(memPool);

    return result ? result - 1 : 0;
}

// simpleNonCoherentPool: verify non-coherent pool functionality
// with respect to map/flush/ilwalidate
int LWNMemoryPoolTest::simpleNonCoherentPool() const
{
    LWNdevice *device = g_lwnDevice;
    LWNqueue *queue = g_lwnQueue;
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;

    int result = 0;

    // these values are arbitrarily chosen since we don't have 
    // APIs that allow us to query constraints related to the values like
    // alignment, max. sizes etc.)
    const int bufferW = 256;
    const int memPages = 1;
    const int pageSize = (64*1024);
    const int memSize =  pageSize * memPages; 
    const int res_offset = memSize/2;

    LWNbufferBuilder *bufferBuilder = lwnDeviceCreateBufferBuilder(device);
    lwnBufferBuilderSetDefaults(bufferBuilder);

    // create a GPU pool 
    // create a pool
    LWNmemoryPool *memPool = lwnDeviceCreateMemoryPool(device, NULL, memSize, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);

    void* poolMap = lwnMemoryPoolMap(memPool);
    if (memPool && !poolMap) {
        return -1;
    }

    // create some buffers and textures in it
    LWNbuffer *bo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, memPool, res_offset, bufferW);
    LWNbufferAddress bo_addr = lwnBufferGetAddress(bo);
    volatile unsigned char *data = (unsigned char *)lwnBufferMap(bo);

    if (!data) {
        result -= 1;
    } else {
        for (int i = 0; i < bufferW; ++i) {
            data[i] = (unsigned char)(0xFF - i*(256.f/bufferW));
        }
    }

    // app needs to flush the mapped range for the GPU to see the data
    lwnMemoryPoolFlushMappedRange(memPool, res_offset, bufferW);

    // cause potential overwrite when implementation copies the whole pool back from sparse to sysmem
    lwnMemoryPoolIlwalidateMappedRange(memPool, 0, memSize);
    lwnQueueFinish(queue);  

    // TBD: use buffer to see if it has coherence
    // maybe draw with uniform???

    // create a texture in the pool
    // it should not get created in a coherent pool
    LWNtextureBuilder *textureBuilder = lwnDeviceCreateTextureBuilder(device);
    lwnTextureBuilderSetDefaults(textureBuilder);
    lwnTextureBuilderSetSize2D(textureBuilder, offscreenWidth, offscreenHeight);
    lwnTextureBuilderSetTarget(textureBuilder, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetSamples(textureBuilder, 0);
    lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_RGBA8);

    // offset needs to be manually adjusted if you change any of the other allocations from this pool
    // (overlap etc.)
    LWNtexture *tex = lwnTextureBuilderCreateTextureFromPool(textureBuilder, memPool, /*offset*/0);
    if (!tex) {
        result -= 2;
    }

    lwnTextureFree(tex);

    // verify data
    for (int i = 0; i < bufferW; ++i) {
        if (data[i] != (0xFF - i*(256.f/bufferW))) {
            result -= 3;
            break;
        }
    }

    LWNbuffer *final_bo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, memPool, 0, bufferW);
    LWNbufferAddress final_bo_addr = lwnBufferGetAddress(final_bo);

    lwnCommandBufferCopyBufferToBuffer(queueCB, bo_addr, final_bo_addr, bufferW, LWN_COPY_FLAGS_NONE);
    queueCB.submit();
    // what type of barrier/flush here for CPU coherency?
    lwnQueueFinish(queue);  

    // make GPU writes visible
    lwnMemoryPoolIlwalidateMappedRange(memPool, 0, memSize);

    lwnQueueFinish(queue);  

    lwnBufferFree(bo);

    data = (unsigned char *)lwnBufferMap(final_bo);
    lwnQueueFinish(queue);  

    // verify data
    for (int i = 0; i < bufferW; ++i) {
        if (data[i] != (0xFF - i*(256.f/bufferW))) {
            result -= 4;
            break;
        }
    }

    lwnBufferFree(final_bo);

    lwnTextureBuilderFree(textureBuilder);
    lwnBufferBuilderFree(bufferBuilder);
    lwnMemoryPoolFree(memPool);

    return result ? result - 1 : 0;
}

// simpleGpuOnlyPool: verify GPU only pool functionality
// with respect to map/flush/ilwalidate
int LWNMemoryPoolTest::simpleGpuOnlyPool() const
{
    LWNdevice *device = g_lwnDevice;
    LWNqueue *queue = g_lwnQueue;

    int result = 0;

    // these values are arbitrarily chosen since we don't have 
    // APIs that allow us to query constraints related to the values like
    // alignment, max. sizes etc.)
    const int bufferW = 256;
    const int memPages = 16;
    const int pageSize = (64*1024);
    const int memSize =  pageSize * memPages; 
    const int res_offset = memSize/2;

    LWNbufferBuilder *bufferBuilder = lwnDeviceCreateBufferBuilder(device);
    lwnBufferBuilderSetDefaults(bufferBuilder);

    // create a pool
    LWNmemoryPool *memPool = lwnDeviceCreateMemoryPool(device, NULL, memSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    // create some buffers and textures in it
    LWNbuffer *bo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, memPool, res_offset, bufferW);
    unsigned char *data = (unsigned char *)lwnBufferMap(bo);

    if (data) {
        result -= 1;
    } 

    // cause potential issue when implementation copies back from sparse to sysmem and vice versa
    // these should effectively be no-ops for GPU only pools
    lwnMemoryPoolIlwalidateMappedRange(memPool, 0, memSize);
    lwnMemoryPoolFlushMappedRange(memPool, 0, memSize);
    lwnQueueFinish(queue);  

    // create a texture in the pool
    // it should get created in a GPU only pool
    LWNtextureBuilder *textureBuilder = lwnDeviceCreateTextureBuilder(device);
    lwnTextureBuilderSetDefaults(textureBuilder);
    lwnTextureBuilderSetSize2D(textureBuilder, offscreenWidth, offscreenHeight);
    lwnTextureBuilderSetTarget(textureBuilder, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetSamples(textureBuilder, 0);
    lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_RGBA8);

    // offset needs to be manually adjusted if you change any of the other allocations from this pool
    // (overlap etc.)
    LWNtexture *tex = lwnTextureBuilderCreateTextureFromPool(textureBuilder, memPool, /*offset*/0);
    if (!tex) {
        result -= 1;
    }

    lwnBufferFree(bo);
    lwnTextureFree(tex);

    lwnTextureBuilderFree(textureBuilder);
    lwnBufferBuilderFree(bufferBuilder);
    lwnMemoryPoolFree(memPool);

    return result ? result - 1 : 0;
}

// simplePoolSizes: create differently sized pools of different type
int LWNMemoryPoolTest::simplePoolSizes() const
{
    LWNdevice *device = g_lwnDevice;
    LWNqueue *queue = g_lwnQueue;

    int result = 0;

    // these values are arbitrarily chosen since we don't have 
    // APIs that allow us to query constraints related to the values like
    // alignment, max. sizes etc.)
    const int pageSize = (64*1024);
    struct _poolTypes 
    {
        LWNmemoryPoolFlags  poolFlags;
        bool                mappable;
    }poolTypes[] = 
    { 
        { LWN_MEMORY_POOL_TYPE_CPU_COHERENT,        true},
        { LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT,    true},
        { LWN_MEMORY_POOL_TYPE_GPU_ONLY,            false}
    };

    LWNbufferBuilder *bufferBuilder = lwnDeviceCreateBufferBuilder(device);
    lwnBufferBuilderSetDefaults(bufferBuilder);

    for (size_t poolType = 0; poolType < __GL_ARRAYSIZE(poolTypes); poolType++) {
        int memPages = 1;
        int memSize; 
        do {
            memSize = pageSize * memPages; 

            // create a pool
            LWNmemoryPool *memPool = lwnDeviceCreateMemoryPool(device, NULL, memSize, poolTypes[poolType].poolFlags);
            if (!memPool) {
                break;
            }

            if (poolTypes[poolType].mappable) {
                void* poolMap = lwnMemoryPoolMap(memPool);
                if (!poolMap) {
                    return false;
                }
            }

            // create a buffer half of the pool size
            LWNbuffer *bo = lwnBufferBuilderCreateBufferFromPool(bufferBuilder, memPool, /* offset */ memSize/2, memSize/2);
            unsigned char *data = (unsigned char *)lwnBufferMap(bo);

            if ( (poolTypes[poolType].mappable && !data) ||  (!poolTypes[poolType].mappable && data) ) {
                result -= 1;
                break;
            } 
            
            // cause potential issue when implementation copies back from sparse to sysmem and vice versa
            // these should effectively be no-ops for GPU only pools
            lwnMemoryPoolIlwalidateMappedRange(memPool, 0, memSize);
            lwnMemoryPoolFlushMappedRange(memPool, 0, memSize);
            lwnQueueFinish(queue);  

            lwnBufferFree(bo);
            lwnMemoryPoolFree(memPool);
            memPages <<= 1;
        } while (memSize < (pageSize*0x80));
    }

    lwnBufferBuilderFree(bufferBuilder);

    return result ? result - 1 : 0;
}

void LWNMemoryPoolTest::doGraphics() const
{
    CellIterator2D cell(cellsX, cellsY);
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;

    // clear
    LWNfloat clearColor[] = {0.3, 0.3, 0.3, 1};
    lwnCommandBufferClearColor(queueCB, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    queueCB.submit();

    int result = 0;
    for (int variant = TEST_VARIANT_RT; variant < TEST_VARIANT_LAST; variant++) {
        switch(variant)
        {
        case TEST_VARIANT_RT:
            // allocate a rendertarget from memory pool and clear to green, then present
            result = simpleColorRenderTargetFromPool(false);
            break;
        case TEST_VARIANT_RT_Z:
            // allocate a rendertarget from memory pool and clear to green, then present
            result = simpleColorRenderTargetFromPool(true);
            break;
        case TEST_VARIANT_TEX_FROM_POOL:
            //// allocate a texture and render with it
            result = simpleFromPool(TT_TEXTURE);
            break;
        case TEST_VARIANT_BUF_FROM_POOL:
            // allocate a buffer and and use it to fill texture
            result = simpleFromPool(TT_BUFFER);
            break;
        case TEST_VARIANT_BUFFER_COPY:
            // allocate multiple buffers  and cross copy
            result = simpleBufferToBufferFromPool();
            break;
        case TEST_VARIANT_PRE_BAKED:
            // make prebaked asset and use it
            result = simplePrebakedAsset();
            break;
        case TEST_VARIANT_COHERENT:
            // coherent pool creation/flush/ilwalidate/map
            result = simpleCoherentPool();
            break;
        case TEST_VARIANT_NON_COHERENT:
            // non-coherent pool creation/flush/ilwalidate/map
            result = simpleNonCoherentPool();
            break;
        case TEST_VARIANT_GPU_ONLY:
            // GPU only pool creation/flush/ilwalidate/map
            result = simpleGpuOnlyPool();
            break;
        case TEST_VARIANT_POOL_SIZE:
            // differently sized pool with different pool types
            result = simplePoolSizes();
            break;
        default:
            assert(0);
            break;
        }

        drawResult(cell, result);
    }
}

OGTEST_CppTest(LWNMemoryPoolTest, lwn_mempool,);
