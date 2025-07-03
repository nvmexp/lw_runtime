/*
 * Copyright (c) 2015-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#if defined(LW_HOS)
#include <nn/nn_Log.h>
#else
#define NN_LOG(...)
#endif
#include <lwn/lwn_FuncPtrImpl.h>
#include <lwnTool/lwnTool_GlslcInterface.h>

#include "Simple_Triangle.h"
#include "lwnUtil/lwnUtil_AlignedStorageImpl.h"
#include "lwnUtil/lwnUtil_PoolAllocatorImpl.h"
#include "textured-triangle-glslcbin.h"

// Two triangles that intersect
static float vertexData[] = {-0.5f, -0.5f, 0.5f, 
                              0.5f, -0.5f,  0.5f,
                             -0.5f, 0.5f,  0.5f};

// Simple 0/1 texcoords in rgba8 format (used to be color data)
static uint8_t texcoordData[] = {0, 0, 0xFF, 0xFF,
                                 0xFF, 0, 0, 0xFF,
                                 0, 0xFF, 0, 0xFF};

int offscreenWidth = 1280, offscreenHeight = 720;

static const int commandPoolAllocSize = 16*1024*1024;
static const int controlPoolAllocSize =  4*1024*1024;
static const int commandPoolChunks = 4;
static const int controlPoolChunks = 4;
static const int commandPoolChunkSize = commandPoolAllocSize / commandPoolChunks;
static const int controlPoolChunkSize = controlPoolAllocSize / controlPoolChunks;

static LWNshaderStageBits allShaderStages = LWNshaderStageBits(LWN_SHADER_STAGE_VERTEX_BIT |
                                                               LWN_SHADER_STAGE_FRAGMENT_BIT);

typedef struct {
    float scale[4];
    LWNtextureHandle bindlessTex;
} UniformBlock;

void LWNSampleTestConfig::Init(LWNnativeWindow nativeWindow, LWNformat winTexFormat)
{
    LWNdevice *device = m_c_interface->device;
    LWNqueue *queue = m_c_interface->queue;

    mCmdBuf = lwnDeviceCreateCommandBuffer(device);

#if defined(LW_HOS)
    mCommandPoolMemory = lwnUtil::PoolStorageAlloc(commandPoolAllocSize);
#else
    mCommandPoolMemory = NULL;
#endif

    mCommandPool = lwnDeviceCreateMemoryPool(device, mCommandPoolMemory, commandPoolAllocSize,
                                             LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    mControlPool = new char[controlPoolAllocSize];
    lwnCommandBufferAddCommandMemory(mCmdBuf, mCommandPool, 0, commandPoolChunkSize);
    lwnCommandBufferAddControlMemory(mCmdBuf, mControlPool, controlPoolChunkSize);
    lwnCommandBufferSetMemoryCallback(mCmdBuf, outOfMemory);
    lwnCommandBufferSetMemoryCallbackData(mCmdBuf, this);
    mCommandPoolChunk = 0;
    mControlPoolChunk = 0;

    lwnCommandBufferBeginRecording(mCmdBuf);

    LWNcommandBuffer *queueCB = mCmdBuf;
    LWNcommandHandle queueCBHandle;

    // Initialize the texture ID pool manager.
    g_lwn.m_texIDPool = new LWNsystemTexIDPool(device, queueCB);

    int textureSize, samplerSize;
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_TEXTURE_DESCRIPTOR_SIZE, &textureSize);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SAMPLER_DESCRIPTOR_SIZE, &samplerSize);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_RESERVED_TEXTURE_DESCRIPTORS, &mNumReservedTextureIDs);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_RESERVED_SAMPLER_DESCRIPTORS, &mNumReservedSamplerIDs);

    int numPublicTextures = LWNsystemTexIDPool::GetNumPublicTextures();
    int numPublicSamplers = LWNsystemTexIDPool::GetNumPublicSamplers();

    size_t descriptorPoolSize = ((mNumReservedSamplerIDs + numPublicSamplers) * samplerSize +
                                 (mNumReservedTextureIDs + numPublicTextures) * textureSize);

#if defined(LW_HOS)
    descriptorPoolSize = lwnUtil::PoolStorageSize(descriptorPoolSize);
    mDescriptorPoolMemory = lwnUtil::PoolStorageAlloc(descriptorPoolSize);
#else
    mDescriptorPoolMemory = NULL;
#endif

#if defined(_WIN32)
    mDescriptorPool = lwnDeviceCreateMemoryPool(device, mDescriptorPoolMemory, descriptorPoolSize,
                            LWN_MEMORY_POOL_TYPE_GPU_ONLY);
#else
    mDescriptorPool = lwnDeviceCreateMemoryPool(device, mDescriptorPoolMemory, descriptorPoolSize,
                            LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
#endif
    lwnSamplerPoolInitialize(&mSamplerPool, mDescriptorPool, 0,
                                 mNumReservedSamplerIDs + numPublicSamplers);
    lwnTexturePoolInitialize(&mTexturePool, mDescriptorPool,
                                 (mNumReservedSamplerIDs + numPublicSamplers) * samplerSize,
                                 mNumReservedTextureIDs + numPublicTextures);
    lwnCommandBufferSetSamplerPool(queueCB, &mSamplerPool);
    lwnCommandBufferSetTexturePool(queueCB, &mTexturePool);
    mNextTextureID = mNumReservedTextureIDs;
    mNextSamplerID = mNumReservedSamplerIDs;

    mBufferPoolAllocator = new MemoryPoolAllocator(device, NULL, 0, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    mTexturePoolAllocator = new MemoryPoolAllocator(device, NULL, 0, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    mBufferBuilder = lwnDeviceCreateBufferBuilder(device);
    mTextureBuilder = lwnDeviceCreateTextureBuilder(device);

    // The following section uses a pre-baked GLSLCoutput binary containing two shaders: a vertex
    // and a fragment shader.  The program will iterate over the GLSLCoutput sections, find the shader
    // data for each stage, and initialize mShaderProgram.  lwnProgramSetShaders is then called to
    // set up the display pipeline to reference the program.

    // Create the shader program.
    mShaderProgram = lwnDeviceCreateProgram(device);

    // The GLSLCoutput is pre-baked for this sample.  It was generated offline
    // using LwnGLSLC API 17.21, GPU 1.16.
    const GLSLCoutput * glslcOutput = (const GLSLCoutput *)(&textured_triangle_bin[0]);

    const char * shaderGpuCode[2] = { NULL };
    const char * shaderControl[2] = { NULL };
    unsigned int gpuCodeSize[2] = { 0 };

    // Find the gpu data and control sections in the GLSLCoutput required for LWN.
    int lwrrShaderNdx = 0;
    for (unsigned int i = 0; i < glslcOutput->numSections; ++i) {
        GLSLCsectionTypeEnum type = glslcOutput->headers[i].genericHeader.common.type;

        // If the section is GPU_CODE section, we want to use it.
        if (type == GLSLC_SECTION_TYPE_GPU_CODE) {
            GLSLCgpuCodeHeader gpuCodeHeader =
                (GLSLCgpuCodeHeader)(glslcOutput->headers[i].gpuCodeHeader);

            // Find the offset to the actual section's data
            char * data = (char *)glslcOutput + gpuCodeHeader.common.dataOffset;

            // Obtain gpu data and control by offsetting into the data.
            shaderGpuCode[lwrrShaderNdx] = data + gpuCodeHeader.dataOffset;
            shaderControl[lwrrShaderNdx] = data + gpuCodeHeader.controlOffset;
            gpuCodeSize[lwrrShaderNdx] = gpuCodeHeader.dataSize;

            lwrrShaderNdx++;
        }
    }

    // We should only have two GPU code sections since we are only compiling a vertex and a fragment shader.
    assert(lwrrShaderNdx == 2);

    // Build the buffers holding the programs and get CPU mapped pointers to them
    lwnBufferBuilderSetDefaults(mBufferBuilder);

    // Set up the shader data structures to send to LWN
    LWNshaderData shaderData[2];
    memset(&shaderData[0], 0, sizeof(LWNshaderData)*2);

    for (int i = 0; i < 2; ++i) {
        LWNbuffer * progBuf = mBufferPoolAllocator->allocBuffer(mBufferBuilder, BUFFER_ALIGN_VERTEX_BIT, gpuCodeSize[i]);
        void * buffCpuPtr = lwnBufferMap(progBuf);

        // Copy over the GPU code data to the buffers.
        memcpy(buffCpuPtr, shaderGpuCode[i], gpuCodeSize[i]);

        shaderData[i].data = lwnBufferGetAddress(progBuf);
        shaderData[i].control = shaderControl[i];
    }

    // Set the program's shader data and initialize the shader program.
    lwnProgramSetShaders(mShaderProgram, 2, &(shaderData[0]));

    // Set the state vector to use two vertex attributes.
    //
    // Interleaved pos+color
    // position = attrib 0 = 3*float at relativeoffset 0
    // texcoord = attrib 1 = rgba8 at relativeoffset 0
    lwlwertexAttribStateSetDefaults(mVertexAttribs + 0);
    lwlwertexAttribStateSetDefaults(mVertexAttribs + 1);
    lwlwertexStreamStateSetDefaults(mVertexStreams + 0);
    lwlwertexStreamStateSetDefaults(mVertexStreams + 1);
    lwlwertexAttribStateSetFormat(mVertexAttribs + 0, LWN_FORMAT_RGB32F, 0);
    lwlwertexAttribStateSetFormat(mVertexAttribs + 1, LWN_FORMAT_RGBA8, 0);
    lwlwertexAttribStateSetStreamIndex(mVertexAttribs + 0, 0);
    lwlwertexAttribStateSetStreamIndex(mVertexAttribs + 1, 1);
    lwlwertexStreamStateSetStride(mVertexStreams + 0, 12);
    lwlwertexStreamStateSetStride(mVertexStreams + 1, 4);

    // Create a vertex buffer and fill it with data
    lwnBufferBuilderSetDefaults(mBufferBuilder);
    mVertexBuffer = mBufferPoolAllocator->allocBuffer(mBufferBuilder, BUFFER_ALIGN_VERTEX_BIT, sizeof(vertexData)+sizeof(texcoordData));

    // create persistent mapping
    void *ptr = lwnBufferMap(mVertexBuffer);

    // fill ptr with vertex data followed by tex coord data
    memcpy(ptr, vertexData, sizeof(vertexData));
    memcpy((char *)ptr + sizeof(vertexData), texcoordData, sizeof(texcoordData));

    // Create an index buffer and fill it with data
    unsigned short indexData[6] = { 0, 1, 2, 3, 4, 5 };

    mIndexBuffer = mBufferPoolAllocator->allocBuffer(mBufferBuilder, BUFFER_ALIGN_INDEX_BIT, sizeof(indexData));
    ptr = lwnBufferMap(mIndexBuffer);
    memcpy(ptr, indexData, sizeof(indexData));

    lwnTextureBuilderSetDefaults(mTextureBuilder);
    lwnTextureBuilderSetFlags(mTextureBuilder, LWN_TEXTURE_FLAGS_DISPLAY_BIT |
                                               LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT);
    lwnTextureBuilderSetSize2D(mTextureBuilder, offscreenWidth, offscreenHeight);
    lwnTextureBuilderSetTarget(mTextureBuilder, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetFormat(mTextureBuilder, winTexFormat);

    for (int i = 0; i < NUM_BUFFERS; i++) {
        mRenderTargetTextures[i] = mTexturePoolAllocator->allocTexture(mTextureBuilder);
        lwnTexturePoolRegisterTexture(&mTexturePool, mNextTextureID++, mRenderTargetTextures[i], NULL);
    }

    lwnTextureBuilderSetFlags(mTextureBuilder, LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT);
    lwnTextureBuilderSetFormat(mTextureBuilder, LWN_FORMAT_DEPTH24_STENCIL8);
    mDepthTexture = mTexturePoolAllocator->allocTexture(mTextureBuilder);
    mDepthTextureID = mNextTextureID++;
    lwnTexturePoolRegisterTexture(&mTexturePool, mDepthTextureID, mDepthTexture, NULL);

    mSamplerBuilder = lwnDeviceCreateSamplerBuilder(device);
    mSampler = lwnSamplerBuilderCreateSampler(mSamplerBuilder);
    mSamplerID = mNextSamplerID++;
    lwnSamplerPoolRegisterSampler(&mSamplerPool, mSamplerID, mSampler);

    mUniformBlockBuffer = mBufferPoolAllocator->allocBuffer(mBufferBuilder, BUFFER_ALIGN_UNIFORM_BIT, sizeof(UniformBlock));

    LWNwindowBuilder wb;
    lwnWindowBuilderSetDefaults(&wb);
    lwnWindowBuilderSetDevice(&wb, device);
    lwnWindowBuilderSetNativeWindow(&wb, nativeWindow);
    lwnWindowBuilderSetTextures(&wb, NUM_BUFFERS, mRenderTargetTextures);
    lwnWindowInitialize(&mWindow, &wb);
    lwnSyncInitialize(&mTextureAvailableSync, device);

    ///////////////////////////////////////////////////////////
    // Generate a texture
    const int texWidth = 1000;
    const int texHeight = 1000;
    lwnTextureBuilderSetDefaults(mTextureBuilder);
    lwnTextureBuilderSetTarget(mTextureBuilder, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetFormat(mTextureBuilder, LWN_FORMAT_RGBA8);
    lwnTextureBuilderSetSize2D(mTextureBuilder, texWidth, texHeight);
    mTexture = mTexturePoolAllocator->allocTexture(mTextureBuilder);
    mTextureID = mNextTextureID++;
    lwnTexturePoolRegisterTexture(&mTexturePool, mTextureID, mTexture, NULL);

    lwnBufferBuilderSetDefaults(mBufferBuilder);
    LWNbuffer *pbo = mBufferPoolAllocator->allocBuffer(mBufferBuilder, BUFFER_ALIGN_COPY_READ_BIT, texWidth*texHeight * 4);
    LWNbufferAddress pboAddr = lwnBufferGetAddress(pbo);

    unsigned char *texdata = (unsigned char *)lwnBufferMap(pbo);

    // Bottom right
    float redBR = 0.0f;
    float greenBR = 0.0f;
    float blueBR = 1.0f;

    // Bottom Left
    float redBL = 0.0f;
    float greenBL = 1.0f;
    float blueBL = 0.0f;

    // Off Screen
    float redEndOff = 0.0f;
    float greenEndOff = 0.0f;
    float blueEndOff = 0.0f;

    // Top Left
    float redEndTL = 1.0f;
    float greenEndTL = 0.0f;
    float blueEndTL = 0.0f;

    float redBRInt = (redEndOff - redBR) / texHeight;
    float greenBRInt = (greenEndOff - greenBR) / texHeight;
    float blueBRInt = (blueEndOff - blueBR) / texHeight;

    float redBLInt = (redEndTL - redBL) / texHeight;
    float greenBLInt = (greenEndTL - greenBL) / texHeight;
    float blueBLInt = (blueEndTL - blueBL) / texHeight;

    int checkerCounterX = 1;
    int checkerCounterY = 1;

    bool checkerX = false; // XO
    bool checkerY = false; // OX
    int checkerSquareSize = 100;

    for (int j = 0; j < texWidth; ++j)
    {
        for (int i = 0; i < texHeight; ++i)
        {
            float xTex = (float)(texHeight - i) / (float)texHeight;
            if (i < 100 && j < 100)
            {
                float r = (1.0f - xTex) * redBR + xTex * redBL;
                float g = (1.0f - xTex) * greenBR + xTex * greenBL;
                float b = (1.0f - xTex) * blueBR + xTex * blueBL;

                texdata[4 * (j * texWidth + i) + 0] = (unsigned char)(r * 0xFF); //0x61 * ((i + j) & 1);    // R
                texdata[4 * (j * texWidth + i) + 1] = (unsigned char)(g * 0xFF); //0xB7 * ((i + j) & 1);    // G
                texdata[4 * (j * texWidth + i) + 2] = (unsigned char)(b * 0xFF); //0x58 * ((i + j) & 1);    // B
                texdata[4 * (j * texWidth + i) + 3] = 0xFF;                                               // A
            }
            else
            {
                if (checkerX == checkerY)
                {
                    texdata[4 * (j * texWidth + i) + 0] = 255;
                    texdata[4 * (j * texWidth + i) + 1] = 255;
                    texdata[4 * (j * texWidth + i) + 2] = 255;
                    texdata[4 * (j * texWidth + i) + 3] = 255;
                }
                else
                {
                    float r = (1.0f - xTex) * redBR + xTex * redBL;
                    float g = (1.0f - xTex) * greenBR + xTex * greenBL;
                    float b = (1.0f - xTex) * blueBR + xTex * blueBL;

                    texdata[4 * (j * texWidth + i) + 0] = (unsigned char)(r * 0xFF); //0x61 * ((i + j) & 1);    // R
                    texdata[4 * (j * texWidth + i) + 1] = (unsigned char)(g * 0xFF); //0xB7 * ((i + j) & 1);    // G
                    texdata[4 * (j * texWidth + i) + 2] = (unsigned char)(b * 0xFF); //0x58 * ((i + j) & 1);    // B
                    texdata[4 * (j * texWidth + i) + 3] = 0xFF;                                               // A
                }
            }
            if (!(checkerCounterX++ % checkerSquareSize))
                checkerX = !checkerX;
        }

        redBR += redBRInt;
        redBL += redBLInt;
        greenBR += greenBRInt;
        greenBL += greenBLInt;
        blueBR += blueBRInt;
        blueBL += blueBLInt;

        checkerCounterX = 1;
        if (!(checkerCounterY++ % checkerSquareSize))
            checkerY = !checkerY;
    }

    // Download the texture data
    LWNcopyRegion copyRegion = { 0, 0, 0, texWidth, texHeight, 1 };
    lwnCommandBufferCopyBufferToTexture(mCmdBuf, pboAddr, mTexture, NULL, &copyRegion, LWN_COPY_FLAGS_NONE);

    // Flush all init commands
    queueCBHandle = lwnCommandBufferEndRecording(queueCB);
    lwnQueueSubmitCommands(queue, 1, &queueCBHandle);

    lwnQueueFinish(queue);
    mBufferPoolAllocator->freeBuffer(pbo);
}

void LWNSampleTestConfig::cDisplay()
{
    LWNdevice *device = m_c_interface->device;
    LWNqueue *queue = m_c_interface->queue;
    LWNcommandBuffer *queueCB = mCmdBuf;
    LWNcommandHandle queueCBHandle;
    int bufferIndex;

    lwnWindowAcquireTexture(&mWindow, &mTextureAvailableSync, &bufferIndex);
    lwnQueueWaitSync(queue, &mTextureAvailableSync);

    lwnCommandBufferBeginRecording(queueCB);
    lwnCommandBufferSetRenderTargets(queueCB, 1, &mRenderTargetTextures[bufferIndex], NULL, mDepthTexture, NULL);

    // Set up a uniform buffer holding transformation code as well as the texture
    // handle for "-bindless".
    float scale = 1.5f;

    // Bindless requires a combined texture/sampler handle
    LWNtextureHandle texHandle = lwnDeviceGetTextureHandle(device, mTextureID, mSamplerID);

    UniformBlock uboData;
    uboData.bindlessTex = texHandle;
    uboData.scale[0] = scale;
    uboData.scale[1] = scale;
    uboData.scale[2] = 1.0f + 1.0f / 65536.0;
    uboData.scale[3] = 1.0f;

    void *ptr = lwnBufferMap(mUniformBlockBuffer);
    memcpy(ptr, &uboData, sizeof(UniformBlock));

    // Get a handle to be used for setting the buffer as a uniform buffer
    LWNbufferAddress uboAddr = lwnBufferGetAddress(mUniformBlockBuffer);

    lwnCommandBufferSetViewport(queueCB, 0, 0, offscreenWidth, offscreenHeight);

    // Clear
    float clearColor[] = {0,0,0,1};
    lwnCommandBufferClearColor(queueCB, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    lwnCommandBufferClearDepthStencil(queueCB, 1.0, LWN_TRUE, 0, 0);

    // Bind the program, vertex state, and any required control structures.
    lwnCommandBufferBindProgram(queueCB, mShaderProgram, allShaderStages);
    lwnCommandBufferBindVertexAttribState(queueCB, 2, mVertexAttribs);
    lwnCommandBufferBindVertexStreamState(queueCB, 2, mVertexStreams);

    LWNbufferAddress vboAddr = lwnBufferGetAddress(mVertexBuffer);
    lwnCommandBufferBindVertexBuffer(queueCB, 0, vboAddr, sizeof(vertexData));
    lwnCommandBufferBindVertexBuffer(queueCB, 1, vboAddr + sizeof(vertexData), sizeof(texcoordData));
    lwnCommandBufferBindUniformBuffer(queueCB, LWN_SHADER_STAGE_VERTEX, 0, uboAddr, sizeof(uboData));
    lwnCommandBufferBindUniformBuffer(queueCB, LWN_SHADER_STAGE_FRAGMENT, 0, uboAddr, sizeof(uboData));

    // Draw
    LWNbufferAddress iboAddr = lwnBufferGetAddress(mIndexBuffer);
    lwnCommandBufferDrawElements(queueCB, LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_SHORT, 6, iboAddr);

    queueCBHandle = lwnCommandBufferEndRecording(queueCB);
    lwnQueueSubmitCommands(queue, 1, &queueCBHandle);

    lwnQueuePresentTexture(queue, &mWindow, bufferIndex);
}

void LWNSampleTestConfig::Deinit()
{
    // Make sure we're done with all of the objects before deallocating
    LWNqueue *queue = m_c_interface->queue;
    lwnQueueFinish(queue);

    // Clean up
    lwnProgramFree(mShaderProgram);
    lwnBufferBuilderFree(mBufferBuilder);
    mBufferPoolAllocator->freeBuffer(mVertexBuffer);
    mBufferPoolAllocator->freeBuffer(mIndexBuffer);
    mBufferPoolAllocator->freeBuffer(mUniformBlockBuffer);
    lwnTextureBuilderFree(mTextureBuilder);
    mTexturePoolAllocator->freeTexture(mTexture);
    lwnWindowFinalize(&mWindow);
    lwnSyncFinalize(&mTextureAvailableSync);
    for (int i = 0; i < NUM_BUFFERS; i++) {
        mTexturePoolAllocator->freeTexture(mRenderTargetTextures[i]);
    }
    mTexturePoolAllocator->freeTexture(mDepthTexture);
    lwnSamplerBuilderFree(mSamplerBuilder);
    lwnSamplerFree(mSampler);
    lwnSamplerPoolFinalize(&mSamplerPool);
    lwnTexturePoolFinalize(&mTexturePool);
    lwnMemoryPoolFree(mDescriptorPool);
    delete [] mControlPool;
    lwnCommandBufferFree(mCmdBuf);
    lwnMemoryPoolFree(mCommandPool);
    delete mBufferPoolAllocator;
    delete mTexturePoolAllocator;
}


void LWNAPIENTRY
    LWNSampleTestConfig::outOfMemory(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event,
                                     size_t minSize, void *callbackData)
{
    LWNSampleTestConfig *cfg = (LWNSampleTestConfig *) callbackData;
    assert(cmdBuf == cfg->mCmdBuf);

    // Note: This code makes no attempt to prevent the application from
    // overwriting parts of the commandbuffer still in use by, or not yet
    // submitted to the GPU.
    switch (event) {
        case LWN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY:
            cfg->mCommandPoolChunk = (cfg->mCommandPoolChunk + 1) % commandPoolChunks;
            lwnCommandBufferAddCommandMemory(cfg->mCmdBuf, cfg->mCommandPool, cfg->mCommandPoolChunk * commandPoolChunkSize, commandPoolChunkSize);
            break;
        case LWN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_CONTROL_MEMORY:
            cfg->mControlPoolChunk = (cfg->mControlPoolChunk + 1) % controlPoolChunks;
            lwnCommandBufferAddControlMemory(cfg->mCmdBuf, cfg->mControlPool + cfg->mControlPoolChunk * controlPoolChunkSize, controlPoolChunkSize);
            break;
        default:
            assert(0);
    }
}

