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
// lwnexample.h
//

#include <lwn/lwn.h>
#include <lwn/lwn_FuncPtrInline.h>
#include "lwnutil.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "lwnUtil/lwnUtil_PoolAllocator.h"

using namespace lwnUtil;

#define NUM_BUFFERS 2

// "Global" LWN objects created at initialization time, to be used by C code.
struct LWNSampleTestCInterface {
    LWNdevice *device;
    LWNqueue *queue;
};

class LWNSampleTestConfig
{
public:
    LWNSampleTestConfig() {}
    ~LWNSampleTestConfig() {}
    static struct LWNSampleTestCInterface       *m_c_interface;


    void cDisplay(void);                // render using C interface

    void Init(LWNnativeWindow nativeWindow, LWNformat winTexFormat);
    void Deinit();

    static void LWNAPIENTRY outOfMemory(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event,
                                        size_t minSize, void *callbackData);
private:
    LWNcommandBuffer       *mCmdBuf;
    void                   *mCommandPoolMemory;
    LWNmemoryPool          *mCommandPool;
    char                   *mControlPool;
    int                     mCommandPoolChunk;
    int                     mControlPoolChunk;
    void                   *mDescriptorPoolMemory;
    LWNmemoryPool          *mDescriptorPool;
    MemoryPoolAllocator    *mBufferPoolAllocator;
    MemoryPoolAllocator    *mTexturePoolAllocator;
    int                     mNumReservedTextureIDs;
    int                     mNumReservedSamplerIDs;
    int                     mNextTextureID;
    int                     mNextSamplerID;
    LWNprogram             *mShaderProgram;
    LWLwertexAttribState    mVertexAttribs[2];
    LWLwertexStreamState    mVertexStreams[2];
    LWNbufferBuilder       *mBufferBuilder;
    LWNtextureBuilder      *mTextureBuilder;
    LWNbuffer              *mVertexBuffer;
    LWNbuffer              *mIndexBuffer;
    LWNbuffer              *mUniformBlockBuffer;
    LWNtexture             *mRenderTargetTextures[NUM_BUFFERS];
    LWNtexture             *mDepthTexture;
    LWNtexture             *mTexture;
    int                     mDepthTextureID;
    int                     mTextureID;
    LWNsamplerBuilder      *mSamplerBuilder;
    LWNsampler             *mSampler;
    int                     mSamplerID;
    LWNwindow               mWindow;
    LWNsync                 mTextureAvailableSync;
    LWNsamplerPool          mSamplerPool;
    LWNtexturePool          mTexturePool;
};
