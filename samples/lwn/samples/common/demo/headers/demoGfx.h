/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/

#ifndef _DEMO_GFX_H_
#define _DEMO_GFX_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <demoGfxTypes.h>

#include "string.h"

#if defined _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <stdio.h>
#include <time.h>
#include <math.h>

extern u32 DEMOGfxOffscreenWidth, DEMOGfxOffscreenHeight;

void DEMOGfxInit(int argc, char **argv, LWNnativeWindow nativeWindow, int numPresentTextures = 2, int presentInterval = 1);
void DEMOGfxShutdown(void);

void DEMOGfxBeforeRender(void);
void DEMOGfxPresentRender(void);
void DEMOGfxDoneRender(void);

void DEMOGfxCreateDevice(LWNnativeWindow nativeWindow, int numPresentTextures, int presentInterval);
void DEMOGfxReleaseDevice();

void DEMOGfxSetContextState(DEMOGfxContextState *pContextState);
DEMOGfxContextState* DEMOGfxGetContextState(void);
void DEMOGfxInitContextState(DEMOGfxContextState *pContextState);

void DEMOGfxCreateShaders(DEMOGfxShader* pShader, const char* vertexShader, const char* pixelShader);
void DEMOGfxSetShaderAttribute(DEMOGfxShader *pShader, const DEMOGfxShaderAttributeData* pData);
void DEMOGfxSetShaders(DEMOGfxShader* pShader);
void DEMOGfxBindProgram(DEMOGfxShader* pShader);
void DEMOGfxReleaseShaders(DEMOGfxShader* pShader);

void DEMOGfxSetStatesDefault();

void DEMOGfxSetColorControl(u32 multiWrite, u32 specialOp, u32 blendEnable, u32 rop3);
void DEMOGfxSetBlendControl(int target,                            
                            LWNblendFunc colorFuncSrc, LWNblendFunc colorFuncDst, LWNblendEquation colorEquation,
                            LWNblendFunc alphaFuncSrc, LWNblendFunc alphaFuncDst, LWNblendEquation alphaEquation);
void DEMOGfxSetDepthControl(int stencilEnable, int depthEnable, int depthWriteEnable, LWNdepthFunc depthFunc);
void DEMOGfxSetPolygonControl(int lwllFront, int lwllBack, LWNfrontFace frontFace, int polyMode,
                              LWNpolygonMode frontMode, LWNpolygonMode backMode,
                              int offsetFront, int offsetBack, int offsetPointsLines);
void DEMOGfxSetStencilControl(int stencilRef, int stencilTestMask, int stencilWriteMask,
                              int backStencilRef, int backStencilTestMask, int backStencilWriteMask);
void DEMOGfxSetChannelMasks(int target0All);

void DEMOGfxCreateVertexBuffer(DEMOGfxVertexData* pData, u32 size);
void DEMOGfxSetVertexBuffer(DEMOGfxVertexData* pData, const void* pVertexData, u32 offset, u32 size);
void DEMOGfxBindVertexBuffer(DEMOGfxVertexData* pData, u32 index, u32 offset, u32 size);
void DEMOGfxReleaseVertexBuffer(DEMOGfxVertexData* pData);

void DEMOGfxCreateIndexBuffer(DEMOGfxIndexData* pData, u32 size);
void DEMOGfxSetIndexBuffer(DEMOGfxIndexData* pData, const void* pIndexData, u32 offset, u32 size);
void DEMOGfxReleaseIndexBuffer(DEMOGfxIndexData* pData);

void DEMOGfxCreateUniformBuffer(DEMOGfxUniformData* pData, u32 size);
void DEMOGfxSetUniformBuffer(DEMOGfxUniformData* pData, const void* pUniformData, u32 offset, u32 size);
void DEMOGfxBindUniformBuffer(const DEMOGfxUniformData* pData, LWNshaderStage stage, u32 index, u32 offset, u32 size);
void DEMOGfxReleaseUniformBuffer(DEMOGfxUniformData* pData);

void DEMOGfxCreateTextureBuffer(DEMOGfxTexture* pTexture, LWNtextureTarget target, u32 levels, LWNformat format, u32 width, u32 height, u32 depth, u32 samples, u32 size);
void DEMOGfxSetTextureBuffer(DEMOGfxTexture* pTexture, s32 bufferOffset, const void* pTextureData, u32 level, int xOffset, int yOffset, int zOffset, int width, int height, int depth, u32 size);
void DEMOGfxBindTextureBuffer(const DEMOGfxTexture* pTexture, LWNshaderStage stage, u32 index);
void DEMOGfxReleaseTextureBuffer(DEMOGfxTexture* pTexture);

void DEMOGfxClearColor(f32 r, f32 g, f32 b, f32 a);
void DEMOGfxClearDepthStencil(f32 depth, u32 stencil);

void DEMOGfxDrawArrays(LWNdrawPrimitive mode, u32 first, u32 count);
void DEMOGfxDrawElements(DEMOGfxIndexData *pData, LWNdrawPrimitive mode, LWNindexType type, u32 count, u32 offset);

void *DEMOGfxLoadTGAImage(u8* pTgaData, u32 fileSize, DEMOGfxTGAInfo *pInfo, u32 *pSize);
void DEMOGfxReleaseTGAImage(void *pBuffer);

void DEMOGfxSetIncludeCPUPerf(bool value);
void DEMOGfxSetUseCommandBuffer(bool value);
void DEMOGfxSetCommandBufferTransient(bool value);

void DEMOGfxResetLWNPerformace(void);
void DEMOGfxPrintLWNPerformace(void);

struct DEMOGfxGPUTimeStamp
{
    void Init();
    void ReportCounter();
    uint64_t GetTimeStamp();

private:
    LWNbuffer *buffer;
    LWNbufferAddress address;
    uint64_t* value;
};

#ifdef __cplusplus
}
#endif

#endif // _DEMO_GFX_H_
