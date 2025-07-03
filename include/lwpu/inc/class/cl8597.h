/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2003-2004 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _cl_gt214_tesla_h_
#define _cl_gt214_tesla_h_

/* This file is generated - do not edit. */

#include "lwtypes.h"

#define GT214_TESLA    0x8597

typedef volatile struct _cl8597_tag0 {
    LwU32 SetObject;
    LwU32 Reserved_0x04[0x3F];
    LwU32 NoOperation;
    LwU32 Notify;
    LwU32 Reserved_0x108[0x2];
    LwU32 WaitForIdle;
    LwU32 Reserved_0x114[0xB];
    LwU32 PmTrigger;
    LwU32 SetContextDmaPeerSemaphore;
    LwU32 Reserved_0x148[0xE];
    LwU32 SetContextDmaNotify;
    LwU32 SetCtxDmaZeta;
    LwU32 SetCtxDmaSemaphore;
    LwU32 SetCtxDmaVertex;
    LwU32 SetCtxDmaShaderThreadMemory;
    LwU32 SetCtxDmaShaderThreadStack;
    LwU32 SetCtxDmaShaderProgram;
    LwU32 SetCtxDmaTextureSampler;
    LwU32 SetCtxDmaTextureHeaders;
    LwU32 SetCtxDmaTexture;
    LwU32 SetCtxDmaStreaming;
    LwU32 SetCtxDmaClipId;
    LwU32 Reserved_0x1B0[0x4];
    LwU32 SetCtxDmaColor[0x8];
    LwU32 Reserved_0x1E0[0x8];
    struct {
        LwU32 A;
        LwU32 B;
        LwU32 Format;
        LwU32 BlockSize;
        LwU32 ArrayPitch;
        LwU32 Mark;
        LwU32 Reserved_0x18[0x2];
    } SetCt[0x8];
    struct {
        LwU32 M;
    } SetVertexData1f[0x10];
    struct {
        LwU32 M;
    } SetVertexData2h[0x10];
    struct {
        LwU32 M[0x2];
    } SetVertexData2f[0x10];
    struct {
        LwU32 M[0x3];
        LwU32 Reserved_0x0C[0x1];
    } SetVertexData3f[0x10];
    struct {
        LwU32 M[0x4];
    } SetVertexData4f[0x10];
    struct {
        LwU32 M[0x2];
    } SetVertexData4h[0x10];
    struct {
        LwU32 M;
    } SetVertexData2s[0x10];
    struct {
        LwU32 M;
    } SetVertexDataScaled2s[0x10];
    struct {
        LwU32 M[0x2];
    } SetVertexData4s[0x10];
    struct {
        LwU32 M[0x2];
    } SetVertexDataScaled4s[0x10];
    struct {
        LwU32 M;
    } SetVertexData4ub[0x10];
    struct {
        LwU32 M;
    } SetVertexData4sb[0x10];
    struct {
        LwU32 M;
    } SetVertexDataScaled4ub[0x10];
    struct {
        LwU32 M;
    } SetVertexDataScaled4sb[0x10];
    struct {
        LwU32 Format;
        LwU32 LocationA;
        LwU32 LocationB;
        LwU32 Frequency;
    } SetVertexStream[0x10];
    struct {
        LwU32 ScaleX;
        LwU32 ScaleY;
        LwU32 ScaleZ;
        LwU32 OffsetX;
        LwU32 OffsetY;
        LwU32 OffsetZ;
        LwU32 Reserved_0x18[0x2];
    } SetViewport[0x10];
    struct {
        LwU32 Horizontal;
        LwU32 Vertical;
        LwU32 MinZ;
        LwU32 MaxZ;
    } SetViewportClip[0x10];
    struct {
        LwU32 Horizontal;
        LwU32 Vertical;
    } SetWindowClip[0x8];
    struct {
        LwU32 X;
        LwU32 Y;
    } SetClipIdExtent[0x4];
    LwU32 SetLocalRegisterFileLoadBalanceTimeout;
    LwU32 SetTickControl;
    LwU32 SetStatisticsCounter;
    LwU32 SetClearRectHorizontal;
    LwU32 SetClearRectVertical;
    LwU32 SetVertexArrayStart;
    LwU32 DrawVertexArray;
    LwU32 SetViewportZClip;
    LwU32 SetColorClearValue[0x4];
    LwU32 SetZClearValue;
    LwU32 SetShaderThreadStackA;
    LwU32 SetShaderThreadStackB;
    LwU32 SetShaderThreadStackC;
    LwU32 SetStencilClearValue;
    LwU32 SetStreamingTrigger;
    LwU32 SetStreamingBufferSize;
    LwU32 SetFrontPolygonMode;
    LwU32 SetBackPolygonMode;
    LwU32 SetPolySmooth;
    LwU32 SetZtMark;
    LwU32 SetZlwllDirFormat;
    LwU32 SetPolyOffsetPoint;
    LwU32 SetPolyOffsetLine;
    LwU32 SetPolyOffsetFill;
    LwU32 Reserved_0xDCC[0x1];
    LwU32 SetDaOutputAttributeSkipMask[0x2];
    LwU32 SetZlwllCriterion;
    LwU32 SetDaAttributeCacheLine;
    LwU32 SetPsZombie;
    LwU32 SetSmTimeoutInterval;
    LwU32 SetDaPrimitiveRestartVertexArray;
    LwU32 SetDrawInlineVertexVabUpdate;
    LwU32 SetPrimitivesPerTpc;
    LwU32 SetTickControlEarlyZ;
    LwU32 SetWindowOffsetX;
    LwU32 SetWindowOffsetY;
    struct {
        LwU32 Enable;
        LwU32 Horizontal;
        LwU32 Vertical;
        LwU32 Reserved_0x0C[0x1];
    } SetScissor[0x10];
    LwU32 LoadConstantSelector;
    LwU32 LoadConstant[0x10];
    LwU32 SetShaderThreadMemoryThrottle;
    LwU32 SetShaderThreadMemoryThrottleControl;
    LwU32 SetShaderThreadStackThrottle;
    LwU32 SetShaderThreadStackThrottleControl;
    LwU32 SetBackStencilFuncRef;
    LwU32 SetBackStencilMask;
    LwU32 SetBackStencilFuncMask;
    LwU32 SetSlwllOccludersX0Y0;
    LwU32 SetSlwllOccludersX4kY0;
    LwU32 SetSlwllOccludersX0Y4k;
    LwU32 SetSlwllOccludersX4kY4k;
    LwU32 SetGsProgramA;
    LwU32 SetGsProgramB;
    LwU32 SetLwmcovgControl;
    LwU32 SetVsProgramA;
    LwU32 SetVsProgramB;
    LwU32 SetVertexStreamSubstituteA;
    LwU32 SetVertexStreamSubstituteB;
    LwU32 SetLineModePolygonClip;
    LwU32 SetSingleCtWriteControl;
    LwU32 SetLwbemapAddressModeOverride;
    LwU32 SetFrstrPerformanceControl;
    LwU32 SetDepthBoundsMin;
    LwU32 SetDepthBoundsMax;
    LwU32 SetPsProgramA;
    LwU32 SetPsProgramB;
    LwU32 SetInterTpcArbitrationControl;
    LwU32 SetNonmultisampledZ;
    LwU32 SetSwath;
    LwU32 SetTpcMaskWait;
    LwU32 SetSampleMaskX0Y0;
    LwU32 SetSampleMaskX1Y0;
    LwU32 SetSampleMaskX0Y1;
    LwU32 SetSampleMaskX1Y1;
    LwU32 SetSurfaceClipIdMemoryA;
    LwU32 SetSurfaceClipIdMemoryB;
    LwU32 SetAttributeViewportIndexSlot;
    LwU32 SetDaAttributeSchedulerPolicy;
    LwU32 SetBlendOptControl;
    LwU32 SetZtA;
    LwU32 SetZtB;
    LwU32 SetZtFormat;
    LwU32 SetZtBlockSize;
    LwU32 SetZtArrayPitch;
    LwU32 SetSurfaceClipHorizontal;
    LwU32 SetSurfaceClipVertical;
    LwU32 SetPlanarQuadClip;
    LwU32 SetVertexStreamInstance[0x10];
    LwU32 SetVsIbufAllocation;
    LwU32 SetGsIbufAllocation;
    LwU32 SetSpareNoop02;
    LwU32 SetSpareNoop03;
    LwU32 SetSpareNoop04;
    LwU32 SetSpareNoop05;
    LwU32 SetSpareNoop06;
    LwU32 SetSpareNoop07;
    LwU32 SetSpareNoop08;
    LwU32 SetSpareNoop09;
    LwU32 SetSpareNoop10;
    LwU32 SetSpareNoop11;
    LwU32 SetSpareNoop12;
    LwU32 SetSpareNoop13;
    LwU32 SetInstrumentationMethodHeader;
    LwU32 SetInstrumentationMethodData;
    struct {
        LwU32 A;
        LwU32 B;
    } SetVertexStreamLimit[0x10];
    LwU32 SetLineQuality;
    LwU32 SetZlwllRegion;
    LwU32 SetZlwllStatsToSm;
    LwU32 NoOperationDataHi;
    LwU32 SetDepthBiasControl;
    LwU32 PmTriggerEnd;
    LwU32 SetVertexIdBase;
    LwU32 IncrementPrimitiveId;
    LwU32 SetDaOutputAttributeSkipMaskA[0x2];
    LwU32 SetDaOutputAttributeSkipMaskB[0x2];
    LwU32 SetDaOutputAttributeMaskA[0x2];
    LwU32 SetDaOutputAttributeMaskB[0x2];
    LwU32 SetBlendPerFormatEnable;
    LwU32 FlushPendingWrites;
    LwU32 Reserved_0x1148[0x1];
    LwU32 SetVabDataControl;
    LwU32 SetVabData[0x4];
    LwU32 SetVertexAttributeA[0x10];
    LwU32 SetVertexAttributeB[0x10];
    LwU32 Reserved_0x11E0[0xE];
    LwU32 SetKind3dCtCheckEnable;
    LwU32 SetCtSelect;
    LwU32 SetCompressionThreshold;
    LwU32 SetCtSizeC;
    LwU32 SetZtSizeA;
    LwU32 SetZtSizeB;
    LwU32 SetZtSizeC;
    LwU32 SetSamplerBinding;
    LwU32 PrefetchTextureSampler;
    LwU32 DrawAuto;
    struct {
        LwU32 A;
        LwU32 B;
    } SetCtSize[0x8];
    LwU32 LoadConstantBufferTableA;
    LwU32 LoadConstantBufferTableB;
    LwU32 LoadConstantBufferTableC;
    LwU32 SetTpcScreenSpacePartition;
    LwU32 SetApiCallLimit;
    LwU32 SetStreamingControl;
    LwU32 SetPsOutputRegister;
    LwU32 SetVsWait;
    LwU32 LoadThreadBalanceControl;
    LwU32 Reserved_0x12A4[0x1];
    LwU32 SetShaderL1CacheControl;
    LwU32 SetShaderScheduling;
    LwU32 LoadLocalRegisterFileLoadBalanceControlA;
    LwU32 LoadLocalRegisterFileLoadBalanceControlB;
    LwU32 LoadLocalRegisterFileLoadBalanceControlC;
    LwU32 SetMemorySurfaceRoot;
    LwU32 SetMemorySurface;
    LwU32 SetMemorySurfaceAttr;
    LwU32 SetBoundingBoxLwll;
    LwU32 SetDepthTest;
    LwU32 SetFillMode;
    LwU32 SetShadeMode;
    LwU32 SetShaderThreadMemoryA;
    LwU32 SetShaderThreadMemoryB;
    LwU32 SetShaderThreadMemoryC;
    LwU32 SetBlendStatePerTarget;
    LwU32 SetDepthWrite;
    LwU32 SetAlphaTest;
    LwU32 SetShaderPerformanceCounterValue[0x4];
    LwU32 SetInlineIndex4x8Align;
    LwU32 DrawInlineIndex4x8;
    LwU32 D3dSetLwllMode;
    LwU32 SetDepthFunc;
    LwU32 SetAlphaRef;
    LwU32 SetAlphaFunc;
    LwU32 SetDrawAutoStride;
    LwU32 SetBlendConstRed;
    LwU32 SetBlendConstGreen;
    LwU32 SetBlendConstBlue;
    LwU32 SetBlendConstAlpha;
    LwU32 SetShaderPerformanceCounterTrapControl;
    LwU32 IlwalidateSamplerCache;
    LwU32 IlwalidateTextureHeaderCache;
    LwU32 IlwalidateTextureDataCache;
    LwU32 SetBlendSeparateForAlpha;
    LwU32 SetBlendColorOp;
    LwU32 SetBlendColorSourceCoeff;
    LwU32 SetBlendColorDestCoeff;
    LwU32 SetBlendAlphaOp;
    LwU32 SetBlendAlphaSourceCoeff;
    LwU32 SetGlobalColorKey;
    LwU32 SetBlendAlphaDestCoeff;
    LwU32 SetSingleRopControl;
    LwU32 SetBlend[0x8];
    LwU32 SetStencilTest;
    LwU32 SetStencilOpFail;
    LwU32 SetStencilOpZfail;
    LwU32 SetStencilOpZpass;
    LwU32 SetStencilFunc;
    LwU32 SetStencilFuncRef;
    LwU32 SetStencilMask;
    LwU32 SetStencilFuncMask;
    LwU32 SetGeomThreadLoad;
    LwU32 SetDrawAutoStart;
    LwU32 SetPsSaturate;
    LwU32 SetWindowOrigin;
    LwU32 SetLineWidthFloat;
    LwU32 SetVsTexture;
    LwU32 SetGsTexture;
    LwU32 SetPsTexture;
    LwU32 SetPointSpriteAttribute[0x10];
    LwU32 SetPsWait;
    LwU32 SetNonPsWait;
    LwU32 SetEarlyZHysteresis;
    LwU32 SetVsProgramStart;
    LwU32 SetGsProgramStart;
    LwU32 SetPsProgramStart;
    LwU32 SetLineMultisampleOverride;
    LwU32 SetIbufAllocation;
    LwU32 SetGsMaxOutputVertexCount;
    LwU32 PeerSemaphoreReleaseOffset;
    LwU32 PeerSemaphoreRelease;
    LwU32 IlwalidateDaDmaCache;
    LwU32 SetReduceDstColor;
    LwU32 SetGlobalBaseVertexIndex;
    LwU32 SetGlobalBaseInstanceIndex;
    LwU32 SetClearControl;
    LwU32 IlwalidateShaderCache;
    LwU32 BindVsTextureSampler;
    LwU32 BindVsTextureHeader;
    LwU32 BindGsTextureSampler;
    LwU32 BindGsTextureHeader;
    LwU32 BindPsTextureSampler;
    LwU32 BindPsTextureHeader;
    LwU32 Reserved_0x145C[0x3];
    LwU32 BindVsExtraTextureSampler;
    LwU32 BindVsExtraTextureHeader;
    LwU32 BindGsExtraTextureSampler;
    LwU32 BindGsExtraTextureHeader;
    LwU32 BindPsExtraTextureSampler;
    LwU32 BindPsExtraTextureHeader;
    LwU32 SetStreamingReorder[0x20];
    LwU32 Reserved_0x1500[0x1];
    LwU32 SetSurfaceClipIdHeight;
    LwU32 SetClipIdClearRectHorizontal;
    LwU32 SetClipIdClearRectVertical;
    LwU32 SetUserClipEnable;
    LwU32 SetZpassPixelCount;
    LwU32 SetPointSize;
    LwU32 SetZlwllStats;
    LwU32 SetPointSprite;
    LwU32 Reserved_0x1524[0x1];
    LwU32 SetShaderExceptions;
    LwU32 LoadLocalRegisterFileLoadBalanceControlD;
    LwU32 ClearReportValue;
    LwU32 SetAntiAliasEnable;
    LwU32 SetZtSelect;
    LwU32 SetAntiAliasAlphaControl;
    LwU32 SetPsInputInterpolationType[0x4];
    LwU32 SetRenderEnableA;
    LwU32 SetRenderEnableB;
    LwU32 SetRenderEnableC;
    LwU32 SetTexSamplerPoolA;
    LwU32 SetTexSamplerPoolB;
    LwU32 SetTexSamplerPoolC;
    LwU32 SetShaderErrorTrapControl;
    LwU32 SetSlopeScaleDepthBias;
    LwU32 SetAntiAliasedLine;
    LwU32 SetTexHeaderPoolA;
    LwU32 SetTexHeaderPoolB;
    LwU32 SetTexHeaderPoolC;
    LwU32 SetShaderPerformanceCounterControl[0x4];
    LwU32 SetActiveZlwllRegion;
    LwU32 SetTwoSidedStencilTest;
    LwU32 SetBackStencilOpFail;
    LwU32 SetBackStencilOpZfail;
    LwU32 SetBackStencilOpZpass;
    LwU32 SetBackStencilFunc;
    LwU32 SetPhaseIdControl;
    LwU32 SetMrtPerformanceControl;
    LwU32 PrefetchShaderInstructions;
    LwU32 SetVcaa;
    LwU32 SetSrgbWrite;
    LwU32 SetDepthBias;
    LwU32 Reserved_0x15C0[0x1];
    LwU32 SetFp32BlendRounding;
    LwU32 SetVcaaSampleMaskInteraction;
    LwU32 SetRtLayer;
    LwU32 SetAntiAlias;
    LwU32 D3dBegin;
    LwU32 D3dEnd;
    LwU32 OglBegin;
    LwU32 OglEnd;
    LwU32 SetEdgeFlag;
    LwU32 DrawInlineIndex;
    LwU32 SetInlineIndex2x16Align;
    LwU32 DrawInlineIndex2x16;
    LwU32 SetVertexGlobalBaseOffsetA;
    LwU32 SetVertexGlobalBaseOffsetB;
    LwU32 Reserved_0x15FC[0x11];
    LwU32 DrawInlineVertex;
    LwU32 SetDaPrimitiveRestart;
    LwU32 SetDaPrimitiveRestartIndex;
    LwU32 SetDaOutput;
    LwU32 SetDaOutputAttributeMask[0x2];
    LwU32 SetAntiAliasedPoint;
    LwU32 SetPointCenterMode;
    LwU32 SetPointSpriteControl;
    LwU32 SetLwbemapInterFaceFiltering;
    LwU32 SetLineSmoothParameters;
    LwU32 SetLineStipple;
    LwU32 SetLineSmoothEdgeTable[0x4];
    LwU32 SetLineStippleParameters;
    LwU32 SetProvokingVertex;
    LwU32 SetTwoSidedLight;
    LwU32 SetPolygonStipple;
    LwU32 SetShaderControl;
    LwU32 BindConstantBuffer;
    LwU32 SetShaderAddressRegister;
    LwU32 SetHybridAntiAliasControl;
    LwU32 SetAlphaToCoverageDitherControl;
    LwU32 Reserved_0x16A4[0x1];
    LwU32 SetShaderNanSaturation;
    LwU32 SetVsOutputCount;
    LwU32 SetVsRegisterCount;
    LwU32 SetAlphaToCoverageOverride;
    LwU32 SetVsOutbufCount;
    LwU32 SetVsOutputPosition;
    LwU32 SetVsOutputReorder[0x10];
    LwU32 SetPolygonStipplePattern[0x20];
    LwU32 SetStreamingStartOffset[0x4];
    LwU32 Reserved_0x1790[0x2];
    LwU32 SetGsEnable;
    LwU32 Reserved_0x179C[0x1];
    LwU32 SetGsRegisterCount;
    LwU32 Reserved_0x17A4[0x1];
    LwU32 SetGsOutbufCount;
    LwU32 SetGsOutputCount;
    LwU32 SetGsOutputTopology;
    LwU32 SetPipelineOutput;
    LwU32 SetStreamingOutput;
    LwU32 Reserved_0x17BC[0x10];
    LwU32 SetGsOutputPosition;
    LwU32 SetGsOutputReorder[0x1F];
    LwU32 SetDepthBiasClamp;
    LwU32 SetVertexStreamInstanceA[0x10];
    LwU32 SetVertexStreamInstanceB[0x10];
    LwU32 SetAttributeViewportIndex;
    LwU32 SetAttributeColor;
    LwU32 SetAttributeUserClip;
    LwU32 SetAttributeRtArrayIndex;
    LwU32 SetAttributePointSize;
    LwU32 SetAttributePrimitiveId;
    LwU32 OglSetLwll;
    LwU32 OglSetFrontFace;
    LwU32 OglSetLwllFace;
    LwU32 SetViewportPixel;
    LwU32 SetPsSampleMaskOutput;
    LwU32 SetViewportScaleOffset;
    LwU32 Reserved_0x1930[0x3];
    LwU32 SetViewportClipControl;
    LwU32 SetUserClipOp;
    LwU32 DrawZeroIndex;
    LwU32 SetFastPolymode;
    LwU32 SetWindowClipEnable;
    LwU32 SetWindowClipType;
    LwU32 ClearZlwllSurface;
    LwU32 IlwalidateZlwll;
    LwU32 IlwalidateAllZlwllRegions;
    LwU32 SetXbarTickArbitration;
    LwU32 Reserved_0x1964[0x1];
    LwU32 SetZlwll;
    LwU32 SetZlwllBounds;
    LwU32 SetVisibleEarlyZ;
    LwU32 Reserved_0x1974[0x1];
    LwU32 ZlwllSync;
    LwU32 SetClipIdTest;
    LwU32 SetSurfaceClipIdWidth;
    LwU32 SetClipId;
    LwU32 SetPsInput;
    LwU32 SetPsRegisterCount;
    LwU32 Reserved_0x1990[0x4];
    LwU32 SetPsRegisterAllocation;
    LwU32 Reserved_0x19A4[0x1];
    LwU32 SetPsControl;
    LwU32 Reserved_0x19AC[0x4];
    LwU32 SetDepthBoundsTest;
    LwU32 SetBlendFloatOption;
    LwU32 SetLogicOp;
    LwU32 SetLogicOpFunc;
    LwU32 SetZCompression;
    LwU32 ClearSurface;
    LwU32 ClearClipIdSurface;
    LwU32 SetLineSnapGrid;
    LwU32 SetNonLineSnapGrid;
    LwU32 SetColorCompression[0x8];
    LwU32 SetCtWrite[0x8];
    LwU32 SetZPlanePrecision;
    LwU32 TestForQuadro;
    LwU32 SetOctverticesPerTpc;
    LwU32 PipeNop;
    LwU32 SetSpare00;
    LwU32 SetSpare01;
    LwU32 SetSpare02;
    LwU32 SetSpare03;
    struct {
        LwU32 Entries[0x4];
    } SetAnisoAngleTable[0x4];
    struct {
        LwU32 A;
        LwU32 B;
        LwU32 C;
        LwU32 BufferBytes;
    } SetStreaming[0x4];
    LwU32 SetVertexAttrib[0x10];
    LwU32 SetReportSemaphoreA;
    LwU32 SetReportSemaphoreB;
    LwU32 SetReportSemaphoreC;
    LwU32 SetReportSemaphoreD;
    LwU32 Reserved_0x1B10[0xB];
    LwU32 SetVsOutputReorderA;
    LwU32 SetVsOutputReorderB[0x10];
    LwU32 SetVsOutputReorderC[0xF];
    LwU32 Reserved_0x1BBC[0x11];
    struct {
        LwU32 Format;
        LwU32 LocationA;
        LwU32 LocationB;
        LwU32 Frequency;
    } SetVertexStreamA[0x10];
    struct {
        LwU32 Format;
        LwU32 LocationA;
        LwU32 LocationB;
        LwU32 Frequency;
    } SetVertexStreamB[0x10];
    struct {
        LwU32 SeparateForAlpha;
        LwU32 ColorOp;
        LwU32 ColorSourceCoeff;
        LwU32 ColorDestCoeff;
        LwU32 AlphaOp;
        LwU32 AlphaSourceCoeff;
        LwU32 AlphaDestCoeff;
        LwU32 Reserved_0x1C[0x1];
    } SetBlendPerTarget[0x8];
    struct {
        LwU32 A;
        LwU32 B;
    } SetVertexStreamLimitA[0x10];
    struct {
        LwU32 A;
        LwU32 B;
    } SetVertexStreamLimitB[0x10];
} gt214_tesla_t;


#define LW8597_SET_OBJECT                                                                                  0x0000
#define LW8597_SET_OBJECT_POINTER                                                                            15:0

#define LW8597_NO_OPERATION                                                                                0x0100
#define LW8597_NO_OPERATION_V                                                                                31:0

#define LW8597_NOTIFY                                                                                      0x0104
#define LW8597_NOTIFY_TYPE                                                                                   31:0
#define LW8597_NOTIFY_TYPE_WRITE_ONLY                                                                  0x00000000
#define LW8597_NOTIFY_TYPE_WRITE_THEN_AWAKEN                                                           0x00000001

#define LW8597_WAIT_FOR_IDLE                                                                               0x0110
#define LW8597_WAIT_FOR_IDLE_V                                                                               31:0

#define LW8597_PM_TRIGGER                                                                                  0x0140
#define LW8597_PM_TRIGGER_V                                                                                  31:0

#define LW8597_SET_CONTEXT_DMA_PEER_SEMAPHORE                                                              0x0144
#define LW8597_SET_CONTEXT_DMA_PEER_SEMAPHORE_V                                                              31:0

#define LW8597_SET_CONTEXT_DMA_NOTIFY                                                                      0x0180
#define LW8597_SET_CONTEXT_DMA_NOTIFY_HANDLE                                                                 31:0

#define LW8597_SET_CTX_DMA_ZETA                                                                            0x0184
#define LW8597_SET_CTX_DMA_ZETA_HANDLE                                                                       31:0

#define LW8597_SET_CTX_DMA_SEMAPHORE                                                                       0x0188
#define LW8597_SET_CTX_DMA_SEMAPHORE_HANDLE                                                                  31:0

#define LW8597_SET_CTX_DMA_VERTEX                                                                          0x018c
#define LW8597_SET_CTX_DMA_VERTEX_HANDLE                                                                     31:0

#define LW8597_SET_CTX_DMA_SHADER_THREAD_MEMORY                                                            0x0190
#define LW8597_SET_CTX_DMA_SHADER_THREAD_MEMORY_HANDLE                                                       31:0

#define LW8597_SET_CTX_DMA_SHADER_THREAD_STACK                                                             0x0194
#define LW8597_SET_CTX_DMA_SHADER_THREAD_STACK_HANDLE                                                        31:0

#define LW8597_SET_CTX_DMA_SHADER_PROGRAM                                                                  0x0198
#define LW8597_SET_CTX_DMA_SHADER_PROGRAM_HANDLE                                                             31:0

#define LW8597_SET_CTX_DMA_TEXTURE_SAMPLER                                                                 0x019c
#define LW8597_SET_CTX_DMA_TEXTURE_SAMPLER_HANDLE                                                            31:0

#define LW8597_SET_CTX_DMA_TEXTURE_HEADERS                                                                 0x01a0
#define LW8597_SET_CTX_DMA_TEXTURE_HEADERS_HANDLE                                                            31:0

#define LW8597_SET_CTX_DMA_TEXTURE                                                                         0x01a4
#define LW8597_SET_CTX_DMA_TEXTURE_HANDLE                                                                    31:0

#define LW8597_SET_CTX_DMA_STREAMING                                                                       0x01a8
#define LW8597_SET_CTX_DMA_STREAMING_HANDLE                                                                  31:0

#define LW8597_SET_CTX_DMA_CLIP_ID                                                                         0x01ac
#define LW8597_SET_CTX_DMA_CLIP_ID_HANDLE                                                                    31:0

#define LW8597_SET_CTX_DMA_COLOR(i)                                                                (0x01c0+(i)*4)
#define LW8597_SET_CTX_DMA_COLOR_HANDLE                                                                      31:0

#define LW8597_SET_CT_A(j)                                                                        (0x0200+(j)*32)
#define LW8597_SET_CT_A_OFFSET_UPPER                                                                          7:0

#define LW8597_SET_CT_B(j)                                                                        (0x0204+(j)*32)
#define LW8597_SET_CT_B_OFFSET_LOWER                                                                         31:0

#define LW8597_SET_CT_FORMAT(j)                                                                   (0x0208+(j)*32)
#define LW8597_SET_CT_FORMAT_V                                                                                7:0
#define LW8597_SET_CT_FORMAT_V_DISABLED                                                                0x00000000
#define LW8597_SET_CT_FORMAT_V_RF32_GF32_BF32_AF32                                                     0x000000C0
#define LW8597_SET_CT_FORMAT_V_RS32_GS32_BS32_AS32                                                     0x000000C1
#define LW8597_SET_CT_FORMAT_V_RU32_GU32_BU32_AU32                                                     0x000000C2
#define LW8597_SET_CT_FORMAT_V_RF32_GF32_BF32_X32                                                      0x000000C3
#define LW8597_SET_CT_FORMAT_V_RS32_GS32_BS32_X32                                                      0x000000C4
#define LW8597_SET_CT_FORMAT_V_RU32_GU32_BU32_X32                                                      0x000000C5
#define LW8597_SET_CT_FORMAT_V_R16_G16_B16_A16                                                         0x000000C6
#define LW8597_SET_CT_FORMAT_V_RN16_GN16_BN16_AN16                                                     0x000000C7
#define LW8597_SET_CT_FORMAT_V_RS16_GS16_BS16_AS16                                                     0x000000C8
#define LW8597_SET_CT_FORMAT_V_RU16_GU16_BU16_AU16                                                     0x000000C9
#define LW8597_SET_CT_FORMAT_V_RF16_GF16_BF16_AF16                                                     0x000000CA
#define LW8597_SET_CT_FORMAT_V_RF32_GF32                                                               0x000000CB
#define LW8597_SET_CT_FORMAT_V_RS32_GS32                                                               0x000000CC
#define LW8597_SET_CT_FORMAT_V_RU32_GU32                                                               0x000000CD
#define LW8597_SET_CT_FORMAT_V_RF16_GF16_BF16_X16                                                      0x000000CE
#define LW8597_SET_CT_FORMAT_V_A8R8G8B8                                                                0x000000CF
#define LW8597_SET_CT_FORMAT_V_A8RL8GL8BL8                                                             0x000000D0
#define LW8597_SET_CT_FORMAT_V_A2B10G10R10                                                             0x000000D1
#define LW8597_SET_CT_FORMAT_V_AU2BU10GU10RU10                                                         0x000000D2
#define LW8597_SET_CT_FORMAT_V_A8B8G8R8                                                                0x000000D5
#define LW8597_SET_CT_FORMAT_V_A8BL8GL8RL8                                                             0x000000D6
#define LW8597_SET_CT_FORMAT_V_AN8BN8GN8RN8                                                            0x000000D7
#define LW8597_SET_CT_FORMAT_V_AS8BS8GS8RS8                                                            0x000000D8
#define LW8597_SET_CT_FORMAT_V_AU8BU8GU8RU8                                                            0x000000D9
#define LW8597_SET_CT_FORMAT_V_R16_G16                                                                 0x000000DA
#define LW8597_SET_CT_FORMAT_V_RN16_GN16                                                               0x000000DB
#define LW8597_SET_CT_FORMAT_V_RS16_GS16                                                               0x000000DC
#define LW8597_SET_CT_FORMAT_V_RU16_GU16                                                               0x000000DD
#define LW8597_SET_CT_FORMAT_V_RF16_GF16                                                               0x000000DE
#define LW8597_SET_CT_FORMAT_V_A2R10G10B10                                                             0x000000DF
#define LW8597_SET_CT_FORMAT_V_BF10GF11RF11                                                            0x000000E0
#define LW8597_SET_CT_FORMAT_V_RS32                                                                    0x000000E3
#define LW8597_SET_CT_FORMAT_V_RU32                                                                    0x000000E4
#define LW8597_SET_CT_FORMAT_V_RF32                                                                    0x000000E5
#define LW8597_SET_CT_FORMAT_V_X8R8G8B8                                                                0x000000E6
#define LW8597_SET_CT_FORMAT_V_X8RL8GL8BL8                                                             0x000000E7
#define LW8597_SET_CT_FORMAT_V_R5G6B5                                                                  0x000000E8
#define LW8597_SET_CT_FORMAT_V_A1R5G5B5                                                                0x000000E9
#define LW8597_SET_CT_FORMAT_V_G8R8                                                                    0x000000EA
#define LW8597_SET_CT_FORMAT_V_GN8RN8                                                                  0x000000EB
#define LW8597_SET_CT_FORMAT_V_GS8RS8                                                                  0x000000EC
#define LW8597_SET_CT_FORMAT_V_GU8RU8                                                                  0x000000ED
#define LW8597_SET_CT_FORMAT_V_R16                                                                     0x000000EE
#define LW8597_SET_CT_FORMAT_V_RN16                                                                    0x000000EF
#define LW8597_SET_CT_FORMAT_V_RS16                                                                    0x000000F0
#define LW8597_SET_CT_FORMAT_V_RU16                                                                    0x000000F1
#define LW8597_SET_CT_FORMAT_V_RF16                                                                    0x000000F2
#define LW8597_SET_CT_FORMAT_V_R8                                                                      0x000000F3
#define LW8597_SET_CT_FORMAT_V_RN8                                                                     0x000000F4
#define LW8597_SET_CT_FORMAT_V_RS8                                                                     0x000000F5
#define LW8597_SET_CT_FORMAT_V_RU8                                                                     0x000000F6
#define LW8597_SET_CT_FORMAT_V_A8                                                                      0x000000F7
#define LW8597_SET_CT_FORMAT_V_X1R5G5B5                                                                0x000000F8
#define LW8597_SET_CT_FORMAT_V_X8B8G8R8                                                                0x000000F9
#define LW8597_SET_CT_FORMAT_V_X8BL8GL8RL8                                                             0x000000FA
#define LW8597_SET_CT_FORMAT_V_Z1R5G5B5                                                                0x000000FB
#define LW8597_SET_CT_FORMAT_V_O1R5G5B5                                                                0x000000FC
#define LW8597_SET_CT_FORMAT_V_Z8R8G8B8                                                                0x000000FD
#define LW8597_SET_CT_FORMAT_V_O8R8G8B8                                                                0x000000FE
#define LW8597_SET_CT_FORMAT_V_R32                                                                     0x000000FF

#define LW8597_SET_CT_BLOCK_SIZE(j)                                                               (0x020c+(j)*32)
#define LW8597_SET_CT_BLOCK_SIZE_WIDTH                                                                        3:0
#define LW8597_SET_CT_BLOCK_SIZE_WIDTH_ONE_GOB                                                         0x00000000
#define LW8597_SET_CT_BLOCK_SIZE_WIDTH_TWO_GOBS                                                        0x00000001
#define LW8597_SET_CT_BLOCK_SIZE_HEIGHT                                                                       7:4
#define LW8597_SET_CT_BLOCK_SIZE_HEIGHT_ONE_GOB                                                        0x00000000
#define LW8597_SET_CT_BLOCK_SIZE_HEIGHT_TWO_GOBS                                                       0x00000001
#define LW8597_SET_CT_BLOCK_SIZE_HEIGHT_FOUR_GOBS                                                      0x00000002
#define LW8597_SET_CT_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                                                     0x00000003
#define LW8597_SET_CT_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                                                   0x00000004
#define LW8597_SET_CT_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                                                 0x00000005
#define LW8597_SET_CT_BLOCK_SIZE_DEPTH                                                                       11:8
#define LW8597_SET_CT_BLOCK_SIZE_DEPTH_ONE_GOB                                                         0x00000000
#define LW8597_SET_CT_BLOCK_SIZE_DEPTH_TWO_GOBS                                                        0x00000001
#define LW8597_SET_CT_BLOCK_SIZE_DEPTH_FOUR_GOBS                                                       0x00000002
#define LW8597_SET_CT_BLOCK_SIZE_DEPTH_EIGHT_GOBS                                                      0x00000003
#define LW8597_SET_CT_BLOCK_SIZE_DEPTH_SIXTEEN_GOBS                                                    0x00000004
#define LW8597_SET_CT_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS                                                  0x00000005

#define LW8597_SET_CT_ARRAY_PITCH(j)                                                              (0x0210+(j)*32)
#define LW8597_SET_CT_ARRAY_PITCH_V                                                                          31:0

#define LW8597_SET_CT_MARK(j)                                                                     (0x0214+(j)*32)
#define LW8597_SET_CT_MARK_IEEE_CLEAN                                                                         0:0
#define LW8597_SET_CT_MARK_IEEE_CLEAN_FALSE                                                            0x00000000
#define LW8597_SET_CT_MARK_IEEE_CLEAN_TRUE                                                             0x00000001

#define LW8597_SET_VERTEX_DATA1F_M(j)                                                              (0x0300+(j)*4)
#define LW8597_SET_VERTEX_DATA1F_M_V                                                                         31:0

#define LW8597_SET_VERTEX_DATA2H_M(j)                                                              (0x0340+(j)*4)
#define LW8597_SET_VERTEX_DATA2H_M_V0                                                                        15:0
#define LW8597_SET_VERTEX_DATA2H_M_V1                                                                       31:16

#define LW8597_SET_VERTEX_DATA2F_M(i,j)                                                      (0x0380+(i)*8+(j)*4)
#define LW8597_SET_VERTEX_DATA2F_M_V                                                                         31:0

#define LW8597_SET_VERTEX_DATA3F_M(i,j)                                                     (0x0400+(i)*16+(j)*4)
#define LW8597_SET_VERTEX_DATA3F_M_V                                                                         31:0

#define LW8597_SET_VERTEX_DATA4F_M(i,j)                                                     (0x0500+(i)*16+(j)*4)
#define LW8597_SET_VERTEX_DATA4F_M_V                                                                         31:0

#define LW8597_SET_VERTEX_DATA4H_M(i,j)                                                      (0x0600+(i)*8+(j)*4)
#define LW8597_SET_VERTEX_DATA4H_M_V0                                                                        15:0
#define LW8597_SET_VERTEX_DATA4H_M_V1                                                                       31:16

#define LW8597_SET_VERTEX_DATA2S_M(j)                                                              (0x0680+(j)*4)
#define LW8597_SET_VERTEX_DATA2S_M_V0                                                                        15:0
#define LW8597_SET_VERTEX_DATA2S_M_V1                                                                       31:16

#define LW8597_SET_VERTEX_DATA_SCALED2S_M(j)                                                       (0x06c0+(j)*4)
#define LW8597_SET_VERTEX_DATA_SCALED2S_M_V0                                                                 15:0
#define LW8597_SET_VERTEX_DATA_SCALED2S_M_V1                                                                31:16

#define LW8597_SET_VERTEX_DATA4S_M(i,j)                                                      (0x0700+(i)*8+(j)*4)
#define LW8597_SET_VERTEX_DATA4S_M_V0                                                                        15:0
#define LW8597_SET_VERTEX_DATA4S_M_V1                                                                       31:16

#define LW8597_SET_VERTEX_DATA_SCALED4S_M(i,j)                                               (0x0780+(i)*8+(j)*4)
#define LW8597_SET_VERTEX_DATA_SCALED4S_M_V0                                                                 15:0
#define LW8597_SET_VERTEX_DATA_SCALED4S_M_V1                                                                31:16

#define LW8597_SET_VERTEX_DATA4UB_M(j)                                                             (0x0800+(j)*4)
#define LW8597_SET_VERTEX_DATA4UB_M_V0                                                                        7:0
#define LW8597_SET_VERTEX_DATA4UB_M_V1                                                                       15:8
#define LW8597_SET_VERTEX_DATA4UB_M_V2                                                                      23:16
#define LW8597_SET_VERTEX_DATA4UB_M_V3                                                                      31:24

#define LW8597_SET_VERTEX_DATA4SB_M(j)                                                             (0x0840+(j)*4)
#define LW8597_SET_VERTEX_DATA4SB_M_V0                                                                        7:0
#define LW8597_SET_VERTEX_DATA4SB_M_V1                                                                       15:8
#define LW8597_SET_VERTEX_DATA4SB_M_V2                                                                      23:16
#define LW8597_SET_VERTEX_DATA4SB_M_V3                                                                      31:24

#define LW8597_SET_VERTEX_DATA_SCALED4UB_M(j)                                                      (0x0880+(j)*4)
#define LW8597_SET_VERTEX_DATA_SCALED4UB_M_V0                                                                 7:0
#define LW8597_SET_VERTEX_DATA_SCALED4UB_M_V1                                                                15:8
#define LW8597_SET_VERTEX_DATA_SCALED4UB_M_V2                                                               23:16
#define LW8597_SET_VERTEX_DATA_SCALED4UB_M_V3                                                               31:24

#define LW8597_SET_VERTEX_DATA_SCALED4SB_M(j)                                                      (0x08c0+(j)*4)
#define LW8597_SET_VERTEX_DATA_SCALED4SB_M_V0                                                                 7:0
#define LW8597_SET_VERTEX_DATA_SCALED4SB_M_V1                                                                15:8
#define LW8597_SET_VERTEX_DATA_SCALED4SB_M_V2                                                               23:16
#define LW8597_SET_VERTEX_DATA_SCALED4SB_M_V3                                                               31:24

#define LW8597_SET_VERTEX_STREAM_FORMAT(j)                                                        (0x0900+(j)*16)
#define LW8597_SET_VERTEX_STREAM_FORMAT_STRIDE                                                               11:0
#define LW8597_SET_VERTEX_STREAM_FORMAT_ENABLE                                                              29:29
#define LW8597_SET_VERTEX_STREAM_FORMAT_ENABLE_FALSE                                                   0x00000000
#define LW8597_SET_VERTEX_STREAM_FORMAT_ENABLE_TRUE                                                    0x00000001

#define LW8597_SET_VERTEX_STREAM_LOCATION_A(j)                                                    (0x0904+(j)*16)
#define LW8597_SET_VERTEX_STREAM_LOCATION_A_OFFSET_UPPER                                                      7:0

#define LW8597_SET_VERTEX_STREAM_LOCATION_B(j)                                                    (0x0908+(j)*16)
#define LW8597_SET_VERTEX_STREAM_LOCATION_B_OFFSET_LOWER                                                     31:0

#define LW8597_SET_VERTEX_STREAM_FREQUENCY(j)                                                     (0x090c+(j)*16)
#define LW8597_SET_VERTEX_STREAM_FREQUENCY_V                                                                 31:0

#define LW8597_SET_VIEWPORT_SCALE_X(j)                                                            (0x0a00+(j)*32)
#define LW8597_SET_VIEWPORT_SCALE_X_V                                                                        31:0

#define LW8597_SET_VIEWPORT_SCALE_Y(j)                                                            (0x0a04+(j)*32)
#define LW8597_SET_VIEWPORT_SCALE_Y_V                                                                        31:0

#define LW8597_SET_VIEWPORT_SCALE_Z(j)                                                            (0x0a08+(j)*32)
#define LW8597_SET_VIEWPORT_SCALE_Z_V                                                                        31:0

#define LW8597_SET_VIEWPORT_OFFSET_X(j)                                                           (0x0a0c+(j)*32)
#define LW8597_SET_VIEWPORT_OFFSET_X_V                                                                       31:0

#define LW8597_SET_VIEWPORT_OFFSET_Y(j)                                                           (0x0a10+(j)*32)
#define LW8597_SET_VIEWPORT_OFFSET_Y_V                                                                       31:0

#define LW8597_SET_VIEWPORT_OFFSET_Z(j)                                                           (0x0a14+(j)*32)
#define LW8597_SET_VIEWPORT_OFFSET_Z_V                                                                       31:0

#define LW8597_SET_VIEWPORT_CLIP_HORIZONTAL(j)                                                    (0x0c00+(j)*16)
#define LW8597_SET_VIEWPORT_CLIP_HORIZONTAL_X0                                                               15:0
#define LW8597_SET_VIEWPORT_CLIP_HORIZONTAL_WIDTH                                                           31:16

#define LW8597_SET_VIEWPORT_CLIP_VERTICAL(j)                                                      (0x0c04+(j)*16)
#define LW8597_SET_VIEWPORT_CLIP_VERTICAL_Y0                                                                 15:0
#define LW8597_SET_VIEWPORT_CLIP_VERTICAL_HEIGHT                                                            31:16

#define LW8597_SET_VIEWPORT_CLIP_MIN_Z(j)                                                         (0x0c08+(j)*16)
#define LW8597_SET_VIEWPORT_CLIP_MIN_Z_V                                                                     31:0

#define LW8597_SET_VIEWPORT_CLIP_MAX_Z(j)                                                         (0x0c0c+(j)*16)
#define LW8597_SET_VIEWPORT_CLIP_MAX_Z_V                                                                     31:0

#define LW8597_SET_WINDOW_CLIP_HORIZONTAL(j)                                                       (0x0d00+(j)*8)
#define LW8597_SET_WINDOW_CLIP_HORIZONTAL_XMIN                                                               15:0
#define LW8597_SET_WINDOW_CLIP_HORIZONTAL_XMAX                                                              31:16

#define LW8597_SET_WINDOW_CLIP_VERTICAL(j)                                                         (0x0d04+(j)*8)
#define LW8597_SET_WINDOW_CLIP_VERTICAL_YMIN                                                                 15:0
#define LW8597_SET_WINDOW_CLIP_VERTICAL_YMAX                                                                31:16

#define LW8597_SET_CLIP_ID_EXTENT_X(j)                                                             (0x0d40+(j)*8)
#define LW8597_SET_CLIP_ID_EXTENT_X_MINX                                                                     15:0
#define LW8597_SET_CLIP_ID_EXTENT_X_WIDTH                                                                   31:16

#define LW8597_SET_CLIP_ID_EXTENT_Y(j)                                                             (0x0d44+(j)*8)
#define LW8597_SET_CLIP_ID_EXTENT_Y_MINY                                                                     15:0
#define LW8597_SET_CLIP_ID_EXTENT_Y_HEIGHT                                                                  31:16

#define LW8597_SET_LOCAL_REGISTER_FILE_LOAD_BALANCE_TIMEOUT                                                0x0d60
#define LW8597_SET_LOCAL_REGISTER_FILE_LOAD_BALANCE_TIMEOUT_V                                                15:0

#define LW8597_SET_TICK_CONTROL                                                                            0x0d64
#define LW8597_SET_TICK_CONTROL_Z_ENABLE                                                                      0:0
#define LW8597_SET_TICK_CONTROL_Z_ENABLE_FALSE                                                         0x00000000
#define LW8597_SET_TICK_CONTROL_Z_ENABLE_TRUE                                                          0x00000001
#define LW8597_SET_TICK_CONTROL_COLOR_ENABLE                                                                  1:1
#define LW8597_SET_TICK_CONTROL_COLOR_ENABLE_FALSE                                                     0x00000000
#define LW8597_SET_TICK_CONTROL_COLOR_ENABLE_TRUE                                                      0x00000001
#define LW8597_SET_TICK_CONTROL_MAX_TICK_COUNT                                                                5:2
#define LW8597_SET_TICK_CONTROL_ACLWM_TIMEOUT                                                                13:6
#define LW8597_SET_TICK_CONTROL_ZBAR_TICK_WINDOW                                                            19:14
#define LW8597_SET_TICK_CONTROL_CBAR_TICK_WINDOW                                                            25:20
#define LW8597_SET_TICK_CONTROL_PUDDLE_AREA                                                                 31:26

#define LW8597_SET_STATISTICS_COUNTER                                                                      0x0d68
#define LW8597_SET_STATISTICS_COUNTER_DA_VERTICES_GENERATED_ENABLE                                            0:0
#define LW8597_SET_STATISTICS_COUNTER_DA_VERTICES_GENERATED_ENABLE_FALSE                               0x00000000
#define LW8597_SET_STATISTICS_COUNTER_DA_VERTICES_GENERATED_ENABLE_TRUE                                0x00000001
#define LW8597_SET_STATISTICS_COUNTER_DA_PRIMITIVES_GENERATED_ENABLE                                          1:1
#define LW8597_SET_STATISTICS_COUNTER_DA_PRIMITIVES_GENERATED_ENABLE_FALSE                             0x00000000
#define LW8597_SET_STATISTICS_COUNTER_DA_PRIMITIVES_GENERATED_ENABLE_TRUE                              0x00000001
#define LW8597_SET_STATISTICS_COUNTER_VS_ILWOCATIONS_ENABLE                                                   2:2
#define LW8597_SET_STATISTICS_COUNTER_VS_ILWOCATIONS_ENABLE_FALSE                                      0x00000000
#define LW8597_SET_STATISTICS_COUNTER_VS_ILWOCATIONS_ENABLE_TRUE                                       0x00000001
#define LW8597_SET_STATISTICS_COUNTER_GS_ILWOCATIONS_ENABLE                                                   3:3
#define LW8597_SET_STATISTICS_COUNTER_GS_ILWOCATIONS_ENABLE_FALSE                                      0x00000000
#define LW8597_SET_STATISTICS_COUNTER_GS_ILWOCATIONS_ENABLE_TRUE                                       0x00000001
#define LW8597_SET_STATISTICS_COUNTER_GS_PRIMITIVES_GENERATED_ENABLE                                          4:4
#define LW8597_SET_STATISTICS_COUNTER_GS_PRIMITIVES_GENERATED_ENABLE_FALSE                             0x00000000
#define LW8597_SET_STATISTICS_COUNTER_GS_PRIMITIVES_GENERATED_ENABLE_TRUE                              0x00000001
#define LW8597_SET_STATISTICS_COUNTER_STREAMING_PRIMITIVES_SUCCEEDED_ENABLE                                   5:5
#define LW8597_SET_STATISTICS_COUNTER_STREAMING_PRIMITIVES_SUCCEEDED_ENABLE_FALSE                      0x00000000
#define LW8597_SET_STATISTICS_COUNTER_STREAMING_PRIMITIVES_SUCCEEDED_ENABLE_TRUE                       0x00000001
#define LW8597_SET_STATISTICS_COUNTER_STREAMING_PRIMITIVES_NEEDED_ENABLE                                      6:6
#define LW8597_SET_STATISTICS_COUNTER_STREAMING_PRIMITIVES_NEEDED_ENABLE_FALSE                         0x00000000
#define LW8597_SET_STATISTICS_COUNTER_STREAMING_PRIMITIVES_NEEDED_ENABLE_TRUE                          0x00000001
#define LW8597_SET_STATISTICS_COUNTER_CLIPPER_ILWOCATIONS_ENABLE                                              7:7
#define LW8597_SET_STATISTICS_COUNTER_CLIPPER_ILWOCATIONS_ENABLE_FALSE                                 0x00000000
#define LW8597_SET_STATISTICS_COUNTER_CLIPPER_ILWOCATIONS_ENABLE_TRUE                                  0x00000001
#define LW8597_SET_STATISTICS_COUNTER_CLIPPER_PRIMITIVES_GENERATED_ENABLE                                     8:8
#define LW8597_SET_STATISTICS_COUNTER_CLIPPER_PRIMITIVES_GENERATED_ENABLE_FALSE                        0x00000000
#define LW8597_SET_STATISTICS_COUNTER_CLIPPER_PRIMITIVES_GENERATED_ENABLE_TRUE                         0x00000001
#define LW8597_SET_STATISTICS_COUNTER_PS_ILWOCATIONS_ENABLE                                                   9:9
#define LW8597_SET_STATISTICS_COUNTER_PS_ILWOCATIONS_ENABLE_FALSE                                      0x00000000
#define LW8597_SET_STATISTICS_COUNTER_PS_ILWOCATIONS_ENABLE_TRUE                                       0x00000001
#define LW8597_SET_STATISTICS_COUNTER_STREAMING_VERTICES_SUCCEEDED_ENABLE                                   10:10
#define LW8597_SET_STATISTICS_COUNTER_STREAMING_VERTICES_SUCCEEDED_ENABLE_FALSE                        0x00000000
#define LW8597_SET_STATISTICS_COUNTER_STREAMING_VERTICES_SUCCEEDED_ENABLE_TRUE                         0x00000001

#define LW8597_SET_CLEAR_RECT_HORIZONTAL                                                                   0x0d6c
#define LW8597_SET_CLEAR_RECT_HORIZONTAL_XMIN                                                                15:0
#define LW8597_SET_CLEAR_RECT_HORIZONTAL_XMAX                                                               31:16

#define LW8597_SET_CLEAR_RECT_VERTICAL                                                                     0x0d70
#define LW8597_SET_CLEAR_RECT_VERTICAL_YMIN                                                                  15:0
#define LW8597_SET_CLEAR_RECT_VERTICAL_YMAX                                                                 31:16

#define LW8597_SET_VERTEX_ARRAY_START                                                                      0x0d74
#define LW8597_SET_VERTEX_ARRAY_START_V                                                                      31:0

#define LW8597_DRAW_VERTEX_ARRAY                                                                           0x0d78
#define LW8597_DRAW_VERTEX_ARRAY_COUNT                                                                       31:0

#define LW8597_SET_VIEWPORT_Z_CLIP                                                                         0x0d7c
#define LW8597_SET_VIEWPORT_Z_CLIP_RANGE                                                                      0:0
#define LW8597_SET_VIEWPORT_Z_CLIP_RANGE_NEGATIVE_W_TO_POSITIVE_W                                      0x00000000
#define LW8597_SET_VIEWPORT_Z_CLIP_RANGE_ZERO_TO_POSITIVE_W                                            0x00000001

#define LW8597_SET_COLOR_CLEAR_VALUE(i)                                                            (0x0d80+(i)*4)
#define LW8597_SET_COLOR_CLEAR_VALUE_V                                                                       31:0

#define LW8597_SET_Z_CLEAR_VALUE                                                                           0x0d90
#define LW8597_SET_Z_CLEAR_VALUE_V                                                                           31:0

#define LW8597_SET_SHADER_THREAD_STACK_A                                                                   0x0d94
#define LW8597_SET_SHADER_THREAD_STACK_A_OFFSET_UPPER                                                         7:0

#define LW8597_SET_SHADER_THREAD_STACK_B                                                                   0x0d98
#define LW8597_SET_SHADER_THREAD_STACK_B_OFFSET_LOWER                                                        31:0

#define LW8597_SET_SHADER_THREAD_STACK_C                                                                   0x0d9c
#define LW8597_SET_SHADER_THREAD_STACK_C_SIZE                                                                 3:0
#define LW8597_SET_SHADER_THREAD_STACK_C_SIZE__0                                                       0x00000000
#define LW8597_SET_SHADER_THREAD_STACK_C_SIZE__1                                                       0x00000001
#define LW8597_SET_SHADER_THREAD_STACK_C_SIZE__2                                                       0x00000002
#define LW8597_SET_SHADER_THREAD_STACK_C_SIZE__4                                                       0x00000003
#define LW8597_SET_SHADER_THREAD_STACK_C_SIZE__8                                                       0x00000004
#define LW8597_SET_SHADER_THREAD_STACK_C_SIZE__16                                                      0x00000005
#define LW8597_SET_SHADER_THREAD_STACK_C_SIZE__32                                                      0x00000006
#define LW8597_SET_SHADER_THREAD_STACK_C_SIZE__64                                                      0x00000007
#define LW8597_SET_SHADER_THREAD_STACK_C_SIZE__128                                                     0x00000008
#define LW8597_SET_SHADER_THREAD_STACK_C_SIZE__256                                                     0x00000009
#define LW8597_SET_SHADER_THREAD_STACK_C_SIZE__512                                                     0x0000000A
#define LW8597_SET_SHADER_THREAD_STACK_C_SIZE__1024                                                    0x0000000B
#define LW8597_SET_SHADER_THREAD_STACK_C_SIZE__2048                                                    0x0000000C
#define LW8597_SET_SHADER_THREAD_STACK_C_SIZE__4096                                                    0x0000000D

#define LW8597_SET_STENCIL_CLEAR_VALUE                                                                     0x0da0
#define LW8597_SET_STENCIL_CLEAR_VALUE_V                                                                      7:0

#define LW8597_SET_STREAMING_TRIGGER                                                                       0x0da4
#define LW8597_SET_STREAMING_TRIGGER_BEGIN                                                                   31:0
#define LW8597_SET_STREAMING_TRIGGER_BEGIN_FALSE                                                       0x00000000
#define LW8597_SET_STREAMING_TRIGGER_BEGIN_TRUE                                                        0x00000001

#define LW8597_SET_STREAMING_BUFFER_SIZE                                                                   0x0da8
#define LW8597_SET_STREAMING_BUFFER_SIZE_PRIMITIVE_COUNT                                                     31:0

#define LW8597_SET_FRONT_POLYGON_MODE                                                                      0x0dac
#define LW8597_SET_FRONT_POLYGON_MODE_V                                                                      31:0
#define LW8597_SET_FRONT_POLYGON_MODE_V_POINT                                                          0x00001B00
#define LW8597_SET_FRONT_POLYGON_MODE_V_LINE                                                           0x00001B01
#define LW8597_SET_FRONT_POLYGON_MODE_V_FILL                                                           0x00001B02

#define LW8597_SET_BACK_POLYGON_MODE                                                                       0x0db0
#define LW8597_SET_BACK_POLYGON_MODE_V                                                                       31:0
#define LW8597_SET_BACK_POLYGON_MODE_V_POINT                                                           0x00001B00
#define LW8597_SET_BACK_POLYGON_MODE_V_LINE                                                            0x00001B01
#define LW8597_SET_BACK_POLYGON_MODE_V_FILL                                                            0x00001B02

#define LW8597_SET_POLY_SMOOTH                                                                             0x0db4
#define LW8597_SET_POLY_SMOOTH_ENABLE                                                                        31:0
#define LW8597_SET_POLY_SMOOTH_ENABLE_FALSE                                                            0x00000000
#define LW8597_SET_POLY_SMOOTH_ENABLE_TRUE                                                             0x00000001

#define LW8597_SET_ZT_MARK                                                                                 0x0db8
#define LW8597_SET_ZT_MARK_IEEE_CLEAN                                                                         0:0
#define LW8597_SET_ZT_MARK_IEEE_CLEAN_FALSE                                                            0x00000000
#define LW8597_SET_ZT_MARK_IEEE_CLEAN_TRUE                                                             0x00000001

#define LW8597_SET_ZLWLL_DIR_FORMAT                                                                        0x0dbc
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZDIR                                                                     15:0
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZDIR_LESS                                                          0x00000000
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZDIR_GREATER                                                       0x00000001
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT                                                                 31:16
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_MSB                                                        0x00000000
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_FP                                                         0x00000001
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZTRICK                                                     0x00000002
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZF32_1                                                     0x00000003
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_MSB                                                        0x00000000
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_FP                                                         0x00000001
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZTRICK                                                     0x00000002
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZF32_1                                                     0x00000003
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_MSB                                                        0x00000000
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_FP                                                         0x00000001
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZTRICK                                                     0x00000002
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZF32_1                                                     0x00000003
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_MSB                                                        0x00000000
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_FP                                                         0x00000001
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZTRICK                                                     0x00000002
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZF32_1                                                     0x00000003
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_MSB                                                        0x00000000
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_FP                                                         0x00000001
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZTRICK                                                     0x00000002
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZF32_1                                                     0x00000003
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_MSB                                                        0x00000000
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_FP                                                         0x00000001
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZTRICK                                                     0x00000002
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZF32_1                                                     0x00000003
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_MSB                                                        0x00000000
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_FP                                                         0x00000001
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZTRICK                                                     0x00000002
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZF32_1                                                     0x00000003
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_MSB                                                        0x00000000
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_FP                                                         0x00000001
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZTRICK                                                     0x00000002
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZF32_1                                                     0x00000003
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_MSB                                                        0x00000000
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_FP                                                         0x00000001
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZTRICK                                                     0x00000002
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZF32_1                                                     0x00000003
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_MSB                                                        0x00000000
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_FP                                                         0x00000001
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZTRICK                                                     0x00000002
#define LW8597_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZF32_1                                                     0x00000003

#define LW8597_SET_POLY_OFFSET_POINT                                                                       0x0dc0
#define LW8597_SET_POLY_OFFSET_POINT_ENABLE                                                                  31:0
#define LW8597_SET_POLY_OFFSET_POINT_ENABLE_FALSE                                                      0x00000000
#define LW8597_SET_POLY_OFFSET_POINT_ENABLE_TRUE                                                       0x00000001

#define LW8597_SET_POLY_OFFSET_LINE                                                                        0x0dc4
#define LW8597_SET_POLY_OFFSET_LINE_ENABLE                                                                   31:0
#define LW8597_SET_POLY_OFFSET_LINE_ENABLE_FALSE                                                       0x00000000
#define LW8597_SET_POLY_OFFSET_LINE_ENABLE_TRUE                                                        0x00000001

#define LW8597_SET_POLY_OFFSET_FILL                                                                        0x0dc8
#define LW8597_SET_POLY_OFFSET_FILL_ENABLE                                                                   31:0
#define LW8597_SET_POLY_OFFSET_FILL_ENABLE_FALSE                                                       0x00000000
#define LW8597_SET_POLY_OFFSET_FILL_ENABLE_TRUE                                                        0x00000001

#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK(i)                                                (0x0dd0+(i)*4)
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE0_COMP0                                             0:0
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE0_COMP0_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE0_COMP0_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE0_COMP1                                             1:1
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE0_COMP1_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE0_COMP1_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE0_COMP2                                             2:2
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE0_COMP2_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE0_COMP2_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE0_COMP3                                             3:3
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE0_COMP3_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE0_COMP3_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE1_COMP0                                             4:4
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE1_COMP0_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE1_COMP0_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE1_COMP1                                             5:5
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE1_COMP1_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE1_COMP1_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE1_COMP2                                             6:6
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE1_COMP2_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE1_COMP2_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE1_COMP3                                             7:7
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE1_COMP3_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE1_COMP3_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE2_COMP0                                             8:8
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE2_COMP0_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE2_COMP0_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE2_COMP1                                             9:9
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE2_COMP1_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE2_COMP1_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE2_COMP2                                           10:10
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE2_COMP2_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE2_COMP2_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE2_COMP3                                           11:11
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE2_COMP3_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE2_COMP3_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE3_COMP0                                           12:12
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE3_COMP0_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE3_COMP0_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE3_COMP1                                           13:13
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE3_COMP1_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE3_COMP1_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE3_COMP2                                           14:14
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE3_COMP2_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE3_COMP2_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE3_COMP3                                           15:15
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE3_COMP3_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE3_COMP3_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE4_COMP0                                           16:16
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE4_COMP0_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE4_COMP0_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE4_COMP1                                           17:17
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE4_COMP1_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE4_COMP1_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE4_COMP2                                           18:18
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE4_COMP2_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE4_COMP2_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE4_COMP3                                           19:19
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE4_COMP3_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE4_COMP3_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE5_COMP0                                           20:20
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE5_COMP0_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE5_COMP0_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE5_COMP1                                           21:21
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE5_COMP1_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE5_COMP1_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE5_COMP2                                           22:22
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE5_COMP2_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE5_COMP2_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE5_COMP3                                           23:23
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE5_COMP3_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE5_COMP3_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE6_COMP0                                           24:24
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE6_COMP0_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE6_COMP0_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE6_COMP1                                           25:25
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE6_COMP1_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE6_COMP1_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE6_COMP2                                           26:26
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE6_COMP2_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE6_COMP2_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE6_COMP3                                           27:27
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE6_COMP3_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE6_COMP3_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE7_COMP0                                           28:28
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE7_COMP0_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE7_COMP0_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE7_COMP1                                           29:29
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE7_COMP1_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE7_COMP1_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE7_COMP2                                           30:30
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE7_COMP2_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE7_COMP2_TRUE                                 0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE7_COMP3                                           31:31
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE7_COMP3_FALSE                                0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_ATTRIBUTE7_COMP3_TRUE                                 0x00000001

#define LW8597_SET_ZLWLL_CRITERION                                                                         0x0dd8
#define LW8597_SET_ZLWLL_CRITERION_SFUNC                                                                      7:0
#define LW8597_SET_ZLWLL_CRITERION_SFUNC_NEVER                                                         0x00000000
#define LW8597_SET_ZLWLL_CRITERION_SFUNC_LESS                                                          0x00000001
#define LW8597_SET_ZLWLL_CRITERION_SFUNC_EQUAL                                                         0x00000002
#define LW8597_SET_ZLWLL_CRITERION_SFUNC_LEQUAL                                                        0x00000003
#define LW8597_SET_ZLWLL_CRITERION_SFUNC_GREATER                                                       0x00000004
#define LW8597_SET_ZLWLL_CRITERION_SFUNC_NOTEQUAL                                                      0x00000005
#define LW8597_SET_ZLWLL_CRITERION_SFUNC_GEQUAL                                                        0x00000006
#define LW8597_SET_ZLWLL_CRITERION_SFUNC_ALWAYS                                                        0x00000007
#define LW8597_SET_ZLWLL_CRITERION_NO_ILWALIDATE                                                              8:8
#define LW8597_SET_ZLWLL_CRITERION_NO_ILWALIDATE_FALSE                                                 0x00000000
#define LW8597_SET_ZLWLL_CRITERION_NO_ILWALIDATE_TRUE                                                  0x00000001
#define LW8597_SET_ZLWLL_CRITERION_FORCE_MATCH                                                                9:9
#define LW8597_SET_ZLWLL_CRITERION_FORCE_MATCH_FALSE                                                   0x00000000
#define LW8597_SET_ZLWLL_CRITERION_FORCE_MATCH_TRUE                                                    0x00000001
#define LW8597_SET_ZLWLL_CRITERION_SREF                                                                     23:16
#define LW8597_SET_ZLWLL_CRITERION_SMASK                                                                    31:24

#define LW8597_SET_DA_ATTRIBUTE_CACHE_LINE                                                                 0x0ddc
#define LW8597_SET_DA_ATTRIBUTE_CACHE_LINE_V                                                                  1:0
#define LW8597_SET_DA_ATTRIBUTE_CACHE_LINE_V_SIZE128                                                   0x00000000
#define LW8597_SET_DA_ATTRIBUTE_CACHE_LINE_V_SIZE64                                                    0x00000001
#define LW8597_SET_DA_ATTRIBUTE_CACHE_LINE_V_SIZE32                                                    0x00000002

#define LW8597_SET_PS_ZOMBIE                                                                               0x0de0
#define LW8597_SET_PS_ZOMBIE_OPTIMIZATION                                                                     0:0
#define LW8597_SET_PS_ZOMBIE_OPTIMIZATION_ZOMBIE_QUADS_REMOVED                                         0x00000000
#define LW8597_SET_PS_ZOMBIE_OPTIMIZATION_ALL_THREADS_COMPLETE                                         0x00000001

#define LW8597_SET_SM_TIMEOUT_INTERVAL                                                                     0x0de4
#define LW8597_SET_SM_TIMEOUT_INTERVAL_COUNTER_BIT                                                            5:0

#define LW8597_SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY                                                       0x0de8
#define LW8597_SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY_ENABLE                                                   0:0
#define LW8597_SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY_ENABLE_FALSE                                      0x00000000
#define LW8597_SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY_ENABLE_TRUE                                       0x00000001

#define LW8597_SET_DRAW_INLINE_VERTEX_VAB_UPDATE                                                           0x0dec
#define LW8597_SET_DRAW_INLINE_VERTEX_VAB_UPDATE_ENABLE                                                       0:0
#define LW8597_SET_DRAW_INLINE_VERTEX_VAB_UPDATE_ENABLE_FALSE                                          0x00000000
#define LW8597_SET_DRAW_INLINE_VERTEX_VAB_UPDATE_ENABLE_TRUE                                           0x00000001

#define LW8597_SET_PRIMITIVES_PER_TPC                                                                      0x0df0
#define LW8597_SET_PRIMITIVES_PER_TPC_LIMIT_ENABLE                                                            0:0
#define LW8597_SET_PRIMITIVES_PER_TPC_LIMIT_ENABLE_FALSE                                               0x00000000
#define LW8597_SET_PRIMITIVES_PER_TPC_LIMIT_ENABLE_TRUE                                                0x00000001
#define LW8597_SET_PRIMITIVES_PER_TPC_V                                                                      11:4

#define LW8597_SET_TICK_CONTROL_EARLY_Z                                                                    0x0df4
#define LW8597_SET_TICK_CONTROL_EARLY_Z_Z_ENABLE                                                              0:0
#define LW8597_SET_TICK_CONTROL_EARLY_Z_Z_ENABLE_FALSE                                                 0x00000000
#define LW8597_SET_TICK_CONTROL_EARLY_Z_Z_ENABLE_TRUE                                                  0x00000001
#define LW8597_SET_TICK_CONTROL_EARLY_Z_COLOR_ENABLE                                                          1:1
#define LW8597_SET_TICK_CONTROL_EARLY_Z_COLOR_ENABLE_FALSE                                             0x00000000
#define LW8597_SET_TICK_CONTROL_EARLY_Z_COLOR_ENABLE_TRUE                                              0x00000001
#define LW8597_SET_TICK_CONTROL_EARLY_Z_MAX_TICK_COUNT                                                        5:2
#define LW8597_SET_TICK_CONTROL_EARLY_Z_ACLWM_TIMEOUT                                                        13:6
#define LW8597_SET_TICK_CONTROL_EARLY_Z_ZBAR_TICK_WINDOW                                                    19:14
#define LW8597_SET_TICK_CONTROL_EARLY_Z_CBAR_TICK_WINDOW                                                    25:20
#define LW8597_SET_TICK_CONTROL_EARLY_Z_PUDDLE_AREA                                                         31:26

#define LW8597_SET_WINDOW_OFFSET_X                                                                         0x0df8
#define LW8597_SET_WINDOW_OFFSET_X_V                                                                         15:0

#define LW8597_SET_WINDOW_OFFSET_Y                                                                         0x0dfc
#define LW8597_SET_WINDOW_OFFSET_Y_V                                                                         15:0

#define LW8597_SET_SCISSOR_ENABLE(j)                                                              (0x0e00+(j)*16)
#define LW8597_SET_SCISSOR_ENABLE_V                                                                           0:0
#define LW8597_SET_SCISSOR_ENABLE_V_FALSE                                                              0x00000000
#define LW8597_SET_SCISSOR_ENABLE_V_TRUE                                                               0x00000001

#define LW8597_SET_SCISSOR_HORIZONTAL(j)                                                          (0x0e04+(j)*16)
#define LW8597_SET_SCISSOR_HORIZONTAL_XMIN                                                                   15:0
#define LW8597_SET_SCISSOR_HORIZONTAL_XMAX                                                                  31:16

#define LW8597_SET_SCISSOR_VERTICAL(j)                                                            (0x0e08+(j)*16)
#define LW8597_SET_SCISSOR_VERTICAL_YMIN                                                                     15:0
#define LW8597_SET_SCISSOR_VERTICAL_YMAX                                                                    31:16

#define LW8597_LOAD_CONSTANT_SELECTOR                                                                      0x0f00
#define LW8597_LOAD_CONSTANT_SELECTOR_TABLE_INDEX                                                             7:0
#define LW8597_LOAD_CONSTANT_SELECTOR_CONSTANT_INDEX                                                         23:8

#define LW8597_LOAD_CONSTANT(i)                                                                    (0x0f04+(i)*4)
#define LW8597_LOAD_CONSTANT_V                                                                               31:0

#define LW8597_SET_SHADER_THREAD_MEMORY_THROTTLE                                                           0x0f44
#define LW8597_SET_SHADER_THREAD_MEMORY_THROTTLE_MAX_TIDS_PER_SM                                              2:0
#define LW8597_SET_SHADER_THREAD_MEMORY_THROTTLE_MAX_TIDS_PER_SM__1                                    0x00000000
#define LW8597_SET_SHADER_THREAD_MEMORY_THROTTLE_MAX_TIDS_PER_SM__2                                    0x00000001
#define LW8597_SET_SHADER_THREAD_MEMORY_THROTTLE_MAX_TIDS_PER_SM__4                                    0x00000002
#define LW8597_SET_SHADER_THREAD_MEMORY_THROTTLE_MAX_TIDS_PER_SM__8                                    0x00000003
#define LW8597_SET_SHADER_THREAD_MEMORY_THROTTLE_MAX_TIDS_PER_SM__16                                   0x00000004
#define LW8597_SET_SHADER_THREAD_MEMORY_THROTTLE_MAX_TIDS_PER_SM__24                                   0x00000005
#define LW8597_SET_SHADER_THREAD_MEMORY_THROTTLE_MAX_TIDS_PER_SM_HW_MAX                                0x00000007

#define LW8597_SET_SHADER_THREAD_MEMORY_THROTTLE_CONTROL                                                   0x0f48
#define LW8597_SET_SHADER_THREAD_MEMORY_THROTTLE_CONTROL_V                                                    2:0
#define LW8597_SET_SHADER_THREAD_MEMORY_THROTTLE_CONTROL_V_USE_THROTTLE_MAX                            0x00000000
#define LW8597_SET_SHADER_THREAD_MEMORY_THROTTLE_CONTROL_V_USE_HW_MAX                                  0x00000001

#define LW8597_SET_SHADER_THREAD_STACK_THROTTLE                                                            0x0f4c
#define LW8597_SET_SHADER_THREAD_STACK_THROTTLE_MAX_TIDS_PER_SM                                               2:0
#define LW8597_SET_SHADER_THREAD_STACK_THROTTLE_MAX_TIDS_PER_SM__1                                     0x00000000
#define LW8597_SET_SHADER_THREAD_STACK_THROTTLE_MAX_TIDS_PER_SM__2                                     0x00000001
#define LW8597_SET_SHADER_THREAD_STACK_THROTTLE_MAX_TIDS_PER_SM__4                                     0x00000002
#define LW8597_SET_SHADER_THREAD_STACK_THROTTLE_MAX_TIDS_PER_SM__8                                     0x00000003
#define LW8597_SET_SHADER_THREAD_STACK_THROTTLE_MAX_TIDS_PER_SM__16                                    0x00000004
#define LW8597_SET_SHADER_THREAD_STACK_THROTTLE_MAX_TIDS_PER_SM__24                                    0x00000005
#define LW8597_SET_SHADER_THREAD_STACK_THROTTLE_MAX_TIDS_PER_SM_HW_MAX                                 0x00000007

#define LW8597_SET_SHADER_THREAD_STACK_THROTTLE_CONTROL                                                    0x0f50
#define LW8597_SET_SHADER_THREAD_STACK_THROTTLE_CONTROL_V                                                     2:0
#define LW8597_SET_SHADER_THREAD_STACK_THROTTLE_CONTROL_V_USE_THROTTLE_MAX                             0x00000000
#define LW8597_SET_SHADER_THREAD_STACK_THROTTLE_CONTROL_V_USE_HW_MAX                                   0x00000001

#define LW8597_SET_BACK_STENCIL_FUNC_REF                                                                   0x0f54
#define LW8597_SET_BACK_STENCIL_FUNC_REF_V                                                                    7:0

#define LW8597_SET_BACK_STENCIL_MASK                                                                       0x0f58
#define LW8597_SET_BACK_STENCIL_MASK_V                                                                        7:0

#define LW8597_SET_BACK_STENCIL_FUNC_MASK                                                                  0x0f5c
#define LW8597_SET_BACK_STENCIL_FUNC_MASK_V                                                                   7:0

#define LW8597_SET_SLWLL_OCCLUDERS_X0_Y0                                                                   0x0f60
#define LW8597_SET_SLWLL_OCCLUDERS_X0_Y0_MASK                                                                15:0
#define LW8597_SET_SLWLL_OCCLUDERS_X0_Y0_Y                                                                  23:16
#define LW8597_SET_SLWLL_OCCLUDERS_X0_Y0_X                                                                  31:24

#define LW8597_SET_SLWLL_OCCLUDERS_X4K_Y0                                                                  0x0f64
#define LW8597_SET_SLWLL_OCCLUDERS_X4K_Y0_MASK                                                               15:0
#define LW8597_SET_SLWLL_OCCLUDERS_X4K_Y0_Y                                                                 23:16
#define LW8597_SET_SLWLL_OCCLUDERS_X4K_Y0_X                                                                 31:24

#define LW8597_SET_SLWLL_OCCLUDERS_X0_Y4K                                                                  0x0f68
#define LW8597_SET_SLWLL_OCCLUDERS_X0_Y4K_MASK                                                               15:0
#define LW8597_SET_SLWLL_OCCLUDERS_X0_Y4K_Y                                                                 23:16
#define LW8597_SET_SLWLL_OCCLUDERS_X0_Y4K_X                                                                 31:24

#define LW8597_SET_SLWLL_OCCLUDERS_X4K_Y4K                                                                 0x0f6c
#define LW8597_SET_SLWLL_OCCLUDERS_X4K_Y4K_MASK                                                              15:0
#define LW8597_SET_SLWLL_OCCLUDERS_X4K_Y4K_Y                                                                23:16
#define LW8597_SET_SLWLL_OCCLUDERS_X4K_Y4K_X                                                                31:24

#define LW8597_SET_GS_PROGRAM_A                                                                            0x0f70
#define LW8597_SET_GS_PROGRAM_A_OFFSET_UPPER                                                                  7:0

#define LW8597_SET_GS_PROGRAM_B                                                                            0x0f74
#define LW8597_SET_GS_PROGRAM_B_OFFSET_LOWER                                                                 31:0

#define LW8597_SET_LWMCOVG_CONTROL                                                                         0x0f78
#define LW8597_SET_LWMCOVG_CONTROL_LWMCOVG_Z_ENABLE                                                           0:0
#define LW8597_SET_LWMCOVG_CONTROL_LWMCOVG_Z_ENABLE_FALSE                                              0x00000000
#define LW8597_SET_LWMCOVG_CONTROL_LWMCOVG_Z_ENABLE_TRUE                                               0x00000001
#define LW8597_SET_LWMCOVG_CONTROL_LWMCOVG_C_ENABLE                                                           1:1
#define LW8597_SET_LWMCOVG_CONTROL_LWMCOVG_C_ENABLE_FALSE                                              0x00000000
#define LW8597_SET_LWMCOVG_CONTROL_LWMCOVG_C_ENABLE_TRUE                                               0x00000001
#define LW8597_SET_LWMCOVG_CONTROL_CONFLICT_NEXT_ENABLE                                                       2:2
#define LW8597_SET_LWMCOVG_CONTROL_CONFLICT_NEXT_ENABLE_FALSE                                          0x00000000
#define LW8597_SET_LWMCOVG_CONTROL_CONFLICT_NEXT_ENABLE_TRUE                                           0x00000001
#define LW8597_SET_LWMCOVG_CONTROL_CONFLICT_NEAR_LIMIT                                                        8:3
#define LW8597_SET_LWMCOVG_CONTROL_CONFLICT_FAR_LIMIT                                                        14:9
#define LW8597_SET_LWMCOVG_CONTROL_HOLDING_CYCLES                                                           20:15

#define LW8597_SET_VS_PROGRAM_A                                                                            0x0f7c
#define LW8597_SET_VS_PROGRAM_A_OFFSET_UPPER                                                                  7:0

#define LW8597_SET_VS_PROGRAM_B                                                                            0x0f80
#define LW8597_SET_VS_PROGRAM_B_OFFSET_LOWER                                                                 31:0

#define LW8597_SET_VERTEX_STREAM_SUBSTITUTE_A                                                              0x0f84
#define LW8597_SET_VERTEX_STREAM_SUBSTITUTE_A_ADDRESS_UPPER                                                   7:0

#define LW8597_SET_VERTEX_STREAM_SUBSTITUTE_B                                                              0x0f88
#define LW8597_SET_VERTEX_STREAM_SUBSTITUTE_B_ADDRESS_LOWER                                                  31:0

#define LW8597_SET_LINE_MODE_POLYGON_CLIP                                                                  0x0f8c
#define LW8597_SET_LINE_MODE_POLYGON_CLIP_GENERATED_EDGE                                                      0:0
#define LW8597_SET_LINE_MODE_POLYGON_CLIP_GENERATED_EDGE_DRAW_LINE                                     0x00000000
#define LW8597_SET_LINE_MODE_POLYGON_CLIP_GENERATED_EDGE_DO_NOT_DRAW_LINE                              0x00000001

#define LW8597_SET_SINGLE_CT_WRITE_CONTROL                                                                 0x0f90
#define LW8597_SET_SINGLE_CT_WRITE_CONTROL_ENABLE                                                             0:0
#define LW8597_SET_SINGLE_CT_WRITE_CONTROL_ENABLE_FALSE                                                0x00000000
#define LW8597_SET_SINGLE_CT_WRITE_CONTROL_ENABLE_TRUE                                                 0x00000001

#define LW8597_SET_LWBEMAP_ADDRESS_MODE_OVERRIDE                                                           0x0f94
#define LW8597_SET_LWBEMAP_ADDRESS_MODE_OVERRIDE_ENABLE                                                      31:0
#define LW8597_SET_LWBEMAP_ADDRESS_MODE_OVERRIDE_ENABLE_FALSE                                          0x00000000
#define LW8597_SET_LWBEMAP_ADDRESS_MODE_OVERRIDE_ENABLE_TRUE                                           0x00000001

#define LW8597_SET_FRSTR_PERFORMANCE_CONTROL                                                               0x0f98
#define LW8597_SET_FRSTR_PERFORMANCE_CONTROL_RAST16X4_ENABLE                                                  0:0
#define LW8597_SET_FRSTR_PERFORMANCE_CONTROL_RAST16X4_ENABLE_FALSE                                     0x00000000
#define LW8597_SET_FRSTR_PERFORMANCE_CONTROL_RAST16X4_ENABLE_TRUE                                      0x00000001

#define LW8597_SET_DEPTH_BOUNDS_MIN                                                                        0x0f9c
#define LW8597_SET_DEPTH_BOUNDS_MIN_V                                                                        31:0

#define LW8597_SET_DEPTH_BOUNDS_MAX                                                                        0x0fa0
#define LW8597_SET_DEPTH_BOUNDS_MAX_V                                                                        31:0

#define LW8597_SET_PS_PROGRAM_A                                                                            0x0fa4
#define LW8597_SET_PS_PROGRAM_A_OFFSET_UPPER                                                                  7:0

#define LW8597_SET_PS_PROGRAM_B                                                                            0x0fa8
#define LW8597_SET_PS_PROGRAM_B_OFFSET_LOWER                                                                 31:0

#define LW8597_SET_INTER_TPC_ARBITRATION_CONTROL                                                           0x0fac
#define LW8597_SET_INTER_TPC_ARBITRATION_CONTROL_ENABLE                                                       0:0
#define LW8597_SET_INTER_TPC_ARBITRATION_CONTROL_ENABLE_FALSE                                          0x00000000
#define LW8597_SET_INTER_TPC_ARBITRATION_CONTROL_ENABLE_TRUE                                           0x00000001
#define LW8597_SET_INTER_TPC_ARBITRATION_CONTROL_WAVEFRONT_WINDOW_SIZE                                       11:4
#define LW8597_SET_INTER_TPC_ARBITRATION_CONTROL_TEXTURE_PHASE_WINDOW_SIZE                                  19:12
#define LW8597_SET_INTER_TPC_ARBITRATION_CONTROL_LOAD_PIXELS_BEFORE_ATTRIBS                                   1:1
#define LW8597_SET_INTER_TPC_ARBITRATION_CONTROL_LOAD_PIXELS_BEFORE_ATTRIBS_FALSE                      0x00000000
#define LW8597_SET_INTER_TPC_ARBITRATION_CONTROL_LOAD_PIXELS_BEFORE_ATTRIBS_TRUE                       0x00000001

#define LW8597_SET_NONMULTISAMPLED_Z                                                                       0x0fb0
#define LW8597_SET_NONMULTISAMPLED_Z_V                                                                        0:0
#define LW8597_SET_NONMULTISAMPLED_Z_V_PER_SAMPLE                                                      0x00000000
#define LW8597_SET_NONMULTISAMPLED_Z_V_AT_PIXEL_CENTER                                                 0x00000001

#define LW8597_SET_SWATH                                                                                   0x0fb4
#define LW8597_SET_SWATH_HEIGHT                                                                               2:0
#define LW8597_SET_SWATH_HEIGHT_ONE_32X32_TILE                                                         0x00000000
#define LW8597_SET_SWATH_HEIGHT_TWO_32X32_TILES                                                        0x00000001
#define LW8597_SET_SWATH_HEIGHT_FOUR_32X32_TILES                                                       0x00000002

#define LW8597_SET_TPC_MASK_WAIT                                                                           0x0fb8
#define LW8597_SET_TPC_MASK_WAIT_COUNT                                                                        4:0

#define LW8597_SET_SAMPLE_MASK_X0_Y0                                                                       0x0fbc
#define LW8597_SET_SAMPLE_MASK_X0_Y0_V                                                                       15:0

#define LW8597_SET_SAMPLE_MASK_X1_Y0                                                                       0x0fc0
#define LW8597_SET_SAMPLE_MASK_X1_Y0_V                                                                       15:0

#define LW8597_SET_SAMPLE_MASK_X0_Y1                                                                       0x0fc4
#define LW8597_SET_SAMPLE_MASK_X0_Y1_V                                                                       15:0

#define LW8597_SET_SAMPLE_MASK_X1_Y1                                                                       0x0fc8
#define LW8597_SET_SAMPLE_MASK_X1_Y1_V                                                                       15:0

#define LW8597_SET_SURFACE_CLIP_ID_MEMORY_A                                                                0x0fcc
#define LW8597_SET_SURFACE_CLIP_ID_MEMORY_A_OFFSET_UPPER                                                      7:0

#define LW8597_SET_SURFACE_CLIP_ID_MEMORY_B                                                                0x0fd0
#define LW8597_SET_SURFACE_CLIP_ID_MEMORY_B_OFFSET_LOWER                                                     31:0

#define LW8597_SET_ATTRIBUTE_VIEWPORT_INDEX_SLOT                                                           0x0fd4
#define LW8597_SET_ATTRIBUTE_VIEWPORT_INDEX_SLOT_V                                                            7:0

#define LW8597_SET_DA_ATTRIBUTE_SCHEDULER_POLICY                                                           0x0fd8
#define LW8597_SET_DA_ATTRIBUTE_SCHEDULER_POLICY_SIMPLE_SCHEDULE                                              0:0
#define LW8597_SET_DA_ATTRIBUTE_SCHEDULER_POLICY_SIMPLE_SCHEDULE_FALSE                                 0x00000000
#define LW8597_SET_DA_ATTRIBUTE_SCHEDULER_POLICY_SIMPLE_SCHEDULE_TRUE                                  0x00000001
#define LW8597_SET_DA_ATTRIBUTE_SCHEDULER_POLICY_STREAM_MATCHING                                              4:4
#define LW8597_SET_DA_ATTRIBUTE_SCHEDULER_POLICY_STREAM_MATCHING_FALSE                                 0x00000000
#define LW8597_SET_DA_ATTRIBUTE_SCHEDULER_POLICY_STREAM_MATCHING_TRUE                                  0x00000001

#define LW8597_SET_BLEND_OPT_CONTROL                                                                       0x0fdc
#define LW8597_SET_BLEND_OPT_CONTROL_ALLOW_FLOAT_PIXEL_KILLS                                                  0:0
#define LW8597_SET_BLEND_OPT_CONTROL_ALLOW_FLOAT_PIXEL_KILLS_FALSE                                     0x00000000
#define LW8597_SET_BLEND_OPT_CONTROL_ALLOW_FLOAT_PIXEL_KILLS_TRUE                                      0x00000001

#define LW8597_SET_ZT_A                                                                                    0x0fe0
#define LW8597_SET_ZT_A_OFFSET_UPPER                                                                          7:0

#define LW8597_SET_ZT_B                                                                                    0x0fe4
#define LW8597_SET_ZT_B_OFFSET_LOWER                                                                         31:0

#define LW8597_SET_ZT_FORMAT                                                                               0x0fe8
#define LW8597_SET_ZT_FORMAT_V                                                                                4:0
#define LW8597_SET_ZT_FORMAT_V_Z16                                                                     0x00000013
#define LW8597_SET_ZT_FORMAT_V_Z24S8                                                                   0x00000014
#define LW8597_SET_ZT_FORMAT_V_X8Z24                                                                   0x00000015
#define LW8597_SET_ZT_FORMAT_V_S8Z24                                                                   0x00000016
#define LW8597_SET_ZT_FORMAT_V_V8Z24                                                                   0x00000018
#define LW8597_SET_ZT_FORMAT_V_ZF32                                                                    0x0000000A
#define LW8597_SET_ZT_FORMAT_V_ZF32_X24S8                                                              0x00000019
#define LW8597_SET_ZT_FORMAT_V_X8Z24_X16V8S8                                                           0x0000001D
#define LW8597_SET_ZT_FORMAT_V_ZF32_X16V8X8                                                            0x0000001E
#define LW8597_SET_ZT_FORMAT_V_ZF32_X16V8S8                                                            0x0000001F

#define LW8597_SET_ZT_BLOCK_SIZE                                                                           0x0fec
#define LW8597_SET_ZT_BLOCK_SIZE_WIDTH                                                                        3:0
#define LW8597_SET_ZT_BLOCK_SIZE_WIDTH_ONE_GOB                                                         0x00000000
#define LW8597_SET_ZT_BLOCK_SIZE_HEIGHT                                                                       7:4
#define LW8597_SET_ZT_BLOCK_SIZE_HEIGHT_ONE_GOB                                                        0x00000000
#define LW8597_SET_ZT_BLOCK_SIZE_HEIGHT_TWO_GOBS                                                       0x00000001
#define LW8597_SET_ZT_BLOCK_SIZE_HEIGHT_FOUR_GOBS                                                      0x00000002
#define LW8597_SET_ZT_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                                                     0x00000003
#define LW8597_SET_ZT_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                                                   0x00000004
#define LW8597_SET_ZT_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                                                 0x00000005
#define LW8597_SET_ZT_BLOCK_SIZE_DEPTH                                                                       11:8
#define LW8597_SET_ZT_BLOCK_SIZE_DEPTH_ONE_GOB                                                         0x00000000

#define LW8597_SET_ZT_ARRAY_PITCH                                                                          0x0ff0
#define LW8597_SET_ZT_ARRAY_PITCH_V                                                                          31:0

#define LW8597_SET_SURFACE_CLIP_HORIZONTAL                                                                 0x0ff4
#define LW8597_SET_SURFACE_CLIP_HORIZONTAL_X                                                                 15:0
#define LW8597_SET_SURFACE_CLIP_HORIZONTAL_WIDTH                                                            31:16

#define LW8597_SET_SURFACE_CLIP_VERTICAL                                                                   0x0ff8
#define LW8597_SET_SURFACE_CLIP_VERTICAL_Y                                                                   15:0
#define LW8597_SET_SURFACE_CLIP_VERTICAL_HEIGHT                                                             31:16

#define LW8597_SET_PLANAR_QUAD_CLIP                                                                        0x0ffc
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_LEFT_PLANE                                                         0:0
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_LEFT_PLANE_FALSE                                            0x00000000
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_LEFT_PLANE_TRUE                                             0x00000001
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_RIGHT_PLANE                                                        1:1
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_RIGHT_PLANE_FALSE                                           0x00000000
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_RIGHT_PLANE_TRUE                                            0x00000001
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_BOTTOM_PLANE                                                       2:2
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_BOTTOM_PLANE_FALSE                                          0x00000000
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_BOTTOM_PLANE_TRUE                                           0x00000001
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_TOP_PLANE                                                          3:3
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_TOP_PLANE_FALSE                                             0x00000000
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_TOP_PLANE_TRUE                                              0x00000001
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_NEAR_PLANE                                                         4:4
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_NEAR_PLANE_FALSE                                            0x00000000
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_NEAR_PLANE_TRUE                                             0x00000001
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_FAR_PLANE                                                          5:5
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_FAR_PLANE_FALSE                                             0x00000000
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_FAR_PLANE_TRUE                                              0x00000001
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_W_EQUALS_ZERO_PLANE                                                6:6
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_W_EQUALS_ZERO_PLANE_FALSE                                   0x00000000
#define LW8597_SET_PLANAR_QUAD_CLIP_ENABLE_W_EQUALS_ZERO_PLANE_TRUE                                    0x00000001
#define LW8597_SET_PLANAR_QUAD_CLIP_DISABLE_ALL_QUADS                                                         7:7
#define LW8597_SET_PLANAR_QUAD_CLIP_DISABLE_ALL_QUADS_FALSE                                            0x00000000
#define LW8597_SET_PLANAR_QUAD_CLIP_DISABLE_ALL_QUADS_TRUE                                             0x00000001

#define LW8597_SET_VERTEX_STREAM_INSTANCE(i)                                                       (0x1000+(i)*4)
#define LW8597_SET_VERTEX_STREAM_INSTANCE_IS_INSTANCED                                                        0:0
#define LW8597_SET_VERTEX_STREAM_INSTANCE_IS_INSTANCED_FALSE                                           0x00000000
#define LW8597_SET_VERTEX_STREAM_INSTANCE_IS_INSTANCED_TRUE                                            0x00000001

#define LW8597_SET_VS_IBUF_ALLOCATION                                                                      0x1040
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION                                                             29:24
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_TWO_SM_IN_TPC                                          0x00000000
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_ONE_SM_IN_TPC                                          0x00000001
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_02                                                0x00000002
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_03                                                0x00000003
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_04                                                0x00000004
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_05                                                0x00000005
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_06                                                0x00000006
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_07                                                0x00000007
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_08                                                0x00000008
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_09                                                0x00000009
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_10                                                0x0000000A
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_11                                                0x0000000B
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_12                                                0x0000000C
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_13                                                0x0000000D
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_14                                                0x0000000E
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_15                                                0x0000000F
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_16                                                0x00000010
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_17                                                0x00000011
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_18                                                0x00000012
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_19                                                0x00000013
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_20                                                0x00000014
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_21                                                0x00000015
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_22                                                0x00000016
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_23                                                0x00000017
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_24                                                0x00000018
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_25                                                0x00000019
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_26                                                0x0000001A
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_27                                                0x0000001B
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_28                                                0x0000001C
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_29                                                0x0000001D
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_30                                                0x0000001E
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_31                                                0x0000001F
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_32                                                0x00000020
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_33                                                0x00000021
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_34                                                0x00000022
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_35                                                0x00000023
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_36                                                0x00000024
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_37                                                0x00000025
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_38                                                0x00000026
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_39                                                0x00000027
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_40                                                0x00000028
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_41                                                0x00000029
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_42                                                0x0000002A
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_43                                                0x0000002B
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_44                                                0x0000002C
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_45                                                0x0000002D
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_46                                                0x0000002E
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_47                                                0x0000002F
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_48                                                0x00000030
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_49                                                0x00000031
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_50                                                0x00000032
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_51                                                0x00000033
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_52                                                0x00000034
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_53                                                0x00000035
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_54                                                0x00000036
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_55                                                0x00000037
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_56                                                0x00000038
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_57                                                0x00000039
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_58                                                0x0000003A
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_59                                                0x0000003B
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_60                                                0x0000003C
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_61                                                0x0000003D
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_62                                                0x0000003E
#define LW8597_SET_VS_IBUF_ALLOCATION_CONDITION_NOOP_63                                                0x0000003F
#define LW8597_SET_VS_IBUF_ALLOCATION_SIZE                                                                   14:0

#define LW8597_SET_GS_IBUF_ALLOCATION                                                                      0x1044
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION                                                             29:24
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_TWO_SM_IN_TPC                                          0x00000000
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_ONE_SM_IN_TPC                                          0x00000001
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_02                                                0x00000002
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_03                                                0x00000003
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_04                                                0x00000004
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_05                                                0x00000005
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_06                                                0x00000006
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_07                                                0x00000007
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_08                                                0x00000008
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_09                                                0x00000009
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_10                                                0x0000000A
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_11                                                0x0000000B
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_12                                                0x0000000C
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_13                                                0x0000000D
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_14                                                0x0000000E
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_15                                                0x0000000F
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_16                                                0x00000010
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_17                                                0x00000011
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_18                                                0x00000012
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_19                                                0x00000013
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_20                                                0x00000014
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_21                                                0x00000015
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_22                                                0x00000016
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_23                                                0x00000017
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_24                                                0x00000018
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_25                                                0x00000019
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_26                                                0x0000001A
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_27                                                0x0000001B
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_28                                                0x0000001C
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_29                                                0x0000001D
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_30                                                0x0000001E
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_31                                                0x0000001F
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_32                                                0x00000020
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_33                                                0x00000021
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_34                                                0x00000022
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_35                                                0x00000023
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_36                                                0x00000024
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_37                                                0x00000025
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_38                                                0x00000026
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_39                                                0x00000027
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_40                                                0x00000028
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_41                                                0x00000029
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_42                                                0x0000002A
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_43                                                0x0000002B
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_44                                                0x0000002C
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_45                                                0x0000002D
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_46                                                0x0000002E
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_47                                                0x0000002F
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_48                                                0x00000030
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_49                                                0x00000031
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_50                                                0x00000032
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_51                                                0x00000033
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_52                                                0x00000034
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_53                                                0x00000035
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_54                                                0x00000036
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_55                                                0x00000037
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_56                                                0x00000038
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_57                                                0x00000039
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_58                                                0x0000003A
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_59                                                0x0000003B
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_60                                                0x0000003C
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_61                                                0x0000003D
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_62                                                0x0000003E
#define LW8597_SET_GS_IBUF_ALLOCATION_CONDITION_NOOP_63                                                0x0000003F
#define LW8597_SET_GS_IBUF_ALLOCATION_SIZE                                                                   14:0

#define LW8597_SET_SPARE_NOOP02                                                                            0x1048
#define LW8597_SET_SPARE_NOOP02_V                                                                            31:0

#define LW8597_SET_SPARE_NOOP03                                                                            0x104c
#define LW8597_SET_SPARE_NOOP03_V                                                                            31:0

#define LW8597_SET_SPARE_NOOP04                                                                            0x1050
#define LW8597_SET_SPARE_NOOP04_V                                                                            31:0

#define LW8597_SET_SPARE_NOOP05                                                                            0x1054
#define LW8597_SET_SPARE_NOOP05_V                                                                            31:0

#define LW8597_SET_SPARE_NOOP06                                                                            0x1058
#define LW8597_SET_SPARE_NOOP06_V                                                                            31:0

#define LW8597_SET_SPARE_NOOP07                                                                            0x105c
#define LW8597_SET_SPARE_NOOP07_V                                                                            31:0

#define LW8597_SET_SPARE_NOOP08                                                                            0x1060
#define LW8597_SET_SPARE_NOOP08_V                                                                            31:0

#define LW8597_SET_SPARE_NOOP09                                                                            0x1064
#define LW8597_SET_SPARE_NOOP09_V                                                                            31:0

#define LW8597_SET_SPARE_NOOP10                                                                            0x1068
#define LW8597_SET_SPARE_NOOP10_V                                                                            31:0

#define LW8597_SET_SPARE_NOOP11                                                                            0x106c
#define LW8597_SET_SPARE_NOOP11_V                                                                            31:0

#define LW8597_SET_SPARE_NOOP12                                                                            0x1070
#define LW8597_SET_SPARE_NOOP12_V                                                                            31:0

#define LW8597_SET_SPARE_NOOP13                                                                            0x1074
#define LW8597_SET_SPARE_NOOP13_V                                                                            31:0

#define LW8597_SET_INSTRUMENTATION_METHOD_HEADER                                                           0x1078
#define LW8597_SET_INSTRUMENTATION_METHOD_HEADER_V                                                           31:0

#define LW8597_SET_INSTRUMENTATION_METHOD_DATA                                                             0x107c
#define LW8597_SET_INSTRUMENTATION_METHOD_DATA_V                                                             31:0

#define LW8597_SET_VERTEX_STREAM_LIMIT_A(j)                                                        (0x1080+(j)*8)
#define LW8597_SET_VERTEX_STREAM_LIMIT_A_UPPER                                                                7:0

#define LW8597_SET_VERTEX_STREAM_LIMIT_B(j)                                                        (0x1084+(j)*8)
#define LW8597_SET_VERTEX_STREAM_LIMIT_B_LOWER                                                               31:0

#define LW8597_SET_LINE_QUALITY                                                                            0x1100
#define LW8597_SET_LINE_QUALITY_MITER_ENABLE                                                                  0:0
#define LW8597_SET_LINE_QUALITY_MITER_ENABLE_FALSE                                                     0x00000000
#define LW8597_SET_LINE_QUALITY_MITER_ENABLE_TRUE                                                      0x00000001
#define LW8597_SET_LINE_QUALITY_SHORT_LINE_MERGE_ENABLE                                                       1:1
#define LW8597_SET_LINE_QUALITY_SHORT_LINE_MERGE_ENABLE_FALSE                                          0x00000000
#define LW8597_SET_LINE_QUALITY_SHORT_LINE_MERGE_ENABLE_TRUE                                           0x00000001

#define LW8597_SET_ZLWLL_REGION                                                                            0x1104
#define LW8597_SET_ZLWLL_REGION_WIDTH                                                                        13:0
#define LW8597_SET_ZLWLL_REGION_HEIGHT                                                                      29:16

#define LW8597_SET_ZLWLL_STATS_TO_SM                                                                       0x1108
#define LW8597_SET_ZLWLL_STATS_TO_SM_AFFECT_Z                                                                 0:0
#define LW8597_SET_ZLWLL_STATS_TO_SM_AFFECT_Z_FALSE                                                    0x00000000
#define LW8597_SET_ZLWLL_STATS_TO_SM_AFFECT_Z_TRUE                                                     0x00000001
#define LW8597_SET_ZLWLL_STATS_TO_SM_COUNT_ENABLE                                                             4:4
#define LW8597_SET_ZLWLL_STATS_TO_SM_COUNT_ENABLE_FALSE                                                0x00000000
#define LW8597_SET_ZLWLL_STATS_TO_SM_COUNT_ENABLE_TRUE                                                 0x00000001

#define LW8597_NO_OPERATION_DATA_HI                                                                        0x110c
#define LW8597_NO_OPERATION_DATA_HI_V                                                                        31:0

#define LW8597_SET_DEPTH_BIAS_CONTROL                                                                      0x1110
#define LW8597_SET_DEPTH_BIAS_CONTROL_DEPTH_FORMAT_DEPENDENT                                                  0:0
#define LW8597_SET_DEPTH_BIAS_CONTROL_DEPTH_FORMAT_DEPENDENT_FALSE                                     0x00000000
#define LW8597_SET_DEPTH_BIAS_CONTROL_DEPTH_FORMAT_DEPENDENT_TRUE                                      0x00000001

#define LW8597_PM_TRIGGER_END                                                                              0x1114
#define LW8597_PM_TRIGGER_END_V                                                                              31:0

#define LW8597_SET_VERTEX_ID_BASE                                                                          0x1118
#define LW8597_SET_VERTEX_ID_BASE_V                                                                          31:0

#define LW8597_INCREMENT_PRIMITIVE_ID                                                                      0x111c
#define LW8597_INCREMENT_PRIMITIVE_ID_V                                                                      31:0

#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A(i)                                              (0x1120+(i)*4)
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP0                                           0:0
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP0_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP0_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP1                                           1:1
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP1_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP1_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP2                                           2:2
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP2_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP2_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP3                                           3:3
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP3_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP3_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP0                                           4:4
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP0_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP0_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP1                                           5:5
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP1_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP1_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP2                                           6:6
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP2_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP2_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP3                                           7:7
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP3_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP3_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP0                                           8:8
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP0_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP0_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP1                                           9:9
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP1_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP1_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP2                                         10:10
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP2_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP2_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP3                                         11:11
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP3_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP3_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP0                                         12:12
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP0_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP0_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP1                                         13:13
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP1_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP1_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP2                                         14:14
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP2_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP2_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP3                                         15:15
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP3_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP3_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP0                                         16:16
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP0_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP0_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP1                                         17:17
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP1_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP1_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP2                                         18:18
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP2_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP2_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP3                                         19:19
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP3_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP3_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP0                                         20:20
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP0_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP0_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP1                                         21:21
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP1_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP1_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP2                                         22:22
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP2_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP2_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP3                                         23:23
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP3_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP3_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP0                                         24:24
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP0_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP0_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP1                                         25:25
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP1_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP1_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP2                                         26:26
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP2_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP2_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP3                                         27:27
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP3_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP3_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP0                                         28:28
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP0_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP0_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP1                                         29:29
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP1_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP1_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP2                                         30:30
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP2_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP2_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP3                                         31:31
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP3_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP3_TRUE                               0x00000001

#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B(i)                                              (0x1128+(i)*4)
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP0                                           0:0
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP0_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP0_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP1                                           1:1
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP1_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP1_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP2                                           2:2
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP2_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP2_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP3                                           3:3
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP3_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP3_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP0                                           4:4
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP0_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP0_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP1                                           5:5
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP1_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP1_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP2                                           6:6
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP2_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP2_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP3                                           7:7
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP3_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP3_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP0                                           8:8
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP0_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP0_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP1                                           9:9
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP1_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP1_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP2                                         10:10
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP2_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP2_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP3                                         11:11
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP3_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP3_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP0                                         12:12
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP0_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP0_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP1                                         13:13
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP1_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP1_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP2                                         14:14
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP2_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP2_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP3                                         15:15
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP3_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP3_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP0                                         16:16
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP0_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP0_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP1                                         17:17
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP1_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP1_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP2                                         18:18
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP2_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP2_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP3                                         19:19
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP3_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP3_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP0                                         20:20
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP0_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP0_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP1                                         21:21
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP1_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP1_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP2                                         22:22
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP2_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP2_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP3                                         23:23
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP3_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP3_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP0                                         24:24
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP0_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP0_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP1                                         25:25
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP1_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP1_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP2                                         26:26
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP2_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP2_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP3                                         27:27
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP3_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP3_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP0                                         28:28
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP0_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP0_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP1                                         29:29
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP1_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP1_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP2                                         30:30
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP2_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP2_TRUE                               0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP3                                         31:31
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP3_FALSE                              0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP3_TRUE                               0x00000001

#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A(i)                                                   (0x1130+(i)*4)
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE0_COMP0                                                0:0
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE0_COMP0_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE0_COMP0_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE0_COMP1                                                1:1
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE0_COMP1_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE0_COMP1_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE0_COMP2                                                2:2
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE0_COMP2_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE0_COMP2_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE0_COMP3                                                3:3
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE0_COMP3_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE0_COMP3_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE1_COMP0                                                4:4
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE1_COMP0_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE1_COMP0_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE1_COMP1                                                5:5
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE1_COMP1_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE1_COMP1_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE1_COMP2                                                6:6
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE1_COMP2_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE1_COMP2_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE1_COMP3                                                7:7
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE1_COMP3_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE1_COMP3_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE2_COMP0                                                8:8
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE2_COMP0_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE2_COMP0_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE2_COMP1                                                9:9
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE2_COMP1_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE2_COMP1_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE2_COMP2                                              10:10
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE2_COMP2_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE2_COMP2_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE2_COMP3                                              11:11
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE2_COMP3_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE2_COMP3_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE3_COMP0                                              12:12
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE3_COMP0_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE3_COMP0_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE3_COMP1                                              13:13
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE3_COMP1_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE3_COMP1_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE3_COMP2                                              14:14
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE3_COMP2_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE3_COMP2_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE3_COMP3                                              15:15
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE3_COMP3_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE3_COMP3_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE4_COMP0                                              16:16
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE4_COMP0_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE4_COMP0_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE4_COMP1                                              17:17
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE4_COMP1_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE4_COMP1_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE4_COMP2                                              18:18
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE4_COMP2_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE4_COMP2_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE4_COMP3                                              19:19
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE4_COMP3_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE4_COMP3_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE5_COMP0                                              20:20
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE5_COMP0_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE5_COMP0_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE5_COMP1                                              21:21
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE5_COMP1_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE5_COMP1_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE5_COMP2                                              22:22
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE5_COMP2_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE5_COMP2_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE5_COMP3                                              23:23
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE5_COMP3_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE5_COMP3_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE6_COMP0                                              24:24
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE6_COMP0_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE6_COMP0_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE6_COMP1                                              25:25
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE6_COMP1_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE6_COMP1_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE6_COMP2                                              26:26
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE6_COMP2_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE6_COMP2_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE6_COMP3                                              27:27
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE6_COMP3_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE6_COMP3_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE7_COMP0                                              28:28
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE7_COMP0_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE7_COMP0_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE7_COMP1                                              29:29
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE7_COMP1_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE7_COMP1_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE7_COMP2                                              30:30
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE7_COMP2_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE7_COMP2_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE7_COMP3                                              31:31
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE7_COMP3_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_A_ATTRIBUTE7_COMP3_TRUE                                    0x00000001

#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B(i)                                                   (0x1138+(i)*4)
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE0_COMP0                                                0:0
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE0_COMP0_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE0_COMP0_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE0_COMP1                                                1:1
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE0_COMP1_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE0_COMP1_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE0_COMP2                                                2:2
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE0_COMP2_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE0_COMP2_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE0_COMP3                                                3:3
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE0_COMP3_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE0_COMP3_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE1_COMP0                                                4:4
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE1_COMP0_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE1_COMP0_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE1_COMP1                                                5:5
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE1_COMP1_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE1_COMP1_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE1_COMP2                                                6:6
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE1_COMP2_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE1_COMP2_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE1_COMP3                                                7:7
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE1_COMP3_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE1_COMP3_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE2_COMP0                                                8:8
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE2_COMP0_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE2_COMP0_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE2_COMP1                                                9:9
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE2_COMP1_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE2_COMP1_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE2_COMP2                                              10:10
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE2_COMP2_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE2_COMP2_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE2_COMP3                                              11:11
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE2_COMP3_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE2_COMP3_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE3_COMP0                                              12:12
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE3_COMP0_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE3_COMP0_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE3_COMP1                                              13:13
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE3_COMP1_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE3_COMP1_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE3_COMP2                                              14:14
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE3_COMP2_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE3_COMP2_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE3_COMP3                                              15:15
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE3_COMP3_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE3_COMP3_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE4_COMP0                                              16:16
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE4_COMP0_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE4_COMP0_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE4_COMP1                                              17:17
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE4_COMP1_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE4_COMP1_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE4_COMP2                                              18:18
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE4_COMP2_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE4_COMP2_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE4_COMP3                                              19:19
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE4_COMP3_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE4_COMP3_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE5_COMP0                                              20:20
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE5_COMP0_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE5_COMP0_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE5_COMP1                                              21:21
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE5_COMP1_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE5_COMP1_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE5_COMP2                                              22:22
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE5_COMP2_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE5_COMP2_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE5_COMP3                                              23:23
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE5_COMP3_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE5_COMP3_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE6_COMP0                                              24:24
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE6_COMP0_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE6_COMP0_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE6_COMP1                                              25:25
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE6_COMP1_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE6_COMP1_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE6_COMP2                                              26:26
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE6_COMP2_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE6_COMP2_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE6_COMP3                                              27:27
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE6_COMP3_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE6_COMP3_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE7_COMP0                                              28:28
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE7_COMP0_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE7_COMP0_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE7_COMP1                                              29:29
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE7_COMP1_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE7_COMP1_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE7_COMP2                                              30:30
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE7_COMP2_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE7_COMP2_TRUE                                    0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE7_COMP3                                              31:31
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE7_COMP3_FALSE                                   0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_B_ATTRIBUTE7_COMP3_TRUE                                    0x00000001

#define LW8597_SET_BLEND_PER_FORMAT_ENABLE                                                                 0x1140
#define LW8597_SET_BLEND_PER_FORMAT_ENABLE_SNORM8_UNORM16_SNORM16                                             4:4
#define LW8597_SET_BLEND_PER_FORMAT_ENABLE_SNORM8_UNORM16_SNORM16_FALSE                                0x00000000
#define LW8597_SET_BLEND_PER_FORMAT_ENABLE_SNORM8_UNORM16_SNORM16_TRUE                                 0x00000001

#define LW8597_FLUSH_PENDING_WRITES                                                                        0x1144
#define LW8597_FLUSH_PENDING_WRITES_V                                                                        31:0

#define LW8597_SET_VAB_DATA_CONTROL                                                                        0x114c
#define LW8597_SET_VAB_DATA_CONTROL_VAB_INDEX                                                                 7:0
#define LW8597_SET_VAB_DATA_CONTROL_COMPONENT_COUNT                                                          10:8
#define LW8597_SET_VAB_DATA_CONTROL_COMPONENT_BYTE_WIDTH                                                    14:12
#define LW8597_SET_VAB_DATA_CONTROL_FORMAT                                                                  18:16
#define LW8597_SET_VAB_DATA_CONTROL_FORMAT_UNUSED_ENUM_DO_NOT_USE_BECAUSE_IT_WILL_GO_AWAY              0x00000000
#define LW8597_SET_VAB_DATA_CONTROL_FORMAT_NUM_SNORM                                                   0x00000001
#define LW8597_SET_VAB_DATA_CONTROL_FORMAT_NUM_UNORM                                                   0x00000002
#define LW8597_SET_VAB_DATA_CONTROL_FORMAT_NUM_SINT                                                    0x00000003
#define LW8597_SET_VAB_DATA_CONTROL_FORMAT_NUM_UINT                                                    0x00000004
#define LW8597_SET_VAB_DATA_CONTROL_FORMAT_NUM_USCALED                                                 0x00000005
#define LW8597_SET_VAB_DATA_CONTROL_FORMAT_NUM_SSCALED                                                 0x00000006
#define LW8597_SET_VAB_DATA_CONTROL_FORMAT_NUM_FLOAT                                                   0x00000007

#define LW8597_SET_VAB_DATA(i)                                                                     (0x1150+(i)*4)
#define LW8597_SET_VAB_DATA_V                                                                                31:0

#define LW8597_SET_VERTEX_ATTRIBUTE_A(i)                                                           (0x1160+(i)*4)
#define LW8597_SET_VERTEX_ATTRIBUTE_A_STREAM                                                                  4:0
#define LW8597_SET_VERTEX_ATTRIBUTE_A_SOURCE                                                                  6:6
#define LW8597_SET_VERTEX_ATTRIBUTE_A_SOURCE_ACTIVE                                                    0x00000000
#define LW8597_SET_VERTEX_ATTRIBUTE_A_SOURCE_INACTIVE                                                  0x00000001
#define LW8597_SET_VERTEX_ATTRIBUTE_A_OFFSET                                                                 20:7
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS                                                  26:21
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R32_G32_B32_A32                             0x00000001
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R32_G32_B32                                 0x00000002
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R16_G16_B16_A16                             0x00000003
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R32_G32                                     0x00000004
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R16_G16_B16                                 0x00000005
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_A8B8G8R8                                    0x0000002F
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R8_G8_B8_A8                                 0x0000000A
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_X8B8G8R8                                    0x00000033
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_A2B10G10R10                                 0x00000030
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_B10G11R11                                   0x00000031
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R16_G16                                     0x0000000F
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R32                                         0x00000012
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R8_G8_B8                                    0x00000013
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_G8R8                                        0x00000032
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R8_G8                                       0x00000018
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R16                                         0x0000001B
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R8                                          0x0000001D
#define LW8597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_A8                                          0x00000034
#define LW8597_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE                                                        29:27
#define LW8597_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_UNUSED_ENUM_DO_NOT_USE_BECAUSE_IT_WILL_GO_AWAY             0x00000000
#define LW8597_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_SNORM                                         0x00000001
#define LW8597_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_UNORM                                         0x00000002
#define LW8597_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_SINT                                          0x00000003
#define LW8597_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_UINT                                          0x00000004
#define LW8597_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_USCALED                                       0x00000005
#define LW8597_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_SSCALED                                       0x00000006
#define LW8597_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_FLOAT                                         0x00000007
#define LW8597_SET_VERTEX_ATTRIBUTE_A_SWAP_R_AND_B                                                          31:31
#define LW8597_SET_VERTEX_ATTRIBUTE_A_SWAP_R_AND_B_FALSE                                               0x00000000
#define LW8597_SET_VERTEX_ATTRIBUTE_A_SWAP_R_AND_B_TRUE                                                0x00000001

#define LW8597_SET_VERTEX_ATTRIBUTE_B(i)                                                           (0x11a0+(i)*4)
#define LW8597_SET_VERTEX_ATTRIBUTE_B_STREAM                                                                  4:0
#define LW8597_SET_VERTEX_ATTRIBUTE_B_SOURCE                                                                  6:6
#define LW8597_SET_VERTEX_ATTRIBUTE_B_SOURCE_ACTIVE                                                    0x00000000
#define LW8597_SET_VERTEX_ATTRIBUTE_B_SOURCE_INACTIVE                                                  0x00000001
#define LW8597_SET_VERTEX_ATTRIBUTE_B_OFFSET                                                                 20:7
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS                                                  26:21
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R32_G32_B32_A32                             0x00000001
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R32_G32_B32                                 0x00000002
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R16_G16_B16_A16                             0x00000003
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R32_G32                                     0x00000004
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R16_G16_B16                                 0x00000005
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_A8B8G8R8                                    0x0000002F
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R8_G8_B8_A8                                 0x0000000A
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_X8B8G8R8                                    0x00000033
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_A2B10G10R10                                 0x00000030
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_B10G11R11                                   0x00000031
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R16_G16                                     0x0000000F
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R32                                         0x00000012
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R8_G8_B8                                    0x00000013
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_G8R8                                        0x00000032
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R8_G8                                       0x00000018
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R16                                         0x0000001B
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R8                                          0x0000001D
#define LW8597_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_A8                                          0x00000034
#define LW8597_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE                                                        29:27
#define LW8597_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_UNUSED_ENUM_DO_NOT_USE_BECAUSE_IT_WILL_GO_AWAY             0x00000000
#define LW8597_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_NUM_SNORM                                         0x00000001
#define LW8597_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_NUM_UNORM                                         0x00000002
#define LW8597_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_NUM_SINT                                          0x00000003
#define LW8597_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_NUM_UINT                                          0x00000004
#define LW8597_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_NUM_USCALED                                       0x00000005
#define LW8597_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_NUM_SSCALED                                       0x00000006
#define LW8597_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_NUM_FLOAT                                         0x00000007
#define LW8597_SET_VERTEX_ATTRIBUTE_B_SWAP_R_AND_B                                                          31:31
#define LW8597_SET_VERTEX_ATTRIBUTE_B_SWAP_R_AND_B_FALSE                                               0x00000000
#define LW8597_SET_VERTEX_ATTRIBUTE_B_SWAP_R_AND_B_TRUE                                                0x00000001

#define LW8597_SET_KIND3D_CT_CHECK_ENABLE                                                                  0x1218
#define LW8597_SET_KIND3D_CT_CHECK_ENABLE_V                                                                   0:0
#define LW8597_SET_KIND3D_CT_CHECK_ENABLE_V_FALSE                                                      0x00000000
#define LW8597_SET_KIND3D_CT_CHECK_ENABLE_V_TRUE                                                       0x00000001

#define LW8597_SET_CT_SELECT                                                                               0x121c
#define LW8597_SET_CT_SELECT_TARGET_COUNT                                                                     3:0
#define LW8597_SET_CT_SELECT_TARGET0                                                                          6:4
#define LW8597_SET_CT_SELECT_TARGET1                                                                          9:7
#define LW8597_SET_CT_SELECT_TARGET2                                                                        12:10
#define LW8597_SET_CT_SELECT_TARGET3                                                                        15:13
#define LW8597_SET_CT_SELECT_TARGET4                                                                        18:16
#define LW8597_SET_CT_SELECT_TARGET5                                                                        21:19
#define LW8597_SET_CT_SELECT_TARGET6                                                                        24:22
#define LW8597_SET_CT_SELECT_TARGET7                                                                        27:25

#define LW8597_SET_COMPRESSION_THRESHOLD                                                                   0x1220
#define LW8597_SET_COMPRESSION_THRESHOLD_SAMPLES                                                              3:0
#define LW8597_SET_COMPRESSION_THRESHOLD_SAMPLES__0                                                    0x00000000
#define LW8597_SET_COMPRESSION_THRESHOLD_SAMPLES__1                                                    0x00000001
#define LW8597_SET_COMPRESSION_THRESHOLD_SAMPLES__2                                                    0x00000002
#define LW8597_SET_COMPRESSION_THRESHOLD_SAMPLES__4                                                    0x00000003
#define LW8597_SET_COMPRESSION_THRESHOLD_SAMPLES__8                                                    0x00000004
#define LW8597_SET_COMPRESSION_THRESHOLD_SAMPLES__16                                                   0x00000005
#define LW8597_SET_COMPRESSION_THRESHOLD_SAMPLES__32                                                   0x00000006
#define LW8597_SET_COMPRESSION_THRESHOLD_SAMPLES__64                                                   0x00000007
#define LW8597_SET_COMPRESSION_THRESHOLD_SAMPLES__128                                                  0x00000008
#define LW8597_SET_COMPRESSION_THRESHOLD_SAMPLES__256                                                  0x00000009
#define LW8597_SET_COMPRESSION_THRESHOLD_SAMPLES__512                                                  0x0000000A
#define LW8597_SET_COMPRESSION_THRESHOLD_SAMPLES__1024                                                 0x0000000B
#define LW8597_SET_COMPRESSION_THRESHOLD_SAMPLES__2048                                                 0x0000000C

#define LW8597_SET_CT_SIZE_C                                                                               0x1224
#define LW8597_SET_CT_SIZE_C_THIRD_DIMENSION                                                                 15:0
#define LW8597_SET_CT_SIZE_C_CONTROL                                                                        16:16
#define LW8597_SET_CT_SIZE_C_CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE                                0x00000000
#define LW8597_SET_CT_SIZE_C_CONTROL_THIRD_DIMENSION_DEFINES_DEPTH_SIZE                                0x00000001

#define LW8597_SET_ZT_SIZE_A                                                                               0x1228
#define LW8597_SET_ZT_SIZE_A_WIDTH                                                                           27:0

#define LW8597_SET_ZT_SIZE_B                                                                               0x122c
#define LW8597_SET_ZT_SIZE_B_HEIGHT                                                                          15:0

#define LW8597_SET_ZT_SIZE_C                                                                               0x1230
#define LW8597_SET_ZT_SIZE_C_THIRD_DIMENSION                                                                 15:0
#define LW8597_SET_ZT_SIZE_C_CONTROL                                                                        16:16
#define LW8597_SET_ZT_SIZE_C_CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE                                0x00000000
#define LW8597_SET_ZT_SIZE_C_CONTROL_ARRAY_SIZE_IS_ONE                                                 0x00000001

#define LW8597_SET_SAMPLER_BINDING                                                                         0x1234
#define LW8597_SET_SAMPLER_BINDING_V                                                                          0:0
#define LW8597_SET_SAMPLER_BINDING_V_INDEPENDENTLY                                                     0x00000000
#define LW8597_SET_SAMPLER_BINDING_V_VIA_HEADER_BINDING                                                0x00000001

#define LW8597_PREFETCH_TEXTURE_SAMPLER                                                                    0x1238
#define LW8597_PREFETCH_TEXTURE_SAMPLER_INDEX                                                                21:0

#define LW8597_DRAW_AUTO                                                                                   0x123c
#define LW8597_DRAW_AUTO_BYTE_COUNT                                                                          31:0

#define LW8597_SET_CT_SIZE_A(j)                                                                    (0x1240+(j)*8)
#define LW8597_SET_CT_SIZE_A_WIDTH                                                                           27:0
#define LW8597_SET_CT_SIZE_A_LAYOUT_IN_MEMORY                                                               31:31
#define LW8597_SET_CT_SIZE_A_LAYOUT_IN_MEMORY_BLOCKLINEAR                                              0x00000000
#define LW8597_SET_CT_SIZE_A_LAYOUT_IN_MEMORY_PITCH                                                    0x00000001

#define LW8597_SET_CT_SIZE_B(j)                                                                    (0x1244+(j)*8)
#define LW8597_SET_CT_SIZE_B_HEIGHT                                                                          15:0

#define LW8597_LOAD_CONSTANT_BUFFER_TABLE_A                                                                0x1280
#define LW8597_LOAD_CONSTANT_BUFFER_TABLE_A_OFFSET_UPPER                                                      7:0

#define LW8597_LOAD_CONSTANT_BUFFER_TABLE_B                                                                0x1284
#define LW8597_LOAD_CONSTANT_BUFFER_TABLE_B_OFFSET_LOWER                                                     31:0

#define LW8597_LOAD_CONSTANT_BUFFER_TABLE_C                                                                0x1288
#define LW8597_LOAD_CONSTANT_BUFFER_TABLE_C_SIZE                                                             15:0
#define LW8597_LOAD_CONSTANT_BUFFER_TABLE_C_ENTRY                                                           23:16

#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION                                                              0x128c
#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION_WIDTH_NON_ZONLY                                                 1:0
#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION_WIDTH_NON_ZONLY_ONE_8X8_TILE                             0x00000000
#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION_WIDTH_NON_ZONLY_TWO_8X8_TILES                            0x00000001
#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION_WIDTH_NON_ZONLY_FOUR_8X8_TILES                           0x00000002
#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION_HEIGHT_NON_ZONLY                                                5:4
#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION_HEIGHT_NON_ZONLY_ONE_8X8_TILE                            0x00000000
#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION_HEIGHT_NON_ZONLY_TWO_8X8_TILES                           0x00000001
#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION_HEIGHT_NON_ZONLY_FOUR_8X8_TILES                          0x00000002
#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION_WIDTH_ZONLY                                                     9:8
#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION_WIDTH_ZONLY_ONE_8X8_TILE                                 0x00000000
#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION_WIDTH_ZONLY_TWO_8X8_TILES                                0x00000001
#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION_WIDTH_ZONLY_FOUR_8X8_TILES                               0x00000002
#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION_HEIGHT_ZONLY                                                  13:12
#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION_HEIGHT_ZONLY_ONE_8X8_TILE                                0x00000000
#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION_HEIGHT_ZONLY_TWO_8X8_TILES                               0x00000001
#define LW8597_SET_TPC_SCREEN_SPACE_PARTITION_HEIGHT_ZONLY_FOUR_8X8_TILES                              0x00000002

#define LW8597_SET_API_CALL_LIMIT                                                                          0x1290
#define LW8597_SET_API_CALL_LIMIT_VS                                                                          3:0
#define LW8597_SET_API_CALL_LIMIT_VS__0                                                                0x00000000
#define LW8597_SET_API_CALL_LIMIT_VS__1                                                                0x00000001
#define LW8597_SET_API_CALL_LIMIT_VS__2                                                                0x00000002
#define LW8597_SET_API_CALL_LIMIT_VS__4                                                                0x00000003
#define LW8597_SET_API_CALL_LIMIT_VS__8                                                                0x00000004
#define LW8597_SET_API_CALL_LIMIT_VS__16                                                               0x00000005
#define LW8597_SET_API_CALL_LIMIT_VS__32                                                               0x00000006
#define LW8597_SET_API_CALL_LIMIT_VS__64                                                               0x00000007
#define LW8597_SET_API_CALL_LIMIT_VS__128                                                              0x00000008
#define LW8597_SET_API_CALL_LIMIT_VS_NO_CHECK                                                          0x0000000F
#define LW8597_SET_API_CALL_LIMIT_GS                                                                          7:4
#define LW8597_SET_API_CALL_LIMIT_GS__0                                                                0x00000000
#define LW8597_SET_API_CALL_LIMIT_GS__1                                                                0x00000001
#define LW8597_SET_API_CALL_LIMIT_GS__2                                                                0x00000002
#define LW8597_SET_API_CALL_LIMIT_GS__4                                                                0x00000003
#define LW8597_SET_API_CALL_LIMIT_GS__8                                                                0x00000004
#define LW8597_SET_API_CALL_LIMIT_GS__16                                                               0x00000005
#define LW8597_SET_API_CALL_LIMIT_GS__32                                                               0x00000006
#define LW8597_SET_API_CALL_LIMIT_GS__64                                                               0x00000007
#define LW8597_SET_API_CALL_LIMIT_GS__128                                                              0x00000008
#define LW8597_SET_API_CALL_LIMIT_GS_NO_CHECK                                                          0x0000000F
#define LW8597_SET_API_CALL_LIMIT_PS                                                                         11:8
#define LW8597_SET_API_CALL_LIMIT_PS__0                                                                0x00000000
#define LW8597_SET_API_CALL_LIMIT_PS__1                                                                0x00000001
#define LW8597_SET_API_CALL_LIMIT_PS__2                                                                0x00000002
#define LW8597_SET_API_CALL_LIMIT_PS__4                                                                0x00000003
#define LW8597_SET_API_CALL_LIMIT_PS__8                                                                0x00000004
#define LW8597_SET_API_CALL_LIMIT_PS__16                                                               0x00000005
#define LW8597_SET_API_CALL_LIMIT_PS__32                                                               0x00000006
#define LW8597_SET_API_CALL_LIMIT_PS__64                                                               0x00000007
#define LW8597_SET_API_CALL_LIMIT_PS__128                                                              0x00000008
#define LW8597_SET_API_CALL_LIMIT_PS_NO_CHECK                                                          0x0000000F

#define LW8597_SET_STREAMING_CONTROL                                                                       0x1294
#define LW8597_SET_STREAMING_CONTROL_MODE                                                                     0:0
#define LW8597_SET_STREAMING_CONTROL_MODE_SEB                                                          0x00000000
#define LW8597_SET_STREAMING_CONTROL_MODE_MEB                                                          0x00000001
#define LW8597_SET_STREAMING_CONTROL_SEB_COUNT                                                                6:4
#define LW8597_SET_STREAMING_CONTROL_MEB_STRIDE                                                              19:8
#define LW8597_SET_STREAMING_CONTROL_OVERFLOW_DETECT                                                          1:1
#define LW8597_SET_STREAMING_CONTROL_OVERFLOW_DETECT_PRIMITIVE_COUNT                                   0x00000000
#define LW8597_SET_STREAMING_CONTROL_OVERFLOW_DETECT_BYTE_COUNT                                        0x00000001

#define LW8597_SET_PS_OUTPUT_REGISTER                                                                      0x1298
#define LW8597_SET_PS_OUTPUT_REGISTER_COUNT                                                                   7:0

#define LW8597_SET_VS_WAIT                                                                                 0x129c
#define LW8597_SET_VS_WAIT_CLOCKS                                                                            15:0

#define LW8597_LOAD_THREAD_BALANCE_CONTROL                                                                 0x12a0
#define LW8597_LOAD_THREAD_BALANCE_CONTROL_INITIAL_VALUE                                                      7:0
#define LW8597_LOAD_THREAD_BALANCE_CONTROL_SM                                                                15:8
#define LW8597_LOAD_THREAD_BALANCE_CONTROL_ENABLE                                                           16:16
#define LW8597_LOAD_THREAD_BALANCE_CONTROL_ENABLE_FALSE                                                0x00000000
#define LW8597_LOAD_THREAD_BALANCE_CONTROL_ENABLE_TRUE                                                 0x00000001

#define LW8597_SET_SHADER_L1_CACHE_CONTROL                                                                 0x12a8
#define LW8597_SET_SHADER_L1_CACHE_CONTROL_ICACHE_PREFETCH_ENABLE                                             0:0
#define LW8597_SET_SHADER_L1_CACHE_CONTROL_ICACHE_PREFETCH_ENABLE_FALSE                                0x00000000
#define LW8597_SET_SHADER_L1_CACHE_CONTROL_ICACHE_PREFETCH_ENABLE_TRUE                                 0x00000001
#define LW8597_SET_SHADER_L1_CACHE_CONTROL_ICACHE_PIXEL_ASSOCIATIVITY                                         7:4
#define LW8597_SET_SHADER_L1_CACHE_CONTROL_ICACHE_NONPIXEL_ASSOCIATIVITY                                     11:8
#define LW8597_SET_SHADER_L1_CACHE_CONTROL_DCACHE_PIXEL_ASSOCIATIVITY                                       15:12
#define LW8597_SET_SHADER_L1_CACHE_CONTROL_DCACHE_NONPIXEL_ASSOCIATIVITY                                    19:16

#define LW8597_SET_SHADER_SCHEDULING                                                                       0x12ac
#define LW8597_SET_SHADER_SCHEDULING_MODE                                                                     0:0
#define LW8597_SET_SHADER_SCHEDULING_MODE_OLDEST_THREAD_FIRST                                          0x00000000
#define LW8597_SET_SHADER_SCHEDULING_MODE_ROUND_ROBIN                                                  0x00000001

#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_A                                             0x12b0
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_A_DELTA                                          7:0
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_A_INITIAL_VALUE                                 15:8
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_A_MIN_VALUE                                    23:16
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_A_MAX_VALUE                                    31:24

#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_B                                             0x12b4
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_B_THRESHOLD_LOWER_BOUND                         15:0
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_B_THRESHOLD_UPPER_BOUND                        31:16

#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_C                                             0x12b8
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_C_SM                                             7:0
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_C_ENABLE                                         8:8
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_C_ENABLE_FALSE                            0x00000000
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_C_ENABLE_TRUE                             0x00000001

#define LW8597_SET_MEMORY_SURFACE_ROOT                                                                     0x12bc
#define LW8597_SET_MEMORY_SURFACE_ROOT_V                                                                     31:0

#define LW8597_SET_MEMORY_SURFACE                                                                          0x12c0
#define LW8597_SET_MEMORY_SURFACE_V                                                                          31:0

#define LW8597_SET_MEMORY_SURFACE_ATTR                                                                     0x12c4
#define LW8597_SET_MEMORY_SURFACE_ATTR_DEPTH                                                                  3:0
#define LW8597_SET_MEMORY_SURFACE_ATTR_DEPTH__16                                                       0x00000002
#define LW8597_SET_MEMORY_SURFACE_ATTR_DEPTH__24                                                       0x00000003
#define LW8597_SET_MEMORY_SURFACE_ATTR_DEPTH__32                                                       0x00000004
#define LW8597_SET_MEMORY_SURFACE_ATTR_ZLWLL                                                                  7:4
#define LW8597_SET_MEMORY_SURFACE_ATTR_ZLWLL_NONE                                                      0x00000000
#define LW8597_SET_MEMORY_SURFACE_ATTR_ZLWLL_REQUIRED                                                  0x00000001
#define LW8597_SET_MEMORY_SURFACE_ATTR_ZLWLL_ANY                                                       0x00000002
#define LW8597_SET_MEMORY_SURFACE_ATTR_ZLWLL_SHARED                                                    0x00000003
#define LW8597_SET_MEMORY_SURFACE_ATTR_COMPR                                                                 11:8
#define LW8597_SET_MEMORY_SURFACE_ATTR_COMPR_NONE                                                      0x00000000
#define LW8597_SET_MEMORY_SURFACE_ATTR_COMPR_REQUIRED                                                  0x00000001
#define LW8597_SET_MEMORY_SURFACE_ATTR_COMPR_ANY                                                       0x00000002

#define LW8597_SET_BOUNDING_BOX_LWLL                                                                       0x12c8
#define LW8597_SET_BOUNDING_BOX_LWLL_ENABLE                                                                   0:0
#define LW8597_SET_BOUNDING_BOX_LWLL_ENABLE_FALSE                                                      0x00000000
#define LW8597_SET_BOUNDING_BOX_LWLL_ENABLE_TRUE                                                       0x00000001

#define LW8597_SET_DEPTH_TEST                                                                              0x12cc
#define LW8597_SET_DEPTH_TEST_ENABLE                                                                         31:0
#define LW8597_SET_DEPTH_TEST_ENABLE_FALSE                                                             0x00000000
#define LW8597_SET_DEPTH_TEST_ENABLE_TRUE                                                              0x00000001

#define LW8597_SET_FILL_MODE                                                                               0x12d0
#define LW8597_SET_FILL_MODE_V                                                                               31:0
#define LW8597_SET_FILL_MODE_V_POINT                                                                   0x00000001
#define LW8597_SET_FILL_MODE_V_WIREFRAME                                                               0x00000002
#define LW8597_SET_FILL_MODE_V_SOLID                                                                   0x00000003

#define LW8597_SET_SHADE_MODE                                                                              0x12d4
#define LW8597_SET_SHADE_MODE_V                                                                              31:0
#define LW8597_SET_SHADE_MODE_V_FLAT                                                                   0x00000001
#define LW8597_SET_SHADE_MODE_V_GOURAUD                                                                0x00000002
#define LW8597_SET_SHADE_MODE_V_OGL_FLAT                                                               0x00001D00
#define LW8597_SET_SHADE_MODE_V_OGL_SMOOTH                                                             0x00001D01

#define LW8597_SET_SHADER_THREAD_MEMORY_A                                                                  0x12d8
#define LW8597_SET_SHADER_THREAD_MEMORY_A_OFFSET_UPPER                                                        7:0

#define LW8597_SET_SHADER_THREAD_MEMORY_B                                                                  0x12dc
#define LW8597_SET_SHADER_THREAD_MEMORY_B_OFFSET_LOWER                                                       31:0

#define LW8597_SET_SHADER_THREAD_MEMORY_C                                                                  0x12e0
#define LW8597_SET_SHADER_THREAD_MEMORY_C_SIZE                                                                3:0
#define LW8597_SET_SHADER_THREAD_MEMORY_C_SIZE__0                                                      0x00000000
#define LW8597_SET_SHADER_THREAD_MEMORY_C_SIZE__1                                                      0x00000001
#define LW8597_SET_SHADER_THREAD_MEMORY_C_SIZE__2                                                      0x00000002
#define LW8597_SET_SHADER_THREAD_MEMORY_C_SIZE__4                                                      0x00000003
#define LW8597_SET_SHADER_THREAD_MEMORY_C_SIZE__8                                                      0x00000004
#define LW8597_SET_SHADER_THREAD_MEMORY_C_SIZE__16                                                     0x00000005
#define LW8597_SET_SHADER_THREAD_MEMORY_C_SIZE__32                                                     0x00000006
#define LW8597_SET_SHADER_THREAD_MEMORY_C_SIZE__64                                                     0x00000007
#define LW8597_SET_SHADER_THREAD_MEMORY_C_SIZE__128                                                    0x00000008
#define LW8597_SET_SHADER_THREAD_MEMORY_C_SIZE__256                                                    0x00000009
#define LW8597_SET_SHADER_THREAD_MEMORY_C_SIZE__512                                                    0x0000000A
#define LW8597_SET_SHADER_THREAD_MEMORY_C_SIZE__1024                                                   0x0000000B
#define LW8597_SET_SHADER_THREAD_MEMORY_C_SIZE__2048                                                   0x0000000C
#define LW8597_SET_SHADER_THREAD_MEMORY_C_SIZE__4096                                                   0x0000000D

#define LW8597_SET_BLEND_STATE_PER_TARGET                                                                  0x12e4
#define LW8597_SET_BLEND_STATE_PER_TARGET_ENABLE                                                              0:0
#define LW8597_SET_BLEND_STATE_PER_TARGET_ENABLE_FALSE                                                 0x00000000
#define LW8597_SET_BLEND_STATE_PER_TARGET_ENABLE_TRUE                                                  0x00000001

#define LW8597_SET_DEPTH_WRITE                                                                             0x12e8
#define LW8597_SET_DEPTH_WRITE_ENABLE                                                                        31:0
#define LW8597_SET_DEPTH_WRITE_ENABLE_FALSE                                                            0x00000000
#define LW8597_SET_DEPTH_WRITE_ENABLE_TRUE                                                             0x00000001

#define LW8597_SET_ALPHA_TEST                                                                              0x12ec
#define LW8597_SET_ALPHA_TEST_ENABLE                                                                         31:0
#define LW8597_SET_ALPHA_TEST_ENABLE_FALSE                                                             0x00000000
#define LW8597_SET_ALPHA_TEST_ENABLE_TRUE                                                              0x00000001

#define LW8597_SET_SHADER_PERFORMANCE_COUNTER_VALUE(i)                                             (0x12f0+(i)*4)
#define LW8597_SET_SHADER_PERFORMANCE_COUNTER_VALUE_V                                                        31:0

#define LW8597_SET_INLINE_INDEX4X8_ALIGN                                                                   0x1300
#define LW8597_SET_INLINE_INDEX4X8_ALIGN_COUNT                                                               29:0
#define LW8597_SET_INLINE_INDEX4X8_ALIGN_START                                                              31:30

#define LW8597_DRAW_INLINE_INDEX4X8                                                                        0x1304
#define LW8597_DRAW_INLINE_INDEX4X8_INDEX0                                                                    7:0
#define LW8597_DRAW_INLINE_INDEX4X8_INDEX1                                                                   15:8
#define LW8597_DRAW_INLINE_INDEX4X8_INDEX2                                                                  23:16
#define LW8597_DRAW_INLINE_INDEX4X8_INDEX3                                                                  31:24

#define LW8597_D3D_SET_LWLL_MODE                                                                           0x1308
#define LW8597_D3D_SET_LWLL_MODE_V                                                                           31:0
#define LW8597_D3D_SET_LWLL_MODE_V_NONE                                                                0x00000001
#define LW8597_D3D_SET_LWLL_MODE_V_CW                                                                  0x00000002
#define LW8597_D3D_SET_LWLL_MODE_V_CCW                                                                 0x00000003

#define LW8597_SET_DEPTH_FUNC                                                                              0x130c
#define LW8597_SET_DEPTH_FUNC_V                                                                              31:0
#define LW8597_SET_DEPTH_FUNC_V_OGL_NEVER                                                              0x00000200
#define LW8597_SET_DEPTH_FUNC_V_OGL_LESS                                                               0x00000201
#define LW8597_SET_DEPTH_FUNC_V_OGL_EQUAL                                                              0x00000202
#define LW8597_SET_DEPTH_FUNC_V_OGL_LEQUAL                                                             0x00000203
#define LW8597_SET_DEPTH_FUNC_V_OGL_GREATER                                                            0x00000204
#define LW8597_SET_DEPTH_FUNC_V_OGL_NOTEQUAL                                                           0x00000205
#define LW8597_SET_DEPTH_FUNC_V_OGL_GEQUAL                                                             0x00000206
#define LW8597_SET_DEPTH_FUNC_V_OGL_ALWAYS                                                             0x00000207
#define LW8597_SET_DEPTH_FUNC_V_D3D_NEVER                                                              0x00000001
#define LW8597_SET_DEPTH_FUNC_V_D3D_LESS                                                               0x00000002
#define LW8597_SET_DEPTH_FUNC_V_D3D_EQUAL                                                              0x00000003
#define LW8597_SET_DEPTH_FUNC_V_D3D_LESSEQUAL                                                          0x00000004
#define LW8597_SET_DEPTH_FUNC_V_D3D_GREATER                                                            0x00000005
#define LW8597_SET_DEPTH_FUNC_V_D3D_NOTEQUAL                                                           0x00000006
#define LW8597_SET_DEPTH_FUNC_V_D3D_GREATEREQUAL                                                       0x00000007
#define LW8597_SET_DEPTH_FUNC_V_D3D_ALWAYS                                                             0x00000008

#define LW8597_SET_ALPHA_REF                                                                               0x1310
#define LW8597_SET_ALPHA_REF_V                                                                               31:0

#define LW8597_SET_ALPHA_FUNC                                                                              0x1314
#define LW8597_SET_ALPHA_FUNC_V                                                                              31:0
#define LW8597_SET_ALPHA_FUNC_V_OGL_NEVER                                                              0x00000200
#define LW8597_SET_ALPHA_FUNC_V_OGL_LESS                                                               0x00000201
#define LW8597_SET_ALPHA_FUNC_V_OGL_EQUAL                                                              0x00000202
#define LW8597_SET_ALPHA_FUNC_V_OGL_LEQUAL                                                             0x00000203
#define LW8597_SET_ALPHA_FUNC_V_OGL_GREATER                                                            0x00000204
#define LW8597_SET_ALPHA_FUNC_V_OGL_NOTEQUAL                                                           0x00000205
#define LW8597_SET_ALPHA_FUNC_V_OGL_GEQUAL                                                             0x00000206
#define LW8597_SET_ALPHA_FUNC_V_OGL_ALWAYS                                                             0x00000207
#define LW8597_SET_ALPHA_FUNC_V_D3D_NEVER                                                              0x00000001
#define LW8597_SET_ALPHA_FUNC_V_D3D_LESS                                                               0x00000002
#define LW8597_SET_ALPHA_FUNC_V_D3D_EQUAL                                                              0x00000003
#define LW8597_SET_ALPHA_FUNC_V_D3D_LESSEQUAL                                                          0x00000004
#define LW8597_SET_ALPHA_FUNC_V_D3D_GREATER                                                            0x00000005
#define LW8597_SET_ALPHA_FUNC_V_D3D_NOTEQUAL                                                           0x00000006
#define LW8597_SET_ALPHA_FUNC_V_D3D_GREATEREQUAL                                                       0x00000007
#define LW8597_SET_ALPHA_FUNC_V_D3D_ALWAYS                                                             0x00000008

#define LW8597_SET_DRAW_AUTO_STRIDE                                                                        0x1318
#define LW8597_SET_DRAW_AUTO_STRIDE_V                                                                        11:0

#define LW8597_SET_BLEND_CONST_RED                                                                         0x131c
#define LW8597_SET_BLEND_CONST_RED_V                                                                         31:0

#define LW8597_SET_BLEND_CONST_GREEN                                                                       0x1320
#define LW8597_SET_BLEND_CONST_GREEN_V                                                                       31:0

#define LW8597_SET_BLEND_CONST_BLUE                                                                        0x1324
#define LW8597_SET_BLEND_CONST_BLUE_V                                                                        31:0

#define LW8597_SET_BLEND_CONST_ALPHA                                                                       0x1328
#define LW8597_SET_BLEND_CONST_ALPHA_V                                                                       31:0

#define LW8597_SET_SHADER_PERFORMANCE_COUNTER_TRAP_CONTROL                                                 0x132c
#define LW8597_SET_SHADER_PERFORMANCE_COUNTER_TRAP_CONTROL_MASK                                               3:0

#define LW8597_ILWALIDATE_SAMPLER_CACHE                                                                    0x1330
#define LW8597_ILWALIDATE_SAMPLER_CACHE_LINES                                                                 0:0
#define LW8597_ILWALIDATE_SAMPLER_CACHE_LINES_ALL                                                      0x00000000
#define LW8597_ILWALIDATE_SAMPLER_CACHE_LINES_ONE                                                      0x00000001
#define LW8597_ILWALIDATE_SAMPLER_CACHE_TAG                                                                  25:4

#define LW8597_ILWALIDATE_TEXTURE_HEADER_CACHE                                                             0x1334
#define LW8597_ILWALIDATE_TEXTURE_HEADER_CACHE_LINES                                                          0:0
#define LW8597_ILWALIDATE_TEXTURE_HEADER_CACHE_LINES_ALL                                               0x00000000
#define LW8597_ILWALIDATE_TEXTURE_HEADER_CACHE_LINES_ONE                                               0x00000001
#define LW8597_ILWALIDATE_TEXTURE_HEADER_CACHE_TAG                                                           25:4

#define LW8597_ILWALIDATE_TEXTURE_DATA_CACHE                                                               0x1338
#define LW8597_ILWALIDATE_TEXTURE_DATA_CACHE_LEVELS                                                           5:4
#define LW8597_ILWALIDATE_TEXTURE_DATA_CACHE_LEVELS_L1_ONLY                                            0x00000000
#define LW8597_ILWALIDATE_TEXTURE_DATA_CACHE_LEVELS_L2_ONLY                                            0x00000001
#define LW8597_ILWALIDATE_TEXTURE_DATA_CACHE_LEVELS_L1_AND_L2                                          0x00000002

#define LW8597_SET_BLEND_SEPARATE_FOR_ALPHA                                                                0x133c
#define LW8597_SET_BLEND_SEPARATE_FOR_ALPHA_ENABLE                                                           31:0
#define LW8597_SET_BLEND_SEPARATE_FOR_ALPHA_ENABLE_FALSE                                               0x00000000
#define LW8597_SET_BLEND_SEPARATE_FOR_ALPHA_ENABLE_TRUE                                                0x00000001

#define LW8597_SET_BLEND_COLOR_OP                                                                          0x1340
#define LW8597_SET_BLEND_COLOR_OP_V                                                                          31:0
#define LW8597_SET_BLEND_COLOR_OP_V_OGL_FUNC_SUBTRACT                                                  0x0000800A
#define LW8597_SET_BLEND_COLOR_OP_V_OGL_FUNC_REVERSE_SUBTRACT                                          0x0000800B
#define LW8597_SET_BLEND_COLOR_OP_V_OGL_FUNC_ADD                                                       0x00008006
#define LW8597_SET_BLEND_COLOR_OP_V_OGL_MIN                                                            0x00008007
#define LW8597_SET_BLEND_COLOR_OP_V_OGL_MAX                                                            0x00008008
#define LW8597_SET_BLEND_COLOR_OP_V_D3D_ADD                                                            0x00000001
#define LW8597_SET_BLEND_COLOR_OP_V_D3D_SUBTRACT                                                       0x00000002
#define LW8597_SET_BLEND_COLOR_OP_V_D3D_REVSUBTRACT                                                    0x00000003
#define LW8597_SET_BLEND_COLOR_OP_V_D3D_MIN                                                            0x00000004
#define LW8597_SET_BLEND_COLOR_OP_V_D3D_MAX                                                            0x00000005

#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF                                                                0x1344
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V                                                                31:0
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ZERO                                                 0x00004000
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ONE                                                  0x00004001
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_SRC_COLOR                                            0x00004300
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_SRC_COLOR                                  0x00004301
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_SRC_ALPHA                                            0x00004302
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA                                  0x00004303
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_DST_ALPHA                                            0x00004304
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_DST_ALPHA                                  0x00004305
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_DST_COLOR                                            0x00004306
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_DST_COLOR                                  0x00004307
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_SRC_ALPHA_SATURATE                                   0x00004308
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_CONSTANT_COLOR                                       0x0000C001
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_CONSTANT_COLOR                             0x0000C002
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_CONSTANT_ALPHA                                       0x0000C003
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_CONSTANT_ALPHA                             0x0000C004
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_SRC1COLOR                                            0x0000C900
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ILWSRC1COLOR                                         0x0000C901
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_SRC1ALPHA                                            0x0000C902
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ILWSRC1ALPHA                                         0x0000C903
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ZERO                                                 0x00000001
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ONE                                                  0x00000002
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_SRCCOLOR                                             0x00000003
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ILWSRCCOLOR                                          0x00000004
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_SRCALPHA                                             0x00000005
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ILWSRCALPHA                                          0x00000006
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_DESTALPHA                                            0x00000007
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ILWDESTALPHA                                         0x00000008
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_DESTCOLOR                                            0x00000009
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ILWDESTCOLOR                                         0x0000000A
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_SRCALPHASAT                                          0x0000000B
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_BOTHSRCALPHA                                         0x0000000C
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_BOTHILWSRCALPHA                                      0x0000000D
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_BLENDFACTOR                                          0x0000000E
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ILWBLENDFACTOR                                       0x0000000F
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_SRC1COLOR                                            0x00000010
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ILWSRC1COLOR                                         0x00000011
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_SRC1ALPHA                                            0x00000012
#define LW8597_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ILWSRC1ALPHA                                         0x00000013

#define LW8597_SET_BLEND_COLOR_DEST_COEFF                                                                  0x1348
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V                                                                  31:0
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ZERO                                                   0x00004000
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ONE                                                    0x00004001
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_SRC_COLOR                                              0x00004300
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_SRC_COLOR                                    0x00004301
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_SRC_ALPHA                                              0x00004302
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA                                    0x00004303
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_DST_ALPHA                                              0x00004304
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_DST_ALPHA                                    0x00004305
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_DST_COLOR                                              0x00004306
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_DST_COLOR                                    0x00004307
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_SRC_ALPHA_SATURATE                                     0x00004308
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_CONSTANT_COLOR                                         0x0000C001
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_CONSTANT_COLOR                               0x0000C002
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_CONSTANT_ALPHA                                         0x0000C003
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_CONSTANT_ALPHA                               0x0000C004
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_SRC1COLOR                                              0x0000C900
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ILWSRC1COLOR                                           0x0000C901
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_SRC1ALPHA                                              0x0000C902
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ILWSRC1ALPHA                                           0x0000C903
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ZERO                                                   0x00000001
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ONE                                                    0x00000002
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_SRCCOLOR                                               0x00000003
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ILWSRCCOLOR                                            0x00000004
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_SRCALPHA                                               0x00000005
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ILWSRCALPHA                                            0x00000006
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_DESTALPHA                                              0x00000007
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ILWDESTALPHA                                           0x00000008
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_DESTCOLOR                                              0x00000009
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ILWDESTCOLOR                                           0x0000000A
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_SRCALPHASAT                                            0x0000000B
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_BLENDFACTOR                                            0x0000000E
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ILWBLENDFACTOR                                         0x0000000F
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_SRC1COLOR                                              0x00000010
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ILWSRC1COLOR                                           0x00000011
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_SRC1ALPHA                                              0x00000012
#define LW8597_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ILWSRC1ALPHA                                           0x00000013

#define LW8597_SET_BLEND_ALPHA_OP                                                                          0x134c
#define LW8597_SET_BLEND_ALPHA_OP_V                                                                          31:0
#define LW8597_SET_BLEND_ALPHA_OP_V_OGL_FUNC_SUBTRACT                                                  0x0000800A
#define LW8597_SET_BLEND_ALPHA_OP_V_OGL_FUNC_REVERSE_SUBTRACT                                          0x0000800B
#define LW8597_SET_BLEND_ALPHA_OP_V_OGL_FUNC_ADD                                                       0x00008006
#define LW8597_SET_BLEND_ALPHA_OP_V_OGL_MIN                                                            0x00008007
#define LW8597_SET_BLEND_ALPHA_OP_V_OGL_MAX                                                            0x00008008
#define LW8597_SET_BLEND_ALPHA_OP_V_D3D_ADD                                                            0x00000001
#define LW8597_SET_BLEND_ALPHA_OP_V_D3D_SUBTRACT                                                       0x00000002
#define LW8597_SET_BLEND_ALPHA_OP_V_D3D_REVSUBTRACT                                                    0x00000003
#define LW8597_SET_BLEND_ALPHA_OP_V_D3D_MIN                                                            0x00000004
#define LW8597_SET_BLEND_ALPHA_OP_V_D3D_MAX                                                            0x00000005

#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF                                                                0x1350
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V                                                                31:0
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ZERO                                                 0x00004000
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ONE                                                  0x00004001
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_SRC_COLOR                                            0x00004300
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_SRC_COLOR                                  0x00004301
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_SRC_ALPHA                                            0x00004302
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA                                  0x00004303
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_DST_ALPHA                                            0x00004304
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_DST_ALPHA                                  0x00004305
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_DST_COLOR                                            0x00004306
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_DST_COLOR                                  0x00004307
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_SRC_ALPHA_SATURATE                                   0x00004308
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_CONSTANT_COLOR                                       0x0000C001
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_CONSTANT_COLOR                             0x0000C002
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_CONSTANT_ALPHA                                       0x0000C003
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_CONSTANT_ALPHA                             0x0000C004
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_SRC1COLOR                                            0x0000C900
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ILWSRC1COLOR                                         0x0000C901
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_SRC1ALPHA                                            0x0000C902
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ILWSRC1ALPHA                                         0x0000C903
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ZERO                                                 0x00000001
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ONE                                                  0x00000002
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_SRCCOLOR                                             0x00000003
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ILWSRCCOLOR                                          0x00000004
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_SRCALPHA                                             0x00000005
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ILWSRCALPHA                                          0x00000006
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_DESTALPHA                                            0x00000007
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ILWDESTALPHA                                         0x00000008
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_DESTCOLOR                                            0x00000009
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ILWDESTCOLOR                                         0x0000000A
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_SRCALPHASAT                                          0x0000000B
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_BOTHSRCALPHA                                         0x0000000C
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_BOTHILWSRCALPHA                                      0x0000000D
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_BLENDFACTOR                                          0x0000000E
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ILWBLENDFACTOR                                       0x0000000F
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_SRC1COLOR                                            0x00000010
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ILWSRC1COLOR                                         0x00000011
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_SRC1ALPHA                                            0x00000012
#define LW8597_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ILWSRC1ALPHA                                         0x00000013

#define LW8597_SET_GLOBAL_COLOR_KEY                                                                        0x1354
#define LW8597_SET_GLOBAL_COLOR_KEY_ENABLE                                                                   31:0
#define LW8597_SET_GLOBAL_COLOR_KEY_ENABLE_FALSE                                                       0x00000000
#define LW8597_SET_GLOBAL_COLOR_KEY_ENABLE_TRUE                                                        0x00000001

#define LW8597_SET_BLEND_ALPHA_DEST_COEFF                                                                  0x1358
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V                                                                  31:0
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ZERO                                                   0x00004000
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ONE                                                    0x00004001
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_SRC_COLOR                                              0x00004300
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_SRC_COLOR                                    0x00004301
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_SRC_ALPHA                                              0x00004302
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA                                    0x00004303
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_DST_ALPHA                                              0x00004304
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_DST_ALPHA                                    0x00004305
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_DST_COLOR                                              0x00004306
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_DST_COLOR                                    0x00004307
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_SRC_ALPHA_SATURATE                                     0x00004308
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_CONSTANT_COLOR                                         0x0000C001
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_CONSTANT_COLOR                               0x0000C002
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_CONSTANT_ALPHA                                         0x0000C003
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_CONSTANT_ALPHA                               0x0000C004
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_SRC1COLOR                                              0x0000C900
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ILWSRC1COLOR                                           0x0000C901
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_SRC1ALPHA                                              0x0000C902
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ILWSRC1ALPHA                                           0x0000C903
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ZERO                                                   0x00000001
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ONE                                                    0x00000002
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_SRCCOLOR                                               0x00000003
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ILWSRCCOLOR                                            0x00000004
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_SRCALPHA                                               0x00000005
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ILWSRCALPHA                                            0x00000006
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_DESTALPHA                                              0x00000007
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ILWDESTALPHA                                           0x00000008
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_DESTCOLOR                                              0x00000009
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ILWDESTCOLOR                                           0x0000000A
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_SRCALPHASAT                                            0x0000000B
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_BLENDFACTOR                                            0x0000000E
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ILWBLENDFACTOR                                         0x0000000F
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_SRC1COLOR                                              0x00000010
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ILWSRC1COLOR                                           0x00000011
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_SRC1ALPHA                                              0x00000012
#define LW8597_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ILWSRC1ALPHA                                           0x00000013

#define LW8597_SET_SINGLE_ROP_CONTROL                                                                      0x135c
#define LW8597_SET_SINGLE_ROP_CONTROL_ENABLE                                                                  0:0
#define LW8597_SET_SINGLE_ROP_CONTROL_ENABLE_FALSE                                                     0x00000000
#define LW8597_SET_SINGLE_ROP_CONTROL_ENABLE_TRUE                                                      0x00000001

#define LW8597_SET_BLEND(i)                                                                        (0x1360+(i)*4)
#define LW8597_SET_BLEND_ENABLE                                                                              31:0
#define LW8597_SET_BLEND_ENABLE_FALSE                                                                  0x00000000
#define LW8597_SET_BLEND_ENABLE_TRUE                                                                   0x00000001

#define LW8597_SET_STENCIL_TEST                                                                            0x1380
#define LW8597_SET_STENCIL_TEST_ENABLE                                                                       31:0
#define LW8597_SET_STENCIL_TEST_ENABLE_FALSE                                                           0x00000000
#define LW8597_SET_STENCIL_TEST_ENABLE_TRUE                                                            0x00000001

#define LW8597_SET_STENCIL_OP_FAIL                                                                         0x1384
#define LW8597_SET_STENCIL_OP_FAIL_V                                                                         31:0
#define LW8597_SET_STENCIL_OP_FAIL_V_OGL_KEEP                                                          0x00001E00
#define LW8597_SET_STENCIL_OP_FAIL_V_OGL_ZERO                                                          0x00000000
#define LW8597_SET_STENCIL_OP_FAIL_V_OGL_REPLACE                                                       0x00001E01
#define LW8597_SET_STENCIL_OP_FAIL_V_OGL_INCRSAT                                                       0x00001E02
#define LW8597_SET_STENCIL_OP_FAIL_V_OGL_DECRSAT                                                       0x00001E03
#define LW8597_SET_STENCIL_OP_FAIL_V_OGL_ILWERT                                                        0x0000150A
#define LW8597_SET_STENCIL_OP_FAIL_V_OGL_INCR                                                          0x00008507
#define LW8597_SET_STENCIL_OP_FAIL_V_OGL_DECR                                                          0x00008508
#define LW8597_SET_STENCIL_OP_FAIL_V_D3D_KEEP                                                          0x00000001
#define LW8597_SET_STENCIL_OP_FAIL_V_D3D_ZERO                                                          0x00000002
#define LW8597_SET_STENCIL_OP_FAIL_V_D3D_REPLACE                                                       0x00000003
#define LW8597_SET_STENCIL_OP_FAIL_V_D3D_INCRSAT                                                       0x00000004
#define LW8597_SET_STENCIL_OP_FAIL_V_D3D_DECRSAT                                                       0x00000005
#define LW8597_SET_STENCIL_OP_FAIL_V_D3D_ILWERT                                                        0x00000006
#define LW8597_SET_STENCIL_OP_FAIL_V_D3D_INCR                                                          0x00000007
#define LW8597_SET_STENCIL_OP_FAIL_V_D3D_DECR                                                          0x00000008

#define LW8597_SET_STENCIL_OP_ZFAIL                                                                        0x1388
#define LW8597_SET_STENCIL_OP_ZFAIL_V                                                                        31:0
#define LW8597_SET_STENCIL_OP_ZFAIL_V_OGL_KEEP                                                         0x00001E00
#define LW8597_SET_STENCIL_OP_ZFAIL_V_OGL_ZERO                                                         0x00000000
#define LW8597_SET_STENCIL_OP_ZFAIL_V_OGL_REPLACE                                                      0x00001E01
#define LW8597_SET_STENCIL_OP_ZFAIL_V_OGL_INCRSAT                                                      0x00001E02
#define LW8597_SET_STENCIL_OP_ZFAIL_V_OGL_DECRSAT                                                      0x00001E03
#define LW8597_SET_STENCIL_OP_ZFAIL_V_OGL_ILWERT                                                       0x0000150A
#define LW8597_SET_STENCIL_OP_ZFAIL_V_OGL_INCR                                                         0x00008507
#define LW8597_SET_STENCIL_OP_ZFAIL_V_OGL_DECR                                                         0x00008508
#define LW8597_SET_STENCIL_OP_ZFAIL_V_D3D_KEEP                                                         0x00000001
#define LW8597_SET_STENCIL_OP_ZFAIL_V_D3D_ZERO                                                         0x00000002
#define LW8597_SET_STENCIL_OP_ZFAIL_V_D3D_REPLACE                                                      0x00000003
#define LW8597_SET_STENCIL_OP_ZFAIL_V_D3D_INCRSAT                                                      0x00000004
#define LW8597_SET_STENCIL_OP_ZFAIL_V_D3D_DECRSAT                                                      0x00000005
#define LW8597_SET_STENCIL_OP_ZFAIL_V_D3D_ILWERT                                                       0x00000006
#define LW8597_SET_STENCIL_OP_ZFAIL_V_D3D_INCR                                                         0x00000007
#define LW8597_SET_STENCIL_OP_ZFAIL_V_D3D_DECR                                                         0x00000008

#define LW8597_SET_STENCIL_OP_ZPASS                                                                        0x138c
#define LW8597_SET_STENCIL_OP_ZPASS_V                                                                        31:0
#define LW8597_SET_STENCIL_OP_ZPASS_V_OGL_KEEP                                                         0x00001E00
#define LW8597_SET_STENCIL_OP_ZPASS_V_OGL_ZERO                                                         0x00000000
#define LW8597_SET_STENCIL_OP_ZPASS_V_OGL_REPLACE                                                      0x00001E01
#define LW8597_SET_STENCIL_OP_ZPASS_V_OGL_INCRSAT                                                      0x00001E02
#define LW8597_SET_STENCIL_OP_ZPASS_V_OGL_DECRSAT                                                      0x00001E03
#define LW8597_SET_STENCIL_OP_ZPASS_V_OGL_ILWERT                                                       0x0000150A
#define LW8597_SET_STENCIL_OP_ZPASS_V_OGL_INCR                                                         0x00008507
#define LW8597_SET_STENCIL_OP_ZPASS_V_OGL_DECR                                                         0x00008508
#define LW8597_SET_STENCIL_OP_ZPASS_V_D3D_KEEP                                                         0x00000001
#define LW8597_SET_STENCIL_OP_ZPASS_V_D3D_ZERO                                                         0x00000002
#define LW8597_SET_STENCIL_OP_ZPASS_V_D3D_REPLACE                                                      0x00000003
#define LW8597_SET_STENCIL_OP_ZPASS_V_D3D_INCRSAT                                                      0x00000004
#define LW8597_SET_STENCIL_OP_ZPASS_V_D3D_DECRSAT                                                      0x00000005
#define LW8597_SET_STENCIL_OP_ZPASS_V_D3D_ILWERT                                                       0x00000006
#define LW8597_SET_STENCIL_OP_ZPASS_V_D3D_INCR                                                         0x00000007
#define LW8597_SET_STENCIL_OP_ZPASS_V_D3D_DECR                                                         0x00000008

#define LW8597_SET_STENCIL_FUNC                                                                            0x1390
#define LW8597_SET_STENCIL_FUNC_V                                                                            31:0
#define LW8597_SET_STENCIL_FUNC_V_OGL_NEVER                                                            0x00000200
#define LW8597_SET_STENCIL_FUNC_V_OGL_LESS                                                             0x00000201
#define LW8597_SET_STENCIL_FUNC_V_OGL_EQUAL                                                            0x00000202
#define LW8597_SET_STENCIL_FUNC_V_OGL_LEQUAL                                                           0x00000203
#define LW8597_SET_STENCIL_FUNC_V_OGL_GREATER                                                          0x00000204
#define LW8597_SET_STENCIL_FUNC_V_OGL_NOTEQUAL                                                         0x00000205
#define LW8597_SET_STENCIL_FUNC_V_OGL_GEQUAL                                                           0x00000206
#define LW8597_SET_STENCIL_FUNC_V_OGL_ALWAYS                                                           0x00000207
#define LW8597_SET_STENCIL_FUNC_V_D3D_NEVER                                                            0x00000001
#define LW8597_SET_STENCIL_FUNC_V_D3D_LESS                                                             0x00000002
#define LW8597_SET_STENCIL_FUNC_V_D3D_EQUAL                                                            0x00000003
#define LW8597_SET_STENCIL_FUNC_V_D3D_LESSEQUAL                                                        0x00000004
#define LW8597_SET_STENCIL_FUNC_V_D3D_GREATER                                                          0x00000005
#define LW8597_SET_STENCIL_FUNC_V_D3D_NOTEQUAL                                                         0x00000006
#define LW8597_SET_STENCIL_FUNC_V_D3D_GREATEREQUAL                                                     0x00000007
#define LW8597_SET_STENCIL_FUNC_V_D3D_ALWAYS                                                           0x00000008

#define LW8597_SET_STENCIL_FUNC_REF                                                                        0x1394
#define LW8597_SET_STENCIL_FUNC_REF_V                                                                         7:0

#define LW8597_SET_STENCIL_MASK                                                                            0x1398
#define LW8597_SET_STENCIL_MASK_V                                                                             7:0

#define LW8597_SET_STENCIL_FUNC_MASK                                                                       0x139c
#define LW8597_SET_STENCIL_FUNC_MASK_V                                                                        7:0

#define LW8597_SET_GEOM_THREAD_LOAD                                                                        0x13a0
#define LW8597_SET_GEOM_THREAD_LOAD_POLICY                                                                    1:0
#define LW8597_SET_GEOM_THREAD_LOAD_POLICY_VS_FIRST                                                    0x00000000
#define LW8597_SET_GEOM_THREAD_LOAD_POLICY_DEPTH_FIRST                                                 0x00000001
#define LW8597_SET_GEOM_THREAD_LOAD_POLICY_FULL_SET                                                    0x00000002

#define LW8597_SET_DRAW_AUTO_START                                                                         0x13a4
#define LW8597_SET_DRAW_AUTO_START_BYTE_COUNT                                                                31:0

#define LW8597_SET_PS_SATURATE                                                                             0x13a8
#define LW8597_SET_PS_SATURATE_OUTPUT0                                                                        0:0
#define LW8597_SET_PS_SATURATE_OUTPUT0_FALSE                                                           0x00000000
#define LW8597_SET_PS_SATURATE_OUTPUT0_TRUE                                                            0x00000001
#define LW8597_SET_PS_SATURATE_OUTPUT1                                                                        4:4
#define LW8597_SET_PS_SATURATE_OUTPUT1_FALSE                                                           0x00000000
#define LW8597_SET_PS_SATURATE_OUTPUT1_TRUE                                                            0x00000001
#define LW8597_SET_PS_SATURATE_OUTPUT2                                                                        8:8
#define LW8597_SET_PS_SATURATE_OUTPUT2_FALSE                                                           0x00000000
#define LW8597_SET_PS_SATURATE_OUTPUT2_TRUE                                                            0x00000001
#define LW8597_SET_PS_SATURATE_OUTPUT3                                                                      12:12
#define LW8597_SET_PS_SATURATE_OUTPUT3_FALSE                                                           0x00000000
#define LW8597_SET_PS_SATURATE_OUTPUT3_TRUE                                                            0x00000001
#define LW8597_SET_PS_SATURATE_OUTPUT4                                                                      16:16
#define LW8597_SET_PS_SATURATE_OUTPUT4_FALSE                                                           0x00000000
#define LW8597_SET_PS_SATURATE_OUTPUT4_TRUE                                                            0x00000001
#define LW8597_SET_PS_SATURATE_OUTPUT5                                                                      20:20
#define LW8597_SET_PS_SATURATE_OUTPUT5_FALSE                                                           0x00000000
#define LW8597_SET_PS_SATURATE_OUTPUT5_TRUE                                                            0x00000001
#define LW8597_SET_PS_SATURATE_OUTPUT6                                                                      24:24
#define LW8597_SET_PS_SATURATE_OUTPUT6_FALSE                                                           0x00000000
#define LW8597_SET_PS_SATURATE_OUTPUT6_TRUE                                                            0x00000001
#define LW8597_SET_PS_SATURATE_OUTPUT7                                                                      28:28
#define LW8597_SET_PS_SATURATE_OUTPUT7_FALSE                                                           0x00000000
#define LW8597_SET_PS_SATURATE_OUTPUT7_TRUE                                                            0x00000001

#define LW8597_SET_WINDOW_ORIGIN                                                                           0x13ac
#define LW8597_SET_WINDOW_ORIGIN_MODE                                                                         0:0
#define LW8597_SET_WINDOW_ORIGIN_MODE_UPPER_LEFT                                                       0x00000000
#define LW8597_SET_WINDOW_ORIGIN_MODE_LOWER_LEFT                                                       0x00000001
#define LW8597_SET_WINDOW_ORIGIN_FLIP_Y                                                                       4:4
#define LW8597_SET_WINDOW_ORIGIN_FLIP_Y_FALSE                                                          0x00000000
#define LW8597_SET_WINDOW_ORIGIN_FLIP_Y_TRUE                                                           0x00000001

#define LW8597_SET_LINE_WIDTH_FLOAT                                                                        0x13b0
#define LW8597_SET_LINE_WIDTH_FLOAT_V                                                                        31:0

#define LW8597_SET_VS_TEXTURE                                                                              0x13b4
#define LW8597_SET_VS_TEXTURE_MAX_ACTIVE_SAMPLERS                                                             3:0
#define LW8597_SET_VS_TEXTURE_MAX_ACTIVE_SAMPLERS__1                                                   0x00000000
#define LW8597_SET_VS_TEXTURE_MAX_ACTIVE_SAMPLERS__2                                                   0x00000001
#define LW8597_SET_VS_TEXTURE_MAX_ACTIVE_SAMPLERS__4                                                   0x00000002
#define LW8597_SET_VS_TEXTURE_MAX_ACTIVE_SAMPLERS__8                                                   0x00000003
#define LW8597_SET_VS_TEXTURE_MAX_ACTIVE_SAMPLERS__16                                                  0x00000004
#define LW8597_SET_VS_TEXTURE_MAX_ACTIVE_HEADERS                                                              7:4
#define LW8597_SET_VS_TEXTURE_MAX_ACTIVE_HEADERS__1                                                    0x00000000
#define LW8597_SET_VS_TEXTURE_MAX_ACTIVE_HEADERS__2                                                    0x00000001
#define LW8597_SET_VS_TEXTURE_MAX_ACTIVE_HEADERS__4                                                    0x00000002
#define LW8597_SET_VS_TEXTURE_MAX_ACTIVE_HEADERS__8                                                    0x00000003
#define LW8597_SET_VS_TEXTURE_MAX_ACTIVE_HEADERS__16                                                   0x00000004
#define LW8597_SET_VS_TEXTURE_MAX_ACTIVE_HEADERS__32                                                   0x00000005
#define LW8597_SET_VS_TEXTURE_MAX_ACTIVE_HEADERS__64                                                   0x00000006
#define LW8597_SET_VS_TEXTURE_MAX_ACTIVE_HEADERS__128                                                  0x00000007

#define LW8597_SET_GS_TEXTURE                                                                              0x13b8
#define LW8597_SET_GS_TEXTURE_MAX_ACTIVE_SAMPLERS                                                             3:0
#define LW8597_SET_GS_TEXTURE_MAX_ACTIVE_SAMPLERS__1                                                   0x00000000
#define LW8597_SET_GS_TEXTURE_MAX_ACTIVE_SAMPLERS__2                                                   0x00000001
#define LW8597_SET_GS_TEXTURE_MAX_ACTIVE_SAMPLERS__4                                                   0x00000002
#define LW8597_SET_GS_TEXTURE_MAX_ACTIVE_SAMPLERS__8                                                   0x00000003
#define LW8597_SET_GS_TEXTURE_MAX_ACTIVE_SAMPLERS__16                                                  0x00000004
#define LW8597_SET_GS_TEXTURE_MAX_ACTIVE_HEADERS                                                              7:4
#define LW8597_SET_GS_TEXTURE_MAX_ACTIVE_HEADERS__1                                                    0x00000000
#define LW8597_SET_GS_TEXTURE_MAX_ACTIVE_HEADERS__2                                                    0x00000001
#define LW8597_SET_GS_TEXTURE_MAX_ACTIVE_HEADERS__4                                                    0x00000002
#define LW8597_SET_GS_TEXTURE_MAX_ACTIVE_HEADERS__8                                                    0x00000003
#define LW8597_SET_GS_TEXTURE_MAX_ACTIVE_HEADERS__16                                                   0x00000004
#define LW8597_SET_GS_TEXTURE_MAX_ACTIVE_HEADERS__32                                                   0x00000005
#define LW8597_SET_GS_TEXTURE_MAX_ACTIVE_HEADERS__64                                                   0x00000006
#define LW8597_SET_GS_TEXTURE_MAX_ACTIVE_HEADERS__128                                                  0x00000007

#define LW8597_SET_PS_TEXTURE                                                                              0x13bc
#define LW8597_SET_PS_TEXTURE_MAX_ACTIVE_SAMPLERS                                                             3:0
#define LW8597_SET_PS_TEXTURE_MAX_ACTIVE_SAMPLERS__1                                                   0x00000000
#define LW8597_SET_PS_TEXTURE_MAX_ACTIVE_SAMPLERS__2                                                   0x00000001
#define LW8597_SET_PS_TEXTURE_MAX_ACTIVE_SAMPLERS__4                                                   0x00000002
#define LW8597_SET_PS_TEXTURE_MAX_ACTIVE_SAMPLERS__8                                                   0x00000003
#define LW8597_SET_PS_TEXTURE_MAX_ACTIVE_SAMPLERS__16                                                  0x00000004
#define LW8597_SET_PS_TEXTURE_MAX_ACTIVE_HEADERS                                                              7:4
#define LW8597_SET_PS_TEXTURE_MAX_ACTIVE_HEADERS__1                                                    0x00000000
#define LW8597_SET_PS_TEXTURE_MAX_ACTIVE_HEADERS__2                                                    0x00000001
#define LW8597_SET_PS_TEXTURE_MAX_ACTIVE_HEADERS__4                                                    0x00000002
#define LW8597_SET_PS_TEXTURE_MAX_ACTIVE_HEADERS__8                                                    0x00000003
#define LW8597_SET_PS_TEXTURE_MAX_ACTIVE_HEADERS__16                                                   0x00000004
#define LW8597_SET_PS_TEXTURE_MAX_ACTIVE_HEADERS__32                                                   0x00000005
#define LW8597_SET_PS_TEXTURE_MAX_ACTIVE_HEADERS__64                                                   0x00000006
#define LW8597_SET_PS_TEXTURE_MAX_ACTIVE_HEADERS__128                                                  0x00000007

#define LW8597_SET_POINT_SPRITE_ATTRIBUTE(i)                                                       (0x13c0+(i)*4)
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP00                                                              3:0
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP00_DISABLE                                               0x00000000
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP00_S                                                     0x00000001
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP00_T                                                     0x00000002
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP00_ZERO                                                  0x00000003
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP00_ONE                                                   0x00000004
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP00_INPUT_S                                               0x00000005
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP01                                                              7:4
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP01_DISABLE                                               0x00000000
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP01_S                                                     0x00000001
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP01_T                                                     0x00000002
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP01_ZERO                                                  0x00000003
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP01_ONE                                                   0x00000004
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP01_INPUT_S                                               0x00000005
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP02                                                             11:8
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP02_DISABLE                                               0x00000000
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP02_S                                                     0x00000001
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP02_T                                                     0x00000002
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP02_ZERO                                                  0x00000003
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP02_ONE                                                   0x00000004
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP02_INPUT_S                                               0x00000005
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP03                                                            15:12
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP03_DISABLE                                               0x00000000
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP03_S                                                     0x00000001
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP03_T                                                     0x00000002
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP03_ZERO                                                  0x00000003
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP03_ONE                                                   0x00000004
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP03_INPUT_S                                               0x00000005
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP04                                                            19:16
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP04_DISABLE                                               0x00000000
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP04_S                                                     0x00000001
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP04_T                                                     0x00000002
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP04_ZERO                                                  0x00000003
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP04_ONE                                                   0x00000004
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP04_INPUT_S                                               0x00000005
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP05                                                            23:20
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP05_DISABLE                                               0x00000000
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP05_S                                                     0x00000001
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP05_T                                                     0x00000002
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP05_ZERO                                                  0x00000003
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP05_ONE                                                   0x00000004
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP05_INPUT_S                                               0x00000005
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP06                                                            27:24
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP06_DISABLE                                               0x00000000
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP06_S                                                     0x00000001
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP06_T                                                     0x00000002
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP06_ZERO                                                  0x00000003
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP06_ONE                                                   0x00000004
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP06_INPUT_S                                               0x00000005
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP07                                                            31:28
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP07_DISABLE                                               0x00000000
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP07_S                                                     0x00000001
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP07_T                                                     0x00000002
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP07_ZERO                                                  0x00000003
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP07_ONE                                                   0x00000004
#define LW8597_SET_POINT_SPRITE_ATTRIBUTE_COMP07_INPUT_S                                               0x00000005

#define LW8597_SET_PS_WAIT                                                                                 0x1400
#define LW8597_SET_PS_WAIT_CLOCKS                                                                            31:0

#define LW8597_SET_NON_PS_WAIT                                                                             0x1404
#define LW8597_SET_NON_PS_WAIT_CLOCKS                                                                        31:0

#define LW8597_SET_EARLY_Z_HYSTERESIS                                                                      0x1408
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS                                                                 15:0
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS_INSTANTANEOUS                                             0x00000000
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS__16                                                       0x00000001
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS__32                                                       0x00000002
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS__64                                                       0x00000003
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS__128                                                      0x00000004
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS__256                                                      0x00000005
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS__512                                                      0x00000006
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS__1024                                                     0x00000007
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS__2048                                                     0x00000008
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS__4096                                                     0x00000009
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS__8192                                                     0x0000000A
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS__16384                                                    0x0000000B
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS__32768                                                    0x0000000C
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS__65536                                                    0x0000000D
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS_INFINITE                                                  0x0000000E
#define LW8597_SET_EARLY_Z_HYSTERESIS_CLOCKS_LATEZ_ALWAYS                                              0x0000000F

#define LW8597_SET_VS_PROGRAM_START                                                                        0x140c
#define LW8597_SET_VS_PROGRAM_START_OFFSET                                                                   23:0

#define LW8597_SET_GS_PROGRAM_START                                                                        0x1410
#define LW8597_SET_GS_PROGRAM_START_OFFSET                                                                   23:0

#define LW8597_SET_PS_PROGRAM_START                                                                        0x1414
#define LW8597_SET_PS_PROGRAM_START_OFFSET                                                                   23:0

#define LW8597_SET_LINE_MULTISAMPLE_OVERRIDE                                                               0x1418
#define LW8597_SET_LINE_MULTISAMPLE_OVERRIDE_ENABLE                                                           0:0
#define LW8597_SET_LINE_MULTISAMPLE_OVERRIDE_ENABLE_FALSE                                              0x00000000
#define LW8597_SET_LINE_MULTISAMPLE_OVERRIDE_ENABLE_TRUE                                               0x00000001

#define LW8597_SET_IBUF_ALLOCATION                                                                         0x141c
#define LW8597_SET_IBUF_ALLOCATION_GS                                                                       26:12

#define LW8597_SET_GS_MAX_OUTPUT_VERTEX_COUNT                                                              0x1420
#define LW8597_SET_GS_MAX_OUTPUT_VERTEX_COUNT_V                                                              15:0

#define LW8597_PEER_SEMAPHORE_RELEASE_OFFSET                                                               0x1424
#define LW8597_PEER_SEMAPHORE_RELEASE_OFFSET_V                                                               31:0

#define LW8597_PEER_SEMAPHORE_RELEASE                                                                      0x1428
#define LW8597_PEER_SEMAPHORE_RELEASE_V                                                                      31:0

#define LW8597_ILWALIDATE_DA_DMA_CACHE                                                                     0x142c
#define LW8597_ILWALIDATE_DA_DMA_CACHE_V                                                                      0:0

#define LW8597_SET_REDUCE_DST_COLOR                                                                        0x1430
#define LW8597_SET_REDUCE_DST_COLOR_UNORM_ENABLE                                                              4:4
#define LW8597_SET_REDUCE_DST_COLOR_UNORM_ENABLE_FALSE                                                 0x00000000
#define LW8597_SET_REDUCE_DST_COLOR_UNORM_ENABLE_TRUE                                                  0x00000001
#define LW8597_SET_REDUCE_DST_COLOR_SRGB_ENABLE                                                               8:8
#define LW8597_SET_REDUCE_DST_COLOR_SRGB_ENABLE_FALSE                                                  0x00000000
#define LW8597_SET_REDUCE_DST_COLOR_SRGB_ENABLE_TRUE                                                   0x00000001

#define LW8597_SET_GLOBAL_BASE_VERTEX_INDEX                                                                0x1434
#define LW8597_SET_GLOBAL_BASE_VERTEX_INDEX_V                                                                31:0

#define LW8597_SET_GLOBAL_BASE_INSTANCE_INDEX                                                              0x1438
#define LW8597_SET_GLOBAL_BASE_INSTANCE_INDEX_V                                                              31:0

#define LW8597_SET_CLEAR_CONTROL                                                                           0x143c
#define LW8597_SET_CLEAR_CONTROL_RESPECT_STENCIL_MASK                                                         0:0
#define LW8597_SET_CLEAR_CONTROL_RESPECT_STENCIL_MASK_FALSE                                            0x00000000
#define LW8597_SET_CLEAR_CONTROL_RESPECT_STENCIL_MASK_TRUE                                             0x00000001
#define LW8597_SET_CLEAR_CONTROL_USE_CLEAR_RECT                                                               4:4
#define LW8597_SET_CLEAR_CONTROL_USE_CLEAR_RECT_FALSE                                                  0x00000000
#define LW8597_SET_CLEAR_CONTROL_USE_CLEAR_RECT_TRUE                                                   0x00000001

#define LW8597_ILWALIDATE_SHADER_CACHE                                                                     0x1440
#define LW8597_ILWALIDATE_SHADER_CACHE_V                                                                      1:0
#define LW8597_ILWALIDATE_SHADER_CACHE_V_ALL                                                           0x00000000
#define LW8597_ILWALIDATE_SHADER_CACHE_V_L1                                                            0x00000001
#define LW8597_ILWALIDATE_SHADER_CACHE_V_L1_DATA                                                       0x00000002
#define LW8597_ILWALIDATE_SHADER_CACHE_V_L1_INSTRUCTION                                                0x00000003

#define LW8597_BIND_VS_TEXTURE_SAMPLER                                                                     0x1444
#define LW8597_BIND_VS_TEXTURE_SAMPLER_VALID                                                                  0:0
#define LW8597_BIND_VS_TEXTURE_SAMPLER_VALID_FALSE                                                     0x00000000
#define LW8597_BIND_VS_TEXTURE_SAMPLER_VALID_TRUE                                                      0x00000001
#define LW8597_BIND_VS_TEXTURE_SAMPLER_SAMPLER_SLOT                                                          11:4
#define LW8597_BIND_VS_TEXTURE_SAMPLER_INDEX                                                                24:12

#define LW8597_BIND_VS_TEXTURE_HEADER                                                                      0x1448
#define LW8597_BIND_VS_TEXTURE_HEADER_VALID                                                                   0:0
#define LW8597_BIND_VS_TEXTURE_HEADER_VALID_FALSE                                                      0x00000000
#define LW8597_BIND_VS_TEXTURE_HEADER_VALID_TRUE                                                       0x00000001
#define LW8597_BIND_VS_TEXTURE_HEADER_TEXTURE_SLOT                                                            8:1
#define LW8597_BIND_VS_TEXTURE_HEADER_INDEX                                                                  30:9

#define LW8597_BIND_GS_TEXTURE_SAMPLER                                                                     0x144c
#define LW8597_BIND_GS_TEXTURE_SAMPLER_VALID                                                                  0:0
#define LW8597_BIND_GS_TEXTURE_SAMPLER_VALID_FALSE                                                     0x00000000
#define LW8597_BIND_GS_TEXTURE_SAMPLER_VALID_TRUE                                                      0x00000001
#define LW8597_BIND_GS_TEXTURE_SAMPLER_SAMPLER_SLOT                                                          11:4
#define LW8597_BIND_GS_TEXTURE_SAMPLER_INDEX                                                                24:12

#define LW8597_BIND_GS_TEXTURE_HEADER                                                                      0x1450
#define LW8597_BIND_GS_TEXTURE_HEADER_VALID                                                                   0:0
#define LW8597_BIND_GS_TEXTURE_HEADER_VALID_FALSE                                                      0x00000000
#define LW8597_BIND_GS_TEXTURE_HEADER_VALID_TRUE                                                       0x00000001
#define LW8597_BIND_GS_TEXTURE_HEADER_TEXTURE_SLOT                                                            8:1
#define LW8597_BIND_GS_TEXTURE_HEADER_INDEX                                                                  30:9

#define LW8597_BIND_PS_TEXTURE_SAMPLER                                                                     0x1454
#define LW8597_BIND_PS_TEXTURE_SAMPLER_VALID                                                                  0:0
#define LW8597_BIND_PS_TEXTURE_SAMPLER_VALID_FALSE                                                     0x00000000
#define LW8597_BIND_PS_TEXTURE_SAMPLER_VALID_TRUE                                                      0x00000001
#define LW8597_BIND_PS_TEXTURE_SAMPLER_SAMPLER_SLOT                                                          11:4
#define LW8597_BIND_PS_TEXTURE_SAMPLER_INDEX                                                                24:12

#define LW8597_BIND_PS_TEXTURE_HEADER                                                                      0x1458
#define LW8597_BIND_PS_TEXTURE_HEADER_VALID                                                                   0:0
#define LW8597_BIND_PS_TEXTURE_HEADER_VALID_FALSE                                                      0x00000000
#define LW8597_BIND_PS_TEXTURE_HEADER_VALID_TRUE                                                       0x00000001
#define LW8597_BIND_PS_TEXTURE_HEADER_TEXTURE_SLOT                                                            8:1
#define LW8597_BIND_PS_TEXTURE_HEADER_INDEX                                                                  30:9

#define LW8597_BIND_VS_EXTRA_TEXTURE_SAMPLER                                                               0x1468
#define LW8597_BIND_VS_EXTRA_TEXTURE_SAMPLER_VALID                                                            0:0
#define LW8597_BIND_VS_EXTRA_TEXTURE_SAMPLER_VALID_FALSE                                               0x00000000
#define LW8597_BIND_VS_EXTRA_TEXTURE_SAMPLER_VALID_TRUE                                                0x00000001
#define LW8597_BIND_VS_EXTRA_TEXTURE_SAMPLER_SAMPLER_SLOT                                                    11:4
#define LW8597_BIND_VS_EXTRA_TEXTURE_SAMPLER_INDEX                                                          24:12

#define LW8597_BIND_VS_EXTRA_TEXTURE_HEADER                                                                0x146c
#define LW8597_BIND_VS_EXTRA_TEXTURE_HEADER_VALID                                                             0:0
#define LW8597_BIND_VS_EXTRA_TEXTURE_HEADER_VALID_FALSE                                                0x00000000
#define LW8597_BIND_VS_EXTRA_TEXTURE_HEADER_VALID_TRUE                                                 0x00000001
#define LW8597_BIND_VS_EXTRA_TEXTURE_HEADER_TEXTURE_SLOT                                                      8:1
#define LW8597_BIND_VS_EXTRA_TEXTURE_HEADER_INDEX                                                            30:9

#define LW8597_BIND_GS_EXTRA_TEXTURE_SAMPLER                                                               0x1470
#define LW8597_BIND_GS_EXTRA_TEXTURE_SAMPLER_VALID                                                            0:0
#define LW8597_BIND_GS_EXTRA_TEXTURE_SAMPLER_VALID_FALSE                                               0x00000000
#define LW8597_BIND_GS_EXTRA_TEXTURE_SAMPLER_VALID_TRUE                                                0x00000001
#define LW8597_BIND_GS_EXTRA_TEXTURE_SAMPLER_SAMPLER_SLOT                                                    11:4
#define LW8597_BIND_GS_EXTRA_TEXTURE_SAMPLER_INDEX                                                          24:12

#define LW8597_BIND_GS_EXTRA_TEXTURE_HEADER                                                                0x1474
#define LW8597_BIND_GS_EXTRA_TEXTURE_HEADER_VALID                                                             0:0
#define LW8597_BIND_GS_EXTRA_TEXTURE_HEADER_VALID_FALSE                                                0x00000000
#define LW8597_BIND_GS_EXTRA_TEXTURE_HEADER_VALID_TRUE                                                 0x00000001
#define LW8597_BIND_GS_EXTRA_TEXTURE_HEADER_TEXTURE_SLOT                                                      8:1
#define LW8597_BIND_GS_EXTRA_TEXTURE_HEADER_INDEX                                                            30:9

#define LW8597_BIND_PS_EXTRA_TEXTURE_SAMPLER                                                               0x1478
#define LW8597_BIND_PS_EXTRA_TEXTURE_SAMPLER_VALID                                                            0:0
#define LW8597_BIND_PS_EXTRA_TEXTURE_SAMPLER_VALID_FALSE                                               0x00000000
#define LW8597_BIND_PS_EXTRA_TEXTURE_SAMPLER_VALID_TRUE                                                0x00000001
#define LW8597_BIND_PS_EXTRA_TEXTURE_SAMPLER_SAMPLER_SLOT                                                    11:4
#define LW8597_BIND_PS_EXTRA_TEXTURE_SAMPLER_INDEX                                                          24:12

#define LW8597_BIND_PS_EXTRA_TEXTURE_HEADER                                                                0x147c
#define LW8597_BIND_PS_EXTRA_TEXTURE_HEADER_VALID                                                             0:0
#define LW8597_BIND_PS_EXTRA_TEXTURE_HEADER_VALID_FALSE                                                0x00000000
#define LW8597_BIND_PS_EXTRA_TEXTURE_HEADER_VALID_TRUE                                                 0x00000001
#define LW8597_BIND_PS_EXTRA_TEXTURE_HEADER_TEXTURE_SLOT                                                      8:1
#define LW8597_BIND_PS_EXTRA_TEXTURE_HEADER_INDEX                                                            30:9

#define LW8597_SET_STREAMING_REORDER(i)                                                            (0x1480+(i)*4)
#define LW8597_SET_STREAMING_REORDER_WRITE_ADDRESS_COMP00                                                     6:0
#define LW8597_SET_STREAMING_REORDER_ENABLE_COMP00                                                            7:7
#define LW8597_SET_STREAMING_REORDER_ENABLE_COMP00_FALSE                                               0x00000000
#define LW8597_SET_STREAMING_REORDER_ENABLE_COMP00_TRUE                                                0x00000001
#define LW8597_SET_STREAMING_REORDER_WRITE_ADDRESS_COMP01                                                    14:8
#define LW8597_SET_STREAMING_REORDER_ENABLE_COMP01                                                          15:15
#define LW8597_SET_STREAMING_REORDER_ENABLE_COMP01_FALSE                                               0x00000000
#define LW8597_SET_STREAMING_REORDER_ENABLE_COMP01_TRUE                                                0x00000001
#define LW8597_SET_STREAMING_REORDER_WRITE_ADDRESS_COMP02                                                   22:16
#define LW8597_SET_STREAMING_REORDER_ENABLE_COMP02                                                          23:23
#define LW8597_SET_STREAMING_REORDER_ENABLE_COMP02_FALSE                                               0x00000000
#define LW8597_SET_STREAMING_REORDER_ENABLE_COMP02_TRUE                                                0x00000001
#define LW8597_SET_STREAMING_REORDER_WRITE_ADDRESS_COMP03                                                   30:24
#define LW8597_SET_STREAMING_REORDER_ENABLE_COMP03                                                          31:31
#define LW8597_SET_STREAMING_REORDER_ENABLE_COMP03_FALSE                                               0x00000000
#define LW8597_SET_STREAMING_REORDER_ENABLE_COMP03_TRUE                                                0x00000001

#define LW8597_SET_SURFACE_CLIP_ID_HEIGHT                                                                  0x1504
#define LW8597_SET_SURFACE_CLIP_ID_HEIGHT_V                                                                  31:0

#define LW8597_SET_CLIP_ID_CLEAR_RECT_HORIZONTAL                                                           0x1508
#define LW8597_SET_CLIP_ID_CLEAR_RECT_HORIZONTAL_XMIN                                                        15:0
#define LW8597_SET_CLIP_ID_CLEAR_RECT_HORIZONTAL_XMAX                                                       31:16

#define LW8597_SET_CLIP_ID_CLEAR_RECT_VERTICAL                                                             0x150c
#define LW8597_SET_CLIP_ID_CLEAR_RECT_VERTICAL_YMIN                                                          15:0
#define LW8597_SET_CLIP_ID_CLEAR_RECT_VERTICAL_YMAX                                                         31:16

#define LW8597_SET_USER_CLIP_ENABLE                                                                        0x1510
#define LW8597_SET_USER_CLIP_ENABLE_PLANE0                                                                    0:0
#define LW8597_SET_USER_CLIP_ENABLE_PLANE0_FALSE                                                       0x00000000
#define LW8597_SET_USER_CLIP_ENABLE_PLANE0_TRUE                                                        0x00000001
#define LW8597_SET_USER_CLIP_ENABLE_PLANE1                                                                    1:1
#define LW8597_SET_USER_CLIP_ENABLE_PLANE1_FALSE                                                       0x00000000
#define LW8597_SET_USER_CLIP_ENABLE_PLANE1_TRUE                                                        0x00000001
#define LW8597_SET_USER_CLIP_ENABLE_PLANE2                                                                    2:2
#define LW8597_SET_USER_CLIP_ENABLE_PLANE2_FALSE                                                       0x00000000
#define LW8597_SET_USER_CLIP_ENABLE_PLANE2_TRUE                                                        0x00000001
#define LW8597_SET_USER_CLIP_ENABLE_PLANE3                                                                    3:3
#define LW8597_SET_USER_CLIP_ENABLE_PLANE3_FALSE                                                       0x00000000
#define LW8597_SET_USER_CLIP_ENABLE_PLANE3_TRUE                                                        0x00000001
#define LW8597_SET_USER_CLIP_ENABLE_PLANE4                                                                    4:4
#define LW8597_SET_USER_CLIP_ENABLE_PLANE4_FALSE                                                       0x00000000
#define LW8597_SET_USER_CLIP_ENABLE_PLANE4_TRUE                                                        0x00000001
#define LW8597_SET_USER_CLIP_ENABLE_PLANE5                                                                    5:5
#define LW8597_SET_USER_CLIP_ENABLE_PLANE5_FALSE                                                       0x00000000
#define LW8597_SET_USER_CLIP_ENABLE_PLANE5_TRUE                                                        0x00000001
#define LW8597_SET_USER_CLIP_ENABLE_PLANE6                                                                    6:6
#define LW8597_SET_USER_CLIP_ENABLE_PLANE6_FALSE                                                       0x00000000
#define LW8597_SET_USER_CLIP_ENABLE_PLANE6_TRUE                                                        0x00000001
#define LW8597_SET_USER_CLIP_ENABLE_PLANE7                                                                    7:7
#define LW8597_SET_USER_CLIP_ENABLE_PLANE7_FALSE                                                       0x00000000
#define LW8597_SET_USER_CLIP_ENABLE_PLANE7_TRUE                                                        0x00000001

#define LW8597_SET_ZPASS_PIXEL_COUNT                                                                       0x1514
#define LW8597_SET_ZPASS_PIXEL_COUNT_ENABLE                                                                   0:0
#define LW8597_SET_ZPASS_PIXEL_COUNT_ENABLE_FALSE                                                      0x00000000
#define LW8597_SET_ZPASS_PIXEL_COUNT_ENABLE_TRUE                                                       0x00000001

#define LW8597_SET_POINT_SIZE                                                                              0x1518
#define LW8597_SET_POINT_SIZE_V                                                                              31:0

#define LW8597_SET_ZLWLL_STATS                                                                             0x151c
#define LW8597_SET_ZLWLL_STATS_ENABLE                                                                         0:0
#define LW8597_SET_ZLWLL_STATS_ENABLE_FALSE                                                            0x00000000
#define LW8597_SET_ZLWLL_STATS_ENABLE_TRUE                                                             0x00000001

#define LW8597_SET_POINT_SPRITE                                                                            0x1520
#define LW8597_SET_POINT_SPRITE_ENABLE                                                                       31:0
#define LW8597_SET_POINT_SPRITE_ENABLE_FALSE                                                           0x00000000
#define LW8597_SET_POINT_SPRITE_ENABLE_TRUE                                                            0x00000001

#define LW8597_SET_SHADER_EXCEPTIONS                                                                       0x1528
#define LW8597_SET_SHADER_EXCEPTIONS_ENABLE                                                                   0:0
#define LW8597_SET_SHADER_EXCEPTIONS_ENABLE_FALSE                                                      0x00000000
#define LW8597_SET_SHADER_EXCEPTIONS_ENABLE_TRUE                                                       0x00000001

#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_D                                             0x152c
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_D_AVOID_FUTILE_LOAD_BALANCE                      0:0
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_D_AVOID_FUTILE_LOAD_BALANCE_FALSE             0x00000000
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_D_AVOID_FUTILE_LOAD_BALANCE_TRUE             0x00000001
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_D_USE_IMPLICIT_DELTA                             4:4
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_D_USE_IMPLICIT_DELTA_FALSE                0x00000000
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_D_USE_IMPLICIT_DELTA_TRUE                 0x00000001
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_D_ENABLE_PS_ALLOC_THROTTLE                       8:8
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_D_ENABLE_PS_ALLOC_THROTTLE_FALSE             0x00000000
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_D_ENABLE_PS_ALLOC_THROTTLE_TRUE             0x00000001
#define LW8597_LOAD_LOCAL_REGISTER_FILE_LOAD_BALANCE_CONTROL_D_PS_ALLOC_THROTTLE_THRESHOLD                  23:12

#define LW8597_CLEAR_REPORT_VALUE                                                                          0x1530
#define LW8597_CLEAR_REPORT_VALUE_TYPE                                                                        4:0
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_VERTICES_GENERATED                                           0x00000012
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_PRIMITIVES_GENERATED                                         0x00000013
#define LW8597_CLEAR_REPORT_VALUE_TYPE_VS_ILWOCATIONS                                                  0x00000015
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_ILWOCATIONS                                                  0x0000001A
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_PRIMITIVES_GENERATED                                         0x0000001B
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_SUCCEEDED                                  0x00000010
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_NEEDED                                     0x00000011
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_ILWOCATIONS                                             0x0000001C
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_PRIMITIVES_GENERATED                                    0x0000001D
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZLWLL_STATS                                                     0x00000002
#define LW8597_CLEAR_REPORT_VALUE_TYPE_PS_ILWOCATIONS                                                  0x0000001E
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZPASS_PIXEL_CNT                                                 0x00000001
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_VERTICES_SUCCEEDED                                    0x00000008
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_VERTICES_GENERATED                                           0x00000012
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_PRIMITIVES_GENERATED                                         0x00000013
#define LW8597_CLEAR_REPORT_VALUE_TYPE_VS_ILWOCATIONS                                                  0x00000015
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_ILWOCATIONS                                                  0x0000001A
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_PRIMITIVES_GENERATED                                         0x0000001B
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_SUCCEEDED                                  0x00000010
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_NEEDED                                     0x00000011
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_ILWOCATIONS                                             0x0000001C
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_PRIMITIVES_GENERATED                                    0x0000001D
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZLWLL_STATS                                                     0x00000002
#define LW8597_CLEAR_REPORT_VALUE_TYPE_PS_ILWOCATIONS                                                  0x0000001E
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZPASS_PIXEL_CNT                                                 0x00000001
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_VERTICES_SUCCEEDED                                    0x00000008
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_VERTICES_GENERATED                                           0x00000012
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_PRIMITIVES_GENERATED                                         0x00000013
#define LW8597_CLEAR_REPORT_VALUE_TYPE_VS_ILWOCATIONS                                                  0x00000015
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_ILWOCATIONS                                                  0x0000001A
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_PRIMITIVES_GENERATED                                         0x0000001B
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_SUCCEEDED                                  0x00000010
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_NEEDED                                     0x00000011
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_ILWOCATIONS                                             0x0000001C
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_PRIMITIVES_GENERATED                                    0x0000001D
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZLWLL_STATS                                                     0x00000002
#define LW8597_CLEAR_REPORT_VALUE_TYPE_PS_ILWOCATIONS                                                  0x0000001E
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZPASS_PIXEL_CNT                                                 0x00000001
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_VERTICES_SUCCEEDED                                    0x00000008
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_VERTICES_GENERATED                                           0x00000012
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_PRIMITIVES_GENERATED                                         0x00000013
#define LW8597_CLEAR_REPORT_VALUE_TYPE_VS_ILWOCATIONS                                                  0x00000015
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_ILWOCATIONS                                                  0x0000001A
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_PRIMITIVES_GENERATED                                         0x0000001B
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_SUCCEEDED                                  0x00000010
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_NEEDED                                     0x00000011
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_ILWOCATIONS                                             0x0000001C
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_PRIMITIVES_GENERATED                                    0x0000001D
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZLWLL_STATS                                                     0x00000002
#define LW8597_CLEAR_REPORT_VALUE_TYPE_PS_ILWOCATIONS                                                  0x0000001E
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZPASS_PIXEL_CNT                                                 0x00000001
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_VERTICES_SUCCEEDED                                    0x00000008
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_VERTICES_GENERATED                                           0x00000012
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_PRIMITIVES_GENERATED                                         0x00000013
#define LW8597_CLEAR_REPORT_VALUE_TYPE_VS_ILWOCATIONS                                                  0x00000015
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_ILWOCATIONS                                                  0x0000001A
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_PRIMITIVES_GENERATED                                         0x0000001B
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_SUCCEEDED                                  0x00000010
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_NEEDED                                     0x00000011
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_ILWOCATIONS                                             0x0000001C
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_PRIMITIVES_GENERATED                                    0x0000001D
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZLWLL_STATS                                                     0x00000002
#define LW8597_CLEAR_REPORT_VALUE_TYPE_PS_ILWOCATIONS                                                  0x0000001E
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZPASS_PIXEL_CNT                                                 0x00000001
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_VERTICES_SUCCEEDED                                    0x00000008
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_VERTICES_GENERATED                                           0x00000012
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_PRIMITIVES_GENERATED                                         0x00000013
#define LW8597_CLEAR_REPORT_VALUE_TYPE_VS_ILWOCATIONS                                                  0x00000015
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_ILWOCATIONS                                                  0x0000001A
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_PRIMITIVES_GENERATED                                         0x0000001B
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_SUCCEEDED                                  0x00000010
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_NEEDED                                     0x00000011
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_ILWOCATIONS                                             0x0000001C
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_PRIMITIVES_GENERATED                                    0x0000001D
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZLWLL_STATS                                                     0x00000002
#define LW8597_CLEAR_REPORT_VALUE_TYPE_PS_ILWOCATIONS                                                  0x0000001E
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZPASS_PIXEL_CNT                                                 0x00000001
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_VERTICES_SUCCEEDED                                    0x00000008
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_VERTICES_GENERATED                                           0x00000012
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_PRIMITIVES_GENERATED                                         0x00000013
#define LW8597_CLEAR_REPORT_VALUE_TYPE_VS_ILWOCATIONS                                                  0x00000015
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_ILWOCATIONS                                                  0x0000001A
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_PRIMITIVES_GENERATED                                         0x0000001B
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_SUCCEEDED                                  0x00000010
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_NEEDED                                     0x00000011
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_ILWOCATIONS                                             0x0000001C
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_PRIMITIVES_GENERATED                                    0x0000001D
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZLWLL_STATS                                                     0x00000002
#define LW8597_CLEAR_REPORT_VALUE_TYPE_PS_ILWOCATIONS                                                  0x0000001E
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZPASS_PIXEL_CNT                                                 0x00000001
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_VERTICES_SUCCEEDED                                    0x00000008
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_VERTICES_GENERATED                                           0x00000012
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_PRIMITIVES_GENERATED                                         0x00000013
#define LW8597_CLEAR_REPORT_VALUE_TYPE_VS_ILWOCATIONS                                                  0x00000015
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_ILWOCATIONS                                                  0x0000001A
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_PRIMITIVES_GENERATED                                         0x0000001B
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_SUCCEEDED                                  0x00000010
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_NEEDED                                     0x00000011
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_ILWOCATIONS                                             0x0000001C
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_PRIMITIVES_GENERATED                                    0x0000001D
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZLWLL_STATS                                                     0x00000002
#define LW8597_CLEAR_REPORT_VALUE_TYPE_PS_ILWOCATIONS                                                  0x0000001E
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZPASS_PIXEL_CNT                                                 0x00000001
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_VERTICES_SUCCEEDED                                    0x00000008
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_VERTICES_GENERATED                                           0x00000012
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_PRIMITIVES_GENERATED                                         0x00000013
#define LW8597_CLEAR_REPORT_VALUE_TYPE_VS_ILWOCATIONS                                                  0x00000015
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_ILWOCATIONS                                                  0x0000001A
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_PRIMITIVES_GENERATED                                         0x0000001B
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_SUCCEEDED                                  0x00000010
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_NEEDED                                     0x00000011
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_ILWOCATIONS                                             0x0000001C
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_PRIMITIVES_GENERATED                                    0x0000001D
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZLWLL_STATS                                                     0x00000002
#define LW8597_CLEAR_REPORT_VALUE_TYPE_PS_ILWOCATIONS                                                  0x0000001E
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZPASS_PIXEL_CNT                                                 0x00000001
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_VERTICES_SUCCEEDED                                    0x00000008
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_VERTICES_GENERATED                                           0x00000012
#define LW8597_CLEAR_REPORT_VALUE_TYPE_DA_PRIMITIVES_GENERATED                                         0x00000013
#define LW8597_CLEAR_REPORT_VALUE_TYPE_VS_ILWOCATIONS                                                  0x00000015
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_ILWOCATIONS                                                  0x0000001A
#define LW8597_CLEAR_REPORT_VALUE_TYPE_GS_PRIMITIVES_GENERATED                                         0x0000001B
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_SUCCEEDED                                  0x00000010
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_NEEDED                                     0x00000011
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_ILWOCATIONS                                             0x0000001C
#define LW8597_CLEAR_REPORT_VALUE_TYPE_CLIPPER_PRIMITIVES_GENERATED                                    0x0000001D
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZLWLL_STATS                                                     0x00000002
#define LW8597_CLEAR_REPORT_VALUE_TYPE_PS_ILWOCATIONS                                                  0x0000001E
#define LW8597_CLEAR_REPORT_VALUE_TYPE_ZPASS_PIXEL_CNT                                                 0x00000001
#define LW8597_CLEAR_REPORT_VALUE_TYPE_STREAMING_VERTICES_SUCCEEDED                                    0x00000008

#define LW8597_SET_ANTI_ALIAS_ENABLE                                                                       0x1534
#define LW8597_SET_ANTI_ALIAS_ENABLE_V                                                                       31:0
#define LW8597_SET_ANTI_ALIAS_ENABLE_V_FALSE                                                           0x00000000
#define LW8597_SET_ANTI_ALIAS_ENABLE_V_TRUE                                                            0x00000001

#define LW8597_SET_ZT_SELECT                                                                               0x1538
#define LW8597_SET_ZT_SELECT_TARGET_COUNT                                                                     0:0

#define LW8597_SET_ANTI_ALIAS_ALPHA_CONTROL                                                                0x153c
#define LW8597_SET_ANTI_ALIAS_ALPHA_CONTROL_ALPHA_TO_COVERAGE                                                 3:0
#define LW8597_SET_ANTI_ALIAS_ALPHA_CONTROL_ALPHA_TO_COVERAGE_DISABLE                                  0x00000000
#define LW8597_SET_ANTI_ALIAS_ALPHA_CONTROL_ALPHA_TO_COVERAGE_ENABLE                                   0x00000001
#define LW8597_SET_ANTI_ALIAS_ALPHA_CONTROL_ALPHA_TO_ONE                                                      7:4
#define LW8597_SET_ANTI_ALIAS_ALPHA_CONTROL_ALPHA_TO_ONE_DISABLE                                       0x00000000
#define LW8597_SET_ANTI_ALIAS_ALPHA_CONTROL_ALPHA_TO_ONE_ENABLE                                        0x00000001

#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE(i)                                                  (0x1540+(i)*4)
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP00                                                         0:0
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP00_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP00_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP01                                                         1:1
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP01_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP01_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP02                                                         2:2
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP02_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP02_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP03                                                         3:3
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP03_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP03_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP04                                                         4:4
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP04_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP04_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP05                                                         5:5
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP05_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP05_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP06                                                         6:6
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP06_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP06_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP07                                                         7:7
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP07_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP07_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP08                                                         8:8
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP08_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP08_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP09                                                         9:9
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP09_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP09_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP10                                                       10:10
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP10_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP10_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP11                                                       11:11
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP11_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP11_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP12                                                       12:12
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP12_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP12_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP13                                                       13:13
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP13_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP13_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP14                                                       14:14
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP14_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP14_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP15                                                       15:15
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP15_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP15_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP16                                                       16:16
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP16_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP16_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP17                                                       17:17
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP17_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP17_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP18                                                       18:18
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP18_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP18_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP19                                                       19:19
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP19_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP19_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP20                                                       20:20
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP20_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP20_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP21                                                       21:21
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP21_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP21_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP22                                                       22:22
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP22_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP22_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP23                                                       23:23
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP23_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP23_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP24                                                       24:24
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP24_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP24_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP25                                                       25:25
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP25_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP25_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP26                                                       26:26
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP26_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP26_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP27                                                       27:27
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP27_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP27_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP28                                                       28:28
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP28_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP28_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP29                                                       29:29
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP29_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP29_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP30                                                       30:30
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP30_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP30_NONPERSPECTIVE                                   0x00000001
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP31                                                       31:31
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP31_PERSPECTIVE                                      0x00000000
#define LW8597_SET_PS_INPUT_INTERPOLATION_TYPE_COMP31_NONPERSPECTIVE                                   0x00000001

#define LW8597_SET_RENDER_ENABLE_A                                                                         0x1550
#define LW8597_SET_RENDER_ENABLE_A_OFFSET_UPPER                                                               7:0

#define LW8597_SET_RENDER_ENABLE_B                                                                         0x1554
#define LW8597_SET_RENDER_ENABLE_B_OFFSET_LOWER                                                              31:0

#define LW8597_SET_RENDER_ENABLE_C                                                                         0x1558
#define LW8597_SET_RENDER_ENABLE_C_MODE                                                                       2:0
#define LW8597_SET_RENDER_ENABLE_C_MODE_FALSE                                                          0x00000000
#define LW8597_SET_RENDER_ENABLE_C_MODE_TRUE                                                           0x00000001
#define LW8597_SET_RENDER_ENABLE_C_MODE_CONDITIONAL                                                    0x00000002
#define LW8597_SET_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL                                                0x00000003
#define LW8597_SET_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL                                            0x00000004

#define LW8597_SET_TEX_SAMPLER_POOL_A                                                                      0x155c
#define LW8597_SET_TEX_SAMPLER_POOL_A_OFFSET_UPPER                                                            7:0

#define LW8597_SET_TEX_SAMPLER_POOL_B                                                                      0x1560
#define LW8597_SET_TEX_SAMPLER_POOL_B_OFFSET_LOWER                                                           31:0

#define LW8597_SET_TEX_SAMPLER_POOL_C                                                                      0x1564
#define LW8597_SET_TEX_SAMPLER_POOL_C_MAXIMUM_INDEX                                                          19:0

#define LW8597_SET_SHADER_ERROR_TRAP_CONTROL                                                               0x1568
#define LW8597_SET_SHADER_ERROR_TRAP_CONTROL_MASTER_MASK                                                      0:0
#define LW8597_SET_SHADER_ERROR_TRAP_CONTROL_MASTER_MASK_FALSE                                         0x00000000
#define LW8597_SET_SHADER_ERROR_TRAP_CONTROL_MASTER_MASK_TRUE                                          0x00000001
#define LW8597_SET_SHADER_ERROR_TRAP_CONTROL_SUBSET_MASK                                                     31:1

#define LW8597_SET_SLOPE_SCALE_DEPTH_BIAS                                                                  0x156c
#define LW8597_SET_SLOPE_SCALE_DEPTH_BIAS_V                                                                  31:0

#define LW8597_SET_ANTI_ALIASED_LINE                                                                       0x1570
#define LW8597_SET_ANTI_ALIASED_LINE_ENABLE                                                                  31:0
#define LW8597_SET_ANTI_ALIASED_LINE_ENABLE_FALSE                                                      0x00000000
#define LW8597_SET_ANTI_ALIASED_LINE_ENABLE_TRUE                                                       0x00000001

#define LW8597_SET_TEX_HEADER_POOL_A                                                                       0x1574
#define LW8597_SET_TEX_HEADER_POOL_A_OFFSET_UPPER                                                             7:0

#define LW8597_SET_TEX_HEADER_POOL_B                                                                       0x1578
#define LW8597_SET_TEX_HEADER_POOL_B_OFFSET_LOWER                                                            31:0

#define LW8597_SET_TEX_HEADER_POOL_C                                                                       0x157c
#define LW8597_SET_TEX_HEADER_POOL_C_MAXIMUM_INDEX                                                           21:0

#define LW8597_SET_SHADER_PERFORMANCE_COUNTER_CONTROL(i)                                           (0x1580+(i)*4)
#define LW8597_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_EDGE                                                    0:0
#define LW8597_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_BLOCK                                                   6:4
#define LW8597_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_BLOCK_ACE                                        0x00000000
#define LW8597_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_BLOCK_DIS                                        0x00000001
#define LW8597_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_BLOCK_DSM                                        0x00000002
#define LW8597_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_BLOCK_PIC                                        0x00000003
#define LW8597_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_BLOCK_STP                                        0x00000004
#define LW8597_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_BLOCK_XIU                                        0x00000005
#define LW8597_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_FUNC                                                   23:8
#define LW8597_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_EVENT                                                 31:24

#define LW8597_SET_ACTIVE_ZLWLL_REGION                                                                     0x1590
#define LW8597_SET_ACTIVE_ZLWLL_REGION_ID                                                                     5:0

#define LW8597_SET_TWO_SIDED_STENCIL_TEST                                                                  0x1594
#define LW8597_SET_TWO_SIDED_STENCIL_TEST_ENABLE                                                             31:0
#define LW8597_SET_TWO_SIDED_STENCIL_TEST_ENABLE_FALSE                                                 0x00000000
#define LW8597_SET_TWO_SIDED_STENCIL_TEST_ENABLE_TRUE                                                  0x00000001

#define LW8597_SET_BACK_STENCIL_OP_FAIL                                                                    0x1598
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V                                                                    31:0
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V_OGL_KEEP                                                     0x00001E00
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V_OGL_ZERO                                                     0x00000000
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V_OGL_REPLACE                                                  0x00001E01
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V_OGL_INCRSAT                                                  0x00001E02
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V_OGL_DECRSAT                                                  0x00001E03
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V_OGL_ILWERT                                                   0x0000150A
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V_OGL_INCR                                                     0x00008507
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V_OGL_DECR                                                     0x00008508
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V_D3D_KEEP                                                     0x00000001
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V_D3D_ZERO                                                     0x00000002
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V_D3D_REPLACE                                                  0x00000003
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V_D3D_INCRSAT                                                  0x00000004
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V_D3D_DECRSAT                                                  0x00000005
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V_D3D_ILWERT                                                   0x00000006
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V_D3D_INCR                                                     0x00000007
#define LW8597_SET_BACK_STENCIL_OP_FAIL_V_D3D_DECR                                                     0x00000008

#define LW8597_SET_BACK_STENCIL_OP_ZFAIL                                                                   0x159c
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V                                                                   31:0
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V_OGL_KEEP                                                    0x00001E00
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V_OGL_ZERO                                                    0x00000000
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V_OGL_REPLACE                                                 0x00001E01
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V_OGL_INCRSAT                                                 0x00001E02
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V_OGL_DECRSAT                                                 0x00001E03
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V_OGL_ILWERT                                                  0x0000150A
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V_OGL_INCR                                                    0x00008507
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V_OGL_DECR                                                    0x00008508
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V_D3D_KEEP                                                    0x00000001
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V_D3D_ZERO                                                    0x00000002
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V_D3D_REPLACE                                                 0x00000003
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V_D3D_INCRSAT                                                 0x00000004
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V_D3D_DECRSAT                                                 0x00000005
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V_D3D_ILWERT                                                  0x00000006
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V_D3D_INCR                                                    0x00000007
#define LW8597_SET_BACK_STENCIL_OP_ZFAIL_V_D3D_DECR                                                    0x00000008

#define LW8597_SET_BACK_STENCIL_OP_ZPASS                                                                   0x15a0
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V                                                                   31:0
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V_OGL_KEEP                                                    0x00001E00
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V_OGL_ZERO                                                    0x00000000
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V_OGL_REPLACE                                                 0x00001E01
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V_OGL_INCRSAT                                                 0x00001E02
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V_OGL_DECRSAT                                                 0x00001E03
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V_OGL_ILWERT                                                  0x0000150A
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V_OGL_INCR                                                    0x00008507
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V_OGL_DECR                                                    0x00008508
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V_D3D_KEEP                                                    0x00000001
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V_D3D_ZERO                                                    0x00000002
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V_D3D_REPLACE                                                 0x00000003
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V_D3D_INCRSAT                                                 0x00000004
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V_D3D_DECRSAT                                                 0x00000005
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V_D3D_ILWERT                                                  0x00000006
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V_D3D_INCR                                                    0x00000007
#define LW8597_SET_BACK_STENCIL_OP_ZPASS_V_D3D_DECR                                                    0x00000008

#define LW8597_SET_BACK_STENCIL_FUNC                                                                       0x15a4
#define LW8597_SET_BACK_STENCIL_FUNC_V                                                                       31:0
#define LW8597_SET_BACK_STENCIL_FUNC_V_OGL_NEVER                                                       0x00000200
#define LW8597_SET_BACK_STENCIL_FUNC_V_OGL_LESS                                                        0x00000201
#define LW8597_SET_BACK_STENCIL_FUNC_V_OGL_EQUAL                                                       0x00000202
#define LW8597_SET_BACK_STENCIL_FUNC_V_OGL_LEQUAL                                                      0x00000203
#define LW8597_SET_BACK_STENCIL_FUNC_V_OGL_GREATER                                                     0x00000204
#define LW8597_SET_BACK_STENCIL_FUNC_V_OGL_NOTEQUAL                                                    0x00000205
#define LW8597_SET_BACK_STENCIL_FUNC_V_OGL_GEQUAL                                                      0x00000206
#define LW8597_SET_BACK_STENCIL_FUNC_V_OGL_ALWAYS                                                      0x00000207
#define LW8597_SET_BACK_STENCIL_FUNC_V_D3D_NEVER                                                       0x00000001
#define LW8597_SET_BACK_STENCIL_FUNC_V_D3D_LESS                                                        0x00000002
#define LW8597_SET_BACK_STENCIL_FUNC_V_D3D_EQUAL                                                       0x00000003
#define LW8597_SET_BACK_STENCIL_FUNC_V_D3D_LESSEQUAL                                                   0x00000004
#define LW8597_SET_BACK_STENCIL_FUNC_V_D3D_GREATER                                                     0x00000005
#define LW8597_SET_BACK_STENCIL_FUNC_V_D3D_NOTEQUAL                                                    0x00000006
#define LW8597_SET_BACK_STENCIL_FUNC_V_D3D_GREATEREQUAL                                                0x00000007
#define LW8597_SET_BACK_STENCIL_FUNC_V_D3D_ALWAYS                                                      0x00000008

#define LW8597_SET_PHASE_ID_CONTROL                                                                        0x15a8
#define LW8597_SET_PHASE_ID_CONTROL_WINDOW_SIZE                                                               2:0
#define LW8597_SET_PHASE_ID_CONTROL_LOCK_PHASE                                                                6:4

#define LW8597_SET_MRT_PERFORMANCE_CONTROL                                                                 0x15ac
#define LW8597_SET_MRT_PERFORMANCE_CONTROL_SEGMENT                                                            1:0
#define LW8597_SET_MRT_PERFORMANCE_CONTROL_SEGMENT_LENGTH_8PIXELS                                      0x00000000
#define LW8597_SET_MRT_PERFORMANCE_CONTROL_SEGMENT_LENGTH_16PIXELS                                     0x00000001
#define LW8597_SET_MRT_PERFORMANCE_CONTROL_SEGMENT_LENGTH_32PIXELS                                     0x00000002
#define LW8597_SET_MRT_PERFORMANCE_CONTROL_SEGMENT_LENGTH_64PIXELS                                     0x00000003

#define LW8597_PREFETCH_SHADER_INSTRUCTIONS                                                                0x15b0
#define LW8597_PREFETCH_SHADER_INSTRUCTIONS_VS                                                                0:0
#define LW8597_PREFETCH_SHADER_INSTRUCTIONS_VS_FALSE                                                   0x00000000
#define LW8597_PREFETCH_SHADER_INSTRUCTIONS_VS_TRUE                                                    0x00000001
#define LW8597_PREFETCH_SHADER_INSTRUCTIONS_GS                                                                4:4
#define LW8597_PREFETCH_SHADER_INSTRUCTIONS_GS_FALSE                                                   0x00000000
#define LW8597_PREFETCH_SHADER_INSTRUCTIONS_GS_TRUE                                                    0x00000001
#define LW8597_PREFETCH_SHADER_INSTRUCTIONS_PS                                                                8:8
#define LW8597_PREFETCH_SHADER_INSTRUCTIONS_PS_FALSE                                                   0x00000000
#define LW8597_PREFETCH_SHADER_INSTRUCTIONS_PS_TRUE                                                    0x00000001

#define LW8597_SET_VCAA                                                                                    0x15b4
#define LW8597_SET_VCAA_WRITE_ENABLE                                                                          0:0
#define LW8597_SET_VCAA_WRITE_ENABLE_FALSE                                                             0x00000000
#define LW8597_SET_VCAA_WRITE_ENABLE_TRUE                                                              0x00000001

#define LW8597_SET_SRGB_WRITE                                                                              0x15b8
#define LW8597_SET_SRGB_WRITE_ENABLE                                                                         31:0
#define LW8597_SET_SRGB_WRITE_ENABLE_FALSE                                                             0x00000000
#define LW8597_SET_SRGB_WRITE_ENABLE_TRUE                                                              0x00000001

#define LW8597_SET_DEPTH_BIAS                                                                              0x15bc
#define LW8597_SET_DEPTH_BIAS_V                                                                              31:0

#define LW8597_SET_FP32_BLEND_ROUNDING                                                                     0x15c4
#define LW8597_SET_FP32_BLEND_ROUNDING_V                                                                      0:0
#define LW8597_SET_FP32_BLEND_ROUNDING_V_TRUNCATION                                                    0x00000000
#define LW8597_SET_FP32_BLEND_ROUNDING_V_ROUND_TO_NEAREST_EVEN                                         0x00000001

#define LW8597_SET_VCAA_SAMPLE_MASK_INTERACTION                                                            0x15c8
#define LW8597_SET_VCAA_SAMPLE_MASK_INTERACTION_V                                                             3:0
#define LW8597_SET_VCAA_SAMPLE_MASK_INTERACTION_V_ALLOW_ERRORS_ON_TRIANGLE_EDGES                       0x00000000
#define LW8597_SET_VCAA_SAMPLE_MASK_INTERACTION_V_CORRECTED_BEHAVIOUR                                  0x00000001
#define LW8597_SET_VCAA_SAMPLE_MASK_INTERACTION_V_FORCE_VCAA_COVG_REPLACE                              0x00000002

#define LW8597_SET_RT_LAYER                                                                                0x15cc
#define LW8597_SET_RT_LAYER_V                                                                                15:0
#define LW8597_SET_RT_LAYER_CONTROL                                                                         16:16
#define LW8597_SET_RT_LAYER_CONTROL_V_SELECTS_LAYER                                                    0x00000000
#define LW8597_SET_RT_LAYER_CONTROL_GEOMETRY_SHADER_SELECTS_LAYER                                      0x00000001

#define LW8597_SET_ANTI_ALIAS                                                                              0x15d0
#define LW8597_SET_ANTI_ALIAS_SAMPLES                                                                         6:0
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_1X1                                                         0x00000000
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1                                                         0x00000001
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2                                                         0x00000002
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2                                                         0x00000003
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_D3D                                                     0x00000004
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1_D3D                                                     0x00000005
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_4                                                    0x00000008
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_12                                                   0x00000009
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_VC_8                                                    0x0000000A
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_1X1                                                         0x00000000
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1                                                         0x00000001
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2                                                         0x00000002
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2                                                         0x00000003
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_D3D                                                     0x00000004
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1_D3D                                                     0x00000005
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_4                                                    0x00000008
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_12                                                   0x00000009
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_VC_8                                                    0x0000000A
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_1X1                                                         0x00000000
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1                                                         0x00000001
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2                                                         0x00000002
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2                                                         0x00000003
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_D3D                                                     0x00000004
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1_D3D                                                     0x00000005
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_4                                                    0x00000008
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_12                                                   0x00000009
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_VC_8                                                    0x0000000A
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_1X1                                                         0x00000000
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1                                                         0x00000001
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2                                                         0x00000002
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2                                                         0x00000003
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_D3D                                                     0x00000004
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1_D3D                                                     0x00000005
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_4                                                    0x00000008
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_12                                                   0x00000009
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_VC_8                                                    0x0000000A
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_1X1                                                         0x00000000
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1                                                         0x00000001
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2                                                         0x00000002
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2                                                         0x00000003
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_D3D                                                     0x00000004
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1_D3D                                                     0x00000005
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_4                                                    0x00000008
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_12                                                   0x00000009
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_VC_8                                                    0x0000000A
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_1X1                                                         0x00000000
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1                                                         0x00000001
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2                                                         0x00000002
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2                                                         0x00000003
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_D3D                                                     0x00000004
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1_D3D                                                     0x00000005
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_4                                                    0x00000008
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_12                                                   0x00000009
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_VC_8                                                    0x0000000A
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_1X1                                                         0x00000000
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1                                                         0x00000001
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2                                                         0x00000002
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2                                                         0x00000003
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_D3D                                                     0x00000004
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1_D3D                                                     0x00000005
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_4                                                    0x00000008
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_12                                                   0x00000009
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_VC_8                                                    0x0000000A
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_1X1                                                         0x00000000
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1                                                         0x00000001
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2                                                         0x00000002
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2                                                         0x00000003
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_D3D                                                     0x00000004
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1_D3D                                                     0x00000005
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_4                                                    0x00000008
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_12                                                   0x00000009
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_VC_8                                                    0x0000000A
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_1X1                                                         0x00000000
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1                                                         0x00000001
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2                                                         0x00000002
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2                                                         0x00000003
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_D3D                                                     0x00000004
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1_D3D                                                     0x00000005
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_4                                                    0x00000008
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_12                                                   0x00000009
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_VC_8                                                    0x0000000A
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_1X1                                                         0x00000000
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1                                                         0x00000001
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2                                                         0x00000002
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2                                                         0x00000003
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_D3D                                                     0x00000004
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X1_D3D                                                     0x00000005
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_4                                                    0x00000008
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_12                                                   0x00000009
#define LW8597_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_VC_8                                                    0x0000000A

#define LW8597_D3D_BEGIN                                                                                   0x15d4
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY                                                                  27:0
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_POINTLIST                                                  0x00000001
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_LINELIST                                                   0x00000002
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_LINESTRIP                                                  0x00000003
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_TRIANGLELIST                                               0x00000004
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP                                              0x00000005
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_LINELIST_ADJCY                                             0x0000000A
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJCY                                            0x0000000B
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJCY                                         0x0000000C
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJCY                                        0x0000000D
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_D3D_POINTS                                                 0x00001001
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_D3D_INDEXEDLINELIST                                        0x00001002
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_D3D_INDEXEDTRIANGLELIST                                    0x00001003
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_D3D_LINELIST                                               0x0000100F
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_D3D_LINESTRIP                                              0x00001010
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_D3D_INDEXEDLINESTRIP                                       0x00001011
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_D3D_TRIANGLELIST                                           0x00001012
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_D3D_TRIANGLESTRIP                                          0x00001013
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_D3D_INDEXEDTRIANGLESTRIP                                   0x00001014
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_D3D_TRIANGLEFAN                                            0x00001015
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_D3D_INDEXEDTRIANGLEFAN                                     0x00001016
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_D3D_TRIANGLEFAN_IMM                                        0x00001017
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_D3D_LINELIST_IMM                                           0x00001018
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_D3D_INDEXEDTRIANGLELIST2                                   0x0000101A
#define LW8597_D3D_BEGIN_PRIMITIVE_TOPOLOGY_D3D_INDEXEDLINELIST2                                       0x0000101B
#define LW8597_D3D_BEGIN_INSTANCE                                                                           28:28
#define LW8597_D3D_BEGIN_INSTANCE_FIRST                                                                0x00000000
#define LW8597_D3D_BEGIN_INSTANCE_SUBSEQUENT                                                           0x00000001
#define LW8597_D3D_BEGIN_RESUME                                                                             29:29
#define LW8597_D3D_BEGIN_RESUME_FALSE                                                                  0x00000000
#define LW8597_D3D_BEGIN_RESUME_TRUE                                                                   0x00000001
#define LW8597_D3D_BEGIN_RESUME_INSTANCE                                                                    30:30
#define LW8597_D3D_BEGIN_RESUME_INSTANCE_FALSE                                                         0x00000000
#define LW8597_D3D_BEGIN_RESUME_INSTANCE_TRUE                                                          0x00000001

#define LW8597_D3D_END                                                                                     0x15d8
#define LW8597_D3D_END_V                                                                                      0:0
#define LW8597_D3D_END_OPEN                                                                                   1:1
#define LW8597_D3D_END_OPEN_FALSE                                                                      0x00000000
#define LW8597_D3D_END_OPEN_TRUE                                                                       0x00000001

#define LW8597_OGL_BEGIN                                                                                   0x15dc
#define LW8597_OGL_BEGIN_OP                                                                                  27:0
#define LW8597_OGL_BEGIN_OP_POINTS                                                                     0x00000000
#define LW8597_OGL_BEGIN_OP_LINES                                                                      0x00000001
#define LW8597_OGL_BEGIN_OP_LINE_LOOP                                                                  0x00000002
#define LW8597_OGL_BEGIN_OP_LINE_STRIP                                                                 0x00000003
#define LW8597_OGL_BEGIN_OP_TRIANGLES                                                                  0x00000004
#define LW8597_OGL_BEGIN_OP_TRIANGLE_STRIP                                                             0x00000005
#define LW8597_OGL_BEGIN_OP_TRIANGLE_FAN                                                               0x00000006
#define LW8597_OGL_BEGIN_OP_QUADS                                                                      0x00000007
#define LW8597_OGL_BEGIN_OP_QUAD_STRIP                                                                 0x00000008
#define LW8597_OGL_BEGIN_OP_POLYGON                                                                    0x00000009
#define LW8597_OGL_BEGIN_OP_LINELIST_ADJCY                                                             0x0000000A
#define LW8597_OGL_BEGIN_OP_LINESTRIP_ADJCY                                                            0x0000000B
#define LW8597_OGL_BEGIN_OP_TRIANGLELIST_ADJCY                                                         0x0000000C
#define LW8597_OGL_BEGIN_OP_TRIANGLESTRIP_ADJCY                                                        0x0000000D
#define LW8597_OGL_BEGIN_INSTANCE                                                                           28:28
#define LW8597_OGL_BEGIN_INSTANCE_FIRST                                                                0x00000000
#define LW8597_OGL_BEGIN_INSTANCE_SUBSEQUENT                                                           0x00000001
#define LW8597_OGL_BEGIN_RESUME                                                                             29:29
#define LW8597_OGL_BEGIN_RESUME_FALSE                                                                  0x00000000
#define LW8597_OGL_BEGIN_RESUME_TRUE                                                                   0x00000001
#define LW8597_OGL_BEGIN_RESUME_INSTANCE                                                                    30:30
#define LW8597_OGL_BEGIN_RESUME_INSTANCE_FALSE                                                         0x00000000
#define LW8597_OGL_BEGIN_RESUME_INSTANCE_TRUE                                                          0x00000001

#define LW8597_OGL_END                                                                                     0x15e0
#define LW8597_OGL_END_V                                                                                      0:0
#define LW8597_OGL_END_OPEN                                                                                   1:1
#define LW8597_OGL_END_OPEN_FALSE                                                                      0x00000000
#define LW8597_OGL_END_OPEN_TRUE                                                                       0x00000001

#define LW8597_SET_EDGE_FLAG                                                                               0x15e4
#define LW8597_SET_EDGE_FLAG_V                                                                               31:0
#define LW8597_SET_EDGE_FLAG_V_FALSE                                                                   0x00000000
#define LW8597_SET_EDGE_FLAG_V_TRUE                                                                    0x00000001

#define LW8597_DRAW_INLINE_INDEX                                                                           0x15e8
#define LW8597_DRAW_INLINE_INDEX_V                                                                           31:0

#define LW8597_SET_INLINE_INDEX2X16_ALIGN                                                                  0x15ec
#define LW8597_SET_INLINE_INDEX2X16_ALIGN_COUNT                                                              30:0
#define LW8597_SET_INLINE_INDEX2X16_ALIGN_START_ODD                                                         31:31
#define LW8597_SET_INLINE_INDEX2X16_ALIGN_START_ODD_FALSE                                              0x00000000
#define LW8597_SET_INLINE_INDEX2X16_ALIGN_START_ODD_TRUE                                               0x00000001

#define LW8597_DRAW_INLINE_INDEX2X16                                                                       0x15f0
#define LW8597_DRAW_INLINE_INDEX2X16_EVEN                                                                    15:0
#define LW8597_DRAW_INLINE_INDEX2X16_ODD                                                                    31:16

#define LW8597_SET_VERTEX_GLOBAL_BASE_OFFSET_A                                                             0x15f4
#define LW8597_SET_VERTEX_GLOBAL_BASE_OFFSET_A_UPPER                                                          7:0

#define LW8597_SET_VERTEX_GLOBAL_BASE_OFFSET_B                                                             0x15f8
#define LW8597_SET_VERTEX_GLOBAL_BASE_OFFSET_B_LOWER                                                         31:0

#define LW8597_DRAW_INLINE_VERTEX                                                                          0x1640
#define LW8597_DRAW_INLINE_VERTEX_V                                                                          31:0

#define LW8597_SET_DA_PRIMITIVE_RESTART                                                                    0x1644
#define LW8597_SET_DA_PRIMITIVE_RESTART_ENABLE                                                                0:0
#define LW8597_SET_DA_PRIMITIVE_RESTART_ENABLE_FALSE                                                   0x00000000
#define LW8597_SET_DA_PRIMITIVE_RESTART_ENABLE_TRUE                                                    0x00000001

#define LW8597_SET_DA_PRIMITIVE_RESTART_INDEX                                                              0x1648
#define LW8597_SET_DA_PRIMITIVE_RESTART_INDEX_V                                                              31:0

#define LW8597_SET_DA_OUTPUT                                                                               0x164c
#define LW8597_SET_DA_OUTPUT_VERTEX_ID_ENABLE                                                                 0:0
#define LW8597_SET_DA_OUTPUT_VERTEX_ID_ENABLE_FALSE                                                    0x00000000
#define LW8597_SET_DA_OUTPUT_VERTEX_ID_ENABLE_TRUE                                                     0x00000001
#define LW8597_SET_DA_OUTPUT_INSTANCE_ID_ENABLE                                                               4:4
#define LW8597_SET_DA_OUTPUT_INSTANCE_ID_ENABLE_FALSE                                                  0x00000000
#define LW8597_SET_DA_OUTPUT_INSTANCE_ID_ENABLE_TRUE                                                   0x00000001
#define LW8597_SET_DA_OUTPUT_PRIMITIVE_ID_ENABLE                                                              8:8
#define LW8597_SET_DA_OUTPUT_PRIMITIVE_ID_ENABLE_FALSE                                                 0x00000000
#define LW8597_SET_DA_OUTPUT_PRIMITIVE_ID_ENABLE_TRUE                                                  0x00000001
#define LW8597_SET_DA_OUTPUT_VERTEX_ID_USES_ARRAY_START                                                     12:12
#define LW8597_SET_DA_OUTPUT_VERTEX_ID_USES_ARRAY_START_FALSE                                          0x00000000
#define LW8597_SET_DA_OUTPUT_VERTEX_ID_USES_ARRAY_START_TRUE                                           0x00000001

#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK(i)                                                     (0x1650+(i)*4)
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE0_COMP0                                                  0:0
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE0_COMP0_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE0_COMP0_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE0_COMP1                                                  1:1
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE0_COMP1_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE0_COMP1_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE0_COMP2                                                  2:2
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE0_COMP2_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE0_COMP2_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE0_COMP3                                                  3:3
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE0_COMP3_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE0_COMP3_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE1_COMP0                                                  4:4
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE1_COMP0_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE1_COMP0_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE1_COMP1                                                  5:5
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE1_COMP1_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE1_COMP1_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE1_COMP2                                                  6:6
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE1_COMP2_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE1_COMP2_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE1_COMP3                                                  7:7
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE1_COMP3_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE1_COMP3_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE2_COMP0                                                  8:8
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE2_COMP0_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE2_COMP0_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE2_COMP1                                                  9:9
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE2_COMP1_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE2_COMP1_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE2_COMP2                                                10:10
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE2_COMP2_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE2_COMP2_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE2_COMP3                                                11:11
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE2_COMP3_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE2_COMP3_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE3_COMP0                                                12:12
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE3_COMP0_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE3_COMP0_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE3_COMP1                                                13:13
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE3_COMP1_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE3_COMP1_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE3_COMP2                                                14:14
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE3_COMP2_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE3_COMP2_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE3_COMP3                                                15:15
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE3_COMP3_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE3_COMP3_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE4_COMP0                                                16:16
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE4_COMP0_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE4_COMP0_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE4_COMP1                                                17:17
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE4_COMP1_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE4_COMP1_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE4_COMP2                                                18:18
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE4_COMP2_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE4_COMP2_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE4_COMP3                                                19:19
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE4_COMP3_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE4_COMP3_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE5_COMP0                                                20:20
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE5_COMP0_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE5_COMP0_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE5_COMP1                                                21:21
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE5_COMP1_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE5_COMP1_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE5_COMP2                                                22:22
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE5_COMP2_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE5_COMP2_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE5_COMP3                                                23:23
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE5_COMP3_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE5_COMP3_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE6_COMP0                                                24:24
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE6_COMP0_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE6_COMP0_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE6_COMP1                                                25:25
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE6_COMP1_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE6_COMP1_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE6_COMP2                                                26:26
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE6_COMP2_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE6_COMP2_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE6_COMP3                                                27:27
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE6_COMP3_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE6_COMP3_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE7_COMP0                                                28:28
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE7_COMP0_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE7_COMP0_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE7_COMP1                                                29:29
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE7_COMP1_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE7_COMP1_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE7_COMP2                                                30:30
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE7_COMP2_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE7_COMP2_TRUE                                      0x00000001
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE7_COMP3                                                31:31
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE7_COMP3_FALSE                                     0x00000000
#define LW8597_SET_DA_OUTPUT_ATTRIBUTE_MASK_ATTRIBUTE7_COMP3_TRUE                                      0x00000001

#define LW8597_SET_ANTI_ALIASED_POINT                                                                      0x1658
#define LW8597_SET_ANTI_ALIASED_POINT_ENABLE                                                                 31:0
#define LW8597_SET_ANTI_ALIASED_POINT_ENABLE_FALSE                                                     0x00000000
#define LW8597_SET_ANTI_ALIASED_POINT_ENABLE_TRUE                                                      0x00000001

#define LW8597_SET_POINT_CENTER_MODE                                                                       0x165c
#define LW8597_SET_POINT_CENTER_MODE_V                                                                       31:0
#define LW8597_SET_POINT_CENTER_MODE_V_OGL                                                             0x00000000
#define LW8597_SET_POINT_CENTER_MODE_V_D3D                                                             0x00000001

#define LW8597_SET_POINT_SPRITE_CONTROL                                                                    0x1660
#define LW8597_SET_POINT_SPRITE_CONTROL_ORIGIN                                                                4:4
#define LW8597_SET_POINT_SPRITE_CONTROL_ORIGIN_BOTTOM                                                  0x00000000
#define LW8597_SET_POINT_SPRITE_CONTROL_ORIGIN_TOP                                                     0x00000001

#define LW8597_SET_LWBEMAP_INTER_FACE_FILTERING                                                            0x1664
#define LW8597_SET_LWBEMAP_INTER_FACE_FILTERING_MODE                                                          2:1
#define LW8597_SET_LWBEMAP_INTER_FACE_FILTERING_MODE_USE_WRAP                                          0x00000000
#define LW8597_SET_LWBEMAP_INTER_FACE_FILTERING_MODE_OVERRIDE_WRAP                                     0x00000001
#define LW8597_SET_LWBEMAP_INTER_FACE_FILTERING_MODE_AUTO_SPAN_SEAM                                    0x00000002
#define LW8597_SET_LWBEMAP_INTER_FACE_FILTERING_MODE_AUTO_CROSS_SEAM                                   0x00000003

#define LW8597_SET_LINE_SMOOTH_PARAMETERS                                                                  0x1668
#define LW8597_SET_LINE_SMOOTH_PARAMETERS_FALLOFF                                                            31:0
#define LW8597_SET_LINE_SMOOTH_PARAMETERS_FALLOFF__1_00                                                0x00000000
#define LW8597_SET_LINE_SMOOTH_PARAMETERS_FALLOFF__1_33                                                0x00000001
#define LW8597_SET_LINE_SMOOTH_PARAMETERS_FALLOFF__1_60                                                0x00000002

#define LW8597_SET_LINE_STIPPLE                                                                            0x166c
#define LW8597_SET_LINE_STIPPLE_ENABLE                                                                       31:0
#define LW8597_SET_LINE_STIPPLE_ENABLE_FALSE                                                           0x00000000
#define LW8597_SET_LINE_STIPPLE_ENABLE_TRUE                                                            0x00000001

#define LW8597_SET_LINE_SMOOTH_EDGE_TABLE(i)                                                       (0x1670+(i)*4)
#define LW8597_SET_LINE_SMOOTH_EDGE_TABLE_V0                                                                  7:0
#define LW8597_SET_LINE_SMOOTH_EDGE_TABLE_V1                                                                 15:8
#define LW8597_SET_LINE_SMOOTH_EDGE_TABLE_V2                                                                23:16
#define LW8597_SET_LINE_SMOOTH_EDGE_TABLE_V3                                                                31:24

#define LW8597_SET_LINE_STIPPLE_PARAMETERS                                                                 0x1680
#define LW8597_SET_LINE_STIPPLE_PARAMETERS_FACTOR                                                             7:0
#define LW8597_SET_LINE_STIPPLE_PARAMETERS_PATTERN                                                           23:8

#define LW8597_SET_PROVOKING_VERTEX                                                                        0x1684
#define LW8597_SET_PROVOKING_VERTEX_V                                                                         0:0
#define LW8597_SET_PROVOKING_VERTEX_V_FIRST                                                            0x00000000
#define LW8597_SET_PROVOKING_VERTEX_V_LAST                                                             0x00000001

#define LW8597_SET_TWO_SIDED_LIGHT                                                                         0x1688
#define LW8597_SET_TWO_SIDED_LIGHT_ENABLE                                                                     0:0
#define LW8597_SET_TWO_SIDED_LIGHT_ENABLE_FALSE                                                        0x00000000
#define LW8597_SET_TWO_SIDED_LIGHT_ENABLE_TRUE                                                         0x00000001

#define LW8597_SET_POLYGON_STIPPLE                                                                         0x168c
#define LW8597_SET_POLYGON_STIPPLE_ENABLE                                                                    31:0
#define LW8597_SET_POLYGON_STIPPLE_ENABLE_FALSE                                                        0x00000000
#define LW8597_SET_POLYGON_STIPPLE_ENABLE_TRUE                                                         0x00000001

#define LW8597_SET_SHADER_CONTROL                                                                          0x1690
#define LW8597_SET_SHADER_CONTROL_DEFAULT_PARTIAL                                                             0:0
#define LW8597_SET_SHADER_CONTROL_DEFAULT_PARTIAL_ZERO                                                 0x00000000
#define LW8597_SET_SHADER_CONTROL_DEFAULT_PARTIAL_INFINITY                                             0x00000001
#define LW8597_SET_SHADER_CONTROL_ZERO_TIMES_ANYTHING_IS_ZERO                                               16:16
#define LW8597_SET_SHADER_CONTROL_ZERO_TIMES_ANYTHING_IS_ZERO_FALSE                                    0x00000000
#define LW8597_SET_SHADER_CONTROL_ZERO_TIMES_ANYTHING_IS_ZERO_TRUE                                     0x00000001

#define LW8597_BIND_CONSTANT_BUFFER                                                                        0x1694
#define LW8597_BIND_CONSTANT_BUFFER_VALID                                                                     3:0
#define LW8597_BIND_CONSTANT_BUFFER_VALID_FALSE                                                        0x00000000
#define LW8597_BIND_CONSTANT_BUFFER_VALID_TRUE                                                         0x00000001
#define LW8597_BIND_CONSTANT_BUFFER_SHADER_TYPE                                                               7:4
#define LW8597_BIND_CONSTANT_BUFFER_SHADER_TYPE_VERTEX                                                 0x00000000
#define LW8597_BIND_CONSTANT_BUFFER_SHADER_TYPE_TESSELLATOR                                            0x00000001
#define LW8597_BIND_CONSTANT_BUFFER_SHADER_TYPE_GEOMETRY                                               0x00000002
#define LW8597_BIND_CONSTANT_BUFFER_SHADER_TYPE_PIXEL                                                  0x00000003
#define LW8597_BIND_CONSTANT_BUFFER_SHADER_SLOT                                                              11:8
#define LW8597_BIND_CONSTANT_BUFFER_TABLE_ENTRY                                                             19:12

#define LW8597_SET_SHADER_ADDRESS_REGISTER                                                                 0x1698
#define LW8597_SET_SHADER_ADDRESS_REGISTER_VS_OVERFLOW                                                        0:0
#define LW8597_SET_SHADER_ADDRESS_REGISTER_VS_OVERFLOW_WRAP                                            0x00000000
#define LW8597_SET_SHADER_ADDRESS_REGISTER_VS_OVERFLOW_STICKY                                          0x00000001
#define LW8597_SET_SHADER_ADDRESS_REGISTER_GS_OVERFLOW                                                        4:4
#define LW8597_SET_SHADER_ADDRESS_REGISTER_GS_OVERFLOW_WRAP                                            0x00000000
#define LW8597_SET_SHADER_ADDRESS_REGISTER_GS_OVERFLOW_STICKY                                          0x00000001
#define LW8597_SET_SHADER_ADDRESS_REGISTER_PS_OVERFLOW                                                        8:8
#define LW8597_SET_SHADER_ADDRESS_REGISTER_PS_OVERFLOW_WRAP                                            0x00000000
#define LW8597_SET_SHADER_ADDRESS_REGISTER_PS_OVERFLOW_STICKY                                          0x00000001

#define LW8597_SET_HYBRID_ANTI_ALIAS_CONTROL                                                               0x169c
#define LW8597_SET_HYBRID_ANTI_ALIAS_CONTROL_PASSES                                                           3:0
#define LW8597_SET_HYBRID_ANTI_ALIAS_CONTROL_CENTROID                                                         4:4
#define LW8597_SET_HYBRID_ANTI_ALIAS_CONTROL_CENTROID_PER_FRAGMENT                                     0x00000000
#define LW8597_SET_HYBRID_ANTI_ALIAS_CONTROL_CENTROID_PER_PASS                                         0x00000001

#define LW8597_SET_ALPHA_TO_COVERAGE_DITHER_CONTROL                                                        0x16a0
#define LW8597_SET_ALPHA_TO_COVERAGE_DITHER_CONTROL_DITHER_FOOTPRINT                                          3:0
#define LW8597_SET_ALPHA_TO_COVERAGE_DITHER_CONTROL_DITHER_FOOTPRINT_PIXELS_1X1                        0x00000000
#define LW8597_SET_ALPHA_TO_COVERAGE_DITHER_CONTROL_DITHER_FOOTPRINT_PIXELS_2X2                        0x00000001

#define LW8597_SET_SHADER_NAN_SATURATION                                                                   0x16a8
#define LW8597_SET_SHADER_NAN_SATURATION_SATURATION                                                           0:0
#define LW8597_SET_SHADER_NAN_SATURATION_SATURATION_NAN_RETURNS_NAN                                    0x00000000
#define LW8597_SET_SHADER_NAN_SATURATION_SATURATION_NAN_RETURNS_ZERO                                   0x00000001

#define LW8597_SET_VS_OUTPUT_COUNT                                                                         0x16ac
#define LW8597_SET_VS_OUTPUT_COUNT_V                                                                          7:0

#define LW8597_SET_VS_REGISTER_COUNT                                                                       0x16b0
#define LW8597_SET_VS_REGISTER_COUNT_V                                                                        7:0

#define LW8597_SET_ALPHA_TO_COVERAGE_OVERRIDE                                                              0x16b4
#define LW8597_SET_ALPHA_TO_COVERAGE_OVERRIDE_QUALIFY_BY_ANTI_ALIAS_ENABLE                                    0:0
#define LW8597_SET_ALPHA_TO_COVERAGE_OVERRIDE_QUALIFY_BY_ANTI_ALIAS_ENABLE_DISABLE                     0x00000000
#define LW8597_SET_ALPHA_TO_COVERAGE_OVERRIDE_QUALIFY_BY_ANTI_ALIAS_ENABLE_ENABLE                      0x00000001
#define LW8597_SET_ALPHA_TO_COVERAGE_OVERRIDE_QUALIFY_BY_PS_SAMPLE_MASK_OUTPUT                                1:1
#define LW8597_SET_ALPHA_TO_COVERAGE_OVERRIDE_QUALIFY_BY_PS_SAMPLE_MASK_OUTPUT_DISABLE                 0x00000000
#define LW8597_SET_ALPHA_TO_COVERAGE_OVERRIDE_QUALIFY_BY_PS_SAMPLE_MASK_OUTPUT_ENABLE                  0x00000001

#define LW8597_SET_VS_OUTBUF_COUNT                                                                         0x16b8
#define LW8597_SET_VS_OUTBUF_COUNT_V                                                                          7:0

#define LW8597_SET_VS_OUTPUT_POSITION                                                                      0x16bc
#define LW8597_SET_VS_OUTPUT_POSITION_COMP00                                                                  7:0
#define LW8597_SET_VS_OUTPUT_POSITION_COMP01                                                                 15:8
#define LW8597_SET_VS_OUTPUT_POSITION_COMP02                                                                23:16
#define LW8597_SET_VS_OUTPUT_POSITION_COMP03                                                                31:24

#define LW8597_SET_VS_OUTPUT_REORDER(i)                                                            (0x16c0+(i)*4)
#define LW8597_SET_VS_OUTPUT_REORDER_COMP00                                                                   7:0
#define LW8597_SET_VS_OUTPUT_REORDER_COMP01                                                                  15:8
#define LW8597_SET_VS_OUTPUT_REORDER_COMP02                                                                 23:16
#define LW8597_SET_VS_OUTPUT_REORDER_COMP03                                                                 31:24

#define LW8597_SET_POLYGON_STIPPLE_PATTERN(i)                                                      (0x1700+(i)*4)
#define LW8597_SET_POLYGON_STIPPLE_PATTERN_V                                                                 31:0

#define LW8597_SET_STREAMING_START_OFFSET(i)                                                       (0x1780+(i)*4)
#define LW8597_SET_STREAMING_START_OFFSET_BYTE_COUNT                                                         31:0

#define LW8597_SET_GS_ENABLE                                                                               0x1798
#define LW8597_SET_GS_ENABLE_V                                                                               31:0
#define LW8597_SET_GS_ENABLE_V_FALSE                                                                   0x00000000
#define LW8597_SET_GS_ENABLE_V_TRUE                                                                    0x00000001

#define LW8597_SET_GS_REGISTER_COUNT                                                                       0x17a0
#define LW8597_SET_GS_REGISTER_COUNT_V                                                                        7:0

#define LW8597_SET_GS_OUTBUF_COUNT                                                                         0x17a8
#define LW8597_SET_GS_OUTBUF_COUNT_V                                                                          7:0

#define LW8597_SET_GS_OUTPUT_COUNT                                                                         0x17ac
#define LW8597_SET_GS_OUTPUT_COUNT_V                                                                          7:0

#define LW8597_SET_GS_OUTPUT_TOPOLOGY                                                                      0x17b0
#define LW8597_SET_GS_OUTPUT_TOPOLOGY_V                                                                       1:0
#define LW8597_SET_GS_OUTPUT_TOPOLOGY_V_POINTS                                                         0x00000001
#define LW8597_SET_GS_OUTPUT_TOPOLOGY_V_LINESTRIP                                                      0x00000002
#define LW8597_SET_GS_OUTPUT_TOPOLOGY_V_TRIANGLESTRIP                                                  0x00000003

#define LW8597_SET_PIPELINE_OUTPUT                                                                         0x17b4
#define LW8597_SET_PIPELINE_OUTPUT_ENABLE                                                                    31:0
#define LW8597_SET_PIPELINE_OUTPUT_ENABLE_FALSE                                                        0x00000000
#define LW8597_SET_PIPELINE_OUTPUT_ENABLE_TRUE                                                         0x00000001

#define LW8597_SET_STREAMING_OUTPUT                                                                        0x17b8
#define LW8597_SET_STREAMING_OUTPUT_ENABLE                                                                   31:0
#define LW8597_SET_STREAMING_OUTPUT_ENABLE_FALSE                                                       0x00000000
#define LW8597_SET_STREAMING_OUTPUT_ENABLE_TRUE                                                        0x00000001

#define LW8597_SET_GS_OUTPUT_POSITION                                                                      0x17fc
#define LW8597_SET_GS_OUTPUT_POSITION_COMP00                                                                  7:0
#define LW8597_SET_GS_OUTPUT_POSITION_COMP01                                                                 15:8
#define LW8597_SET_GS_OUTPUT_POSITION_COMP02                                                                23:16
#define LW8597_SET_GS_OUTPUT_POSITION_COMP03                                                                31:24

#define LW8597_SET_GS_OUTPUT_REORDER(i)                                                            (0x1800+(i)*4)
#define LW8597_SET_GS_OUTPUT_REORDER_COMP00                                                                   7:0
#define LW8597_SET_GS_OUTPUT_REORDER_COMP01                                                                  15:8
#define LW8597_SET_GS_OUTPUT_REORDER_COMP02                                                                 23:16
#define LW8597_SET_GS_OUTPUT_REORDER_COMP03                                                                 31:24

#define LW8597_SET_DEPTH_BIAS_CLAMP                                                                        0x187c
#define LW8597_SET_DEPTH_BIAS_CLAMP_V                                                                        31:0

#define LW8597_SET_VERTEX_STREAM_INSTANCE_A(i)                                                     (0x1880+(i)*4)
#define LW8597_SET_VERTEX_STREAM_INSTANCE_A_IS_INSTANCED                                                      0:0
#define LW8597_SET_VERTEX_STREAM_INSTANCE_A_IS_INSTANCED_FALSE                                         0x00000000
#define LW8597_SET_VERTEX_STREAM_INSTANCE_A_IS_INSTANCED_TRUE                                          0x00000001

#define LW8597_SET_VERTEX_STREAM_INSTANCE_B(i)                                                     (0x18c0+(i)*4)
#define LW8597_SET_VERTEX_STREAM_INSTANCE_B_IS_INSTANCED                                                      0:0
#define LW8597_SET_VERTEX_STREAM_INSTANCE_B_IS_INSTANCED_FALSE                                         0x00000000
#define LW8597_SET_VERTEX_STREAM_INSTANCE_B_IS_INSTANCED_TRUE                                          0x00000001

#define LW8597_SET_ATTRIBUTE_VIEWPORT_INDEX                                                                0x1900
#define LW8597_SET_ATTRIBUTE_VIEWPORT_INDEX_ENABLE                                                            0:0
#define LW8597_SET_ATTRIBUTE_VIEWPORT_INDEX_ENABLE_FALSE                                               0x00000000
#define LW8597_SET_ATTRIBUTE_VIEWPORT_INDEX_ENABLE_TRUE                                                0x00000001

#define LW8597_SET_ATTRIBUTE_COLOR                                                                         0x1904
#define LW8597_SET_ATTRIBUTE_COLOR_FRONT_SLOT                                                                 7:0
#define LW8597_SET_ATTRIBUTE_COLOR_BACK_SLOT                                                                 15:8
#define LW8597_SET_ATTRIBUTE_COLOR_COUNT                                                                    23:16
#define LW8597_SET_ATTRIBUTE_COLOR_CLAMP_ENABLE                                                             24:24
#define LW8597_SET_ATTRIBUTE_COLOR_CLAMP_ENABLE_FALSE                                                  0x00000000
#define LW8597_SET_ATTRIBUTE_COLOR_CLAMP_ENABLE_TRUE                                                   0x00000001

#define LW8597_SET_ATTRIBUTE_USER_CLIP                                                                     0x1908
#define LW8597_SET_ATTRIBUTE_USER_CLIP_SLOT                                                                   7:0
#define LW8597_SET_ATTRIBUTE_USER_CLIP_COUNT                                                                 11:8

#define LW8597_SET_ATTRIBUTE_RT_ARRAY_INDEX                                                                0x190c
#define LW8597_SET_ATTRIBUTE_RT_ARRAY_INDEX_REGISTER                                                          7:0

#define LW8597_SET_ATTRIBUTE_POINT_SIZE                                                                    0x1910
#define LW8597_SET_ATTRIBUTE_POINT_SIZE_ENABLE                                                                0:0
#define LW8597_SET_ATTRIBUTE_POINT_SIZE_ENABLE_FALSE                                                   0x00000000
#define LW8597_SET_ATTRIBUTE_POINT_SIZE_ENABLE_TRUE                                                    0x00000001
#define LW8597_SET_ATTRIBUTE_POINT_SIZE_SLOT                                                                 11:4

#define LW8597_SET_ATTRIBUTE_PRIMITIVE_ID                                                                  0x1914
#define LW8597_SET_ATTRIBUTE_PRIMITIVE_ID_SLOT                                                                7:0

#define LW8597_OGL_SET_LWLL                                                                                0x1918
#define LW8597_OGL_SET_LWLL_ENABLE                                                                           31:0
#define LW8597_OGL_SET_LWLL_ENABLE_FALSE                                                               0x00000000
#define LW8597_OGL_SET_LWLL_ENABLE_TRUE                                                                0x00000001

#define LW8597_OGL_SET_FRONT_FACE                                                                          0x191c
#define LW8597_OGL_SET_FRONT_FACE_V                                                                          31:0
#define LW8597_OGL_SET_FRONT_FACE_V_CW                                                                 0x00000900
#define LW8597_OGL_SET_FRONT_FACE_V_CCW                                                                0x00000901

#define LW8597_OGL_SET_LWLL_FACE                                                                           0x1920
#define LW8597_OGL_SET_LWLL_FACE_V                                                                           31:0
#define LW8597_OGL_SET_LWLL_FACE_V_FRONT                                                               0x00000404
#define LW8597_OGL_SET_LWLL_FACE_V_BACK                                                                0x00000405
#define LW8597_OGL_SET_LWLL_FACE_V_FRONT_AND_BACK                                                      0x00000408

#define LW8597_SET_VIEWPORT_PIXEL                                                                          0x1924
#define LW8597_SET_VIEWPORT_PIXEL_CENTER                                                                      0:0
#define LW8597_SET_VIEWPORT_PIXEL_CENTER_AT_HALF_INTEGERS                                              0x00000000
#define LW8597_SET_VIEWPORT_PIXEL_CENTER_AT_INTEGERS                                                   0x00000001

#define LW8597_SET_PS_SAMPLE_MASK_OUTPUT                                                                   0x1928
#define LW8597_SET_PS_SAMPLE_MASK_OUTPUT_ENABLE                                                               0:0
#define LW8597_SET_PS_SAMPLE_MASK_OUTPUT_ENABLE_FALSE                                                  0x00000000
#define LW8597_SET_PS_SAMPLE_MASK_OUTPUT_ENABLE_TRUE                                                   0x00000001
#define LW8597_SET_PS_SAMPLE_MASK_OUTPUT_QUALIFY_BY_ANTI_ALIAS_ENABLE                                         1:1
#define LW8597_SET_PS_SAMPLE_MASK_OUTPUT_QUALIFY_BY_ANTI_ALIAS_ENABLE_DISABLE                          0x00000000
#define LW8597_SET_PS_SAMPLE_MASK_OUTPUT_QUALIFY_BY_ANTI_ALIAS_ENABLE_ENABLE                           0x00000001

#define LW8597_SET_VIEWPORT_SCALE_OFFSET                                                                   0x192c
#define LW8597_SET_VIEWPORT_SCALE_OFFSET_ENABLE                                                               0:0
#define LW8597_SET_VIEWPORT_SCALE_OFFSET_ENABLE_FALSE                                                  0x00000000
#define LW8597_SET_VIEWPORT_SCALE_OFFSET_ENABLE_TRUE                                                   0x00000001

#define LW8597_SET_VIEWPORT_CLIP_CONTROL                                                                   0x193c
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_MIN_Z_ZERO_MAX_Z_ONE                                                 0:0
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_MIN_Z_ZERO_MAX_Z_ONE_FALSE                                    0x00000000
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_MIN_Z_ZERO_MAX_Z_ONE_TRUE                                     0x00000001
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_PIXEL_MIN_Z                                                          3:3
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_PIXEL_MIN_Z_CLIP                                              0x00000000
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_PIXEL_MIN_Z_CLAMP                                             0x00000001
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_PIXEL_MAX_Z                                                          4:4
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_PIXEL_MAX_Z_CLIP                                              0x00000000
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_PIXEL_MAX_Z_CLAMP                                             0x00000001
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_GUARDBAND                                                   7:7
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_GUARDBAND_SCALE_256                                  0x00000000
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_GUARDBAND_SCALE_1                                    0x00000001
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_LINE_POINT_LWLL_GUARDBAND                                          10:10
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_LINE_POINT_LWLL_GUARDBAND_SCALE_256                           0x00000000
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_LINE_POINT_LWLL_GUARDBAND_SCALE_1                             0x00000001
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP                                                      13:11
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP                                      0x00000000
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_PASSTHRU                                        0x00000001
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XY_CLIP                                 0x00000002
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XYZ_CLIP                                0x00000003
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP_NO_Z_LWLL                            0x00000004
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_Z_CLIP                                  0x00000005
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP                                      0x00000000
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_PASSTHRU                                        0x00000001
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XY_CLIP                                 0x00000002
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XYZ_CLIP                                0x00000003
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP_NO_Z_LWLL                            0x00000004
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_Z_CLIP                                  0x00000005
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP                                      0x00000000
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_PASSTHRU                                        0x00000001
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XY_CLIP                                 0x00000002
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XYZ_CLIP                                0x00000003
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP_NO_Z_LWLL                            0x00000004
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_Z_CLIP                                  0x00000005
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP                                      0x00000000
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_PASSTHRU                                        0x00000001
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XY_CLIP                                 0x00000002
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XYZ_CLIP                                0x00000003
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP_NO_Z_LWLL                            0x00000004
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_Z_CLIP                                  0x00000005
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP                                      0x00000000
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_PASSTHRU                                        0x00000001
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XY_CLIP                                 0x00000002
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XYZ_CLIP                                0x00000003
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP_NO_Z_LWLL                            0x00000004
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_Z_CLIP                                  0x00000005
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP                                      0x00000000
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_PASSTHRU                                        0x00000001
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XY_CLIP                                 0x00000002
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XYZ_CLIP                                0x00000003
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP_NO_Z_LWLL                            0x00000004
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_Z_CLIP                                  0x00000005
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP                                      0x00000000
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_PASSTHRU                                        0x00000001
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XY_CLIP                                 0x00000002
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XYZ_CLIP                                0x00000003
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP_NO_Z_LWLL                            0x00000004
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_Z_CLIP                                  0x00000005
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP                                      0x00000000
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_PASSTHRU                                        0x00000001
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XY_CLIP                                 0x00000002
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XYZ_CLIP                                0x00000003
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP_NO_Z_LWLL                            0x00000004
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_Z_CLIP                                  0x00000005
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP                                      0x00000000
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_PASSTHRU                                        0x00000001
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XY_CLIP                                 0x00000002
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XYZ_CLIP                                0x00000003
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP_NO_Z_LWLL                            0x00000004
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_Z_CLIP                                  0x00000005
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP                                      0x00000000
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_PASSTHRU                                        0x00000001
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XY_CLIP                                 0x00000002
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XYZ_CLIP                                0x00000003
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP_NO_Z_LWLL                            0x00000004
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_Z_CLIP                                  0x00000005
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_GUARDBAND_Z                                                 2:1
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_GUARDBAND_Z_SAME_AS_XY_GUARDBAND                     0x00000000
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_GUARDBAND_Z_SCALE_256                                0x00000001
#define LW8597_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_GUARDBAND_Z_SCALE_1                                  0x00000002

#define LW8597_SET_USER_CLIP_OP                                                                            0x1940
#define LW8597_SET_USER_CLIP_OP_PLANE0                                                                        0:0
#define LW8597_SET_USER_CLIP_OP_PLANE0_CLIP                                                            0x00000000
#define LW8597_SET_USER_CLIP_OP_PLANE0_LWLL                                                            0x00000001
#define LW8597_SET_USER_CLIP_OP_PLANE1                                                                        4:4
#define LW8597_SET_USER_CLIP_OP_PLANE1_CLIP                                                            0x00000000
#define LW8597_SET_USER_CLIP_OP_PLANE1_LWLL                                                            0x00000001
#define LW8597_SET_USER_CLIP_OP_PLANE2                                                                        8:8
#define LW8597_SET_USER_CLIP_OP_PLANE2_CLIP                                                            0x00000000
#define LW8597_SET_USER_CLIP_OP_PLANE2_LWLL                                                            0x00000001
#define LW8597_SET_USER_CLIP_OP_PLANE3                                                                      12:12
#define LW8597_SET_USER_CLIP_OP_PLANE3_CLIP                                                            0x00000000
#define LW8597_SET_USER_CLIP_OP_PLANE3_LWLL                                                            0x00000001
#define LW8597_SET_USER_CLIP_OP_PLANE4                                                                      16:16
#define LW8597_SET_USER_CLIP_OP_PLANE4_CLIP                                                            0x00000000
#define LW8597_SET_USER_CLIP_OP_PLANE4_LWLL                                                            0x00000001
#define LW8597_SET_USER_CLIP_OP_PLANE5                                                                      20:20
#define LW8597_SET_USER_CLIP_OP_PLANE5_CLIP                                                            0x00000000
#define LW8597_SET_USER_CLIP_OP_PLANE5_LWLL                                                            0x00000001
#define LW8597_SET_USER_CLIP_OP_PLANE6                                                                      24:24
#define LW8597_SET_USER_CLIP_OP_PLANE6_CLIP                                                            0x00000000
#define LW8597_SET_USER_CLIP_OP_PLANE6_LWLL                                                            0x00000001
#define LW8597_SET_USER_CLIP_OP_PLANE7                                                                      28:28
#define LW8597_SET_USER_CLIP_OP_PLANE7_CLIP                                                            0x00000000
#define LW8597_SET_USER_CLIP_OP_PLANE7_LWLL                                                            0x00000001

#define LW8597_DRAW_ZERO_INDEX                                                                             0x1944
#define LW8597_DRAW_ZERO_INDEX_COUNT                                                                         31:0

#define LW8597_SET_FAST_POLYMODE                                                                           0x1948
#define LW8597_SET_FAST_POLYMODE_ENABLE                                                                       0:0
#define LW8597_SET_FAST_POLYMODE_ENABLE_FALSE                                                          0x00000000
#define LW8597_SET_FAST_POLYMODE_ENABLE_TRUE                                                           0x00000001

#define LW8597_SET_WINDOW_CLIP_ENABLE                                                                      0x194c
#define LW8597_SET_WINDOW_CLIP_ENABLE_V                                                                       0:0
#define LW8597_SET_WINDOW_CLIP_ENABLE_V_FALSE                                                          0x00000000
#define LW8597_SET_WINDOW_CLIP_ENABLE_V_TRUE                                                           0x00000001

#define LW8597_SET_WINDOW_CLIP_TYPE                                                                        0x1950
#define LW8597_SET_WINDOW_CLIP_TYPE_V                                                                         1:0
#define LW8597_SET_WINDOW_CLIP_TYPE_V_INCLUSIVE                                                        0x00000000
#define LW8597_SET_WINDOW_CLIP_TYPE_V_EXCLUSIVE                                                        0x00000001
#define LW8597_SET_WINDOW_CLIP_TYPE_V_CLIPALL                                                          0x00000002

#define LW8597_CLEAR_ZLWLL_SURFACE                                                                         0x1954
#define LW8597_CLEAR_ZLWLL_SURFACE_Z_ENABLE                                                                   0:0
#define LW8597_CLEAR_ZLWLL_SURFACE_Z_ENABLE_FALSE                                                      0x00000000
#define LW8597_CLEAR_ZLWLL_SURFACE_Z_ENABLE_TRUE                                                       0x00000001
#define LW8597_CLEAR_ZLWLL_SURFACE_STENCIL_ENABLE                                                             4:4
#define LW8597_CLEAR_ZLWLL_SURFACE_STENCIL_ENABLE_FALSE                                                0x00000000
#define LW8597_CLEAR_ZLWLL_SURFACE_STENCIL_ENABLE_TRUE                                                 0x00000001

#define LW8597_ILWALIDATE_ZLWLL                                                                            0x1958
#define LW8597_ILWALIDATE_ZLWLL_V                                                                            31:0
#define LW8597_ILWALIDATE_ZLWLL_V_ILWALIDATE                                                           0x00000000

#define LW8597_ILWALIDATE_ALL_ZLWLL_REGIONS                                                                0x195c
#define LW8597_ILWALIDATE_ALL_ZLWLL_REGIONS_V                                                                 0:0
#define LW8597_ILWALIDATE_ALL_ZLWLL_REGIONS_V_ILWALIDATE                                               0x00000000

#define LW8597_SET_XBAR_TICK_ARBITRATION                                                                   0x1960
#define LW8597_SET_XBAR_TICK_ARBITRATION_CBAR                                                                 3:0
#define LW8597_SET_XBAR_TICK_ARBITRATION_CBAR_NORMAL                                                   0x00000000
#define LW8597_SET_XBAR_TICK_ARBITRATION_CBAR_PRIORITIZE_TPCS_WITH_MIN_TICKS                           0x00000001
#define LW8597_SET_XBAR_TICK_ARBITRATION_ZBAR                                                                 7:4
#define LW8597_SET_XBAR_TICK_ARBITRATION_ZBAR_NORMAL                                                   0x00000000
#define LW8597_SET_XBAR_TICK_ARBITRATION_ZBAR_PRIORITIZE_TPCS_WITH_MIN_TICKS                           0x00000001

#define LW8597_SET_ZLWLL                                                                                   0x1968
#define LW8597_SET_ZLWLL_Z_ENABLE                                                                             0:0
#define LW8597_SET_ZLWLL_Z_ENABLE_FALSE                                                                0x00000000
#define LW8597_SET_ZLWLL_Z_ENABLE_TRUE                                                                 0x00000001
#define LW8597_SET_ZLWLL_STENCIL_ENABLE                                                                       4:4
#define LW8597_SET_ZLWLL_STENCIL_ENABLE_FALSE                                                          0x00000000
#define LW8597_SET_ZLWLL_STENCIL_ENABLE_TRUE                                                           0x00000001

#define LW8597_SET_ZLWLL_BOUNDS                                                                            0x196c
#define LW8597_SET_ZLWLL_BOUNDS_Z_MIN_UNBOUNDED_ENABLE                                                        0:0
#define LW8597_SET_ZLWLL_BOUNDS_Z_MIN_UNBOUNDED_ENABLE_FALSE                                           0x00000000
#define LW8597_SET_ZLWLL_BOUNDS_Z_MIN_UNBOUNDED_ENABLE_TRUE                                            0x00000001
#define LW8597_SET_ZLWLL_BOUNDS_Z_MAX_UNBOUNDED_ENABLE                                                        4:4
#define LW8597_SET_ZLWLL_BOUNDS_Z_MAX_UNBOUNDED_ENABLE_FALSE                                           0x00000000
#define LW8597_SET_ZLWLL_BOUNDS_Z_MAX_UNBOUNDED_ENABLE_TRUE                                            0x00000001

#define LW8597_SET_VISIBLE_EARLY_Z                                                                         0x1970
#define LW8597_SET_VISIBLE_EARLY_Z_ENABLE                                                                     0:0
#define LW8597_SET_VISIBLE_EARLY_Z_ENABLE_FALSE                                                        0x00000000
#define LW8597_SET_VISIBLE_EARLY_Z_ENABLE_TRUE                                                         0x00000001

#define LW8597_ZLWLL_SYNC                                                                                  0x1978
#define LW8597_ZLWLL_SYNC_V                                                                                  31:0

#define LW8597_SET_CLIP_ID_TEST                                                                            0x197c
#define LW8597_SET_CLIP_ID_TEST_ENABLE                                                                       31:0
#define LW8597_SET_CLIP_ID_TEST_ENABLE_FALSE                                                           0x00000000
#define LW8597_SET_CLIP_ID_TEST_ENABLE_TRUE                                                            0x00000001

#define LW8597_SET_SURFACE_CLIP_ID_WIDTH                                                                   0x1980
#define LW8597_SET_SURFACE_CLIP_ID_WIDTH_V                                                                   31:0

#define LW8597_SET_CLIP_ID                                                                                 0x1984
#define LW8597_SET_CLIP_ID_V                                                                                 31:0

#define LW8597_SET_PS_INPUT                                                                                0x1988
#define LW8597_SET_PS_INPUT_COUNT                                                                             7:0
#define LW8597_SET_PS_INPUT_START                                                                            15:8
#define LW8597_SET_PS_INPUT_INTERPOLATED_COUNT                                                              23:16
#define LW8597_SET_PS_INPUT_X_ENABLE                                                                        24:24
#define LW8597_SET_PS_INPUT_X_ENABLE_FALSE                                                             0x00000000
#define LW8597_SET_PS_INPUT_X_ENABLE_TRUE                                                              0x00000001
#define LW8597_SET_PS_INPUT_Y_ENABLE                                                                        25:25
#define LW8597_SET_PS_INPUT_Y_ENABLE_FALSE                                                             0x00000000
#define LW8597_SET_PS_INPUT_Y_ENABLE_TRUE                                                              0x00000001
#define LW8597_SET_PS_INPUT_Z_ENABLE                                                                        26:26
#define LW8597_SET_PS_INPUT_Z_ENABLE_FALSE                                                             0x00000000
#define LW8597_SET_PS_INPUT_Z_ENABLE_TRUE                                                              0x00000001
#define LW8597_SET_PS_INPUT_W_ENABLE                                                                        27:27
#define LW8597_SET_PS_INPUT_W_ENABLE_FALSE                                                             0x00000000
#define LW8597_SET_PS_INPUT_W_ENABLE_TRUE                                                              0x00000001
#define LW8597_SET_PS_INPUT_VIEWPORT_INDEX_ENABLE                                                           28:28
#define LW8597_SET_PS_INPUT_VIEWPORT_INDEX_ENABLE_FALSE                                                0x00000000
#define LW8597_SET_PS_INPUT_VIEWPORT_INDEX_ENABLE_TRUE                                                 0x00000001

#define LW8597_SET_PS_REGISTER_COUNT                                                                       0x198c
#define LW8597_SET_PS_REGISTER_COUNT_V                                                                        7:0

#define LW8597_SET_PS_REGISTER_ALLOCATION                                                                  0x19a0
#define LW8597_SET_PS_REGISTER_ALLOCATION_V                                                                  31:0
#define LW8597_SET_PS_REGISTER_ALLOCATION_V_THICK                                                      0x00000001
#define LW8597_SET_PS_REGISTER_ALLOCATION_V_THIN                                                       0x00000002

#define LW8597_SET_PS_CONTROL                                                                              0x19a8
#define LW8597_SET_PS_CONTROL_MRT_ENABLE                                                                      0:0
#define LW8597_SET_PS_CONTROL_MRT_ENABLE_FALSE                                                         0x00000000
#define LW8597_SET_PS_CONTROL_MRT_ENABLE_TRUE                                                          0x00000001
#define LW8597_SET_PS_CONTROL_DEPTH_REPLACE_ENABLE                                                            8:8
#define LW8597_SET_PS_CONTROL_DEPTH_REPLACE_ENABLE_FALSE                                               0x00000000
#define LW8597_SET_PS_CONTROL_DEPTH_REPLACE_ENABLE_TRUE                                                0x00000001
#define LW8597_SET_PS_CONTROL_KILLS_PIXELS                                                                  20:20
#define LW8597_SET_PS_CONTROL_KILLS_PIXELS_FALSE                                                       0x00000000
#define LW8597_SET_PS_CONTROL_KILLS_PIXELS_TRUE                                                        0x00000001

#define LW8597_SET_DEPTH_BOUNDS_TEST                                                                       0x19bc
#define LW8597_SET_DEPTH_BOUNDS_TEST_ENABLE                                                                  31:0
#define LW8597_SET_DEPTH_BOUNDS_TEST_ENABLE_FALSE                                                      0x00000000
#define LW8597_SET_DEPTH_BOUNDS_TEST_ENABLE_TRUE                                                       0x00000001

#define LW8597_SET_BLEND_FLOAT_OPTION                                                                      0x19c0
#define LW8597_SET_BLEND_FLOAT_OPTION_ZERO_TIMES_ANYTHING_IS_ZERO                                             0:0
#define LW8597_SET_BLEND_FLOAT_OPTION_ZERO_TIMES_ANYTHING_IS_ZERO_FALSE                                0x00000000
#define LW8597_SET_BLEND_FLOAT_OPTION_ZERO_TIMES_ANYTHING_IS_ZERO_TRUE                                 0x00000001

#define LW8597_SET_LOGIC_OP                                                                                0x19c4
#define LW8597_SET_LOGIC_OP_ENABLE                                                                           31:0
#define LW8597_SET_LOGIC_OP_ENABLE_FALSE                                                               0x00000000
#define LW8597_SET_LOGIC_OP_ENABLE_TRUE                                                                0x00000001

#define LW8597_SET_LOGIC_OP_FUNC                                                                           0x19c8
#define LW8597_SET_LOGIC_OP_FUNC_V                                                                           31:0
#define LW8597_SET_LOGIC_OP_FUNC_V_CLEAR                                                               0x00001500
#define LW8597_SET_LOGIC_OP_FUNC_V_AND                                                                 0x00001501
#define LW8597_SET_LOGIC_OP_FUNC_V_AND_REVERSE                                                         0x00001502
#define LW8597_SET_LOGIC_OP_FUNC_V_COPY                                                                0x00001503
#define LW8597_SET_LOGIC_OP_FUNC_V_AND_ILWERTED                                                        0x00001504
#define LW8597_SET_LOGIC_OP_FUNC_V_NOOP                                                                0x00001505
#define LW8597_SET_LOGIC_OP_FUNC_V_XOR                                                                 0x00001506
#define LW8597_SET_LOGIC_OP_FUNC_V_OR                                                                  0x00001507
#define LW8597_SET_LOGIC_OP_FUNC_V_NOR                                                                 0x00001508
#define LW8597_SET_LOGIC_OP_FUNC_V_EQUIV                                                               0x00001509
#define LW8597_SET_LOGIC_OP_FUNC_V_ILWERT                                                              0x0000150A
#define LW8597_SET_LOGIC_OP_FUNC_V_OR_REVERSE                                                          0x0000150B
#define LW8597_SET_LOGIC_OP_FUNC_V_COPY_ILWERTED                                                       0x0000150C
#define LW8597_SET_LOGIC_OP_FUNC_V_OR_ILWERTED                                                         0x0000150D
#define LW8597_SET_LOGIC_OP_FUNC_V_NAND                                                                0x0000150E
#define LW8597_SET_LOGIC_OP_FUNC_V_SET                                                                 0x0000150F

#define LW8597_SET_Z_COMPRESSION                                                                           0x19cc
#define LW8597_SET_Z_COMPRESSION_ENABLE                                                                      31:0
#define LW8597_SET_Z_COMPRESSION_ENABLE_FALSE                                                          0x00000000
#define LW8597_SET_Z_COMPRESSION_ENABLE_TRUE                                                           0x00000001

#define LW8597_CLEAR_SURFACE                                                                               0x19d0
#define LW8597_CLEAR_SURFACE_Z_ENABLE                                                                         0:0
#define LW8597_CLEAR_SURFACE_Z_ENABLE_FALSE                                                            0x00000000
#define LW8597_CLEAR_SURFACE_Z_ENABLE_TRUE                                                             0x00000001
#define LW8597_CLEAR_SURFACE_STENCIL_ENABLE                                                                   1:1
#define LW8597_CLEAR_SURFACE_STENCIL_ENABLE_FALSE                                                      0x00000000
#define LW8597_CLEAR_SURFACE_STENCIL_ENABLE_TRUE                                                       0x00000001
#define LW8597_CLEAR_SURFACE_R_ENABLE                                                                         2:2
#define LW8597_CLEAR_SURFACE_R_ENABLE_FALSE                                                            0x00000000
#define LW8597_CLEAR_SURFACE_R_ENABLE_TRUE                                                             0x00000001
#define LW8597_CLEAR_SURFACE_G_ENABLE                                                                         3:3
#define LW8597_CLEAR_SURFACE_G_ENABLE_FALSE                                                            0x00000000
#define LW8597_CLEAR_SURFACE_G_ENABLE_TRUE                                                             0x00000001
#define LW8597_CLEAR_SURFACE_B_ENABLE                                                                         4:4
#define LW8597_CLEAR_SURFACE_B_ENABLE_FALSE                                                            0x00000000
#define LW8597_CLEAR_SURFACE_B_ENABLE_TRUE                                                             0x00000001
#define LW8597_CLEAR_SURFACE_A_ENABLE                                                                         5:5
#define LW8597_CLEAR_SURFACE_A_ENABLE_FALSE                                                            0x00000000
#define LW8597_CLEAR_SURFACE_A_ENABLE_TRUE                                                             0x00000001
#define LW8597_CLEAR_SURFACE_MRT_SELECT                                                                       9:6
#define LW8597_CLEAR_SURFACE_RT_ARRAY_INDEX                                                                 25:10

#define LW8597_CLEAR_CLIP_ID_SURFACE                                                                       0x19d4
#define LW8597_CLEAR_CLIP_ID_SURFACE_V                                                                       31:0

#define LW8597_SET_LINE_SNAP_GRID                                                                          0x19d8
#define LW8597_SET_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL                                                         7:0
#define LW8597_SET_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__4                                               0x00000002
#define LW8597_SET_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__8                                               0x00000003
#define LW8597_SET_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__16                                              0x00000004
#define LW8597_SET_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__32                                              0x00000005
#define LW8597_SET_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__64                                              0x00000006
#define LW8597_SET_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__256                                             0x00000008
#define LW8597_SET_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__1024                                            0x0000000A
#define LW8597_SET_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__4096                                            0x0000000C
#define LW8597_SET_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__16384                                           0x0000000E
#define LW8597_SET_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__65536                                           0x00000010

#define LW8597_SET_NON_LINE_SNAP_GRID                                                                      0x19dc
#define LW8597_SET_NON_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL                                                     7:0
#define LW8597_SET_NON_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__4                                           0x00000002
#define LW8597_SET_NON_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__8                                           0x00000003
#define LW8597_SET_NON_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__16                                          0x00000004
#define LW8597_SET_NON_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__32                                          0x00000005
#define LW8597_SET_NON_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__64                                          0x00000006
#define LW8597_SET_NON_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__256                                         0x00000008
#define LW8597_SET_NON_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__1024                                        0x0000000A
#define LW8597_SET_NON_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__4096                                        0x0000000C
#define LW8597_SET_NON_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__16384                                       0x0000000E
#define LW8597_SET_NON_LINE_SNAP_GRID_LOCATIONS_PER_PIXEL__65536                                       0x00000010

#define LW8597_SET_COLOR_COMPRESSION(i)                                                            (0x19e0+(i)*4)
#define LW8597_SET_COLOR_COMPRESSION_ENABLE                                                                  31:0
#define LW8597_SET_COLOR_COMPRESSION_ENABLE_FALSE                                                      0x00000000
#define LW8597_SET_COLOR_COMPRESSION_ENABLE_TRUE                                                       0x00000001

#define LW8597_SET_CT_WRITE(i)                                                                     (0x1a00+(i)*4)
#define LW8597_SET_CT_WRITE_R_ENABLE                                                                          0:0
#define LW8597_SET_CT_WRITE_R_ENABLE_FALSE                                                             0x00000000
#define LW8597_SET_CT_WRITE_R_ENABLE_TRUE                                                              0x00000001
#define LW8597_SET_CT_WRITE_G_ENABLE                                                                          4:4
#define LW8597_SET_CT_WRITE_G_ENABLE_FALSE                                                             0x00000000
#define LW8597_SET_CT_WRITE_G_ENABLE_TRUE                                                              0x00000001
#define LW8597_SET_CT_WRITE_B_ENABLE                                                                          8:8
#define LW8597_SET_CT_WRITE_B_ENABLE_FALSE                                                             0x00000000
#define LW8597_SET_CT_WRITE_B_ENABLE_TRUE                                                              0x00000001
#define LW8597_SET_CT_WRITE_A_ENABLE                                                                        12:12
#define LW8597_SET_CT_WRITE_A_ENABLE_FALSE                                                             0x00000000
#define LW8597_SET_CT_WRITE_A_ENABLE_TRUE                                                              0x00000001

#define LW8597_SET_Z_PLANE_PRECISION                                                                       0x1a20
#define LW8597_SET_Z_PLANE_PRECISION_MASK_A_AND_B_BIT_COUNT                                                   3:0
#define LW8597_SET_Z_PLANE_PRECISION_MASK_C_BIT_COUNT                                                         7:4

#define LW8597_TEST_FOR_QUADRO                                                                             0x1a24
#define LW8597_TEST_FOR_QUADRO_V                                                                             31:0

#define LW8597_SET_OCTVERTICES_PER_TPC                                                                     0x1a28
#define LW8597_SET_OCTVERTICES_PER_TPC_V                                                                      7:0
#define LW8597_SET_OCTVERTICES_PER_TPC_SWITCH_TPC_ON_FLUSH                                                    8:8
#define LW8597_SET_OCTVERTICES_PER_TPC_SWITCH_TPC_ON_FLUSH_FALSE                                       0x00000000
#define LW8597_SET_OCTVERTICES_PER_TPC_SWITCH_TPC_ON_FLUSH_TRUE                                        0x00000001

#define LW8597_PIPE_NOP                                                                                    0x1a2c
#define LW8597_PIPE_NOP_V                                                                                    31:0

#define LW8597_SET_SPARE00                                                                                 0x1a30
#define LW8597_SET_SPARE00_V                                                                                 31:0

#define LW8597_SET_SPARE01                                                                                 0x1a34
#define LW8597_SET_SPARE01_V                                                                                 31:0

#define LW8597_SET_SPARE02                                                                                 0x1a38
#define LW8597_SET_SPARE02_V                                                                                 31:0

#define LW8597_SET_SPARE03                                                                                 0x1a3c
#define LW8597_SET_SPARE03_V                                                                                 31:0

#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES(i,j)                                           (0x1a40+(i)*16+(j)*4)
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP00                                                           3:0
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP00_ANISO_1_TO_1                                       0x00000000
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP00_ANISO_2_TO_1                                       0x00000001
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP00_ANISO_4_TO_1                                       0x00000002
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP00_ANISO_6_TO_1                                       0x00000003
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP00_ANISO_8_TO_1                                       0x00000004
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP00_ANISO_10_TO_1                                      0x00000005
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP00_ANISO_12_TO_1                                      0x00000006
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP00_ANISO_16_TO_1                                      0x00000007
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP01                                                           7:4
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP01_ANISO_1_TO_1                                       0x00000000
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP01_ANISO_2_TO_1                                       0x00000001
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP01_ANISO_4_TO_1                                       0x00000002
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP01_ANISO_6_TO_1                                       0x00000003
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP01_ANISO_8_TO_1                                       0x00000004
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP01_ANISO_10_TO_1                                      0x00000005
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP01_ANISO_12_TO_1                                      0x00000006
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP01_ANISO_16_TO_1                                      0x00000007
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP02                                                          11:8
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP02_ANISO_1_TO_1                                       0x00000000
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP02_ANISO_2_TO_1                                       0x00000001
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP02_ANISO_4_TO_1                                       0x00000002
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP02_ANISO_6_TO_1                                       0x00000003
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP02_ANISO_8_TO_1                                       0x00000004
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP02_ANISO_10_TO_1                                      0x00000005
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP02_ANISO_12_TO_1                                      0x00000006
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP02_ANISO_16_TO_1                                      0x00000007
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP03                                                         15:12
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP03_ANISO_1_TO_1                                       0x00000000
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP03_ANISO_2_TO_1                                       0x00000001
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP03_ANISO_4_TO_1                                       0x00000002
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP03_ANISO_6_TO_1                                       0x00000003
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP03_ANISO_8_TO_1                                       0x00000004
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP03_ANISO_10_TO_1                                      0x00000005
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP03_ANISO_12_TO_1                                      0x00000006
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP03_ANISO_16_TO_1                                      0x00000007
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP04                                                         19:16
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP04_ANISO_1_TO_1                                       0x00000000
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP04_ANISO_2_TO_1                                       0x00000001
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP04_ANISO_4_TO_1                                       0x00000002
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP04_ANISO_6_TO_1                                       0x00000003
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP04_ANISO_8_TO_1                                       0x00000004
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP04_ANISO_10_TO_1                                      0x00000005
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP04_ANISO_12_TO_1                                      0x00000006
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP04_ANISO_16_TO_1                                      0x00000007
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP05                                                         23:20
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP05_ANISO_1_TO_1                                       0x00000000
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP05_ANISO_2_TO_1                                       0x00000001
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP05_ANISO_4_TO_1                                       0x00000002
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP05_ANISO_6_TO_1                                       0x00000003
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP05_ANISO_8_TO_1                                       0x00000004
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP05_ANISO_10_TO_1                                      0x00000005
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP05_ANISO_12_TO_1                                      0x00000006
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP05_ANISO_16_TO_1                                      0x00000007
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP06                                                         27:24
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP06_ANISO_1_TO_1                                       0x00000000
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP06_ANISO_2_TO_1                                       0x00000001
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP06_ANISO_4_TO_1                                       0x00000002
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP06_ANISO_6_TO_1                                       0x00000003
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP06_ANISO_8_TO_1                                       0x00000004
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP06_ANISO_10_TO_1                                      0x00000005
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP06_ANISO_12_TO_1                                      0x00000006
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP06_ANISO_16_TO_1                                      0x00000007
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP07                                                         31:28
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP07_ANISO_1_TO_1                                       0x00000000
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP07_ANISO_2_TO_1                                       0x00000001
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP07_ANISO_4_TO_1                                       0x00000002
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP07_ANISO_6_TO_1                                       0x00000003
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP07_ANISO_8_TO_1                                       0x00000004
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP07_ANISO_10_TO_1                                      0x00000005
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP07_ANISO_12_TO_1                                      0x00000006
#define LW8597_SET_ANISO_ANGLE_TABLE_ENTRIES_COMP07_ANISO_16_TO_1                                      0x00000007

#define LW8597_SET_STREAMING_A(j)                                                                 (0x1a80+(j)*16)
#define LW8597_SET_STREAMING_A_OFFSET_UPPER                                                                   7:0

#define LW8597_SET_STREAMING_B(j)                                                                 (0x1a84+(j)*16)
#define LW8597_SET_STREAMING_B_OFFSET_LOWER                                                                  31:0

#define LW8597_SET_STREAMING_C(j)                                                                 (0x1a88+(j)*16)
#define LW8597_SET_STREAMING_C_ATTRIBUTE_COUNT                                                                7:0

#define LW8597_SET_STREAMING_BUFFER_BYTES(j)                                                      (0x1a8c+(j)*16)
#define LW8597_SET_STREAMING_BUFFER_BYTES_COUNT                                                              31:0

#define LW8597_SET_VERTEX_ATTRIB(i)                                                                (0x1ac0+(i)*4)
#define LW8597_SET_VERTEX_ATTRIB_STREAM                                                                       3:0
#define LW8597_SET_VERTEX_ATTRIB_SOURCE                                                                       4:4
#define LW8597_SET_VERTEX_ATTRIB_SOURCE_ACTIVE                                                         0x00000000
#define LW8597_SET_VERTEX_ATTRIB_SOURCE_INACTIVE                                                       0x00000001
#define LW8597_SET_VERTEX_ATTRIB_OFFSET                                                                      18:5
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS                                                       24:19
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_R32_G32_B32_A32                                  0x00000001
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_R32_G32_B32                                      0x00000002
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_R16_G16_B16_A16                                  0x00000003
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_R32_G32                                          0x00000004
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_R16_G16_B16                                      0x00000005
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_A8B8G8R8                                         0x0000002F
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_R8_G8_B8_A8                                      0x0000000A
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_X8B8G8R8                                         0x00000033
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_A2B10G10R10                                      0x00000030
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_B10G11R11                                        0x00000031
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_R16_G16                                          0x0000000F
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_R32                                              0x00000012
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_R8_G8_B8                                         0x00000013
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_G8R8                                             0x00000032
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_R8_G8                                            0x00000018
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_R16                                              0x0000001B
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_R8                                               0x0000001D
#define LW8597_SET_VERTEX_ATTRIB_COMPONENT_BIT_WIDTHS_A8                                               0x00000034
#define LW8597_SET_VERTEX_ATTRIB_RGB_TYPE                                                                   27:25
#define LW8597_SET_VERTEX_ATTRIB_RGB_TYPE_UNUSED_ENUM_DO_NOT_USE_BECAUSE_IT_WILL_GO_AWAY               0x00000000
#define LW8597_SET_VERTEX_ATTRIB_RGB_TYPE_NUM_SNORM                                                    0x00000001
#define LW8597_SET_VERTEX_ATTRIB_RGB_TYPE_NUM_UNORM                                                    0x00000002
#define LW8597_SET_VERTEX_ATTRIB_RGB_TYPE_NUM_SINT                                                     0x00000003
#define LW8597_SET_VERTEX_ATTRIB_RGB_TYPE_NUM_UINT                                                     0x00000004
#define LW8597_SET_VERTEX_ATTRIB_RGB_TYPE_NUM_USCALED                                                  0x00000005
#define LW8597_SET_VERTEX_ATTRIB_RGB_TYPE_NUM_SSCALED                                                  0x00000006
#define LW8597_SET_VERTEX_ATTRIB_RGB_TYPE_NUM_FLOAT                                                    0x00000007
#define LW8597_SET_VERTEX_ATTRIB_A_TYPE                                                                     30:28
#define LW8597_SET_VERTEX_ATTRIB_A_TYPE_UNUSED_ENUM_DO_NOT_USE_BECAUSE_IT_WILL_GO_AWAY                 0x00000000
#define LW8597_SET_VERTEX_ATTRIB_A_TYPE_NUM_SNORM                                                      0x00000001
#define LW8597_SET_VERTEX_ATTRIB_A_TYPE_NUM_UNORM                                                      0x00000002
#define LW8597_SET_VERTEX_ATTRIB_A_TYPE_NUM_SINT                                                       0x00000003
#define LW8597_SET_VERTEX_ATTRIB_A_TYPE_NUM_UINT                                                       0x00000004
#define LW8597_SET_VERTEX_ATTRIB_A_TYPE_NUM_USCALED                                                    0x00000005
#define LW8597_SET_VERTEX_ATTRIB_A_TYPE_NUM_SSCALED                                                    0x00000006
#define LW8597_SET_VERTEX_ATTRIB_A_TYPE_NUM_FLOAT                                                      0x00000007
#define LW8597_SET_VERTEX_ATTRIB_SWAP_R_AND_B                                                               31:31
#define LW8597_SET_VERTEX_ATTRIB_SWAP_R_AND_B_FALSE                                                    0x00000000
#define LW8597_SET_VERTEX_ATTRIB_SWAP_R_AND_B_TRUE                                                     0x00000001

#define LW8597_SET_REPORT_SEMAPHORE_A                                                                      0x1b00
#define LW8597_SET_REPORT_SEMAPHORE_A_OFFSET_UPPER                                                            7:0

#define LW8597_SET_REPORT_SEMAPHORE_B                                                                      0x1b04
#define LW8597_SET_REPORT_SEMAPHORE_B_OFFSET_LOWER                                                           31:0

#define LW8597_SET_REPORT_SEMAPHORE_C                                                                      0x1b08
#define LW8597_SET_REPORT_SEMAPHORE_C_PAYLOAD                                                                31:0

#define LW8597_SET_REPORT_SEMAPHORE_D                                                                      0x1b0c
#define LW8597_SET_REPORT_SEMAPHORE_D_OPERATION                                                               1:0
#define LW8597_SET_REPORT_SEMAPHORE_D_OPERATION_RELEASE                                                0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_OPERATION_ACQUIRE                                                0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_OPERATION_REPORT_ONLY                                            0x00000002
#define LW8597_SET_REPORT_SEMAPHORE_D_RELEASE                                                                 4:4
#define LW8597_SET_REPORT_SEMAPHORE_D_RELEASE_AFTER_ALL_PRECEEDING_READS_COMPLETE                      0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_RELEASE_AFTER_ALL_PRECEEDING_WRITES_COMPLETE                     0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_ACQUIRE                                                                 8:8
#define LW8597_SET_REPORT_SEMAPHORE_D_ACQUIRE_BEFORE_ANY_FOLLOWING_WRITES_START                        0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_ACQUIRE_BEFORE_ANY_FOLLOWING_READS_START                         0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION                                                     15:12
#define LW8597_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_NONE                                           0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_DATA_ASSEMBLER                                 0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_VERTEX_SHADER                                  0x00000002
#define LW8597_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_GEOMETRY_SHADER                                0x00000006
#define LW8597_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_STREAMING_OUTPUT                               0x00000005
#define LW8597_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_VPC                                            0x00000004
#define LW8597_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_ZLWLL                                          0x00000007
#define LW8597_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_PIXEL_SHADER                                   0x0000000A
#define LW8597_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_DEPTH_TEST                                     0x0000000C
#define LW8597_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_ALL                                            0x0000000F
#define LW8597_SET_REPORT_SEMAPHORE_D_COMPARISON                                                            16:16
#define LW8597_SET_REPORT_SEMAPHORE_D_COMPARISON_EQ                                                    0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_COMPARISON_GE                                                    0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_AWAKEN_ENABLE                                                         20:20
#define LW8597_SET_REPORT_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                              0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                               0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT                                                                27:23
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_NONE                                                      0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_VERTICES_GENERATED                                     0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_PRIMITIVES_GENERATED                                   0x00000003
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_VS_ILWOCATIONS                                            0x00000005
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_ILWOCATIONS                                            0x00000007
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_PRIMITIVES_GENERATED                                   0x00000009
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_STATUS                                          0x00000004
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_SUCCEEDED                            0x0000000B
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED                               0x0000000D
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_ILWOCATIONS                                       0x0000000F
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_PRIMITIVES_GENERATED                              0x00000011
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS0                                              0x0000000A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS1                                              0x0000000C
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS2                                              0x0000000E
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_PS_ILWOCATIONS                                            0x00000013
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS3                                              0x00000010
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT                                           0x00000002
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT64                                         0x00000015
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_COLOR_TARGET                                   0x00000018
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_ZETA_TARGET                                    0x00000019
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_VERTICES_SUCCEEDED                              0x00000008
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED               0x00000006
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_BYTE_COUNT                                      0x0000001A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_NONE                                                      0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_VERTICES_GENERATED                                     0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_PRIMITIVES_GENERATED                                   0x00000003
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_VS_ILWOCATIONS                                            0x00000005
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_ILWOCATIONS                                            0x00000007
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_PRIMITIVES_GENERATED                                   0x00000009
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_STATUS                                          0x00000004
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_SUCCEEDED                            0x0000000B
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED                               0x0000000D
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_ILWOCATIONS                                       0x0000000F
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_PRIMITIVES_GENERATED                              0x00000011
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS0                                              0x0000000A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS1                                              0x0000000C
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS2                                              0x0000000E
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_PS_ILWOCATIONS                                            0x00000013
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS3                                              0x00000010
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT                                           0x00000002
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT64                                         0x00000015
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_COLOR_TARGET                                   0x00000018
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_ZETA_TARGET                                    0x00000019
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_VERTICES_SUCCEEDED                              0x00000008
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED               0x00000006
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_BYTE_COUNT                                      0x0000001A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_NONE                                                      0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_VERTICES_GENERATED                                     0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_PRIMITIVES_GENERATED                                   0x00000003
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_VS_ILWOCATIONS                                            0x00000005
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_ILWOCATIONS                                            0x00000007
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_PRIMITIVES_GENERATED                                   0x00000009
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_STATUS                                          0x00000004
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_SUCCEEDED                            0x0000000B
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED                               0x0000000D
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_ILWOCATIONS                                       0x0000000F
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_PRIMITIVES_GENERATED                              0x00000011
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS0                                              0x0000000A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS1                                              0x0000000C
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS2                                              0x0000000E
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_PS_ILWOCATIONS                                            0x00000013
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS3                                              0x00000010
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT                                           0x00000002
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT64                                         0x00000015
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_COLOR_TARGET                                   0x00000018
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_ZETA_TARGET                                    0x00000019
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_VERTICES_SUCCEEDED                              0x00000008
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED               0x00000006
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_BYTE_COUNT                                      0x0000001A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_NONE                                                      0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_VERTICES_GENERATED                                     0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_PRIMITIVES_GENERATED                                   0x00000003
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_VS_ILWOCATIONS                                            0x00000005
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_ILWOCATIONS                                            0x00000007
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_PRIMITIVES_GENERATED                                   0x00000009
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_STATUS                                          0x00000004
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_SUCCEEDED                            0x0000000B
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED                               0x0000000D
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_ILWOCATIONS                                       0x0000000F
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_PRIMITIVES_GENERATED                              0x00000011
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS0                                              0x0000000A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS1                                              0x0000000C
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS2                                              0x0000000E
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_PS_ILWOCATIONS                                            0x00000013
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS3                                              0x00000010
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT                                           0x00000002
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT64                                         0x00000015
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_COLOR_TARGET                                   0x00000018
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_ZETA_TARGET                                    0x00000019
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_VERTICES_SUCCEEDED                              0x00000008
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED               0x00000006
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_BYTE_COUNT                                      0x0000001A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_NONE                                                      0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_VERTICES_GENERATED                                     0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_PRIMITIVES_GENERATED                                   0x00000003
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_VS_ILWOCATIONS                                            0x00000005
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_ILWOCATIONS                                            0x00000007
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_PRIMITIVES_GENERATED                                   0x00000009
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_STATUS                                          0x00000004
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_SUCCEEDED                            0x0000000B
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED                               0x0000000D
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_ILWOCATIONS                                       0x0000000F
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_PRIMITIVES_GENERATED                              0x00000011
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS0                                              0x0000000A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS1                                              0x0000000C
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS2                                              0x0000000E
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_PS_ILWOCATIONS                                            0x00000013
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS3                                              0x00000010
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT                                           0x00000002
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT64                                         0x00000015
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_COLOR_TARGET                                   0x00000018
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_ZETA_TARGET                                    0x00000019
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_VERTICES_SUCCEEDED                              0x00000008
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED               0x00000006
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_BYTE_COUNT                                      0x0000001A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_NONE                                                      0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_VERTICES_GENERATED                                     0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_PRIMITIVES_GENERATED                                   0x00000003
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_VS_ILWOCATIONS                                            0x00000005
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_ILWOCATIONS                                            0x00000007
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_PRIMITIVES_GENERATED                                   0x00000009
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_STATUS                                          0x00000004
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_SUCCEEDED                            0x0000000B
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED                               0x0000000D
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_ILWOCATIONS                                       0x0000000F
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_PRIMITIVES_GENERATED                              0x00000011
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS0                                              0x0000000A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS1                                              0x0000000C
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS2                                              0x0000000E
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_PS_ILWOCATIONS                                            0x00000013
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS3                                              0x00000010
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT                                           0x00000002
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT64                                         0x00000015
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_COLOR_TARGET                                   0x00000018
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_ZETA_TARGET                                    0x00000019
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_VERTICES_SUCCEEDED                              0x00000008
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED               0x00000006
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_BYTE_COUNT                                      0x0000001A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_NONE                                                      0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_VERTICES_GENERATED                                     0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_PRIMITIVES_GENERATED                                   0x00000003
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_VS_ILWOCATIONS                                            0x00000005
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_ILWOCATIONS                                            0x00000007
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_PRIMITIVES_GENERATED                                   0x00000009
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_STATUS                                          0x00000004
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_SUCCEEDED                            0x0000000B
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED                               0x0000000D
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_ILWOCATIONS                                       0x0000000F
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_PRIMITIVES_GENERATED                              0x00000011
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS0                                              0x0000000A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS1                                              0x0000000C
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS2                                              0x0000000E
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_PS_ILWOCATIONS                                            0x00000013
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS3                                              0x00000010
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT                                           0x00000002
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT64                                         0x00000015
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_COLOR_TARGET                                   0x00000018
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_ZETA_TARGET                                    0x00000019
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_VERTICES_SUCCEEDED                              0x00000008
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED               0x00000006
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_BYTE_COUNT                                      0x0000001A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_NONE                                                      0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_VERTICES_GENERATED                                     0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_PRIMITIVES_GENERATED                                   0x00000003
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_VS_ILWOCATIONS                                            0x00000005
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_ILWOCATIONS                                            0x00000007
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_PRIMITIVES_GENERATED                                   0x00000009
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_STATUS                                          0x00000004
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_SUCCEEDED                            0x0000000B
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED                               0x0000000D
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_ILWOCATIONS                                       0x0000000F
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_PRIMITIVES_GENERATED                              0x00000011
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS0                                              0x0000000A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS1                                              0x0000000C
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS2                                              0x0000000E
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_PS_ILWOCATIONS                                            0x00000013
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS3                                              0x00000010
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT                                           0x00000002
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT64                                         0x00000015
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_COLOR_TARGET                                   0x00000018
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_ZETA_TARGET                                    0x00000019
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_VERTICES_SUCCEEDED                              0x00000008
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED               0x00000006
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_BYTE_COUNT                                      0x0000001A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_NONE                                                      0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_VERTICES_GENERATED                                     0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_PRIMITIVES_GENERATED                                   0x00000003
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_VS_ILWOCATIONS                                            0x00000005
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_ILWOCATIONS                                            0x00000007
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_PRIMITIVES_GENERATED                                   0x00000009
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_STATUS                                          0x00000004
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_SUCCEEDED                            0x0000000B
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED                               0x0000000D
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_ILWOCATIONS                                       0x0000000F
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_PRIMITIVES_GENERATED                              0x00000011
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS0                                              0x0000000A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS1                                              0x0000000C
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS2                                              0x0000000E
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_PS_ILWOCATIONS                                            0x00000013
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS3                                              0x00000010
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT                                           0x00000002
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT64                                         0x00000015
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_COLOR_TARGET                                   0x00000018
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_ZETA_TARGET                                    0x00000019
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_VERTICES_SUCCEEDED                              0x00000008
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED               0x00000006
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_BYTE_COUNT                                      0x0000001A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_NONE                                                      0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_VERTICES_GENERATED                                     0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_DA_PRIMITIVES_GENERATED                                   0x00000003
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_VS_ILWOCATIONS                                            0x00000005
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_ILWOCATIONS                                            0x00000007
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_GS_PRIMITIVES_GENERATED                                   0x00000009
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_STATUS                                          0x00000004
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_SUCCEEDED                            0x0000000B
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED                               0x0000000D
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_ILWOCATIONS                                       0x0000000F
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_PRIMITIVES_GENERATED                              0x00000011
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS0                                              0x0000000A
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS1                                              0x0000000C
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS2                                              0x0000000E
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_PS_ILWOCATIONS                                            0x00000013
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS3                                              0x00000010
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT                                           0x00000002
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT64                                         0x00000015
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_COLOR_TARGET                                   0x00000018
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_ZETA_TARGET                                    0x00000019
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_VERTICES_SUCCEEDED                              0x00000008
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED               0x00000006
#define LW8597_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_BYTE_COUNT                                      0x0000001A
#define LW8597_SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE                                                        28:28
#define LW8597_SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_FOUR_WORDS                                        0x00000000
#define LW8597_SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD                                          0x00000001
#define LW8597_SET_REPORT_SEMAPHORE_D_SUB_REPORT                                                              7:5

#define LW8597_SET_GR_ZLWLL_BIT_VALUE                                                                      0x1b10
#define LW8597_SET_GR_ZLWLL_BIT_VALUE_V                                                                      31:0

#define LW8597_SET_VS_OUTPUT_REORDER_A                                                                     0x1b3c
#define LW8597_SET_VS_OUTPUT_REORDER_A_COMP00                                                                 7:0
#define LW8597_SET_VS_OUTPUT_REORDER_A_COMP01                                                                15:8
#define LW8597_SET_VS_OUTPUT_REORDER_A_COMP02                                                               23:16
#define LW8597_SET_VS_OUTPUT_REORDER_A_COMP03                                                               31:24

#define LW8597_SET_VS_OUTPUT_REORDER_B(i)                                                          (0x1b40+(i)*4)
#define LW8597_SET_VS_OUTPUT_REORDER_B_COMP00                                                                 7:0
#define LW8597_SET_VS_OUTPUT_REORDER_B_COMP01                                                                15:8
#define LW8597_SET_VS_OUTPUT_REORDER_B_COMP02                                                               23:16
#define LW8597_SET_VS_OUTPUT_REORDER_B_COMP03                                                               31:24

#define LW8597_SET_VS_OUTPUT_REORDER_C(i)                                                          (0x1b80+(i)*4)
#define LW8597_SET_VS_OUTPUT_REORDER_C_COMP00                                                                 7:0
#define LW8597_SET_VS_OUTPUT_REORDER_C_COMP01                                                                15:8
#define LW8597_SET_VS_OUTPUT_REORDER_C_COMP02                                                               23:16
#define LW8597_SET_VS_OUTPUT_REORDER_C_COMP03                                                               31:24

#define LW8597_SET_VERTEX_STREAM_A_FORMAT(j)                                                      (0x1c00+(j)*16)
#define LW8597_SET_VERTEX_STREAM_A_FORMAT_STRIDE                                                             11:0
#define LW8597_SET_VERTEX_STREAM_A_FORMAT_ENABLE                                                            12:12
#define LW8597_SET_VERTEX_STREAM_A_FORMAT_ENABLE_FALSE                                                 0x00000000
#define LW8597_SET_VERTEX_STREAM_A_FORMAT_ENABLE_TRUE                                                  0x00000001

#define LW8597_SET_VERTEX_STREAM_A_LOCATION_A(j)                                                  (0x1c04+(j)*16)
#define LW8597_SET_VERTEX_STREAM_A_LOCATION_A_OFFSET_UPPER                                                    7:0

#define LW8597_SET_VERTEX_STREAM_A_LOCATION_B(j)                                                  (0x1c08+(j)*16)
#define LW8597_SET_VERTEX_STREAM_A_LOCATION_B_OFFSET_LOWER                                                   31:0

#define LW8597_SET_VERTEX_STREAM_A_FREQUENCY(j)                                                   (0x1c0c+(j)*16)
#define LW8597_SET_VERTEX_STREAM_A_FREQUENCY_V                                                               31:0

#define LW8597_SET_VERTEX_STREAM_B_FORMAT(j)                                                      (0x1d00+(j)*16)
#define LW8597_SET_VERTEX_STREAM_B_FORMAT_STRIDE                                                             11:0
#define LW8597_SET_VERTEX_STREAM_B_FORMAT_ENABLE                                                            12:12
#define LW8597_SET_VERTEX_STREAM_B_FORMAT_ENABLE_FALSE                                                 0x00000000
#define LW8597_SET_VERTEX_STREAM_B_FORMAT_ENABLE_TRUE                                                  0x00000001

#define LW8597_SET_VERTEX_STREAM_B_LOCATION_A(j)                                                  (0x1d04+(j)*16)
#define LW8597_SET_VERTEX_STREAM_B_LOCATION_A_OFFSET_UPPER                                                    7:0

#define LW8597_SET_VERTEX_STREAM_B_LOCATION_B(j)                                                  (0x1d08+(j)*16)
#define LW8597_SET_VERTEX_STREAM_B_LOCATION_B_OFFSET_LOWER                                                   31:0

#define LW8597_SET_VERTEX_STREAM_B_FREQUENCY(j)                                                   (0x1d0c+(j)*16)
#define LW8597_SET_VERTEX_STREAM_B_FREQUENCY_V                                                               31:0

#define LW8597_SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA(j)                                         (0x1e00+(j)*32)
#define LW8597_SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA_ENABLE                                                31:0
#define LW8597_SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA_ENABLE_FALSE                                    0x00000000
#define LW8597_SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA_ENABLE_TRUE                                     0x00000001

#define LW8597_SET_BLEND_PER_TARGET_COLOR_OP(j)                                                   (0x1e04+(j)*32)
#define LW8597_SET_BLEND_PER_TARGET_COLOR_OP_V                                                               31:0
#define LW8597_SET_BLEND_PER_TARGET_COLOR_OP_V_OGL_FUNC_SUBTRACT                                       0x0000800A
#define LW8597_SET_BLEND_PER_TARGET_COLOR_OP_V_OGL_FUNC_REVERSE_SUBTRACT                               0x0000800B
#define LW8597_SET_BLEND_PER_TARGET_COLOR_OP_V_OGL_FUNC_ADD                                            0x00008006
#define LW8597_SET_BLEND_PER_TARGET_COLOR_OP_V_OGL_MIN                                                 0x00008007
#define LW8597_SET_BLEND_PER_TARGET_COLOR_OP_V_OGL_MAX                                                 0x00008008
#define LW8597_SET_BLEND_PER_TARGET_COLOR_OP_V_D3D_ADD                                                 0x00000001
#define LW8597_SET_BLEND_PER_TARGET_COLOR_OP_V_D3D_SUBTRACT                                            0x00000002
#define LW8597_SET_BLEND_PER_TARGET_COLOR_OP_V_D3D_REVSUBTRACT                                         0x00000003
#define LW8597_SET_BLEND_PER_TARGET_COLOR_OP_V_D3D_MIN                                                 0x00000004
#define LW8597_SET_BLEND_PER_TARGET_COLOR_OP_V_D3D_MAX                                                 0x00000005

#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF(j)                                         (0x1e08+(j)*32)
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V                                                     31:0
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ZERO                                      0x00004000
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ONE                                       0x00004001
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_SRC_COLOR                                 0x00004300
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_SRC_COLOR                       0x00004301
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_SRC_ALPHA                                 0x00004302
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA                       0x00004303
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_DST_ALPHA                                 0x00004304
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_DST_ALPHA                       0x00004305
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_DST_COLOR                                 0x00004306
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_DST_COLOR                       0x00004307
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_SRC_ALPHA_SATURATE                        0x00004308
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_CONSTANT_COLOR                            0x0000C001
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_CONSTANT_COLOR                  0x0000C002
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_CONSTANT_ALPHA                            0x0000C003
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_CONSTANT_ALPHA                  0x0000C004
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_SRC1COLOR                                 0x0000C900
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ILWSRC1COLOR                              0x0000C901
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_SRC1ALPHA                                 0x0000C902
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ILWSRC1ALPHA                              0x0000C903
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ZERO                                      0x00000001
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ONE                                       0x00000002
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_SRCCOLOR                                  0x00000003
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ILWSRCCOLOR                               0x00000004
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_SRCALPHA                                  0x00000005
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ILWSRCALPHA                               0x00000006
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_DESTALPHA                                 0x00000007
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ILWDESTALPHA                              0x00000008
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_DESTCOLOR                                 0x00000009
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ILWDESTCOLOR                              0x0000000A
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_SRCALPHASAT                               0x0000000B
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_BOTHSRCALPHA                              0x0000000C
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_BOTHILWSRCALPHA                           0x0000000D
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_BLENDFACTOR                               0x0000000E
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ILWBLENDFACTOR                            0x0000000F
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_SRC1COLOR                                 0x00000010
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ILWSRC1COLOR                              0x00000011
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_SRC1ALPHA                                 0x00000012
#define LW8597_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ILWSRC1ALPHA                              0x00000013

#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF(j)                                           (0x1e0c+(j)*32)
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V                                                       31:0
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ZERO                                        0x00004000
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ONE                                         0x00004001
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_SRC_COLOR                                   0x00004300
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_SRC_COLOR                         0x00004301
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_SRC_ALPHA                                   0x00004302
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA                         0x00004303
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_DST_ALPHA                                   0x00004304
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_DST_ALPHA                         0x00004305
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_DST_COLOR                                   0x00004306
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_DST_COLOR                         0x00004307
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_SRC_ALPHA_SATURATE                          0x00004308
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_CONSTANT_COLOR                              0x0000C001
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_CONSTANT_COLOR                    0x0000C002
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_CONSTANT_ALPHA                              0x0000C003
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_CONSTANT_ALPHA                    0x0000C004
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_SRC1COLOR                                   0x0000C900
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ILWSRC1COLOR                                0x0000C901
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_SRC1ALPHA                                   0x0000C902
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ILWSRC1ALPHA                                0x0000C903
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ZERO                                        0x00000001
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ONE                                         0x00000002
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_SRCCOLOR                                    0x00000003
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ILWSRCCOLOR                                 0x00000004
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_SRCALPHA                                    0x00000005
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ILWSRCALPHA                                 0x00000006
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_DESTALPHA                                   0x00000007
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ILWDESTALPHA                                0x00000008
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_DESTCOLOR                                   0x00000009
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ILWDESTCOLOR                                0x0000000A
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_SRCALPHASAT                                 0x0000000B
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_BLENDFACTOR                                 0x0000000E
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ILWBLENDFACTOR                              0x0000000F
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_SRC1COLOR                                   0x00000010
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ILWSRC1COLOR                                0x00000011
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_SRC1ALPHA                                   0x00000012
#define LW8597_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ILWSRC1ALPHA                                0x00000013

#define LW8597_SET_BLEND_PER_TARGET_ALPHA_OP(j)                                                   (0x1e10+(j)*32)
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_OP_V                                                               31:0
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_OP_V_OGL_FUNC_SUBTRACT                                       0x0000800A
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_OP_V_OGL_FUNC_REVERSE_SUBTRACT                               0x0000800B
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_OP_V_OGL_FUNC_ADD                                            0x00008006
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_OP_V_OGL_MIN                                                 0x00008007
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_OP_V_OGL_MAX                                                 0x00008008
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_OP_V_D3D_ADD                                                 0x00000001
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_OP_V_D3D_SUBTRACT                                            0x00000002
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_OP_V_D3D_REVSUBTRACT                                         0x00000003
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_OP_V_D3D_MIN                                                 0x00000004
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_OP_V_D3D_MAX                                                 0x00000005

#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF(j)                                         (0x1e14+(j)*32)
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V                                                     31:0
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ZERO                                      0x00004000
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ONE                                       0x00004001
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_SRC_COLOR                                 0x00004300
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_SRC_COLOR                       0x00004301
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_SRC_ALPHA                                 0x00004302
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA                       0x00004303
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_DST_ALPHA                                 0x00004304
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_DST_ALPHA                       0x00004305
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_DST_COLOR                                 0x00004306
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_DST_COLOR                       0x00004307
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_SRC_ALPHA_SATURATE                        0x00004308
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_CONSTANT_COLOR                            0x0000C001
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_CONSTANT_COLOR                  0x0000C002
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_CONSTANT_ALPHA                            0x0000C003
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_CONSTANT_ALPHA                  0x0000C004
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_SRC1COLOR                                 0x0000C900
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ILWSRC1COLOR                              0x0000C901
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_SRC1ALPHA                                 0x0000C902
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ILWSRC1ALPHA                              0x0000C903
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ZERO                                      0x00000001
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ONE                                       0x00000002
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_SRCCOLOR                                  0x00000003
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ILWSRCCOLOR                               0x00000004
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_SRCALPHA                                  0x00000005
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ILWSRCALPHA                               0x00000006
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_DESTALPHA                                 0x00000007
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ILWDESTALPHA                              0x00000008
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_DESTCOLOR                                 0x00000009
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ILWDESTCOLOR                              0x0000000A
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_SRCALPHASAT                               0x0000000B
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_BOTHSRCALPHA                              0x0000000C
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_BOTHILWSRCALPHA                           0x0000000D
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_BLENDFACTOR                               0x0000000E
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ILWBLENDFACTOR                            0x0000000F
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_SRC1COLOR                                 0x00000010
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ILWSRC1COLOR                              0x00000011
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_SRC1ALPHA                                 0x00000012
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ILWSRC1ALPHA                              0x00000013

#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF(j)                                           (0x1e18+(j)*32)
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V                                                       31:0
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ZERO                                        0x00004000
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ONE                                         0x00004001
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_SRC_COLOR                                   0x00004300
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_SRC_COLOR                         0x00004301
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_SRC_ALPHA                                   0x00004302
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA                         0x00004303
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_DST_ALPHA                                   0x00004304
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_DST_ALPHA                         0x00004305
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_DST_COLOR                                   0x00004306
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_DST_COLOR                         0x00004307
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_SRC_ALPHA_SATURATE                          0x00004308
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_CONSTANT_COLOR                              0x0000C001
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_CONSTANT_COLOR                    0x0000C002
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_CONSTANT_ALPHA                              0x0000C003
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_CONSTANT_ALPHA                    0x0000C004
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_SRC1COLOR                                   0x0000C900
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ILWSRC1COLOR                                0x0000C901
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_SRC1ALPHA                                   0x0000C902
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ILWSRC1ALPHA                                0x0000C903
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ZERO                                        0x00000001
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ONE                                         0x00000002
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_SRCCOLOR                                    0x00000003
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ILWSRCCOLOR                                 0x00000004
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_SRCALPHA                                    0x00000005
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ILWSRCALPHA                                 0x00000006
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_DESTALPHA                                   0x00000007
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ILWDESTALPHA                                0x00000008
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_DESTCOLOR                                   0x00000009
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ILWDESTCOLOR                                0x0000000A
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_SRCALPHASAT                                 0x0000000B
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_BLENDFACTOR                                 0x0000000E
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ILWBLENDFACTOR                              0x0000000F
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_SRC1COLOR                                   0x00000010
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ILWSRC1COLOR                                0x00000011
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_SRC1ALPHA                                   0x00000012
#define LW8597_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ILWSRC1ALPHA                                0x00000013

#define LW8597_SET_VERTEX_STREAM_LIMIT_A_A(j)                                                      (0x1f00+(j)*8)
#define LW8597_SET_VERTEX_STREAM_LIMIT_A_A_UPPER                                                              7:0

#define LW8597_SET_VERTEX_STREAM_LIMIT_A_B(j)                                                      (0x1f04+(j)*8)
#define LW8597_SET_VERTEX_STREAM_LIMIT_A_B_LOWER                                                             31:0

#define LW8597_SET_VERTEX_STREAM_LIMIT_B_A(j)                                                      (0x1f80+(j)*8)
#define LW8597_SET_VERTEX_STREAM_LIMIT_B_A_UPPER                                                              7:0

#define LW8597_SET_VERTEX_STREAM_LIMIT_B_B(j)                                                      (0x1f84+(j)*8)
#define LW8597_SET_VERTEX_STREAM_LIMIT_B_B_LOWER                                                             31:0

#endif /* _cl_gt214_tesla_h_ */
