/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2003-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef HWFE_HOPPER_B_SW_H
#define HWFE_HOPPER_B_SW_H

/* AUTO GENERATED FILE -- DO NOT EDIT */
/* Command: ../../../../class/bin/sw_header.pl hopper_b */

#include "lwtypes.h"

#define HOPPER_B    0xCC97

typedef volatile struct hopper_b_struct {
    LwU32 SetObject;
    LwU32 Reserved_0x04[0x3F];
    LwU32 NoOperation;
    LwU32 SetNotifyA;
    LwU32 SetNotifyB;
    LwU32 Notify;
    LwU32 WaitForIdle;
    LwU32 LoadMmeInstructionRamPointer;
    LwU32 LoadMmeInstructionRam;
    LwU32 LoadMmeStartAddressRamPointer;
    LwU32 LoadMmeStartAddressRam;
    LwU32 SetMmeShadowRamControl;
    LwU32 PeerSemaphoreReleaseOffsetUpper;
    LwU32 PeerSemaphoreReleaseOffset;
    LwU32 SetGlobalRenderEnableA;
    LwU32 SetGlobalRenderEnableB;
    LwU32 SetGlobalRenderEnableC;
    LwU32 SendGoIdle;
    LwU32 PmTrigger;
    LwU32 PmTriggerWfi;
    LwU32 FeAtomicSequenceBegin;
    LwU32 FeAtomicSequenceEnd;
    LwU32 SetInstrumentationMethodHeader;
    LwU32 SetInstrumentationMethodData;
    LwU32 SetReportSemaphorePayloadLower;
    LwU32 SetReportSemaphorePayloadUpper;
    LwU32 SetReportSemaphoreAddressLower;
    LwU32 SetReportSemaphoreAddressUpper;
    LwU32 ReportSemaphoreExelwte;
    LwU32 Reserved_0x16C[0x5];
    LwU32 LineLengthIn;
    LwU32 LineCount;
    LwU32 OffsetOutUpper;
    LwU32 OffsetOut;
    LwU32 PitchOut;
    LwU32 SetDstBlockSize;
    LwU32 SetDstWidth;
    LwU32 SetDstHeight;
    LwU32 SetDstDepth;
    LwU32 SetDstLayer;
    LwU32 SetDstOriginBytesX;
    LwU32 SetDstOriginSamplesY;
    LwU32 LaunchDma;
    LwU32 LoadInlineData;
    LwU32 Reserved_0x1B8[0x9];
    LwU32 SetI2mSemaphoreA;
    LwU32 SetI2mSemaphoreB;
    LwU32 SetI2mSemaphoreC;
    LwU32 Reserved_0x1E8[0x1];
    LwU32 SetMmeSwitchState;
    LwU32 SetI2mSpareNoop00;
    LwU32 SetI2mSpareNoop01;
    LwU32 SetI2mSpareNoop02;
    LwU32 SetI2mSpareNoop03;
    LwU32 RunDsNow;
    LwU32 SetOpportunisticEarlyZHysteresis;
    LwU32 SetGsMode;
    LwU32 SetAliasedLineWidthEnable;
    LwU32 SetApiMandatedEarlyZ;
    LwU32 SetGsDmFifo;
    LwU32 SetL2CacheControlForRopPrefetchReadRequests;
    LwU32 IlwalidateShaderCaches;
    LwU32 SetInstanceCount;
    LwU32 SetPositionWScaledOffsetEnable;
    LwU32 SetMultiView;
    LwU32 SetGoIdleTimeout;
    LwU32 SetConservativeRasterControl;
    LwU32 SetMmeVersion;
    LwU32 SetIndexBufferSizeA;
    LwU32 SetIndexBufferSizeB;
    LwU32 SetRootTableVisibility[0x8];
    LwU32 SetDrawControlA;
    LwU32 SetDrawControlB;
    LwU32 DrawIndexBufferBeginEndA;
    LwU32 DrawIndexBufferBeginEndB;
    LwU32 DrawVertexArrayBeginEndA;
    LwU32 DrawVertexArrayBeginEndB;
    LwU32 SetCheckerboardClipControl;
    LwU32 IlwalidateRasterCacheNoWfi;
    LwU32 SetVariablePixelRateSampleOrder[0xD];
    LwU32 SetShaderLocalMemoryThrottle;
    LwU32 SetColorRenderToZetaSurface;
    LwU32 SetZlwllVisiblePrimOptimization;
    LwU32 SetVpcTestingControl;
    LwU32 SetCheckerboardClipOwnership;
    LwU32 IncrementSyncPoint;
    LwU32 SetVariablePixelRateShadingTableSelect;
    LwU32 SetRootTablePrefetch;
    LwU32 FlushAndIlwalidateRopMiniCache;
    LwU32 SetSurfaceClipIdBlockSize;
    LwU32 SetAlphaCirlwlarBufferSize;
    LwU32 DecompressSurface;
    LwU32 SetZlwllRopBypass;
    LwU32 SetZlwllSubregion;
    LwU32 SetRasterBoundingBox;
    LwU32 PeerSemaphoreRelease;
    LwU32 SetIteratedBlendOptimization;
    LwU32 SetZlwllSubregionAllocation;
    LwU32 AssignZlwllSubregions;
    LwU32 SetPsOutputSampleMaskUsage;
    LwU32 DrawZeroIndex;
    LwU32 SetL1Configuration;
    LwU32 SetRenderEnableControl;
    LwU32 SetSpaVersion;
    LwU32 SetTimesliceBatchLimit;
    LwU32 SetSnapGridLine;
    LwU32 SetSnapGridNonLine;
    LwU32 SetTessellationParameters;
    LwU32 SetTessellationLodU0OrDensity;
    LwU32 SetTessellationLodV0OrDetail;
    LwU32 SetTessellationLodU1OrW0;
    LwU32 SetTessellationLodV1;
    LwU32 SetTgLodInteriorU;
    LwU32 SetTgLodInteriorV;
    LwU32 ReservedTg07;
    LwU32 ReservedTg08;
    LwU32 ReservedTg09;
    LwU32 ReservedTg10;
    LwU32 ReservedTg11;
    LwU32 ReservedTg12;
    LwU32 ReservedTg13;
    LwU32 ReservedTg14;
    LwU32 ReservedTg15;
    LwU32 SetSubtilingPerfKnobA;
    LwU32 SetSubtilingPerfKnobB;
    LwU32 SetSubtilingPerfKnobC;
    LwU32 SetZlwllSubregionToReport;
    LwU32 SetZlwllSubregionReportType;
    LwU32 SetBalancedPrimitiveWorkload;
    LwU32 SetMaxPatchesPerBatch;
    LwU32 SetRasterEnable;
    struct {
        LwU32 Enable;
        LwU32 AddressA;
        LwU32 AddressB;
        LwU32 Size;
        LwU32 LoadWritePointer;
        LwU32 Reserved_0x14[0x3];
    } SetStreamOutBuffer[0x4];
    struct {
        LwU32 ScaleA;
        LwU32 ScaleB;
        LwU32 ReservedA;
        LwU32 ReservedB;
    } SetPositionWScaledOffset[0x10];
    LwU32 SetZRopSliceMap;
    LwU32 SetRootTableSelector;
    LwU32 LoadRootTable;
    LwU32 Reserved_0x50C[0x11];
    LwU32 SetMmeMemAddressA;
    LwU32 SetMmeMemAddressB;
    LwU32 SetMmeDataRamAddress;
    LwU32 MmeDmaRead;
    LwU32 MmeDmaReadFifoed;
    LwU32 MmeDmaWrite;
    LwU32 MmeDmaReduction;
    LwU32 MmeDmaSysmembar;
    LwU32 MmeDmaSync;
    LwU32 SetMmeDataFifoConfig;
    LwU32 Reserved_0x578[0x22];
    struct {
        LwU32 A;
        LwU32 B;
    } SetVertexStreamSize[0x20];
    struct {
        LwU32 Stream;
        LwU32 ComponentCount;
        LwU32 Stride;
        LwU32 Reserved_0x0C[0x1];
    } SetStreamOutControl[0x4];
    LwU32 SetRasterInput;
    LwU32 SetStreamOutput;
    LwU32 SetDaPrimitiveRestartTopologyChange;
    LwU32 Reserved_0x74C[0x1];
    LwU32 SetShadermodel3Reorder;
    LwU32 SetHybridAntiAliasControl;
    LwU32 SetQuadroTune;
    LwU32 Reserved_0x75C[0x1];
    LwU32 SetShadermodel3VsOutputGeneric[0x4];
    LwU32 SetShadermodel3VsOutputTexcoord[0x3];
    LwU32 SetShaderLocalMemoryWindow;
    LwU32 SetShadermodel3PsInputGeneric[0x4];
    LwU32 SetShaderLocalMemoryA;
    LwU32 SetShaderLocalMemoryB;
    LwU32 SetShaderLocalMemoryC;
    LwU32 SetShaderLocalMemoryD;
    LwU32 SetShaderLocalMemoryE;
    LwU32 SetColorZeroBandwidthClear;
    LwU32 SetZZeroBandwidthClear;
    LwU32 SetTextureHeaderVersion;
    LwU32 SetStencilZeroBandwidthClear;
    LwU32 Reserved_0x7B4[0x1];
    LwU32 SetMonitoredFenceSignalAddressBaseA;
    LwU32 SetMonitoredFenceSignalAddressBaseB;
    LwU32 SetZlwllRegionSizeA;
    LwU32 SetZlwllRegionSizeB;
    LwU32 SetZlwllRegionSizeC;
    LwU32 SetZlwllRegionPixelOffsetC;
    LwU32 SetShadermodel3PsInputTexcoord[0x3];
    LwU32 SetLwllBeforeFetch;
    LwU32 SetZlwllRegionLocation;
    LwU32 SetZlwllRegionAliquots;
    LwU32 SetZlwllStorageA;
    LwU32 SetZlwllStorageB;
    LwU32 SetZlwllStorageC;
    LwU32 SetZlwllStorageD;
    LwU32 SetZtReadOnly;
    LwU32 ThrottleSm;
    struct {
        LwU32 A;
        LwU32 B;
        LwU32 Width;
        LwU32 Height;
        LwU32 Format;
        LwU32 Memory;
        LwU32 ThirdDimension;
        LwU32 ArrayPitch;
        LwU32 Layer;
        LwU32 CRopSliceMap;
        LwU32 Reserved_0x28[0x6];
    } SetColorTarget[0x8];
    struct {
        LwU32 ScaleX;
        LwU32 ScaleY;
        LwU32 ScaleZ;
        LwU32 OffsetX;
        LwU32 OffsetY;
        LwU32 OffsetZ;
        LwU32 CoordinateSwizzle;
        LwU32 IncreaseSnapGridPrecision;
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
    LwU32 SetMaxStreamOutputGsInstancesPerTask;
    LwU32 SetApiVisibleCallLimit;
    LwU32 SetStatisticsCounter;
    LwU32 SetClearRectHorizontal;
    LwU32 SetClearRectVertical;
    LwU32 SetVertexArrayStart;
    LwU32 DrawVertexArray;
    LwU32 SetViewportZClip;
    LwU32 SetColorClearValue[0x4];
    LwU32 SetZClearValue;
    LwU32 SetShaderCacheControl;
    LwU32 ForceTransitionToBeta;
    LwU32 SetReduceColorThresholdsEnable;
    LwU32 SetStencilClearValue;
    LwU32 IlwalidateShaderCachesNoWfi;
    LwU32 SetZlwllSerialization;
    LwU32 SetFrontPolygonMode;
    LwU32 SetBackPolygonMode;
    LwU32 SetPolySmooth;
    LwU32 Reserved_0xDB8[0x1];
    LwU32 SetZlwllDirFormat;
    LwU32 SetPolyOffsetPoint;
    LwU32 SetPolyOffsetLine;
    LwU32 SetPolyOffsetFill;
    LwU32 SetPatch;
    LwU32 SetIteratedBlend;
    LwU32 SetIteratedBlendPass;
    LwU32 SetZlwllCriterion;
    LwU32 Reserved_0xDDC[0x1];
    LwU32 PixelShaderBarrier;
    LwU32 SetSmTimeoutInterval;
    LwU32 SetDaPrimitiveRestartVertexArray;
    LwU32 MmeDmaWriteMethodBarrier;
    LwU32 LoadIteratedBlendInstructionPointer;
    LwU32 LoadIteratedBlendInstruction;
    LwU32 SetWindowOffsetX;
    LwU32 SetWindowOffsetY;
    struct {
        LwU32 Enable;
        LwU32 Horizontal;
        LwU32 Vertical;
        LwU32 Reserved_0x0C[0x1];
    } SetScissor[0x10];
    LwU32 SetStA;
    LwU32 SetStB;
    LwU32 SetStBlockSize;
    LwU32 SetStArrayPitch;
    LwU32 Reserved_0xF10[0x1];
    LwU32 SetVpcPerfKnob;
    LwU32 PmLocalTrigger;
    LwU32 SetPostZPsImask;
    LwU32 SetMultiViewRenderTargetArrayIndexOffset[0x4];
    LwU32 SetSpareMultiViewRenderTargetArrayIndexOffset[0x4];
    LwU32 SetConstantColorRendering;
    LwU32 SetConstantColorRenderingRed;
    LwU32 SetConstantColorRenderingGreen;
    LwU32 SetConstantColorRenderingBlue;
    LwU32 SetConstantColorRenderingAlpha;
    LwU32 SetBackStencilFuncRef;
    LwU32 SetBackStencilMask;
    LwU32 SetBackStencilFuncMask;
    LwU32 SetTiledCache;
    LwU32 SetTiledCacheTileSize;
    LwU32 SetTiledCacheBufferInterleave;
    LwU32 SetTiledCacheControl;
    LwU32 SetTiledCacheStateThreshold;
    LwU32 TiledCacheIlwalidateTextureDataCache;
    LwU32 TiledCacheIlwalidate;
    LwU32 TiledCacheBarrier;
    LwU32 TiledCacheFlush;
    LwU32 SetVertexStreamSubstituteA;
    LwU32 SetVertexStreamSubstituteB;
    LwU32 SetLineModePolygonClip;
    LwU32 SetSingleCtWriteControl;
    LwU32 MultiPassIntervalBoundary;
    LwU32 SetVtgWarpWatermarks;
    LwU32 SetDepthBoundsMin;
    LwU32 SetDepthBoundsMax;
    LwU32 SetSampleMask;
    LwU32 SetColorTargetSampleMask;
    LwU32 SetCtMrtEnable;
    LwU32 SetNonmultisampledZ;
    LwU32 SetTir;
    LwU32 SetAntiAliasRaster;
    LwU32 SetSampleMaskX0Y0;
    LwU32 SetSampleMaskX1Y0;
    LwU32 SetSampleMaskX0Y1;
    LwU32 SetSampleMaskX1Y1;
    LwU32 SetSurfaceClipIdMemoryA;
    LwU32 SetSurfaceClipIdMemoryB;
    LwU32 SetTirModulation;
    LwU32 SetTirModulationFunction;
    LwU32 SetBlendOptControl;
    LwU32 SetZtA;
    LwU32 SetZtB;
    LwU32 SetZtFormat;
    LwU32 SetZtBlockSize;
    LwU32 SetZtArrayPitch;
    LwU32 SetSurfaceClipHorizontal;
    LwU32 SetSurfaceClipVertical;
    LwU32 Reserved_0xFFC[0x1];
    LwU32 SetL2CacheControlForVafRequests;
    LwU32 SetViewportMulticast;
    LwU32 SetTessellationLwtHeight;
    LwU32 SetMaxGsInstancesPerTask;
    LwU32 SetMaxGsOutputVerticesPerTask;
    LwU32 SetReservedSwMethod00;
    LwU32 SetGsOutputCbStorageMultiplier;
    LwU32 SetBetaCbStorageConstraint;
    LwU32 SetTiOutputCbStorageMultiplier;
    LwU32 SetAlphaCbStorageConstraint;
    LwU32 SetReservedSwMethod01;
    LwU32 SetReservedSwMethod02;
    LwU32 SetTirModulationCoefficientTable[0x4];
    LwU32 Reserved_0x1040[0x1];
    LwU32 SetSpareNoop01;
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
    LwU32 SetSpareNoop14;
    LwU32 SetSpareNoop15;
    LwU32 SetPixelsFromMemoryDstX0;
    LwU32 SetPixelsFromMemoryDstY0;
    LwU32 SetPixelsFromMemoryDstWidth;
    LwU32 SetPixelsFromMemoryDstHeight;
    LwU32 SetPixelsFromMemoryDuDxFrac;
    LwU32 SetPixelsFromMemoryDuDxInt;
    LwU32 SetPixelsFromMemoryDvDyFrac;
    LwU32 SetPixelsFromMemoryDvDyInt;
    LwU32 SetPixelsFromMemoryTextureSampler;
    LwU32 SetPixelsFromMemoryTextureHeader;
    LwU32 SetPixelsFromMemorySrcX0;
    LwU32 PixelsFromMemorySrcY0;
    LwU32 SetReservedSwMethod03;
    LwU32 SetReservedSwMethod04;
    LwU32 SetReservedSwMethod05;
    LwU32 SetReservedSwMethod06;
    LwU32 SetReservedSwMethod07;
    LwU32 SetReservedSwMethod08;
    LwU32 SetReservedSwMethod09;
    LwU32 SetReduceColorThresholdsUnorm8;
    LwU32 SetReservedSwMethod10;
    LwU32 SetReservedSwMethod11;
    LwU32 SetReservedSwMethod12;
    LwU32 SetReservedSwMethod13;
    LwU32 SetReduceColorThresholdsUnorm10;
    LwU32 SetReduceColorThresholdsUnorm16;
    LwU32 SetReduceColorThresholdsFp11;
    LwU32 SetReduceColorThresholdsFp16;
    LwU32 SetReduceColorThresholdsSrgb8;
    LwU32 UnbindAll;
    LwU32 SetClearSurfaceControl;
    LwU32 SetL2CacheControlForRopNoninterlockedReadRequests;
    LwU32 SetReservedSwMethod14;
    LwU32 SetReservedSwMethod15;
    LwU32 SetTiledCacheControlExtended;
    LwU32 NoOperationDataHi;
    LwU32 SetDepthBiasControl;
    LwU32 PmTriggerEnd;
    LwU32 SetVertexIdBase;
    LwU32 SetStencilCompression;
    LwU32 SetDaOutputAttributeSkipMaskA[0x2];
    LwU32 SetDaOutputAttributeSkipMaskB[0x2];
    LwU32 SetTirControl;
    LwU32 SetMutableMethodControl;
    LwU32 SetPostPsInitialCoverage;
    LwU32 SetFillViaTriangle;
    LwU32 SetBlendPerFormatEnable;
    LwU32 FlushPendingWrites;
    LwU32 SetConservativeRaster;
    LwU32 SetMeshControl;
    LwU32 SetMeshShaderA;
    LwU32 SetMeshShaderB;
    LwU32 SetMeshInitShader;
    LwU32 Reserved_0x115C[0x1];
    LwU32 SetVertexAttributeA[0x10];
    LwU32 SetVertexAttributeB[0x10];
    LwU32 SetAntiAliasSamplePositions[0x4];
    LwU32 SetOffsetRenderTargetIndex;
    LwU32 ForceHeavyweightMethodSync;
    LwU32 SetCoverageToColor;
    LwU32 DecompressZetaSurface;
    LwU32 TiledCacheIlwalidateTextureSamplerCache;
    LwU32 SetScreenStateMask;
    LwU32 SetZtSparse;
    LwU32 SetStSizeA;
    LwU32 SetStSizeB;
    LwU32 DrawVertexArrayBeginEndInstanceFirst;
    LwU32 DrawVertexArrayBeginEndInstanceSubsequent;
    LwU32 SetCtSelect;
    LwU32 SetCompressionThreshold;
    LwU32 SetPixelShaderInterlockControl;
    LwU32 SetZtSizeA;
    LwU32 SetZtSizeB;
    LwU32 SetZtSizeC;
    LwU32 SetSamplerBinding;
    LwU32 SetMultiPassPrimitive;
    LwU32 DrawAuto;
    LwU32 SetPostVtgShaderAttributeSkipMask[0x8];
    LwU32 SetPixelShaderTicketDispenserValue;
    LwU32 SetBackEndCopyA;
    LwU32 SetBackEndCopyB;
    LwU32 SetBackEndCopyC;
    LwU32 SetBackEndCopyD;
    LwU32 SetBackEndCopyE;
    LwU32 SetMultiPassShaderMetaData;
    LwU32 Reserved_0x127C[0x1];
    LwU32 SetCirlwlarBufferSize;
    LwU32 SetVtgRegisterWatermarks;
    LwU32 IlwalidateTextureDataCacheNoWfi;
    LwU32 Reserved_0x128C[0x1];
    LwU32 SetL2CacheControlForRopInterlockedReadRequests;
    LwU32 Reserved_0x1294[0x4];
    LwU32 SetDaPrimitiveRestartIndexTopologyChange;
    struct {
        LwU32 AddressA;
        LwU32 AddressB;
        LwU32 SizeA;
        LwU32 SizeB;
        LwU32 Layer;
        LwU32 ArrayPitch;
        LwU32 BlockSize;
        LwU32 AllocatedSize;
    } SetShadingRateIndexSurface[0x1];
    LwU32 ClearZlwllRegion;
    LwU32 SetDepthTest;
    LwU32 SetFillMode;
    LwU32 SetShadeMode;
    LwU32 SetL2CacheControlForRopNoninterlockedWriteRequests;
    LwU32 SetL2CacheControlForRopInterlockedWriteRequests;
    LwU32 SetAlphaToCoverageDitherControl;
    LwU32 SetBlendStatePerTarget;
    LwU32 SetDepthWrite;
    LwU32 SetAlphaTest;
    LwU32 Reserved_0x12F0[0x4];
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
    LwU32 Reserved_0x132C[0x1];
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
    LwU32 SetStencilFuncMask;
    LwU32 SetStencilMask;
    LwU32 Reserved_0x13A0[0x1];
    LwU32 SetDrawAutoStart;
    LwU32 SetPsSaturate;
    LwU32 SetWindowOrigin;
    LwU32 SetLineWidthFloat;
    LwU32 SetAliasedLineWidthFloat;
    LwU32 Reserved_0x13B8[0x18];
    LwU32 SetLineMultisampleOverride;
    LwU32 Reserved_0x141C[0x2];
    LwU32 IlwalidateSamplerCacheNoWfi;
    LwU32 IlwalidateTextureHeaderCacheNoWfi;
    LwU32 Reserved_0x142C[0x2];
    LwU32 SetGlobalBaseVertexIndex;
    LwU32 SetGlobalBaseInstanceIndex;
    LwU32 Reserved_0x143C[0x2];
    LwU32 SetGsSpillRegionA;
    LwU32 SetGsSpillRegionB;
    LwU32 SetGsSpillRegionC;
    LwU32 SetPsWarpWatermarks;
    LwU32 SetPsRegisterWatermarks;
    LwU32 Reserved_0x1458[0x3];
    LwU32 StoreZlwll;
    LwU32 Reserved_0x1468[0x6];
    struct {
        LwU32 Red;
        LwU32 Green;
        LwU32 Blue;
        LwU32 Reserved_0x0C[0x1];
    } SetIteratedBlendConstant[0x8];
    LwU32 LoadZlwll;
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
    LwU32 Reserved_0x152C[0x1];
    LwU32 ClearReportValue;
    LwU32 SetAntiAliasEnable;
    LwU32 SetZtSelect;
    LwU32 SetAntiAliasAlphaControl;
    LwU32 Reserved_0x1540[0x4];
    LwU32 SetRenderEnableA;
    LwU32 SetRenderEnableB;
    LwU32 SetRenderEnableC;
    LwU32 SetTexSamplerPoolA;
    LwU32 SetTexSamplerPoolB;
    LwU32 SetTexSamplerPoolC;
    LwU32 Reserved_0x1568[0x1];
    LwU32 SetSlopeScaleDepthBias;
    LwU32 SetAntiAliasedLine;
    LwU32 SetTexHeaderPoolA;
    LwU32 SetTexHeaderPoolB;
    LwU32 SetTexHeaderPoolC;
    LwU32 Reserved_0x1580[0x4];
    LwU32 SetActiveZlwllRegion;
    LwU32 SetTwoSidedStencilTest;
    LwU32 SetBackStencilOpFail;
    LwU32 SetBackStencilOpZfail;
    LwU32 SetBackStencilOpZpass;
    LwU32 SetBackStencilFunc;
    LwU32 Reserved_0x15A8[0x4];
    LwU32 SetSrgbWrite;
    LwU32 SetDepthBias;
    LwU32 Reserved_0x15C0[0x2];
    LwU32 SetZlwllRegionFormat;
    LwU32 SetRtLayer;
    LwU32 SetAntiAlias;
    LwU32 Reserved_0x15D4[0x4];
    LwU32 SetEdgeFlag;
    LwU32 DrawInlineIndex;
    LwU32 SetInlineIndex2x16Align;
    LwU32 DrawInlineIndex2x16;
    LwU32 SetVertexGlobalBaseOffsetA;
    LwU32 SetVertexGlobalBaseOffsetB;
    LwU32 SetZlwllRegionPixelOffsetA;
    LwU32 SetZlwllRegionPixelOffsetB;
    LwU32 SetPointSpriteSelect;
    LwU32 Reserved_0x1608[0x2];
    LwU32 SetAttributeDefault;
    LwU32 End;
    LwU32 Begin;
    LwU32 SetVertexIdCopy;
    LwU32 AddToPrimitiveId;
    LwU32 LoadPrimitiveId;
    LwU32 Reserved_0x1628[0x1];
    LwU32 SetShaderBasedLwll;
    LwU32 Reserved_0x1630[0x2];
    LwU32 SetClassVersion;
    LwU32 Reserved_0x163C[0x2];
    LwU32 SetDaPrimitiveRestart;
    LwU32 SetDaPrimitiveRestartIndex;
    LwU32 SetDaOutput;
    LwU32 Reserved_0x1650[0x2];
    LwU32 SetAntiAliasedPoint;
    LwU32 SetPointCenterMode;
    LwU32 Reserved_0x1660[0x2];
    LwU32 SetLineSmoothParameters;
    LwU32 SetLineStipple;
    LwU32 SetLineSmoothEdgeTable[0x4];
    LwU32 SetLineStippleParameters;
    LwU32 SetProvokingVertex;
    LwU32 SetTwoSidedLight;
    LwU32 SetPolygonStipple;
    LwU32 SetShaderControl;
    LwU32 Reserved_0x1694[0x3];
    LwU32 CheckClassVersion;
    LwU32 SetSphVersion;
    LwU32 CheckSphVersion;
    LwU32 Reserved_0x16AC[0x2];
    LwU32 SetAlphaToCoverageOverride;
    LwU32 Reserved_0x16B8[0x1];
    LwU32 SetScgGraphicsPriority;
    LwU32 SetScgGraphicsSchedulingParameters[0x10];
    LwU32 SetPolygonStipplePattern[0x20];
    LwU32 Reserved_0x1780[0x4];
    LwU32 SetAamVersion;
    LwU32 CheckAamVersion;
    LwU32 Reserved_0x1798[0x1];
    LwU32 SetZtLayer;
    LwU32 Reserved_0x17A0[0xA];
    LwU32 SetIndexBufferA;
    LwU32 SetIndexBufferB;
    LwU32 Reserved_0x17D0[0x2];
    LwU32 SetIndexBufferE;
    LwU32 SetIndexBufferF;
    LwU32 DrawIndexBuffer;
    LwU32 DrawIndexBuffer32BeginEndInstanceFirst;
    LwU32 DrawIndexBuffer16BeginEndInstanceFirst;
    LwU32 DrawIndexBuffer8BeginEndInstanceFirst;
    LwU32 DrawIndexBuffer32BeginEndInstanceSubsequent;
    LwU32 DrawIndexBuffer16BeginEndInstanceSubsequent;
    LwU32 DrawIndexBuffer8BeginEndInstanceSubsequent;
    LwU32 Reserved_0x17FC[0x20];
    LwU32 SetDepthBiasClamp;
    LwU32 SetVertexStreamInstanceA[0x10];
    LwU32 SetVertexStreamInstanceB[0x10];
    LwU32 Reserved_0x1900[0x4];
    LwU32 SetAttributePointSize;
    LwU32 Reserved_0x1914[0x1];
    LwU32 OglSetLwll;
    LwU32 OglSetFrontFace;
    LwU32 OglSetLwllFace;
    LwU32 SetViewportPixel;
    LwU32 Reserved_0x1928[0x1];
    LwU32 SetViewportScaleOffset;
    LwU32 Reserved_0x1930[0x3];
    LwU32 SetViewportClipControl;
    LwU32 SetUserClipOp;
    LwU32 SetRenderEnableOverride;
    LwU32 SetPrimitiveTopologyControl;
    LwU32 SetWindowClipEnable;
    LwU32 SetWindowClipType;
    LwU32 Reserved_0x1954[0x1];
    LwU32 IlwalidateZlwll;
    LwU32 Reserved_0x195C[0x3];
    LwU32 SetZlwll;
    LwU32 SetZlwllBounds;
    LwU32 SetPrimitiveTopology;
    LwU32 Reserved_0x1974[0x1];
    LwU32 ZlwllSync;
    LwU32 SetClipIdTest;
    LwU32 SetSurfaceClipIdWidth;
    LwU32 SetClipId;
    LwU32 Reserved_0x1988[0xD];
    LwU32 SetDepthBoundsTest;
    LwU32 SetBlendFloatOption;
    LwU32 SetLogicOp;
    LwU32 SetLogicOpFunc;
    LwU32 SetZCompression;
    LwU32 ClearSurface;
    LwU32 ClearClipIdSurface;
    LwU32 Reserved_0x19D8[0x2];
    LwU32 SetColorCompression[0x8];
    LwU32 SetCtWrite[0x8];
    LwU32 Reserved_0x1A20[0x1];
    LwU32 TestForQuadro;
    LwU32 Reserved_0x1A28[0x1];
    LwU32 PipeNop;
    LwU32 SetSpare00;
    LwU32 SetSpare01;
    LwU32 SetSpare02;
    LwU32 SetSpare03;
    LwU32 Reserved_0x1A40[0x30];
    LwU32 SetReportSemaphoreA;
    LwU32 SetReportSemaphoreB;
    LwU32 SetReportSemaphoreC;
    LwU32 SetReportSemaphoreD;
    LwU32 Reserved_0x1B10[0x3C];
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
    LwU32 Reserved_0x1F00[0x40];
    struct {
        LwU32 Shader;
        LwU32 ReservedB;
        LwU32 ReservedA;
        LwU32 RegisterCount;
        LwU32 Binding;
        LwU32 ProgramAddressA;
        LwU32 ProgramAddressB;
        LwU32 ProgramPrefetch;
        LwU32 ReservedE;
        LwU32 Reserved_0x24[0x7];
    } SetPipeline[0x6];
    LwU32 Reserved_0x2180[0x60];
    LwU32 SetFalcon00;
    LwU32 SetFalcon01;
    LwU32 SetFalcon02;
    LwU32 SetFalcon03;
    LwU32 SetFalcon04;
    LwU32 SetFalcon05;
    LwU32 SetFalcon06;
    LwU32 SetFalcon07;
    LwU32 SetFalcon08;
    LwU32 SetFalcon09;
    LwU32 SetFalcon10;
    LwU32 SetFalcon11;
    LwU32 SetFalcon12;
    LwU32 SetFalcon13;
    LwU32 SetFalcon14;
    LwU32 SetFalcon15;
    LwU32 SetFalcon16;
    LwU32 SetFalcon17;
    LwU32 SetFalcon18;
    LwU32 SetFalcon19;
    LwU32 SetFalcon20;
    LwU32 SetFalcon21;
    LwU32 SetFalcon22;
    LwU32 SetFalcon23;
    LwU32 SetFalcon24;
    LwU32 SetFalcon25;
    LwU32 SetFalcon26;
    LwU32 SetFalcon27;
    LwU32 SetFalcon28;
    LwU32 SetFalcon29;
    LwU32 SetFalcon30;
    LwU32 SetFalcon31;
    LwU32 SetConstantBufferSelectorA;
    LwU32 SetConstantBufferSelectorB;
    LwU32 SetConstantBufferSelectorC;
    LwU32 LoadConstantBufferOffset;
    LwU32 LoadConstantBuffer[0x10];
    LwU32 Reserved_0x23D0[0xC];
    struct {
        LwU32 ReservedA;
        LwU32 ReservedB;
        LwU32 ReservedC;
        LwU32 ReservedD;
        LwU32 ConstantBuffer;
        LwU32 Reserved_0x14[0x3];
    } BindGroup[0x5];
    LwU32 Reserved_0x24A0[0x56];
    LwU32 SetTrapHandlerA;
    LwU32 SetTrapHandlerB;
    LwU32 SetColorClamp;
    LwU32 Reserved_0x2604[0x3];
    struct {
        LwU32 Enable;
        LwU32 Horizontal;
        LwU32 Vertical;
        LwU32 Reserved_0x0C[0x1];
    } SetScissorExclusive[0x10];
    LwU32 Reserved_0x2710[0x3C];
    struct {
        LwU32 Select[0x20];
    } SetStreamOutLayout[0x4];
    struct {
        LwU32 Control;
        LwU32 IndexToRateA;
        LwU32 IndexToRateB;
        LwU32 ReservedA;
    } SetVariablePixelRateShading[0x10];
    LwU32 Reserved_0x2B00[0x1FD];
    LwU32 SetShaderPerformanceSnapshotCounterValue[0x8];
    LwU32 SetShaderPerformanceSnapshotCounterValueUpper[0x8];
    LwU32 EnableShaderPerformanceSnapshotCounter;
    LwU32 DisableShaderPerformanceSnapshotCounter;
    LwU32 SetShaderPerformanceCounterValueUpper[0x8];
    LwU32 SetShaderPerformanceCounterValue[0x8];
    LwU32 SetShaderPerformanceCounterEvent[0x8];
    LwU32 SetShaderPerformanceCounterControlA[0x8];
    LwU32 SetShaderPerformanceCounterControlB[0x8];
    LwU32 SetShaderPerformanceCounterTrapControl;
    LwU32 StartShaderPerformanceCounter;
    LwU32 StopShaderPerformanceCounter;
    LwU32 SetShaderPerformanceCounterSctlFilter;
    LwU32 SetShaderPerformanceCounterCoreMioFilter;
    LwU32 Reserved_0x33F0[0x4];
    LwU32 SetMmeShadowScratch[0x100];
    struct {
        LwU32 Macro;
        LwU32 Data;
    } CallMme[0xE0];
} hopper_b_t;


#define LWCC97_SET_OBJECT                                                                                  0x0000
#define LWCC97_SET_OBJECT_CLASS_ID                                                                           15:0
#define LWCC97_SET_OBJECT_ENGINE_ID                                                                         20:16

#define LWCC97_NO_OPERATION                                                                                0x0100
#define LWCC97_NO_OPERATION_V                                                                                31:0

#define LWCC97_SET_NOTIFY_A                                                                                0x0104
#define LWCC97_SET_NOTIFY_A_ADDRESS_UPPER                                                                    24:0

#define LWCC97_SET_NOTIFY_B                                                                                0x0108
#define LWCC97_SET_NOTIFY_B_ADDRESS_LOWER                                                                    31:0

#define LWCC97_NOTIFY                                                                                      0x010c
#define LWCC97_NOTIFY_TYPE                                                                                   31:0
#define LWCC97_NOTIFY_TYPE_WRITE_ONLY                                                                  0x00000000
#define LWCC97_NOTIFY_TYPE_WRITE_THEN_AWAKEN                                                           0x00000001

#define LWCC97_WAIT_FOR_IDLE                                                                               0x0110
#define LWCC97_WAIT_FOR_IDLE_V                                                                               31:0

#define LWCC97_LOAD_MME_INSTRUCTION_RAM_POINTER                                                            0x0114
#define LWCC97_LOAD_MME_INSTRUCTION_RAM_POINTER_V                                                            31:0

#define LWCC97_LOAD_MME_INSTRUCTION_RAM                                                                    0x0118
#define LWCC97_LOAD_MME_INSTRUCTION_RAM_V                                                                    31:0

#define LWCC97_LOAD_MME_START_ADDRESS_RAM_POINTER                                                          0x011c
#define LWCC97_LOAD_MME_START_ADDRESS_RAM_POINTER_V                                                          31:0

#define LWCC97_LOAD_MME_START_ADDRESS_RAM                                                                  0x0120
#define LWCC97_LOAD_MME_START_ADDRESS_RAM_V                                                                  31:0

#define LWCC97_SET_MME_SHADOW_RAM_CONTROL                                                                  0x0124
#define LWCC97_SET_MME_SHADOW_RAM_CONTROL_MODE                                                                1:0
#define LWCC97_SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_TRACK                                            0x00000000
#define LWCC97_SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_TRACK_WITH_FILTER                                0x00000001
#define LWCC97_SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_PASSTHROUGH                                      0x00000002
#define LWCC97_SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_REPLAY                                           0x00000003

#define LWCC97_PEER_SEMAPHORE_RELEASE_OFFSET_UPPER                                                         0x0128
#define LWCC97_PEER_SEMAPHORE_RELEASE_OFFSET_UPPER_V                                                          7:0

#define LWCC97_PEER_SEMAPHORE_RELEASE_OFFSET                                                               0x012c
#define LWCC97_PEER_SEMAPHORE_RELEASE_OFFSET_V                                                               31:0

#define LWCC97_SET_GLOBAL_RENDER_ENABLE_A                                                                  0x0130
#define LWCC97_SET_GLOBAL_RENDER_ENABLE_A_OFFSET_UPPER                                                        7:0

#define LWCC97_SET_GLOBAL_RENDER_ENABLE_B                                                                  0x0134
#define LWCC97_SET_GLOBAL_RENDER_ENABLE_B_OFFSET_LOWER                                                       31:0

#define LWCC97_SET_GLOBAL_RENDER_ENABLE_C                                                                  0x0138
#define LWCC97_SET_GLOBAL_RENDER_ENABLE_C_MODE                                                                2:0
#define LWCC97_SET_GLOBAL_RENDER_ENABLE_C_MODE_FALSE                                                   0x00000000
#define LWCC97_SET_GLOBAL_RENDER_ENABLE_C_MODE_TRUE                                                    0x00000001
#define LWCC97_SET_GLOBAL_RENDER_ENABLE_C_MODE_CONDITIONAL                                             0x00000002
#define LWCC97_SET_GLOBAL_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL                                         0x00000003
#define LWCC97_SET_GLOBAL_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL                                     0x00000004

#define LWCC97_SEND_GO_IDLE                                                                                0x013c
#define LWCC97_SEND_GO_IDLE_V                                                                                31:0

#define LWCC97_PM_TRIGGER                                                                                  0x0140
#define LWCC97_PM_TRIGGER_V                                                                                  31:0

#define LWCC97_PM_TRIGGER_WFI                                                                              0x0144
#define LWCC97_PM_TRIGGER_WFI_V                                                                              31:0

#define LWCC97_FE_ATOMIC_SEQUENCE_BEGIN                                                                    0x0148
#define LWCC97_FE_ATOMIC_SEQUENCE_BEGIN_V                                                                    31:0

#define LWCC97_FE_ATOMIC_SEQUENCE_END                                                                      0x014c
#define LWCC97_FE_ATOMIC_SEQUENCE_END_V                                                                      31:0

#define LWCC97_SET_INSTRUMENTATION_METHOD_HEADER                                                           0x0150
#define LWCC97_SET_INSTRUMENTATION_METHOD_HEADER_V                                                           31:0

#define LWCC97_SET_INSTRUMENTATION_METHOD_DATA                                                             0x0154
#define LWCC97_SET_INSTRUMENTATION_METHOD_DATA_V                                                             31:0

#define LWCC97_SET_REPORT_SEMAPHORE_PAYLOAD_LOWER                                                          0x0158
#define LWCC97_SET_REPORT_SEMAPHORE_PAYLOAD_LOWER_PAYLOAD_LOWER                                              31:0

#define LWCC97_SET_REPORT_SEMAPHORE_PAYLOAD_UPPER                                                          0x015c
#define LWCC97_SET_REPORT_SEMAPHORE_PAYLOAD_UPPER_PAYLOAD_UPPER                                              31:0

#define LWCC97_SET_REPORT_SEMAPHORE_ADDRESS_LOWER                                                          0x0160
#define LWCC97_SET_REPORT_SEMAPHORE_ADDRESS_LOWER_LOWER                                                      31:0

#define LWCC97_SET_REPORT_SEMAPHORE_ADDRESS_UPPER                                                          0x0164
#define LWCC97_SET_REPORT_SEMAPHORE_ADDRESS_UPPER_UPPER                                                      24:0

#define LWCC97_REPORT_SEMAPHORE_EXELWTE                                                                    0x0168
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_OPERATION                                                             1:0
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_OPERATION_RELEASE                                              0x00000000
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_OPERATION_ACQUIRE                                              0x00000001
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_OPERATION_REPORT_ONLY                                          0x00000002
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_OPERATION_TRAP                                                 0x00000003
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_PIPELINE_LOCATION                                                     5:2
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_PIPELINE_LOCATION_NONE                                         0x00000000
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_PIPELINE_LOCATION_DATA_ASSEMBLER                               0x00000001
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_PIPELINE_LOCATION_VERTEX_SHADER                                0x00000002
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_PIPELINE_LOCATION_TESSELATION_INIT_SHADER                      0x00000008
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_PIPELINE_LOCATION_TESSELATION_SHADER                           0x00000009
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_PIPELINE_LOCATION_GEOMETRY_SHADER                              0x00000006
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_PIPELINE_LOCATION_STREAMING_OUTPUT                             0x00000005
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_PIPELINE_LOCATION_VPC                                          0x00000004
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_PIPELINE_LOCATION_ZLWLL                                        0x00000007
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_PIPELINE_LOCATION_PIXEL_SHADER                                 0x0000000A
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_PIPELINE_LOCATION_DEPTH_TEST                                   0x0000000C
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_PIPELINE_LOCATION_ALL                                          0x0000000F
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_AWAKEN_ENABLE                                                         6:6
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_AWAKEN_ENABLE_FALSE                                            0x00000000
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_AWAKEN_ENABLE_TRUE                                             0x00000001
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT                                                               11:7
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_NONE                                                    0x00000000
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_DA_VERTICES_GENERATED                                   0x00000001
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_DA_PRIMITIVES_GENERATED                                 0x00000003
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_VS_ILWOCATIONS                                          0x00000005
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_TI_ILWOCATIONS                                          0x0000001B
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_TS_ILWOCATIONS                                          0x0000001D
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_TS_PRIMITIVES_GENERATED                                 0x0000001F
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_GS_ILWOCATIONS                                          0x00000007
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_GS_PRIMITIVES_GENERATED                                 0x00000009
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_ALPHA_BETA_CLOCKS                                       0x00000004
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_SCG_CLOCKS                                              0x00000008
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_VTG_PRIMITIVES_OUT                                      0x00000012
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_TOTAL_STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED             0x0000001E
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_STREAMING_PRIMITIVES_SUCCEEDED                          0x0000000B
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_STREAMING_PRIMITIVES_NEEDED                             0x0000000D
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED             0x00000006
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_STREAMING_BYTE_COUNT                                    0x0000001A
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_CLIPPER_ILWOCATIONS                                     0x0000000F
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_CLIPPER_PRIMITIVES_GENERATED                            0x00000011
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_ZLWLL_STATS0                                            0x0000000A
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_ZLWLL_STATS1                                            0x0000000C
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_ZLWLL_STATS2                                            0x0000000E
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_ZLWLL_STATS3                                            0x00000010
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_PS_ILWOCATIONS                                          0x00000013
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_ZPASS_PIXEL_CNT                                         0x00000002
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_ZPASS_PIXEL_CNT64                                       0x00000015
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_TILED_ZPASS_PIXEL_CNT64                                 0x00000017
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_IEEE_CLEAN_COLOR_TARGET                                 0x00000018
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_IEEE_CLEAN_ZETA_TARGET                                  0x00000019
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_BOUNDING_RECTANGLE                                      0x0000001C
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REPORT_TIMESTAMP                                               0x00000014
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_STRUCTURE_SIZE                                                      14:13
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_STRUCTURE_SIZE_SEMAPHORE_FOUR_WORDS                            0x00000000
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_STRUCTURE_SIZE_SEMAPHORE_ONE_WORD                              0x00000001
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_STRUCTURE_SIZE_SEMAPHORE_TWO_WORDS                             0x00000002
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_SUB_REPORT                                                          17:15
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_FLUSH_DISABLE                                                       19:19
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_FLUSH_DISABLE_FALSE                                            0x00000000
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_FLUSH_DISABLE_TRUE                                             0x00000001
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_ROP_FLUSH_DISABLE                                                   18:18
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_ROP_FLUSH_DISABLE_FALSE                                        0x00000000
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_ROP_FLUSH_DISABLE_TRUE                                         0x00000001
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REDUCTION_ENABLE                                                    20:20
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REDUCTION_ENABLE_FALSE                                         0x00000000
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REDUCTION_ENABLE_TRUE                                          0x00000001
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REDUCTION_OP                                                        23:21
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REDUCTION_OP_RED_ADD                                           0x00000000
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REDUCTION_OP_RED_MIN                                           0x00000001
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REDUCTION_OP_RED_MAX                                           0x00000002
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REDUCTION_OP_RED_INC                                           0x00000003
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REDUCTION_OP_RED_DEC                                           0x00000004
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REDUCTION_OP_RED_AND                                           0x00000005
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REDUCTION_OP_RED_OR                                            0x00000006
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REDUCTION_OP_RED_XOR                                           0x00000007
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REDUCTION_FORMAT                                                    25:24
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REDUCTION_FORMAT_UNSIGNED                                      0x00000000
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_REDUCTION_FORMAT_SIGNED                                        0x00000001
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_PAYLOAD_SIZE64                                                      27:27
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_PAYLOAD_SIZE64_FALSE                                           0x00000000
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_PAYLOAD_SIZE64_TRUE                                            0x00000001
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_TRAP_TYPE                                                           29:28
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_TRAP_TYPE_TRAP_NONE                                            0x00000000
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_TRAP_TYPE_TRAP_UNCONDITIONAL                                   0x00000001
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_TRAP_TYPE_TRAP_CONDITIONAL                                     0x00000002
#define LWCC97_REPORT_SEMAPHORE_EXELWTE_TRAP_TYPE_TRAP_CONDITIONAL_EXT                                 0x00000003

#define LWCC97_LINE_LENGTH_IN                                                                              0x0180
#define LWCC97_LINE_LENGTH_IN_VALUE                                                                          31:0

#define LWCC97_LINE_COUNT                                                                                  0x0184
#define LWCC97_LINE_COUNT_VALUE                                                                              31:0

#define LWCC97_OFFSET_OUT_UPPER                                                                            0x0188
#define LWCC97_OFFSET_OUT_UPPER_VALUE                                                                        24:0

#define LWCC97_OFFSET_OUT                                                                                  0x018c
#define LWCC97_OFFSET_OUT_VALUE                                                                              31:0

#define LWCC97_PITCH_OUT                                                                                   0x0190
#define LWCC97_PITCH_OUT_VALUE                                                                               31:0

#define LWCC97_SET_DST_BLOCK_SIZE                                                                          0x0194
#define LWCC97_SET_DST_BLOCK_SIZE_WIDTH                                                                       3:0
#define LWCC97_SET_DST_BLOCK_SIZE_WIDTH_ONE_GOB                                                        0x00000000
#define LWCC97_SET_DST_BLOCK_SIZE_HEIGHT                                                                      7:4
#define LWCC97_SET_DST_BLOCK_SIZE_HEIGHT_ONE_GOB                                                       0x00000000
#define LWCC97_SET_DST_BLOCK_SIZE_HEIGHT_TWO_GOBS                                                      0x00000001
#define LWCC97_SET_DST_BLOCK_SIZE_HEIGHT_FOUR_GOBS                                                     0x00000002
#define LWCC97_SET_DST_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                                                    0x00000003
#define LWCC97_SET_DST_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                                                  0x00000004
#define LWCC97_SET_DST_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                                                0x00000005
#define LWCC97_SET_DST_BLOCK_SIZE_DEPTH                                                                      11:8
#define LWCC97_SET_DST_BLOCK_SIZE_DEPTH_ONE_GOB                                                        0x00000000
#define LWCC97_SET_DST_BLOCK_SIZE_DEPTH_TWO_GOBS                                                       0x00000001
#define LWCC97_SET_DST_BLOCK_SIZE_DEPTH_FOUR_GOBS                                                      0x00000002
#define LWCC97_SET_DST_BLOCK_SIZE_DEPTH_EIGHT_GOBS                                                     0x00000003
#define LWCC97_SET_DST_BLOCK_SIZE_DEPTH_SIXTEEN_GOBS                                                   0x00000004
#define LWCC97_SET_DST_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS                                                 0x00000005

#define LWCC97_SET_DST_WIDTH                                                                               0x0198
#define LWCC97_SET_DST_WIDTH_V                                                                               31:0

#define LWCC97_SET_DST_HEIGHT                                                                              0x019c
#define LWCC97_SET_DST_HEIGHT_V                                                                              31:0

#define LWCC97_SET_DST_DEPTH                                                                               0x01a0
#define LWCC97_SET_DST_DEPTH_V                                                                               31:0

#define LWCC97_SET_DST_LAYER                                                                               0x01a4
#define LWCC97_SET_DST_LAYER_V                                                                               31:0

#define LWCC97_SET_DST_ORIGIN_BYTES_X                                                                      0x01a8
#define LWCC97_SET_DST_ORIGIN_BYTES_X_V                                                                      20:0

#define LWCC97_SET_DST_ORIGIN_SAMPLES_Y                                                                    0x01ac
#define LWCC97_SET_DST_ORIGIN_SAMPLES_Y_V                                                                    16:0

#define LWCC97_LAUNCH_DMA                                                                                  0x01b0
#define LWCC97_LAUNCH_DMA_DST_MEMORY_LAYOUT                                                                   0:0
#define LWCC97_LAUNCH_DMA_DST_MEMORY_LAYOUT_BLOCKLINEAR                                                0x00000000
#define LWCC97_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH                                                      0x00000001
#define LWCC97_LAUNCH_DMA_COMPLETION_TYPE                                                                     5:4
#define LWCC97_LAUNCH_DMA_COMPLETION_TYPE_FLUSH_DISABLE                                                0x00000000
#define LWCC97_LAUNCH_DMA_COMPLETION_TYPE_FLUSH_ONLY                                                   0x00000001
#define LWCC97_LAUNCH_DMA_COMPLETION_TYPE_RELEASE_SEMAPHORE                                            0x00000002
#define LWCC97_LAUNCH_DMA_INTERRUPT_TYPE                                                                      9:8
#define LWCC97_LAUNCH_DMA_INTERRUPT_TYPE_NONE                                                          0x00000000
#define LWCC97_LAUNCH_DMA_INTERRUPT_TYPE_INTERRUPT                                                     0x00000001
#define LWCC97_LAUNCH_DMA_SEMAPHORE_STRUCT_SIZE                                                             12:12
#define LWCC97_LAUNCH_DMA_SEMAPHORE_STRUCT_SIZE_FOUR_WORDS                                             0x00000000
#define LWCC97_LAUNCH_DMA_SEMAPHORE_STRUCT_SIZE_ONE_WORD                                               0x00000001
#define LWCC97_LAUNCH_DMA_REDUCTION_ENABLE                                                                    1:1
#define LWCC97_LAUNCH_DMA_REDUCTION_ENABLE_FALSE                                                       0x00000000
#define LWCC97_LAUNCH_DMA_REDUCTION_ENABLE_TRUE                                                        0x00000001
#define LWCC97_LAUNCH_DMA_REDUCTION_OP                                                                      15:13
#define LWCC97_LAUNCH_DMA_REDUCTION_OP_RED_ADD                                                         0x00000000
#define LWCC97_LAUNCH_DMA_REDUCTION_OP_RED_MIN                                                         0x00000001
#define LWCC97_LAUNCH_DMA_REDUCTION_OP_RED_MAX                                                         0x00000002
#define LWCC97_LAUNCH_DMA_REDUCTION_OP_RED_INC                                                         0x00000003
#define LWCC97_LAUNCH_DMA_REDUCTION_OP_RED_DEC                                                         0x00000004
#define LWCC97_LAUNCH_DMA_REDUCTION_OP_RED_AND                                                         0x00000005
#define LWCC97_LAUNCH_DMA_REDUCTION_OP_RED_OR                                                          0x00000006
#define LWCC97_LAUNCH_DMA_REDUCTION_OP_RED_XOR                                                         0x00000007
#define LWCC97_LAUNCH_DMA_REDUCTION_FORMAT                                                                    3:2
#define LWCC97_LAUNCH_DMA_REDUCTION_FORMAT_UNSIGNED_32                                                 0x00000000
#define LWCC97_LAUNCH_DMA_REDUCTION_FORMAT_SIGNED_32                                                   0x00000001
#define LWCC97_LAUNCH_DMA_SYSMEMBAR_DISABLE                                                                   6:6
#define LWCC97_LAUNCH_DMA_SYSMEMBAR_DISABLE_FALSE                                                      0x00000000
#define LWCC97_LAUNCH_DMA_SYSMEMBAR_DISABLE_TRUE                                                       0x00000001

#define LWCC97_LOAD_INLINE_DATA                                                                            0x01b4
#define LWCC97_LOAD_INLINE_DATA_V                                                                            31:0

#define LWCC97_SET_I2M_SEMAPHORE_A                                                                         0x01dc
#define LWCC97_SET_I2M_SEMAPHORE_A_OFFSET_UPPER                                                              24:0

#define LWCC97_SET_I2M_SEMAPHORE_B                                                                         0x01e0
#define LWCC97_SET_I2M_SEMAPHORE_B_OFFSET_LOWER                                                              31:0

#define LWCC97_SET_I2M_SEMAPHORE_C                                                                         0x01e4
#define LWCC97_SET_I2M_SEMAPHORE_C_PAYLOAD                                                                   31:0

#define LWCC97_SET_MME_SWITCH_STATE                                                                        0x01ec
#define LWCC97_SET_MME_SWITCH_STATE_VALID                                                                     0:0
#define LWCC97_SET_MME_SWITCH_STATE_VALID_FALSE                                                        0x00000000
#define LWCC97_SET_MME_SWITCH_STATE_VALID_TRUE                                                         0x00000001
#define LWCC97_SET_MME_SWITCH_STATE_SAVE_MACRO                                                               11:4
#define LWCC97_SET_MME_SWITCH_STATE_RESTORE_MACRO                                                           19:12

#define LWCC97_SET_I2M_SPARE_NOOP00                                                                        0x01f0
#define LWCC97_SET_I2M_SPARE_NOOP00_V                                                                        31:0

#define LWCC97_SET_I2M_SPARE_NOOP01                                                                        0x01f4
#define LWCC97_SET_I2M_SPARE_NOOP01_V                                                                        31:0

#define LWCC97_SET_I2M_SPARE_NOOP02                                                                        0x01f8
#define LWCC97_SET_I2M_SPARE_NOOP02_V                                                                        31:0

#define LWCC97_SET_I2M_SPARE_NOOP03                                                                        0x01fc
#define LWCC97_SET_I2M_SPARE_NOOP03_V                                                                        31:0

#define LWCC97_RUN_DS_NOW                                                                                  0x0200
#define LWCC97_RUN_DS_NOW_V                                                                                  31:0

#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS                                                        0x0204
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD                           4:0
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD_INSTANTANEOUS             0x00000000
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__16                0x00000001
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__32                0x00000002
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__64                0x00000003
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__128               0x00000004
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__256               0x00000005
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__512               0x00000006
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__1024              0x00000007
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__2048              0x00000008
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__4096              0x00000009
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__8192              0x0000000A
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__16384             0x0000000B
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__32768             0x0000000C
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__65536             0x0000000D
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__131072             0x0000000E
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__262144             0x0000000F
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__524288             0x00000010
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__1048576             0x00000011
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__2097152             0x00000012
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD__4194304             0x00000013
#define LWCC97_SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS_ACLWMULATED_PRIM_AREA_THRESHOLD_LATEZ_ALWAYS             0x0000001F

#define LWCC97_SET_GS_MODE                                                                                 0x0208
#define LWCC97_SET_GS_MODE_TYPE                                                                               0:0
#define LWCC97_SET_GS_MODE_TYPE_ANY                                                                    0x00000000
#define LWCC97_SET_GS_MODE_TYPE_FAST_GS                                                                0x00000001

#define LWCC97_SET_ALIASED_LINE_WIDTH_ENABLE                                                               0x020c
#define LWCC97_SET_ALIASED_LINE_WIDTH_ENABLE_V                                                                0:0
#define LWCC97_SET_ALIASED_LINE_WIDTH_ENABLE_V_FALSE                                                   0x00000000
#define LWCC97_SET_ALIASED_LINE_WIDTH_ENABLE_V_TRUE                                                    0x00000001

#define LWCC97_SET_API_MANDATED_EARLY_Z                                                                    0x0210
#define LWCC97_SET_API_MANDATED_EARLY_Z_ENABLE                                                                0:0
#define LWCC97_SET_API_MANDATED_EARLY_Z_ENABLE_FALSE                                                   0x00000000
#define LWCC97_SET_API_MANDATED_EARLY_Z_ENABLE_TRUE                                                    0x00000001

#define LWCC97_SET_GS_DM_FIFO                                                                              0x0214
#define LWCC97_SET_GS_DM_FIFO_SIZE_RASTER_ON                                                                 12:0
#define LWCC97_SET_GS_DM_FIFO_SIZE_RASTER_OFF                                                               28:16
#define LWCC97_SET_GS_DM_FIFO_SPILL_ENABLED                                                                 31:31
#define LWCC97_SET_GS_DM_FIFO_SPILL_ENABLED_FALSE                                                      0x00000000
#define LWCC97_SET_GS_DM_FIFO_SPILL_ENABLED_TRUE                                                       0x00000001

#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_PREFETCH_READ_REQUESTS                                         0x0218
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_PREFETCH_READ_REQUESTS_POLICY                                     5:4
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_PREFETCH_READ_REQUESTS_POLICY_EVICT_FIRST                  0x00000000
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_PREFETCH_READ_REQUESTS_POLICY_EVICT_NORMAL                 0x00000001
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_PREFETCH_READ_REQUESTS_POLICY_EVICT_LAST                   0x00000002
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_PREFETCH_READ_REQUESTS_ZROP_POLICY                                1:0
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_PREFETCH_READ_REQUESTS_ZROP_POLICY_EVICT_FIRST             0x00000000
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_PREFETCH_READ_REQUESTS_ZROP_POLICY_EVICT_NORMAL             0x00000001
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_PREFETCH_READ_REQUESTS_ZROP_POLICY_EVICT_LAST              0x00000002

#define LWCC97_ILWALIDATE_SHADER_CACHES                                                                    0x021c
#define LWCC97_ILWALIDATE_SHADER_CACHES_INSTRUCTION                                                           0:0
#define LWCC97_ILWALIDATE_SHADER_CACHES_INSTRUCTION_FALSE                                              0x00000000
#define LWCC97_ILWALIDATE_SHADER_CACHES_INSTRUCTION_TRUE                                               0x00000001
#define LWCC97_ILWALIDATE_SHADER_CACHES_DATA                                                                  4:4
#define LWCC97_ILWALIDATE_SHADER_CACHES_DATA_FALSE                                                     0x00000000
#define LWCC97_ILWALIDATE_SHADER_CACHES_DATA_TRUE                                                      0x00000001
#define LWCC97_ILWALIDATE_SHADER_CACHES_CONSTANT                                                            12:12
#define LWCC97_ILWALIDATE_SHADER_CACHES_CONSTANT_FALSE                                                 0x00000000
#define LWCC97_ILWALIDATE_SHADER_CACHES_CONSTANT_TRUE                                                  0x00000001
#define LWCC97_ILWALIDATE_SHADER_CACHES_LOCKS                                                                 1:1
#define LWCC97_ILWALIDATE_SHADER_CACHES_LOCKS_FALSE                                                    0x00000000
#define LWCC97_ILWALIDATE_SHADER_CACHES_LOCKS_TRUE                                                     0x00000001
#define LWCC97_ILWALIDATE_SHADER_CACHES_FLUSH_DATA                                                            2:2
#define LWCC97_ILWALIDATE_SHADER_CACHES_FLUSH_DATA_FALSE                                               0x00000000
#define LWCC97_ILWALIDATE_SHADER_CACHES_FLUSH_DATA_TRUE                                                0x00000001

#define LWCC97_SET_INSTANCE_COUNT                                                                          0x0220
#define LWCC97_SET_INSTANCE_COUNT_V                                                                          31:0

#define LWCC97_SET_POSITION_W_SCALED_OFFSET_ENABLE                                                         0x0224
#define LWCC97_SET_POSITION_W_SCALED_OFFSET_ENABLE_ENABLE                                                     0:0
#define LWCC97_SET_POSITION_W_SCALED_OFFSET_ENABLE_ENABLE_FALSE                                        0x00000000
#define LWCC97_SET_POSITION_W_SCALED_OFFSET_ENABLE_ENABLE_TRUE                                         0x00000001

#define LWCC97_SET_MULTI_VIEW                                                                              0x0228
#define LWCC97_SET_MULTI_VIEW_VIEW_COUNT                                                                      3:0
#define LWCC97_SET_MULTI_VIEW_RENDER_TARGET_INDEX_OFFSET                                                     14:4
#define LWCC97_SET_MULTI_VIEW_INDEPENDENT_VIEWPORT_MASK_ENABLE                                              16:16
#define LWCC97_SET_MULTI_VIEW_INDEPENDENT_VIEWPORT_MASK_ENABLE_FALSE                                   0x00000000
#define LWCC97_SET_MULTI_VIEW_INDEPENDENT_VIEWPORT_MASK_ENABLE_TRUE                                    0x00000001
#define LWCC97_SET_MULTI_VIEW_SECOND_X_ONLY                                                                 15:15
#define LWCC97_SET_MULTI_VIEW_SECOND_X_ONLY_FALSE                                                      0x00000000
#define LWCC97_SET_MULTI_VIEW_SECOND_X_ONLY_TRUE                                                       0x00000001

#define LWCC97_SET_GO_IDLE_TIMEOUT                                                                         0x022c
#define LWCC97_SET_GO_IDLE_TIMEOUT_V                                                                         31:0

#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL                                                             0x0230
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_EXTRA_PRIM_BLOAT                                               3:0
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_EXTRA_PRIM_BLOAT_BLOAT_0                                0x00000000
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_EXTRA_PRIM_BLOAT_BLOAT_25                               0x00000001
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_EXTRA_PRIM_BLOAT_BLOAT_50                               0x00000002
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_EXTRA_PRIM_BLOAT_BLOAT_75                               0x00000003
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_EXTRA_PRIM_BLOAT_BLOAT_PRI                              0x00000004
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_COPY_INNER_TO_OUTER                                            4:4
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_COPY_INNER_TO_OUTER_FALSE                               0x00000000
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_COPY_INNER_TO_OUTER_TRUE                                0x00000001
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_TRIANGLE_SNAP_MODE                                            10:8
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_TRIANGLE_SNAP_MODE_MODE_POST_SNAP                       0x00000000
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_TRIANGLE_SNAP_MODE_MODE_PRE_SNAP                        0x00000001
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_TRIANGLE_SNAP_MODE_MODE_PRI                             0x00000002
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_LINE_AND_POINT_SNAP_MODE                                     14:12
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_LINE_AND_POINT_SNAP_MODE_MODE_POST_SNAP                 0x00000000
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_LINE_AND_POINT_SNAP_MODE_MODE_PRE_SNAP                  0x00000001
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_LINE_AND_POINT_SNAP_MODE_MODE_PRI                       0x00000002
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_UNCERTAINTY_REGION_SIZE                                      19:16
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_UNCERTAINTY_REGION_SIZE_SIZE_64                         0x00000000
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_UNCERTAINTY_REGION_SIZE_SIZE_128                        0x00000001
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_UNCERTAINTY_REGION_SIZE_SIZE_256                        0x00000002
#define LWCC97_SET_CONSERVATIVE_RASTER_CONTROL_UNCERTAINTY_REGION_SIZE_SIZE_512                        0x00000003

#define LWCC97_SET_MME_VERSION                                                                             0x0234
#define LWCC97_SET_MME_VERSION_MAJOR                                                                          7:0

#define LWCC97_SET_INDEX_BUFFER_SIZE_A                                                                     0x0238
#define LWCC97_SET_INDEX_BUFFER_SIZE_A_UPPER                                                                  7:0

#define LWCC97_SET_INDEX_BUFFER_SIZE_B                                                                     0x023c
#define LWCC97_SET_INDEX_BUFFER_SIZE_B_LOWER                                                                 31:0

#define LWCC97_SET_ROOT_TABLE_VISIBILITY(i)                                                        (0x0240+(i)*4)
#define LWCC97_SET_ROOT_TABLE_VISIBILITY_BINDING_GROUP0_ENABLE                                                1:0
#define LWCC97_SET_ROOT_TABLE_VISIBILITY_BINDING_GROUP1_ENABLE                                                5:4
#define LWCC97_SET_ROOT_TABLE_VISIBILITY_BINDING_GROUP2_ENABLE                                                9:8
#define LWCC97_SET_ROOT_TABLE_VISIBILITY_BINDING_GROUP3_ENABLE                                              13:12
#define LWCC97_SET_ROOT_TABLE_VISIBILITY_BINDING_GROUP4_ENABLE                                              17:16

#define LWCC97_SET_DRAW_CONTROL_A                                                                          0x0260
#define LWCC97_SET_DRAW_CONTROL_A_TOPOLOGY                                                                    3:0
#define LWCC97_SET_DRAW_CONTROL_A_TOPOLOGY_POINTS                                                      0x00000000
#define LWCC97_SET_DRAW_CONTROL_A_TOPOLOGY_LINES                                                       0x00000001
#define LWCC97_SET_DRAW_CONTROL_A_TOPOLOGY_LINE_LOOP                                                   0x00000002
#define LWCC97_SET_DRAW_CONTROL_A_TOPOLOGY_LINE_STRIP                                                  0x00000003
#define LWCC97_SET_DRAW_CONTROL_A_TOPOLOGY_TRIANGLES                                                   0x00000004
#define LWCC97_SET_DRAW_CONTROL_A_TOPOLOGY_TRIANGLE_STRIP                                              0x00000005
#define LWCC97_SET_DRAW_CONTROL_A_TOPOLOGY_TRIANGLE_FAN                                                0x00000006
#define LWCC97_SET_DRAW_CONTROL_A_TOPOLOGY_QUADS                                                       0x00000007
#define LWCC97_SET_DRAW_CONTROL_A_TOPOLOGY_QUAD_STRIP                                                  0x00000008
#define LWCC97_SET_DRAW_CONTROL_A_TOPOLOGY_POLYGON                                                     0x00000009
#define LWCC97_SET_DRAW_CONTROL_A_TOPOLOGY_LINELIST_ADJCY                                              0x0000000A
#define LWCC97_SET_DRAW_CONTROL_A_TOPOLOGY_LINESTRIP_ADJCY                                             0x0000000B
#define LWCC97_SET_DRAW_CONTROL_A_TOPOLOGY_TRIANGLELIST_ADJCY                                          0x0000000C
#define LWCC97_SET_DRAW_CONTROL_A_TOPOLOGY_TRIANGLESTRIP_ADJCY                                         0x0000000D
#define LWCC97_SET_DRAW_CONTROL_A_TOPOLOGY_PATCH                                                       0x0000000E
#define LWCC97_SET_DRAW_CONTROL_A_PRIMITIVE_ID                                                                4:4
#define LWCC97_SET_DRAW_CONTROL_A_PRIMITIVE_ID_FIRST                                                   0x00000000
#define LWCC97_SET_DRAW_CONTROL_A_PRIMITIVE_ID_UNCHANGED                                               0x00000001
#define LWCC97_SET_DRAW_CONTROL_A_INSTANCE_ID                                                                 6:5
#define LWCC97_SET_DRAW_CONTROL_A_INSTANCE_ID_FIRST                                                    0x00000000
#define LWCC97_SET_DRAW_CONTROL_A_INSTANCE_ID_SUBSEQUENT                                               0x00000001
#define LWCC97_SET_DRAW_CONTROL_A_INSTANCE_ID_UNCHANGED                                                0x00000002
#define LWCC97_SET_DRAW_CONTROL_A_SPLIT_MODE                                                                  8:7
#define LWCC97_SET_DRAW_CONTROL_A_SPLIT_MODE_NORMAL_BEGIN_NORMAL_END                                   0x00000000
#define LWCC97_SET_DRAW_CONTROL_A_SPLIT_MODE_NORMAL_BEGIN_OPEN_END                                     0x00000001
#define LWCC97_SET_DRAW_CONTROL_A_SPLIT_MODE_OPEN_BEGIN_OPEN_END                                       0x00000002
#define LWCC97_SET_DRAW_CONTROL_A_SPLIT_MODE_OPEN_BEGIN_NORMAL_END                                     0x00000003
#define LWCC97_SET_DRAW_CONTROL_A_INSTANCE_ITERATE_ENABLE                                                     9:9
#define LWCC97_SET_DRAW_CONTROL_A_INSTANCE_ITERATE_ENABLE_FALSE                                        0x00000000
#define LWCC97_SET_DRAW_CONTROL_A_INSTANCE_ITERATE_ENABLE_TRUE                                         0x00000001
#define LWCC97_SET_DRAW_CONTROL_A_IGNORE_GLOBAL_BASE_VERTEX_INDEX                                           10:10
#define LWCC97_SET_DRAW_CONTROL_A_IGNORE_GLOBAL_BASE_VERTEX_INDEX_FALSE                                0x00000000
#define LWCC97_SET_DRAW_CONTROL_A_IGNORE_GLOBAL_BASE_VERTEX_INDEX_TRUE                                 0x00000001
#define LWCC97_SET_DRAW_CONTROL_A_IGNORE_GLOBAL_BASE_INSTANCE_INDEX                                         11:11
#define LWCC97_SET_DRAW_CONTROL_A_IGNORE_GLOBAL_BASE_INSTANCE_INDEX_FALSE                              0x00000000
#define LWCC97_SET_DRAW_CONTROL_A_IGNORE_GLOBAL_BASE_INSTANCE_INDEX_TRUE                               0x00000001

#define LWCC97_SET_DRAW_CONTROL_B                                                                          0x0264
#define LWCC97_SET_DRAW_CONTROL_B_INSTANCE_COUNT                                                             31:0

#define LWCC97_DRAW_INDEX_BUFFER_BEGIN_END_A                                                               0x0268
#define LWCC97_DRAW_INDEX_BUFFER_BEGIN_END_A_FIRST                                                           31:0

#define LWCC97_DRAW_INDEX_BUFFER_BEGIN_END_B                                                               0x026c
#define LWCC97_DRAW_INDEX_BUFFER_BEGIN_END_B_COUNT                                                           31:0

#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_A                                                               0x0270
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_A_START                                                           31:0

#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_B                                                               0x0274
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_B_COUNT                                                           31:0

#define LWCC97_SET_CHECKERBOARD_CLIP_CONTROL                                                               0x0278
#define LWCC97_SET_CHECKERBOARD_CLIP_CONTROL_ENABLE                                                           0:0
#define LWCC97_SET_CHECKERBOARD_CLIP_CONTROL_ENABLE_FALSE                                              0x00000000
#define LWCC97_SET_CHECKERBOARD_CLIP_CONTROL_ENABLE_TRUE                                               0x00000001
#define LWCC97_SET_CHECKERBOARD_CLIP_CONTROL_WIDTH_LOG2                                                       7:4
#define LWCC97_SET_CHECKERBOARD_CLIP_CONTROL_HEIGHT_LOG2                                                     11:8

#define LWCC97_ILWALIDATE_RASTER_CACHE_NO_WFI                                                              0x027c
#define LWCC97_ILWALIDATE_RASTER_CACHE_NO_WFI_V                                                               0:0

#define LWCC97_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(i)                                             (0x0280+(i)*4)
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER_V                                                        31:0

#define LWCC97_SET_SHADER_LOCAL_MEMORY_THROTTLE                                                            0x02b4
#define LWCC97_SET_SHADER_LOCAL_MEMORY_THROTTLE_ENABLED                                                       0:0
#define LWCC97_SET_SHADER_LOCAL_MEMORY_THROTTLE_ENABLED_FALSE                                          0x00000000
#define LWCC97_SET_SHADER_LOCAL_MEMORY_THROTTLE_ENABLED_TRUE                                           0x00000001

#define LWCC97_SET_COLOR_RENDER_TO_ZETA_SURFACE                                                            0x02b8
#define LWCC97_SET_COLOR_RENDER_TO_ZETA_SURFACE_V                                                             0:0
#define LWCC97_SET_COLOR_RENDER_TO_ZETA_SURFACE_V_FALSE                                                0x00000000
#define LWCC97_SET_COLOR_RENDER_TO_ZETA_SURFACE_V_TRUE                                                 0x00000001

#define LWCC97_SET_ZLWLL_VISIBLE_PRIM_OPTIMIZATION                                                         0x02bc
#define LWCC97_SET_ZLWLL_VISIBLE_PRIM_OPTIMIZATION_V                                                          0:0
#define LWCC97_SET_ZLWLL_VISIBLE_PRIM_OPTIMIZATION_V_FALSE                                             0x00000000
#define LWCC97_SET_ZLWLL_VISIBLE_PRIM_OPTIMIZATION_V_TRUE                                              0x00000001

#define LWCC97_SET_VPC_TESTING_CONTROL                                                                     0x02c0
#define LWCC97_SET_VPC_TESTING_CONTROL_PER_PRIMITIVE_FRUSTUM_AND_USER_CLIP_PLANE_LWLLING_ENABLE                    0:0
#define LWCC97_SET_VPC_TESTING_CONTROL_PER_PRIMITIVE_FRUSTUM_AND_USER_CLIP_PLANE_LWLLING_ENABLE_FALSE             0x00000000
#define LWCC97_SET_VPC_TESTING_CONTROL_PER_PRIMITIVE_FRUSTUM_AND_USER_CLIP_PLANE_LWLLING_ENABLE_TRUE             0x00000001

#define LWCC97_SET_CHECKERBOARD_CLIP_OWNERSHIP                                                             0x02c4
#define LWCC97_SET_CHECKERBOARD_CLIP_OWNERSHIP_MASK                                                          31:0

#define LWCC97_INCREMENT_SYNC_POINT                                                                        0x02c8
#define LWCC97_INCREMENT_SYNC_POINT_INDEX                                                                    11:0
#define LWCC97_INCREMENT_SYNC_POINT_CLEAN_L2                                                                16:16
#define LWCC97_INCREMENT_SYNC_POINT_CLEAN_L2_FALSE                                                     0x00000000
#define LWCC97_INCREMENT_SYNC_POINT_CLEAN_L2_TRUE                                                      0x00000001
#define LWCC97_INCREMENT_SYNC_POINT_CONDITION                                                               20:20
#define LWCC97_INCREMENT_SYNC_POINT_CONDITION_STREAM_OUT_WRITES_DONE                                   0x00000000
#define LWCC97_INCREMENT_SYNC_POINT_CONDITION_ROP_WRITES_DONE                                          0x00000001

#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_TABLE_SELECT                                                0x02cc
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_TABLE_SELECT_SOURCE                                            1:0
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_TABLE_SELECT_SOURCE_FROM_VIEWPORT_INDEX                 0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_TABLE_SELECT_SOURCE_FROM_VPRS_TABLE_INDEX               0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_TABLE_SELECT_SOURCE_FROM_CONSTANT                       0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_TABLE_SELECT_SOURCE_RESERVED                            0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_TABLE_SELECT_SOURCE_CONSTANT_VALUE                             7:4

#define LWCC97_SET_ROOT_TABLE_PREFETCH                                                                     0x02d0
#define LWCC97_SET_ROOT_TABLE_PREFETCH_STAGE_ENABLES                                                          5:0

#define LWCC97_FLUSH_AND_ILWALIDATE_ROP_MINI_CACHE                                                         0x02d4
#define LWCC97_FLUSH_AND_ILWALIDATE_ROP_MINI_CACHE_V                                                          0:0

#define LWCC97_SET_SURFACE_CLIP_ID_BLOCK_SIZE                                                              0x02d8
#define LWCC97_SET_SURFACE_CLIP_ID_BLOCK_SIZE_WIDTH                                                           3:0
#define LWCC97_SET_SURFACE_CLIP_ID_BLOCK_SIZE_WIDTH_ONE_GOB                                            0x00000000
#define LWCC97_SET_SURFACE_CLIP_ID_BLOCK_SIZE_HEIGHT                                                          7:4
#define LWCC97_SET_SURFACE_CLIP_ID_BLOCK_SIZE_HEIGHT_ONE_GOB                                           0x00000000
#define LWCC97_SET_SURFACE_CLIP_ID_BLOCK_SIZE_HEIGHT_TWO_GOBS                                          0x00000001
#define LWCC97_SET_SURFACE_CLIP_ID_BLOCK_SIZE_HEIGHT_FOUR_GOBS                                         0x00000002
#define LWCC97_SET_SURFACE_CLIP_ID_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                                        0x00000003
#define LWCC97_SET_SURFACE_CLIP_ID_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                                      0x00000004
#define LWCC97_SET_SURFACE_CLIP_ID_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                                    0x00000005
#define LWCC97_SET_SURFACE_CLIP_ID_BLOCK_SIZE_DEPTH                                                          11:8
#define LWCC97_SET_SURFACE_CLIP_ID_BLOCK_SIZE_DEPTH_ONE_GOB                                            0x00000000

#define LWCC97_SET_ALPHA_CIRLWLAR_BUFFER_SIZE                                                              0x02dc
#define LWCC97_SET_ALPHA_CIRLWLAR_BUFFER_SIZE_CACHE_LINES_PER_SM                                             13:0

#define LWCC97_DECOMPRESS_SURFACE                                                                          0x02e0
#define LWCC97_DECOMPRESS_SURFACE_MRT_SELECT                                                                  2:0
#define LWCC97_DECOMPRESS_SURFACE_RT_ARRAY_INDEX                                                             19:4

#define LWCC97_SET_ZLWLL_ROP_BYPASS                                                                        0x02e4
#define LWCC97_SET_ZLWLL_ROP_BYPASS_ENABLE                                                                    0:0
#define LWCC97_SET_ZLWLL_ROP_BYPASS_ENABLE_FALSE                                                       0x00000000
#define LWCC97_SET_ZLWLL_ROP_BYPASS_ENABLE_TRUE                                                        0x00000001
#define LWCC97_SET_ZLWLL_ROP_BYPASS_NO_STALL                                                                  4:4
#define LWCC97_SET_ZLWLL_ROP_BYPASS_NO_STALL_FALSE                                                     0x00000000
#define LWCC97_SET_ZLWLL_ROP_BYPASS_NO_STALL_TRUE                                                      0x00000001
#define LWCC97_SET_ZLWLL_ROP_BYPASS_LWLL_EVERYTHING                                                           8:8
#define LWCC97_SET_ZLWLL_ROP_BYPASS_LWLL_EVERYTHING_FALSE                                              0x00000000
#define LWCC97_SET_ZLWLL_ROP_BYPASS_LWLL_EVERYTHING_TRUE                                               0x00000001
#define LWCC97_SET_ZLWLL_ROP_BYPASS_THRESHOLD                                                               15:12

#define LWCC97_SET_ZLWLL_SUBREGION                                                                         0x02e8
#define LWCC97_SET_ZLWLL_SUBREGION_ENABLE                                                                     0:0
#define LWCC97_SET_ZLWLL_SUBREGION_ENABLE_FALSE                                                        0x00000000
#define LWCC97_SET_ZLWLL_SUBREGION_ENABLE_TRUE                                                         0x00000001
#define LWCC97_SET_ZLWLL_SUBREGION_NORMALIZED_ALIQUOTS                                                       27:4

#define LWCC97_SET_RASTER_BOUNDING_BOX                                                                     0x02ec
#define LWCC97_SET_RASTER_BOUNDING_BOX_MODE                                                                   0:0
#define LWCC97_SET_RASTER_BOUNDING_BOX_MODE_BOUNDING_BOX                                               0x00000000
#define LWCC97_SET_RASTER_BOUNDING_BOX_MODE_FULL_VIEWPORT                                              0x00000001
#define LWCC97_SET_RASTER_BOUNDING_BOX_PAD                                                                   11:4

#define LWCC97_PEER_SEMAPHORE_RELEASE                                                                      0x02f0
#define LWCC97_PEER_SEMAPHORE_RELEASE_V                                                                      31:0

#define LWCC97_SET_ITERATED_BLEND_OPTIMIZATION                                                             0x02f4
#define LWCC97_SET_ITERATED_BLEND_OPTIMIZATION_NOOP                                                           1:0
#define LWCC97_SET_ITERATED_BLEND_OPTIMIZATION_NOOP_NEVER                                              0x00000000
#define LWCC97_SET_ITERATED_BLEND_OPTIMIZATION_NOOP_SOURCE_RGBA_0000                                   0x00000001
#define LWCC97_SET_ITERATED_BLEND_OPTIMIZATION_NOOP_SOURCE_ALPHA_0                                     0x00000002
#define LWCC97_SET_ITERATED_BLEND_OPTIMIZATION_NOOP_SOURCE_RGBA_0001                                   0x00000003

#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION                                                              0x02f8
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_SUBREGION_ID                                                    7:0
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_ALIQUOTS                                                       23:8
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_FORMAT                                                        27:24
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_FORMAT_Z_16X16X2_4X4                                     0x00000000
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_FORMAT_ZS_16X16_4X4                                      0x00000001
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_FORMAT_Z_16X16_4X2                                       0x00000002
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_FORMAT_Z_16X16_2X4                                       0x00000003
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_FORMAT_Z_16X8_4X4                                        0x00000004
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_FORMAT_Z_8X8_4X2                                         0x00000005
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_FORMAT_Z_8X8_2X4                                         0x00000006
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_FORMAT_Z_16X16_4X8                                       0x00000007
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_FORMAT_Z_4X8_2X2                                         0x00000008
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_FORMAT_ZS_16X8_4X2                                       0x00000009
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_FORMAT_ZS_16X8_2X4                                       0x0000000A
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_FORMAT_ZS_8X8_2X2                                        0x0000000B
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_FORMAT_Z_4X8_1X1                                         0x0000000C
#define LWCC97_SET_ZLWLL_SUBREGION_ALLOCATION_FORMAT_NONE                                              0x0000000F

#define LWCC97_ASSIGN_ZLWLL_SUBREGIONS                                                                     0x02fc
#define LWCC97_ASSIGN_ZLWLL_SUBREGIONS_ALGORITHM                                                              1:0
#define LWCC97_ASSIGN_ZLWLL_SUBREGIONS_ALGORITHM_Static                                                0x00000000
#define LWCC97_ASSIGN_ZLWLL_SUBREGIONS_ALGORITHM_Adaptive                                              0x00000001

#define LWCC97_SET_PS_OUTPUT_SAMPLE_MASK_USAGE                                                             0x0300
#define LWCC97_SET_PS_OUTPUT_SAMPLE_MASK_USAGE_ENABLE                                                         0:0
#define LWCC97_SET_PS_OUTPUT_SAMPLE_MASK_USAGE_ENABLE_FALSE                                            0x00000000
#define LWCC97_SET_PS_OUTPUT_SAMPLE_MASK_USAGE_ENABLE_TRUE                                             0x00000001
#define LWCC97_SET_PS_OUTPUT_SAMPLE_MASK_USAGE_QUALIFY_BY_ANTI_ALIAS_ENABLE                                   1:1
#define LWCC97_SET_PS_OUTPUT_SAMPLE_MASK_USAGE_QUALIFY_BY_ANTI_ALIAS_ENABLE_DISABLE                    0x00000000
#define LWCC97_SET_PS_OUTPUT_SAMPLE_MASK_USAGE_QUALIFY_BY_ANTI_ALIAS_ENABLE_ENABLE                     0x00000001

#define LWCC97_DRAW_ZERO_INDEX                                                                             0x0304
#define LWCC97_DRAW_ZERO_INDEX_COUNT                                                                         31:0

#define LWCC97_SET_L1_CONFIGURATION                                                                        0x0308
#define LWCC97_SET_L1_CONFIGURATION_DIRECTLY_ADDRESSABLE_MEMORY                                               2:0
#define LWCC97_SET_L1_CONFIGURATION_DIRECTLY_ADDRESSABLE_MEMORY_SIZE_16KB                              0x00000001
#define LWCC97_SET_L1_CONFIGURATION_DIRECTLY_ADDRESSABLE_MEMORY_SIZE_48KB                              0x00000003

#define LWCC97_SET_RENDER_ENABLE_CONTROL                                                                   0x030c
#define LWCC97_SET_RENDER_ENABLE_CONTROL_CONDITIONAL_LOAD_CONSTANT_BUFFER                                     0:0
#define LWCC97_SET_RENDER_ENABLE_CONTROL_CONDITIONAL_LOAD_CONSTANT_BUFFER_FALSE                        0x00000000
#define LWCC97_SET_RENDER_ENABLE_CONTROL_CONDITIONAL_LOAD_CONSTANT_BUFFER_TRUE                         0x00000001

#define LWCC97_SET_SPA_VERSION                                                                             0x0310
#define LWCC97_SET_SPA_VERSION_MINOR                                                                          7:0
#define LWCC97_SET_SPA_VERSION_MAJOR                                                                         15:8

#define LWCC97_SET_TIMESLICE_BATCH_LIMIT                                                                   0x0314
#define LWCC97_SET_TIMESLICE_BATCH_LIMIT_BATCH_LIMIT                                                         15:0

#define LWCC97_SET_SNAP_GRID_LINE                                                                          0x0318
#define LWCC97_SET_SNAP_GRID_LINE_LOCATIONS_PER_PIXEL                                                         3:0
#define LWCC97_SET_SNAP_GRID_LINE_LOCATIONS_PER_PIXEL__2X2                                             0x00000001
#define LWCC97_SET_SNAP_GRID_LINE_LOCATIONS_PER_PIXEL__4X4                                             0x00000002
#define LWCC97_SET_SNAP_GRID_LINE_LOCATIONS_PER_PIXEL__8X8                                             0x00000003
#define LWCC97_SET_SNAP_GRID_LINE_LOCATIONS_PER_PIXEL__16X16                                           0x00000004
#define LWCC97_SET_SNAP_GRID_LINE_LOCATIONS_PER_PIXEL__32X32                                           0x00000005
#define LWCC97_SET_SNAP_GRID_LINE_LOCATIONS_PER_PIXEL__64X64                                           0x00000006
#define LWCC97_SET_SNAP_GRID_LINE_LOCATIONS_PER_PIXEL__128X128                                         0x00000007
#define LWCC97_SET_SNAP_GRID_LINE_LOCATIONS_PER_PIXEL__256X256                                         0x00000008
#define LWCC97_SET_SNAP_GRID_LINE_ROUNDING_MODE                                                               8:8
#define LWCC97_SET_SNAP_GRID_LINE_ROUNDING_MODE_RTNE                                                   0x00000000
#define LWCC97_SET_SNAP_GRID_LINE_ROUNDING_MODE_TESLA                                                  0x00000001

#define LWCC97_SET_SNAP_GRID_NON_LINE                                                                      0x031c
#define LWCC97_SET_SNAP_GRID_NON_LINE_LOCATIONS_PER_PIXEL                                                     3:0
#define LWCC97_SET_SNAP_GRID_NON_LINE_LOCATIONS_PER_PIXEL__2X2                                         0x00000001
#define LWCC97_SET_SNAP_GRID_NON_LINE_LOCATIONS_PER_PIXEL__4X4                                         0x00000002
#define LWCC97_SET_SNAP_GRID_NON_LINE_LOCATIONS_PER_PIXEL__8X8                                         0x00000003
#define LWCC97_SET_SNAP_GRID_NON_LINE_LOCATIONS_PER_PIXEL__16X16                                       0x00000004
#define LWCC97_SET_SNAP_GRID_NON_LINE_LOCATIONS_PER_PIXEL__32X32                                       0x00000005
#define LWCC97_SET_SNAP_GRID_NON_LINE_LOCATIONS_PER_PIXEL__64X64                                       0x00000006
#define LWCC97_SET_SNAP_GRID_NON_LINE_LOCATIONS_PER_PIXEL__128X128                                     0x00000007
#define LWCC97_SET_SNAP_GRID_NON_LINE_LOCATIONS_PER_PIXEL__256X256                                     0x00000008
#define LWCC97_SET_SNAP_GRID_NON_LINE_ROUNDING_MODE                                                           8:8
#define LWCC97_SET_SNAP_GRID_NON_LINE_ROUNDING_MODE_RTNE                                               0x00000000
#define LWCC97_SET_SNAP_GRID_NON_LINE_ROUNDING_MODE_TESLA                                              0x00000001

#define LWCC97_SET_TESSELLATION_PARAMETERS                                                                 0x0320
#define LWCC97_SET_TESSELLATION_PARAMETERS_DOMAIN_TYPE                                                        1:0
#define LWCC97_SET_TESSELLATION_PARAMETERS_DOMAIN_TYPE_ISOLINE                                         0x00000000
#define LWCC97_SET_TESSELLATION_PARAMETERS_DOMAIN_TYPE_TRIANGLE                                        0x00000001
#define LWCC97_SET_TESSELLATION_PARAMETERS_DOMAIN_TYPE_QUAD                                            0x00000002
#define LWCC97_SET_TESSELLATION_PARAMETERS_SPACING                                                            5:4
#define LWCC97_SET_TESSELLATION_PARAMETERS_SPACING_INTEGER                                             0x00000000
#define LWCC97_SET_TESSELLATION_PARAMETERS_SPACING_FRACTIONAL_ODD                                      0x00000001
#define LWCC97_SET_TESSELLATION_PARAMETERS_SPACING_FRACTIONAL_EVEN                                     0x00000002
#define LWCC97_SET_TESSELLATION_PARAMETERS_OUTPUT_PRIMITIVES                                                  9:8
#define LWCC97_SET_TESSELLATION_PARAMETERS_OUTPUT_PRIMITIVES_POINTS                                    0x00000000
#define LWCC97_SET_TESSELLATION_PARAMETERS_OUTPUT_PRIMITIVES_LINES                                     0x00000001
#define LWCC97_SET_TESSELLATION_PARAMETERS_OUTPUT_PRIMITIVES_TRIANGLES_CW                              0x00000002
#define LWCC97_SET_TESSELLATION_PARAMETERS_OUTPUT_PRIMITIVES_TRIANGLES_CCW                             0x00000003

#define LWCC97_SET_TESSELLATION_LOD_U0_OR_DENSITY                                                          0x0324
#define LWCC97_SET_TESSELLATION_LOD_U0_OR_DENSITY_V                                                          31:0

#define LWCC97_SET_TESSELLATION_LOD_V0_OR_DETAIL                                                           0x0328
#define LWCC97_SET_TESSELLATION_LOD_V0_OR_DETAIL_V                                                           31:0

#define LWCC97_SET_TESSELLATION_LOD_U1_OR_W0                                                               0x032c
#define LWCC97_SET_TESSELLATION_LOD_U1_OR_W0_V                                                               31:0

#define LWCC97_SET_TESSELLATION_LOD_V1                                                                     0x0330
#define LWCC97_SET_TESSELLATION_LOD_V1_V                                                                     31:0

#define LWCC97_SET_TG_LOD_INTERIOR_U                                                                       0x0334
#define LWCC97_SET_TG_LOD_INTERIOR_U_V                                                                       31:0

#define LWCC97_SET_TG_LOD_INTERIOR_V                                                                       0x0338
#define LWCC97_SET_TG_LOD_INTERIOR_V_V                                                                       31:0

#define LWCC97_RESERVED_TG07                                                                               0x033c
#define LWCC97_RESERVED_TG07_V                                                                                0:0

#define LWCC97_RESERVED_TG08                                                                               0x0340
#define LWCC97_RESERVED_TG08_V                                                                                0:0

#define LWCC97_RESERVED_TG09                                                                               0x0344
#define LWCC97_RESERVED_TG09_V                                                                                0:0

#define LWCC97_RESERVED_TG10                                                                               0x0348
#define LWCC97_RESERVED_TG10_V                                                                                0:0

#define LWCC97_RESERVED_TG11                                                                               0x034c
#define LWCC97_RESERVED_TG11_V                                                                                0:0

#define LWCC97_RESERVED_TG12                                                                               0x0350
#define LWCC97_RESERVED_TG12_V                                                                                0:0

#define LWCC97_RESERVED_TG13                                                                               0x0354
#define LWCC97_RESERVED_TG13_V                                                                                0:0

#define LWCC97_RESERVED_TG14                                                                               0x0358
#define LWCC97_RESERVED_TG14_V                                                                                0:0

#define LWCC97_RESERVED_TG15                                                                               0x035c
#define LWCC97_RESERVED_TG15_V                                                                                0:0

#define LWCC97_SET_SUBTILING_PERF_KNOB_A                                                                   0x0360
#define LWCC97_SET_SUBTILING_PERF_KNOB_A_FRACTION_OF_SPM_REGISTER_FILE_PER_SUBTILE                            7:0
#define LWCC97_SET_SUBTILING_PERF_KNOB_A_FRACTION_OF_SPM_PIXEL_OUTPUT_BUFFER_PER_SUBTILE                     15:8
#define LWCC97_SET_SUBTILING_PERF_KNOB_A_FRACTION_OF_SPM_TRIANGLE_RAM_PER_SUBTILE                           23:16
#define LWCC97_SET_SUBTILING_PERF_KNOB_A_FRACTION_OF_MAX_QUADS_PER_SUBTILE                                  31:24

#define LWCC97_SET_SUBTILING_PERF_KNOB_B                                                                   0x0364
#define LWCC97_SET_SUBTILING_PERF_KNOB_B_FRACTION_OF_MAX_PRIMITIVES_PER_SUBTILE                               7:0

#define LWCC97_SET_SUBTILING_PERF_KNOB_C                                                                   0x0368
#define LWCC97_SET_SUBTILING_PERF_KNOB_C_RESERVED                                                             0:0

#define LWCC97_SET_ZLWLL_SUBREGION_TO_REPORT                                                               0x036c
#define LWCC97_SET_ZLWLL_SUBREGION_TO_REPORT_ENABLE                                                           0:0
#define LWCC97_SET_ZLWLL_SUBREGION_TO_REPORT_ENABLE_FALSE                                              0x00000000
#define LWCC97_SET_ZLWLL_SUBREGION_TO_REPORT_ENABLE_TRUE                                               0x00000001
#define LWCC97_SET_ZLWLL_SUBREGION_TO_REPORT_SUBREGION_ID                                                    11:4

#define LWCC97_SET_ZLWLL_SUBREGION_REPORT_TYPE                                                             0x0370
#define LWCC97_SET_ZLWLL_SUBREGION_REPORT_TYPE_ENABLE                                                         0:0
#define LWCC97_SET_ZLWLL_SUBREGION_REPORT_TYPE_ENABLE_FALSE                                            0x00000000
#define LWCC97_SET_ZLWLL_SUBREGION_REPORT_TYPE_ENABLE_TRUE                                             0x00000001
#define LWCC97_SET_ZLWLL_SUBREGION_REPORT_TYPE_TYPE                                                           6:4
#define LWCC97_SET_ZLWLL_SUBREGION_REPORT_TYPE_TYPE_DEPTH_TEST                                         0x00000000
#define LWCC97_SET_ZLWLL_SUBREGION_REPORT_TYPE_TYPE_DEPTH_TEST_NO_ACCEPT                               0x00000001
#define LWCC97_SET_ZLWLL_SUBREGION_REPORT_TYPE_TYPE_DEPTH_TEST_LATE_Z                                  0x00000002
#define LWCC97_SET_ZLWLL_SUBREGION_REPORT_TYPE_TYPE_STENCIL_TEST                                       0x00000003

#define LWCC97_SET_BALANCED_PRIMITIVE_WORKLOAD                                                             0x0374
#define LWCC97_SET_BALANCED_PRIMITIVE_WORKLOAD_IN_UNPARTITIONED_MODE                                          0:0
#define LWCC97_SET_BALANCED_PRIMITIVE_WORKLOAD_IN_UNPARTITIONED_MODE_FALSE                             0x00000000
#define LWCC97_SET_BALANCED_PRIMITIVE_WORKLOAD_IN_UNPARTITIONED_MODE_TRUE                              0x00000001
#define LWCC97_SET_BALANCED_PRIMITIVE_WORKLOAD_IN_TIMESLICED_MODE                                             4:4
#define LWCC97_SET_BALANCED_PRIMITIVE_WORKLOAD_IN_TIMESLICED_MODE_FALSE                                0x00000000
#define LWCC97_SET_BALANCED_PRIMITIVE_WORKLOAD_IN_TIMESLICED_MODE_TRUE                                 0x00000001
#define LWCC97_SET_BALANCED_PRIMITIVE_WORKLOAD_BY_PES_IN_UNPARTITIONED_MODE                                   8:8
#define LWCC97_SET_BALANCED_PRIMITIVE_WORKLOAD_BY_PES_IN_UNPARTITIONED_MODE_FALSE                      0x00000000
#define LWCC97_SET_BALANCED_PRIMITIVE_WORKLOAD_BY_PES_IN_UNPARTITIONED_MODE_TRUE                       0x00000001
#define LWCC97_SET_BALANCED_PRIMITIVE_WORKLOAD_BY_PES_IN_TIMESLICED_MODE                                      9:9
#define LWCC97_SET_BALANCED_PRIMITIVE_WORKLOAD_BY_PES_IN_TIMESLICED_MODE_FALSE                         0x00000000
#define LWCC97_SET_BALANCED_PRIMITIVE_WORKLOAD_BY_PES_IN_TIMESLICED_MODE_TRUE                          0x00000001

#define LWCC97_SET_MAX_PATCHES_PER_BATCH                                                                   0x0378
#define LWCC97_SET_MAX_PATCHES_PER_BATCH_V                                                                    5:0

#define LWCC97_SET_RASTER_ENABLE                                                                           0x037c
#define LWCC97_SET_RASTER_ENABLE_V                                                                            0:0
#define LWCC97_SET_RASTER_ENABLE_V_FALSE                                                               0x00000000
#define LWCC97_SET_RASTER_ENABLE_V_TRUE                                                                0x00000001

#define LWCC97_SET_STREAM_OUT_BUFFER_ENABLE(j)                                                    (0x0380+(j)*32)
#define LWCC97_SET_STREAM_OUT_BUFFER_ENABLE_V                                                                 0:0
#define LWCC97_SET_STREAM_OUT_BUFFER_ENABLE_V_FALSE                                                    0x00000000
#define LWCC97_SET_STREAM_OUT_BUFFER_ENABLE_V_TRUE                                                     0x00000001

#define LWCC97_SET_STREAM_OUT_BUFFER_ADDRESS_A(j)                                                 (0x0384+(j)*32)
#define LWCC97_SET_STREAM_OUT_BUFFER_ADDRESS_A_UPPER                                                          7:0

#define LWCC97_SET_STREAM_OUT_BUFFER_ADDRESS_B(j)                                                 (0x0388+(j)*32)
#define LWCC97_SET_STREAM_OUT_BUFFER_ADDRESS_B_LOWER                                                         31:0

#define LWCC97_SET_STREAM_OUT_BUFFER_SIZE(j)                                                      (0x038c+(j)*32)
#define LWCC97_SET_STREAM_OUT_BUFFER_SIZE_BYTES                                                              31:0

#define LWCC97_SET_STREAM_OUT_BUFFER_LOAD_WRITE_POINTER(j)                                        (0x0390+(j)*32)
#define LWCC97_SET_STREAM_OUT_BUFFER_LOAD_WRITE_POINTER_START_OFFSET                                         31:0

#define LWCC97_SET_POSITION_W_SCALED_OFFSET_SCALE_A(j)                                            (0x0400+(j)*16)
#define LWCC97_SET_POSITION_W_SCALED_OFFSET_SCALE_A_V                                                        31:0

#define LWCC97_SET_POSITION_W_SCALED_OFFSET_SCALE_B(j)                                            (0x0404+(j)*16)
#define LWCC97_SET_POSITION_W_SCALED_OFFSET_SCALE_B_V                                                        31:0

#define LWCC97_SET_POSITION_W_SCALED_OFFSET_RESERVED_A(j)                                         (0x0408+(j)*16)
#define LWCC97_SET_POSITION_W_SCALED_OFFSET_RESERVED_A_V                                                     31:0

#define LWCC97_SET_POSITION_W_SCALED_OFFSET_RESERVED_B(j)                                         (0x040c+(j)*16)
#define LWCC97_SET_POSITION_W_SCALED_OFFSET_RESERVED_B_V                                                     31:0

#define LWCC97_SET_Z_ROP_SLICE_MAP                                                                         0x0500
#define LWCC97_SET_Z_ROP_SLICE_MAP_VIRTUAL_ADDRESS_MASK                                                      31:0

#define LWCC97_SET_ROOT_TABLE_SELECTOR                                                                     0x0504
#define LWCC97_SET_ROOT_TABLE_SELECTOR_ROOT_TABLE                                                             2:0
#define LWCC97_SET_ROOT_TABLE_SELECTOR_OFFSET                                                                15:8

#define LWCC97_LOAD_ROOT_TABLE                                                                             0x0508
#define LWCC97_LOAD_ROOT_TABLE_V                                                                             31:0

#define LWCC97_SET_MME_MEM_ADDRESS_A                                                                       0x0550
#define LWCC97_SET_MME_MEM_ADDRESS_A_UPPER                                                                   24:0

#define LWCC97_SET_MME_MEM_ADDRESS_B                                                                       0x0554
#define LWCC97_SET_MME_MEM_ADDRESS_B_LOWER                                                                   31:0

#define LWCC97_SET_MME_DATA_RAM_ADDRESS                                                                    0x0558
#define LWCC97_SET_MME_DATA_RAM_ADDRESS_WORD                                                                 31:0

#define LWCC97_MME_DMA_READ                                                                                0x055c
#define LWCC97_MME_DMA_READ_LENGTH                                                                           31:0

#define LWCC97_MME_DMA_READ_FIFOED                                                                         0x0560
#define LWCC97_MME_DMA_READ_FIFOED_LENGTH                                                                    31:0

#define LWCC97_MME_DMA_WRITE                                                                               0x0564
#define LWCC97_MME_DMA_WRITE_LENGTH                                                                          31:0

#define LWCC97_MME_DMA_REDUCTION                                                                           0x0568
#define LWCC97_MME_DMA_REDUCTION_REDUCTION_OP                                                                 2:0
#define LWCC97_MME_DMA_REDUCTION_REDUCTION_OP_RED_ADD                                                  0x00000000
#define LWCC97_MME_DMA_REDUCTION_REDUCTION_OP_RED_MIN                                                  0x00000001
#define LWCC97_MME_DMA_REDUCTION_REDUCTION_OP_RED_MAX                                                  0x00000002
#define LWCC97_MME_DMA_REDUCTION_REDUCTION_OP_RED_INC                                                  0x00000003
#define LWCC97_MME_DMA_REDUCTION_REDUCTION_OP_RED_DEC                                                  0x00000004
#define LWCC97_MME_DMA_REDUCTION_REDUCTION_OP_RED_AND                                                  0x00000005
#define LWCC97_MME_DMA_REDUCTION_REDUCTION_OP_RED_OR                                                   0x00000006
#define LWCC97_MME_DMA_REDUCTION_REDUCTION_OP_RED_XOR                                                  0x00000007
#define LWCC97_MME_DMA_REDUCTION_REDUCTION_FORMAT                                                             5:4
#define LWCC97_MME_DMA_REDUCTION_REDUCTION_FORMAT_UNSIGNED                                             0x00000000
#define LWCC97_MME_DMA_REDUCTION_REDUCTION_FORMAT_SIGNED                                               0x00000001
#define LWCC97_MME_DMA_REDUCTION_REDUCTION_SIZE                                                               8:8
#define LWCC97_MME_DMA_REDUCTION_REDUCTION_SIZE_FOUR_BYTES                                             0x00000000
#define LWCC97_MME_DMA_REDUCTION_REDUCTION_SIZE_EIGHT_BYTES                                            0x00000001

#define LWCC97_MME_DMA_SYSMEMBAR                                                                           0x056c
#define LWCC97_MME_DMA_SYSMEMBAR_V                                                                            0:0

#define LWCC97_MME_DMA_SYNC                                                                                0x0570
#define LWCC97_MME_DMA_SYNC_VALUE                                                                            31:0

#define LWCC97_SET_MME_DATA_FIFO_CONFIG                                                                    0x0574
#define LWCC97_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE                                                             2:0
#define LWCC97_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_0KB                                             0x00000000
#define LWCC97_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_4KB                                             0x00000001
#define LWCC97_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_8KB                                             0x00000002
#define LWCC97_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_12KB                                            0x00000003
#define LWCC97_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_16KB                                            0x00000004

#define LWCC97_SET_VERTEX_STREAM_SIZE_A(j)                                                         (0x0600+(j)*8)
#define LWCC97_SET_VERTEX_STREAM_SIZE_A_UPPER                                                                 7:0

#define LWCC97_SET_VERTEX_STREAM_SIZE_B(j)                                                         (0x0604+(j)*8)
#define LWCC97_SET_VERTEX_STREAM_SIZE_B_LOWER                                                                31:0

#define LWCC97_SET_STREAM_OUT_CONTROL_STREAM(j)                                                   (0x0700+(j)*16)
#define LWCC97_SET_STREAM_OUT_CONTROL_STREAM_SELECT                                                           1:0

#define LWCC97_SET_STREAM_OUT_CONTROL_COMPONENT_COUNT(j)                                          (0x0704+(j)*16)
#define LWCC97_SET_STREAM_OUT_CONTROL_COMPONENT_COUNT_MAX                                                     7:0

#define LWCC97_SET_STREAM_OUT_CONTROL_STRIDE(j)                                                   (0x0708+(j)*16)
#define LWCC97_SET_STREAM_OUT_CONTROL_STRIDE_BYTES                                                           31:0

#define LWCC97_SET_RASTER_INPUT                                                                            0x0740
#define LWCC97_SET_RASTER_INPUT_STREAM_SELECT                                                                 1:0

#define LWCC97_SET_STREAM_OUTPUT                                                                           0x0744
#define LWCC97_SET_STREAM_OUTPUT_ENABLE                                                                       0:0
#define LWCC97_SET_STREAM_OUTPUT_ENABLE_FALSE                                                          0x00000000
#define LWCC97_SET_STREAM_OUTPUT_ENABLE_TRUE                                                           0x00000001

#define LWCC97_SET_DA_PRIMITIVE_RESTART_TOPOLOGY_CHANGE                                                    0x0748
#define LWCC97_SET_DA_PRIMITIVE_RESTART_TOPOLOGY_CHANGE_ENABLE                                                0:0
#define LWCC97_SET_DA_PRIMITIVE_RESTART_TOPOLOGY_CHANGE_ENABLE_FALSE                                   0x00000000
#define LWCC97_SET_DA_PRIMITIVE_RESTART_TOPOLOGY_CHANGE_ENABLE_TRUE                                    0x00000001

#define LWCC97_SET_SHADERMODEL3_REORDER                                                                    0x0750
#define LWCC97_SET_SHADERMODEL3_REORDER_ENABLE                                                                0:0
#define LWCC97_SET_SHADERMODEL3_REORDER_ENABLE_FALSE                                                   0x00000000
#define LWCC97_SET_SHADERMODEL3_REORDER_ENABLE_TRUE                                                    0x00000001

#define LWCC97_SET_HYBRID_ANTI_ALIAS_CONTROL                                                               0x0754
#define LWCC97_SET_HYBRID_ANTI_ALIAS_CONTROL_PASSES                                                           3:0
#define LWCC97_SET_HYBRID_ANTI_ALIAS_CONTROL_CENTROID                                                         4:4
#define LWCC97_SET_HYBRID_ANTI_ALIAS_CONTROL_CENTROID_PER_FRAGMENT                                     0x00000000
#define LWCC97_SET_HYBRID_ANTI_ALIAS_CONTROL_CENTROID_PER_PASS                                         0x00000001
#define LWCC97_SET_HYBRID_ANTI_ALIAS_CONTROL_PASSES_EXTENDED                                                  5:5

#define LWCC97_SET_QUADRO_TUNE                                                                             0x0758
#define LWCC97_SET_QUADRO_TUNE_V                                                                             31:0

#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC(i)                                               (0x0760+(i)*4)
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00                                                     3:0
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00_POSITION                                     0x00000000
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00_BLENDWEIGHT                                  0x00000001
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00_BLENDINDICES                                 0x00000002
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00_NORMAL                                       0x00000003
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00_PSIZE                                        0x00000004
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00_TEXCOORD                                     0x00000005
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00_TANGENT                                      0x00000006
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00_BINORMAL                                     0x00000007
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00_TESSFACTOR                                   0x00000008
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00_POSITIONT                                    0x00000009
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00_COLOR                                        0x0000000A
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00_FOG                                          0x0000000B
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00_DEPTH                                        0x0000000C
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00_SAMPLE                                       0x0000000D
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00_RESERVED_USAGE_A                             0x0000000E
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE00_RESERVED_USAGE_B                             0x0000000F
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_INDEX00                                                     7:4
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01                                                    11:8
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01_POSITION                                     0x00000000
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01_BLENDWEIGHT                                  0x00000001
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01_BLENDINDICES                                 0x00000002
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01_NORMAL                                       0x00000003
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01_PSIZE                                        0x00000004
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01_TEXCOORD                                     0x00000005
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01_TANGENT                                      0x00000006
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01_BINORMAL                                     0x00000007
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01_TESSFACTOR                                   0x00000008
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01_POSITIONT                                    0x00000009
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01_COLOR                                        0x0000000A
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01_FOG                                          0x0000000B
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01_DEPTH                                        0x0000000C
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01_SAMPLE                                       0x0000000D
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01_RESERVED_USAGE_A                             0x0000000E
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE01_RESERVED_USAGE_B                             0x0000000F
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_INDEX01                                                   15:12
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02                                                   19:16
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02_POSITION                                     0x00000000
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02_BLENDWEIGHT                                  0x00000001
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02_BLENDINDICES                                 0x00000002
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02_NORMAL                                       0x00000003
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02_PSIZE                                        0x00000004
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02_TEXCOORD                                     0x00000005
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02_TANGENT                                      0x00000006
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02_BINORMAL                                     0x00000007
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02_TESSFACTOR                                   0x00000008
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02_POSITIONT                                    0x00000009
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02_COLOR                                        0x0000000A
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02_FOG                                          0x0000000B
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02_DEPTH                                        0x0000000C
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02_SAMPLE                                       0x0000000D
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02_RESERVED_USAGE_A                             0x0000000E
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE02_RESERVED_USAGE_B                             0x0000000F
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_INDEX02                                                   23:20
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03                                                   27:24
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03_POSITION                                     0x00000000
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03_BLENDWEIGHT                                  0x00000001
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03_BLENDINDICES                                 0x00000002
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03_NORMAL                                       0x00000003
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03_PSIZE                                        0x00000004
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03_TEXCOORD                                     0x00000005
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03_TANGENT                                      0x00000006
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03_BINORMAL                                     0x00000007
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03_TESSFACTOR                                   0x00000008
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03_POSITIONT                                    0x00000009
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03_COLOR                                        0x0000000A
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03_FOG                                          0x0000000B
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03_DEPTH                                        0x0000000C
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03_SAMPLE                                       0x0000000D
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03_RESERVED_USAGE_A                             0x0000000E
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_USAGE03_RESERVED_USAGE_B                             0x0000000F
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_GENERIC_INDEX03                                                   31:28

#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_TEXCOORD(i)                                              (0x0770+(i)*4)
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_TEXCOORD_INDEX00                                                    7:4
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_TEXCOORD_INDEX01                                                  15:12
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_TEXCOORD_INDEX02                                                  23:20
#define LWCC97_SET_SHADERMODEL3_VS_OUTPUT_TEXCOORD_INDEX03                                                  31:28

#define LWCC97_SET_SHADER_LOCAL_MEMORY_WINDOW                                                              0x077c
#define LWCC97_SET_SHADER_LOCAL_MEMORY_WINDOW_BASE_ADDRESS                                                   31:0

#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC(i)                                                (0x0780+(i)*4)
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00                                                      3:0
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00_POSITION                                      0x00000000
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00_BLENDWEIGHT                                   0x00000001
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00_BLENDINDICES                                  0x00000002
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00_NORMAL                                        0x00000003
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00_PSIZE                                         0x00000004
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00_TEXCOORD                                      0x00000005
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00_TANGENT                                       0x00000006
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00_BINORMAL                                      0x00000007
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00_TESSFACTOR                                    0x00000008
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00_POSITIONT                                     0x00000009
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00_COLOR                                         0x0000000A
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00_FOG                                           0x0000000B
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00_DEPTH                                         0x0000000C
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00_SAMPLE                                        0x0000000D
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00_RESERVED_USAGE_A                              0x0000000E
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE00_RESERVED_USAGE_B                              0x0000000F
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_INDEX00                                                      7:4
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01                                                     11:8
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01_POSITION                                      0x00000000
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01_BLENDWEIGHT                                   0x00000001
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01_BLENDINDICES                                  0x00000002
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01_NORMAL                                        0x00000003
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01_PSIZE                                         0x00000004
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01_TEXCOORD                                      0x00000005
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01_TANGENT                                       0x00000006
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01_BINORMAL                                      0x00000007
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01_TESSFACTOR                                    0x00000008
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01_POSITIONT                                     0x00000009
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01_COLOR                                         0x0000000A
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01_FOG                                           0x0000000B
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01_DEPTH                                         0x0000000C
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01_SAMPLE                                        0x0000000D
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01_RESERVED_USAGE_A                              0x0000000E
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE01_RESERVED_USAGE_B                              0x0000000F
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_INDEX01                                                    15:12
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02                                                    19:16
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02_POSITION                                      0x00000000
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02_BLENDWEIGHT                                   0x00000001
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02_BLENDINDICES                                  0x00000002
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02_NORMAL                                        0x00000003
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02_PSIZE                                         0x00000004
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02_TEXCOORD                                      0x00000005
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02_TANGENT                                       0x00000006
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02_BINORMAL                                      0x00000007
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02_TESSFACTOR                                    0x00000008
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02_POSITIONT                                     0x00000009
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02_COLOR                                         0x0000000A
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02_FOG                                           0x0000000B
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02_DEPTH                                         0x0000000C
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02_SAMPLE                                        0x0000000D
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02_RESERVED_USAGE_A                              0x0000000E
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE02_RESERVED_USAGE_B                              0x0000000F
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_INDEX02                                                    23:20
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03                                                    27:24
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03_POSITION                                      0x00000000
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03_BLENDWEIGHT                                   0x00000001
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03_BLENDINDICES                                  0x00000002
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03_NORMAL                                        0x00000003
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03_PSIZE                                         0x00000004
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03_TEXCOORD                                      0x00000005
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03_TANGENT                                       0x00000006
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03_BINORMAL                                      0x00000007
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03_TESSFACTOR                                    0x00000008
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03_POSITIONT                                     0x00000009
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03_COLOR                                         0x0000000A
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03_FOG                                           0x0000000B
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03_DEPTH                                         0x0000000C
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03_SAMPLE                                        0x0000000D
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03_RESERVED_USAGE_A                              0x0000000E
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_USAGE03_RESERVED_USAGE_B                              0x0000000F
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_GENERIC_INDEX03                                                    31:28

#define LWCC97_SET_SHADER_LOCAL_MEMORY_A                                                                   0x0790
#define LWCC97_SET_SHADER_LOCAL_MEMORY_A_ADDRESS_UPPER                                                        7:0

#define LWCC97_SET_SHADER_LOCAL_MEMORY_B                                                                   0x0794
#define LWCC97_SET_SHADER_LOCAL_MEMORY_B_ADDRESS_LOWER                                                       31:0

#define LWCC97_SET_SHADER_LOCAL_MEMORY_C                                                                   0x0798
#define LWCC97_SET_SHADER_LOCAL_MEMORY_C_SIZE_UPPER                                                           6:0

#define LWCC97_SET_SHADER_LOCAL_MEMORY_D                                                                   0x079c
#define LWCC97_SET_SHADER_LOCAL_MEMORY_D_SIZE_LOWER                                                          31:0

#define LWCC97_SET_SHADER_LOCAL_MEMORY_E                                                                   0x07a0
#define LWCC97_SET_SHADER_LOCAL_MEMORY_E_DEFAULT_SIZE_PER_WARP                                               25:0

#define LWCC97_SET_COLOR_ZERO_BANDWIDTH_CLEAR                                                              0x07a4
#define LWCC97_SET_COLOR_ZERO_BANDWIDTH_CLEAR_SLOT_DISABLE_MASK                                              14:0

#define LWCC97_SET_Z_ZERO_BANDWIDTH_CLEAR                                                                  0x07a8
#define LWCC97_SET_Z_ZERO_BANDWIDTH_CLEAR_SLOT_DISABLE_MASK                                                  14:0

#define LWCC97_SET_TEXTURE_HEADER_VERSION                                                                  0x07ac
#define LWCC97_SET_TEXTURE_HEADER_VERSION_MAJOR                                                               7:0

#define LWCC97_SET_STENCIL_ZERO_BANDWIDTH_CLEAR                                                            0x07b0
#define LWCC97_SET_STENCIL_ZERO_BANDWIDTH_CLEAR_SLOT_DISABLE_MASK                                            14:0

#define LWCC97_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_A                                                   0x07b8
#define LWCC97_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_A_LOWER                                               31:0

#define LWCC97_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_B                                                   0x07bc
#define LWCC97_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_B_UPPER                                               24:0

#define LWCC97_SET_ZLWLL_REGION_SIZE_A                                                                     0x07c0
#define LWCC97_SET_ZLWLL_REGION_SIZE_A_WIDTH                                                                 15:0

#define LWCC97_SET_ZLWLL_REGION_SIZE_B                                                                     0x07c4
#define LWCC97_SET_ZLWLL_REGION_SIZE_B_HEIGHT                                                                15:0

#define LWCC97_SET_ZLWLL_REGION_SIZE_C                                                                     0x07c8
#define LWCC97_SET_ZLWLL_REGION_SIZE_C_DEPTH                                                                 15:0

#define LWCC97_SET_ZLWLL_REGION_PIXEL_OFFSET_C                                                             0x07cc
#define LWCC97_SET_ZLWLL_REGION_PIXEL_OFFSET_C_DEPTH                                                         15:0

#define LWCC97_SET_SHADERMODEL3_PS_INPUT_TEXCOORD(i)                                               (0x07d0+(i)*4)
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_TEXCOORD_INDEX00                                                     7:4
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_TEXCOORD_INDEX01                                                   15:12
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_TEXCOORD_INDEX02                                                   23:20
#define LWCC97_SET_SHADERMODEL3_PS_INPUT_TEXCOORD_INDEX03                                                   31:28

#define LWCC97_SET_LWLL_BEFORE_FETCH                                                                       0x07dc
#define LWCC97_SET_LWLL_BEFORE_FETCH_FETCH_STREAMS_ONCE                                                       0:0
#define LWCC97_SET_LWLL_BEFORE_FETCH_FETCH_STREAMS_ONCE_FALSE                                          0x00000000
#define LWCC97_SET_LWLL_BEFORE_FETCH_FETCH_STREAMS_ONCE_TRUE                                           0x00000001

#define LWCC97_SET_ZLWLL_REGION_LOCATION                                                                   0x07e0
#define LWCC97_SET_ZLWLL_REGION_LOCATION_START_ALIQUOT                                                       15:0
#define LWCC97_SET_ZLWLL_REGION_LOCATION_ALIQUOT_COUNT                                                      31:16

#define LWCC97_SET_ZLWLL_REGION_ALIQUOTS                                                                   0x07e4
#define LWCC97_SET_ZLWLL_REGION_ALIQUOTS_PER_LAYER                                                           15:0

#define LWCC97_SET_ZLWLL_STORAGE_A                                                                         0x07e8
#define LWCC97_SET_ZLWLL_STORAGE_A_ADDRESS_UPPER                                                              7:0

#define LWCC97_SET_ZLWLL_STORAGE_B                                                                         0x07ec
#define LWCC97_SET_ZLWLL_STORAGE_B_ADDRESS_LOWER                                                             31:0

#define LWCC97_SET_ZLWLL_STORAGE_C                                                                         0x07f0
#define LWCC97_SET_ZLWLL_STORAGE_C_LIMIT_ADDRESS_UPPER                                                        7:0

#define LWCC97_SET_ZLWLL_STORAGE_D                                                                         0x07f4
#define LWCC97_SET_ZLWLL_STORAGE_D_LIMIT_ADDRESS_LOWER                                                       31:0

#define LWCC97_SET_ZT_READ_ONLY                                                                            0x07f8
#define LWCC97_SET_ZT_READ_ONLY_ENABLE_Z                                                                      0:0
#define LWCC97_SET_ZT_READ_ONLY_ENABLE_Z_FALSE                                                         0x00000000
#define LWCC97_SET_ZT_READ_ONLY_ENABLE_Z_TRUE                                                          0x00000001
#define LWCC97_SET_ZT_READ_ONLY_ENABLE_STENCIL                                                                4:4
#define LWCC97_SET_ZT_READ_ONLY_ENABLE_STENCIL_FALSE                                                   0x00000000
#define LWCC97_SET_ZT_READ_ONLY_ENABLE_STENCIL_TRUE                                                    0x00000001

#define LWCC97_THROTTLE_SM                                                                                 0x07fc
#define LWCC97_THROTTLE_SM_MULTIPLY_ADD                                                                       0:0
#define LWCC97_THROTTLE_SM_MULTIPLY_ADD_FALSE                                                          0x00000000
#define LWCC97_THROTTLE_SM_MULTIPLY_ADD_TRUE                                                           0x00000001

#define LWCC97_SET_COLOR_TARGET_A(j)                                                              (0x0800+(j)*64)
#define LWCC97_SET_COLOR_TARGET_A_OFFSET_UPPER                                                                7:0

#define LWCC97_SET_COLOR_TARGET_B(j)                                                              (0x0804+(j)*64)
#define LWCC97_SET_COLOR_TARGET_B_OFFSET_LOWER                                                               31:0

#define LWCC97_SET_COLOR_TARGET_WIDTH(j)                                                          (0x0808+(j)*64)
#define LWCC97_SET_COLOR_TARGET_WIDTH_V                                                                      27:0

#define LWCC97_SET_COLOR_TARGET_HEIGHT(j)                                                         (0x080c+(j)*64)
#define LWCC97_SET_COLOR_TARGET_HEIGHT_V                                                                     16:0

#define LWCC97_SET_COLOR_TARGET_FORMAT(j)                                                         (0x0810+(j)*64)
#define LWCC97_SET_COLOR_TARGET_FORMAT_V                                                                      7:0
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_DISABLED                                                      0x00000000
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RF32_GF32_BF32_AF32                                           0x000000C0
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RS32_GS32_BS32_AS32                                           0x000000C1
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RU32_GU32_BU32_AU32                                           0x000000C2
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RF32_GF32_BF32_X32                                            0x000000C3
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RS32_GS32_BS32_X32                                            0x000000C4
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RU32_GU32_BU32_X32                                            0x000000C5
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_R16_G16_B16_A16                                               0x000000C6
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RN16_GN16_BN16_AN16                                           0x000000C7
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RS16_GS16_BS16_AS16                                           0x000000C8
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RU16_GU16_BU16_AU16                                           0x000000C9
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RF16_GF16_BF16_AF16                                           0x000000CA
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RF32_GF32                                                     0x000000CB
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RS32_GS32                                                     0x000000CC
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RU32_GU32                                                     0x000000CD
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RF16_GF16_BF16_X16                                            0x000000CE
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_A8R8G8B8                                                      0x000000CF
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_A8RL8GL8BL8                                                   0x000000D0
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_A2B10G10R10                                                   0x000000D1
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_AU2BU10GU10RU10                                               0x000000D2
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_A8B8G8R8                                                      0x000000D5
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_A8BL8GL8RL8                                                   0x000000D6
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_AN8BN8GN8RN8                                                  0x000000D7
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_AS8BS8GS8RS8                                                  0x000000D8
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_AU8BU8GU8RU8                                                  0x000000D9
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_R16_G16                                                       0x000000DA
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RN16_GN16                                                     0x000000DB
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RS16_GS16                                                     0x000000DC
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RU16_GU16                                                     0x000000DD
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RF16_GF16                                                     0x000000DE
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_A2R10G10B10                                                   0x000000DF
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_BF10GF11RF11                                                  0x000000E0
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RS32                                                          0x000000E3
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RU32                                                          0x000000E4
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RF32                                                          0x000000E5
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_X8R8G8B8                                                      0x000000E6
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_X8RL8GL8BL8                                                   0x000000E7
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_R5G6B5                                                        0x000000E8
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_A1R5G5B5                                                      0x000000E9
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_G8R8                                                          0x000000EA
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_GN8RN8                                                        0x000000EB
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_GS8RS8                                                        0x000000EC
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_GU8RU8                                                        0x000000ED
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_R16                                                           0x000000EE
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RN16                                                          0x000000EF
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RS16                                                          0x000000F0
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RU16                                                          0x000000F1
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RF16                                                          0x000000F2
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_R8                                                            0x000000F3
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RN8                                                           0x000000F4
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RS8                                                           0x000000F5
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RU8                                                           0x000000F6
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_A8                                                            0x000000F7
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_X1R5G5B5                                                      0x000000F8
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_X8B8G8R8                                                      0x000000F9
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_X8BL8GL8RL8                                                   0x000000FA
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_Z1R5G5B5                                                      0x000000FB
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_O1R5G5B5                                                      0x000000FC
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_Z8R8G8B8                                                      0x000000FD
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_O8R8G8B8                                                      0x000000FE
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_R32                                                           0x000000FF
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_A16                                                           0x00000040
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_AF16                                                          0x00000041
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_AF32                                                          0x00000042
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_A8R8                                                          0x00000043
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_R16_A16                                                       0x00000044
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RF16_AF16                                                     0x00000045
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_RF32_AF32                                                     0x00000046
#define LWCC97_SET_COLOR_TARGET_FORMAT_V_B8G8R8A8                                                      0x00000047

#define LWCC97_SET_COLOR_TARGET_MEMORY(j)                                                         (0x0814+(j)*64)
#define LWCC97_SET_COLOR_TARGET_MEMORY_BLOCK_WIDTH                                                            3:0
#define LWCC97_SET_COLOR_TARGET_MEMORY_BLOCK_WIDTH_ONE_GOB                                             0x00000000
#define LWCC97_SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT                                                           7:4
#define LWCC97_SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT_ONE_GOB                                            0x00000000
#define LWCC97_SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT_TWO_GOBS                                           0x00000001
#define LWCC97_SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT_FOUR_GOBS                                          0x00000002
#define LWCC97_SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT_EIGHT_GOBS                                         0x00000003
#define LWCC97_SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT_SIXTEEN_GOBS                                       0x00000004
#define LWCC97_SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT_THIRTYTWO_GOBS                                     0x00000005
#define LWCC97_SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH                                                           11:8
#define LWCC97_SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH_ONE_GOB                                             0x00000000
#define LWCC97_SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH_TWO_GOBS                                            0x00000001
#define LWCC97_SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH_FOUR_GOBS                                           0x00000002
#define LWCC97_SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH_EIGHT_GOBS                                          0x00000003
#define LWCC97_SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH_SIXTEEN_GOBS                                        0x00000004
#define LWCC97_SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH_THIRTYTWO_GOBS                                      0x00000005
#define LWCC97_SET_COLOR_TARGET_MEMORY_LAYOUT                                                               12:12
#define LWCC97_SET_COLOR_TARGET_MEMORY_LAYOUT_BLOCKLINEAR                                              0x00000000
#define LWCC97_SET_COLOR_TARGET_MEMORY_LAYOUT_PITCH                                                    0x00000001
#define LWCC97_SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL                                              16:16
#define LWCC97_SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE             0x00000000
#define LWCC97_SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_DEPTH_SIZE             0x00000001

#define LWCC97_SET_COLOR_TARGET_THIRD_DIMENSION(j)                                                (0x0818+(j)*64)
#define LWCC97_SET_COLOR_TARGET_THIRD_DIMENSION_V                                                            27:0

#define LWCC97_SET_COLOR_TARGET_ARRAY_PITCH(j)                                                    (0x081c+(j)*64)
#define LWCC97_SET_COLOR_TARGET_ARRAY_PITCH_V                                                                31:0

#define LWCC97_SET_COLOR_TARGET_LAYER(j)                                                          (0x0820+(j)*64)
#define LWCC97_SET_COLOR_TARGET_LAYER_OFFSET                                                                 15:0

#define LWCC97_SET_COLOR_TARGET_C_ROP_SLICE_MAP(j)                                                (0x0824+(j)*64)
#define LWCC97_SET_COLOR_TARGET_C_ROP_SLICE_MAP_VIRTUAL_ADDRESS_MASK                                         31:0

#define LWCC97_SET_VIEWPORT_SCALE_X(j)                                                            (0x0a00+(j)*32)
#define LWCC97_SET_VIEWPORT_SCALE_X_V                                                                        31:0

#define LWCC97_SET_VIEWPORT_SCALE_Y(j)                                                            (0x0a04+(j)*32)
#define LWCC97_SET_VIEWPORT_SCALE_Y_V                                                                        31:0

#define LWCC97_SET_VIEWPORT_SCALE_Z(j)                                                            (0x0a08+(j)*32)
#define LWCC97_SET_VIEWPORT_SCALE_Z_V                                                                        31:0

#define LWCC97_SET_VIEWPORT_OFFSET_X(j)                                                           (0x0a0c+(j)*32)
#define LWCC97_SET_VIEWPORT_OFFSET_X_V                                                                       31:0

#define LWCC97_SET_VIEWPORT_OFFSET_Y(j)                                                           (0x0a10+(j)*32)
#define LWCC97_SET_VIEWPORT_OFFSET_Y_V                                                                       31:0

#define LWCC97_SET_VIEWPORT_OFFSET_Z(j)                                                           (0x0a14+(j)*32)
#define LWCC97_SET_VIEWPORT_OFFSET_Z_V                                                                       31:0

#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE(j)                                                 (0x0a18+(j)*32)
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_X                                                              2:0
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_X_POS_X                                                 0x00000000
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_X_NEG_X                                                 0x00000001
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_X_POS_Y                                                 0x00000002
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_X_NEG_Y                                                 0x00000003
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_X_POS_Z                                                 0x00000004
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_X_NEG_Z                                                 0x00000005
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_X_POS_W                                                 0x00000006
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_X_NEG_W                                                 0x00000007
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Y                                                              6:4
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Y_POS_X                                                 0x00000000
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Y_NEG_X                                                 0x00000001
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Y_POS_Y                                                 0x00000002
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Y_NEG_Y                                                 0x00000003
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Y_POS_Z                                                 0x00000004
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Y_NEG_Z                                                 0x00000005
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Y_POS_W                                                 0x00000006
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Y_NEG_W                                                 0x00000007
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Z                                                             10:8
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Z_POS_X                                                 0x00000000
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Z_NEG_X                                                 0x00000001
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Z_POS_Y                                                 0x00000002
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Z_NEG_Y                                                 0x00000003
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Z_POS_Z                                                 0x00000004
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Z_NEG_Z                                                 0x00000005
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Z_POS_W                                                 0x00000006
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_Z_NEG_W                                                 0x00000007
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_W                                                            14:12
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_W_POS_X                                                 0x00000000
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_W_NEG_X                                                 0x00000001
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_W_POS_Y                                                 0x00000002
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_W_NEG_Y                                                 0x00000003
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_W_POS_Z                                                 0x00000004
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_W_NEG_Z                                                 0x00000005
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_W_POS_W                                                 0x00000006
#define LWCC97_SET_VIEWPORT_COORDINATE_SWIZZLE_W_NEG_W                                                 0x00000007

#define LWCC97_SET_VIEWPORT_INCREASE_SNAP_GRID_PRECISION(j)                                       (0x0a1c+(j)*32)
#define LWCC97_SET_VIEWPORT_INCREASE_SNAP_GRID_PRECISION_X_BITS                                               4:0
#define LWCC97_SET_VIEWPORT_INCREASE_SNAP_GRID_PRECISION_Y_BITS                                              12:8

#define LWCC97_SET_VIEWPORT_CLIP_HORIZONTAL(j)                                                    (0x0c00+(j)*16)
#define LWCC97_SET_VIEWPORT_CLIP_HORIZONTAL_X0                                                               15:0
#define LWCC97_SET_VIEWPORT_CLIP_HORIZONTAL_WIDTH                                                           31:16

#define LWCC97_SET_VIEWPORT_CLIP_VERTICAL(j)                                                      (0x0c04+(j)*16)
#define LWCC97_SET_VIEWPORT_CLIP_VERTICAL_Y0                                                                 15:0
#define LWCC97_SET_VIEWPORT_CLIP_VERTICAL_HEIGHT                                                            31:16

#define LWCC97_SET_VIEWPORT_CLIP_MIN_Z(j)                                                         (0x0c08+(j)*16)
#define LWCC97_SET_VIEWPORT_CLIP_MIN_Z_V                                                                     31:0

#define LWCC97_SET_VIEWPORT_CLIP_MAX_Z(j)                                                         (0x0c0c+(j)*16)
#define LWCC97_SET_VIEWPORT_CLIP_MAX_Z_V                                                                     31:0

#define LWCC97_SET_WINDOW_CLIP_HORIZONTAL(j)                                                       (0x0d00+(j)*8)
#define LWCC97_SET_WINDOW_CLIP_HORIZONTAL_XMIN                                                               15:0
#define LWCC97_SET_WINDOW_CLIP_HORIZONTAL_XMAX                                                              31:16

#define LWCC97_SET_WINDOW_CLIP_VERTICAL(j)                                                         (0x0d04+(j)*8)
#define LWCC97_SET_WINDOW_CLIP_VERTICAL_YMIN                                                                 15:0
#define LWCC97_SET_WINDOW_CLIP_VERTICAL_YMAX                                                                31:16

#define LWCC97_SET_CLIP_ID_EXTENT_X(j)                                                             (0x0d40+(j)*8)
#define LWCC97_SET_CLIP_ID_EXTENT_X_MINX                                                                     15:0
#define LWCC97_SET_CLIP_ID_EXTENT_X_WIDTH                                                                   31:16

#define LWCC97_SET_CLIP_ID_EXTENT_Y(j)                                                             (0x0d44+(j)*8)
#define LWCC97_SET_CLIP_ID_EXTENT_Y_MINY                                                                     15:0
#define LWCC97_SET_CLIP_ID_EXTENT_Y_HEIGHT                                                                  31:16

#define LWCC97_SET_MAX_STREAM_OUTPUT_GS_INSTANCES_PER_TASK                                                 0x0d60
#define LWCC97_SET_MAX_STREAM_OUTPUT_GS_INSTANCES_PER_TASK_V                                                 10:0

#define LWCC97_SET_API_VISIBLE_CALL_LIMIT                                                                  0x0d64
#define LWCC97_SET_API_VISIBLE_CALL_LIMIT_V                                                                   3:0
#define LWCC97_SET_API_VISIBLE_CALL_LIMIT_V__0                                                         0x00000000
#define LWCC97_SET_API_VISIBLE_CALL_LIMIT_V__1                                                         0x00000001
#define LWCC97_SET_API_VISIBLE_CALL_LIMIT_V__2                                                         0x00000002
#define LWCC97_SET_API_VISIBLE_CALL_LIMIT_V__4                                                         0x00000003
#define LWCC97_SET_API_VISIBLE_CALL_LIMIT_V__8                                                         0x00000004
#define LWCC97_SET_API_VISIBLE_CALL_LIMIT_V__16                                                        0x00000005
#define LWCC97_SET_API_VISIBLE_CALL_LIMIT_V__32                                                        0x00000006
#define LWCC97_SET_API_VISIBLE_CALL_LIMIT_V__64                                                        0x00000007
#define LWCC97_SET_API_VISIBLE_CALL_LIMIT_V__128                                                       0x00000008
#define LWCC97_SET_API_VISIBLE_CALL_LIMIT_V_NO_CHECK                                                   0x0000000F

#define LWCC97_SET_STATISTICS_COUNTER                                                                      0x0d68
#define LWCC97_SET_STATISTICS_COUNTER_DA_VERTICES_GENERATED_ENABLE                                            0:0
#define LWCC97_SET_STATISTICS_COUNTER_DA_VERTICES_GENERATED_ENABLE_FALSE                               0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_DA_VERTICES_GENERATED_ENABLE_TRUE                                0x00000001
#define LWCC97_SET_STATISTICS_COUNTER_DA_PRIMITIVES_GENERATED_ENABLE                                          1:1
#define LWCC97_SET_STATISTICS_COUNTER_DA_PRIMITIVES_GENERATED_ENABLE_FALSE                             0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_DA_PRIMITIVES_GENERATED_ENABLE_TRUE                              0x00000001
#define LWCC97_SET_STATISTICS_COUNTER_VS_ILWOCATIONS_ENABLE                                                   2:2
#define LWCC97_SET_STATISTICS_COUNTER_VS_ILWOCATIONS_ENABLE_FALSE                                      0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_VS_ILWOCATIONS_ENABLE_TRUE                                       0x00000001
#define LWCC97_SET_STATISTICS_COUNTER_GS_ILWOCATIONS_ENABLE                                                   3:3
#define LWCC97_SET_STATISTICS_COUNTER_GS_ILWOCATIONS_ENABLE_FALSE                                      0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_GS_ILWOCATIONS_ENABLE_TRUE                                       0x00000001
#define LWCC97_SET_STATISTICS_COUNTER_GS_PRIMITIVES_GENERATED_ENABLE                                          4:4
#define LWCC97_SET_STATISTICS_COUNTER_GS_PRIMITIVES_GENERATED_ENABLE_FALSE                             0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_GS_PRIMITIVES_GENERATED_ENABLE_TRUE                              0x00000001
#define LWCC97_SET_STATISTICS_COUNTER_STREAMING_PRIMITIVES_SUCCEEDED_ENABLE                                   5:5
#define LWCC97_SET_STATISTICS_COUNTER_STREAMING_PRIMITIVES_SUCCEEDED_ENABLE_FALSE                      0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_STREAMING_PRIMITIVES_SUCCEEDED_ENABLE_TRUE                       0x00000001
#define LWCC97_SET_STATISTICS_COUNTER_STREAMING_PRIMITIVES_NEEDED_ENABLE                                      6:6
#define LWCC97_SET_STATISTICS_COUNTER_STREAMING_PRIMITIVES_NEEDED_ENABLE_FALSE                         0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_STREAMING_PRIMITIVES_NEEDED_ENABLE_TRUE                          0x00000001
#define LWCC97_SET_STATISTICS_COUNTER_CLIPPER_ILWOCATIONS_ENABLE                                              7:7
#define LWCC97_SET_STATISTICS_COUNTER_CLIPPER_ILWOCATIONS_ENABLE_FALSE                                 0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_CLIPPER_ILWOCATIONS_ENABLE_TRUE                                  0x00000001
#define LWCC97_SET_STATISTICS_COUNTER_CLIPPER_PRIMITIVES_GENERATED_ENABLE                                     8:8
#define LWCC97_SET_STATISTICS_COUNTER_CLIPPER_PRIMITIVES_GENERATED_ENABLE_FALSE                        0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_CLIPPER_PRIMITIVES_GENERATED_ENABLE_TRUE                         0x00000001
#define LWCC97_SET_STATISTICS_COUNTER_PS_ILWOCATIONS_ENABLE                                                   9:9
#define LWCC97_SET_STATISTICS_COUNTER_PS_ILWOCATIONS_ENABLE_FALSE                                      0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_PS_ILWOCATIONS_ENABLE_TRUE                                       0x00000001
#define LWCC97_SET_STATISTICS_COUNTER_TI_ILWOCATIONS_ENABLE                                                 11:11
#define LWCC97_SET_STATISTICS_COUNTER_TI_ILWOCATIONS_ENABLE_FALSE                                      0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_TI_ILWOCATIONS_ENABLE_TRUE                                       0x00000001
#define LWCC97_SET_STATISTICS_COUNTER_TS_ILWOCATIONS_ENABLE                                                 12:12
#define LWCC97_SET_STATISTICS_COUNTER_TS_ILWOCATIONS_ENABLE_FALSE                                      0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_TS_ILWOCATIONS_ENABLE_TRUE                                       0x00000001
#define LWCC97_SET_STATISTICS_COUNTER_TS_PRIMITIVES_GENERATED_ENABLE                                        13:13
#define LWCC97_SET_STATISTICS_COUNTER_TS_PRIMITIVES_GENERATED_ENABLE_FALSE                             0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_TS_PRIMITIVES_GENERATED_ENABLE_TRUE                              0x00000001
#define LWCC97_SET_STATISTICS_COUNTER_TOTAL_STREAMING_PRIMITIVES_NEEDED_SUCCEEDED_ENABLE                    14:14
#define LWCC97_SET_STATISTICS_COUNTER_TOTAL_STREAMING_PRIMITIVES_NEEDED_SUCCEEDED_ENABLE_FALSE             0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_TOTAL_STREAMING_PRIMITIVES_NEEDED_SUCCEEDED_ENABLE_TRUE             0x00000001
#define LWCC97_SET_STATISTICS_COUNTER_VTG_PRIMITIVES_OUT_ENABLE                                             10:10
#define LWCC97_SET_STATISTICS_COUNTER_VTG_PRIMITIVES_OUT_ENABLE_FALSE                                  0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_VTG_PRIMITIVES_OUT_ENABLE_TRUE                                   0x00000001
#define LWCC97_SET_STATISTICS_COUNTER_ALPHA_BETA_CLOCKS_ENABLE                                              15:15
#define LWCC97_SET_STATISTICS_COUNTER_ALPHA_BETA_CLOCKS_ENABLE_FALSE                                   0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_ALPHA_BETA_CLOCKS_ENABLE_TRUE                                    0x00000001
#define LWCC97_SET_STATISTICS_COUNTER_SCG_CLOCKS_ENABLE                                                     16:16
#define LWCC97_SET_STATISTICS_COUNTER_SCG_CLOCKS_ENABLE_FALSE                                          0x00000000
#define LWCC97_SET_STATISTICS_COUNTER_SCG_CLOCKS_ENABLE_TRUE                                           0x00000001

#define LWCC97_SET_CLEAR_RECT_HORIZONTAL                                                                   0x0d6c
#define LWCC97_SET_CLEAR_RECT_HORIZONTAL_XMIN                                                                15:0
#define LWCC97_SET_CLEAR_RECT_HORIZONTAL_XMAX                                                               31:16

#define LWCC97_SET_CLEAR_RECT_VERTICAL                                                                     0x0d70
#define LWCC97_SET_CLEAR_RECT_VERTICAL_YMIN                                                                  15:0
#define LWCC97_SET_CLEAR_RECT_VERTICAL_YMAX                                                                 31:16

#define LWCC97_SET_VERTEX_ARRAY_START                                                                      0x0d74
#define LWCC97_SET_VERTEX_ARRAY_START_V                                                                      31:0

#define LWCC97_DRAW_VERTEX_ARRAY                                                                           0x0d78
#define LWCC97_DRAW_VERTEX_ARRAY_COUNT                                                                       31:0

#define LWCC97_SET_VIEWPORT_Z_CLIP                                                                         0x0d7c
#define LWCC97_SET_VIEWPORT_Z_CLIP_RANGE                                                                      0:0
#define LWCC97_SET_VIEWPORT_Z_CLIP_RANGE_NEGATIVE_W_TO_POSITIVE_W                                      0x00000000
#define LWCC97_SET_VIEWPORT_Z_CLIP_RANGE_ZERO_TO_POSITIVE_W                                            0x00000001

#define LWCC97_SET_COLOR_CLEAR_VALUE(i)                                                            (0x0d80+(i)*4)
#define LWCC97_SET_COLOR_CLEAR_VALUE_V                                                                       31:0

#define LWCC97_SET_Z_CLEAR_VALUE                                                                           0x0d90
#define LWCC97_SET_Z_CLEAR_VALUE_V                                                                           31:0

#define LWCC97_SET_SHADER_CACHE_CONTROL                                                                    0x0d94
#define LWCC97_SET_SHADER_CACHE_CONTROL_ICACHE_PREFETCH_ENABLE                                                0:0
#define LWCC97_SET_SHADER_CACHE_CONTROL_ICACHE_PREFETCH_ENABLE_FALSE                                   0x00000000
#define LWCC97_SET_SHADER_CACHE_CONTROL_ICACHE_PREFETCH_ENABLE_TRUE                                    0x00000001

#define LWCC97_FORCE_TRANSITION_TO_BETA                                                                    0x0d98
#define LWCC97_FORCE_TRANSITION_TO_BETA_V                                                                     0:0

#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_ENABLE                                                          0x0d9c
#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_ENABLE_V                                                           0:0
#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_ENABLE_V_FALSE                                              0x00000000
#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_ENABLE_V_TRUE                                               0x00000001

#define LWCC97_SET_STENCIL_CLEAR_VALUE                                                                     0x0da0
#define LWCC97_SET_STENCIL_CLEAR_VALUE_V                                                                      7:0

#define LWCC97_ILWALIDATE_SHADER_CACHES_NO_WFI                                                             0x0da4
#define LWCC97_ILWALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION                                                    0:0
#define LWCC97_ILWALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION_FALSE                                       0x00000000
#define LWCC97_ILWALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION_TRUE                                        0x00000001
#define LWCC97_ILWALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA                                                    4:4
#define LWCC97_ILWALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA_FALSE                                       0x00000000
#define LWCC97_ILWALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA_TRUE                                        0x00000001
#define LWCC97_ILWALIDATE_SHADER_CACHES_NO_WFI_CONSTANT                                                     12:12
#define LWCC97_ILWALIDATE_SHADER_CACHES_NO_WFI_CONSTANT_FALSE                                          0x00000000
#define LWCC97_ILWALIDATE_SHADER_CACHES_NO_WFI_CONSTANT_TRUE                                           0x00000001

#define LWCC97_SET_ZLWLL_SERIALIZATION                                                                     0x0da8
#define LWCC97_SET_ZLWLL_SERIALIZATION_ENABLE                                                                 0:0
#define LWCC97_SET_ZLWLL_SERIALIZATION_ENABLE_FALSE                                                    0x00000000
#define LWCC97_SET_ZLWLL_SERIALIZATION_ENABLE_TRUE                                                     0x00000001
#define LWCC97_SET_ZLWLL_SERIALIZATION_APPLIED                                                                5:4
#define LWCC97_SET_ZLWLL_SERIALIZATION_APPLIED_ALWAYS                                                  0x00000000
#define LWCC97_SET_ZLWLL_SERIALIZATION_APPLIED_LATE_Z                                                  0x00000001
#define LWCC97_SET_ZLWLL_SERIALIZATION_APPLIED_OUT_OF_GAMUT_Z                                          0x00000002
#define LWCC97_SET_ZLWLL_SERIALIZATION_APPLIED_LATE_Z_OR_OUT_OF_GAMUT_Z                                0x00000003

#define LWCC97_SET_FRONT_POLYGON_MODE                                                                      0x0dac
#define LWCC97_SET_FRONT_POLYGON_MODE_V                                                                      31:0
#define LWCC97_SET_FRONT_POLYGON_MODE_V_POINT                                                          0x00001B00
#define LWCC97_SET_FRONT_POLYGON_MODE_V_LINE                                                           0x00001B01
#define LWCC97_SET_FRONT_POLYGON_MODE_V_FILL                                                           0x00001B02

#define LWCC97_SET_BACK_POLYGON_MODE                                                                       0x0db0
#define LWCC97_SET_BACK_POLYGON_MODE_V                                                                       31:0
#define LWCC97_SET_BACK_POLYGON_MODE_V_POINT                                                           0x00001B00
#define LWCC97_SET_BACK_POLYGON_MODE_V_LINE                                                            0x00001B01
#define LWCC97_SET_BACK_POLYGON_MODE_V_FILL                                                            0x00001B02

#define LWCC97_SET_POLY_SMOOTH                                                                             0x0db4
#define LWCC97_SET_POLY_SMOOTH_ENABLE                                                                         0:0
#define LWCC97_SET_POLY_SMOOTH_ENABLE_FALSE                                                            0x00000000
#define LWCC97_SET_POLY_SMOOTH_ENABLE_TRUE                                                             0x00000001

#define LWCC97_SET_ZLWLL_DIR_FORMAT                                                                        0x0dbc
#define LWCC97_SET_ZLWLL_DIR_FORMAT_ZDIR                                                                     15:0
#define LWCC97_SET_ZLWLL_DIR_FORMAT_ZDIR_LESS                                                          0x00000000
#define LWCC97_SET_ZLWLL_DIR_FORMAT_ZDIR_GREATER                                                       0x00000001
#define LWCC97_SET_ZLWLL_DIR_FORMAT_ZFORMAT                                                                 31:16
#define LWCC97_SET_ZLWLL_DIR_FORMAT_ZFORMAT_MSB                                                        0x00000000
#define LWCC97_SET_ZLWLL_DIR_FORMAT_ZFORMAT_FP                                                         0x00000001
#define LWCC97_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZTRICK                                                     0x00000002
#define LWCC97_SET_ZLWLL_DIR_FORMAT_ZFORMAT_ZF32_1                                                     0x00000003

#define LWCC97_SET_POLY_OFFSET_POINT                                                                       0x0dc0
#define LWCC97_SET_POLY_OFFSET_POINT_ENABLE                                                                   0:0
#define LWCC97_SET_POLY_OFFSET_POINT_ENABLE_FALSE                                                      0x00000000
#define LWCC97_SET_POLY_OFFSET_POINT_ENABLE_TRUE                                                       0x00000001

#define LWCC97_SET_POLY_OFFSET_LINE                                                                        0x0dc4
#define LWCC97_SET_POLY_OFFSET_LINE_ENABLE                                                                    0:0
#define LWCC97_SET_POLY_OFFSET_LINE_ENABLE_FALSE                                                       0x00000000
#define LWCC97_SET_POLY_OFFSET_LINE_ENABLE_TRUE                                                        0x00000001

#define LWCC97_SET_POLY_OFFSET_FILL                                                                        0x0dc8
#define LWCC97_SET_POLY_OFFSET_FILL_ENABLE                                                                    0:0
#define LWCC97_SET_POLY_OFFSET_FILL_ENABLE_FALSE                                                       0x00000000
#define LWCC97_SET_POLY_OFFSET_FILL_ENABLE_TRUE                                                        0x00000001

#define LWCC97_SET_PATCH                                                                                   0x0dcc
#define LWCC97_SET_PATCH_SIZE                                                                                 7:0

#define LWCC97_SET_ITERATED_BLEND                                                                          0x0dd0
#define LWCC97_SET_ITERATED_BLEND_ENABLE                                                                      0:0
#define LWCC97_SET_ITERATED_BLEND_ENABLE_FALSE                                                         0x00000000
#define LWCC97_SET_ITERATED_BLEND_ENABLE_TRUE                                                          0x00000001
#define LWCC97_SET_ITERATED_BLEND_ALPHA_ENABLE                                                                1:1
#define LWCC97_SET_ITERATED_BLEND_ALPHA_ENABLE_FALSE                                                   0x00000000
#define LWCC97_SET_ITERATED_BLEND_ALPHA_ENABLE_TRUE                                                    0x00000001

#define LWCC97_SET_ITERATED_BLEND_PASS                                                                     0x0dd4
#define LWCC97_SET_ITERATED_BLEND_PASS_COUNT                                                                  7:0

#define LWCC97_SET_ZLWLL_CRITERION                                                                         0x0dd8
#define LWCC97_SET_ZLWLL_CRITERION_SFUNC                                                                      7:0
#define LWCC97_SET_ZLWLL_CRITERION_SFUNC_NEVER                                                         0x00000000
#define LWCC97_SET_ZLWLL_CRITERION_SFUNC_LESS                                                          0x00000001
#define LWCC97_SET_ZLWLL_CRITERION_SFUNC_EQUAL                                                         0x00000002
#define LWCC97_SET_ZLWLL_CRITERION_SFUNC_LEQUAL                                                        0x00000003
#define LWCC97_SET_ZLWLL_CRITERION_SFUNC_GREATER                                                       0x00000004
#define LWCC97_SET_ZLWLL_CRITERION_SFUNC_NOTEQUAL                                                      0x00000005
#define LWCC97_SET_ZLWLL_CRITERION_SFUNC_GEQUAL                                                        0x00000006
#define LWCC97_SET_ZLWLL_CRITERION_SFUNC_ALWAYS                                                        0x00000007
#define LWCC97_SET_ZLWLL_CRITERION_NO_ILWALIDATE                                                              8:8
#define LWCC97_SET_ZLWLL_CRITERION_NO_ILWALIDATE_FALSE                                                 0x00000000
#define LWCC97_SET_ZLWLL_CRITERION_NO_ILWALIDATE_TRUE                                                  0x00000001
#define LWCC97_SET_ZLWLL_CRITERION_FORCE_MATCH                                                                9:9
#define LWCC97_SET_ZLWLL_CRITERION_FORCE_MATCH_FALSE                                                   0x00000000
#define LWCC97_SET_ZLWLL_CRITERION_FORCE_MATCH_TRUE                                                    0x00000001
#define LWCC97_SET_ZLWLL_CRITERION_SREF                                                                     23:16
#define LWCC97_SET_ZLWLL_CRITERION_SMASK                                                                    31:24

#define LWCC97_PIXEL_SHADER_BARRIER                                                                        0x0de0
#define LWCC97_PIXEL_SHADER_BARRIER_SYSMEMBAR_ENABLE                                                          0:0
#define LWCC97_PIXEL_SHADER_BARRIER_SYSMEMBAR_ENABLE_FALSE                                             0x00000000
#define LWCC97_PIXEL_SHADER_BARRIER_SYSMEMBAR_ENABLE_TRUE                                              0x00000001
#define LWCC97_PIXEL_SHADER_BARRIER_BARRIER_LOCATION                                                          1:1
#define LWCC97_PIXEL_SHADER_BARRIER_BARRIER_LOCATION_BLOCK_BEFORE_PS                                   0x00000000
#define LWCC97_PIXEL_SHADER_BARRIER_BARRIER_LOCATION_BLOCK_BEFORE_PS_AND_ZTEST                         0x00000001

#define LWCC97_SET_SM_TIMEOUT_INTERVAL                                                                     0x0de4
#define LWCC97_SET_SM_TIMEOUT_INTERVAL_COUNTER_BIT                                                            5:0

#define LWCC97_SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY                                                       0x0de8
#define LWCC97_SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY_ENABLE                                                   0:0
#define LWCC97_SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY_ENABLE_FALSE                                      0x00000000
#define LWCC97_SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY_ENABLE_TRUE                                       0x00000001

#define LWCC97_MME_DMA_WRITE_METHOD_BARRIER                                                                0x0dec
#define LWCC97_MME_DMA_WRITE_METHOD_BARRIER_V                                                                 0:0

#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_POINTER                                                     0x0df0
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_POINTER_V                                                      7:0

#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION                                                             0x0df4
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_TEST_CC                                                        2:0
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_TEST_CC_FALSE                                           0x00000000
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_TEST_CC_TRUE                                            0x00000001
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_TEST_CC_EQ                                              0x00000002
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_TEST_CC_NE                                              0x00000003
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_TEST_CC_LT                                              0x00000004
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_TEST_CC_LE                                              0x00000005
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_TEST_CC_GT                                              0x00000006
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_TEST_CC_GE                                              0x00000007
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERATION                                                      5:3
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERATION_ADD_PRODUCTS                                  0x00000000
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERATION_SUB_PRODUCTS                                  0x00000001
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERATION_MIN                                           0x00000002
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERATION_MAX                                           0x00000003
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERATION_RCP                                           0x00000004
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERATION_ADD                                           0x00000005
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERATION_SUBTRACT                                      0x00000006
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_CONSTANT_INPUT_SELECT                                          8:6
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_CONSTANT_INPUT_SELECT_CONSTANT0                         0x00000000
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_CONSTANT_INPUT_SELECT_CONSTANT1                         0x00000001
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_CONSTANT_INPUT_SELECT_CONSTANT2                         0x00000002
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_CONSTANT_INPUT_SELECT_CONSTANT3                         0x00000003
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_CONSTANT_INPUT_SELECT_CONSTANT4                         0x00000004
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_CONSTANT_INPUT_SELECT_CONSTANT5                         0x00000005
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_CONSTANT_INPUT_SELECT_CONSTANT6                         0x00000006
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_CONSTANT_INPUT_SELECT_CONSTANT7                         0x00000007
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_A_SELECT                                              11:9
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_A_SELECT_SRC_RGB                                0x00000000
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_A_SELECT_DEST_RGB                               0x00000001
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_A_SELECT_SRC_AAA                                0x00000002
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_A_SELECT_DEST_AAA                               0x00000003
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_A_SELECT_TEMP0_RGB                              0x00000004
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_A_SELECT_TEMP1_RGB                              0x00000005
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_A_SELECT_TEMP2_RGB                              0x00000006
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_A_SELECT_PBR_RGB                                0x00000007
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_B_SELECT                                             15:12
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_B_SELECT_ZERO                                   0x00000000
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_B_SELECT_ONE                                    0x00000001
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_B_SELECT_SRC_RGB                                0x00000002
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_B_SELECT_SRC_AAA                                0x00000003
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_B_SELECT_ONE_MINUS_SRC_AAA                      0x00000004
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_B_SELECT_DEST_RGB                               0x00000005
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_B_SELECT_DEST_AAA                               0x00000006
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_B_SELECT_ONE_MINUS_DEST_AAA                     0x00000007
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_B_SELECT_TEMP0_RGB                              0x00000009
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_B_SELECT_TEMP1_RGB                              0x0000000A
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_B_SELECT_TEMP2_RGB                              0x0000000B
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_B_SELECT_PBR_RGB                                0x0000000C
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_B_SELECT_CONSTANT_RGB                           0x0000000D
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_B_SELECT_ZERO_A_TIMES_B                         0x0000000E
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_C_SELECT                                             18:16
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_C_SELECT_SRC_RGB                                0x00000000
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_C_SELECT_DEST_RGB                               0x00000001
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_C_SELECT_SRC_AAA                                0x00000002
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_C_SELECT_DEST_AAA                               0x00000003
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_C_SELECT_TEMP0_RGB                              0x00000004
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_C_SELECT_TEMP1_RGB                              0x00000005
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_C_SELECT_TEMP2_RGB                              0x00000006
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_C_SELECT_PBR_RGB                                0x00000007
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_D_SELECT                                             22:19
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_D_SELECT_ZERO                                   0x00000000
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_D_SELECT_ONE                                    0x00000001
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_D_SELECT_SRC_RGB                                0x00000002
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_D_SELECT_SRC_AAA                                0x00000003
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_D_SELECT_ONE_MINUS_SRC_AAA                      0x00000004
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_D_SELECT_DEST_RGB                               0x00000005
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_D_SELECT_DEST_AAA                               0x00000006
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_D_SELECT_ONE_MINUS_DEST_AAA                     0x00000007
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_D_SELECT_TEMP0_RGB                              0x00000009
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_D_SELECT_TEMP1_RGB                              0x0000000A
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_D_SELECT_TEMP2_RGB                              0x0000000B
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_D_SELECT_PBR_RGB                                0x0000000C
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_D_SELECT_CONSTANT_RGB                           0x0000000D
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OPERAND_D_SELECT_ZERO_C_TIMES_D                         0x0000000E
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OUTPUT_SWIZZLE                                               25:23
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OUTPUT_SWIZZLE_RGB                                      0x00000000
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OUTPUT_SWIZZLE_GBR                                      0x00000001
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OUTPUT_SWIZZLE_RRR                                      0x00000002
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OUTPUT_SWIZZLE_GGG                                      0x00000003
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OUTPUT_SWIZZLE_BBB                                      0x00000004
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OUTPUT_SWIZZLE_R_TO_A                                   0x00000005
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OUTPUT_WRITE_MASK                                            27:26
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OUTPUT_WRITE_MASK_RGB                                   0x00000000
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OUTPUT_WRITE_MASK_R_ONLY                                0x00000001
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OUTPUT_WRITE_MASK_G_ONLY                                0x00000002
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_OUTPUT_WRITE_MASK_B_ONLY                                0x00000003
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_PASS_OUTPUT                                                  29:28
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_PASS_OUTPUT_TEMP0                                       0x00000000
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_PASS_OUTPUT_TEMP1                                       0x00000001
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_PASS_OUTPUT_TEMP2                                       0x00000002
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_PASS_OUTPUT_NONE                                        0x00000003
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_SET_CC                                                       31:31
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_SET_CC_FALSE                                            0x00000000
#define LWCC97_LOAD_ITERATED_BLEND_INSTRUCTION_SET_CC_TRUE                                             0x00000001

#define LWCC97_SET_WINDOW_OFFSET_X                                                                         0x0df8
#define LWCC97_SET_WINDOW_OFFSET_X_V                                                                         16:0

#define LWCC97_SET_WINDOW_OFFSET_Y                                                                         0x0dfc
#define LWCC97_SET_WINDOW_OFFSET_Y_V                                                                         17:0

#define LWCC97_SET_SCISSOR_ENABLE(j)                                                              (0x0e00+(j)*16)
#define LWCC97_SET_SCISSOR_ENABLE_V                                                                           0:0
#define LWCC97_SET_SCISSOR_ENABLE_V_FALSE                                                              0x00000000
#define LWCC97_SET_SCISSOR_ENABLE_V_TRUE                                                               0x00000001

#define LWCC97_SET_SCISSOR_HORIZONTAL(j)                                                          (0x0e04+(j)*16)
#define LWCC97_SET_SCISSOR_HORIZONTAL_XMIN                                                                   15:0
#define LWCC97_SET_SCISSOR_HORIZONTAL_XMAX                                                                  31:16

#define LWCC97_SET_SCISSOR_VERTICAL(j)                                                            (0x0e08+(j)*16)
#define LWCC97_SET_SCISSOR_VERTICAL_YMIN                                                                     15:0
#define LWCC97_SET_SCISSOR_VERTICAL_YMAX                                                                    31:16

#define LWCC97_SET_ST_A                                                                                    0x0f00
#define LWCC97_SET_ST_A_OFFSET_UPPER                                                                          7:0

#define LWCC97_SET_ST_B                                                                                    0x0f04
#define LWCC97_SET_ST_B_OFFSET_LOWER                                                                         31:0

#define LWCC97_SET_ST_BLOCK_SIZE                                                                           0x0f08
#define LWCC97_SET_ST_BLOCK_SIZE_WIDTH                                                                        3:0
#define LWCC97_SET_ST_BLOCK_SIZE_WIDTH_ONE_GOB                                                         0x00000000
#define LWCC97_SET_ST_BLOCK_SIZE_HEIGHT                                                                       7:4
#define LWCC97_SET_ST_BLOCK_SIZE_HEIGHT_ONE_GOB                                                        0x00000000
#define LWCC97_SET_ST_BLOCK_SIZE_HEIGHT_TWO_GOBS                                                       0x00000001
#define LWCC97_SET_ST_BLOCK_SIZE_HEIGHT_FOUR_GOBS                                                      0x00000002
#define LWCC97_SET_ST_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                                                     0x00000003
#define LWCC97_SET_ST_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                                                   0x00000004
#define LWCC97_SET_ST_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                                                 0x00000005
#define LWCC97_SET_ST_BLOCK_SIZE_DEPTH                                                                       11:8
#define LWCC97_SET_ST_BLOCK_SIZE_DEPTH_ONE_GOB                                                         0x00000000

#define LWCC97_SET_ST_ARRAY_PITCH                                                                          0x0f0c
#define LWCC97_SET_ST_ARRAY_PITCH_V                                                                          31:0

#define LWCC97_SET_VPC_PERF_KNOB                                                                           0x0f14
#define LWCC97_SET_VPC_PERF_KNOB_LWLLED_SMALL_LINES                                                           7:0
#define LWCC97_SET_VPC_PERF_KNOB_LWLLED_SMALL_TRIANGLES                                                      15:8
#define LWCC97_SET_VPC_PERF_KNOB_NONLWLLED_LINES_AND_POINTS                                                 23:16
#define LWCC97_SET_VPC_PERF_KNOB_NONLWLLED_TRIANGLES                                                        31:24

#define LWCC97_PM_LOCAL_TRIGGER                                                                            0x0f18
#define LWCC97_PM_LOCAL_TRIGGER_BOOKMARK                                                                     15:0

#define LWCC97_SET_POST_Z_PS_IMASK                                                                         0x0f1c
#define LWCC97_SET_POST_Z_PS_IMASK_ENABLE                                                                     0:0
#define LWCC97_SET_POST_Z_PS_IMASK_ENABLE_FALSE                                                        0x00000000
#define LWCC97_SET_POST_Z_PS_IMASK_ENABLE_TRUE                                                         0x00000001

#define LWCC97_SET_MULTI_VIEW_RENDER_TARGET_ARRAY_INDEX_OFFSET(i)                                  (0x0f20+(i)*4)
#define LWCC97_SET_MULTI_VIEW_RENDER_TARGET_ARRAY_INDEX_OFFSET_V                                             10:0

#define LWCC97_SET_SPARE_MULTI_VIEW_RENDER_TARGET_ARRAY_INDEX_OFFSET(i)                            (0x0f30+(i)*4)
#define LWCC97_SET_SPARE_MULTI_VIEW_RENDER_TARGET_ARRAY_INDEX_OFFSET_V                                       31:0

#define LWCC97_SET_CONSTANT_COLOR_RENDERING                                                                0x0f40
#define LWCC97_SET_CONSTANT_COLOR_RENDERING_ENABLE                                                            0:0
#define LWCC97_SET_CONSTANT_COLOR_RENDERING_ENABLE_FALSE                                               0x00000000
#define LWCC97_SET_CONSTANT_COLOR_RENDERING_ENABLE_TRUE                                                0x00000001

#define LWCC97_SET_CONSTANT_COLOR_RENDERING_RED                                                            0x0f44
#define LWCC97_SET_CONSTANT_COLOR_RENDERING_RED_V                                                            31:0

#define LWCC97_SET_CONSTANT_COLOR_RENDERING_GREEN                                                          0x0f48
#define LWCC97_SET_CONSTANT_COLOR_RENDERING_GREEN_V                                                          31:0

#define LWCC97_SET_CONSTANT_COLOR_RENDERING_BLUE                                                           0x0f4c
#define LWCC97_SET_CONSTANT_COLOR_RENDERING_BLUE_V                                                           31:0

#define LWCC97_SET_CONSTANT_COLOR_RENDERING_ALPHA                                                          0x0f50
#define LWCC97_SET_CONSTANT_COLOR_RENDERING_ALPHA_V                                                          31:0

#define LWCC97_SET_BACK_STENCIL_FUNC_REF                                                                   0x0f54
#define LWCC97_SET_BACK_STENCIL_FUNC_REF_V                                                                    7:0

#define LWCC97_SET_BACK_STENCIL_MASK                                                                       0x0f58
#define LWCC97_SET_BACK_STENCIL_MASK_V                                                                        7:0

#define LWCC97_SET_BACK_STENCIL_FUNC_MASK                                                                  0x0f5c
#define LWCC97_SET_BACK_STENCIL_FUNC_MASK_V                                                                   7:0

#define LWCC97_SET_TILED_CACHE                                                                             0x0f60
#define LWCC97_SET_TILED_CACHE_ENABLE                                                                         0:0
#define LWCC97_SET_TILED_CACHE_ENABLE_FALSE                                                            0x00000000
#define LWCC97_SET_TILED_CACHE_ENABLE_TRUE                                                             0x00000001

#define LWCC97_SET_TILED_CACHE_TILE_SIZE                                                                   0x0f64
#define LWCC97_SET_TILED_CACHE_TILE_SIZE_PIXELS_X                                                            15:0
#define LWCC97_SET_TILED_CACHE_TILE_SIZE_PIXELS_Y                                                           31:16

#define LWCC97_SET_TILED_CACHE_BUFFER_INTERLEAVE                                                           0x0f68
#define LWCC97_SET_TILED_CACHE_BUFFER_INTERLEAVE_BETA_CB_COUNT                                                2:0
#define LWCC97_SET_TILED_CACHE_BUFFER_INTERLEAVE_PAGE_POOL_CB_COUNT                                          10:8
#define LWCC97_SET_TILED_CACHE_BUFFER_INTERLEAVE_CBTBL_INDEX_COUNT                                            5:3
#define LWCC97_SET_TILED_CACHE_BUFFER_INTERLEAVE_BINNER_COUNT                                               14:12
#define LWCC97_SET_TILED_CACHE_BUFFER_INTERLEAVE_RTV_CB_COUNT                                               18:16

#define LWCC97_SET_TILED_CACHE_CONTROL                                                                     0x0f6c
#define LWCC97_SET_TILED_CACHE_CONTROL_MIX_EARLY_LATE_Z_ENABLE                                                0:0
#define LWCC97_SET_TILED_CACHE_CONTROL_MIX_EARLY_LATE_Z_ENABLE_FALSE                                   0x00000000
#define LWCC97_SET_TILED_CACHE_CONTROL_MIX_EARLY_LATE_Z_ENABLE_TRUE                                    0x00000001
#define LWCC97_SET_TILED_CACHE_CONTROL_LAZY_STATE_UPDATE_ENABLE                                               1:1
#define LWCC97_SET_TILED_CACHE_CONTROL_LAZY_STATE_UPDATE_ENABLE_FALSE                                  0x00000000
#define LWCC97_SET_TILED_CACHE_CONTROL_LAZY_STATE_UPDATE_ENABLE_TRUE                                   0x00000001
#define LWCC97_SET_TILED_CACHE_CONTROL_LEVEL0_BBOX_PRIM_COUNT                                                14:2
#define LWCC97_SET_TILED_CACHE_CONTROL_LEVEL1_BBOX_PAD_X                                                    22:15
#define LWCC97_SET_TILED_CACHE_CONTROL_LEVEL1_BBOX_PAD_Y                                                    30:23

#define LWCC97_SET_TILED_CACHE_STATE_THRESHOLD                                                             0x0f70
#define LWCC97_SET_TILED_CACHE_STATE_THRESHOLD_ACTION_WHEN_COUNT_REACHED                                      1:0
#define LWCC97_SET_TILED_CACHE_STATE_THRESHOLD_ACTION_WHEN_COUNT_REACHED_NONE                          0x00000000
#define LWCC97_SET_TILED_CACHE_STATE_THRESHOLD_ACTION_WHEN_COUNT_REACHED_FLUSH_WITH_TILED_CACHING             0x00000001
#define LWCC97_SET_TILED_CACHE_STATE_THRESHOLD_ACTION_WHEN_COUNT_REACHED_FLUSH_WITHOUT_TILED_CACHING             0x00000002
#define LWCC97_SET_TILED_CACHE_STATE_THRESHOLD_COUNT                                                         23:4

#define LWCC97_TILED_CACHE_ILWALIDATE_TEXTURE_DATA_CACHE                                                   0x0f74
#define LWCC97_TILED_CACHE_ILWALIDATE_TEXTURE_DATA_CACHE_V                                                    0:0

#define LWCC97_TILED_CACHE_ILWALIDATE                                                                      0x0f78
#define LWCC97_TILED_CACHE_ILWALIDATE_TARGET_TYPE_SELECT                                                      0:0
#define LWCC97_TILED_CACHE_ILWALIDATE_TARGET_TYPE_SELECT_COLOR                                         0x00000000
#define LWCC97_TILED_CACHE_ILWALIDATE_TARGET_TYPE_SELECT_ZETA                                          0x00000001
#define LWCC97_TILED_CACHE_ILWALIDATE_TARGET_SELECT                                                           6:4
#define LWCC97_TILED_CACHE_ILWALIDATE_LAYER_SELECT                                                           23:8
#define LWCC97_TILED_CACHE_ILWALIDATE_COMP_BIT_ONLY                                                         24:24
#define LWCC97_TILED_CACHE_ILWALIDATE_COMP_BIT_ONLY_FALSE                                              0x00000000
#define LWCC97_TILED_CACHE_ILWALIDATE_COMP_BIT_ONLY_TRUE                                               0x00000001

#define LWCC97_TILED_CACHE_BARRIER                                                                         0x0f7c
#define LWCC97_TILED_CACHE_BARRIER_V                                                                          0:0
#define LWCC97_TILED_CACHE_BARRIER_BARRIER_LOCATION                                                           1:1
#define LWCC97_TILED_CACHE_BARRIER_BARRIER_LOCATION_BLOCK_BEFORE_PS                                    0x00000000
#define LWCC97_TILED_CACHE_BARRIER_BARRIER_LOCATION_BLOCK_BEFORE_PS_AND_ZTEST                          0x00000001

#define LWCC97_TILED_CACHE_FLUSH                                                                           0x0f80
#define LWCC97_TILED_CACHE_FLUSH_TYPE                                                                         0:0
#define LWCC97_TILED_CACHE_FLUSH_TYPE_FLUSH_WITH_TILED_CACHING                                         0x00000000
#define LWCC97_TILED_CACHE_FLUSH_TYPE_FLUSH_WITHOUT_TILED_CACHING                                      0x00000001

#define LWCC97_SET_VERTEX_STREAM_SUBSTITUTE_A                                                              0x0f84
#define LWCC97_SET_VERTEX_STREAM_SUBSTITUTE_A_ADDRESS_UPPER                                                   7:0

#define LWCC97_SET_VERTEX_STREAM_SUBSTITUTE_B                                                              0x0f88
#define LWCC97_SET_VERTEX_STREAM_SUBSTITUTE_B_ADDRESS_LOWER                                                  31:0

#define LWCC97_SET_LINE_MODE_POLYGON_CLIP                                                                  0x0f8c
#define LWCC97_SET_LINE_MODE_POLYGON_CLIP_GENERATED_EDGE                                                      0:0
#define LWCC97_SET_LINE_MODE_POLYGON_CLIP_GENERATED_EDGE_DRAW_LINE                                     0x00000000
#define LWCC97_SET_LINE_MODE_POLYGON_CLIP_GENERATED_EDGE_DO_NOT_DRAW_LINE                              0x00000001

#define LWCC97_SET_SINGLE_CT_WRITE_CONTROL                                                                 0x0f90
#define LWCC97_SET_SINGLE_CT_WRITE_CONTROL_ENABLE                                                             0:0
#define LWCC97_SET_SINGLE_CT_WRITE_CONTROL_ENABLE_FALSE                                                0x00000000
#define LWCC97_SET_SINGLE_CT_WRITE_CONTROL_ENABLE_TRUE                                                 0x00000001

#define LWCC97_MULTI_PASS_INTERVAL_BOUNDARY                                                                0x0f94
#define LWCC97_MULTI_PASS_INTERVAL_BOUNDARY_PASS_ENABLE                                                       3:0
#define LWCC97_MULTI_PASS_INTERVAL_BOUNDARY_PASS0_STATE                                                       5:4
#define LWCC97_MULTI_PASS_INTERVAL_BOUNDARY_PASS1_STATE                                                       9:8
#define LWCC97_MULTI_PASS_INTERVAL_BOUNDARY_PASS2_STATE                                                     13:12
#define LWCC97_MULTI_PASS_INTERVAL_BOUNDARY_PASS3_STATE                                                     17:16

#define LWCC97_SET_VTG_WARP_WATERMARKS                                                                     0x0f98
#define LWCC97_SET_VTG_WARP_WATERMARKS_LOW                                                                   15:0
#define LWCC97_SET_VTG_WARP_WATERMARKS_HIGH                                                                 31:16

#define LWCC97_SET_DEPTH_BOUNDS_MIN                                                                        0x0f9c
#define LWCC97_SET_DEPTH_BOUNDS_MIN_V                                                                        31:0

#define LWCC97_SET_DEPTH_BOUNDS_MAX                                                                        0x0fa0
#define LWCC97_SET_DEPTH_BOUNDS_MAX_V                                                                        31:0

#define LWCC97_SET_SAMPLE_MASK                                                                             0x0fa4
#define LWCC97_SET_SAMPLE_MASK_RASTER_OUT_ENABLE                                                              0:0
#define LWCC97_SET_SAMPLE_MASK_RASTER_OUT_ENABLE_FALSE                                                 0x00000000
#define LWCC97_SET_SAMPLE_MASK_RASTER_OUT_ENABLE_TRUE                                                  0x00000001
#define LWCC97_SET_SAMPLE_MASK_COLOR_TARGET_ENABLE                                                            4:4
#define LWCC97_SET_SAMPLE_MASK_COLOR_TARGET_ENABLE_FALSE                                               0x00000000
#define LWCC97_SET_SAMPLE_MASK_COLOR_TARGET_ENABLE_TRUE                                                0x00000001

#define LWCC97_SET_COLOR_TARGET_SAMPLE_MASK                                                                0x0fa8
#define LWCC97_SET_COLOR_TARGET_SAMPLE_MASK_V                                                                15:0

#define LWCC97_SET_CT_MRT_ENABLE                                                                           0x0fac
#define LWCC97_SET_CT_MRT_ENABLE_V                                                                            0:0
#define LWCC97_SET_CT_MRT_ENABLE_V_FALSE                                                               0x00000000
#define LWCC97_SET_CT_MRT_ENABLE_V_TRUE                                                                0x00000001

#define LWCC97_SET_NONMULTISAMPLED_Z                                                                       0x0fb0
#define LWCC97_SET_NONMULTISAMPLED_Z_V                                                                        0:0
#define LWCC97_SET_NONMULTISAMPLED_Z_V_PER_SAMPLE                                                      0x00000000
#define LWCC97_SET_NONMULTISAMPLED_Z_V_AT_PIXEL_CENTER                                                 0x00000001

#define LWCC97_SET_TIR                                                                                     0x0fb4
#define LWCC97_SET_TIR_MODE                                                                                   1:0
#define LWCC97_SET_TIR_MODE_DISABLED                                                                   0x00000000
#define LWCC97_SET_TIR_MODE_RASTER_N_TARGET_M                                                          0x00000001

#define LWCC97_SET_ANTI_ALIAS_RASTER                                                                       0x0fb8
#define LWCC97_SET_ANTI_ALIAS_RASTER_SAMPLES                                                                  2:0
#define LWCC97_SET_ANTI_ALIAS_RASTER_SAMPLES_MODE_1X1                                                  0x00000000
#define LWCC97_SET_ANTI_ALIAS_RASTER_SAMPLES_MODE_2X2                                                  0x00000002
#define LWCC97_SET_ANTI_ALIAS_RASTER_SAMPLES_MODE_4X2_D3D                                              0x00000004
#define LWCC97_SET_ANTI_ALIAS_RASTER_SAMPLES_MODE_2X1_D3D                                              0x00000005
#define LWCC97_SET_ANTI_ALIAS_RASTER_SAMPLES_MODE_4X4                                                  0x00000006

#define LWCC97_SET_SAMPLE_MASK_X0_Y0                                                                       0x0fbc
#define LWCC97_SET_SAMPLE_MASK_X0_Y0_V                                                                       15:0

#define LWCC97_SET_SAMPLE_MASK_X1_Y0                                                                       0x0fc0
#define LWCC97_SET_SAMPLE_MASK_X1_Y0_V                                                                       15:0

#define LWCC97_SET_SAMPLE_MASK_X0_Y1                                                                       0x0fc4
#define LWCC97_SET_SAMPLE_MASK_X0_Y1_V                                                                       15:0

#define LWCC97_SET_SAMPLE_MASK_X1_Y1                                                                       0x0fc8
#define LWCC97_SET_SAMPLE_MASK_X1_Y1_V                                                                       15:0

#define LWCC97_SET_SURFACE_CLIP_ID_MEMORY_A                                                                0x0fcc
#define LWCC97_SET_SURFACE_CLIP_ID_MEMORY_A_OFFSET_UPPER                                                      7:0

#define LWCC97_SET_SURFACE_CLIP_ID_MEMORY_B                                                                0x0fd0
#define LWCC97_SET_SURFACE_CLIP_ID_MEMORY_B_OFFSET_LOWER                                                     31:0

#define LWCC97_SET_TIR_MODULATION                                                                          0x0fd4
#define LWCC97_SET_TIR_MODULATION_COMPONENT_SELECT                                                            1:0
#define LWCC97_SET_TIR_MODULATION_COMPONENT_SELECT_NO_MODULATION                                       0x00000000
#define LWCC97_SET_TIR_MODULATION_COMPONENT_SELECT_MODULATE_RGB                                        0x00000001
#define LWCC97_SET_TIR_MODULATION_COMPONENT_SELECT_MODULATE_ALPHA_ONLY                                 0x00000002
#define LWCC97_SET_TIR_MODULATION_COMPONENT_SELECT_MODULATE_RGBA                                       0x00000003

#define LWCC97_SET_TIR_MODULATION_FUNCTION                                                                 0x0fd8
#define LWCC97_SET_TIR_MODULATION_FUNCTION_SELECT                                                             0:0
#define LWCC97_SET_TIR_MODULATION_FUNCTION_SELECT_LINEAR                                               0x00000000
#define LWCC97_SET_TIR_MODULATION_FUNCTION_SELECT_TABLE                                                0x00000001

#define LWCC97_SET_BLEND_OPT_CONTROL                                                                       0x0fdc
#define LWCC97_SET_BLEND_OPT_CONTROL_ALLOW_FLOAT_PIXEL_KILLS                                                  0:0
#define LWCC97_SET_BLEND_OPT_CONTROL_ALLOW_FLOAT_PIXEL_KILLS_FALSE                                     0x00000000
#define LWCC97_SET_BLEND_OPT_CONTROL_ALLOW_FLOAT_PIXEL_KILLS_TRUE                                      0x00000001

#define LWCC97_SET_ZT_A                                                                                    0x0fe0
#define LWCC97_SET_ZT_A_OFFSET_UPPER                                                                          7:0

#define LWCC97_SET_ZT_B                                                                                    0x0fe4
#define LWCC97_SET_ZT_B_OFFSET_LOWER                                                                         31:0

#define LWCC97_SET_ZT_FORMAT                                                                               0x0fe8
#define LWCC97_SET_ZT_FORMAT_V                                                                                4:0
#define LWCC97_SET_ZT_FORMAT_V_Z16                                                                     0x00000013
#define LWCC97_SET_ZT_FORMAT_V_Z24S8                                                                   0x00000014
#define LWCC97_SET_ZT_FORMAT_V_X8Z24                                                                   0x00000015
#define LWCC97_SET_ZT_FORMAT_V_S8Z24                                                                   0x00000016
#define LWCC97_SET_ZT_FORMAT_V_S8                                                                      0x00000017
#define LWCC97_SET_ZT_FORMAT_V_ZF32                                                                    0x0000000A
#define LWCC97_SET_ZT_FORMAT_V_ZF32_X24S8                                                              0x00000019
#define LWCC97_SET_ZT_FORMAT_STENCIL_IS_SEPARATE                                                              8:8
#define LWCC97_SET_ZT_FORMAT_STENCIL_IS_SEPARATE_FALSE                                                 0x00000000
#define LWCC97_SET_ZT_FORMAT_STENCIL_IS_SEPARATE_TRUE                                                  0x00000001

#define LWCC97_SET_ZT_BLOCK_SIZE                                                                           0x0fec
#define LWCC97_SET_ZT_BLOCK_SIZE_WIDTH                                                                        3:0
#define LWCC97_SET_ZT_BLOCK_SIZE_WIDTH_ONE_GOB                                                         0x00000000
#define LWCC97_SET_ZT_BLOCK_SIZE_HEIGHT                                                                       7:4
#define LWCC97_SET_ZT_BLOCK_SIZE_HEIGHT_ONE_GOB                                                        0x00000000
#define LWCC97_SET_ZT_BLOCK_SIZE_HEIGHT_TWO_GOBS                                                       0x00000001
#define LWCC97_SET_ZT_BLOCK_SIZE_HEIGHT_FOUR_GOBS                                                      0x00000002
#define LWCC97_SET_ZT_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                                                     0x00000003
#define LWCC97_SET_ZT_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                                                   0x00000004
#define LWCC97_SET_ZT_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                                                 0x00000005
#define LWCC97_SET_ZT_BLOCK_SIZE_DEPTH                                                                       11:8
#define LWCC97_SET_ZT_BLOCK_SIZE_DEPTH_ONE_GOB                                                         0x00000000

#define LWCC97_SET_ZT_ARRAY_PITCH                                                                          0x0ff0
#define LWCC97_SET_ZT_ARRAY_PITCH_V                                                                          31:0

#define LWCC97_SET_SURFACE_CLIP_HORIZONTAL                                                                 0x0ff4
#define LWCC97_SET_SURFACE_CLIP_HORIZONTAL_X                                                                 15:0
#define LWCC97_SET_SURFACE_CLIP_HORIZONTAL_WIDTH                                                            31:16

#define LWCC97_SET_SURFACE_CLIP_VERTICAL                                                                   0x0ff8
#define LWCC97_SET_SURFACE_CLIP_VERTICAL_Y                                                                   15:0
#define LWCC97_SET_SURFACE_CLIP_VERTICAL_HEIGHT                                                             31:16

#define LWCC97_SET_L2_CACHE_CONTROL_FOR_VAF_REQUESTS                                                       0x1000
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_VAF_REQUESTS_SYSTEM_MEMORY_VOLATILE                                   0:0
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_VAF_REQUESTS_SYSTEM_MEMORY_VOLATILE_FALSE                      0x00000000
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_VAF_REQUESTS_SYSTEM_MEMORY_VOLATILE_TRUE                       0x00000001
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_VAF_REQUESTS_POLICY                                                   5:4
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_VAF_REQUESTS_POLICY_EVICT_FIRST                                0x00000000
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_VAF_REQUESTS_POLICY_EVICT_NORMAL                               0x00000001
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_VAF_REQUESTS_POLICY_EVICT_LAST                                 0x00000002

#define LWCC97_SET_VIEWPORT_MULTICAST                                                                      0x1004
#define LWCC97_SET_VIEWPORT_MULTICAST_ORDER                                                                   0:0
#define LWCC97_SET_VIEWPORT_MULTICAST_ORDER_VIEWPORT_ORDER                                             0x00000000
#define LWCC97_SET_VIEWPORT_MULTICAST_ORDER_PRIMITIVE_ORDER                                            0x00000001

#define LWCC97_SET_TESSELLATION_LWT_HEIGHT                                                                 0x1008
#define LWCC97_SET_TESSELLATION_LWT_HEIGHT_V                                                                  4:0

#define LWCC97_SET_MAX_GS_INSTANCES_PER_TASK                                                               0x100c
#define LWCC97_SET_MAX_GS_INSTANCES_PER_TASK_V                                                               10:0

#define LWCC97_SET_MAX_GS_OUTPUT_VERTICES_PER_TASK                                                         0x1010
#define LWCC97_SET_MAX_GS_OUTPUT_VERTICES_PER_TASK_V                                                         15:0

#define LWCC97_SET_RESERVED_SW_METHOD00                                                                    0x1014
#define LWCC97_SET_RESERVED_SW_METHOD00_V                                                                    31:0

#define LWCC97_SET_GS_OUTPUT_CB_STORAGE_MULTIPLIER                                                         0x1018
#define LWCC97_SET_GS_OUTPUT_CB_STORAGE_MULTIPLIER_V                                                          9:0

#define LWCC97_SET_BETA_CB_STORAGE_CONSTRAINT                                                              0x101c
#define LWCC97_SET_BETA_CB_STORAGE_CONSTRAINT_ENABLE                                                          0:0
#define LWCC97_SET_BETA_CB_STORAGE_CONSTRAINT_ENABLE_FALSE                                             0x00000000
#define LWCC97_SET_BETA_CB_STORAGE_CONSTRAINT_ENABLE_TRUE                                              0x00000001

#define LWCC97_SET_TI_OUTPUT_CB_STORAGE_MULTIPLIER                                                         0x1020
#define LWCC97_SET_TI_OUTPUT_CB_STORAGE_MULTIPLIER_V                                                          9:0

#define LWCC97_SET_ALPHA_CB_STORAGE_CONSTRAINT                                                             0x1024
#define LWCC97_SET_ALPHA_CB_STORAGE_CONSTRAINT_ENABLE                                                         0:0
#define LWCC97_SET_ALPHA_CB_STORAGE_CONSTRAINT_ENABLE_FALSE                                            0x00000000
#define LWCC97_SET_ALPHA_CB_STORAGE_CONSTRAINT_ENABLE_TRUE                                             0x00000001

#define LWCC97_SET_RESERVED_SW_METHOD01                                                                    0x1028
#define LWCC97_SET_RESERVED_SW_METHOD01_V                                                                    31:0

#define LWCC97_SET_RESERVED_SW_METHOD02                                                                    0x102c
#define LWCC97_SET_RESERVED_SW_METHOD02_V                                                                    31:0

#define LWCC97_SET_TIR_MODULATION_COEFFICIENT_TABLE(i)                                             (0x1030+(i)*4)
#define LWCC97_SET_TIR_MODULATION_COEFFICIENT_TABLE_V0                                                        7:0
#define LWCC97_SET_TIR_MODULATION_COEFFICIENT_TABLE_V1                                                       15:8
#define LWCC97_SET_TIR_MODULATION_COEFFICIENT_TABLE_V2                                                      23:16
#define LWCC97_SET_TIR_MODULATION_COEFFICIENT_TABLE_V3                                                      31:24

#define LWCC97_SET_SPARE_NOOP01                                                                            0x1044
#define LWCC97_SET_SPARE_NOOP01_V                                                                            31:0

#define LWCC97_SET_SPARE_NOOP02                                                                            0x1048
#define LWCC97_SET_SPARE_NOOP02_V                                                                            31:0

#define LWCC97_SET_SPARE_NOOP03                                                                            0x104c
#define LWCC97_SET_SPARE_NOOP03_V                                                                            31:0

#define LWCC97_SET_SPARE_NOOP04                                                                            0x1050
#define LWCC97_SET_SPARE_NOOP04_V                                                                            31:0

#define LWCC97_SET_SPARE_NOOP05                                                                            0x1054
#define LWCC97_SET_SPARE_NOOP05_V                                                                            31:0

#define LWCC97_SET_SPARE_NOOP06                                                                            0x1058
#define LWCC97_SET_SPARE_NOOP06_V                                                                            31:0

#define LWCC97_SET_SPARE_NOOP07                                                                            0x105c
#define LWCC97_SET_SPARE_NOOP07_V                                                                            31:0

#define LWCC97_SET_SPARE_NOOP08                                                                            0x1060
#define LWCC97_SET_SPARE_NOOP08_V                                                                            31:0

#define LWCC97_SET_SPARE_NOOP09                                                                            0x1064
#define LWCC97_SET_SPARE_NOOP09_V                                                                            31:0

#define LWCC97_SET_SPARE_NOOP10                                                                            0x1068
#define LWCC97_SET_SPARE_NOOP10_V                                                                            31:0

#define LWCC97_SET_SPARE_NOOP11                                                                            0x106c
#define LWCC97_SET_SPARE_NOOP11_V                                                                            31:0

#define LWCC97_SET_SPARE_NOOP12                                                                            0x1070
#define LWCC97_SET_SPARE_NOOP12_V                                                                            31:0

#define LWCC97_SET_SPARE_NOOP13                                                                            0x1074
#define LWCC97_SET_SPARE_NOOP13_V                                                                            31:0

#define LWCC97_SET_SPARE_NOOP14                                                                            0x1078
#define LWCC97_SET_SPARE_NOOP14_V                                                                            31:0

#define LWCC97_SET_SPARE_NOOP15                                                                            0x107c
#define LWCC97_SET_SPARE_NOOP15_V                                                                            31:0

#define LWCC97_SET_PIXELS_FROM_MEMORY_DST_X0                                                               0x1080
#define LWCC97_SET_PIXELS_FROM_MEMORY_DST_X0_V                                                               31:0

#define LWCC97_SET_PIXELS_FROM_MEMORY_DST_Y0                                                               0x1084
#define LWCC97_SET_PIXELS_FROM_MEMORY_DST_Y0_V                                                               31:0

#define LWCC97_SET_PIXELS_FROM_MEMORY_DST_WIDTH                                                            0x1088
#define LWCC97_SET_PIXELS_FROM_MEMORY_DST_WIDTH_V                                                            31:0

#define LWCC97_SET_PIXELS_FROM_MEMORY_DST_HEIGHT                                                           0x108c
#define LWCC97_SET_PIXELS_FROM_MEMORY_DST_HEIGHT_V                                                           31:0

#define LWCC97_SET_PIXELS_FROM_MEMORY_DU_DX_FRAC                                                           0x1090
#define LWCC97_SET_PIXELS_FROM_MEMORY_DU_DX_FRAC_V                                                           31:0

#define LWCC97_SET_PIXELS_FROM_MEMORY_DU_DX_INT                                                            0x1094
#define LWCC97_SET_PIXELS_FROM_MEMORY_DU_DX_INT_V                                                            31:0

#define LWCC97_SET_PIXELS_FROM_MEMORY_DV_DY_FRAC                                                           0x1098
#define LWCC97_SET_PIXELS_FROM_MEMORY_DV_DY_FRAC_V                                                           31:0

#define LWCC97_SET_PIXELS_FROM_MEMORY_DV_DY_INT                                                            0x109c
#define LWCC97_SET_PIXELS_FROM_MEMORY_DV_DY_INT_V                                                            31:0

#define LWCC97_SET_PIXELS_FROM_MEMORY_TEXTURE_SAMPLER                                                      0x10a0
#define LWCC97_SET_PIXELS_FROM_MEMORY_TEXTURE_SAMPLER_INDEX                                                  11:0

#define LWCC97_SET_PIXELS_FROM_MEMORY_TEXTURE_HEADER                                                       0x10a4
#define LWCC97_SET_PIXELS_FROM_MEMORY_TEXTURE_HEADER_INDEX                                                   19:0

#define LWCC97_SET_PIXELS_FROM_MEMORY_SRC_X0                                                               0x10a8
#define LWCC97_SET_PIXELS_FROM_MEMORY_SRC_X0_V                                                               31:0

#define LWCC97_PIXELS_FROM_MEMORY_SRC_Y0                                                                   0x10ac
#define LWCC97_PIXELS_FROM_MEMORY_SRC_Y0_V                                                                   31:0

#define LWCC97_SET_RESERVED_SW_METHOD03                                                                    0x10b0
#define LWCC97_SET_RESERVED_SW_METHOD03_V                                                                    31:0

#define LWCC97_SET_RESERVED_SW_METHOD04                                                                    0x10b4
#define LWCC97_SET_RESERVED_SW_METHOD04_V                                                                    31:0

#define LWCC97_SET_RESERVED_SW_METHOD05                                                                    0x10b8
#define LWCC97_SET_RESERVED_SW_METHOD05_V                                                                    31:0

#define LWCC97_SET_RESERVED_SW_METHOD06                                                                    0x10bc
#define LWCC97_SET_RESERVED_SW_METHOD06_V                                                                    31:0

#define LWCC97_SET_RESERVED_SW_METHOD07                                                                    0x10c0
#define LWCC97_SET_RESERVED_SW_METHOD07_V                                                                    31:0

#define LWCC97_SET_RESERVED_SW_METHOD08                                                                    0x10c4
#define LWCC97_SET_RESERVED_SW_METHOD08_V                                                                    31:0

#define LWCC97_SET_RESERVED_SW_METHOD09                                                                    0x10c8
#define LWCC97_SET_RESERVED_SW_METHOD09_V                                                                    31:0

#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_UNORM8                                                          0x10cc
#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_UNORM8_ALL_COVERED_ALL_HIT_ONCE                                    7:0
#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_UNORM8_ALL_COVERED                                               23:16

#define LWCC97_SET_RESERVED_SW_METHOD10                                                                    0x10d0
#define LWCC97_SET_RESERVED_SW_METHOD10_V                                                                    31:0

#define LWCC97_SET_RESERVED_SW_METHOD11                                                                    0x10d4
#define LWCC97_SET_RESERVED_SW_METHOD11_V                                                                    31:0

#define LWCC97_SET_RESERVED_SW_METHOD12                                                                    0x10d8
#define LWCC97_SET_RESERVED_SW_METHOD12_V                                                                    31:0

#define LWCC97_SET_RESERVED_SW_METHOD13                                                                    0x10dc
#define LWCC97_SET_RESERVED_SW_METHOD13_V                                                                    31:0

#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_UNORM10                                                         0x10e0
#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_UNORM10_ALL_COVERED_ALL_HIT_ONCE                                   7:0
#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_UNORM10_ALL_COVERED                                              23:16

#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_UNORM16                                                         0x10e4
#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_UNORM16_ALL_COVERED_ALL_HIT_ONCE                                   7:0
#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_UNORM16_ALL_COVERED                                              23:16

#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_FP11                                                            0x10e8
#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_FP11_ALL_COVERED_ALL_HIT_ONCE                                      5:0
#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_FP11_ALL_COVERED                                                 21:16

#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_FP16                                                            0x10ec
#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_FP16_ALL_COVERED_ALL_HIT_ONCE                                      7:0
#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_FP16_ALL_COVERED                                                 23:16

#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_SRGB8                                                           0x10f0
#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_SRGB8_ALL_COVERED_ALL_HIT_ONCE                                     7:0
#define LWCC97_SET_REDUCE_COLOR_THRESHOLDS_SRGB8_ALL_COVERED                                                23:16

#define LWCC97_UNBIND_ALL                                                                                  0x10f4
#define LWCC97_UNBIND_ALL_CONSTANT_BUFFERS                                                                    8:8
#define LWCC97_UNBIND_ALL_CONSTANT_BUFFERS_FALSE                                                       0x00000000
#define LWCC97_UNBIND_ALL_CONSTANT_BUFFERS_TRUE                                                        0x00000001

#define LWCC97_SET_CLEAR_SURFACE_CONTROL                                                                   0x10f8
#define LWCC97_SET_CLEAR_SURFACE_CONTROL_RESPECT_STENCIL_MASK                                                 0:0
#define LWCC97_SET_CLEAR_SURFACE_CONTROL_RESPECT_STENCIL_MASK_FALSE                                    0x00000000
#define LWCC97_SET_CLEAR_SURFACE_CONTROL_RESPECT_STENCIL_MASK_TRUE                                     0x00000001
#define LWCC97_SET_CLEAR_SURFACE_CONTROL_USE_CLEAR_RECT                                                       4:4
#define LWCC97_SET_CLEAR_SURFACE_CONTROL_USE_CLEAR_RECT_FALSE                                          0x00000000
#define LWCC97_SET_CLEAR_SURFACE_CONTROL_USE_CLEAR_RECT_TRUE                                           0x00000001
#define LWCC97_SET_CLEAR_SURFACE_CONTROL_USE_SCISSOR0                                                         8:8
#define LWCC97_SET_CLEAR_SURFACE_CONTROL_USE_SCISSOR0_FALSE                                            0x00000000
#define LWCC97_SET_CLEAR_SURFACE_CONTROL_USE_SCISSOR0_TRUE                                             0x00000001
#define LWCC97_SET_CLEAR_SURFACE_CONTROL_USE_VIEWPORT_CLIP0                                                 12:12
#define LWCC97_SET_CLEAR_SURFACE_CONTROL_USE_VIEWPORT_CLIP0_FALSE                                      0x00000000
#define LWCC97_SET_CLEAR_SURFACE_CONTROL_USE_VIEWPORT_CLIP0_TRUE                                       0x00000001

#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS                                   0x10fc
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS_POLICY                               5:4
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS_POLICY_EVICT_FIRST             0x00000000
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS_POLICY_EVICT_NORMAL             0x00000001
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS_POLICY_EVICT_LAST             0x00000002
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS_ZROP_POLICY                          1:0
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS_ZROP_POLICY_EVICT_FIRST             0x00000000
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS_ZROP_POLICY_EVICT_NORMAL             0x00000001
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS_ZROP_POLICY_EVICT_LAST             0x00000002

#define LWCC97_SET_RESERVED_SW_METHOD14                                                                    0x1100
#define LWCC97_SET_RESERVED_SW_METHOD14_V                                                                    31:0

#define LWCC97_SET_RESERVED_SW_METHOD15                                                                    0x1104
#define LWCC97_SET_RESERVED_SW_METHOD15_V                                                                    31:0

#define LWCC97_SET_TILED_CACHE_CONTROL_EXTENDED                                                            0x1108
#define LWCC97_SET_TILED_CACHE_CONTROL_EXTENDED_LEVEL1_BBOX_PRIM_COUNT                                        5:0
#define LWCC97_SET_TILED_CACHE_CONTROL_EXTENDED_MAX_CYCLES_WITHOUT_BINNER_DRAIN                              11:6
#define LWCC97_SET_TILED_CACHE_CONTROL_EXTENDED_AREA_THRESHOLD                                              16:12

#define LWCC97_NO_OPERATION_DATA_HI                                                                        0x110c
#define LWCC97_NO_OPERATION_DATA_HI_V                                                                        31:0

#define LWCC97_SET_DEPTH_BIAS_CONTROL                                                                      0x1110
#define LWCC97_SET_DEPTH_BIAS_CONTROL_DEPTH_FORMAT_DEPENDENT                                                  0:0
#define LWCC97_SET_DEPTH_BIAS_CONTROL_DEPTH_FORMAT_DEPENDENT_FALSE                                     0x00000000
#define LWCC97_SET_DEPTH_BIAS_CONTROL_DEPTH_FORMAT_DEPENDENT_TRUE                                      0x00000001

#define LWCC97_PM_TRIGGER_END                                                                              0x1114
#define LWCC97_PM_TRIGGER_END_V                                                                              31:0

#define LWCC97_SET_VERTEX_ID_BASE                                                                          0x1118
#define LWCC97_SET_VERTEX_ID_BASE_V                                                                          31:0

#define LWCC97_SET_STENCIL_COMPRESSION                                                                     0x111c
#define LWCC97_SET_STENCIL_COMPRESSION_ENABLE                                                                 0:0
#define LWCC97_SET_STENCIL_COMPRESSION_ENABLE_FALSE                                                    0x00000000
#define LWCC97_SET_STENCIL_COMPRESSION_ENABLE_TRUE                                                     0x00000001

#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A(i)                                              (0x1120+(i)*4)
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP0                                           0:0
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP0_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP0_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP1                                           1:1
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP1_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP1_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP2                                           2:2
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP2_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP2_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP3                                           3:3
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP3_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE0_COMP3_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP0                                           4:4
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP0_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP0_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP1                                           5:5
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP1_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP1_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP2                                           6:6
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP2_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP2_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP3                                           7:7
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP3_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE1_COMP3_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP0                                           8:8
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP0_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP0_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP1                                           9:9
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP1_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP1_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP2                                         10:10
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP2_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP2_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP3                                         11:11
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP3_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE2_COMP3_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP0                                         12:12
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP0_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP0_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP1                                         13:13
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP1_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP1_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP2                                         14:14
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP2_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP2_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP3                                         15:15
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP3_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE3_COMP3_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP0                                         16:16
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP0_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP0_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP1                                         17:17
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP1_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP1_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP2                                         18:18
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP2_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP2_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP3                                         19:19
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP3_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE4_COMP3_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP0                                         20:20
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP0_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP0_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP1                                         21:21
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP1_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP1_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP2                                         22:22
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP2_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP2_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP3                                         23:23
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP3_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE5_COMP3_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP0                                         24:24
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP0_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP0_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP1                                         25:25
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP1_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP1_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP2                                         26:26
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP2_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP2_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP3                                         27:27
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP3_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE6_COMP3_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP0                                         28:28
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP0_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP0_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP1                                         29:29
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP1_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP1_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP2                                         30:30
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP2_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP2_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP3                                         31:31
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP3_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_A_ATTRIBUTE7_COMP3_TRUE                               0x00000001

#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B(i)                                              (0x1128+(i)*4)
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP0                                           0:0
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP0_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP0_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP1                                           1:1
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP1_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP1_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP2                                           2:2
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP2_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP2_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP3                                           3:3
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP3_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE0_COMP3_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP0                                           4:4
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP0_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP0_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP1                                           5:5
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP1_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP1_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP2                                           6:6
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP2_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP2_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP3                                           7:7
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP3_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE1_COMP3_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP0                                           8:8
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP0_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP0_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP1                                           9:9
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP1_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP1_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP2                                         10:10
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP2_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP2_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP3                                         11:11
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP3_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE2_COMP3_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP0                                         12:12
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP0_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP0_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP1                                         13:13
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP1_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP1_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP2                                         14:14
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP2_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP2_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP3                                         15:15
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP3_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE3_COMP3_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP0                                         16:16
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP0_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP0_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP1                                         17:17
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP1_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP1_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP2                                         18:18
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP2_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP2_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP3                                         19:19
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP3_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE4_COMP3_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP0                                         20:20
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP0_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP0_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP1                                         21:21
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP1_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP1_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP2                                         22:22
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP2_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP2_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP3                                         23:23
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP3_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE5_COMP3_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP0                                         24:24
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP0_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP0_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP1                                         25:25
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP1_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP1_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP2                                         26:26
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP2_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP2_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP3                                         27:27
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP3_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE6_COMP3_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP0                                         28:28
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP0_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP0_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP1                                         29:29
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP1_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP1_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP2                                         30:30
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP2_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP2_TRUE                               0x00000001
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP3                                         31:31
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP3_FALSE                              0x00000000
#define LWCC97_SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK_B_ATTRIBUTE7_COMP3_TRUE                               0x00000001

#define LWCC97_SET_TIR_CONTROL                                                                             0x1130
#define LWCC97_SET_TIR_CONTROL_Z_PASS_PIXEL_COUNT_USE_RASTER_SAMPLES                                          0:0
#define LWCC97_SET_TIR_CONTROL_Z_PASS_PIXEL_COUNT_USE_RASTER_SAMPLES_DISABLE                           0x00000000
#define LWCC97_SET_TIR_CONTROL_Z_PASS_PIXEL_COUNT_USE_RASTER_SAMPLES_ENABLE                            0x00000001
#define LWCC97_SET_TIR_CONTROL_ALPHA_TO_COVERAGE_USE_RASTER_SAMPLES                                           4:4
#define LWCC97_SET_TIR_CONTROL_ALPHA_TO_COVERAGE_USE_RASTER_SAMPLES_DISABLE                            0x00000000
#define LWCC97_SET_TIR_CONTROL_ALPHA_TO_COVERAGE_USE_RASTER_SAMPLES_ENABLE                             0x00000001
#define LWCC97_SET_TIR_CONTROL_REDUCE_COVERAGE                                                                1:1
#define LWCC97_SET_TIR_CONTROL_REDUCE_COVERAGE_DISABLE                                                 0x00000000
#define LWCC97_SET_TIR_CONTROL_REDUCE_COVERAGE_ENABLE                                                  0x00000001
#define LWCC97_SET_TIR_CONTROL_REDUCTION_MODE                                                                 2:2
#define LWCC97_SET_TIR_CONTROL_REDUCTION_MODE_AFFINITY_MAP                                             0x00000000
#define LWCC97_SET_TIR_CONTROL_REDUCTION_MODE_TRUNCATION                                               0x00000001

#define LWCC97_SET_MUTABLE_METHOD_CONTROL                                                                  0x1134
#define LWCC97_SET_MUTABLE_METHOD_CONTROL_TREAT_MUTABLE_AS_HEAVYWEIGHT                                        0:0
#define LWCC97_SET_MUTABLE_METHOD_CONTROL_TREAT_MUTABLE_AS_HEAVYWEIGHT_FALSE                           0x00000000
#define LWCC97_SET_MUTABLE_METHOD_CONTROL_TREAT_MUTABLE_AS_HEAVYWEIGHT_TRUE                            0x00000001

#define LWCC97_SET_POST_PS_INITIAL_COVERAGE                                                                0x1138
#define LWCC97_SET_POST_PS_INITIAL_COVERAGE_USE_PRE_PS_COVERAGE                                               0:0
#define LWCC97_SET_POST_PS_INITIAL_COVERAGE_USE_PRE_PS_COVERAGE_FALSE                                  0x00000000
#define LWCC97_SET_POST_PS_INITIAL_COVERAGE_USE_PRE_PS_COVERAGE_TRUE                                   0x00000001

#define LWCC97_SET_FILL_VIA_TRIANGLE                                                                       0x113c
#define LWCC97_SET_FILL_VIA_TRIANGLE_MODE                                                                     1:0
#define LWCC97_SET_FILL_VIA_TRIANGLE_MODE_DISABLED                                                     0x00000000
#define LWCC97_SET_FILL_VIA_TRIANGLE_MODE_FILL_ALL                                                     0x00000001
#define LWCC97_SET_FILL_VIA_TRIANGLE_MODE_FILL_BBOX                                                    0x00000002

#define LWCC97_SET_BLEND_PER_FORMAT_ENABLE                                                                 0x1140
#define LWCC97_SET_BLEND_PER_FORMAT_ENABLE_SNORM8_UNORM16_SNORM16                                             4:4
#define LWCC97_SET_BLEND_PER_FORMAT_ENABLE_SNORM8_UNORM16_SNORM16_FALSE                                0x00000000
#define LWCC97_SET_BLEND_PER_FORMAT_ENABLE_SNORM8_UNORM16_SNORM16_TRUE                                 0x00000001

#define LWCC97_FLUSH_PENDING_WRITES                                                                        0x1144
#define LWCC97_FLUSH_PENDING_WRITES_SM_DOES_GLOBAL_STORE                                                      0:0

#define LWCC97_SET_CONSERVATIVE_RASTER                                                                     0x1148
#define LWCC97_SET_CONSERVATIVE_RASTER_ENABLE                                                                 0:0
#define LWCC97_SET_CONSERVATIVE_RASTER_ENABLE_FALSE                                                    0x00000000
#define LWCC97_SET_CONSERVATIVE_RASTER_ENABLE_TRUE                                                     0x00000001

#define LWCC97_SET_MESH_CONTROL                                                                            0x114c
#define LWCC97_SET_MESH_CONTROL_ENABLE                                                                        0:0
#define LWCC97_SET_MESH_CONTROL_ENABLE_FALSE                                                           0x00000000
#define LWCC97_SET_MESH_CONTROL_ENABLE_TRUE                                                            0x00000001

#define LWCC97_SET_MESH_SHADER_A                                                                           0x1150
#define LWCC97_SET_MESH_SHADER_A_OUTPUT_TOPOLOGY                                                              3:0
#define LWCC97_SET_MESH_SHADER_A_OUTPUT_TOPOLOGY_POINTS                                                0x00000000
#define LWCC97_SET_MESH_SHADER_A_OUTPUT_TOPOLOGY_LINES                                                 0x00000001
#define LWCC97_SET_MESH_SHADER_A_OUTPUT_TOPOLOGY_LINE_LOOP                                             0x00000002
#define LWCC97_SET_MESH_SHADER_A_OUTPUT_TOPOLOGY_LINE_STRIP                                            0x00000003
#define LWCC97_SET_MESH_SHADER_A_OUTPUT_TOPOLOGY_TRIANGLES                                             0x00000004
#define LWCC97_SET_MESH_SHADER_A_OUTPUT_TOPOLOGY_TRIANGLE_STRIP                                        0x00000005
#define LWCC97_SET_MESH_SHADER_A_OUTPUT_TOPOLOGY_TRIANGLE_FAN                                          0x00000006
#define LWCC97_SET_MESH_SHADER_A_OUTPUT_TOPOLOGY_QUADS                                                 0x00000007
#define LWCC97_SET_MESH_SHADER_A_OUTPUT_TOPOLOGY_QUAD_STRIP                                            0x00000008
#define LWCC97_SET_MESH_SHADER_A_OUTPUT_TOPOLOGY_POLYGON                                               0x00000009
#define LWCC97_SET_MESH_SHADER_A_OUTPUT_TOPOLOGY_LINELIST_ADJCY                                        0x0000000A
#define LWCC97_SET_MESH_SHADER_A_OUTPUT_TOPOLOGY_LINESTRIP_ADJCY                                       0x0000000B
#define LWCC97_SET_MESH_SHADER_A_OUTPUT_TOPOLOGY_TRIANGLELIST_ADJCY                                    0x0000000C
#define LWCC97_SET_MESH_SHADER_A_OUTPUT_TOPOLOGY_TRIANGLESTRIP_ADJCY                                   0x0000000D
#define LWCC97_SET_MESH_SHADER_A_OUTPUT_TOPOLOGY_PATCH                                                 0x0000000E
#define LWCC97_SET_MESH_SHADER_A_MAX_VERTEX                                                                  12:4
#define LWCC97_SET_MESH_SHADER_A_MAX_PRIMITIVE                                                              22:13

#define LWCC97_SET_MESH_SHADER_B                                                                           0x1154
#define LWCC97_SET_MESH_SHADER_B_SHARED_MEM_LINES                                                            16:8
#define LWCC97_SET_MESH_SHADER_B_THREAD_COUNT                                                               25:20

#define LWCC97_SET_MESH_INIT_SHADER                                                                        0x1158
#define LWCC97_SET_MESH_INIT_SHADER_THREAD_COUNT                                                              5:0
#define LWCC97_SET_MESH_INIT_SHADER_LOCAL_BUFFER_LINES                                                      20:12
#define LWCC97_SET_MESH_INIT_SHADER_OUTPUT_TO_M_S_LINES                                                     30:22

#define LWCC97_SET_VERTEX_ATTRIBUTE_A(i)                                                           (0x1160+(i)*4)
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_STREAM                                                                  4:0
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_SOURCE                                                                  6:6
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_SOURCE_ACTIVE                                                    0x00000000
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_SOURCE_INACTIVE                                                  0x00000001
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_OFFSET                                                                 20:7
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS                                                  26:21
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R32_G32_B32_A32                             0x00000001
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R32_G32_B32                                 0x00000002
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R16_G16_B16_A16                             0x00000003
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R32_G32                                     0x00000004
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R16_G16_B16                                 0x00000005
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_A8B8G8R8                                    0x0000002F
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R8_G8_B8_A8                                 0x0000000A
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_X8B8G8R8                                    0x00000033
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_A2B10G10R10                                 0x00000030
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_B10G11R11                                   0x00000031
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R16_G16                                     0x0000000F
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R32                                         0x00000012
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R8_G8_B8                                    0x00000013
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_G8R8                                        0x00000032
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R8_G8                                       0x00000018
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R16                                         0x0000001B
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R8                                          0x0000001D
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_A8                                          0x00000034
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE                                                        29:27
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_UNUSED_ENUM_DO_NOT_USE_BECAUSE_IT_WILL_GO_AWAY             0x00000000
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_SNORM                                         0x00000001
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_UNORM                                         0x00000002
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_SINT                                          0x00000003
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_UINT                                          0x00000004
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_USCALED                                       0x00000005
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_SSCALED                                       0x00000006
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_FLOAT                                         0x00000007
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_SWAP_R_AND_B                                                          31:31
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_SWAP_R_AND_B_FALSE                                               0x00000000
#define LWCC97_SET_VERTEX_ATTRIBUTE_A_SWAP_R_AND_B_TRUE                                                0x00000001

#define LWCC97_SET_VERTEX_ATTRIBUTE_B(i)                                                           (0x11a0+(i)*4)
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_STREAM                                                                  4:0
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_SOURCE                                                                  6:6
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_SOURCE_ACTIVE                                                    0x00000000
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_SOURCE_INACTIVE                                                  0x00000001
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_OFFSET                                                                 20:7
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS                                                  26:21
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R32_G32_B32_A32                             0x00000001
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R32_G32_B32                                 0x00000002
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R16_G16_B16_A16                             0x00000003
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R32_G32                                     0x00000004
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R16_G16_B16                                 0x00000005
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_A8B8G8R8                                    0x0000002F
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R8_G8_B8_A8                                 0x0000000A
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_X8B8G8R8                                    0x00000033
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_A2B10G10R10                                 0x00000030
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_B10G11R11                                   0x00000031
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R16_G16                                     0x0000000F
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R32                                         0x00000012
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R8_G8_B8                                    0x00000013
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_G8R8                                        0x00000032
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R8_G8                                       0x00000018
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R16                                         0x0000001B
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R8                                          0x0000001D
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_A8                                          0x00000034
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE                                                        29:27
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_UNUSED_ENUM_DO_NOT_USE_BECAUSE_IT_WILL_GO_AWAY             0x00000000
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_NUM_SNORM                                         0x00000001
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_NUM_UNORM                                         0x00000002
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_NUM_SINT                                          0x00000003
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_NUM_UINT                                          0x00000004
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_NUM_USCALED                                       0x00000005
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_NUM_SSCALED                                       0x00000006
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_NUM_FLOAT                                         0x00000007
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_SWAP_R_AND_B                                                          31:31
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_SWAP_R_AND_B_FALSE                                               0x00000000
#define LWCC97_SET_VERTEX_ATTRIBUTE_B_SWAP_R_AND_B_TRUE                                                0x00000001

#define LWCC97_SET_ANTI_ALIAS_SAMPLE_POSITIONS(i)                                                  (0x11e0+(i)*4)
#define LWCC97_SET_ANTI_ALIAS_SAMPLE_POSITIONS_X0                                                             3:0
#define LWCC97_SET_ANTI_ALIAS_SAMPLE_POSITIONS_Y0                                                             7:4
#define LWCC97_SET_ANTI_ALIAS_SAMPLE_POSITIONS_X1                                                            11:8
#define LWCC97_SET_ANTI_ALIAS_SAMPLE_POSITIONS_Y1                                                           15:12
#define LWCC97_SET_ANTI_ALIAS_SAMPLE_POSITIONS_X2                                                           19:16
#define LWCC97_SET_ANTI_ALIAS_SAMPLE_POSITIONS_Y2                                                           23:20
#define LWCC97_SET_ANTI_ALIAS_SAMPLE_POSITIONS_X3                                                           27:24
#define LWCC97_SET_ANTI_ALIAS_SAMPLE_POSITIONS_Y3                                                           31:28

#define LWCC97_SET_OFFSET_RENDER_TARGET_INDEX                                                              0x11f0
#define LWCC97_SET_OFFSET_RENDER_TARGET_INDEX_BY_VIEWPORT_INDEX                                               0:0
#define LWCC97_SET_OFFSET_RENDER_TARGET_INDEX_BY_VIEWPORT_INDEX_FALSE                                  0x00000000
#define LWCC97_SET_OFFSET_RENDER_TARGET_INDEX_BY_VIEWPORT_INDEX_TRUE                                   0x00000001

#define LWCC97_FORCE_HEAVYWEIGHT_METHOD_SYNC                                                               0x11f4
#define LWCC97_FORCE_HEAVYWEIGHT_METHOD_SYNC_V                                                               31:0

#define LWCC97_SET_COVERAGE_TO_COLOR                                                                       0x11f8
#define LWCC97_SET_COVERAGE_TO_COLOR_ENABLE                                                                   0:0
#define LWCC97_SET_COVERAGE_TO_COLOR_ENABLE_FALSE                                                      0x00000000
#define LWCC97_SET_COVERAGE_TO_COLOR_ENABLE_TRUE                                                       0x00000001
#define LWCC97_SET_COVERAGE_TO_COLOR_CT_SELECT                                                                6:4

#define LWCC97_DECOMPRESS_ZETA_SURFACE                                                                     0x11fc
#define LWCC97_DECOMPRESS_ZETA_SURFACE_Z_ENABLE                                                               0:0
#define LWCC97_DECOMPRESS_ZETA_SURFACE_Z_ENABLE_FALSE                                                  0x00000000
#define LWCC97_DECOMPRESS_ZETA_SURFACE_Z_ENABLE_TRUE                                                   0x00000001
#define LWCC97_DECOMPRESS_ZETA_SURFACE_STENCIL_ENABLE                                                         4:4
#define LWCC97_DECOMPRESS_ZETA_SURFACE_STENCIL_ENABLE_FALSE                                            0x00000000
#define LWCC97_DECOMPRESS_ZETA_SURFACE_STENCIL_ENABLE_TRUE                                             0x00000001

#define LWCC97_TILED_CACHE_ILWALIDATE_TEXTURE_SAMPLER_CACHE                                                0x1200
#define LWCC97_TILED_CACHE_ILWALIDATE_TEXTURE_SAMPLER_CACHE_V                                                 0:0

#define LWCC97_SET_SCREEN_STATE_MASK                                                                       0x1204
#define LWCC97_SET_SCREEN_STATE_MASK_MASK                                                                     3:0

#define LWCC97_SET_ZT_SPARSE                                                                               0x1208
#define LWCC97_SET_ZT_SPARSE_ENABLE                                                                           0:0
#define LWCC97_SET_ZT_SPARSE_ENABLE_FALSE                                                              0x00000000
#define LWCC97_SET_ZT_SPARSE_ENABLE_TRUE                                                               0x00000001
#define LWCC97_SET_ZT_SPARSE_UNMAPPED_COMPARE                                                                 1:1
#define LWCC97_SET_ZT_SPARSE_UNMAPPED_COMPARE_ZT_SPARSE_UNMAPPED_0                                     0x00000000
#define LWCC97_SET_ZT_SPARSE_UNMAPPED_COMPARE_ZT_SPARSE_FAIL_ALWAYS                                    0x00000001

#define LWCC97_SET_ST_SIZE_A                                                                               0x120c
#define LWCC97_SET_ST_SIZE_A_WIDTH                                                                           27:0

#define LWCC97_SET_ST_SIZE_B                                                                               0x1210
#define LWCC97_SET_ST_SIZE_B_HEIGHT                                                                          17:0

#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST                                                  0x1214
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_START_INDEX                                        15:0
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_COUNT                                             27:16
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY                                          31:28
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_POINTS                              0x00000000
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINES                               0x00000001
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINE_LOOP                           0x00000002
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINE_STRIP                          0x00000003
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLES                           0x00000004
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLE_STRIP                      0x00000005
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLE_FAN                        0x00000006
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_QUADS                               0x00000007
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_QUAD_STRIP                          0x00000008
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_POLYGON                             0x00000009
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINELIST_ADJCY                      0x0000000A
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINESTRIP_ADJCY                     0x0000000B
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLELIST_ADJCY                  0x0000000C
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLESTRIP_ADJCY                 0x0000000D
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_PATCH                               0x0000000E

#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT                                             0x1218
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_START_INDEX                                   15:0
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_COUNT                                        27:16
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY                                     31:28
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_POINTS                         0x00000000
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINES                          0x00000001
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINE_LOOP                      0x00000002
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINE_STRIP                     0x00000003
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLES                      0x00000004
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLE_STRIP                 0x00000005
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLE_FAN                   0x00000006
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_QUADS                          0x00000007
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_QUAD_STRIP                     0x00000008
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_POLYGON                        0x00000009
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINELIST_ADJCY                 0x0000000A
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINESTRIP_ADJCY                0x0000000B
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLELIST_ADJCY             0x0000000C
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLESTRIP_ADJCY             0x0000000D
#define LWCC97_DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_PATCH                          0x0000000E

#define LWCC97_SET_CT_SELECT                                                                               0x121c
#define LWCC97_SET_CT_SELECT_TARGET_COUNT                                                                     3:0
#define LWCC97_SET_CT_SELECT_TARGET0                                                                          6:4
#define LWCC97_SET_CT_SELECT_TARGET1                                                                          9:7
#define LWCC97_SET_CT_SELECT_TARGET2                                                                        12:10
#define LWCC97_SET_CT_SELECT_TARGET3                                                                        15:13
#define LWCC97_SET_CT_SELECT_TARGET4                                                                        18:16
#define LWCC97_SET_CT_SELECT_TARGET5                                                                        21:19
#define LWCC97_SET_CT_SELECT_TARGET6                                                                        24:22
#define LWCC97_SET_CT_SELECT_TARGET7                                                                        27:25

#define LWCC97_SET_COMPRESSION_THRESHOLD                                                                   0x1220
#define LWCC97_SET_COMPRESSION_THRESHOLD_SAMPLES                                                              3:0
#define LWCC97_SET_COMPRESSION_THRESHOLD_SAMPLES__0                                                    0x00000000
#define LWCC97_SET_COMPRESSION_THRESHOLD_SAMPLES__1                                                    0x00000001
#define LWCC97_SET_COMPRESSION_THRESHOLD_SAMPLES__2                                                    0x00000002
#define LWCC97_SET_COMPRESSION_THRESHOLD_SAMPLES__4                                                    0x00000003
#define LWCC97_SET_COMPRESSION_THRESHOLD_SAMPLES__8                                                    0x00000004
#define LWCC97_SET_COMPRESSION_THRESHOLD_SAMPLES__16                                                   0x00000005
#define LWCC97_SET_COMPRESSION_THRESHOLD_SAMPLES__32                                                   0x00000006
#define LWCC97_SET_COMPRESSION_THRESHOLD_SAMPLES__64                                                   0x00000007
#define LWCC97_SET_COMPRESSION_THRESHOLD_SAMPLES__128                                                  0x00000008
#define LWCC97_SET_COMPRESSION_THRESHOLD_SAMPLES__256                                                  0x00000009
#define LWCC97_SET_COMPRESSION_THRESHOLD_SAMPLES__512                                                  0x0000000A
#define LWCC97_SET_COMPRESSION_THRESHOLD_SAMPLES__1024                                                 0x0000000B
#define LWCC97_SET_COMPRESSION_THRESHOLD_SAMPLES__2048                                                 0x0000000C

#define LWCC97_SET_PIXEL_SHADER_INTERLOCK_CONTROL                                                          0x1224
#define LWCC97_SET_PIXEL_SHADER_INTERLOCK_CONTROL_TILE_COALESCER_MODE                                         1:0
#define LWCC97_SET_PIXEL_SHADER_INTERLOCK_CONTROL_TILE_COALESCER_MODE_NO_CONFLICT_DETECT               0x00000000
#define LWCC97_SET_PIXEL_SHADER_INTERLOCK_CONTROL_TILE_COALESCER_MODE_CONFLICT_DETECT_SAMPLE             0x00000001
#define LWCC97_SET_PIXEL_SHADER_INTERLOCK_CONTROL_TILE_COALESCER_MODE_CONFLICT_DETECT_PIXEL             0x00000002
#define LWCC97_SET_PIXEL_SHADER_INTERLOCK_CONTROL_TILE_COALESCER_MODE_CONFLICT_DETECT_VPRS             0x00000003
#define LWCC97_SET_PIXEL_SHADER_INTERLOCK_CONTROL_TILE_COALESCER_TILE_SIZE                                    2:2
#define LWCC97_SET_PIXEL_SHADER_INTERLOCK_CONTROL_TILE_COALESCER_TILE_SIZE_TC_TILE_SIZE_16X16             0x00000000
#define LWCC97_SET_PIXEL_SHADER_INTERLOCK_CONTROL_TILE_COALESCER_TILE_SIZE_TC_TILE_SIZE_8X8             0x00000001
#define LWCC97_SET_PIXEL_SHADER_INTERLOCK_CONTROL_TILE_COALESCER_FRAGMENT_ORDER                               3:3
#define LWCC97_SET_PIXEL_SHADER_INTERLOCK_CONTROL_TILE_COALESCER_FRAGMENT_ORDER_TC_FRAGMENT_ORDERED             0x00000000
#define LWCC97_SET_PIXEL_SHADER_INTERLOCK_CONTROL_TILE_COALESCER_FRAGMENT_ORDER_TC_FRAGMENT_UNORDERED             0x00000001

#define LWCC97_SET_ZT_SIZE_A                                                                               0x1228
#define LWCC97_SET_ZT_SIZE_A_WIDTH                                                                           27:0

#define LWCC97_SET_ZT_SIZE_B                                                                               0x122c
#define LWCC97_SET_ZT_SIZE_B_HEIGHT                                                                          17:0

#define LWCC97_SET_ZT_SIZE_C                                                                               0x1230
#define LWCC97_SET_ZT_SIZE_C_THIRD_DIMENSION                                                                 15:0
#define LWCC97_SET_ZT_SIZE_C_CONTROL                                                                        16:16
#define LWCC97_SET_ZT_SIZE_C_CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE                                0x00000000
#define LWCC97_SET_ZT_SIZE_C_CONTROL_ARRAY_SIZE_IS_ONE                                                 0x00000001

#define LWCC97_SET_SAMPLER_BINDING                                                                         0x1234
#define LWCC97_SET_SAMPLER_BINDING_V                                                                          0:0
#define LWCC97_SET_SAMPLER_BINDING_V_INDEPENDENTLY                                                     0x00000000
#define LWCC97_SET_SAMPLER_BINDING_V_VIA_HEADER_BINDING                                                0x00000001

#define LWCC97_SET_MULTI_PASS_PRIMITIVE                                                                    0x1238
#define LWCC97_SET_MULTI_PASS_PRIMITIVE_MASK                                                                  3:0

#define LWCC97_DRAW_AUTO                                                                                   0x123c
#define LWCC97_DRAW_AUTO_BYTE_COUNT                                                                          31:0

#define LWCC97_SET_POST_VTG_SHADER_ATTRIBUTE_SKIP_MASK(i)                                          (0x1240+(i)*4)
#define LWCC97_SET_POST_VTG_SHADER_ATTRIBUTE_SKIP_MASK_V                                                     31:0

#define LWCC97_SET_PIXEL_SHADER_TICKET_DISPENSER_VALUE                                                     0x1260
#define LWCC97_SET_PIXEL_SHADER_TICKET_DISPENSER_VALUE_TICKET_DISPENSER_INDEX                                 7:0
#define LWCC97_SET_PIXEL_SHADER_TICKET_DISPENSER_VALUE_TICKET_DISPENSER_VALUE                                23:8

#define LWCC97_SET_BACK_END_COPY_A                                                                         0x1264
#define LWCC97_SET_BACK_END_COPY_A_DWORDS                                                                     7:0
#define LWCC97_SET_BACK_END_COPY_A_SATURATE32_ENABLE                                                          8:8
#define LWCC97_SET_BACK_END_COPY_A_SATURATE32_ENABLE_FALSE                                             0x00000000
#define LWCC97_SET_BACK_END_COPY_A_SATURATE32_ENABLE_TRUE                                              0x00000001
#define LWCC97_SET_BACK_END_COPY_A_TIMESTAMP_ENABLE                                                         12:12
#define LWCC97_SET_BACK_END_COPY_A_TIMESTAMP_ENABLE_FALSE                                              0x00000000
#define LWCC97_SET_BACK_END_COPY_A_TIMESTAMP_ENABLE_TRUE                                               0x00000001

#define LWCC97_SET_BACK_END_COPY_B                                                                         0x1268
#define LWCC97_SET_BACK_END_COPY_B_SRC_ADDRESS_UPPER                                                          7:0

#define LWCC97_SET_BACK_END_COPY_C                                                                         0x126c
#define LWCC97_SET_BACK_END_COPY_C_SRC_ADDRESS_LOWER                                                         31:0

#define LWCC97_SET_BACK_END_COPY_D                                                                         0x1270
#define LWCC97_SET_BACK_END_COPY_D_DEST_ADDRESS_UPPER                                                         7:0

#define LWCC97_SET_BACK_END_COPY_E                                                                         0x1274
#define LWCC97_SET_BACK_END_COPY_E_DEST_ADDRESS_LOWER                                                        31:0

#define LWCC97_SET_MULTI_PASS_SHADER_META_DATA                                                             0x1278
#define LWCC97_SET_MULTI_PASS_SHADER_META_DATA_V                                                             31:0

#define LWCC97_SET_CIRLWLAR_BUFFER_SIZE                                                                    0x1280
#define LWCC97_SET_CIRLWLAR_BUFFER_SIZE_CACHE_LINES_PER_SM                                                   19:0

#define LWCC97_SET_VTG_REGISTER_WATERMARKS                                                                 0x1284
#define LWCC97_SET_VTG_REGISTER_WATERMARKS_LOW                                                               15:0
#define LWCC97_SET_VTG_REGISTER_WATERMARKS_HIGH                                                             31:16

#define LWCC97_ILWALIDATE_TEXTURE_DATA_CACHE_NO_WFI                                                        0x1288
#define LWCC97_ILWALIDATE_TEXTURE_DATA_CACHE_NO_WFI_LINES                                                     0:0
#define LWCC97_ILWALIDATE_TEXTURE_DATA_CACHE_NO_WFI_LINES_ALL                                          0x00000000
#define LWCC97_ILWALIDATE_TEXTURE_DATA_CACHE_NO_WFI_LINES_ONE                                          0x00000001
#define LWCC97_ILWALIDATE_TEXTURE_DATA_CACHE_NO_WFI_TAG                                                      25:4

#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS                                      0x1290
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS_POLICY                                  5:4
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS_POLICY_EVICT_FIRST               0x00000000
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS_POLICY_EVICT_NORMAL              0x00000001
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS_POLICY_EVICT_LAST                0x00000002
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS_ZROP_POLICY                             1:0
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS_ZROP_POLICY_EVICT_FIRST             0x00000000
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS_ZROP_POLICY_EVICT_NORMAL             0x00000001
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS_ZROP_POLICY_EVICT_LAST             0x00000002

#define LWCC97_SET_DA_PRIMITIVE_RESTART_INDEX_TOPOLOGY_CHANGE                                              0x12a4
#define LWCC97_SET_DA_PRIMITIVE_RESTART_INDEX_TOPOLOGY_CHANGE_V                                              31:0

#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_ADDRESS_A(j)                                        (0x12a8+(j)*32)
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_ADDRESS_A_OFFSET_UPPER                                          7:0

#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_ADDRESS_B(j)                                        (0x12ac+(j)*32)
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_ADDRESS_B_OFFSET_LOWER                                         31:0

#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_SIZE_A(j)                                           (0x12b0+(j)*32)
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_SIZE_A_WIDTH                                                   11:0
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_SIZE_A_HEIGHT                                                 27:16

#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_SIZE_B(j)                                           (0x12b4+(j)*32)
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_SIZE_B_ARRAY_SIZE                                              11:0

#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_LAYER(j)                                            (0x12b8+(j)*32)
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_LAYER_OFFSET                                                   11:0

#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_ARRAY_PITCH(j)                                      (0x12bc+(j)*32)
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_ARRAY_PITCH_V                                                  28:0

#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_BLOCK_SIZE(j)                                       (0x12c0+(j)*32)
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_BLOCK_SIZE_WIDTH                                                3:0
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_BLOCK_SIZE_WIDTH_ONE_GOB                                 0x00000000
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_BLOCK_SIZE_HEIGHT                                               7:4
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_BLOCK_SIZE_HEIGHT_ONE_GOB                                0x00000000
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_BLOCK_SIZE_HEIGHT_TWO_GOBS                               0x00000001
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_BLOCK_SIZE_HEIGHT_FOUR_GOBS                              0x00000002
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                             0x00000003
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                           0x00000004
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                         0x00000005
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_BLOCK_SIZE_DEPTH                                               11:8
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_BLOCK_SIZE_DEPTH_ONE_GOB                                 0x00000000

#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_ALLOCATED_SIZE(j)                                   (0x12c4+(j)*32)
#define LWCC97_SET_SHADING_RATE_INDEX_SURFACE_ALLOCATED_SIZE_WIDTH                                           11:0

#define LWCC97_CLEAR_ZLWLL_REGION                                                                          0x12c8
#define LWCC97_CLEAR_ZLWLL_REGION_Z_ENABLE                                                                    0:0
#define LWCC97_CLEAR_ZLWLL_REGION_Z_ENABLE_FALSE                                                       0x00000000
#define LWCC97_CLEAR_ZLWLL_REGION_Z_ENABLE_TRUE                                                        0x00000001
#define LWCC97_CLEAR_ZLWLL_REGION_STENCIL_ENABLE                                                              4:4
#define LWCC97_CLEAR_ZLWLL_REGION_STENCIL_ENABLE_FALSE                                                 0x00000000
#define LWCC97_CLEAR_ZLWLL_REGION_STENCIL_ENABLE_TRUE                                                  0x00000001
#define LWCC97_CLEAR_ZLWLL_REGION_USE_CLEAR_RECT                                                              1:1
#define LWCC97_CLEAR_ZLWLL_REGION_USE_CLEAR_RECT_FALSE                                                 0x00000000
#define LWCC97_CLEAR_ZLWLL_REGION_USE_CLEAR_RECT_TRUE                                                  0x00000001
#define LWCC97_CLEAR_ZLWLL_REGION_USE_RT_ARRAY_INDEX                                                          2:2
#define LWCC97_CLEAR_ZLWLL_REGION_USE_RT_ARRAY_INDEX_FALSE                                             0x00000000
#define LWCC97_CLEAR_ZLWLL_REGION_USE_RT_ARRAY_INDEX_TRUE                                              0x00000001
#define LWCC97_CLEAR_ZLWLL_REGION_RT_ARRAY_INDEX                                                             20:5
#define LWCC97_CLEAR_ZLWLL_REGION_MAKE_CONSERVATIVE                                                           3:3
#define LWCC97_CLEAR_ZLWLL_REGION_MAKE_CONSERVATIVE_FALSE                                              0x00000000
#define LWCC97_CLEAR_ZLWLL_REGION_MAKE_CONSERVATIVE_TRUE                                               0x00000001

#define LWCC97_SET_DEPTH_TEST                                                                              0x12cc
#define LWCC97_SET_DEPTH_TEST_ENABLE                                                                          0:0
#define LWCC97_SET_DEPTH_TEST_ENABLE_FALSE                                                             0x00000000
#define LWCC97_SET_DEPTH_TEST_ENABLE_TRUE                                                              0x00000001

#define LWCC97_SET_FILL_MODE                                                                               0x12d0
#define LWCC97_SET_FILL_MODE_V                                                                               31:0
#define LWCC97_SET_FILL_MODE_V_POINT                                                                   0x00000001
#define LWCC97_SET_FILL_MODE_V_WIREFRAME                                                               0x00000002
#define LWCC97_SET_FILL_MODE_V_SOLID                                                                   0x00000003

#define LWCC97_SET_SHADE_MODE                                                                              0x12d4
#define LWCC97_SET_SHADE_MODE_V                                                                              31:0
#define LWCC97_SET_SHADE_MODE_V_FLAT                                                                   0x00000001
#define LWCC97_SET_SHADE_MODE_V_GOURAUD                                                                0x00000002
#define LWCC97_SET_SHADE_MODE_V_OGL_FLAT                                                               0x00001D00
#define LWCC97_SET_SHADE_MODE_V_OGL_SMOOTH                                                             0x00001D01

#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS                                  0x12d8
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS_POLICY                              5:4
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS_POLICY_EVICT_FIRST             0x00000000
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS_POLICY_EVICT_NORMAL             0x00000001
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS_POLICY_EVICT_LAST             0x00000002
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS_ZROP_POLICY                         1:0
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS_ZROP_POLICY_EVICT_FIRST             0x00000000
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS_ZROP_POLICY_EVICT_NORMAL             0x00000001
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS_ZROP_POLICY_EVICT_LAST             0x00000002

#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS                                     0x12dc
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS_POLICY                                 5:4
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS_POLICY_EVICT_FIRST              0x00000000
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS_POLICY_EVICT_NORMAL             0x00000001
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS_POLICY_EVICT_LAST               0x00000002
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS_ZROP_POLICY                            1:0
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS_ZROP_POLICY_EVICT_FIRST             0x00000000
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS_ZROP_POLICY_EVICT_NORMAL             0x00000001
#define LWCC97_SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS_ZROP_POLICY_EVICT_LAST             0x00000002

#define LWCC97_SET_ALPHA_TO_COVERAGE_DITHER_CONTROL                                                        0x12e0
#define LWCC97_SET_ALPHA_TO_COVERAGE_DITHER_CONTROL_DITHER_FOOTPRINT                                          3:0
#define LWCC97_SET_ALPHA_TO_COVERAGE_DITHER_CONTROL_DITHER_FOOTPRINT_PIXELS_1X1                        0x00000000
#define LWCC97_SET_ALPHA_TO_COVERAGE_DITHER_CONTROL_DITHER_FOOTPRINT_PIXELS_2X2                        0x00000001
#define LWCC97_SET_ALPHA_TO_COVERAGE_DITHER_CONTROL_DITHER_FOOTPRINT_PIXELS_1X1_VIRTUAL_SAMPLES             0x00000002

#define LWCC97_SET_BLEND_STATE_PER_TARGET                                                                  0x12e4
#define LWCC97_SET_BLEND_STATE_PER_TARGET_ENABLE                                                              0:0
#define LWCC97_SET_BLEND_STATE_PER_TARGET_ENABLE_FALSE                                                 0x00000000
#define LWCC97_SET_BLEND_STATE_PER_TARGET_ENABLE_TRUE                                                  0x00000001

#define LWCC97_SET_DEPTH_WRITE                                                                             0x12e8
#define LWCC97_SET_DEPTH_WRITE_ENABLE                                                                         0:0
#define LWCC97_SET_DEPTH_WRITE_ENABLE_FALSE                                                            0x00000000
#define LWCC97_SET_DEPTH_WRITE_ENABLE_TRUE                                                             0x00000001

#define LWCC97_SET_ALPHA_TEST                                                                              0x12ec
#define LWCC97_SET_ALPHA_TEST_ENABLE                                                                          0:0
#define LWCC97_SET_ALPHA_TEST_ENABLE_FALSE                                                             0x00000000
#define LWCC97_SET_ALPHA_TEST_ENABLE_TRUE                                                              0x00000001

#define LWCC97_SET_INLINE_INDEX4X8_ALIGN                                                                   0x1300
#define LWCC97_SET_INLINE_INDEX4X8_ALIGN_COUNT                                                               29:0
#define LWCC97_SET_INLINE_INDEX4X8_ALIGN_START                                                              31:30

#define LWCC97_DRAW_INLINE_INDEX4X8                                                                        0x1304
#define LWCC97_DRAW_INLINE_INDEX4X8_INDEX0                                                                    7:0
#define LWCC97_DRAW_INLINE_INDEX4X8_INDEX1                                                                   15:8
#define LWCC97_DRAW_INLINE_INDEX4X8_INDEX2                                                                  23:16
#define LWCC97_DRAW_INLINE_INDEX4X8_INDEX3                                                                  31:24

#define LWCC97_D3D_SET_LWLL_MODE                                                                           0x1308
#define LWCC97_D3D_SET_LWLL_MODE_V                                                                           31:0
#define LWCC97_D3D_SET_LWLL_MODE_V_NONE                                                                0x00000001
#define LWCC97_D3D_SET_LWLL_MODE_V_CW                                                                  0x00000002
#define LWCC97_D3D_SET_LWLL_MODE_V_CCW                                                                 0x00000003

#define LWCC97_SET_DEPTH_FUNC                                                                              0x130c
#define LWCC97_SET_DEPTH_FUNC_V                                                                              31:0
#define LWCC97_SET_DEPTH_FUNC_V_OGL_NEVER                                                              0x00000200
#define LWCC97_SET_DEPTH_FUNC_V_OGL_LESS                                                               0x00000201
#define LWCC97_SET_DEPTH_FUNC_V_OGL_EQUAL                                                              0x00000202
#define LWCC97_SET_DEPTH_FUNC_V_OGL_LEQUAL                                                             0x00000203
#define LWCC97_SET_DEPTH_FUNC_V_OGL_GREATER                                                            0x00000204
#define LWCC97_SET_DEPTH_FUNC_V_OGL_NOTEQUAL                                                           0x00000205
#define LWCC97_SET_DEPTH_FUNC_V_OGL_GEQUAL                                                             0x00000206
#define LWCC97_SET_DEPTH_FUNC_V_OGL_ALWAYS                                                             0x00000207
#define LWCC97_SET_DEPTH_FUNC_V_D3D_NEVER                                                              0x00000001
#define LWCC97_SET_DEPTH_FUNC_V_D3D_LESS                                                               0x00000002
#define LWCC97_SET_DEPTH_FUNC_V_D3D_EQUAL                                                              0x00000003
#define LWCC97_SET_DEPTH_FUNC_V_D3D_LESSEQUAL                                                          0x00000004
#define LWCC97_SET_DEPTH_FUNC_V_D3D_GREATER                                                            0x00000005
#define LWCC97_SET_DEPTH_FUNC_V_D3D_NOTEQUAL                                                           0x00000006
#define LWCC97_SET_DEPTH_FUNC_V_D3D_GREATEREQUAL                                                       0x00000007
#define LWCC97_SET_DEPTH_FUNC_V_D3D_ALWAYS                                                             0x00000008

#define LWCC97_SET_ALPHA_REF                                                                               0x1310
#define LWCC97_SET_ALPHA_REF_V                                                                               31:0

#define LWCC97_SET_ALPHA_FUNC                                                                              0x1314
#define LWCC97_SET_ALPHA_FUNC_V                                                                              31:0
#define LWCC97_SET_ALPHA_FUNC_V_OGL_NEVER                                                              0x00000200
#define LWCC97_SET_ALPHA_FUNC_V_OGL_LESS                                                               0x00000201
#define LWCC97_SET_ALPHA_FUNC_V_OGL_EQUAL                                                              0x00000202
#define LWCC97_SET_ALPHA_FUNC_V_OGL_LEQUAL                                                             0x00000203
#define LWCC97_SET_ALPHA_FUNC_V_OGL_GREATER                                                            0x00000204
#define LWCC97_SET_ALPHA_FUNC_V_OGL_NOTEQUAL                                                           0x00000205
#define LWCC97_SET_ALPHA_FUNC_V_OGL_GEQUAL                                                             0x00000206
#define LWCC97_SET_ALPHA_FUNC_V_OGL_ALWAYS                                                             0x00000207
#define LWCC97_SET_ALPHA_FUNC_V_D3D_NEVER                                                              0x00000001
#define LWCC97_SET_ALPHA_FUNC_V_D3D_LESS                                                               0x00000002
#define LWCC97_SET_ALPHA_FUNC_V_D3D_EQUAL                                                              0x00000003
#define LWCC97_SET_ALPHA_FUNC_V_D3D_LESSEQUAL                                                          0x00000004
#define LWCC97_SET_ALPHA_FUNC_V_D3D_GREATER                                                            0x00000005
#define LWCC97_SET_ALPHA_FUNC_V_D3D_NOTEQUAL                                                           0x00000006
#define LWCC97_SET_ALPHA_FUNC_V_D3D_GREATEREQUAL                                                       0x00000007
#define LWCC97_SET_ALPHA_FUNC_V_D3D_ALWAYS                                                             0x00000008

#define LWCC97_SET_DRAW_AUTO_STRIDE                                                                        0x1318
#define LWCC97_SET_DRAW_AUTO_STRIDE_V                                                                        11:0

#define LWCC97_SET_BLEND_CONST_RED                                                                         0x131c
#define LWCC97_SET_BLEND_CONST_RED_V                                                                         31:0

#define LWCC97_SET_BLEND_CONST_GREEN                                                                       0x1320
#define LWCC97_SET_BLEND_CONST_GREEN_V                                                                       31:0

#define LWCC97_SET_BLEND_CONST_BLUE                                                                        0x1324
#define LWCC97_SET_BLEND_CONST_BLUE_V                                                                        31:0

#define LWCC97_SET_BLEND_CONST_ALPHA                                                                       0x1328
#define LWCC97_SET_BLEND_CONST_ALPHA_V                                                                       31:0

#define LWCC97_ILWALIDATE_SAMPLER_CACHE                                                                    0x1330
#define LWCC97_ILWALIDATE_SAMPLER_CACHE_LINES                                                                 0:0
#define LWCC97_ILWALIDATE_SAMPLER_CACHE_LINES_ALL                                                      0x00000000
#define LWCC97_ILWALIDATE_SAMPLER_CACHE_LINES_ONE                                                      0x00000001
#define LWCC97_ILWALIDATE_SAMPLER_CACHE_TAG                                                                  25:4

#define LWCC97_ILWALIDATE_TEXTURE_HEADER_CACHE                                                             0x1334
#define LWCC97_ILWALIDATE_TEXTURE_HEADER_CACHE_LINES                                                          0:0
#define LWCC97_ILWALIDATE_TEXTURE_HEADER_CACHE_LINES_ALL                                               0x00000000
#define LWCC97_ILWALIDATE_TEXTURE_HEADER_CACHE_LINES_ONE                                               0x00000001
#define LWCC97_ILWALIDATE_TEXTURE_HEADER_CACHE_TAG                                                           25:4

#define LWCC97_ILWALIDATE_TEXTURE_DATA_CACHE                                                               0x1338
#define LWCC97_ILWALIDATE_TEXTURE_DATA_CACHE_LINES                                                            0:0
#define LWCC97_ILWALIDATE_TEXTURE_DATA_CACHE_LINES_ALL                                                 0x00000000
#define LWCC97_ILWALIDATE_TEXTURE_DATA_CACHE_LINES_ONE                                                 0x00000001
#define LWCC97_ILWALIDATE_TEXTURE_DATA_CACHE_TAG                                                             25:4

#define LWCC97_SET_BLEND_SEPARATE_FOR_ALPHA                                                                0x133c
#define LWCC97_SET_BLEND_SEPARATE_FOR_ALPHA_ENABLE                                                            0:0
#define LWCC97_SET_BLEND_SEPARATE_FOR_ALPHA_ENABLE_FALSE                                               0x00000000
#define LWCC97_SET_BLEND_SEPARATE_FOR_ALPHA_ENABLE_TRUE                                                0x00000001

#define LWCC97_SET_BLEND_COLOR_OP                                                                          0x1340
#define LWCC97_SET_BLEND_COLOR_OP_V                                                                          31:0
#define LWCC97_SET_BLEND_COLOR_OP_V_OGL_FUNC_SUBTRACT                                                  0x0000800A
#define LWCC97_SET_BLEND_COLOR_OP_V_OGL_FUNC_REVERSE_SUBTRACT                                          0x0000800B
#define LWCC97_SET_BLEND_COLOR_OP_V_OGL_FUNC_ADD                                                       0x00008006
#define LWCC97_SET_BLEND_COLOR_OP_V_OGL_MIN                                                            0x00008007
#define LWCC97_SET_BLEND_COLOR_OP_V_OGL_MAX                                                            0x00008008
#define LWCC97_SET_BLEND_COLOR_OP_V_D3D_ADD                                                            0x00000001
#define LWCC97_SET_BLEND_COLOR_OP_V_D3D_SUBTRACT                                                       0x00000002
#define LWCC97_SET_BLEND_COLOR_OP_V_D3D_REVSUBTRACT                                                    0x00000003
#define LWCC97_SET_BLEND_COLOR_OP_V_D3D_MIN                                                            0x00000004
#define LWCC97_SET_BLEND_COLOR_OP_V_D3D_MAX                                                            0x00000005

#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF                                                                0x1344
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V                                                                31:0
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ZERO                                                 0x00004000
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ONE                                                  0x00004001
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_SRC_COLOR                                            0x00004300
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_SRC_COLOR                                  0x00004301
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_SRC_ALPHA                                            0x00004302
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA                                  0x00004303
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_DST_ALPHA                                            0x00004304
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_DST_ALPHA                                  0x00004305
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_DST_COLOR                                            0x00004306
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_DST_COLOR                                  0x00004307
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_SRC_ALPHA_SATURATE                                   0x00004308
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_CONSTANT_COLOR                                       0x0000C001
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_CONSTANT_COLOR                             0x0000C002
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_CONSTANT_ALPHA                                       0x0000C003
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_CONSTANT_ALPHA                             0x0000C004
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_SRC1COLOR                                            0x0000C900
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ILWSRC1COLOR                                         0x0000C901
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_SRC1ALPHA                                            0x0000C902
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_ILWSRC1ALPHA                                         0x0000C903
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ZERO                                                 0x00000001
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ONE                                                  0x00000002
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_SRCCOLOR                                             0x00000003
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ILWSRCCOLOR                                          0x00000004
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_SRCALPHA                                             0x00000005
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ILWSRCALPHA                                          0x00000006
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_DESTALPHA                                            0x00000007
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ILWDESTALPHA                                         0x00000008
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_DESTCOLOR                                            0x00000009
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ILWDESTCOLOR                                         0x0000000A
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_SRCALPHASAT                                          0x0000000B
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_BOTHSRCALPHA                                         0x0000000C
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_BOTHILWSRCALPHA                                      0x0000000D
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_BLENDFACTOR                                          0x0000000E
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ILWBLENDFACTOR                                       0x0000000F
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_SRC1COLOR                                            0x00000010
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ILWSRC1COLOR                                         0x00000011
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_SRC1ALPHA                                            0x00000012
#define LWCC97_SET_BLEND_COLOR_SOURCE_COEFF_V_D3D_ILWSRC1ALPHA                                         0x00000013

#define LWCC97_SET_BLEND_COLOR_DEST_COEFF                                                                  0x1348
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V                                                                  31:0
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ZERO                                                   0x00004000
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ONE                                                    0x00004001
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_SRC_COLOR                                              0x00004300
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_SRC_COLOR                                    0x00004301
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_SRC_ALPHA                                              0x00004302
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA                                    0x00004303
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_DST_ALPHA                                              0x00004304
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_DST_ALPHA                                    0x00004305
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_DST_COLOR                                              0x00004306
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_DST_COLOR                                    0x00004307
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_SRC_ALPHA_SATURATE                                     0x00004308
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_CONSTANT_COLOR                                         0x0000C001
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_CONSTANT_COLOR                               0x0000C002
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_CONSTANT_ALPHA                                         0x0000C003
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_CONSTANT_ALPHA                               0x0000C004
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_SRC1COLOR                                              0x0000C900
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ILWSRC1COLOR                                           0x0000C901
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_SRC1ALPHA                                              0x0000C902
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_OGL_ILWSRC1ALPHA                                           0x0000C903
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ZERO                                                   0x00000001
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ONE                                                    0x00000002
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_SRCCOLOR                                               0x00000003
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ILWSRCCOLOR                                            0x00000004
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_SRCALPHA                                               0x00000005
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ILWSRCALPHA                                            0x00000006
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_DESTALPHA                                              0x00000007
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ILWDESTALPHA                                           0x00000008
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_DESTCOLOR                                              0x00000009
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ILWDESTCOLOR                                           0x0000000A
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_SRCALPHASAT                                            0x0000000B
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_BLENDFACTOR                                            0x0000000E
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ILWBLENDFACTOR                                         0x0000000F
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_SRC1COLOR                                              0x00000010
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ILWSRC1COLOR                                           0x00000011
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_SRC1ALPHA                                              0x00000012
#define LWCC97_SET_BLEND_COLOR_DEST_COEFF_V_D3D_ILWSRC1ALPHA                                           0x00000013

#define LWCC97_SET_BLEND_ALPHA_OP                                                                          0x134c
#define LWCC97_SET_BLEND_ALPHA_OP_V                                                                          31:0
#define LWCC97_SET_BLEND_ALPHA_OP_V_OGL_FUNC_SUBTRACT                                                  0x0000800A
#define LWCC97_SET_BLEND_ALPHA_OP_V_OGL_FUNC_REVERSE_SUBTRACT                                          0x0000800B
#define LWCC97_SET_BLEND_ALPHA_OP_V_OGL_FUNC_ADD                                                       0x00008006
#define LWCC97_SET_BLEND_ALPHA_OP_V_OGL_MIN                                                            0x00008007
#define LWCC97_SET_BLEND_ALPHA_OP_V_OGL_MAX                                                            0x00008008
#define LWCC97_SET_BLEND_ALPHA_OP_V_D3D_ADD                                                            0x00000001
#define LWCC97_SET_BLEND_ALPHA_OP_V_D3D_SUBTRACT                                                       0x00000002
#define LWCC97_SET_BLEND_ALPHA_OP_V_D3D_REVSUBTRACT                                                    0x00000003
#define LWCC97_SET_BLEND_ALPHA_OP_V_D3D_MIN                                                            0x00000004
#define LWCC97_SET_BLEND_ALPHA_OP_V_D3D_MAX                                                            0x00000005

#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF                                                                0x1350
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V                                                                31:0
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ZERO                                                 0x00004000
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ONE                                                  0x00004001
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_SRC_COLOR                                            0x00004300
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_SRC_COLOR                                  0x00004301
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_SRC_ALPHA                                            0x00004302
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA                                  0x00004303
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_DST_ALPHA                                            0x00004304
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_DST_ALPHA                                  0x00004305
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_DST_COLOR                                            0x00004306
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_DST_COLOR                                  0x00004307
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_SRC_ALPHA_SATURATE                                   0x00004308
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_CONSTANT_COLOR                                       0x0000C001
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_CONSTANT_COLOR                             0x0000C002
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_CONSTANT_ALPHA                                       0x0000C003
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_CONSTANT_ALPHA                             0x0000C004
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_SRC1COLOR                                            0x0000C900
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ILWSRC1COLOR                                         0x0000C901
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_SRC1ALPHA                                            0x0000C902
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_ILWSRC1ALPHA                                         0x0000C903
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ZERO                                                 0x00000001
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ONE                                                  0x00000002
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_SRCCOLOR                                             0x00000003
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ILWSRCCOLOR                                          0x00000004
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_SRCALPHA                                             0x00000005
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ILWSRCALPHA                                          0x00000006
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_DESTALPHA                                            0x00000007
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ILWDESTALPHA                                         0x00000008
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_DESTCOLOR                                            0x00000009
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ILWDESTCOLOR                                         0x0000000A
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_SRCALPHASAT                                          0x0000000B
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_BOTHSRCALPHA                                         0x0000000C
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_BOTHILWSRCALPHA                                      0x0000000D
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_BLENDFACTOR                                          0x0000000E
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ILWBLENDFACTOR                                       0x0000000F
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_SRC1COLOR                                            0x00000010
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ILWSRC1COLOR                                         0x00000011
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_SRC1ALPHA                                            0x00000012
#define LWCC97_SET_BLEND_ALPHA_SOURCE_COEFF_V_D3D_ILWSRC1ALPHA                                         0x00000013

#define LWCC97_SET_GLOBAL_COLOR_KEY                                                                        0x1354
#define LWCC97_SET_GLOBAL_COLOR_KEY_ENABLE                                                                    0:0
#define LWCC97_SET_GLOBAL_COLOR_KEY_ENABLE_FALSE                                                       0x00000000
#define LWCC97_SET_GLOBAL_COLOR_KEY_ENABLE_TRUE                                                        0x00000001

#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF                                                                  0x1358
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V                                                                  31:0
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ZERO                                                   0x00004000
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ONE                                                    0x00004001
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_SRC_COLOR                                              0x00004300
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_SRC_COLOR                                    0x00004301
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_SRC_ALPHA                                              0x00004302
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA                                    0x00004303
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_DST_ALPHA                                              0x00004304
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_DST_ALPHA                                    0x00004305
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_DST_COLOR                                              0x00004306
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_DST_COLOR                                    0x00004307
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_SRC_ALPHA_SATURATE                                     0x00004308
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_CONSTANT_COLOR                                         0x0000C001
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_CONSTANT_COLOR                               0x0000C002
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_CONSTANT_ALPHA                                         0x0000C003
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_CONSTANT_ALPHA                               0x0000C004
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_SRC1COLOR                                              0x0000C900
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ILWSRC1COLOR                                           0x0000C901
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_SRC1ALPHA                                              0x0000C902
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ILWSRC1ALPHA                                           0x0000C903
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ZERO                                                   0x00000001
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ONE                                                    0x00000002
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_SRCCOLOR                                               0x00000003
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ILWSRCCOLOR                                            0x00000004
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_SRCALPHA                                               0x00000005
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ILWSRCALPHA                                            0x00000006
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_DESTALPHA                                              0x00000007
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ILWDESTALPHA                                           0x00000008
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_DESTCOLOR                                              0x00000009
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ILWDESTCOLOR                                           0x0000000A
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_SRCALPHASAT                                            0x0000000B
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_BLENDFACTOR                                            0x0000000E
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ILWBLENDFACTOR                                         0x0000000F
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_SRC1COLOR                                              0x00000010
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ILWSRC1COLOR                                           0x00000011
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_SRC1ALPHA                                              0x00000012
#define LWCC97_SET_BLEND_ALPHA_DEST_COEFF_V_D3D_ILWSRC1ALPHA                                           0x00000013

#define LWCC97_SET_SINGLE_ROP_CONTROL                                                                      0x135c
#define LWCC97_SET_SINGLE_ROP_CONTROL_ENABLE                                                                  0:0
#define LWCC97_SET_SINGLE_ROP_CONTROL_ENABLE_FALSE                                                     0x00000000
#define LWCC97_SET_SINGLE_ROP_CONTROL_ENABLE_TRUE                                                      0x00000001

#define LWCC97_SET_BLEND(i)                                                                        (0x1360+(i)*4)
#define LWCC97_SET_BLEND_ENABLE                                                                               0:0
#define LWCC97_SET_BLEND_ENABLE_FALSE                                                                  0x00000000
#define LWCC97_SET_BLEND_ENABLE_TRUE                                                                   0x00000001

#define LWCC97_SET_STENCIL_TEST                                                                            0x1380
#define LWCC97_SET_STENCIL_TEST_ENABLE                                                                        0:0
#define LWCC97_SET_STENCIL_TEST_ENABLE_FALSE                                                           0x00000000
#define LWCC97_SET_STENCIL_TEST_ENABLE_TRUE                                                            0x00000001

#define LWCC97_SET_STENCIL_OP_FAIL                                                                         0x1384
#define LWCC97_SET_STENCIL_OP_FAIL_V                                                                         31:0
#define LWCC97_SET_STENCIL_OP_FAIL_V_OGL_KEEP                                                          0x00001E00
#define LWCC97_SET_STENCIL_OP_FAIL_V_OGL_ZERO                                                          0x00000000
#define LWCC97_SET_STENCIL_OP_FAIL_V_OGL_REPLACE                                                       0x00001E01
#define LWCC97_SET_STENCIL_OP_FAIL_V_OGL_INCRSAT                                                       0x00001E02
#define LWCC97_SET_STENCIL_OP_FAIL_V_OGL_DECRSAT                                                       0x00001E03
#define LWCC97_SET_STENCIL_OP_FAIL_V_OGL_ILWERT                                                        0x0000150A
#define LWCC97_SET_STENCIL_OP_FAIL_V_OGL_INCR                                                          0x00008507
#define LWCC97_SET_STENCIL_OP_FAIL_V_OGL_DECR                                                          0x00008508
#define LWCC97_SET_STENCIL_OP_FAIL_V_D3D_KEEP                                                          0x00000001
#define LWCC97_SET_STENCIL_OP_FAIL_V_D3D_ZERO                                                          0x00000002
#define LWCC97_SET_STENCIL_OP_FAIL_V_D3D_REPLACE                                                       0x00000003
#define LWCC97_SET_STENCIL_OP_FAIL_V_D3D_INCRSAT                                                       0x00000004
#define LWCC97_SET_STENCIL_OP_FAIL_V_D3D_DECRSAT                                                       0x00000005
#define LWCC97_SET_STENCIL_OP_FAIL_V_D3D_ILWERT                                                        0x00000006
#define LWCC97_SET_STENCIL_OP_FAIL_V_D3D_INCR                                                          0x00000007
#define LWCC97_SET_STENCIL_OP_FAIL_V_D3D_DECR                                                          0x00000008

#define LWCC97_SET_STENCIL_OP_ZFAIL                                                                        0x1388
#define LWCC97_SET_STENCIL_OP_ZFAIL_V                                                                        31:0
#define LWCC97_SET_STENCIL_OP_ZFAIL_V_OGL_KEEP                                                         0x00001E00
#define LWCC97_SET_STENCIL_OP_ZFAIL_V_OGL_ZERO                                                         0x00000000
#define LWCC97_SET_STENCIL_OP_ZFAIL_V_OGL_REPLACE                                                      0x00001E01
#define LWCC97_SET_STENCIL_OP_ZFAIL_V_OGL_INCRSAT                                                      0x00001E02
#define LWCC97_SET_STENCIL_OP_ZFAIL_V_OGL_DECRSAT                                                      0x00001E03
#define LWCC97_SET_STENCIL_OP_ZFAIL_V_OGL_ILWERT                                                       0x0000150A
#define LWCC97_SET_STENCIL_OP_ZFAIL_V_OGL_INCR                                                         0x00008507
#define LWCC97_SET_STENCIL_OP_ZFAIL_V_OGL_DECR                                                         0x00008508
#define LWCC97_SET_STENCIL_OP_ZFAIL_V_D3D_KEEP                                                         0x00000001
#define LWCC97_SET_STENCIL_OP_ZFAIL_V_D3D_ZERO                                                         0x00000002
#define LWCC97_SET_STENCIL_OP_ZFAIL_V_D3D_REPLACE                                                      0x00000003
#define LWCC97_SET_STENCIL_OP_ZFAIL_V_D3D_INCRSAT                                                      0x00000004
#define LWCC97_SET_STENCIL_OP_ZFAIL_V_D3D_DECRSAT                                                      0x00000005
#define LWCC97_SET_STENCIL_OP_ZFAIL_V_D3D_ILWERT                                                       0x00000006
#define LWCC97_SET_STENCIL_OP_ZFAIL_V_D3D_INCR                                                         0x00000007
#define LWCC97_SET_STENCIL_OP_ZFAIL_V_D3D_DECR                                                         0x00000008

#define LWCC97_SET_STENCIL_OP_ZPASS                                                                        0x138c
#define LWCC97_SET_STENCIL_OP_ZPASS_V                                                                        31:0
#define LWCC97_SET_STENCIL_OP_ZPASS_V_OGL_KEEP                                                         0x00001E00
#define LWCC97_SET_STENCIL_OP_ZPASS_V_OGL_ZERO                                                         0x00000000
#define LWCC97_SET_STENCIL_OP_ZPASS_V_OGL_REPLACE                                                      0x00001E01
#define LWCC97_SET_STENCIL_OP_ZPASS_V_OGL_INCRSAT                                                      0x00001E02
#define LWCC97_SET_STENCIL_OP_ZPASS_V_OGL_DECRSAT                                                      0x00001E03
#define LWCC97_SET_STENCIL_OP_ZPASS_V_OGL_ILWERT                                                       0x0000150A
#define LWCC97_SET_STENCIL_OP_ZPASS_V_OGL_INCR                                                         0x00008507
#define LWCC97_SET_STENCIL_OP_ZPASS_V_OGL_DECR                                                         0x00008508
#define LWCC97_SET_STENCIL_OP_ZPASS_V_D3D_KEEP                                                         0x00000001
#define LWCC97_SET_STENCIL_OP_ZPASS_V_D3D_ZERO                                                         0x00000002
#define LWCC97_SET_STENCIL_OP_ZPASS_V_D3D_REPLACE                                                      0x00000003
#define LWCC97_SET_STENCIL_OP_ZPASS_V_D3D_INCRSAT                                                      0x00000004
#define LWCC97_SET_STENCIL_OP_ZPASS_V_D3D_DECRSAT                                                      0x00000005
#define LWCC97_SET_STENCIL_OP_ZPASS_V_D3D_ILWERT                                                       0x00000006
#define LWCC97_SET_STENCIL_OP_ZPASS_V_D3D_INCR                                                         0x00000007
#define LWCC97_SET_STENCIL_OP_ZPASS_V_D3D_DECR                                                         0x00000008

#define LWCC97_SET_STENCIL_FUNC                                                                            0x1390
#define LWCC97_SET_STENCIL_FUNC_V                                                                            31:0
#define LWCC97_SET_STENCIL_FUNC_V_OGL_NEVER                                                            0x00000200
#define LWCC97_SET_STENCIL_FUNC_V_OGL_LESS                                                             0x00000201
#define LWCC97_SET_STENCIL_FUNC_V_OGL_EQUAL                                                            0x00000202
#define LWCC97_SET_STENCIL_FUNC_V_OGL_LEQUAL                                                           0x00000203
#define LWCC97_SET_STENCIL_FUNC_V_OGL_GREATER                                                          0x00000204
#define LWCC97_SET_STENCIL_FUNC_V_OGL_NOTEQUAL                                                         0x00000205
#define LWCC97_SET_STENCIL_FUNC_V_OGL_GEQUAL                                                           0x00000206
#define LWCC97_SET_STENCIL_FUNC_V_OGL_ALWAYS                                                           0x00000207
#define LWCC97_SET_STENCIL_FUNC_V_D3D_NEVER                                                            0x00000001
#define LWCC97_SET_STENCIL_FUNC_V_D3D_LESS                                                             0x00000002
#define LWCC97_SET_STENCIL_FUNC_V_D3D_EQUAL                                                            0x00000003
#define LWCC97_SET_STENCIL_FUNC_V_D3D_LESSEQUAL                                                        0x00000004
#define LWCC97_SET_STENCIL_FUNC_V_D3D_GREATER                                                          0x00000005
#define LWCC97_SET_STENCIL_FUNC_V_D3D_NOTEQUAL                                                         0x00000006
#define LWCC97_SET_STENCIL_FUNC_V_D3D_GREATEREQUAL                                                     0x00000007
#define LWCC97_SET_STENCIL_FUNC_V_D3D_ALWAYS                                                           0x00000008

#define LWCC97_SET_STENCIL_FUNC_REF                                                                        0x1394
#define LWCC97_SET_STENCIL_FUNC_REF_V                                                                         7:0

#define LWCC97_SET_STENCIL_FUNC_MASK                                                                       0x1398
#define LWCC97_SET_STENCIL_FUNC_MASK_V                                                                        7:0

#define LWCC97_SET_STENCIL_MASK                                                                            0x139c
#define LWCC97_SET_STENCIL_MASK_V                                                                             7:0

#define LWCC97_SET_DRAW_AUTO_START                                                                         0x13a4
#define LWCC97_SET_DRAW_AUTO_START_BYTE_COUNT                                                                31:0

#define LWCC97_SET_PS_SATURATE                                                                             0x13a8
#define LWCC97_SET_PS_SATURATE_OUTPUT0                                                                        0:0
#define LWCC97_SET_PS_SATURATE_OUTPUT0_FALSE                                                           0x00000000
#define LWCC97_SET_PS_SATURATE_OUTPUT0_TRUE                                                            0x00000001
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE0                                                                   1:1
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE0_ZERO_TO_PLUS_ONE                                           0x00000000
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE0_MINUS_ONE_TO_PLUS_ONE                                      0x00000001
#define LWCC97_SET_PS_SATURATE_OUTPUT1                                                                        4:4
#define LWCC97_SET_PS_SATURATE_OUTPUT1_FALSE                                                           0x00000000
#define LWCC97_SET_PS_SATURATE_OUTPUT1_TRUE                                                            0x00000001
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE1                                                                   5:5
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE1_ZERO_TO_PLUS_ONE                                           0x00000000
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE1_MINUS_ONE_TO_PLUS_ONE                                      0x00000001
#define LWCC97_SET_PS_SATURATE_OUTPUT2                                                                        8:8
#define LWCC97_SET_PS_SATURATE_OUTPUT2_FALSE                                                           0x00000000
#define LWCC97_SET_PS_SATURATE_OUTPUT2_TRUE                                                            0x00000001
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE2                                                                   9:9
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE2_ZERO_TO_PLUS_ONE                                           0x00000000
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE2_MINUS_ONE_TO_PLUS_ONE                                      0x00000001
#define LWCC97_SET_PS_SATURATE_OUTPUT3                                                                      12:12
#define LWCC97_SET_PS_SATURATE_OUTPUT3_FALSE                                                           0x00000000
#define LWCC97_SET_PS_SATURATE_OUTPUT3_TRUE                                                            0x00000001
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE3                                                                 13:13
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE3_ZERO_TO_PLUS_ONE                                           0x00000000
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE3_MINUS_ONE_TO_PLUS_ONE                                      0x00000001
#define LWCC97_SET_PS_SATURATE_OUTPUT4                                                                      16:16
#define LWCC97_SET_PS_SATURATE_OUTPUT4_FALSE                                                           0x00000000
#define LWCC97_SET_PS_SATURATE_OUTPUT4_TRUE                                                            0x00000001
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE4                                                                 17:17
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE4_ZERO_TO_PLUS_ONE                                           0x00000000
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE4_MINUS_ONE_TO_PLUS_ONE                                      0x00000001
#define LWCC97_SET_PS_SATURATE_OUTPUT5                                                                      20:20
#define LWCC97_SET_PS_SATURATE_OUTPUT5_FALSE                                                           0x00000000
#define LWCC97_SET_PS_SATURATE_OUTPUT5_TRUE                                                            0x00000001
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE5                                                                 21:21
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE5_ZERO_TO_PLUS_ONE                                           0x00000000
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE5_MINUS_ONE_TO_PLUS_ONE                                      0x00000001
#define LWCC97_SET_PS_SATURATE_OUTPUT6                                                                      24:24
#define LWCC97_SET_PS_SATURATE_OUTPUT6_FALSE                                                           0x00000000
#define LWCC97_SET_PS_SATURATE_OUTPUT6_TRUE                                                            0x00000001
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE6                                                                 25:25
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE6_ZERO_TO_PLUS_ONE                                           0x00000000
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE6_MINUS_ONE_TO_PLUS_ONE                                      0x00000001
#define LWCC97_SET_PS_SATURATE_OUTPUT7                                                                      28:28
#define LWCC97_SET_PS_SATURATE_OUTPUT7_FALSE                                                           0x00000000
#define LWCC97_SET_PS_SATURATE_OUTPUT7_TRUE                                                            0x00000001
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE7                                                                 29:29
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE7_ZERO_TO_PLUS_ONE                                           0x00000000
#define LWCC97_SET_PS_SATURATE_CLAMP_RANGE7_MINUS_ONE_TO_PLUS_ONE                                      0x00000001

#define LWCC97_SET_WINDOW_ORIGIN                                                                           0x13ac
#define LWCC97_SET_WINDOW_ORIGIN_MODE                                                                         0:0
#define LWCC97_SET_WINDOW_ORIGIN_MODE_UPPER_LEFT                                                       0x00000000
#define LWCC97_SET_WINDOW_ORIGIN_MODE_LOWER_LEFT                                                       0x00000001
#define LWCC97_SET_WINDOW_ORIGIN_FLIP_Y                                                                       4:4
#define LWCC97_SET_WINDOW_ORIGIN_FLIP_Y_FALSE                                                          0x00000000
#define LWCC97_SET_WINDOW_ORIGIN_FLIP_Y_TRUE                                                           0x00000001

#define LWCC97_SET_LINE_WIDTH_FLOAT                                                                        0x13b0
#define LWCC97_SET_LINE_WIDTH_FLOAT_V                                                                        31:0

#define LWCC97_SET_ALIASED_LINE_WIDTH_FLOAT                                                                0x13b4
#define LWCC97_SET_ALIASED_LINE_WIDTH_FLOAT_V                                                                31:0

#define LWCC97_SET_LINE_MULTISAMPLE_OVERRIDE                                                               0x1418
#define LWCC97_SET_LINE_MULTISAMPLE_OVERRIDE_ENABLE                                                           0:0
#define LWCC97_SET_LINE_MULTISAMPLE_OVERRIDE_ENABLE_FALSE                                              0x00000000
#define LWCC97_SET_LINE_MULTISAMPLE_OVERRIDE_ENABLE_TRUE                                               0x00000001

#define LWCC97_ILWALIDATE_SAMPLER_CACHE_NO_WFI                                                             0x1424
#define LWCC97_ILWALIDATE_SAMPLER_CACHE_NO_WFI_LINES                                                          0:0
#define LWCC97_ILWALIDATE_SAMPLER_CACHE_NO_WFI_LINES_ALL                                               0x00000000
#define LWCC97_ILWALIDATE_SAMPLER_CACHE_NO_WFI_LINES_ONE                                               0x00000001
#define LWCC97_ILWALIDATE_SAMPLER_CACHE_NO_WFI_TAG                                                           25:4

#define LWCC97_ILWALIDATE_TEXTURE_HEADER_CACHE_NO_WFI                                                      0x1428
#define LWCC97_ILWALIDATE_TEXTURE_HEADER_CACHE_NO_WFI_LINES                                                   0:0
#define LWCC97_ILWALIDATE_TEXTURE_HEADER_CACHE_NO_WFI_LINES_ALL                                        0x00000000
#define LWCC97_ILWALIDATE_TEXTURE_HEADER_CACHE_NO_WFI_LINES_ONE                                        0x00000001
#define LWCC97_ILWALIDATE_TEXTURE_HEADER_CACHE_NO_WFI_TAG                                                    25:4

#define LWCC97_SET_GLOBAL_BASE_VERTEX_INDEX                                                                0x1434
#define LWCC97_SET_GLOBAL_BASE_VERTEX_INDEX_V                                                                31:0

#define LWCC97_SET_GLOBAL_BASE_INSTANCE_INDEX                                                              0x1438
#define LWCC97_SET_GLOBAL_BASE_INSTANCE_INDEX_V                                                              31:0

#define LWCC97_SET_GS_SPILL_REGION_A                                                                       0x1444
#define LWCC97_SET_GS_SPILL_REGION_A_ADDRESS_UPPER                                                            7:0

#define LWCC97_SET_GS_SPILL_REGION_B                                                                       0x1448
#define LWCC97_SET_GS_SPILL_REGION_B_ADDRESS_LOWER                                                           31:0

#define LWCC97_SET_GS_SPILL_REGION_C                                                                       0x144c
#define LWCC97_SET_GS_SPILL_REGION_C_SIZE                                                                    31:0

#define LWCC97_SET_PS_WARP_WATERMARKS                                                                      0x1450
#define LWCC97_SET_PS_WARP_WATERMARKS_LOW                                                                    15:0
#define LWCC97_SET_PS_WARP_WATERMARKS_HIGH                                                                  31:16

#define LWCC97_SET_PS_REGISTER_WATERMARKS                                                                  0x1454
#define LWCC97_SET_PS_REGISTER_WATERMARKS_LOW                                                                15:0
#define LWCC97_SET_PS_REGISTER_WATERMARKS_HIGH                                                              31:16

#define LWCC97_STORE_ZLWLL                                                                                 0x1464
#define LWCC97_STORE_ZLWLL_V                                                                                  0:0

#define LWCC97_SET_ITERATED_BLEND_CONSTANT_RED(j)                                                 (0x1480+(j)*16)
#define LWCC97_SET_ITERATED_BLEND_CONSTANT_RED_V                                                             15:0

#define LWCC97_SET_ITERATED_BLEND_CONSTANT_GREEN(j)                                               (0x1484+(j)*16)
#define LWCC97_SET_ITERATED_BLEND_CONSTANT_GREEN_V                                                           15:0

#define LWCC97_SET_ITERATED_BLEND_CONSTANT_BLUE(j)                                                (0x1488+(j)*16)
#define LWCC97_SET_ITERATED_BLEND_CONSTANT_BLUE_V                                                            15:0

#define LWCC97_LOAD_ZLWLL                                                                                  0x1500
#define LWCC97_LOAD_ZLWLL_V                                                                                   0:0

#define LWCC97_SET_SURFACE_CLIP_ID_HEIGHT                                                                  0x1504
#define LWCC97_SET_SURFACE_CLIP_ID_HEIGHT_V                                                                  31:0

#define LWCC97_SET_CLIP_ID_CLEAR_RECT_HORIZONTAL                                                           0x1508
#define LWCC97_SET_CLIP_ID_CLEAR_RECT_HORIZONTAL_XMIN                                                        15:0
#define LWCC97_SET_CLIP_ID_CLEAR_RECT_HORIZONTAL_XMAX                                                       31:16

#define LWCC97_SET_CLIP_ID_CLEAR_RECT_VERTICAL                                                             0x150c
#define LWCC97_SET_CLIP_ID_CLEAR_RECT_VERTICAL_YMIN                                                          15:0
#define LWCC97_SET_CLIP_ID_CLEAR_RECT_VERTICAL_YMAX                                                         31:16

#define LWCC97_SET_USER_CLIP_ENABLE                                                                        0x1510
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE0                                                                    0:0
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE0_FALSE                                                       0x00000000
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE0_TRUE                                                        0x00000001
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE1                                                                    1:1
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE1_FALSE                                                       0x00000000
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE1_TRUE                                                        0x00000001
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE2                                                                    2:2
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE2_FALSE                                                       0x00000000
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE2_TRUE                                                        0x00000001
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE3                                                                    3:3
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE3_FALSE                                                       0x00000000
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE3_TRUE                                                        0x00000001
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE4                                                                    4:4
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE4_FALSE                                                       0x00000000
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE4_TRUE                                                        0x00000001
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE5                                                                    5:5
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE5_FALSE                                                       0x00000000
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE5_TRUE                                                        0x00000001
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE6                                                                    6:6
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE6_FALSE                                                       0x00000000
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE6_TRUE                                                        0x00000001
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE7                                                                    7:7
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE7_FALSE                                                       0x00000000
#define LWCC97_SET_USER_CLIP_ENABLE_PLANE7_TRUE                                                        0x00000001

#define LWCC97_SET_ZPASS_PIXEL_COUNT                                                                       0x1514
#define LWCC97_SET_ZPASS_PIXEL_COUNT_ENABLE                                                                   0:0
#define LWCC97_SET_ZPASS_PIXEL_COUNT_ENABLE_FALSE                                                      0x00000000
#define LWCC97_SET_ZPASS_PIXEL_COUNT_ENABLE_TRUE                                                       0x00000001

#define LWCC97_SET_POINT_SIZE                                                                              0x1518
#define LWCC97_SET_POINT_SIZE_V                                                                              31:0

#define LWCC97_SET_ZLWLL_STATS                                                                             0x151c
#define LWCC97_SET_ZLWLL_STATS_ENABLE                                                                         0:0
#define LWCC97_SET_ZLWLL_STATS_ENABLE_FALSE                                                            0x00000000
#define LWCC97_SET_ZLWLL_STATS_ENABLE_TRUE                                                             0x00000001

#define LWCC97_SET_POINT_SPRITE                                                                            0x1520
#define LWCC97_SET_POINT_SPRITE_ENABLE                                                                        0:0
#define LWCC97_SET_POINT_SPRITE_ENABLE_FALSE                                                           0x00000000
#define LWCC97_SET_POINT_SPRITE_ENABLE_TRUE                                                            0x00000001

#define LWCC97_SET_SHADER_EXCEPTIONS                                                                       0x1528
#define LWCC97_SET_SHADER_EXCEPTIONS_ENABLE                                                                   0:0
#define LWCC97_SET_SHADER_EXCEPTIONS_ENABLE_FALSE                                                      0x00000000
#define LWCC97_SET_SHADER_EXCEPTIONS_ENABLE_TRUE                                                       0x00000001

#define LWCC97_CLEAR_REPORT_VALUE                                                                          0x1530
#define LWCC97_CLEAR_REPORT_VALUE_TYPE                                                                        4:0
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_DA_VERTICES_GENERATED                                           0x00000012
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_DA_PRIMITIVES_GENERATED                                         0x00000013
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_VS_ILWOCATIONS                                                  0x00000015
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_TI_ILWOCATIONS                                                  0x00000016
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_TS_ILWOCATIONS                                                  0x00000017
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_TS_PRIMITIVES_GENERATED                                         0x00000018
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_GS_ILWOCATIONS                                                  0x0000001A
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_GS_PRIMITIVES_GENERATED                                         0x0000001B
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_VTG_PRIMITIVES_OUT                                              0x0000001F
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_SUCCEEDED                                  0x00000010
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_STREAMING_PRIMITIVES_NEEDED                                     0x00000011
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_TOTAL_STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED               0x00000003
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_CLIPPER_ILWOCATIONS                                             0x0000001C
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_CLIPPER_PRIMITIVES_GENERATED                                    0x0000001D
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_ZLWLL_STATS                                                     0x00000002
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_PS_ILWOCATIONS                                                  0x0000001E
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_ZPASS_PIXEL_CNT                                                 0x00000001
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_ALPHA_BETA_CLOCKS                                               0x00000004
#define LWCC97_CLEAR_REPORT_VALUE_TYPE_SCG_CLOCKS                                                      0x00000009

#define LWCC97_SET_ANTI_ALIAS_ENABLE                                                                       0x1534
#define LWCC97_SET_ANTI_ALIAS_ENABLE_V                                                                        0:0
#define LWCC97_SET_ANTI_ALIAS_ENABLE_V_FALSE                                                           0x00000000
#define LWCC97_SET_ANTI_ALIAS_ENABLE_V_TRUE                                                            0x00000001

#define LWCC97_SET_ZT_SELECT                                                                               0x1538
#define LWCC97_SET_ZT_SELECT_TARGET_COUNT                                                                     0:0

#define LWCC97_SET_ANTI_ALIAS_ALPHA_CONTROL                                                                0x153c
#define LWCC97_SET_ANTI_ALIAS_ALPHA_CONTROL_ALPHA_TO_COVERAGE                                                 0:0
#define LWCC97_SET_ANTI_ALIAS_ALPHA_CONTROL_ALPHA_TO_COVERAGE_DISABLE                                  0x00000000
#define LWCC97_SET_ANTI_ALIAS_ALPHA_CONTROL_ALPHA_TO_COVERAGE_ENABLE                                   0x00000001
#define LWCC97_SET_ANTI_ALIAS_ALPHA_CONTROL_ALPHA_TO_ONE                                                      4:4
#define LWCC97_SET_ANTI_ALIAS_ALPHA_CONTROL_ALPHA_TO_ONE_DISABLE                                       0x00000000
#define LWCC97_SET_ANTI_ALIAS_ALPHA_CONTROL_ALPHA_TO_ONE_ENABLE                                        0x00000001

#define LWCC97_SET_RENDER_ENABLE_A                                                                         0x1550
#define LWCC97_SET_RENDER_ENABLE_A_OFFSET_UPPER                                                               7:0

#define LWCC97_SET_RENDER_ENABLE_B                                                                         0x1554
#define LWCC97_SET_RENDER_ENABLE_B_OFFSET_LOWER                                                              31:0

#define LWCC97_SET_RENDER_ENABLE_C                                                                         0x1558
#define LWCC97_SET_RENDER_ENABLE_C_MODE                                                                       2:0
#define LWCC97_SET_RENDER_ENABLE_C_MODE_FALSE                                                          0x00000000
#define LWCC97_SET_RENDER_ENABLE_C_MODE_TRUE                                                           0x00000001
#define LWCC97_SET_RENDER_ENABLE_C_MODE_CONDITIONAL                                                    0x00000002
#define LWCC97_SET_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL                                                0x00000003
#define LWCC97_SET_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL                                            0x00000004

#define LWCC97_SET_TEX_SAMPLER_POOL_A                                                                      0x155c
#define LWCC97_SET_TEX_SAMPLER_POOL_A_OFFSET_UPPER                                                            7:0

#define LWCC97_SET_TEX_SAMPLER_POOL_B                                                                      0x1560
#define LWCC97_SET_TEX_SAMPLER_POOL_B_OFFSET_LOWER                                                           31:0

#define LWCC97_SET_TEX_SAMPLER_POOL_C                                                                      0x1564
#define LWCC97_SET_TEX_SAMPLER_POOL_C_MAXIMUM_INDEX                                                          19:0

#define LWCC97_SET_SLOPE_SCALE_DEPTH_BIAS                                                                  0x156c
#define LWCC97_SET_SLOPE_SCALE_DEPTH_BIAS_V                                                                  31:0

#define LWCC97_SET_ANTI_ALIASED_LINE                                                                       0x1570
#define LWCC97_SET_ANTI_ALIASED_LINE_ENABLE                                                                   0:0
#define LWCC97_SET_ANTI_ALIASED_LINE_ENABLE_FALSE                                                      0x00000000
#define LWCC97_SET_ANTI_ALIASED_LINE_ENABLE_TRUE                                                       0x00000001

#define LWCC97_SET_TEX_HEADER_POOL_A                                                                       0x1574
#define LWCC97_SET_TEX_HEADER_POOL_A_OFFSET_UPPER                                                             7:0

#define LWCC97_SET_TEX_HEADER_POOL_B                                                                       0x1578
#define LWCC97_SET_TEX_HEADER_POOL_B_OFFSET_LOWER                                                            31:0

#define LWCC97_SET_TEX_HEADER_POOL_C                                                                       0x157c
#define LWCC97_SET_TEX_HEADER_POOL_C_MAXIMUM_INDEX                                                           21:0

#define LWCC97_SET_ACTIVE_ZLWLL_REGION                                                                     0x1590
#define LWCC97_SET_ACTIVE_ZLWLL_REGION_ID                                                                     5:0

#define LWCC97_SET_TWO_SIDED_STENCIL_TEST                                                                  0x1594
#define LWCC97_SET_TWO_SIDED_STENCIL_TEST_ENABLE                                                              0:0
#define LWCC97_SET_TWO_SIDED_STENCIL_TEST_ENABLE_FALSE                                                 0x00000000
#define LWCC97_SET_TWO_SIDED_STENCIL_TEST_ENABLE_TRUE                                                  0x00000001

#define LWCC97_SET_BACK_STENCIL_OP_FAIL                                                                    0x1598
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V                                                                    31:0
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V_OGL_KEEP                                                     0x00001E00
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V_OGL_ZERO                                                     0x00000000
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V_OGL_REPLACE                                                  0x00001E01
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V_OGL_INCRSAT                                                  0x00001E02
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V_OGL_DECRSAT                                                  0x00001E03
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V_OGL_ILWERT                                                   0x0000150A
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V_OGL_INCR                                                     0x00008507
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V_OGL_DECR                                                     0x00008508
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V_D3D_KEEP                                                     0x00000001
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V_D3D_ZERO                                                     0x00000002
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V_D3D_REPLACE                                                  0x00000003
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V_D3D_INCRSAT                                                  0x00000004
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V_D3D_DECRSAT                                                  0x00000005
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V_D3D_ILWERT                                                   0x00000006
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V_D3D_INCR                                                     0x00000007
#define LWCC97_SET_BACK_STENCIL_OP_FAIL_V_D3D_DECR                                                     0x00000008

#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL                                                                   0x159c
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V                                                                   31:0
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V_OGL_KEEP                                                    0x00001E00
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V_OGL_ZERO                                                    0x00000000
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V_OGL_REPLACE                                                 0x00001E01
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V_OGL_INCRSAT                                                 0x00001E02
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V_OGL_DECRSAT                                                 0x00001E03
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V_OGL_ILWERT                                                  0x0000150A
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V_OGL_INCR                                                    0x00008507
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V_OGL_DECR                                                    0x00008508
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V_D3D_KEEP                                                    0x00000001
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V_D3D_ZERO                                                    0x00000002
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V_D3D_REPLACE                                                 0x00000003
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V_D3D_INCRSAT                                                 0x00000004
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V_D3D_DECRSAT                                                 0x00000005
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V_D3D_ILWERT                                                  0x00000006
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V_D3D_INCR                                                    0x00000007
#define LWCC97_SET_BACK_STENCIL_OP_ZFAIL_V_D3D_DECR                                                    0x00000008

#define LWCC97_SET_BACK_STENCIL_OP_ZPASS                                                                   0x15a0
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V                                                                   31:0
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V_OGL_KEEP                                                    0x00001E00
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V_OGL_ZERO                                                    0x00000000
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V_OGL_REPLACE                                                 0x00001E01
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V_OGL_INCRSAT                                                 0x00001E02
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V_OGL_DECRSAT                                                 0x00001E03
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V_OGL_ILWERT                                                  0x0000150A
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V_OGL_INCR                                                    0x00008507
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V_OGL_DECR                                                    0x00008508
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V_D3D_KEEP                                                    0x00000001
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V_D3D_ZERO                                                    0x00000002
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V_D3D_REPLACE                                                 0x00000003
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V_D3D_INCRSAT                                                 0x00000004
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V_D3D_DECRSAT                                                 0x00000005
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V_D3D_ILWERT                                                  0x00000006
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V_D3D_INCR                                                    0x00000007
#define LWCC97_SET_BACK_STENCIL_OP_ZPASS_V_D3D_DECR                                                    0x00000008

#define LWCC97_SET_BACK_STENCIL_FUNC                                                                       0x15a4
#define LWCC97_SET_BACK_STENCIL_FUNC_V                                                                       31:0
#define LWCC97_SET_BACK_STENCIL_FUNC_V_OGL_NEVER                                                       0x00000200
#define LWCC97_SET_BACK_STENCIL_FUNC_V_OGL_LESS                                                        0x00000201
#define LWCC97_SET_BACK_STENCIL_FUNC_V_OGL_EQUAL                                                       0x00000202
#define LWCC97_SET_BACK_STENCIL_FUNC_V_OGL_LEQUAL                                                      0x00000203
#define LWCC97_SET_BACK_STENCIL_FUNC_V_OGL_GREATER                                                     0x00000204
#define LWCC97_SET_BACK_STENCIL_FUNC_V_OGL_NOTEQUAL                                                    0x00000205
#define LWCC97_SET_BACK_STENCIL_FUNC_V_OGL_GEQUAL                                                      0x00000206
#define LWCC97_SET_BACK_STENCIL_FUNC_V_OGL_ALWAYS                                                      0x00000207
#define LWCC97_SET_BACK_STENCIL_FUNC_V_D3D_NEVER                                                       0x00000001
#define LWCC97_SET_BACK_STENCIL_FUNC_V_D3D_LESS                                                        0x00000002
#define LWCC97_SET_BACK_STENCIL_FUNC_V_D3D_EQUAL                                                       0x00000003
#define LWCC97_SET_BACK_STENCIL_FUNC_V_D3D_LESSEQUAL                                                   0x00000004
#define LWCC97_SET_BACK_STENCIL_FUNC_V_D3D_GREATER                                                     0x00000005
#define LWCC97_SET_BACK_STENCIL_FUNC_V_D3D_NOTEQUAL                                                    0x00000006
#define LWCC97_SET_BACK_STENCIL_FUNC_V_D3D_GREATEREQUAL                                                0x00000007
#define LWCC97_SET_BACK_STENCIL_FUNC_V_D3D_ALWAYS                                                      0x00000008

#define LWCC97_SET_SRGB_WRITE                                                                              0x15b8
#define LWCC97_SET_SRGB_WRITE_ENABLE                                                                          0:0
#define LWCC97_SET_SRGB_WRITE_ENABLE_FALSE                                                             0x00000000
#define LWCC97_SET_SRGB_WRITE_ENABLE_TRUE                                                              0x00000001

#define LWCC97_SET_DEPTH_BIAS                                                                              0x15bc
#define LWCC97_SET_DEPTH_BIAS_V                                                                              31:0

#define LWCC97_SET_ZLWLL_REGION_FORMAT                                                                     0x15c8
#define LWCC97_SET_ZLWLL_REGION_FORMAT_TYPE                                                                   3:0
#define LWCC97_SET_ZLWLL_REGION_FORMAT_TYPE_Z_4X4                                                      0x00000000
#define LWCC97_SET_ZLWLL_REGION_FORMAT_TYPE_ZS_4X4                                                     0x00000001
#define LWCC97_SET_ZLWLL_REGION_FORMAT_TYPE_Z_4X2                                                      0x00000002
#define LWCC97_SET_ZLWLL_REGION_FORMAT_TYPE_Z_2X4                                                      0x00000003
#define LWCC97_SET_ZLWLL_REGION_FORMAT_TYPE_Z_16X8_4X4                                                 0x00000004
#define LWCC97_SET_ZLWLL_REGION_FORMAT_TYPE_Z_8X8_4X2                                                  0x00000005
#define LWCC97_SET_ZLWLL_REGION_FORMAT_TYPE_Z_8X8_2X4                                                  0x00000006
#define LWCC97_SET_ZLWLL_REGION_FORMAT_TYPE_Z_16X16_4X8                                                0x00000007
#define LWCC97_SET_ZLWLL_REGION_FORMAT_TYPE_Z_4X8_2X2                                                  0x00000008
#define LWCC97_SET_ZLWLL_REGION_FORMAT_TYPE_ZS_16X8_4X2                                                0x00000009
#define LWCC97_SET_ZLWLL_REGION_FORMAT_TYPE_ZS_16X8_2X4                                                0x0000000A
#define LWCC97_SET_ZLWLL_REGION_FORMAT_TYPE_ZS_8X8_2X2                                                 0x0000000B
#define LWCC97_SET_ZLWLL_REGION_FORMAT_TYPE_Z_4X8_1X1                                                  0x0000000C

#define LWCC97_SET_RT_LAYER                                                                                0x15cc
#define LWCC97_SET_RT_LAYER_V                                                                                15:0
#define LWCC97_SET_RT_LAYER_CONTROL                                                                         16:16
#define LWCC97_SET_RT_LAYER_CONTROL_V_SELECTS_LAYER                                                    0x00000000
#define LWCC97_SET_RT_LAYER_CONTROL_GEOMETRY_SHADER_SELECTS_LAYER                                      0x00000001

#define LWCC97_SET_ANTI_ALIAS                                                                              0x15d0
#define LWCC97_SET_ANTI_ALIAS_SAMPLES                                                                         3:0
#define LWCC97_SET_ANTI_ALIAS_SAMPLES_MODE_1X1                                                         0x00000000
#define LWCC97_SET_ANTI_ALIAS_SAMPLES_MODE_2X1                                                         0x00000001
#define LWCC97_SET_ANTI_ALIAS_SAMPLES_MODE_2X2                                                         0x00000002
#define LWCC97_SET_ANTI_ALIAS_SAMPLES_MODE_4X2                                                         0x00000003
#define LWCC97_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_D3D                                                     0x00000004
#define LWCC97_SET_ANTI_ALIAS_SAMPLES_MODE_2X1_D3D                                                     0x00000005
#define LWCC97_SET_ANTI_ALIAS_SAMPLES_MODE_4X4                                                         0x00000006
#define LWCC97_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_4                                                    0x00000008
#define LWCC97_SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_12                                                   0x00000009
#define LWCC97_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_VC_8                                                    0x0000000A
#define LWCC97_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_VC_24                                                   0x0000000B

#define LWCC97_SET_EDGE_FLAG                                                                               0x15e4
#define LWCC97_SET_EDGE_FLAG_V                                                                                0:0
#define LWCC97_SET_EDGE_FLAG_V_FALSE                                                                   0x00000000
#define LWCC97_SET_EDGE_FLAG_V_TRUE                                                                    0x00000001

#define LWCC97_DRAW_INLINE_INDEX                                                                           0x15e8
#define LWCC97_DRAW_INLINE_INDEX_V                                                                           31:0

#define LWCC97_SET_INLINE_INDEX2X16_ALIGN                                                                  0x15ec
#define LWCC97_SET_INLINE_INDEX2X16_ALIGN_COUNT                                                              30:0
#define LWCC97_SET_INLINE_INDEX2X16_ALIGN_START_ODD                                                         31:31
#define LWCC97_SET_INLINE_INDEX2X16_ALIGN_START_ODD_FALSE                                              0x00000000
#define LWCC97_SET_INLINE_INDEX2X16_ALIGN_START_ODD_TRUE                                               0x00000001

#define LWCC97_DRAW_INLINE_INDEX2X16                                                                       0x15f0
#define LWCC97_DRAW_INLINE_INDEX2X16_EVEN                                                                    15:0
#define LWCC97_DRAW_INLINE_INDEX2X16_ODD                                                                    31:16

#define LWCC97_SET_VERTEX_GLOBAL_BASE_OFFSET_A                                                             0x15f4
#define LWCC97_SET_VERTEX_GLOBAL_BASE_OFFSET_A_UPPER                                                          7:0

#define LWCC97_SET_VERTEX_GLOBAL_BASE_OFFSET_B                                                             0x15f8
#define LWCC97_SET_VERTEX_GLOBAL_BASE_OFFSET_B_LOWER                                                         31:0

#define LWCC97_SET_ZLWLL_REGION_PIXEL_OFFSET_A                                                             0x15fc
#define LWCC97_SET_ZLWLL_REGION_PIXEL_OFFSET_A_WIDTH                                                         15:0

#define LWCC97_SET_ZLWLL_REGION_PIXEL_OFFSET_B                                                             0x1600
#define LWCC97_SET_ZLWLL_REGION_PIXEL_OFFSET_B_HEIGHT                                                        15:0

#define LWCC97_SET_POINT_SPRITE_SELECT                                                                     0x1604
#define LWCC97_SET_POINT_SPRITE_SELECT_RMODE                                                                  1:0
#define LWCC97_SET_POINT_SPRITE_SELECT_RMODE_ZERO                                                      0x00000000
#define LWCC97_SET_POINT_SPRITE_SELECT_RMODE_FROM_R                                                    0x00000001
#define LWCC97_SET_POINT_SPRITE_SELECT_RMODE_FROM_S                                                    0x00000002
#define LWCC97_SET_POINT_SPRITE_SELECT_ORIGIN                                                                 2:2
#define LWCC97_SET_POINT_SPRITE_SELECT_ORIGIN_BOTTOM                                                   0x00000000
#define LWCC97_SET_POINT_SPRITE_SELECT_ORIGIN_TOP                                                      0x00000001
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE0                                                               3:3
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE0_PASSTHROUGH                                            0x00000000
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE0_GENERATE                                               0x00000001
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE1                                                               4:4
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE1_PASSTHROUGH                                            0x00000000
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE1_GENERATE                                               0x00000001
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE2                                                               5:5
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE2_PASSTHROUGH                                            0x00000000
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE2_GENERATE                                               0x00000001
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE3                                                               6:6
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE3_PASSTHROUGH                                            0x00000000
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE3_GENERATE                                               0x00000001
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE4                                                               7:7
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE4_PASSTHROUGH                                            0x00000000
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE4_GENERATE                                               0x00000001
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE5                                                               8:8
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE5_PASSTHROUGH                                            0x00000000
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE5_GENERATE                                               0x00000001
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE6                                                               9:9
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE6_PASSTHROUGH                                            0x00000000
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE6_GENERATE                                               0x00000001
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE7                                                             10:10
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE7_PASSTHROUGH                                            0x00000000
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE7_GENERATE                                               0x00000001
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE8                                                             11:11
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE8_PASSTHROUGH                                            0x00000000
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE8_GENERATE                                               0x00000001
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE9                                                             12:12
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE9_PASSTHROUGH                                            0x00000000
#define LWCC97_SET_POINT_SPRITE_SELECT_TEXTURE9_GENERATE                                               0x00000001

#define LWCC97_SET_ATTRIBUTE_DEFAULT                                                                       0x1610
#define LWCC97_SET_ATTRIBUTE_DEFAULT_COLOR_FRONT_DIFFUSE                                                      0:0
#define LWCC97_SET_ATTRIBUTE_DEFAULT_COLOR_FRONT_DIFFUSE_VECTOR_0001                                   0x00000000
#define LWCC97_SET_ATTRIBUTE_DEFAULT_COLOR_FRONT_DIFFUSE_VECTOR_1111                                   0x00000001
#define LWCC97_SET_ATTRIBUTE_DEFAULT_COLOR_FRONT_SPELWLAR                                                     1:1
#define LWCC97_SET_ATTRIBUTE_DEFAULT_COLOR_FRONT_SPELWLAR_VECTOR_0000                                  0x00000000
#define LWCC97_SET_ATTRIBUTE_DEFAULT_COLOR_FRONT_SPELWLAR_VECTOR_0001                                  0x00000001
#define LWCC97_SET_ATTRIBUTE_DEFAULT_GENERIC_VECTOR                                                           2:2
#define LWCC97_SET_ATTRIBUTE_DEFAULT_GENERIC_VECTOR_VECTOR_0000                                        0x00000000
#define LWCC97_SET_ATTRIBUTE_DEFAULT_GENERIC_VECTOR_VECTOR_0001                                        0x00000001
#define LWCC97_SET_ATTRIBUTE_DEFAULT_FIXED_FNC_TEXTURE                                                        3:3
#define LWCC97_SET_ATTRIBUTE_DEFAULT_FIXED_FNC_TEXTURE_VECTOR_0000                                     0x00000000
#define LWCC97_SET_ATTRIBUTE_DEFAULT_FIXED_FNC_TEXTURE_VECTOR_0001                                     0x00000001
#define LWCC97_SET_ATTRIBUTE_DEFAULT_DX9_COLOR0                                                               4:4
#define LWCC97_SET_ATTRIBUTE_DEFAULT_DX9_COLOR0_VECTOR_0001                                            0x00000000
#define LWCC97_SET_ATTRIBUTE_DEFAULT_DX9_COLOR0_VECTOR_1111                                            0x00000001
#define LWCC97_SET_ATTRIBUTE_DEFAULT_DX9_COLOR1_TO_COLOR15                                                    5:5
#define LWCC97_SET_ATTRIBUTE_DEFAULT_DX9_COLOR1_TO_COLOR15_VECTOR_0000                                 0x00000000
#define LWCC97_SET_ATTRIBUTE_DEFAULT_DX9_COLOR1_TO_COLOR15_VECTOR_0001                                 0x00000001

#define LWCC97_END                                                                                         0x1614
#define LWCC97_END_V                                                                                          0:0

#define LWCC97_BEGIN                                                                                       0x1618
#define LWCC97_BEGIN_OP                                                                                      15:0
#define LWCC97_BEGIN_OP_POINTS                                                                         0x00000000
#define LWCC97_BEGIN_OP_LINES                                                                          0x00000001
#define LWCC97_BEGIN_OP_LINE_LOOP                                                                      0x00000002
#define LWCC97_BEGIN_OP_LINE_STRIP                                                                     0x00000003
#define LWCC97_BEGIN_OP_TRIANGLES                                                                      0x00000004
#define LWCC97_BEGIN_OP_TRIANGLE_STRIP                                                                 0x00000005
#define LWCC97_BEGIN_OP_TRIANGLE_FAN                                                                   0x00000006
#define LWCC97_BEGIN_OP_QUADS                                                                          0x00000007
#define LWCC97_BEGIN_OP_QUAD_STRIP                                                                     0x00000008
#define LWCC97_BEGIN_OP_POLYGON                                                                        0x00000009
#define LWCC97_BEGIN_OP_LINELIST_ADJCY                                                                 0x0000000A
#define LWCC97_BEGIN_OP_LINESTRIP_ADJCY                                                                0x0000000B
#define LWCC97_BEGIN_OP_TRIANGLELIST_ADJCY                                                             0x0000000C
#define LWCC97_BEGIN_OP_TRIANGLESTRIP_ADJCY                                                            0x0000000D
#define LWCC97_BEGIN_OP_PATCH                                                                          0x0000000E
#define LWCC97_BEGIN_PRIMITIVE_ID                                                                           24:24
#define LWCC97_BEGIN_PRIMITIVE_ID_FIRST                                                                0x00000000
#define LWCC97_BEGIN_PRIMITIVE_ID_UNCHANGED                                                            0x00000001
#define LWCC97_BEGIN_INSTANCE_ID                                                                            27:26
#define LWCC97_BEGIN_INSTANCE_ID_FIRST                                                                 0x00000000
#define LWCC97_BEGIN_INSTANCE_ID_SUBSEQUENT                                                            0x00000001
#define LWCC97_BEGIN_INSTANCE_ID_UNCHANGED                                                             0x00000002
#define LWCC97_BEGIN_SPLIT_MODE                                                                             30:29
#define LWCC97_BEGIN_SPLIT_MODE_NORMAL_BEGIN_NORMAL_END                                                0x00000000
#define LWCC97_BEGIN_SPLIT_MODE_NORMAL_BEGIN_OPEN_END                                                  0x00000001
#define LWCC97_BEGIN_SPLIT_MODE_OPEN_BEGIN_OPEN_END                                                    0x00000002
#define LWCC97_BEGIN_SPLIT_MODE_OPEN_BEGIN_NORMAL_END                                                  0x00000003
#define LWCC97_BEGIN_INSTANCE_ITERATE_ENABLE                                                                31:31
#define LWCC97_BEGIN_INSTANCE_ITERATE_ENABLE_FALSE                                                     0x00000000
#define LWCC97_BEGIN_INSTANCE_ITERATE_ENABLE_TRUE                                                      0x00000001

#define LWCC97_SET_VERTEX_ID_COPY                                                                          0x161c
#define LWCC97_SET_VERTEX_ID_COPY_ENABLE                                                                      0:0
#define LWCC97_SET_VERTEX_ID_COPY_ENABLE_FALSE                                                         0x00000000
#define LWCC97_SET_VERTEX_ID_COPY_ENABLE_TRUE                                                          0x00000001
#define LWCC97_SET_VERTEX_ID_COPY_ATTRIBUTE_SLOT                                                             11:4

#define LWCC97_ADD_TO_PRIMITIVE_ID                                                                         0x1620
#define LWCC97_ADD_TO_PRIMITIVE_ID_V                                                                         31:0

#define LWCC97_LOAD_PRIMITIVE_ID                                                                           0x1624
#define LWCC97_LOAD_PRIMITIVE_ID_V                                                                           31:0

#define LWCC97_SET_SHADER_BASED_LWLL                                                                       0x162c
#define LWCC97_SET_SHADER_BASED_LWLL_BATCH_LWLL_ENABLE                                                        1:1
#define LWCC97_SET_SHADER_BASED_LWLL_BATCH_LWLL_ENABLE_FALSE                                           0x00000000
#define LWCC97_SET_SHADER_BASED_LWLL_BATCH_LWLL_ENABLE_TRUE                                            0x00000001
#define LWCC97_SET_SHADER_BASED_LWLL_BEFORE_FETCH_ENABLE                                                      0:0
#define LWCC97_SET_SHADER_BASED_LWLL_BEFORE_FETCH_ENABLE_FALSE                                         0x00000000
#define LWCC97_SET_SHADER_BASED_LWLL_BEFORE_FETCH_ENABLE_TRUE                                          0x00000001

#define LWCC97_SET_CLASS_VERSION                                                                           0x1638
#define LWCC97_SET_CLASS_VERSION_LWRRENT                                                                     15:0
#define LWCC97_SET_CLASS_VERSION_OLDEST_SUPPORTED                                                           31:16

#define LWCC97_SET_DA_PRIMITIVE_RESTART                                                                    0x1644
#define LWCC97_SET_DA_PRIMITIVE_RESTART_ENABLE                                                                0:0
#define LWCC97_SET_DA_PRIMITIVE_RESTART_ENABLE_FALSE                                                   0x00000000
#define LWCC97_SET_DA_PRIMITIVE_RESTART_ENABLE_TRUE                                                    0x00000001

#define LWCC97_SET_DA_PRIMITIVE_RESTART_INDEX                                                              0x1648
#define LWCC97_SET_DA_PRIMITIVE_RESTART_INDEX_V                                                              31:0

#define LWCC97_SET_DA_OUTPUT                                                                               0x164c
#define LWCC97_SET_DA_OUTPUT_VERTEX_ID_USES_ARRAY_START                                                     12:12
#define LWCC97_SET_DA_OUTPUT_VERTEX_ID_USES_ARRAY_START_FALSE                                          0x00000000
#define LWCC97_SET_DA_OUTPUT_VERTEX_ID_USES_ARRAY_START_TRUE                                           0x00000001

#define LWCC97_SET_ANTI_ALIASED_POINT                                                                      0x1658
#define LWCC97_SET_ANTI_ALIASED_POINT_ENABLE                                                                  0:0
#define LWCC97_SET_ANTI_ALIASED_POINT_ENABLE_FALSE                                                     0x00000000
#define LWCC97_SET_ANTI_ALIASED_POINT_ENABLE_TRUE                                                      0x00000001

#define LWCC97_SET_POINT_CENTER_MODE                                                                       0x165c
#define LWCC97_SET_POINT_CENTER_MODE_V                                                                       31:0
#define LWCC97_SET_POINT_CENTER_MODE_V_OGL                                                             0x00000000
#define LWCC97_SET_POINT_CENTER_MODE_V_D3D                                                             0x00000001

#define LWCC97_SET_LINE_SMOOTH_PARAMETERS                                                                  0x1668
#define LWCC97_SET_LINE_SMOOTH_PARAMETERS_FALLOFF                                                            31:0
#define LWCC97_SET_LINE_SMOOTH_PARAMETERS_FALLOFF__1_00                                                0x00000000
#define LWCC97_SET_LINE_SMOOTH_PARAMETERS_FALLOFF__1_33                                                0x00000001
#define LWCC97_SET_LINE_SMOOTH_PARAMETERS_FALLOFF__1_60                                                0x00000002

#define LWCC97_SET_LINE_STIPPLE                                                                            0x166c
#define LWCC97_SET_LINE_STIPPLE_ENABLE                                                                        0:0
#define LWCC97_SET_LINE_STIPPLE_ENABLE_FALSE                                                           0x00000000
#define LWCC97_SET_LINE_STIPPLE_ENABLE_TRUE                                                            0x00000001

#define LWCC97_SET_LINE_SMOOTH_EDGE_TABLE(i)                                                       (0x1670+(i)*4)
#define LWCC97_SET_LINE_SMOOTH_EDGE_TABLE_V0                                                                  7:0
#define LWCC97_SET_LINE_SMOOTH_EDGE_TABLE_V1                                                                 15:8
#define LWCC97_SET_LINE_SMOOTH_EDGE_TABLE_V2                                                                23:16
#define LWCC97_SET_LINE_SMOOTH_EDGE_TABLE_V3                                                                31:24

#define LWCC97_SET_LINE_STIPPLE_PARAMETERS                                                                 0x1680
#define LWCC97_SET_LINE_STIPPLE_PARAMETERS_FACTOR                                                             7:0
#define LWCC97_SET_LINE_STIPPLE_PARAMETERS_PATTERN                                                           23:8

#define LWCC97_SET_PROVOKING_VERTEX                                                                        0x1684
#define LWCC97_SET_PROVOKING_VERTEX_V                                                                         0:0
#define LWCC97_SET_PROVOKING_VERTEX_V_FIRST                                                            0x00000000
#define LWCC97_SET_PROVOKING_VERTEX_V_LAST                                                             0x00000001

#define LWCC97_SET_TWO_SIDED_LIGHT                                                                         0x1688
#define LWCC97_SET_TWO_SIDED_LIGHT_ENABLE                                                                     0:0
#define LWCC97_SET_TWO_SIDED_LIGHT_ENABLE_FALSE                                                        0x00000000
#define LWCC97_SET_TWO_SIDED_LIGHT_ENABLE_TRUE                                                         0x00000001

#define LWCC97_SET_POLYGON_STIPPLE                                                                         0x168c
#define LWCC97_SET_POLYGON_STIPPLE_ENABLE                                                                     0:0
#define LWCC97_SET_POLYGON_STIPPLE_ENABLE_FALSE                                                        0x00000000
#define LWCC97_SET_POLYGON_STIPPLE_ENABLE_TRUE                                                         0x00000001

#define LWCC97_SET_SHADER_CONTROL                                                                          0x1690
#define LWCC97_SET_SHADER_CONTROL_DEFAULT_PARTIAL                                                             0:0
#define LWCC97_SET_SHADER_CONTROL_DEFAULT_PARTIAL_ZERO                                                 0x00000000
#define LWCC97_SET_SHADER_CONTROL_DEFAULT_PARTIAL_INFINITY                                             0x00000001
#define LWCC97_SET_SHADER_CONTROL_FP32_NAN_BEHAVIOR                                                           1:1
#define LWCC97_SET_SHADER_CONTROL_FP32_NAN_BEHAVIOR_LEGACY                                             0x00000000
#define LWCC97_SET_SHADER_CONTROL_FP32_NAN_BEHAVIOR_FP64_COMPATIBLE                                    0x00000001
#define LWCC97_SET_SHADER_CONTROL_FP32_F2I_NAN_BEHAVIOR                                                       2:2
#define LWCC97_SET_SHADER_CONTROL_FP32_F2I_NAN_BEHAVIOR_PASS_ZERO                                      0x00000000
#define LWCC97_SET_SHADER_CONTROL_FP32_F2I_NAN_BEHAVIOR_PASS_INDEFINITE                                0x00000001

#define LWCC97_CHECK_CLASS_VERSION                                                                         0x16a0
#define LWCC97_CHECK_CLASS_VERSION_LWRRENT                                                                   15:0
#define LWCC97_CHECK_CLASS_VERSION_OLDEST_SUPPORTED                                                         31:16

#define LWCC97_SET_SPH_VERSION                                                                             0x16a4
#define LWCC97_SET_SPH_VERSION_LWRRENT                                                                       15:0
#define LWCC97_SET_SPH_VERSION_OLDEST_SUPPORTED                                                             31:16

#define LWCC97_CHECK_SPH_VERSION                                                                           0x16a8
#define LWCC97_CHECK_SPH_VERSION_LWRRENT                                                                     15:0
#define LWCC97_CHECK_SPH_VERSION_OLDEST_SUPPORTED                                                           31:16

#define LWCC97_SET_ALPHA_TO_COVERAGE_OVERRIDE                                                              0x16b4
#define LWCC97_SET_ALPHA_TO_COVERAGE_OVERRIDE_QUALIFY_BY_ANTI_ALIAS_ENABLE                                    0:0
#define LWCC97_SET_ALPHA_TO_COVERAGE_OVERRIDE_QUALIFY_BY_ANTI_ALIAS_ENABLE_DISABLE                     0x00000000
#define LWCC97_SET_ALPHA_TO_COVERAGE_OVERRIDE_QUALIFY_BY_ANTI_ALIAS_ENABLE_ENABLE                      0x00000001
#define LWCC97_SET_ALPHA_TO_COVERAGE_OVERRIDE_QUALIFY_BY_PS_SAMPLE_MASK_OUTPUT                                1:1
#define LWCC97_SET_ALPHA_TO_COVERAGE_OVERRIDE_QUALIFY_BY_PS_SAMPLE_MASK_OUTPUT_DISABLE                 0x00000000
#define LWCC97_SET_ALPHA_TO_COVERAGE_OVERRIDE_QUALIFY_BY_PS_SAMPLE_MASK_OUTPUT_ENABLE                  0x00000001

#define LWCC97_SET_SCG_GRAPHICS_PRIORITY                                                                   0x16bc
#define LWCC97_SET_SCG_GRAPHICS_PRIORITY_PRIORITY                                                             5:0

#define LWCC97_SET_SCG_GRAPHICS_SCHEDULING_PARAMETERS(i)                                           (0x16c0+(i)*4)
#define LWCC97_SET_SCG_GRAPHICS_SCHEDULING_PARAMETERS_V                                                      31:0

#define LWCC97_SET_POLYGON_STIPPLE_PATTERN(i)                                                      (0x1700+(i)*4)
#define LWCC97_SET_POLYGON_STIPPLE_PATTERN_V                                                                 31:0

#define LWCC97_SET_AAM_VERSION                                                                             0x1790
#define LWCC97_SET_AAM_VERSION_LWRRENT                                                                       15:0
#define LWCC97_SET_AAM_VERSION_OLDEST_SUPPORTED                                                             31:16

#define LWCC97_CHECK_AAM_VERSION                                                                           0x1794
#define LWCC97_CHECK_AAM_VERSION_LWRRENT                                                                     15:0
#define LWCC97_CHECK_AAM_VERSION_OLDEST_SUPPORTED                                                           31:16

#define LWCC97_SET_ZT_LAYER                                                                                0x179c
#define LWCC97_SET_ZT_LAYER_OFFSET                                                                           15:0

#define LWCC97_SET_INDEX_BUFFER_A                                                                          0x17c8
#define LWCC97_SET_INDEX_BUFFER_A_ADDRESS_UPPER                                                               7:0

#define LWCC97_SET_INDEX_BUFFER_B                                                                          0x17cc
#define LWCC97_SET_INDEX_BUFFER_B_ADDRESS_LOWER                                                              31:0

#define LWCC97_SET_INDEX_BUFFER_E                                                                          0x17d8
#define LWCC97_SET_INDEX_BUFFER_E_INDEX_SIZE                                                                  1:0
#define LWCC97_SET_INDEX_BUFFER_E_INDEX_SIZE_ONE_BYTE                                                  0x00000000
#define LWCC97_SET_INDEX_BUFFER_E_INDEX_SIZE_TWO_BYTES                                                 0x00000001
#define LWCC97_SET_INDEX_BUFFER_E_INDEX_SIZE_FOUR_BYTES                                                0x00000002

#define LWCC97_SET_INDEX_BUFFER_F                                                                          0x17dc
#define LWCC97_SET_INDEX_BUFFER_F_FIRST                                                                      31:0

#define LWCC97_DRAW_INDEX_BUFFER                                                                           0x17e0
#define LWCC97_DRAW_INDEX_BUFFER_COUNT                                                                       31:0

#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST                                                0x17e4
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_FIRST                                            15:0
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_COUNT                                           27:16
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY                                        31:28
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_POINTS                            0x00000000
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINES                             0x00000001
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINE_LOOP                         0x00000002
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINE_STRIP                        0x00000003
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLES                         0x00000004
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLE_STRIP                    0x00000005
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLE_FAN                      0x00000006
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_QUADS                             0x00000007
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_QUAD_STRIP                        0x00000008
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_POLYGON                           0x00000009
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINELIST_ADJCY                    0x0000000A
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINESTRIP_ADJCY                   0x0000000B
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLELIST_ADJCY                0x0000000C
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLESTRIP_ADJCY               0x0000000D
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_PATCH                             0x0000000E

#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST                                                0x17e8
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_FIRST                                            15:0
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_COUNT                                           27:16
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY                                        31:28
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_POINTS                            0x00000000
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINES                             0x00000001
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINE_LOOP                         0x00000002
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINE_STRIP                        0x00000003
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLES                         0x00000004
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLE_STRIP                    0x00000005
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLE_FAN                      0x00000006
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_QUADS                             0x00000007
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_QUAD_STRIP                        0x00000008
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_POLYGON                           0x00000009
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINELIST_ADJCY                    0x0000000A
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINESTRIP_ADJCY                   0x0000000B
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLELIST_ADJCY                0x0000000C
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLESTRIP_ADJCY               0x0000000D
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_PATCH                             0x0000000E

#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST                                                 0x17ec
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_FIRST                                             15:0
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_COUNT                                            27:16
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_TOPOLOGY                                         31:28
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_POINTS                             0x00000000
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINES                              0x00000001
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINE_LOOP                          0x00000002
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINE_STRIP                         0x00000003
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLES                          0x00000004
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLE_STRIP                     0x00000005
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLE_FAN                       0x00000006
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_QUADS                              0x00000007
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_QUAD_STRIP                         0x00000008
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_POLYGON                            0x00000009
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINELIST_ADJCY                     0x0000000A
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_LINESTRIP_ADJCY                    0x0000000B
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLELIST_ADJCY                 0x0000000C
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_TRIANGLESTRIP_ADJCY                0x0000000D
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST_TOPOLOGY_PATCH                              0x0000000E

#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT                                           0x17f0
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_FIRST                                       15:0
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_COUNT                                      27:16
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY                                   31:28
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_POINTS                       0x00000000
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINES                        0x00000001
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINE_LOOP                    0x00000002
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINE_STRIP                   0x00000003
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLES                    0x00000004
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLE_STRIP               0x00000005
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLE_FAN                 0x00000006
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_QUADS                        0x00000007
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_QUAD_STRIP                   0x00000008
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_POLYGON                      0x00000009
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINELIST_ADJCY               0x0000000A
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINESTRIP_ADJCY              0x0000000B
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLELIST_ADJCY             0x0000000C
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLESTRIP_ADJCY             0x0000000D
#define LWCC97_DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_PATCH                        0x0000000E

#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT                                           0x17f4
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_FIRST                                       15:0
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_COUNT                                      27:16
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY                                   31:28
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_POINTS                       0x00000000
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINES                        0x00000001
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINE_LOOP                    0x00000002
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINE_STRIP                   0x00000003
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLES                    0x00000004
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLE_STRIP               0x00000005
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLE_FAN                 0x00000006
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_QUADS                        0x00000007
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_QUAD_STRIP                   0x00000008
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_POLYGON                      0x00000009
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINELIST_ADJCY               0x0000000A
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINESTRIP_ADJCY              0x0000000B
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLELIST_ADJCY             0x0000000C
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLESTRIP_ADJCY             0x0000000D
#define LWCC97_DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_PATCH                        0x0000000E

#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT                                            0x17f8
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_FIRST                                        15:0
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_COUNT                                       27:16
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY                                    31:28
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_POINTS                        0x00000000
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINES                         0x00000001
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINE_LOOP                     0x00000002
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINE_STRIP                    0x00000003
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLES                     0x00000004
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLE_STRIP                0x00000005
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLE_FAN                  0x00000006
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_QUADS                         0x00000007
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_QUAD_STRIP                    0x00000008
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_POLYGON                       0x00000009
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINELIST_ADJCY                0x0000000A
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_LINESTRIP_ADJCY               0x0000000B
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLELIST_ADJCY             0x0000000C
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_TRIANGLESTRIP_ADJCY             0x0000000D
#define LWCC97_DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT_TOPOLOGY_PATCH                         0x0000000E

#define LWCC97_SET_DEPTH_BIAS_CLAMP                                                                        0x187c
#define LWCC97_SET_DEPTH_BIAS_CLAMP_V                                                                        31:0

#define LWCC97_SET_VERTEX_STREAM_INSTANCE_A(i)                                                     (0x1880+(i)*4)
#define LWCC97_SET_VERTEX_STREAM_INSTANCE_A_IS_INSTANCED                                                      0:0
#define LWCC97_SET_VERTEX_STREAM_INSTANCE_A_IS_INSTANCED_FALSE                                         0x00000000
#define LWCC97_SET_VERTEX_STREAM_INSTANCE_A_IS_INSTANCED_TRUE                                          0x00000001

#define LWCC97_SET_VERTEX_STREAM_INSTANCE_B(i)                                                     (0x18c0+(i)*4)
#define LWCC97_SET_VERTEX_STREAM_INSTANCE_B_IS_INSTANCED                                                      0:0
#define LWCC97_SET_VERTEX_STREAM_INSTANCE_B_IS_INSTANCED_FALSE                                         0x00000000
#define LWCC97_SET_VERTEX_STREAM_INSTANCE_B_IS_INSTANCED_TRUE                                          0x00000001

#define LWCC97_SET_ATTRIBUTE_POINT_SIZE                                                                    0x1910
#define LWCC97_SET_ATTRIBUTE_POINT_SIZE_ENABLE                                                                0:0
#define LWCC97_SET_ATTRIBUTE_POINT_SIZE_ENABLE_FALSE                                                   0x00000000
#define LWCC97_SET_ATTRIBUTE_POINT_SIZE_ENABLE_TRUE                                                    0x00000001
#define LWCC97_SET_ATTRIBUTE_POINT_SIZE_SLOT                                                                 11:4

#define LWCC97_OGL_SET_LWLL                                                                                0x1918
#define LWCC97_OGL_SET_LWLL_ENABLE                                                                            0:0
#define LWCC97_OGL_SET_LWLL_ENABLE_FALSE                                                               0x00000000
#define LWCC97_OGL_SET_LWLL_ENABLE_TRUE                                                                0x00000001

#define LWCC97_OGL_SET_FRONT_FACE                                                                          0x191c
#define LWCC97_OGL_SET_FRONT_FACE_V                                                                          31:0
#define LWCC97_OGL_SET_FRONT_FACE_V_CW                                                                 0x00000900
#define LWCC97_OGL_SET_FRONT_FACE_V_CCW                                                                0x00000901

#define LWCC97_OGL_SET_LWLL_FACE                                                                           0x1920
#define LWCC97_OGL_SET_LWLL_FACE_V                                                                           31:0
#define LWCC97_OGL_SET_LWLL_FACE_V_FRONT                                                               0x00000404
#define LWCC97_OGL_SET_LWLL_FACE_V_BACK                                                                0x00000405
#define LWCC97_OGL_SET_LWLL_FACE_V_FRONT_AND_BACK                                                      0x00000408

#define LWCC97_SET_VIEWPORT_PIXEL                                                                          0x1924
#define LWCC97_SET_VIEWPORT_PIXEL_CENTER                                                                      0:0
#define LWCC97_SET_VIEWPORT_PIXEL_CENTER_AT_HALF_INTEGERS                                              0x00000000
#define LWCC97_SET_VIEWPORT_PIXEL_CENTER_AT_INTEGERS                                                   0x00000001

#define LWCC97_SET_VIEWPORT_SCALE_OFFSET                                                                   0x192c
#define LWCC97_SET_VIEWPORT_SCALE_OFFSET_ENABLE                                                               0:0
#define LWCC97_SET_VIEWPORT_SCALE_OFFSET_ENABLE_FALSE                                                  0x00000000
#define LWCC97_SET_VIEWPORT_SCALE_OFFSET_ENABLE_TRUE                                                   0x00000001

#define LWCC97_SET_VIEWPORT_CLIP_CONTROL                                                                   0x193c
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_MIN_Z_ZERO_MAX_Z_ONE                                                 0:0
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_MIN_Z_ZERO_MAX_Z_ONE_FALSE                                    0x00000000
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_MIN_Z_ZERO_MAX_Z_ONE_TRUE                                     0x00000001
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_Z_CLIP_RANGE                                                       17:16
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_Z_CLIP_RANGE_USE_FIELD_MIN_Z_ZERO_MAX_Z_ONE                   0x00000000
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_Z_CLIP_RANGE_MIN_Z_MAX_Z                                      0x00000001
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_Z_CLIP_RANGE_ZERO_ONE                                         0x00000002
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_Z_CLIP_RANGE_MINUS_INF_PLUS_INF                               0x00000003
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_PIXEL_MIN_Z                                                          3:3
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_PIXEL_MIN_Z_CLIP                                              0x00000000
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_PIXEL_MIN_Z_CLAMP                                             0x00000001
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_PIXEL_MAX_Z                                                          4:4
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_PIXEL_MAX_Z_CLIP                                              0x00000000
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_PIXEL_MAX_Z_CLAMP                                             0x00000001
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_GUARDBAND                                                   7:7
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_GUARDBAND_SCALE_256                                  0x00000000
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_GUARDBAND_SCALE_1                                    0x00000001
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_LINE_POINT_LWLL_GUARDBAND                                          10:10
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_LINE_POINT_LWLL_GUARDBAND_SCALE_256                           0x00000000
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_LINE_POINT_LWLL_GUARDBAND_SCALE_1                             0x00000001
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP                                                      13:11
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP                                      0x00000000
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_PASSTHRU                                        0x00000001
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XY_CLIP                                 0x00000002
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_XYZ_CLIP                                0x00000003
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_CLIP_NO_Z_LWLL                            0x00000004
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_FRUSTUM_Z_CLIP                                  0x00000005
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP_WZERO_TRI_FILL_OR_CLIP                          0x00000006
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_GUARDBAND_Z                                                 2:1
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_GUARDBAND_Z_SAME_AS_XY_GUARDBAND                     0x00000000
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_GUARDBAND_Z_SCALE_256                                0x00000001
#define LWCC97_SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_GUARDBAND_Z_SCALE_1                                  0x00000002

#define LWCC97_SET_USER_CLIP_OP                                                                            0x1940
#define LWCC97_SET_USER_CLIP_OP_PLANE0                                                                        0:0
#define LWCC97_SET_USER_CLIP_OP_PLANE0_CLIP                                                            0x00000000
#define LWCC97_SET_USER_CLIP_OP_PLANE0_LWLL                                                            0x00000001
#define LWCC97_SET_USER_CLIP_OP_PLANE1                                                                        4:4
#define LWCC97_SET_USER_CLIP_OP_PLANE1_CLIP                                                            0x00000000
#define LWCC97_SET_USER_CLIP_OP_PLANE1_LWLL                                                            0x00000001
#define LWCC97_SET_USER_CLIP_OP_PLANE2                                                                        8:8
#define LWCC97_SET_USER_CLIP_OP_PLANE2_CLIP                                                            0x00000000
#define LWCC97_SET_USER_CLIP_OP_PLANE2_LWLL                                                            0x00000001
#define LWCC97_SET_USER_CLIP_OP_PLANE3                                                                      12:12
#define LWCC97_SET_USER_CLIP_OP_PLANE3_CLIP                                                            0x00000000
#define LWCC97_SET_USER_CLIP_OP_PLANE3_LWLL                                                            0x00000001
#define LWCC97_SET_USER_CLIP_OP_PLANE4                                                                      16:16
#define LWCC97_SET_USER_CLIP_OP_PLANE4_CLIP                                                            0x00000000
#define LWCC97_SET_USER_CLIP_OP_PLANE4_LWLL                                                            0x00000001
#define LWCC97_SET_USER_CLIP_OP_PLANE5                                                                      20:20
#define LWCC97_SET_USER_CLIP_OP_PLANE5_CLIP                                                            0x00000000
#define LWCC97_SET_USER_CLIP_OP_PLANE5_LWLL                                                            0x00000001
#define LWCC97_SET_USER_CLIP_OP_PLANE6                                                                      24:24
#define LWCC97_SET_USER_CLIP_OP_PLANE6_CLIP                                                            0x00000000
#define LWCC97_SET_USER_CLIP_OP_PLANE6_LWLL                                                            0x00000001
#define LWCC97_SET_USER_CLIP_OP_PLANE7                                                                      28:28
#define LWCC97_SET_USER_CLIP_OP_PLANE7_CLIP                                                            0x00000000
#define LWCC97_SET_USER_CLIP_OP_PLANE7_LWLL                                                            0x00000001

#define LWCC97_SET_RENDER_ENABLE_OVERRIDE                                                                  0x1944
#define LWCC97_SET_RENDER_ENABLE_OVERRIDE_MODE                                                                1:0
#define LWCC97_SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE                                       0x00000000
#define LWCC97_SET_RENDER_ENABLE_OVERRIDE_MODE_ALWAYS_RENDER                                           0x00000001
#define LWCC97_SET_RENDER_ENABLE_OVERRIDE_MODE_NEVER_RENDER                                            0x00000002

#define LWCC97_SET_PRIMITIVE_TOPOLOGY_CONTROL                                                              0x1948
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_CONTROL_OVERRIDE                                                        0:0
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_CONTROL_OVERRIDE_USE_TOPOLOGY_IN_BEGIN_METHODS                   0x00000000
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_CONTROL_OVERRIDE_USE_SEPARATE_TOPOLOGY_STATE                     0x00000001

#define LWCC97_SET_WINDOW_CLIP_ENABLE                                                                      0x194c
#define LWCC97_SET_WINDOW_CLIP_ENABLE_V                                                                       0:0
#define LWCC97_SET_WINDOW_CLIP_ENABLE_V_FALSE                                                          0x00000000
#define LWCC97_SET_WINDOW_CLIP_ENABLE_V_TRUE                                                           0x00000001

#define LWCC97_SET_WINDOW_CLIP_TYPE                                                                        0x1950
#define LWCC97_SET_WINDOW_CLIP_TYPE_V                                                                         1:0
#define LWCC97_SET_WINDOW_CLIP_TYPE_V_INCLUSIVE                                                        0x00000000
#define LWCC97_SET_WINDOW_CLIP_TYPE_V_EXCLUSIVE                                                        0x00000001
#define LWCC97_SET_WINDOW_CLIP_TYPE_V_CLIPALL                                                          0x00000002

#define LWCC97_ILWALIDATE_ZLWLL                                                                            0x1958
#define LWCC97_ILWALIDATE_ZLWLL_V                                                                            31:0
#define LWCC97_ILWALIDATE_ZLWLL_V_ILWALIDATE                                                           0x00000000

#define LWCC97_SET_ZLWLL                                                                                   0x1968
#define LWCC97_SET_ZLWLL_Z_ENABLE                                                                             0:0
#define LWCC97_SET_ZLWLL_Z_ENABLE_FALSE                                                                0x00000000
#define LWCC97_SET_ZLWLL_Z_ENABLE_TRUE                                                                 0x00000001
#define LWCC97_SET_ZLWLL_STENCIL_ENABLE                                                                       4:4
#define LWCC97_SET_ZLWLL_STENCIL_ENABLE_FALSE                                                          0x00000000
#define LWCC97_SET_ZLWLL_STENCIL_ENABLE_TRUE                                                           0x00000001

#define LWCC97_SET_ZLWLL_BOUNDS                                                                            0x196c
#define LWCC97_SET_ZLWLL_BOUNDS_Z_MIN_UNBOUNDED_ENABLE                                                        0:0
#define LWCC97_SET_ZLWLL_BOUNDS_Z_MIN_UNBOUNDED_ENABLE_FALSE                                           0x00000000
#define LWCC97_SET_ZLWLL_BOUNDS_Z_MIN_UNBOUNDED_ENABLE_TRUE                                            0x00000001
#define LWCC97_SET_ZLWLL_BOUNDS_Z_MAX_UNBOUNDED_ENABLE                                                        4:4
#define LWCC97_SET_ZLWLL_BOUNDS_Z_MAX_UNBOUNDED_ENABLE_FALSE                                           0x00000000
#define LWCC97_SET_ZLWLL_BOUNDS_Z_MAX_UNBOUNDED_ENABLE_TRUE                                            0x00000001

#define LWCC97_SET_PRIMITIVE_TOPOLOGY                                                                      0x1970
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V                                                                      15:0
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_POINTLIST                                                      0x00000001
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LINELIST                                                       0x00000002
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LINESTRIP                                                      0x00000003
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_TRIANGLELIST                                                   0x00000004
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_TRIANGLESTRIP                                                  0x00000005
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LINELIST_ADJCY                                                 0x0000000A
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LINESTRIP_ADJCY                                                0x0000000B
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_TRIANGLELIST_ADJCY                                             0x0000000C
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_TRIANGLESTRIP_ADJCY                                            0x0000000D
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_PATCHLIST                                                      0x0000000E
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LEGACY_POINTS                                                  0x00001001
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LEGACY_INDEXEDLINELIST                                         0x00001002
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LEGACY_INDEXEDTRIANGLELIST                                     0x00001003
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LEGACY_LINELIST                                                0x0000100F
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LEGACY_LINESTRIP                                               0x00001010
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LEGACY_INDEXEDLINESTRIP                                        0x00001011
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LEGACY_TRIANGLELIST                                            0x00001012
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LEGACY_TRIANGLESTRIP                                           0x00001013
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LEGACY_INDEXEDTRIANGLESTRIP                                    0x00001014
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LEGACY_TRIANGLEFAN                                             0x00001015
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LEGACY_INDEXEDTRIANGLEFAN                                      0x00001016
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LEGACY_TRIANGLEFAN_IMM                                         0x00001017
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LEGACY_LINELIST_IMM                                            0x00001018
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LEGACY_INDEXEDTRIANGLELIST2                                    0x0000101A
#define LWCC97_SET_PRIMITIVE_TOPOLOGY_V_LEGACY_INDEXEDLINELIST2                                        0x0000101B

#define LWCC97_ZLWLL_SYNC                                                                                  0x1978
#define LWCC97_ZLWLL_SYNC_V                                                                                  31:0

#define LWCC97_SET_CLIP_ID_TEST                                                                            0x197c
#define LWCC97_SET_CLIP_ID_TEST_ENABLE                                                                        0:0
#define LWCC97_SET_CLIP_ID_TEST_ENABLE_FALSE                                                           0x00000000
#define LWCC97_SET_CLIP_ID_TEST_ENABLE_TRUE                                                            0x00000001

#define LWCC97_SET_SURFACE_CLIP_ID_WIDTH                                                                   0x1980
#define LWCC97_SET_SURFACE_CLIP_ID_WIDTH_V                                                                   31:0

#define LWCC97_SET_CLIP_ID                                                                                 0x1984
#define LWCC97_SET_CLIP_ID_V                                                                                 31:0

#define LWCC97_SET_DEPTH_BOUNDS_TEST                                                                       0x19bc
#define LWCC97_SET_DEPTH_BOUNDS_TEST_ENABLE                                                                   0:0
#define LWCC97_SET_DEPTH_BOUNDS_TEST_ENABLE_FALSE                                                      0x00000000
#define LWCC97_SET_DEPTH_BOUNDS_TEST_ENABLE_TRUE                                                       0x00000001

#define LWCC97_SET_BLEND_FLOAT_OPTION                                                                      0x19c0
#define LWCC97_SET_BLEND_FLOAT_OPTION_ZERO_TIMES_ANYTHING_IS_ZERO                                             0:0
#define LWCC97_SET_BLEND_FLOAT_OPTION_ZERO_TIMES_ANYTHING_IS_ZERO_FALSE                                0x00000000
#define LWCC97_SET_BLEND_FLOAT_OPTION_ZERO_TIMES_ANYTHING_IS_ZERO_TRUE                                 0x00000001

#define LWCC97_SET_LOGIC_OP                                                                                0x19c4
#define LWCC97_SET_LOGIC_OP_ENABLE                                                                            0:0
#define LWCC97_SET_LOGIC_OP_ENABLE_FALSE                                                               0x00000000
#define LWCC97_SET_LOGIC_OP_ENABLE_TRUE                                                                0x00000001

#define LWCC97_SET_LOGIC_OP_FUNC                                                                           0x19c8
#define LWCC97_SET_LOGIC_OP_FUNC_V                                                                           31:0
#define LWCC97_SET_LOGIC_OP_FUNC_V_CLEAR                                                               0x00001500
#define LWCC97_SET_LOGIC_OP_FUNC_V_AND                                                                 0x00001501
#define LWCC97_SET_LOGIC_OP_FUNC_V_AND_REVERSE                                                         0x00001502
#define LWCC97_SET_LOGIC_OP_FUNC_V_COPY                                                                0x00001503
#define LWCC97_SET_LOGIC_OP_FUNC_V_AND_ILWERTED                                                        0x00001504
#define LWCC97_SET_LOGIC_OP_FUNC_V_NOOP                                                                0x00001505
#define LWCC97_SET_LOGIC_OP_FUNC_V_XOR                                                                 0x00001506
#define LWCC97_SET_LOGIC_OP_FUNC_V_OR                                                                  0x00001507
#define LWCC97_SET_LOGIC_OP_FUNC_V_NOR                                                                 0x00001508
#define LWCC97_SET_LOGIC_OP_FUNC_V_EQUIV                                                               0x00001509
#define LWCC97_SET_LOGIC_OP_FUNC_V_ILWERT                                                              0x0000150A
#define LWCC97_SET_LOGIC_OP_FUNC_V_OR_REVERSE                                                          0x0000150B
#define LWCC97_SET_LOGIC_OP_FUNC_V_COPY_ILWERTED                                                       0x0000150C
#define LWCC97_SET_LOGIC_OP_FUNC_V_OR_ILWERTED                                                         0x0000150D
#define LWCC97_SET_LOGIC_OP_FUNC_V_NAND                                                                0x0000150E
#define LWCC97_SET_LOGIC_OP_FUNC_V_SET                                                                 0x0000150F

#define LWCC97_SET_Z_COMPRESSION                                                                           0x19cc
#define LWCC97_SET_Z_COMPRESSION_ENABLE                                                                       0:0
#define LWCC97_SET_Z_COMPRESSION_ENABLE_FALSE                                                          0x00000000
#define LWCC97_SET_Z_COMPRESSION_ENABLE_TRUE                                                           0x00000001

#define LWCC97_CLEAR_SURFACE                                                                               0x19d0
#define LWCC97_CLEAR_SURFACE_Z_ENABLE                                                                         0:0
#define LWCC97_CLEAR_SURFACE_Z_ENABLE_FALSE                                                            0x00000000
#define LWCC97_CLEAR_SURFACE_Z_ENABLE_TRUE                                                             0x00000001
#define LWCC97_CLEAR_SURFACE_STENCIL_ENABLE                                                                   1:1
#define LWCC97_CLEAR_SURFACE_STENCIL_ENABLE_FALSE                                                      0x00000000
#define LWCC97_CLEAR_SURFACE_STENCIL_ENABLE_TRUE                                                       0x00000001
#define LWCC97_CLEAR_SURFACE_R_ENABLE                                                                         2:2
#define LWCC97_CLEAR_SURFACE_R_ENABLE_FALSE                                                            0x00000000
#define LWCC97_CLEAR_SURFACE_R_ENABLE_TRUE                                                             0x00000001
#define LWCC97_CLEAR_SURFACE_G_ENABLE                                                                         3:3
#define LWCC97_CLEAR_SURFACE_G_ENABLE_FALSE                                                            0x00000000
#define LWCC97_CLEAR_SURFACE_G_ENABLE_TRUE                                                             0x00000001
#define LWCC97_CLEAR_SURFACE_B_ENABLE                                                                         4:4
#define LWCC97_CLEAR_SURFACE_B_ENABLE_FALSE                                                            0x00000000
#define LWCC97_CLEAR_SURFACE_B_ENABLE_TRUE                                                             0x00000001
#define LWCC97_CLEAR_SURFACE_A_ENABLE                                                                         5:5
#define LWCC97_CLEAR_SURFACE_A_ENABLE_FALSE                                                            0x00000000
#define LWCC97_CLEAR_SURFACE_A_ENABLE_TRUE                                                             0x00000001
#define LWCC97_CLEAR_SURFACE_MRT_SELECT                                                                       9:6
#define LWCC97_CLEAR_SURFACE_RT_ARRAY_INDEX                                                                 25:10

#define LWCC97_CLEAR_CLIP_ID_SURFACE                                                                       0x19d4
#define LWCC97_CLEAR_CLIP_ID_SURFACE_V                                                                       31:0

#define LWCC97_SET_COLOR_COMPRESSION(i)                                                            (0x19e0+(i)*4)
#define LWCC97_SET_COLOR_COMPRESSION_ENABLE                                                                   0:0
#define LWCC97_SET_COLOR_COMPRESSION_ENABLE_FALSE                                                      0x00000000
#define LWCC97_SET_COLOR_COMPRESSION_ENABLE_TRUE                                                       0x00000001

#define LWCC97_SET_CT_WRITE(i)                                                                     (0x1a00+(i)*4)
#define LWCC97_SET_CT_WRITE_R_ENABLE                                                                          0:0
#define LWCC97_SET_CT_WRITE_R_ENABLE_FALSE                                                             0x00000000
#define LWCC97_SET_CT_WRITE_R_ENABLE_TRUE                                                              0x00000001
#define LWCC97_SET_CT_WRITE_G_ENABLE                                                                          4:4
#define LWCC97_SET_CT_WRITE_G_ENABLE_FALSE                                                             0x00000000
#define LWCC97_SET_CT_WRITE_G_ENABLE_TRUE                                                              0x00000001
#define LWCC97_SET_CT_WRITE_B_ENABLE                                                                          8:8
#define LWCC97_SET_CT_WRITE_B_ENABLE_FALSE                                                             0x00000000
#define LWCC97_SET_CT_WRITE_B_ENABLE_TRUE                                                              0x00000001
#define LWCC97_SET_CT_WRITE_A_ENABLE                                                                        12:12
#define LWCC97_SET_CT_WRITE_A_ENABLE_FALSE                                                             0x00000000
#define LWCC97_SET_CT_WRITE_A_ENABLE_TRUE                                                              0x00000001

#define LWCC97_TEST_FOR_QUADRO                                                                             0x1a24
#define LWCC97_TEST_FOR_QUADRO_V                                                                             31:0

#define LWCC97_PIPE_NOP                                                                                    0x1a2c
#define LWCC97_PIPE_NOP_V                                                                                    31:0

#define LWCC97_SET_SPARE00                                                                                 0x1a30
#define LWCC97_SET_SPARE00_V                                                                                 31:0

#define LWCC97_SET_SPARE01                                                                                 0x1a34
#define LWCC97_SET_SPARE01_V                                                                                 31:0

#define LWCC97_SET_SPARE02                                                                                 0x1a38
#define LWCC97_SET_SPARE02_V                                                                                 31:0

#define LWCC97_SET_SPARE03                                                                                 0x1a3c
#define LWCC97_SET_SPARE03_V                                                                                 31:0

#define LWCC97_SET_REPORT_SEMAPHORE_A                                                                      0x1b00
#define LWCC97_SET_REPORT_SEMAPHORE_A_OFFSET_UPPER                                                           24:0

#define LWCC97_SET_REPORT_SEMAPHORE_B                                                                      0x1b04
#define LWCC97_SET_REPORT_SEMAPHORE_B_OFFSET_LOWER                                                           31:0

#define LWCC97_SET_REPORT_SEMAPHORE_C                                                                      0x1b08
#define LWCC97_SET_REPORT_SEMAPHORE_C_PAYLOAD                                                                31:0

#define LWCC97_SET_REPORT_SEMAPHORE_D                                                                      0x1b0c
#define LWCC97_SET_REPORT_SEMAPHORE_D_OPERATION                                                               1:0
#define LWCC97_SET_REPORT_SEMAPHORE_D_OPERATION_RELEASE                                                0x00000000
#define LWCC97_SET_REPORT_SEMAPHORE_D_OPERATION_ACQUIRE                                                0x00000001
#define LWCC97_SET_REPORT_SEMAPHORE_D_OPERATION_REPORT_ONLY                                            0x00000002
#define LWCC97_SET_REPORT_SEMAPHORE_D_OPERATION_TRAP                                                   0x00000003
#define LWCC97_SET_REPORT_SEMAPHORE_D_RELEASE                                                                 4:4
#define LWCC97_SET_REPORT_SEMAPHORE_D_RELEASE_AFTER_ALL_PRECEEDING_READS_COMPLETE                      0x00000000
#define LWCC97_SET_REPORT_SEMAPHORE_D_RELEASE_AFTER_ALL_PRECEEDING_WRITES_COMPLETE                     0x00000001
#define LWCC97_SET_REPORT_SEMAPHORE_D_ACQUIRE                                                                 8:8
#define LWCC97_SET_REPORT_SEMAPHORE_D_ACQUIRE_BEFORE_ANY_FOLLOWING_WRITES_START                        0x00000000
#define LWCC97_SET_REPORT_SEMAPHORE_D_ACQUIRE_BEFORE_ANY_FOLLOWING_READS_START                         0x00000001
#define LWCC97_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION                                                     15:12
#define LWCC97_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_NONE                                           0x00000000
#define LWCC97_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_DATA_ASSEMBLER                                 0x00000001
#define LWCC97_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_VERTEX_SHADER                                  0x00000002
#define LWCC97_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_TESSELATION_INIT_SHADER                        0x00000008
#define LWCC97_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_TESSELATION_SHADER                             0x00000009
#define LWCC97_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_GEOMETRY_SHADER                                0x00000006
#define LWCC97_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_STREAMING_OUTPUT                               0x00000005
#define LWCC97_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_VPC                                            0x00000004
#define LWCC97_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_ZLWLL                                          0x00000007
#define LWCC97_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_PIXEL_SHADER                                   0x0000000A
#define LWCC97_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_DEPTH_TEST                                     0x0000000C
#define LWCC97_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_ALL                                            0x0000000F
#define LWCC97_SET_REPORT_SEMAPHORE_D_COMPARISON                                                            16:16
#define LWCC97_SET_REPORT_SEMAPHORE_D_COMPARISON_EQ                                                    0x00000000
#define LWCC97_SET_REPORT_SEMAPHORE_D_COMPARISON_GE                                                    0x00000001
#define LWCC97_SET_REPORT_SEMAPHORE_D_AWAKEN_ENABLE                                                         20:20
#define LWCC97_SET_REPORT_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                              0x00000000
#define LWCC97_SET_REPORT_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                               0x00000001
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT                                                                27:23
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_NONE                                                      0x00000000
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_DA_VERTICES_GENERATED                                     0x00000001
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_DA_PRIMITIVES_GENERATED                                   0x00000003
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_VS_ILWOCATIONS                                            0x00000005
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_TI_ILWOCATIONS                                            0x0000001B
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_TS_ILWOCATIONS                                            0x0000001D
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_TS_PRIMITIVES_GENERATED                                   0x0000001F
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_GS_ILWOCATIONS                                            0x00000007
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_GS_PRIMITIVES_GENERATED                                   0x00000009
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_ALPHA_BETA_CLOCKS                                         0x00000004
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_SCG_CLOCKS                                                0x00000008
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_VTG_PRIMITIVES_OUT                                        0x00000012
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_TOTAL_STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED             0x0000001E
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_SUCCEEDED                            0x0000000B
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED                               0x0000000D
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED               0x00000006
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_BYTE_COUNT                                      0x0000001A
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_ILWOCATIONS                                       0x0000000F
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_CLIPPER_PRIMITIVES_GENERATED                              0x00000011
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS0                                              0x0000000A
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS1                                              0x0000000C
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS2                                              0x0000000E
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_ZLWLL_STATS3                                              0x00000010
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_PS_ILWOCATIONS                                            0x00000013
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT                                           0x00000002
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT64                                         0x00000015
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_TILED_ZPASS_PIXEL_CNT64                                   0x00000017
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_COLOR_TARGET                                   0x00000018
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_IEEE_CLEAN_ZETA_TARGET                                    0x00000019
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_BOUNDING_RECTANGLE                                        0x0000001C
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_TIMESTAMP                                                 0x00000014
#define LWCC97_SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE                                                        28:28
#define LWCC97_SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_FOUR_WORDS                                        0x00000000
#define LWCC97_SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD                                          0x00000001
#define LWCC97_SET_REPORT_SEMAPHORE_D_SUB_REPORT                                                              7:5
#define LWCC97_SET_REPORT_SEMAPHORE_D_REPORT_DWORD_NUMBER                                                   21:21
#define LWCC97_SET_REPORT_SEMAPHORE_D_FLUSH_DISABLE                                                           2:2
#define LWCC97_SET_REPORT_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                              0x00000000
#define LWCC97_SET_REPORT_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                               0x00000001
#define LWCC97_SET_REPORT_SEMAPHORE_D_REDUCTION_ENABLE                                                        3:3
#define LWCC97_SET_REPORT_SEMAPHORE_D_REDUCTION_ENABLE_FALSE                                           0x00000000
#define LWCC97_SET_REPORT_SEMAPHORE_D_REDUCTION_ENABLE_TRUE                                            0x00000001
#define LWCC97_SET_REPORT_SEMAPHORE_D_REDUCTION_OP                                                           11:9
#define LWCC97_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_ADD                                             0x00000000
#define LWCC97_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_MIN                                             0x00000001
#define LWCC97_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_MAX                                             0x00000002
#define LWCC97_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_INC                                             0x00000003
#define LWCC97_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_DEC                                             0x00000004
#define LWCC97_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_AND                                             0x00000005
#define LWCC97_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_OR                                              0x00000006
#define LWCC97_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_XOR                                             0x00000007
#define LWCC97_SET_REPORT_SEMAPHORE_D_REDUCTION_FORMAT                                                      18:17
#define LWCC97_SET_REPORT_SEMAPHORE_D_REDUCTION_FORMAT_UNSIGNED_32                                     0x00000000
#define LWCC97_SET_REPORT_SEMAPHORE_D_REDUCTION_FORMAT_SIGNED_32                                       0x00000001
#define LWCC97_SET_REPORT_SEMAPHORE_D_CONDITIONAL_TRAP                                                      19:19
#define LWCC97_SET_REPORT_SEMAPHORE_D_CONDITIONAL_TRAP_FALSE                                           0x00000000
#define LWCC97_SET_REPORT_SEMAPHORE_D_CONDITIONAL_TRAP_TRUE                                            0x00000001

#define LWCC97_SET_VERTEX_STREAM_A_FORMAT(j)                                                      (0x1c00+(j)*16)
#define LWCC97_SET_VERTEX_STREAM_A_FORMAT_STRIDE                                                             11:0
#define LWCC97_SET_VERTEX_STREAM_A_FORMAT_ENABLE                                                            12:12
#define LWCC97_SET_VERTEX_STREAM_A_FORMAT_ENABLE_FALSE                                                 0x00000000
#define LWCC97_SET_VERTEX_STREAM_A_FORMAT_ENABLE_TRUE                                                  0x00000001

#define LWCC97_SET_VERTEX_STREAM_A_LOCATION_A(j)                                                  (0x1c04+(j)*16)
#define LWCC97_SET_VERTEX_STREAM_A_LOCATION_A_OFFSET_UPPER                                                    7:0

#define LWCC97_SET_VERTEX_STREAM_A_LOCATION_B(j)                                                  (0x1c08+(j)*16)
#define LWCC97_SET_VERTEX_STREAM_A_LOCATION_B_OFFSET_LOWER                                                   31:0

#define LWCC97_SET_VERTEX_STREAM_A_FREQUENCY(j)                                                   (0x1c0c+(j)*16)
#define LWCC97_SET_VERTEX_STREAM_A_FREQUENCY_V                                                               31:0

#define LWCC97_SET_VERTEX_STREAM_B_FORMAT(j)                                                      (0x1d00+(j)*16)
#define LWCC97_SET_VERTEX_STREAM_B_FORMAT_STRIDE                                                             11:0
#define LWCC97_SET_VERTEX_STREAM_B_FORMAT_ENABLE                                                            12:12
#define LWCC97_SET_VERTEX_STREAM_B_FORMAT_ENABLE_FALSE                                                 0x00000000
#define LWCC97_SET_VERTEX_STREAM_B_FORMAT_ENABLE_TRUE                                                  0x00000001

#define LWCC97_SET_VERTEX_STREAM_B_LOCATION_A(j)                                                  (0x1d04+(j)*16)
#define LWCC97_SET_VERTEX_STREAM_B_LOCATION_A_OFFSET_UPPER                                                    7:0

#define LWCC97_SET_VERTEX_STREAM_B_LOCATION_B(j)                                                  (0x1d08+(j)*16)
#define LWCC97_SET_VERTEX_STREAM_B_LOCATION_B_OFFSET_LOWER                                                   31:0

#define LWCC97_SET_VERTEX_STREAM_B_FREQUENCY(j)                                                   (0x1d0c+(j)*16)
#define LWCC97_SET_VERTEX_STREAM_B_FREQUENCY_V                                                               31:0

#define LWCC97_SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA(j)                                         (0x1e00+(j)*32)
#define LWCC97_SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA_ENABLE                                                 0:0
#define LWCC97_SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA_ENABLE_FALSE                                    0x00000000
#define LWCC97_SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA_ENABLE_TRUE                                     0x00000001

#define LWCC97_SET_BLEND_PER_TARGET_COLOR_OP(j)                                                   (0x1e04+(j)*32)
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_OP_V                                                               31:0
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_OP_V_OGL_FUNC_SUBTRACT                                       0x0000800A
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_OP_V_OGL_FUNC_REVERSE_SUBTRACT                               0x0000800B
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_OP_V_OGL_FUNC_ADD                                            0x00008006
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_OP_V_OGL_MIN                                                 0x00008007
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_OP_V_OGL_MAX                                                 0x00008008
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_OP_V_D3D_ADD                                                 0x00000001
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_OP_V_D3D_SUBTRACT                                            0x00000002
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_OP_V_D3D_REVSUBTRACT                                         0x00000003
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_OP_V_D3D_MIN                                                 0x00000004
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_OP_V_D3D_MAX                                                 0x00000005

#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF(j)                                         (0x1e08+(j)*32)
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V                                                     31:0
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ZERO                                      0x00004000
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ONE                                       0x00004001
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_SRC_COLOR                                 0x00004300
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_SRC_COLOR                       0x00004301
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_SRC_ALPHA                                 0x00004302
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA                       0x00004303
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_DST_ALPHA                                 0x00004304
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_DST_ALPHA                       0x00004305
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_DST_COLOR                                 0x00004306
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_DST_COLOR                       0x00004307
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_SRC_ALPHA_SATURATE                        0x00004308
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_CONSTANT_COLOR                            0x0000C001
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_CONSTANT_COLOR                  0x0000C002
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_CONSTANT_ALPHA                            0x0000C003
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ONE_MINUS_CONSTANT_ALPHA                  0x0000C004
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_SRC1COLOR                                 0x0000C900
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ILWSRC1COLOR                              0x0000C901
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_SRC1ALPHA                                 0x0000C902
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ILWSRC1ALPHA                              0x0000C903
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ZERO                                      0x00000001
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ONE                                       0x00000002
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_SRCCOLOR                                  0x00000003
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ILWSRCCOLOR                               0x00000004
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_SRCALPHA                                  0x00000005
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ILWSRCALPHA                               0x00000006
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_DESTALPHA                                 0x00000007
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ILWDESTALPHA                              0x00000008
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_DESTCOLOR                                 0x00000009
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ILWDESTCOLOR                              0x0000000A
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_SRCALPHASAT                               0x0000000B
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_BOTHSRCALPHA                              0x0000000C
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_BOTHILWSRCALPHA                           0x0000000D
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_BLENDFACTOR                               0x0000000E
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ILWBLENDFACTOR                            0x0000000F
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_SRC1COLOR                                 0x00000010
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ILWSRC1COLOR                              0x00000011
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_SRC1ALPHA                                 0x00000012
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_D3D_ILWSRC1ALPHA                              0x00000013

#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF(j)                                           (0x1e0c+(j)*32)
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V                                                       31:0
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ZERO                                        0x00004000
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ONE                                         0x00004001
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_SRC_COLOR                                   0x00004300
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_SRC_COLOR                         0x00004301
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_SRC_ALPHA                                   0x00004302
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA                         0x00004303
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_DST_ALPHA                                   0x00004304
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_DST_ALPHA                         0x00004305
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_DST_COLOR                                   0x00004306
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_DST_COLOR                         0x00004307
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_SRC_ALPHA_SATURATE                          0x00004308
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_CONSTANT_COLOR                              0x0000C001
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_CONSTANT_COLOR                    0x0000C002
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_CONSTANT_ALPHA                              0x0000C003
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_CONSTANT_ALPHA                    0x0000C004
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_SRC1COLOR                                   0x0000C900
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ILWSRC1COLOR                                0x0000C901
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_SRC1ALPHA                                   0x0000C902
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_ILWSRC1ALPHA                                0x0000C903
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ZERO                                        0x00000001
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ONE                                         0x00000002
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_SRCCOLOR                                    0x00000003
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ILWSRCCOLOR                                 0x00000004
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_SRCALPHA                                    0x00000005
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ILWSRCALPHA                                 0x00000006
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_DESTALPHA                                   0x00000007
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ILWDESTALPHA                                0x00000008
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_DESTCOLOR                                   0x00000009
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ILWDESTCOLOR                                0x0000000A
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_SRCALPHASAT                                 0x0000000B
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_BLENDFACTOR                                 0x0000000E
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ILWBLENDFACTOR                              0x0000000F
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_SRC1COLOR                                   0x00000010
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ILWSRC1COLOR                                0x00000011
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_SRC1ALPHA                                   0x00000012
#define LWCC97_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_D3D_ILWSRC1ALPHA                                0x00000013

#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_OP(j)                                                   (0x1e10+(j)*32)
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_OP_V                                                               31:0
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_OP_V_OGL_FUNC_SUBTRACT                                       0x0000800A
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_OP_V_OGL_FUNC_REVERSE_SUBTRACT                               0x0000800B
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_OP_V_OGL_FUNC_ADD                                            0x00008006
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_OP_V_OGL_MIN                                                 0x00008007
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_OP_V_OGL_MAX                                                 0x00008008
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_OP_V_D3D_ADD                                                 0x00000001
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_OP_V_D3D_SUBTRACT                                            0x00000002
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_OP_V_D3D_REVSUBTRACT                                         0x00000003
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_OP_V_D3D_MIN                                                 0x00000004
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_OP_V_D3D_MAX                                                 0x00000005

#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF(j)                                         (0x1e14+(j)*32)
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V                                                     31:0
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ZERO                                      0x00004000
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ONE                                       0x00004001
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_SRC_COLOR                                 0x00004300
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_SRC_COLOR                       0x00004301
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_SRC_ALPHA                                 0x00004302
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA                       0x00004303
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_DST_ALPHA                                 0x00004304
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_DST_ALPHA                       0x00004305
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_DST_COLOR                                 0x00004306
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_DST_COLOR                       0x00004307
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_SRC_ALPHA_SATURATE                        0x00004308
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_CONSTANT_COLOR                            0x0000C001
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_CONSTANT_COLOR                  0x0000C002
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_CONSTANT_ALPHA                            0x0000C003
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ONE_MINUS_CONSTANT_ALPHA                  0x0000C004
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_SRC1COLOR                                 0x0000C900
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ILWSRC1COLOR                              0x0000C901
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_SRC1ALPHA                                 0x0000C902
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_ILWSRC1ALPHA                              0x0000C903
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ZERO                                      0x00000001
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ONE                                       0x00000002
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_SRCCOLOR                                  0x00000003
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ILWSRCCOLOR                               0x00000004
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_SRCALPHA                                  0x00000005
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ILWSRCALPHA                               0x00000006
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_DESTALPHA                                 0x00000007
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ILWDESTALPHA                              0x00000008
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_DESTCOLOR                                 0x00000009
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ILWDESTCOLOR                              0x0000000A
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_SRCALPHASAT                               0x0000000B
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_BOTHSRCALPHA                              0x0000000C
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_BOTHILWSRCALPHA                           0x0000000D
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_BLENDFACTOR                               0x0000000E
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ILWBLENDFACTOR                            0x0000000F
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_SRC1COLOR                                 0x00000010
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ILWSRC1COLOR                              0x00000011
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_SRC1ALPHA                                 0x00000012
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_D3D_ILWSRC1ALPHA                              0x00000013

#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF(j)                                           (0x1e18+(j)*32)
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V                                                       31:0
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ZERO                                        0x00004000
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ONE                                         0x00004001
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_SRC_COLOR                                   0x00004300
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_SRC_COLOR                         0x00004301
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_SRC_ALPHA                                   0x00004302
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA                         0x00004303
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_DST_ALPHA                                   0x00004304
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_DST_ALPHA                         0x00004305
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_DST_COLOR                                   0x00004306
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_DST_COLOR                         0x00004307
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_SRC_ALPHA_SATURATE                          0x00004308
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_CONSTANT_COLOR                              0x0000C001
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_CONSTANT_COLOR                    0x0000C002
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_CONSTANT_ALPHA                              0x0000C003
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ONE_MINUS_CONSTANT_ALPHA                    0x0000C004
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_SRC1COLOR                                   0x0000C900
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ILWSRC1COLOR                                0x0000C901
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_SRC1ALPHA                                   0x0000C902
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_ILWSRC1ALPHA                                0x0000C903
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ZERO                                        0x00000001
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ONE                                         0x00000002
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_SRCCOLOR                                    0x00000003
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ILWSRCCOLOR                                 0x00000004
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_SRCALPHA                                    0x00000005
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ILWSRCALPHA                                 0x00000006
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_DESTALPHA                                   0x00000007
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ILWDESTALPHA                                0x00000008
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_DESTCOLOR                                   0x00000009
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ILWDESTCOLOR                                0x0000000A
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_SRCALPHASAT                                 0x0000000B
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_BLENDFACTOR                                 0x0000000E
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ILWBLENDFACTOR                              0x0000000F
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_SRC1COLOR                                   0x00000010
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ILWSRC1COLOR                                0x00000011
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_SRC1ALPHA                                   0x00000012
#define LWCC97_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_D3D_ILWSRC1ALPHA                                0x00000013

#define LWCC97_SET_PIPELINE_SHADER(j)                                                             (0x2000+(j)*64)
#define LWCC97_SET_PIPELINE_SHADER_ENABLE                                                                     0:0
#define LWCC97_SET_PIPELINE_SHADER_ENABLE_FALSE                                                        0x00000000
#define LWCC97_SET_PIPELINE_SHADER_ENABLE_TRUE                                                         0x00000001
#define LWCC97_SET_PIPELINE_SHADER_TYPE                                                                       7:4
#define LWCC97_SET_PIPELINE_SHADER_TYPE_VERTEX_LWLL_BEFORE_FETCH                                       0x00000000
#define LWCC97_SET_PIPELINE_SHADER_TYPE_VERTEX                                                         0x00000001
#define LWCC97_SET_PIPELINE_SHADER_TYPE_TESSELLATION_INIT                                              0x00000002
#define LWCC97_SET_PIPELINE_SHADER_TYPE_TESSELLATION                                                   0x00000003
#define LWCC97_SET_PIPELINE_SHADER_TYPE_GEOMETRY                                                       0x00000004
#define LWCC97_SET_PIPELINE_SHADER_TYPE_PIXEL                                                          0x00000005

#define LWCC97_SET_PIPELINE_RESERVED_B(j)                                                         (0x2004+(j)*64)
#define LWCC97_SET_PIPELINE_RESERVED_B_V                                                                      0:0

#define LWCC97_SET_PIPELINE_RESERVED_A(j)                                                         (0x2008+(j)*64)
#define LWCC97_SET_PIPELINE_RESERVED_A_V                                                                      0:0

#define LWCC97_SET_PIPELINE_REGISTER_COUNT(j)                                                     (0x200c+(j)*64)
#define LWCC97_SET_PIPELINE_REGISTER_COUNT_V                                                                  8:0

#define LWCC97_SET_PIPELINE_BINDING(j)                                                            (0x2010+(j)*64)
#define LWCC97_SET_PIPELINE_BINDING_GROUP                                                                     2:0

#define LWCC97_SET_PIPELINE_PROGRAM_ADDRESS_A(j)                                                  (0x2014+(j)*64)
#define LWCC97_SET_PIPELINE_PROGRAM_ADDRESS_A_UPPER                                                           7:0

#define LWCC97_SET_PIPELINE_PROGRAM_ADDRESS_B(j)                                                  (0x2018+(j)*64)
#define LWCC97_SET_PIPELINE_PROGRAM_ADDRESS_B_LOWER                                                          31:0

#define LWCC97_SET_PIPELINE_PROGRAM_PREFETCH(j)                                                   (0x201c+(j)*64)
#define LWCC97_SET_PIPELINE_PROGRAM_PREFETCH_SIZE_IN_BLOCKS                                                   6:0

#define LWCC97_SET_PIPELINE_RESERVED_E(j)                                                         (0x2020+(j)*64)
#define LWCC97_SET_PIPELINE_RESERVED_E_V                                                                      0:0

#define LWCC97_SET_FALCON00                                                                                0x2300
#define LWCC97_SET_FALCON00_V                                                                                31:0

#define LWCC97_SET_FALCON01                                                                                0x2304
#define LWCC97_SET_FALCON01_V                                                                                31:0

#define LWCC97_SET_FALCON02                                                                                0x2308
#define LWCC97_SET_FALCON02_V                                                                                31:0

#define LWCC97_SET_FALCON03                                                                                0x230c
#define LWCC97_SET_FALCON03_V                                                                                31:0

#define LWCC97_SET_FALCON04                                                                                0x2310
#define LWCC97_SET_FALCON04_V                                                                                31:0

#define LWCC97_SET_FALCON05                                                                                0x2314
#define LWCC97_SET_FALCON05_V                                                                                31:0

#define LWCC97_SET_FALCON06                                                                                0x2318
#define LWCC97_SET_FALCON06_V                                                                                31:0

#define LWCC97_SET_FALCON07                                                                                0x231c
#define LWCC97_SET_FALCON07_V                                                                                31:0

#define LWCC97_SET_FALCON08                                                                                0x2320
#define LWCC97_SET_FALCON08_V                                                                                31:0

#define LWCC97_SET_FALCON09                                                                                0x2324
#define LWCC97_SET_FALCON09_V                                                                                31:0

#define LWCC97_SET_FALCON10                                                                                0x2328
#define LWCC97_SET_FALCON10_V                                                                                31:0

#define LWCC97_SET_FALCON11                                                                                0x232c
#define LWCC97_SET_FALCON11_V                                                                                31:0

#define LWCC97_SET_FALCON12                                                                                0x2330
#define LWCC97_SET_FALCON12_V                                                                                31:0

#define LWCC97_SET_FALCON13                                                                                0x2334
#define LWCC97_SET_FALCON13_V                                                                                31:0

#define LWCC97_SET_FALCON14                                                                                0x2338
#define LWCC97_SET_FALCON14_V                                                                                31:0

#define LWCC97_SET_FALCON15                                                                                0x233c
#define LWCC97_SET_FALCON15_V                                                                                31:0

#define LWCC97_SET_FALCON16                                                                                0x2340
#define LWCC97_SET_FALCON16_V                                                                                31:0

#define LWCC97_SET_FALCON17                                                                                0x2344
#define LWCC97_SET_FALCON17_V                                                                                31:0

#define LWCC97_SET_FALCON18                                                                                0x2348
#define LWCC97_SET_FALCON18_V                                                                                31:0

#define LWCC97_SET_FALCON19                                                                                0x234c
#define LWCC97_SET_FALCON19_V                                                                                31:0

#define LWCC97_SET_FALCON20                                                                                0x2350
#define LWCC97_SET_FALCON20_V                                                                                31:0

#define LWCC97_SET_FALCON21                                                                                0x2354
#define LWCC97_SET_FALCON21_V                                                                                31:0

#define LWCC97_SET_FALCON22                                                                                0x2358
#define LWCC97_SET_FALCON22_V                                                                                31:0

#define LWCC97_SET_FALCON23                                                                                0x235c
#define LWCC97_SET_FALCON23_V                                                                                31:0

#define LWCC97_SET_FALCON24                                                                                0x2360
#define LWCC97_SET_FALCON24_V                                                                                31:0

#define LWCC97_SET_FALCON25                                                                                0x2364
#define LWCC97_SET_FALCON25_V                                                                                31:0

#define LWCC97_SET_FALCON26                                                                                0x2368
#define LWCC97_SET_FALCON26_V                                                                                31:0

#define LWCC97_SET_FALCON27                                                                                0x236c
#define LWCC97_SET_FALCON27_V                                                                                31:0

#define LWCC97_SET_FALCON28                                                                                0x2370
#define LWCC97_SET_FALCON28_V                                                                                31:0

#define LWCC97_SET_FALCON29                                                                                0x2374
#define LWCC97_SET_FALCON29_V                                                                                31:0

#define LWCC97_SET_FALCON30                                                                                0x2378
#define LWCC97_SET_FALCON30_V                                                                                31:0

#define LWCC97_SET_FALCON31                                                                                0x237c
#define LWCC97_SET_FALCON31_V                                                                                31:0

#define LWCC97_SET_CONSTANT_BUFFER_SELECTOR_A                                                              0x2380
#define LWCC97_SET_CONSTANT_BUFFER_SELECTOR_A_SIZE                                                           16:0

#define LWCC97_SET_CONSTANT_BUFFER_SELECTOR_B                                                              0x2384
#define LWCC97_SET_CONSTANT_BUFFER_SELECTOR_B_ADDRESS_UPPER                                                   7:0

#define LWCC97_SET_CONSTANT_BUFFER_SELECTOR_C                                                              0x2388
#define LWCC97_SET_CONSTANT_BUFFER_SELECTOR_C_ADDRESS_LOWER                                                  31:0

#define LWCC97_LOAD_CONSTANT_BUFFER_OFFSET                                                                 0x238c
#define LWCC97_LOAD_CONSTANT_BUFFER_OFFSET_V                                                                 15:0

#define LWCC97_LOAD_CONSTANT_BUFFER(i)                                                             (0x2390+(i)*4)
#define LWCC97_LOAD_CONSTANT_BUFFER_V                                                                        31:0

#define LWCC97_BIND_GROUP_RESERVED_A(j)                                                           (0x2400+(j)*32)
#define LWCC97_BIND_GROUP_RESERVED_A_V                                                                        0:0

#define LWCC97_BIND_GROUP_RESERVED_B(j)                                                           (0x2404+(j)*32)
#define LWCC97_BIND_GROUP_RESERVED_B_V                                                                        0:0

#define LWCC97_BIND_GROUP_RESERVED_C(j)                                                           (0x2408+(j)*32)
#define LWCC97_BIND_GROUP_RESERVED_C_V                                                                        0:0

#define LWCC97_BIND_GROUP_RESERVED_D(j)                                                           (0x240c+(j)*32)
#define LWCC97_BIND_GROUP_RESERVED_D_V                                                                        0:0

#define LWCC97_BIND_GROUP_CONSTANT_BUFFER(j)                                                      (0x2410+(j)*32)
#define LWCC97_BIND_GROUP_CONSTANT_BUFFER_VALID                                                               0:0
#define LWCC97_BIND_GROUP_CONSTANT_BUFFER_VALID_FALSE                                                  0x00000000
#define LWCC97_BIND_GROUP_CONSTANT_BUFFER_VALID_TRUE                                                   0x00000001
#define LWCC97_BIND_GROUP_CONSTANT_BUFFER_SHADER_SLOT                                                         8:4

#define LWCC97_SET_TRAP_HANDLER_A                                                                          0x25f8
#define LWCC97_SET_TRAP_HANDLER_A_ADDRESS_UPPER                                                              16:0

#define LWCC97_SET_TRAP_HANDLER_B                                                                          0x25fc
#define LWCC97_SET_TRAP_HANDLER_B_ADDRESS_LOWER                                                              31:0

#define LWCC97_SET_COLOR_CLAMP                                                                             0x2600
#define LWCC97_SET_COLOR_CLAMP_ENABLE                                                                         0:0
#define LWCC97_SET_COLOR_CLAMP_ENABLE_FALSE                                                            0x00000000
#define LWCC97_SET_COLOR_CLAMP_ENABLE_TRUE                                                             0x00000001

#define LWCC97_SET_SCISSOR_EXCLUSIVE_ENABLE(j)                                                    (0x2610+(j)*16)
#define LWCC97_SET_SCISSOR_EXCLUSIVE_ENABLE_V                                                                 0:0
#define LWCC97_SET_SCISSOR_EXCLUSIVE_ENABLE_V_FALSE                                                    0x00000000
#define LWCC97_SET_SCISSOR_EXCLUSIVE_ENABLE_V_TRUE                                                     0x00000001

#define LWCC97_SET_SCISSOR_EXCLUSIVE_HORIZONTAL(j)                                                (0x2614+(j)*16)
#define LWCC97_SET_SCISSOR_EXCLUSIVE_HORIZONTAL_XMIN                                                         15:0
#define LWCC97_SET_SCISSOR_EXCLUSIVE_HORIZONTAL_XMAX                                                        31:16

#define LWCC97_SET_SCISSOR_EXCLUSIVE_VERTICAL(j)                                                  (0x2618+(j)*16)
#define LWCC97_SET_SCISSOR_EXCLUSIVE_VERTICAL_YMIN                                                           15:0
#define LWCC97_SET_SCISSOR_EXCLUSIVE_VERTICAL_YMAX                                                          31:16

#define LWCC97_SET_STREAM_OUT_LAYOUT_SELECT(i,j)                                           (0x2800+(i)*128+(j)*4)
#define LWCC97_SET_STREAM_OUT_LAYOUT_SELECT_ATTRIBUTE_NUMBER00                                                7:0
#define LWCC97_SET_STREAM_OUT_LAYOUT_SELECT_ATTRIBUTE_NUMBER01                                               15:8
#define LWCC97_SET_STREAM_OUT_LAYOUT_SELECT_ATTRIBUTE_NUMBER02                                              23:16
#define LWCC97_SET_STREAM_OUT_LAYOUT_SELECT_ATTRIBUTE_NUMBER03                                              31:24

#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_CONTROL(j)                                         (0x2a00+(j)*16)
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_CONTROL_ENABLE                                                 0:0
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_CONTROL_ENABLE_FALSE                                    0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_CONTROL_ENABLE_TRUE                                     0x00000001

#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A(j)                                 (0x2a04+(j)*16)
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX0                                    3:0
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX0_PS_X0_LWLL_RASTER_PIXELS             0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX0_PS_X16_PER_RASTER_PIXEL             0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX0_PS_X8_PER_RASTER_PIXEL             0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX0_PS_X4_PER_RASTER_PIXEL             0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX0_PS_X2_PER_RASTER_PIXEL             0x00000004
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX0_PS_X1_PER_RASTER_PIXEL             0x00000005
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX0_PS_X1_PER_2X1_RASTER_PIXELS             0x00000006
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX0_PS_X1_PER_1X2_RASTER_PIXELS             0x00000007
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX0_PS_X1_PER_2X2_RASTER_PIXELS             0x00000008
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX0_PS_X1_PER_4X2_RASTER_PIXELS             0x00000009
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX0_PS_X1_PER_2X4_RASTER_PIXELS             0x0000000A
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX0_PS_X1_PER_4X4_RASTER_PIXELS             0x0000000B
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX1                                    7:4
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX1_PS_X0_LWLL_RASTER_PIXELS             0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX1_PS_X16_PER_RASTER_PIXEL             0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX1_PS_X8_PER_RASTER_PIXEL             0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX1_PS_X4_PER_RASTER_PIXEL             0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX1_PS_X2_PER_RASTER_PIXEL             0x00000004
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX1_PS_X1_PER_RASTER_PIXEL             0x00000005
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX1_PS_X1_PER_2X1_RASTER_PIXELS             0x00000006
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX1_PS_X1_PER_1X2_RASTER_PIXELS             0x00000007
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX1_PS_X1_PER_2X2_RASTER_PIXELS             0x00000008
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX1_PS_X1_PER_4X2_RASTER_PIXELS             0x00000009
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX1_PS_X1_PER_2X4_RASTER_PIXELS             0x0000000A
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX1_PS_X1_PER_4X4_RASTER_PIXELS             0x0000000B
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX2                                   11:8
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX2_PS_X0_LWLL_RASTER_PIXELS             0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX2_PS_X16_PER_RASTER_PIXEL             0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX2_PS_X8_PER_RASTER_PIXEL             0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX2_PS_X4_PER_RASTER_PIXEL             0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX2_PS_X2_PER_RASTER_PIXEL             0x00000004
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX2_PS_X1_PER_RASTER_PIXEL             0x00000005
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX2_PS_X1_PER_2X1_RASTER_PIXELS             0x00000006
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX2_PS_X1_PER_1X2_RASTER_PIXELS             0x00000007
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX2_PS_X1_PER_2X2_RASTER_PIXELS             0x00000008
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX2_PS_X1_PER_4X2_RASTER_PIXELS             0x00000009
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX2_PS_X1_PER_2X4_RASTER_PIXELS             0x0000000A
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX2_PS_X1_PER_4X4_RASTER_PIXELS             0x0000000B
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX3                                  15:12
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX3_PS_X0_LWLL_RASTER_PIXELS             0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX3_PS_X16_PER_RASTER_PIXEL             0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX3_PS_X8_PER_RASTER_PIXEL             0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX3_PS_X4_PER_RASTER_PIXEL             0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX3_PS_X2_PER_RASTER_PIXEL             0x00000004
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX3_PS_X1_PER_RASTER_PIXEL             0x00000005
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX3_PS_X1_PER_2X1_RASTER_PIXELS             0x00000006
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX3_PS_X1_PER_1X2_RASTER_PIXELS             0x00000007
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX3_PS_X1_PER_2X2_RASTER_PIXELS             0x00000008
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX3_PS_X1_PER_4X2_RASTER_PIXELS             0x00000009
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX3_PS_X1_PER_2X4_RASTER_PIXELS             0x0000000A
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX3_PS_X1_PER_4X4_RASTER_PIXELS             0x0000000B
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX4                                  19:16
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX4_PS_X0_LWLL_RASTER_PIXELS             0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX4_PS_X16_PER_RASTER_PIXEL             0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX4_PS_X8_PER_RASTER_PIXEL             0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX4_PS_X4_PER_RASTER_PIXEL             0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX4_PS_X2_PER_RASTER_PIXEL             0x00000004
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX4_PS_X1_PER_RASTER_PIXEL             0x00000005
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX4_PS_X1_PER_2X1_RASTER_PIXELS             0x00000006
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX4_PS_X1_PER_1X2_RASTER_PIXELS             0x00000007
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX4_PS_X1_PER_2X2_RASTER_PIXELS             0x00000008
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX4_PS_X1_PER_4X2_RASTER_PIXELS             0x00000009
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX4_PS_X1_PER_2X4_RASTER_PIXELS             0x0000000A
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX4_PS_X1_PER_4X4_RASTER_PIXELS             0x0000000B
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX5                                  23:20
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX5_PS_X0_LWLL_RASTER_PIXELS             0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX5_PS_X16_PER_RASTER_PIXEL             0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX5_PS_X8_PER_RASTER_PIXEL             0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX5_PS_X4_PER_RASTER_PIXEL             0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX5_PS_X2_PER_RASTER_PIXEL             0x00000004
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX5_PS_X1_PER_RASTER_PIXEL             0x00000005
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX5_PS_X1_PER_2X1_RASTER_PIXELS             0x00000006
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX5_PS_X1_PER_1X2_RASTER_PIXELS             0x00000007
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX5_PS_X1_PER_2X2_RASTER_PIXELS             0x00000008
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX5_PS_X1_PER_4X2_RASTER_PIXELS             0x00000009
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX5_PS_X1_PER_2X4_RASTER_PIXELS             0x0000000A
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX5_PS_X1_PER_4X4_RASTER_PIXELS             0x0000000B
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX6                                  27:24
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX6_PS_X0_LWLL_RASTER_PIXELS             0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX6_PS_X16_PER_RASTER_PIXEL             0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX6_PS_X8_PER_RASTER_PIXEL             0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX6_PS_X4_PER_RASTER_PIXEL             0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX6_PS_X2_PER_RASTER_PIXEL             0x00000004
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX6_PS_X1_PER_RASTER_PIXEL             0x00000005
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX6_PS_X1_PER_2X1_RASTER_PIXELS             0x00000006
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX6_PS_X1_PER_1X2_RASTER_PIXELS             0x00000007
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX6_PS_X1_PER_2X2_RASTER_PIXELS             0x00000008
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX6_PS_X1_PER_4X2_RASTER_PIXELS             0x00000009
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX6_PS_X1_PER_2X4_RASTER_PIXELS             0x0000000A
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX6_PS_X1_PER_4X4_RASTER_PIXELS             0x0000000B
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX7                                  31:28
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX7_PS_X0_LWLL_RASTER_PIXELS             0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX7_PS_X16_PER_RASTER_PIXEL             0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX7_PS_X8_PER_RASTER_PIXEL             0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX7_PS_X4_PER_RASTER_PIXEL             0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX7_PS_X2_PER_RASTER_PIXEL             0x00000004
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX7_PS_X1_PER_RASTER_PIXEL             0x00000005
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX7_PS_X1_PER_2X1_RASTER_PIXELS             0x00000006
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX7_PS_X1_PER_1X2_RASTER_PIXELS             0x00000007
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX7_PS_X1_PER_2X2_RASTER_PIXELS             0x00000008
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX7_PS_X1_PER_4X2_RASTER_PIXELS             0x00000009
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX7_PS_X1_PER_2X4_RASTER_PIXELS             0x0000000A
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_A_RATE_INDEX7_PS_X1_PER_4X4_RASTER_PIXELS             0x0000000B

#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B(j)                                 (0x2a08+(j)*16)
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX8                                    3:0
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX8_PS_X0_LWLL_RASTER_PIXELS             0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX8_PS_X16_PER_RASTER_PIXEL             0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX8_PS_X8_PER_RASTER_PIXEL             0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX8_PS_X4_PER_RASTER_PIXEL             0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX8_PS_X2_PER_RASTER_PIXEL             0x00000004
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX8_PS_X1_PER_RASTER_PIXEL             0x00000005
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX8_PS_X1_PER_2X1_RASTER_PIXELS             0x00000006
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX8_PS_X1_PER_1X2_RASTER_PIXELS             0x00000007
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX8_PS_X1_PER_2X2_RASTER_PIXELS             0x00000008
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX8_PS_X1_PER_4X2_RASTER_PIXELS             0x00000009
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX8_PS_X1_PER_2X4_RASTER_PIXELS             0x0000000A
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX8_PS_X1_PER_4X4_RASTER_PIXELS             0x0000000B
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX9                                    7:4
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX9_PS_X0_LWLL_RASTER_PIXELS             0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX9_PS_X16_PER_RASTER_PIXEL             0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX9_PS_X8_PER_RASTER_PIXEL             0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX9_PS_X4_PER_RASTER_PIXEL             0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX9_PS_X2_PER_RASTER_PIXEL             0x00000004
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX9_PS_X1_PER_RASTER_PIXEL             0x00000005
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX9_PS_X1_PER_2X1_RASTER_PIXELS             0x00000006
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX9_PS_X1_PER_1X2_RASTER_PIXELS             0x00000007
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX9_PS_X1_PER_2X2_RASTER_PIXELS             0x00000008
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX9_PS_X1_PER_4X2_RASTER_PIXELS             0x00000009
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX9_PS_X1_PER_2X4_RASTER_PIXELS             0x0000000A
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX9_PS_X1_PER_4X4_RASTER_PIXELS             0x0000000B
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX10                                  11:8
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX10_PS_X0_LWLL_RASTER_PIXELS             0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX10_PS_X16_PER_RASTER_PIXEL             0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX10_PS_X8_PER_RASTER_PIXEL             0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX10_PS_X4_PER_RASTER_PIXEL             0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX10_PS_X2_PER_RASTER_PIXEL             0x00000004
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX10_PS_X1_PER_RASTER_PIXEL             0x00000005
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX10_PS_X1_PER_2X1_RASTER_PIXELS             0x00000006
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX10_PS_X1_PER_1X2_RASTER_PIXELS             0x00000007
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX10_PS_X1_PER_2X2_RASTER_PIXELS             0x00000008
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX10_PS_X1_PER_4X2_RASTER_PIXELS             0x00000009
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX10_PS_X1_PER_2X4_RASTER_PIXELS             0x0000000A
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX10_PS_X1_PER_4X4_RASTER_PIXELS             0x0000000B
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX11                                 15:12
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX11_PS_X0_LWLL_RASTER_PIXELS             0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX11_PS_X16_PER_RASTER_PIXEL             0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX11_PS_X8_PER_RASTER_PIXEL             0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX11_PS_X4_PER_RASTER_PIXEL             0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX11_PS_X2_PER_RASTER_PIXEL             0x00000004
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX11_PS_X1_PER_RASTER_PIXEL             0x00000005
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX11_PS_X1_PER_2X1_RASTER_PIXELS             0x00000006
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX11_PS_X1_PER_1X2_RASTER_PIXELS             0x00000007
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX11_PS_X1_PER_2X2_RASTER_PIXELS             0x00000008
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX11_PS_X1_PER_4X2_RASTER_PIXELS             0x00000009
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX11_PS_X1_PER_2X4_RASTER_PIXELS             0x0000000A
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX11_PS_X1_PER_4X4_RASTER_PIXELS             0x0000000B
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX12                                 19:16
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX12_PS_X0_LWLL_RASTER_PIXELS             0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX12_PS_X16_PER_RASTER_PIXEL             0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX12_PS_X8_PER_RASTER_PIXEL             0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX12_PS_X4_PER_RASTER_PIXEL             0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX12_PS_X2_PER_RASTER_PIXEL             0x00000004
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX12_PS_X1_PER_RASTER_PIXEL             0x00000005
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX12_PS_X1_PER_2X1_RASTER_PIXELS             0x00000006
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX12_PS_X1_PER_1X2_RASTER_PIXELS             0x00000007
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX12_PS_X1_PER_2X2_RASTER_PIXELS             0x00000008
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX12_PS_X1_PER_4X2_RASTER_PIXELS             0x00000009
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX12_PS_X1_PER_2X4_RASTER_PIXELS             0x0000000A
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX12_PS_X1_PER_4X4_RASTER_PIXELS             0x0000000B
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX13                                 23:20
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX13_PS_X0_LWLL_RASTER_PIXELS             0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX13_PS_X16_PER_RASTER_PIXEL             0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX13_PS_X8_PER_RASTER_PIXEL             0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX13_PS_X4_PER_RASTER_PIXEL             0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX13_PS_X2_PER_RASTER_PIXEL             0x00000004
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX13_PS_X1_PER_RASTER_PIXEL             0x00000005
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX13_PS_X1_PER_2X1_RASTER_PIXELS             0x00000006
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX13_PS_X1_PER_1X2_RASTER_PIXELS             0x00000007
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX13_PS_X1_PER_2X2_RASTER_PIXELS             0x00000008
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX13_PS_X1_PER_4X2_RASTER_PIXELS             0x00000009
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX13_PS_X1_PER_2X4_RASTER_PIXELS             0x0000000A
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX13_PS_X1_PER_4X4_RASTER_PIXELS             0x0000000B
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX14                                 27:24
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX14_PS_X0_LWLL_RASTER_PIXELS             0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX14_PS_X16_PER_RASTER_PIXEL             0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX14_PS_X8_PER_RASTER_PIXEL             0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX14_PS_X4_PER_RASTER_PIXEL             0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX14_PS_X2_PER_RASTER_PIXEL             0x00000004
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX14_PS_X1_PER_RASTER_PIXEL             0x00000005
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX14_PS_X1_PER_2X1_RASTER_PIXELS             0x00000006
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX14_PS_X1_PER_1X2_RASTER_PIXELS             0x00000007
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX14_PS_X1_PER_2X2_RASTER_PIXELS             0x00000008
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX14_PS_X1_PER_4X2_RASTER_PIXELS             0x00000009
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX14_PS_X1_PER_2X4_RASTER_PIXELS             0x0000000A
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX14_PS_X1_PER_4X4_RASTER_PIXELS             0x0000000B
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX15                                 31:28
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX15_PS_X0_LWLL_RASTER_PIXELS             0x00000000
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX15_PS_X16_PER_RASTER_PIXEL             0x00000001
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX15_PS_X8_PER_RASTER_PIXEL             0x00000002
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX15_PS_X4_PER_RASTER_PIXEL             0x00000003
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX15_PS_X2_PER_RASTER_PIXEL             0x00000004
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX15_PS_X1_PER_RASTER_PIXEL             0x00000005
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX15_PS_X1_PER_2X1_RASTER_PIXELS             0x00000006
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX15_PS_X1_PER_1X2_RASTER_PIXELS             0x00000007
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX15_PS_X1_PER_2X2_RASTER_PIXELS             0x00000008
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX15_PS_X1_PER_4X2_RASTER_PIXELS             0x00000009
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX15_PS_X1_PER_2X4_RASTER_PIXELS             0x0000000A
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE_B_RATE_INDEX15_PS_X1_PER_4X4_RASTER_PIXELS             0x0000000B

#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_RESERVED_A(j)                                      (0x2a0c+(j)*16)
#define LWCC97_SET_VARIABLE_PIXEL_RATE_SHADING_RESERVED_A_V                                                  31:0

#define LWCC97_SET_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_VALUE(i)                                    (0x32f4+(i)*4)
#define LWCC97_SET_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_VALUE_V                                               31:0

#define LWCC97_SET_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_VALUE_UPPER(i)                              (0x3314+(i)*4)
#define LWCC97_SET_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_VALUE_UPPER_V                                         31:0

#define LWCC97_ENABLE_SHADER_PERFORMANCE_SNAPSHOT_COUNTER                                                  0x3334
#define LWCC97_ENABLE_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_V                                                   0:0

#define LWCC97_DISABLE_SHADER_PERFORMANCE_SNAPSHOT_COUNTER                                                 0x3338
#define LWCC97_DISABLE_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_V                                                  0:0

#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_VALUE_UPPER(i)                                       (0x333c+(i)*4)
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_VALUE_UPPER_V                                                  31:0

#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_VALUE(i)                                             (0x335c+(i)*4)
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_VALUE_V                                                        31:0

#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_EVENT(i)                                             (0x337c+(i)*4)
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_EVENT_EVENT                                                     7:0

#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A(i)                                         (0x339c+(i)*4)
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT0                                                1:0
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT0                                           4:2
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT1                                                6:5
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT1                                           9:7
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT2                                              11:10
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT2                                         14:12
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT3                                              16:15
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT3                                         19:17
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT4                                              21:20
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT4                                         24:22
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT5                                              26:25
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT5                                         29:27
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_SPARE                                               31:30

#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_B(i)                                         (0x33bc+(i)*4)
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_B_EDGE                                                  0:0
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_B_MODE                                                  2:1
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_B_WINDOWED                                              3:3
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_B_FUNC                                                 19:4

#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_TRAP_CONTROL                                                 0x33dc
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_TRAP_CONTROL_MASK                                               7:0

#define LWCC97_START_SHADER_PERFORMANCE_COUNTER                                                            0x33e0
#define LWCC97_START_SHADER_PERFORMANCE_COUNTER_COUNTER_MASK                                                  7:0

#define LWCC97_STOP_SHADER_PERFORMANCE_COUNTER                                                             0x33e4
#define LWCC97_STOP_SHADER_PERFORMANCE_COUNTER_COUNTER_MASK                                                   7:0

#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_SCTL_FILTER                                                  0x33e8
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_SCTL_FILTER_V                                                  31:0

#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CORE_MIO_FILTER                                              0x33ec
#define LWCC97_SET_SHADER_PERFORMANCE_COUNTER_CORE_MIO_FILTER_V                                              31:0

#define LWCC97_SET_MME_SHADOW_SCRATCH(i)                                                           (0x3400+(i)*4)
#define LWCC97_SET_MME_SHADOW_SCRATCH_V                                                                      31:0

#define LWCC97_CALL_MME_MACRO(j)                                                                   (0x3800+(j)*8)
#define LWCC97_CALL_MME_MACRO_V                                                                              31:0

#define LWCC97_CALL_MME_DATA(j)                                                                    (0x3804+(j)*8)
#define LWCC97_CALL_MME_DATA_V                                                                               31:0

#endif /* HWFE_HOPPER_B_SW_H */
