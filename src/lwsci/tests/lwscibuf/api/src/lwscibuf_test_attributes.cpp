/*
 * Copyright (c) 2020-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_test_attributes.h"

const char* LwSciBufAttrKeyToString(LwSciBufAttrKey key)
{
    switch (key) {
    case LwSciBufGeneralAttrKey_Types:
        return "LwSciBufGeneralAttrKey_Types";
    case LwSciBufGeneralAttrKey_NeedCpuAccess:
        return "LwSciBufGeneralAttrKey_NeedCpuAccess";
    case LwSciBufGeneralAttrKey_RequiredPerm:
        return "LwSciBufGeneralAttrKey_RequiredPerm";
    case LwSciBufGeneralAttrKey_EnableCpuCache:
        return "LwSciBufGeneralAttrKey_EnableCpuCache";
    case LwSciBufGeneralAttrKey_GpuId:
        return "LwSciBufGeneralAttrKey_GpuId";
    case LwSciBufGeneralAttrKey_CpuNeedSwCacheCoherency:
        return "LwSciBufGeneralAttrKey_CpuNeedSwCacheCoherency";
    case LwSciBufGeneralAttrKey_ActualPerm:
        return "LwSciBufGeneralAttrKey_ActualPerm";
    case LwSciBufGeneralAttrKey_VidMem_GpuId:
        return "LwSciBufGeneralAttrKey_VidMem_GpuId";
    case LwSciBufGeneralAttrKey_EnableGpuCache:
        return "LwSciBufGeneralAttrKey_EnableGpuCache";
    case LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency:
        return "LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency";
    case LwSciBufGeneralAttrKey_EnableGpuCompression:
        return "LwSciBufGeneralAttrKey_EnableGpuCompression";
    case LwSciBufRawBufferAttrKey_Size:
        return "LwSciBufRawBufferAttrKey_Size";
    case LwSciBufRawBufferAttrKey_Align:
        return "LwSciBufRawBufferAttrKey_Align";
    case LwSciBufImageAttrKey_Layout:
        return "LwSciBufImageAttrKey_Layout";
    case LwSciBufImageAttrKey_TopPadding:
        return "LwSciBufImageAttrKey_TopPadding";
    case LwSciBufImageAttrKey_BottomPadding:
        return "LwSciBufImageAttrKey_BottomPadding";
    case LwSciBufImageAttrKey_LeftPadding:
        return "LwSciBufImageAttrKey_LeftPadding";
    case LwSciBufImageAttrKey_RightPadding:
        return "LwSciBufImageAttrKey_RightPadding";
    case LwSciBufImageAttrKey_VprFlag:
        return "LwSciBufImageAttrKey_VprFlag";
    case LwSciBufImageAttrKey_Size:
        return "LwSciBufImageAttrKey_Size";
    case LwSciBufImageAttrKey_Alignment:
        return "LwSciBufImageAttrKey_Alignment";
    case LwSciBufImageAttrKey_PlaneCount:
        return "LwSciBufImageAttrKey_PlaneCount";
    case LwSciBufImageAttrKey_PlaneColorFormat:
        return "LwSciBufImageAttrKey_PlaneColorFormat";
    case LwSciBufImageAttrKey_PlaneColorStd:
        return "LwSciBufImageAttrKey_PlaneColorStd";
    case LwSciBufImageAttrKey_PlaneBaseAddrAlign:
        return "LwSciBufImageAttrKey_PlaneBaseAddrAlign";
    case LwSciBufImageAttrKey_PlaneWidth:
        return "LwSciBufImageAttrKey_PlaneWidth";
    case LwSciBufImageAttrKey_PlaneHeight:
        return "LwSciBufImageAttrKey_PlaneHeight";
    case LwSciBufImageAttrKey_ScanType:
        return "LwSciBufImageAttrKey_ScanType";
    case LwSciBufImageAttrKey_PlaneBitsPerPixel:
        return "LwSciBufImageAttrKey_PlaneBitsPerPixel";
    case LwSciBufImageAttrKey_PlaneOffset:
        return "LwSciBufImageAttrKey_PlaneOffset";
    case LwSciBufImageAttrKey_PlaneDatatype:
        return "LwSciBufImageAttrKey_PlaneDatatype";
    case LwSciBufImageAttrKey_PlaneChannelCount:
        return "LwSciBufImageAttrKey_PlaneChannelCount";
    case LwSciBufImageAttrKey_PlaneSecondFieldOffset:
        return "LwSciBufImageAttrKey_PlaneSecondFieldOffset";
    case LwSciBufImageAttrKey_PlanePitch:
        return "LwSciBufImageAttrKey_PlanePitch";
    case LwSciBufImageAttrKey_PlaneAlignedHeight:
        return "LwSciBufImageAttrKey_PlaneAlignedHeight";
    case LwSciBufImageAttrKey_PlaneAlignedSize:
        return "LwSciBufImageAttrKey_PlaneAlignedSize";
    case LwSciBufImageAttrKey_ImageCount:
        return "LwSciBufImageAttrKey_ImageCount";
    case LwSciBufImageAttrKey_SurfType:
        return "LwSciBufImageAttrKey_SurfType";
    case LwSciBufImageAttrKey_SurfMemLayout:
        return "LwSciBufImageAttrKey_SurfMemLayout";
    case LwSciBufImageAttrKey_SurfSampleType:
        return "LwSciBufImageAttrKey_SurfSampleType";
    case LwSciBufImageAttrKey_SurfBPC:
        return "LwSciBufImageAttrKey_SurfBPC";
    case LwSciBufImageAttrKey_SurfComponentOrder:
        return "LwSciBufImageAttrKey_SurfComponentOrder";
    case LwSciBufImageAttrKey_SurfWidthBase:
        return "LwSciBufImageAttrKey_SurfWidthBase";
    case LwSciBufImageAttrKey_SurfHeightBase:
        return "LwSciBufImageAttrKey_SurfHeightBase";
    case LwSciBufTensorAttrKey_DataType:
        return "LwSciBufTensorAttrKey_DataType";
    case LwSciBufTensorAttrKey_NumDims:
        return "LwSciBufTensorAttrKey_NumDims";
    case LwSciBufTensorAttrKey_SizePerDim:
        return "LwSciBufTensorAttrKey_SizePerDim";
    case LwSciBufTensorAttrKey_AlignmentPerDim:
        return "LwSciBufTensorAttrKey_AlignmentPerDim";
    case LwSciBufTensorAttrKey_StridesPerDim:
        return "LwSciBufTensorAttrKey_StridesPerDim";
    case LwSciBufTensorAttrKey_PixelFormat:
        return "LwSciBufTensorAttrKey_PixelFormat";
    case LwSciBufTensorAttrKey_BaseAddrAlign:
        return "LwSciBufTensorAttrKey_BaseAddrAlign";
    case LwSciBufTensorAttrKey_Size:
        return "LwSciBufTensorAttrKey_Size";
    case LwSciBufArrayAttrKey_DataType:
        return "LwSciBufArrayAttrKey_DataType";
    case LwSciBufArrayAttrKey_Stride:
        return "LwSciBufArrayAttrKey_Stride";
    case LwSciBufArrayAttrKey_Capacity:
        return "LwSciBufArrayAttrKey_Capacity";
    case LwSciBufArrayAttrKey_Size:
        return "LwSciBufArrayAttrKey_Size";
    case LwSciBufArrayAttrKey_Alignment:
        return "LwSciBufArrayAttrKey_Alignment";
    case LwSciBufPyramidAttrKey_NumLevels:
        return "LwSciBufPyramidAttrKey_NumLevels";
    case LwSciBufPyramidAttrKey_Scale:
        return "LwSciBufPyramidAttrKey_Scale";
    case LwSciBufPyramidAttrKey_LevelOffset:
        return "LwSciBufPyramidAttrKey_LevelOffset";
    case LwSciBufPyramidAttrKey_LevelSize:
        return "LwSciBufPyramidAttrKey_LevelSize";
    case LwSciBufPyramidAttrKey_Alignment:
        return "LwSciBufPyramidAttrKey_Alignment";
    default:
        return "Unknown Attribute";
    }
}

const char* LwSciBufInternalAttrKeyToString(LwSciBufInternalAttrKey key)
{
    switch (key) {
    case LwSciBufInternalGeneralAttrKey_EngineArray:
        return "LwSciBufInternalGeneralAttrKey_EngineArray";
    case LwSciBufInternalGeneralAttrKey_MemDomainArray:
        return "LwSciBufInternalGeneralAttrKey_MemDomainArray";
    case LwSciBufInternalImageAttrKey_PlaneGobSize:
        return "LwSciBufInternalImageAttrKey_PlaneGobSize";
    case LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX:
        return "LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX";
    case LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY:
        return "LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY";
    case LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ:
        return "LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ";
    case LwSciBufInternalAttrKey_LwMediaPrivateFirst:
        return "LwSciBufInternalAttrKey_LwMediaPrivateFirst";
    case LwSciBufInternalAttrKey_LwMediaPrivateLast:
        return "LwSciBufInternalAttrKey_LwMediaPrivateLast";
    case LwSciBufInternalAttrKey_DriveworksPrivateFirst:
        return "LwSciBufInternalAttrKey_DriveworksPrivateFirst";
    case LwSciBufInternalAttrKey_DriveworksPrivateLast:
        return "LwSciBufInternalAttrKey_DriveworksPrivateLast";
    default:
        return "Unknown InternalAttribute";
    }
}
