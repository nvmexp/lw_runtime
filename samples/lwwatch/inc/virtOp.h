#ifndef __VIRTOP_H
#define __VIRTOP_H

#include "lwBlockLinear.h"

// Callback for virtual memory accesses
typedef enum VCB_ID
{
    VCB_ID_NONE         = 0,
    VCB_ID_READVIRTUAL  = 1,
    VCB_ID_WRITEVIRTUAL = 2,
    VCB_ID_BLTOLINEAR   = 3,
    VCB_ID_FILLVIRTUAL  = 4,
} VCB_ID;

typedef enum VCBMEMTYPE
{
    MT_GPUVIRTUAL,
    MT_CPUADDRESS,
    MT_PHYSICALADDRESS,
} VCBMEMTYPE;

typedef struct VCB_PARAM
{
    VCB_ID Id;

    VCBMEMTYPE memType;

    BOOL bStatus;
} VCB_PARAM;

typedef struct READVIRTUAL_VCB_PARAM
{
    // Must be first
    VCB_PARAM vcbParam;

    LwU32 lwrOffset;
    LwU8* pData;
    LwU32 bufferSize;
} READVIRTUAL_VCB_PARAM;

LW_STATUS readVirtualCB(LwU64 va, void* buffer, LwU32 length, VCB_PARAM* pParam);
                          
typedef struct WRITEVIRTUAL_VCB_PARAM
{
    // Must be first
    VCB_PARAM vcbParam;

    LwU32 lwrOffset;
    LwU8* pData;
    LwU32 bufferSize;
} WRITEVIRTUAL_VCB_PARAM;

LW_STATUS writeVirtualCB(LwU64 va, void* buffer, LwU32 length, VCB_PARAM* pParam);

typedef struct FILLVIRTUAL_VCB_PARAM
{
    // Must be first
    VCB_PARAM vcbParam;

    LwU32 data;
} FILLVIRTUAL_VCB_PARAM;

LW_STATUS fillVirtualCB(LwU64 va, void* buffer, LwU32 length, VCB_PARAM* pParam);

typedef struct BLTOLINEAR_LWRREAD
{
    LwU32 xBlock, yBlock, zBlock;
    LwU32 xGob, yGob, zGob;
} BLTOLINEAR_LWRREAD;

typedef struct BLTOLINEAR_VCB_PARAM
{
    // Must be first
    VCB_PARAM vcbParam;

    LwU32 gobWidth, gobHeight;
    LwBlockLinearImageInfo blockLinearInfo;
    LwBlockLinearTexParams texParams;

    BLTOLINEAR_LWRREAD lwrRead;
    LwU32 pitch, slicePitch;
    LwU8 *pOutBuf;
    VMemTypes vMemType;
} BLTOLINEAR_VCB_PARAM;

LW_STATUS virtDisplayVirtual(VMemTypes vMemType, LwU32 chId, LwU64 va,
                        LwU32 width, LwU32 height, //No depth since it displays a 2D image
                        LwU32 logBlockWidth, LwU32 logBlockHeight, LwU32 logBlockDepth, //Still need block depth though!
                        LwU32 format, LwU32 gobWidth, LwU32 gobHeight);

typedef LwU32 (*virtualCallback)(LwU64 va, void *buffer, LwU32 length, VCB_PARAM *pParam);

LW_STATUS vmemDoVirtualOp(VMemSpace* pVMemSpace, LwU64 va, LwU32 length, LwU32 isWrite, 
                          virtualCallback virtualCB, VCB_PARAM* pParam);

LwU32 coordToBL(LwU32 cx, LwU32 cy, LwU32 width, LwU32 height,
                LwU32 logBlockWidth, LwU32 logBlockHeight, LwU32 logBlockDepth,
                LwU32 logGobWidth, LwU32 logGobHeight, LwU32 logGobDepth, LwU32 format);

LwU32 blToCoord(LwU32 offsetBL, LwU32 * cx, LwU32 * cy, LwU32 width, LwU32 height,
                LwU32 logBlockWidth, LwU32 logBlockHeight, LwU32 logBlockDepth,
                LwU32 logGobWidth, LwU32 logGobHeight, LwU32 logGobDepth, LwU32 format);

typedef enum _TEXEL_FORMATS
{
    TF_X8R8G8B8,
    TF_A8R8G8B8,
    TF_R5G6B5,
    TF_A1R5G5B5,
    TF_A16B16G16R16F,
    TF_R16F,
    TF_R16UN,
    TF_R32F,
    TF_A32B32G32R32F,
    TF_A2R10G10B10,
    TF_A4R4G4B4,
    TF_S8Z24,
    TF_Z24S8,
    TF_Y8,
    TF_YUY2,
    TF_UYVY,
    TF_A8B8G8R8,
    TF_R11G11B10F,
    TF_LW12,
    TF_LW24,
    TF_YV12,
    TF_UV,
    TF_R32G32F,
    TF_Z24X8_X16V8S8,
    TF_Z32F_X16V8S8,
    TF_DXT1,
    TF_DXT23,
    TF_DXT45,
    TF_AYUV,
    TF_P010,
    TF_YY16,
    TF_UV16,
    TF_COUNT
} TEXEL_FORMATS;

#endif //__VIRTOP_H
