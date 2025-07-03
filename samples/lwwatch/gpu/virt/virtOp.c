/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2005-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "os.h"
#include "vmem.h"
#include "chip.h"
#include "virtOp.h"

#include "class/cle3f1.h"      // TEGRA_VASPACE_A
#include "g00x/g000/dev_master.h" // For chip architecture definitions

//Read 128KB chunks
#define BUFFERSIZE   (0x20*0x1000)

#ifndef MAX
#define MAX(a, b)   (((a) > (b)) ? (a) : (b))
#endif

typedef struct
{
    LwBool bValid;
    LwU64  size;
    LwU64  offset;
} DoVirtualOpArg;

static LW_STATUS _doVirtualOpPteFunc
(
    VMemSpace           *pVMemSpace,
    GMMU_APERTURE        aperture,
    LwU64                va,
    LwU64                entryAddr,
    LwU32                level,
    LwU32                sublevel,
    LwU32                index,
    const MMU_FMT_LEVEL *pFmtLevel,
    const MMU_FMT_PTE   *pFmtPte,
    GMMU_ENTRY_VALUE    *pPte,
    LwBool               valid,
    LwBool              *pDone,
    void                *pArg
)
{
    DoVirtualOpArg *pDoVirtualOpArg = (DoVirtualOpArg*)pArg;

    if (valid)
    {
        pDoVirtualOpArg->bValid = valid;
        pDoVirtualOpArg->size   = mmuFmtLevelPageSize(pFmtLevel);
        pDoVirtualOpArg->offset = va % pDoVirtualOpArg->size;

        *pDone = TRUE;
    }
    else
    {
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

static LW_STATUS _doVirtualOpPdeFunc
(
    VMemSpace           *pVMemSpace,
    GMMU_APERTURE        aperture,
    LwU64                va,
    LwU64                entryAddr,
    LwU32                level,
    LwU32                sublevel,
    LwU32                index,
    const MMU_FMT_LEVEL *pFmtLevel,
    const MMU_FMT_PDE   *pFmtPde,
    GMMU_ENTRY_VALUE    *pPde,
    LwBool               valid,
    LwBool              *pDone,
    void                *pArg
)
{
    DoVirtualOpArg *pDoVirtualOpArg = (DoVirtualOpArg*)pArg;

    if (!valid)
    {
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

//-----------------------------------------------------
// vmemDoVirtualOp
//
//-----------------------------------------------------
LW_STATUS
vmemDoVirtualOp
(
    VMemSpace              *pVMemSpace,
    LwU64                   va,
    LwU32                   length,
    LwU32                   isWrite, //IFB path doesn't handle READ *AND* WRITE since it doesn't reset the RDWR_ADDR register after the read
    virtualCallback         virtualCB,
    VCB_PARAM              *pParam
)
{
    LW_STATUS    status = LW_OK, retStatus = LW_OK;

    LwU64 bar1Offset = 0;
    LwU64 bar1PhysAddr = lwBar1 & ~(BIT(8)-1); //drop the bottom 8 bits (could be non-zero!)

    LwU32 bufferSize = min(length, BUFFERSIZE);
    LwU8* tbuffer;

    LwU32 lwrOffset = 0;
    VMEM_LWHAL_IFACES *pLwrVmem = &pVmem[indexGpu];

    DoVirtualOpArg    args = {0};
    VMemTableWalkInfo info = {0};

    LwU64 pa = 0;
    GMMU_APERTURE aperture = {0};
    LwBool bSysmem = LW_FALSE;

    if (pParam->memType == MT_GPUVIRTUAL && lwBar1 == 0)
    {
        dprintf("lw: %s - lwBar1 is 0 - aborting\n", __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    tbuffer = malloc(bufferSize);
    if (!tbuffer)
    {
        dprintf("lw: %s - failed to allocate temp buffer (size = 0x%08x) - aborting\n",
                    __FUNCTION__, bufferSize);
        return LW_ERR_GENERIC;
    }

    info.pteFunc = _doVirtualOpPteFunc;
    info.pdeFunc = _doVirtualOpPdeFunc;
    info.pArg    = &args;

    while (length > 0)
    {
        LwU32            walkSize;
        LwU32            lengthLeft;

        if (osCheckControlC())
            break;

        args.bValid = FALSE;
        args.size   = 0;
        args.offset = 0;

        status = vmemTableWalk(pVMemSpace, va, &info, LW_FALSE);
        if (status != LW_OK)
            break;

        walkSize   = (LwU32)min(length, args.size - (va % args.size));
        lengthLeft = (LwU32)min(length, walkSize);

        if ((pParam->memType == MT_GPUVIRTUAL) && (pVMemSpace->class != TEGRA_VASPACE_A))
        {
            // We're asked to read GPU virtual, which may reflect back to sysmem, so check.
            if ((status = vmemVToP(pVMemSpace, va, &pa, &aperture, LW_FALSE)) != LW_OK)
            {
                dprintf("lw: %s - vmemVToP failed\n", __FUNCTION__);
                goto exit;
            }
            if ((aperture == GMMU_APERTURE_SYS_NONCOH) || (aperture == GMMU_APERTURE_SYS_COH))
            {
                bSysmem = LW_TRUE;
            }

            if (!bSysmem)
            {
                CHECK_EXIT(vmemBeginBar1Mapping(pVMemSpace, va));
            }

            // PTE changed so reset the offset
            lwrOffset  = 0;
            bar1Offset = args.offset;
        }

        while (lengthLeft > 0)
        {
            LwU32 lwrJobSize = (LwU32)min(bufferSize, lengthLeft);

            if (!isWrite)
            {
                LwU64 sizeRead;
                switch (pParam->memType)
                {
                case MT_GPUVIRTUAL:
                {
                    // Check for non-CheetAh type VA space
                    if (pVMemSpace->class != TEGRA_VASPACE_A)
                    {
                        if (bSysmem)
                        {
                            status = readPhysicalMem(pa + lwrOffset, tbuffer, lwrJobSize, &sizeRead);
                        }   
                        else
                        {                     
                            status = readPhysicalMem(bar1PhysAddr + bar1Offset + lwrOffset, tbuffer,
                                                    lwrJobSize, &sizeRead);
                        }
                    }
                    else
                    {
                        status = pLwrVmem->vmemRead(pVMemSpace, va + lwrOffset, lwrJobSize, tbuffer);
                        sizeRead = lwrJobSize;
                    }
                    break;
                }
                case MT_PHYSICALADDRESS:
                {
                    lwrJobSize = min(lwrJobSize, 4096 * 8);
                    status = readPhysicalMem(va + lwrOffset, tbuffer, lwrJobSize, &sizeRead);
                    break;
                }
                case MT_CPUADDRESS:
                {
                    LwU64 cpuAddr64 = va + lwrOffset;
                    void* pData = (void*)(LwUPtr)cpuAddr64;

                    memcpy(tbuffer, pData, lwrJobSize);
                    sizeRead = lwrJobSize;
                    break;
                }
                default:
                    dprintf("lw: %s: Unknown memtype (%d)\n", __FUNCTION__, pParam->memType);
                    status = LW_ERR_GENERIC;

                }

                if (osCheckControlC())
                    status = LW_ERR_GENERIC;

                if (status == LW_ERR_GENERIC)
                    break;

                if (sizeRead != lwrJobSize)
                {
                    dprintf("lw: %s: sizeRead != requested read size\n", __FUNCTION__);
                    status = LW_ERR_GENERIC;
                    break;
                }
                // If reading/writing/filling a large chunk of memory give user an indication of progress (if requested)
                if (pParam->bStatus)
                {
                    dprintf(".");
                    if (length == lwrJobSize) //last iteration
                        dprintf("\n");
                }
            }

            // Call the callback
            status = virtualCB(va + lwrOffset, tbuffer, lwrJobSize, pParam);
            if (status == LW_ERR_GENERIC)
                break;

            if (isWrite)
            {
                LwU64 sizeWritten;

                switch (pParam->memType)
                {
                case MT_GPUVIRTUAL:
                {
                    // Check for non-CheetAh type VA space
                    if (pVMemSpace->class != TEGRA_VASPACE_A)
                    {
                        if (bSysmem)
                        {
                            status = writePhysicalMem(pa + lwrOffset, tbuffer, lwrJobSize, &sizeWritten);
                        }
                        else
                        {                        
                            status = writePhysicalMem(bar1PhysAddr + bar1Offset + lwrOffset, tbuffer,
                                                      lwrJobSize, &sizeWritten);
                        }
                    }
                    else
                    {
                        status = pLwrVmem->vmemWrite(pVMemSpace, va + lwrOffset, lwrJobSize, tbuffer);
                        sizeWritten = lwrJobSize;
                    }
                    break;
                }
                case MT_PHYSICALADDRESS:
                case MT_CPUADDRESS:
                default:
                    dprintf("lw: %s: memtype not supported - aborting\n", __FUNCTION__);
                    status = LW_ERR_GENERIC;
                    break;
                }

                if (osCheckControlC())
                    status = LW_ERR_GENERIC;

                if (status == LW_ERR_GENERIC)
                    break;

                if (sizeWritten != lwrJobSize)
                {
                    dprintf("lw: %s: sizeWritten != requested write size\n", __FUNCTION__);
                    status = LW_ERR_GENERIC;
                    break;
                }
            }

            lengthLeft -= lwrJobSize;
            lwrOffset  += lwrJobSize;
        }

        retStatus = status;

        if (pParam->memType == MT_GPUVIRTUAL)
        {
            // No IFB or BAR1 mapping required for CheetAh VA space
            assert(pVMemSpace->class != TEGRA_VASPACE_A);
            CHECK_EXIT(vmemEndBar1Mapping(pVMemSpace, va));
        }

        if (retStatus != LW_OK)
        {
            status = retStatus;
            break;
        }
        length -= walkSize;
        va     += walkSize;

    }
    retStatus = status;

exit:

    free(tbuffer);

    return retStatus;
}

//-----------------------------------------------------
// readVirtualCB
//
//-----------------------------------------------------

LW_STATUS readVirtualCB(LwU64 va, void* buffer, LwU32 length, VCB_PARAM* pParam)
{
    READVIRTUAL_VCB_PARAM* pReadVirtualParam = (READVIRTUAL_VCB_PARAM*) pParam;
    LwU32 spaceLeft = pReadVirtualParam->bufferSize - pReadVirtualParam->lwrOffset;
    LwU32 readLen = min(spaceLeft, length);

    memcpy(pReadVirtualParam->pData+pReadVirtualParam->lwrOffset, buffer, readLen);

    pReadVirtualParam->lwrOffset += readLen;

    return LW_OK;
}

//-----------------------------------------------------
// writeVirtualCB
//
//-----------------------------------------------------

LW_STATUS writeVirtualCB(LwU64 va, void* buffer, LwU32 length, VCB_PARAM* pParam)
{
    WRITEVIRTUAL_VCB_PARAM* pWriteVirtualParam = (WRITEVIRTUAL_VCB_PARAM*) pParam;
    LwU32 spaceLeft = pWriteVirtualParam->bufferSize - pWriteVirtualParam->lwrOffset;
    LwU32 writeLen = min(spaceLeft, length);

    memcpy(buffer, pWriteVirtualParam->pData+pWriteVirtualParam->lwrOffset, writeLen);

    pWriteVirtualParam->lwrOffset += writeLen;

    return LW_OK;
}

//-----------------------------------------------------
// fillVirtualCB
//
//-----------------------------------------------------
LW_STATUS fillVirtualCB(LwU64 va, void* buffer, LwU32 length, VCB_PARAM* pParam)
{
    LwU32 i;
    FILLVIRTUAL_VCB_PARAM* fillVirtual = (FILLVIRTUAL_VCB_PARAM*) pParam;
    LwU32* pData;

    if (!fillVirtual || fillVirtual->vcbParam.Id != VCB_ID_FILLVIRTUAL)
    {
        dprintf("lw: Bad param for %s\n", __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    pData = (LwU32*) buffer;

    for (i = 0; i < length/4; ++i)
    {
        pData[i] = fillVirtual->data;
    }

    return LW_OK;
}

//-----------------------------------------------------
// virtDisplayVirtual
//
//-----------------------------------------------------
#define LW_DISPLAYVIRTUAL_FORMAT_TYPE                           5:0
//Use TEXEL_FORMATS enum

#define LW_DISPLAYVIRTUAL_FORMAT_DISPLAY                      31:30
#define LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_COLOR           0x00000000
#define LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_DEPTH           0x00000000
#define LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_ALPHA           0x00000002
#define LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_STENCIL         0x00000002
#define LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_VCAA            0x00000001

#define LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_PIXEL                24:24
#define LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_PIXEL_EVEN      0x00000000
#define LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_PIXEL_ODD       0x00000001

#define BMP_SOI                 (4)

#define ILWALID_PACKET_SIZE     (~0)

#define MAKE_DWORD(a, r, g, b)          \
    ((((a) & 0xff) << 24) | (((r) & 0xff) << 16) | (((g) & 0xff) << 8) | (((b) & 0xff) << 0))

static LW_STATUS ColwertFromARGB8(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU32* pInFormatBuffer = (LwU32*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;
    LwU32 valA;

    switch (DRF_VAL(_DISPLAYVIRTUAL, _FORMAT, _DISPLAY, flags))
    {
    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_ALPHA:
        valA = (*pInFormatBuffer >> 24) & 0xff;
        *pOutFormatBuffer = MAKE_DWORD(valA, valA, valA, valA);
        break;

    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_COLOR:
        *pOutFormatBuffer = *pInFormatBuffer;
        break;
    }

    return LW_OK;
}

static LW_STATUS ColwertFromR5G6B5(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU16* pInFormatBuffer = (LwU16*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;
    float fValR, fValG, fValB;

    fValB = (float) ((*pInFormatBuffer >>  0) & 0x1f) / 32.0f;
    fValG = (float) ((*pInFormatBuffer >>  5) & 0x3f) / 64.0f;
    fValR = (float) ((*pInFormatBuffer >> 11) & 0x1f) / 32.0f;
    *pOutFormatBuffer = MAKE_DWORD(0,
                                   (LwU32) (fValR * 255.0f),
                                   (LwU32) (fValG * 255.0f),
                                   (LwU32) (fValB * 255.0f));
    return LW_OK;
}


static LW_STATUS ColwertFromA1R5G5B5(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU16* pInFormatBuffer = (LwU16*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;
    float fValA = 0.0f, fValR = 0.0f, fValG = 0.0f, fValB = 0.0f;

    switch (DRF_VAL(_DISPLAYVIRTUAL, _FORMAT, _DISPLAY, flags))
    {
    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_ALPHA:
        fValA = (float) ((*pInFormatBuffer >> 15) & 0x1);
        fValR = fValG = fValB = fValA;
        break;

    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_COLOR:
        fValR = (float) ((*pInFormatBuffer >>  0) & 0x1f) / 32.0f;
        fValG = (float) ((*pInFormatBuffer >>  5) & 0x1f) / 32.0f;
        fValB = (float) ((*pInFormatBuffer >> 10) & 0x1f) / 32.0f;
        fValA = (float) ((*pInFormatBuffer >> 15) & 0x1);
        break;
    }

    *pOutFormatBuffer = MAKE_DWORD((LwU32) (fValA * 255.0f),
                                   (LwU32) (fValR * 255.0f),
                                   (LwU32) (fValG * 255.0f),
                                   (LwU32) (fValB * 255.0f));
    return LW_OK;
}

static LW_STATUS ColwertFromR16UN(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU16* pInFormatBuffer = (LwU16*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;
    float fValR;

    fValR = ((float) pInFormatBuffer[0]) / 65535.0f;
    *pOutFormatBuffer = MAKE_DWORD(0xff,
                                   (LwU32) (fValR * 255.0f),
                                   0,
                                   0);

    return LW_OK;
}

#define LWEXT_FLOAT_SIGN        31:31
#define LWEXT_FLOAT_SIGN_POS        0
#define LWEXT_FLOAT_SIGN_NEG        1
#define LWEXT_FLOAT_EXP         30:23
#define LWEXT_FLOAT_EXP_BIAS      127
#define LWEXT_FLOAT_MANTISSA     22:0

#define LWEXT_HFLOAT_SIGN        15:15
#define LWEXT_HFLOAT_SIGN_POS        0
#define LWEXT_HFLOAT_SIGN_NEG        1
#define LWEXT_HFLOAT_EXP         14:10
#define LWEXT_HFLOAT_EXP_BIAS       15
#define LWEXT_HFLOAT_MANTISSA      9:0

static float HalfFloatToFloat(LwU16 HalfFloat)
{
    LwU32 value;

    if (DRF_VAL(EXT, _HFLOAT, _EXP, HalfFloat) == DRF_MASK(LWEXT_HFLOAT_EXP) &&
        DRF_VAL(EXT, _HFLOAT, _MANTISSA, HalfFloat) == DRF_MASK(LWEXT_HFLOAT_MANTISSA))
    {
        if (FLD_TEST_DRF(EXT, _HFLOAT, _SIGN, _POS, HalfFloat))
        {
            value = DRF_DEF(EXT, _FLOAT, _SIGN, _POS) |
                    DRF_SHIFTMASK(LWEXT_FLOAT_EXP) |
                    DRF_NUM(EXT, _FLOAT, _MANTISSA, 0);
        }
        else
        {
            value = DRF_DEF(EXT, _FLOAT, _SIGN, _NEG) |
                    DRF_SHIFTMASK(LWEXT_FLOAT_EXP) |
                    DRF_NUM(EXT, _FLOAT, _MANTISSA, 0);
        }
    }
/*
    //No denorm!
    else if (DRF_VAL(EXT, _FLOAT, _EXP, HalfFloat) == 0 &&
             DRF_VAL(EXT, _FLOAT, _MANTISSA, HalfFloat) != 0)
    {
        dprintf("Denorm\n");
    }
*/
    else
    {
        if (DRF_VAL(EXT, _HFLOAT, _EXP, HalfFloat) == 0 &&
            DRF_VAL(EXT, _HFLOAT, _MANTISSA, HalfFloat) == 0)
        {
            if (FLD_TEST_DRF(EXT, _HFLOAT, _SIGN, _POS, HalfFloat))
            {
                value = DRF_DEF(EXT, _FLOAT, _SIGN, _POS) |
                        DRF_NUM(EXT, _FLOAT, _EXP, 0) |
                        DRF_NUM(EXT, _FLOAT, _MANTISSA, 0);
            }
            else
            {
                value = DRF_DEF(EXT, _FLOAT, _SIGN, _NEG) |
                        DRF_NUM(EXT, _FLOAT, _EXP, 0) |
                        DRF_NUM(EXT, _FLOAT, _MANTISSA, 0);
            }
        }
        else
        {
            LwU32 exp, mant;

            value = FLD_TEST_DRF(EXT, _HFLOAT, _SIGN, _NEG, HalfFloat) ?
                                    DRF_DEF(EXT, _FLOAT, _SIGN, _NEG) :
                                    DRF_DEF(EXT, _FLOAT, _SIGN, _POS);
            exp = DRF_VAL(EXT, _HFLOAT, _EXP, HalfFloat);
            mant = DRF_VAL(EXT, _HFLOAT, _MANTISSA, HalfFloat);
            exp += LWEXT_FLOAT_EXP_BIAS - LWEXT_HFLOAT_EXP_BIAS;
            mant <<= DRF_SIZE(LWEXT_FLOAT_MANTISSA) - DRF_SIZE(LWEXT_HFLOAT_MANTISSA);
            value |= DRF_NUM(EXT, _FLOAT, _EXP, exp) |
                     DRF_NUM(EXT, _FLOAT, _MANTISSA, mant);
        }
    }

    return *((float*)&value);
}

static LW_STATUS ColwertFromA16B16G16R16F(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU16* pInFormatBuffer = (LwU16*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;
    float fValA, fValR, fValG, fValB;

    LwU8 showAlpha = DRF_VAL(_DISPLAYVIRTUAL, _FORMAT, _DISPLAY, flags) == LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_ALPHA;
    fValA = HalfFloatToFloat(pInFormatBuffer[3]);
    if (showAlpha)
        fValB = fValG = fValR = fValA;
    else
    {
        fValR = HalfFloatToFloat(pInFormatBuffer[0]);
        fValG = HalfFloatToFloat(pInFormatBuffer[1]);
        fValB = HalfFloatToFloat(pInFormatBuffer[2]);
    }
    *pOutFormatBuffer = MAKE_DWORD((LwU32) (fValA * 255.0f),
                                   (LwU32) (fValR * 255.0f),
                                   (LwU32) (fValG * 255.0f),
                                   (LwU32) (fValB * 255.0f));

    return LW_OK;
}

static LW_STATUS ColwertFromR16F(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU16* pInFormatBuffer = (LwU16*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;
    float fValR;

    fValR = HalfFloatToFloat(pInFormatBuffer[0]);
    *pOutFormatBuffer = MAKE_DWORD(0xff,
                                   (LwU32) (fValR * 255.0f),
                                   0,
                                   0);

    return LW_OK;
}

static LW_STATUS ColwertFromR32F(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    float* pInFormatBuffer = (float*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;
    float fValR = *pInFormatBuffer;
    LwU32 iValR = (LwU32) (fValR * 255.0f);
    *pOutFormatBuffer = MAKE_DWORD(0, iValR, iValR, iValR);

    return LW_OK;
}

static LW_STATUS ColwertFromR32G32F(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    float* pInFormatBuffer = (float*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;
    float fValR, fValG;

    fValR = pInFormatBuffer[0];
    fValG = pInFormatBuffer[1];
    *pOutFormatBuffer = MAKE_DWORD(0, (LwU32) (fValR * 255.0f), (LwU32) (fValG * 255.0f), 0);

    return LW_OK;
}

static LW_STATUS ColwertFromA32B32G32R32F(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    float* pInFormatBuffer = (float*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;
    float fValA, fValR, fValG, fValB;

    LwU8 showAlpha = DRF_VAL(_DISPLAYVIRTUAL, _FORMAT, _DISPLAY, flags) == LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_ALPHA;

    fValA = pInFormatBuffer[3];
    if (showAlpha)
        fValB = fValG = fValR = fValA;
    else
    {
        fValR = pInFormatBuffer[0];
        fValG = pInFormatBuffer[1];
        fValB = pInFormatBuffer[2];
    }
    *pOutFormatBuffer = MAKE_DWORD((LwU32) (fValA * 255.0f),
                                   (LwU32) (fValR * 255.0f),
                                   (LwU32) (fValG * 255.0f),
                                   (LwU32) (fValB * 255.0f));
    return LW_OK;
}

static LW_STATUS ColwertFromA2R10G10B10(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU32* pInFormatBuffer = (LwU32*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;

    float fValA, fValR, fValG, fValB;
    LwU8 showAlpha = DRF_VAL(_DISPLAYVIRTUAL, _FORMAT, _DISPLAY, flags) == LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_ALPHA;

    fValA = (float) ((*pInFormatBuffer >> 30) & 0x003) / 3.0f;
    if (showAlpha)
        fValB = fValG = fValR = fValA;
    else
    {
        fValR = (float) ((*pInFormatBuffer >> 20) & 0x3ff) / 1023.0f;
        fValG = (float) ((*pInFormatBuffer >> 10) & 0x3ff) / 1023.0f;
        fValB = (float) ((*pInFormatBuffer >>  0) & 0x3ff) / 1023.0f;
    }
    *pOutFormatBuffer = MAKE_DWORD(0,
                                   (LwU32) (fValR * 255.0f),
                                   (LwU32) (fValG * 255.0f),
                                   (LwU32) (fValB * 255.0f));

    return LW_OK;
}

static LW_STATUS ColwertFromA4R4G4B4(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU16* pInFormatBuffer = (LwU16*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;

    float fValA, fValR, fValG, fValB;
    LwU8 showAlpha = DRF_VAL(_DISPLAYVIRTUAL, _FORMAT, _DISPLAY, flags) == LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_ALPHA;

    fValA = (float) ((*pInFormatBuffer >> 12) & 0xf) / 15.0f;
    if (showAlpha)
        fValB = fValG = fValR = fValA;
    else
    {
        fValR = (float) ((*pInFormatBuffer >>  8) & 0xf) / 15.0f;
        fValG = (float) ((*pInFormatBuffer >>  4) & 0xf) / 15.0f;
        fValB = (float) ((*pInFormatBuffer >>  0) & 0xf) / 15.0f;
    }
    *pOutFormatBuffer = MAKE_DWORD((LwU32) (fValA * 255.0f),
                                   (LwU32) (fValR * 255.0f),
                                   (LwU32) (fValG * 255.0f),
                                   (LwU32) (fValB * 255.0f));

    return LW_OK;
}

static float f10f11tof32(LwU32 exponent, LwU32 mantissa, LwU32 mantissaShift)
{
    LwU32 retval;

    if (0 == exponent)
    {
        if (0 == mantissa)
            retval = 0;                     //zero
        else
            retval = (mantissa << mantissaShift);  //denorm
    }
    else if (31 == exponent)
    {
        if (0 == mantissa)
            retval = 0x7f800000;            //infinity
        else
            retval = 0x7fffffff;            //NaN
    }
    else
    {
        retval = ((exponent + 112) << 23) | (mantissa << mantissaShift);
    }

    return *(float*)&retval;
}

static float f10tof32(LwU32 float10)
{
    const LwU32 exp  = (float10 & 0x3e0) >> 5;
    const LwU32 mant = (float10 & 0x01f);
    return f10f11tof32(exp, mant, 18);
}

static float f11tof32(LwU32 float11)
{
    const LwU32 exp  = (float11 & 0x7c0) >> 6;
    const LwU32 mant = (float11 & 0x03f);
    return f10f11tof32(exp, mant, 17);
}

static LW_STATUS ColwertFromR11G11B10(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU32* pInFormatBuffer = (LwU32*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;

    float fValR, fValG, fValB;

    fValR = f11tof32((*pInFormatBuffer)       & 0x7ff);
    fValG = f11tof32((*pInFormatBuffer >> 11) & 0x7ff);
    fValB = f10tof32((*pInFormatBuffer >> 22) & 0x3ff);

    pOutFormatBuffer[0]  = MAKE_DWORD(                   0,
                                   (LwU32) (fValR * 255.0f),
                                   (LwU32) (fValG * 255.0f),
                                   (LwU32) (fValB * 255.0f));

    return LW_OK;
}

static LW_STATUS ColwertFromS8Z24(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU32* pInFormatBuffer = (LwU32*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;
    LwU32 val = 0;

    switch (DRF_VAL(_DISPLAYVIRTUAL, _FORMAT, _DISPLAY, flags))
    {
    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_STENCIL:
    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_VCAA:
        val = (*pInFormatBuffer >> 24) & 0xff;
        break;

    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_DEPTH:
        val = (LwU32) (((float) (*pInFormatBuffer & 0x00ffffff) / ((1 << 24)-1)) * 0xff);
        break;
    }
    *pOutFormatBuffer = MAKE_DWORD(0xff, val, val, val);

    return LW_OK;
}

static LW_STATUS ColwertFromZ24S8(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU32* pInFormatBuffer = (LwU32*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;
    LwU32 val = 0;

    switch (DRF_VAL(_DISPLAYVIRTUAL, _FORMAT, _DISPLAY, flags))
    {
    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_STENCIL:
    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_VCAA:
        val = *pInFormatBuffer & 0xff;
        break;

    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_DEPTH:
        val = (LwU32) (((float) ((*pInFormatBuffer >> 8) & 0x00ffffff) / ((1 << 24)-1)) * 0xff);
        break;
    }
    *pOutFormatBuffer = MAKE_DWORD(0xff, val, val, val);

    return LW_OK;
}

static LW_STATUS ColwertFromZ24X8_X16V8S8(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU32* pInFormatBuffer = (LwU32*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;
    LwU32 val = 0;

    switch (DRF_VAL(_DISPLAYVIRTUAL, _FORMAT, _DISPLAY, flags))
    {
    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_VCAA:
        val = (pInFormatBuffer[1] >> 8) & 0xff;
        break;
    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_STENCIL:
        val = (pInFormatBuffer[1] & 0xff);
        break;
    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_DEPTH:
        val = (LwU32) (((float) ((pInFormatBuffer[0] >> 8) & 0x00ffffff) / ((1 << 24)-1)) * 0xff);
        break;
    }
    *pOutFormatBuffer = MAKE_DWORD(0xff, val, val, val);

    return LW_OK;
}

static LW_STATUS ColwertFromZ32F_X16V8S8(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU32* pInFormatBuffer = (LwU32*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;
    LwU32 val = 0;

    switch (DRF_VAL(_DISPLAYVIRTUAL, _FORMAT, _DISPLAY, flags))
    {
    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_VCAA:
        val = (pInFormatBuffer[1] >> 8) & 0xff;
        break;
    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_STENCIL:
        val = (pInFormatBuffer[1] & 0xff);
        break;
    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_DEPTH:
        val = (LwU32)(*(float*)pInFormatBuffer * 255.0f);
        break;
    }

    *pOutFormatBuffer = MAKE_DWORD(0xff, val, val, val);

    return LW_OK;
}

static LwU32 ColwertYUVToRGB(LwU8 Y, LwU8 U, LwU8 V)
{
    float C = (float)Y-16;
    float D = (float)U-128;
    float E = (float)V-128;

    LwU8 R = (LwU8)min(MAX(((LwS32)( 298 * C           + 409 * E + 128)),0) >> 8,255);
    LwU8 G = (LwU8)min(MAX(((LwS32)( 298 * C - 100 * D - 208 * E + 128)),0) >> 8,255);
    LwU8 B = (LwU8)min(MAX(((LwS32)( 298 * C + 516 * D           + 128)),0) >> 8,255);

    return MAKE_DWORD(0xff, R, G, B);
}

static LW_STATUS ColwertFromUYVY(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    BOOL bEven = DRF_VAL(_DISPLAYVIRTUAL, _FORMAT, _DISPLAY_PIXEL, flags) == LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_PIXEL_EVEN;

    static LwU8 y1Val;
    static LwU8 uVal;

    if (bEven)
    {
        uVal  = pInBuffer[0];
        y1Val = pInBuffer[1];
    }
    else
    {
        LwU32 *pOut = (LwU32 *)pOutBuffer;
        LwU8 vVal  = pInBuffer[0];
        LwU8 y2Val = pInBuffer[1];

        *(pOut-1) = ColwertYUVToRGB(y1Val, uVal, vVal);
        *pOut     = ColwertYUVToRGB(y2Val, uVal, vVal);
    }

    return LW_OK;
}

static LW_STATUS ColwertFromYUY2(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    BOOL bEven = DRF_VAL(_DISPLAYVIRTUAL, _FORMAT, _DISPLAY_PIXEL, flags) == LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_PIXEL_EVEN;

    static LwU8 y1Val;
    static LwU8 uVal;

    if (bEven)
    {
        y1Val = pInBuffer[0];
        uVal  = pInBuffer[1];
    }
    else
    {
        LwU32 *pOut = (LwU32 *)pOutBuffer;
        LwU8 y2Val = pInBuffer[0];
        LwU8 vVal  = pInBuffer[1];

        *(pOut-1) = ColwertYUVToRGB(y1Val, uVal, vVal);
        *pOut     = ColwertYUVToRGB(y2Val, uVal, vVal);
    }

    return LW_OK;
}

static LW_STATUS ColwertFromUV(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU8 yVal = 0xc0; //
    LwU8 vVal = pInBuffer[0];
    LwU8 uVal = pInBuffer[1];

    LwU32 *pOut = (LwU32 *)pOutBuffer;

    *pOut = ColwertYUVToRGB(yVal, uVal, vVal);

    return LW_OK;
}

static LW_STATUS ColwertFromY8(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU8 yVal = *pInBuffer;

    // assuming u and v is 0x80.

    LwU32 *pOut = (LwU32 *)pOutBuffer;

    float C = (float)yVal-16;

    LwU8 R = (LwU8)min(((LwU32)( 298 * C + 128)) >> 8,255);
    LwU8 G = (LwU8)min(((LwU32)( 298 * C + 128)) >> 8,255);
    LwU8 B = (LwU8)min(((LwU32)( 298 * C + 128)) >> 8,255);

    *pOut = MAKE_DWORD(0xff, R, G, B);

    return LW_OK;
}

static LW_STATUS ColwertFromUV16(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU16 *pChroma = (LwU16 *) pInBuffer;
    LwU8 yVal = 0xc0; //
    LwU8 vVal = (LwU8) (pChroma[0] >> 8);
    LwU8 uVal = (LwU8) (pChroma[1] >> 8);

    LwU32 *pOut = (LwU32 *)pOutBuffer;

    *pOut = ColwertYUVToRGB(yVal, uVal, vVal);

    return LW_OK;
}

static LW_STATUS ColwertFromY16(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU16 *pLuma = (LwU16 *) pInBuffer;
    LwU8 yVal = (LwU8) ((*pLuma) >> 8);

    // assuming u and v is 0x80.

    LwU32 *pOut = (LwU32 *)pOutBuffer;

    float C = (float)yVal-16;

    LwU8 R = (LwU8)min(((LwU32)( 298 * C + 128)) >> 8,255);
    LwU8 G = (LwU8)min(((LwU32)( 298 * C + 128)) >> 8,255);
    LwU8 B = (LwU8)min(((LwU32)( 298 * C + 128)) >> 8,255);

    *pOut = MAKE_DWORD(0xff, R, G, B);

    return LW_OK;
}

static LW_STATUS ColwertFromYUV420(LwU8* pInBufferY, LwU8* pInBufferUV, LwU8* pOutBuffer, LwU32 width, LwU32 height, LwU32 inPitch, LwU32 outPitch, LwU32 format, LwU32 chromaSize)
{

    LwU32 w, h, c, chromaInc, chromaPitch, lumaFieldPitch;
    LwU8 *pLuma, *pLumaTopField, *pLumaBtmField, *pChromaU, *pChromaV,  *pChromaUTop, *pChromaVTop, *pChromaUBtm, *pChromaVBtm, *pbRGBOut, *pbOut, y1, y2, u, v;
    LwU32 *pdwRGBOut;
    BOOL bFieldChroma = FALSE;


    pbRGBOut = pbOut = pOutBuffer;
    pLuma = pLumaTopField = pInBufferY;
    pLumaBtmField = pLumaTopField + inPitch;
    pChromaUBtm = pChromaUTop = pChromaU = pInBufferUV;
    pChromaVBtm = pChromaVTop = pChromaV = pChromaU+1;
    chromaPitch = inPitch;
    lumaFieldPitch = inPitch << 1;
    chromaInc = 2;

    if (format == TF_YV12)
    {
        chromaInc = 1;
        pChromaVBtm = pChromaVTop = pChromaV = pInBufferUV;
        pChromaUBtm = pChromaUTop = pChromaU = pInBufferUV + (chromaSize >> 1);
        chromaPitch >>= 1;
    }

    if (format == TF_LW24)
    {
        pLumaBtmField = pLumaTopField + inPitch * (height >> 1);
        lumaFieldPitch = inPitch;
        pChromaUBtm = pChromaUTop + (chromaSize >> 1);
        pChromaVBtm = pChromaUBtm + 1;
        bFieldChroma = TRUE;
    }

    for (h = 0; h < height; h++)
    {
        pdwRGBOut = (LwU32 *) pbRGBOut;

        for (w =0, c=0; w < width; w+=2, c +=chromaInc)
        {

            y1 = pLuma[w];
            y2 = pLuma[w+1];

            u = pChromaU[c];
            v = pChromaV[c];

            *pdwRGBOut++ = ColwertYUVToRGB(y1, u, v);
            *pdwRGBOut++ = ColwertYUVToRGB(y2, u, v);
        }

        if (h & 1)
        {
            pLumaTopField += lumaFieldPitch;
            pLumaBtmField += lumaFieldPitch;

            if (!bFieldChroma || ((h & 0x3) == 0x03))
            {
                pChromaUTop += chromaPitch;
                pChromaVTop += chromaPitch;

                pChromaUBtm += chromaPitch;
                pChromaVBtm += chromaPitch;
            }

            pChromaU = pChromaUTop;
            pChromaV = pChromaVTop;

            pLuma = pLumaTopField;
        }
        else
        {
            pLuma = pLumaBtmField;
            pChromaU = pChromaUBtm;
            pChromaV = pChromaVBtm;
        }

        pbRGBOut += outPitch;
    }


    return LW_OK;
}

static LW_STATUS ColwertFromP010(LwU8* pInBufferY, LwU8* pInBufferUV, LwU8* pOutBuffer, LwU32 width, LwU32 height, LwU32 inPitch, LwU32 outPitch, LwU32 format, LwU32 chromaSize)
{

    LwU32 w, h, c, chromaInc, chromaPitch, lumaFieldPitch;
    LwU8  *pbRGBOut, *pLumaTopField, *pLumaBtmField, *pChromaUTop, *pChromaVTop, *pChromaUBtm, *pChromaVBtm;
    LwU16 *pLuma, *pChromaU, *pChromaV, y1, y2, u, v;
    LwU32 *pdwRGBOut;
    LwU8  ya, yb, u8, v8;
    BOOL bFieldChroma = FALSE;

    void *pInY, *pInUV;

    pInY = pInBufferY;
    pInUV = pInBufferUV;

    pbRGBOut = pOutBuffer;
    pLumaTopField = pInY;
    pLuma = (LwU16*) pLumaTopField;
    pLumaBtmField = pLumaTopField + inPitch;
    pChromaUBtm = pChromaUTop = pInUV;
    pChromaU = (LwU16 *)pInUV;
    pChromaV = pChromaU+1;
    pChromaVBtm = pChromaVTop = (LwU8 *) pChromaV;
    chromaPitch = inPitch;
    lumaFieldPitch = inPitch << 1;
    chromaInc = 2;

    for (h = 0; h < height; h++)
    {
        pdwRGBOut = (LwU32 *) pbRGBOut;

        for (w =0, c=0; w < width; w+=2, c +=chromaInc)
        {
            y1 = pLuma[w];
            y2 = pLuma[w+1];

            u = pChromaU[c];
            v = pChromaV[c];

            ya = (LwU8) (y1 >> 8);
            yb = (LwU8) (y2 >> 8);
            u8 = (LwU8) (u >> 8);
            v8 = (LwU8) (v >> 8);

            *pdwRGBOut++ = ColwertYUVToRGB(ya, u8, v8);
            *pdwRGBOut++ = ColwertYUVToRGB(yb, u8, v8);
        }

        if (h & 1)
        {
            pLumaTopField += lumaFieldPitch;
            pLumaBtmField += lumaFieldPitch;

            if (!bFieldChroma || ((h & 0x3) == 0x03))
            {
                pChromaUTop += chromaPitch;
                pChromaVTop += chromaPitch;

                pChromaUBtm += chromaPitch;
                pChromaVBtm += chromaPitch;
            }

            pChromaU = (LwU16 *) pChromaUTop;
            pChromaV = (LwU16 *) pChromaVTop;

            pLuma = (LwU16 *) pLumaTopField;
        }
        else
        {
            pLuma = (LwU16 *) pLumaBtmField;
            pChromaU = (LwU16 *) pChromaUBtm;
            pChromaV = (LwU16 *) pChromaVBtm;
        }

        pbRGBOut += outPitch;
    }


    return LW_OK;
}

static LW_STATUS ColwertFromAYUV(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{

    LwU32 valAYUV = *(LwU32*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;
    LwU32 valA;
    LwU8  valY, valU, valV;

    valA = valAYUV  & 0xff000000;
    valY = (LwU8) (valAYUV >> 16);
    valU = (LwU8) (valAYUV >> 8);
    valV = (LwU8) valAYUV;

    if (valA)
    {
        *pOutFormatBuffer = ColwertYUVToRGB(valY, valU, valV) | valA;
    }
    else
    {
        *pOutFormatBuffer = 0;
    }

    return LW_OK;
}

static LW_STATUS ColwertFromABGR8(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU32* pInFormatBuffer = (LwU32*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;
    LwU32 valA, valB, valG, valR;

    valA = (*pInFormatBuffer >> 24) & 0xff;
    switch (DRF_VAL(_DISPLAYVIRTUAL, _FORMAT, _DISPLAY, flags))
    {
    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_ALPHA:
        valR = valG = valB = valA;
        break;

    case LW_DISPLAYVIRTUAL_FORMAT_DISPLAY_COLOR:
        valR = (*pInFormatBuffer >>  0) & 0xff;
        valG = (*pInFormatBuffer >>  8) & 0xff;
        valB = (*pInFormatBuffer >> 16) & 0xff;
        break;

    default:
        valR = valG = valB = valA = 0;
        break;
    }

    *pOutFormatBuffer = MAKE_DWORD(valA, valR, valG, valB);

    return LW_OK;
}

static LW_STATUS ColwertFromA8(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU8 val = *pInBuffer;

    LwU32 *pOut = (LwU32 *)pOutBuffer;

    *pOut = MAKE_DWORD(0xff, val, val, val);

    return LW_OK;
}

static LW_STATUS ColwertFromR8G8(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU32* pInFormatBuffer = (LwU32*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;
    LwU32 valG, valR;

    valR = (*pInFormatBuffer >>  8) & 0xff;
    valG = (*pInFormatBuffer >>  0) & 0xff;

    *pOutFormatBuffer = MAKE_DWORD(0xff, valR, valG, 0x0);

    return LW_OK;
}

static LW_STATUS ColwertFromR16(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU16* pInFormatBuffer = (LwU16*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;

    LwU16 valR = pInFormatBuffer[0];

    *pOutFormatBuffer = MAKE_DWORD(0xff, valR, valR, valR);

    return LW_OK;
}

static LW_STATUS ColwertFromR16G16(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU16* pInFormatBuffer = (LwU16*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;

    LwU16 valR = pInFormatBuffer[0];
    LwU16 valG = pInFormatBuffer[1];

    *pOutFormatBuffer = MAKE_DWORD(0xff, valR, valG, 0);

    return LW_OK;
}

static LW_STATUS ColwertFromR16G16F(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU16* pInFormatBuffer = (LwU16*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;

    float fValR = HalfFloatToFloat(pInFormatBuffer[0]);
    float fValG = HalfFloatToFloat(pInFormatBuffer[1]);

    *pOutFormatBuffer = MAKE_DWORD(0xff,
                                   (LwU32)(fValR * 255.0f),
                                   (LwU32)(fValG * 255.0f),
                                   0);

    return LW_OK;
}

static LW_STATUS ColwertFromR32(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU32 val = *pInBuffer;

    LwU32 *pOut = (LwU32 *)pOutBuffer;

    *pOut = MAKE_DWORD(0xff, val, val, val);

    return LW_OK;
}

static LW_STATUS ColwertFromG16R16(LwU8* pInBuffer, LwU8* pOutBuffer, LwU32 flags)
{
    LwU16* pInFormatBuffer = (LwU16*) pInBuffer;
    LwU32* pOutFormatBuffer = (LwU32*) pOutBuffer;

    LwU16 valG = pInFormatBuffer[0];
    LwU16 valR = pInFormatBuffer[1];

    *pOutFormatBuffer = MAKE_DWORD(0xff, valR, valG, 0);

    return LW_OK;
}

#pragma pack(push, 1)
typedef struct
{
    LwU16 color0;
    LwU16 color1;
    LwU32 indices;
} DXT1Block;

static LW_STATUS ColwertFromDXT1(LwU8 *inBuffer, LwU8 *outBuffer, LwU32 flags)
{
    LwU32 i;
    DXT1Block* block = (DXT1Block*)inBuffer;
    LwU8 c0_r, c0_g, c0_b;
    LwU8 c1_r, c1_g, c1_b;
    LwU8 red, green, blue;
    LwU32 color_table[4];
    LwU8 color_idx;

    ColwertFromR5G6B5((LwU8*)&block->color0, (LwU8*)&color_table[0], 0);
    ColwertFromR5G6B5((LwU8*)&block->color1, (LwU8*)&color_table[1], 0);

    // force alpha to 255
    color_table[0] |= 0xff000000;
    color_table[1] |= 0xff000000;

    // extract the components
    c0_r = (LwU8)((color_table[0] >> 16) & 0xff);
    c0_g = (LwU8)((color_table[0] >> 8) & 0xff);
    c0_b = (LwU8)(color_table[0] & 0xff);

    c1_r = (LwU8)((color_table[1] >> 16) & 0xff);
    c1_g = (LwU8)((color_table[1] >> 8) & 0xff);
    c1_b = (LwU8)(color_table[1] & 0xff);

    // interpolate to determine the other two values in the color table
    if(block->color0 > block->color1)
    {
        red = (2 * c0_r + c1_r) / 3;
        green = (2 * c0_g + c1_g) / 3;
        blue = (2 * c0_b + c1_b) / 3;

        color_table[2] = MAKE_DWORD(0xff, red, green, blue);

        red = (c0_r + 2 * c1_r) / 3;
        green = (c0_g + 2 * c1_g) / 3;
        blue = (c0_b + 2 * c1_b) / 3;

        color_table[3] = MAKE_DWORD(0xff, red, green, blue);
    }
    else
    {
        red = (c0_r + c1_r) / 2;
        green = (c0_g + c1_g) / 2;
        blue = (c0_b + c1_b) / 2;

        color_table[2] = MAKE_DWORD(0xff, red, green, blue);
        color_table[3] = 0;
    }

    for(i=0;i<16;i++)
    {
        color_idx = (LwU8)((block->indices >> (2 * i)) & 0x3);
        ((LwU32*)outBuffer)[i] = color_table[color_idx];
    }

    return LW_OK;
}

typedef struct
{
    LwU64 alpha_table;
    LwU16 color0;
    LwU16 color1;
    LwU32 color_indices;
} DXT23Block;

static LW_STATUS ColwertFromDXT23(LwU8 *inBuffer, LwU8 *outBuffer, LwU32 flags)
{
    DXT23Block* block = (DXT23Block*)inBuffer;
    LwU32 color_table[4];
    LwU8 alpha, red, green, blue;
    LwU8 c0_r, c0_g, c0_b;
    LwU8 c1_r, c1_g, c1_b;
    LwU8 color_idx;
    LwU32 i;

    // extract the components for color0 and color1
    ColwertFromR5G6B5((LwU8*)&block->color0, (LwU8*)&color_table[0], 0);
    ColwertFromR5G6B5((LwU8*)&block->color1, (LwU8*)&color_table[1], 0);
    c0_r = (LwU8)((color_table[0] >> 16) & 0xff);
    c0_g = (LwU8)((color_table[0] >> 8) & 0xff);
    c0_b = (LwU8)(color_table[0] & 0xff);
    c1_r = (LwU8)((color_table[1] >> 16) & 0xff);
    c1_g = (LwU8)((color_table[1] >> 8) & 0xff);
    c1_b = (LwU8)(color_table[1] & 0xff);

    // interpolate between color0 and color1 to get color2 and color3
    red = (2 * c0_r + c1_r) / 3;
    green = (2 * c0_g + c1_g) / 3;
    blue = (2 * c0_b + c1_b) / 3;

    color_table[2] = MAKE_DWORD(0xff, red, green, blue);

    red = (c0_r + 2 * c1_r) / 3;
    green = (c0_g + 2 * c1_g) / 3;
    blue = (c0_b + 2 * c1_b) / 3;

    color_table[3] = MAKE_DWORD(0xff, red, green, blue);

    for (i = 0; i < 16; i++)
    {
        color_idx = (LwU8)((block->color_indices >> (2 * i)) & 0x3);
        alpha = (LwU8)((block->alpha_table >> (4*i)) & 0xf);
        // alpha values are only 0-15 so shift by an extra 4 bits
        ((LwU32*)outBuffer)[i] = alpha << 28 | color_table[color_idx];
    }

    return LW_OK;
}

typedef struct
{
    LwU8 alpha0;
    LwU8 alpha1;
    LwU32 alpha_indices_lower;
    LwU16 alpha_indices_upper;
    LwU16 color0;
    LwU16 color1;
    LwU32 color_indices;
} DXT45Block;

static LW_STATUS ColwertFromDXT45(LwU8 *inBuffer, LwU8 *outBuffer, LwU32 flags)
{
    DXT45Block* block = (DXT45Block*)inBuffer;
    LwU8 alpha_table[8];
    LwU32 color_table[4];
    LwU64 alpha_indices;
    LwU8 red, green, blue;
    LwU8 c0_r, c0_g, c0_b;
    LwU8 c1_r, c1_g, c1_b;
    LwU32 i;
    LwU8 alpha_idx;
    LwU8 color_idx;

    alpha_table[0] = block->alpha0;
    alpha_table[1] = block->alpha1;

    if (alpha_table[0] > alpha_table[1])
    {
        // interpolate between alpha0 and alpha1
        // to get alpha2 through alpha7
        for (i = 2; i < 8; i++)
        {
            alpha_table[i] = (LwU8)(((8-i) * alpha_table[0] + (i-1) * alpha_table[1]) / 7);
        }
    }
    else
    {
        // interpolate between alpha0 and alpha1
        // to get alpha2 through alpha5
        // in this mode alpha6 is always 0 and alpha7 is always 255
        for (i = 2; i < 6; i++)
        {
            alpha_table[i] = (LwU8)(((6-i) * alpha_table[0] + (i-1)*alpha_table[1]) / 5);
        }
        alpha_table[6] = 0;
        alpha_table[7] = 255;
    }

    // table of sixteen 3-bit indices
    alpha_indices = block->alpha_indices_lower | ((LwU64)block->alpha_indices_upper) << 32;

    // extract the components for color0 and color1
    ColwertFromR5G6B5((LwU8*)&block->color0, (LwU8*)&color_table[0], 0);
    ColwertFromR5G6B5((LwU8*)&block->color1, (LwU8*)&color_table[1], 0);
    c0_r = (LwU8)((color_table[0] >> 16) & 0xff);
    c0_g = (LwU8)((color_table[0] >> 8) & 0xff);
    c0_b = (LwU8)(color_table[0] & 0xff);
    c1_r = (LwU8)((color_table[1] >> 16) & 0xff);
    c1_g = (LwU8)((color_table[1] >> 8) & 0xff);
    c1_b = (LwU8)(color_table[1] & 0xff);

    // interpolate between color0 and color1 to get color2 and color3
    red = (2 * c0_r + c1_r) / 3;
    green = (2 * c0_g + c1_g) / 3;
    blue = (2 * c0_b + c1_b) / 3;

    color_table[2] = MAKE_DWORD(0xff, red, green, blue);

    red = (c0_r + 2 * c1_r) / 3;
    green = (c0_g + 2 * c1_g) / 3;
    blue = (c0_b + 2 * c1_b) / 3;

    color_table[3] = MAKE_DWORD(0xff, red, green, blue);

    for (i = 0; i < 16; i++)
    {
        alpha_idx = (LwU8)((alpha_indices >> (3 * i)) & 0x7);
        color_idx = (LwU8)((block->color_indices >> (2 * i)) & 0x3);

        ((LwU32*)outBuffer)[i] = ((LwU32)alpha_table[alpha_idx]) << 24 | color_table[color_idx];
    }

    return LW_OK;
}

#pragma pack(pop)


/*
     An 8x8 DXT texture before:

     -----------------------------------
     |  0   1   2   3 | 16  17  18  19 |
     |  4   5   6   7 | 20  21  22  23 |
     |  8   9  10  11 | 24  25  26  27 |
     | 12  13  14  15 | 28  29  30  31 |
     | --------------------------------|
     | 32  33  34  35 | 48  49  50  51 |
     | 36  37  38  39 | 52  53  54  55 |
     | 40  41  42  43 | 56  57  58  59 |
     | 44  45  46  47 | 60  61  62  63 |
     -----------------------------------

     and after:

     ----------------------------------
     |  0   1   2   3   4   5   6   7 |
     |  8   9  10  11  12  13  14  15 |
     | 16  17  18  19  20  21  22  23 |
     | 24  25  26  27  28  29  30  31 |
     | 32  33  34  35  36  37  38  39 |
     | 40  41  42  43  44  45  46  47 |
     | 48  49  50  51  52  53  54  55 |
     | 56  57  58  59  60  61  62  63 |
     ----------------------------------

     Each DXT "texel" expands to a 4x4 block of pixels
     Also note that at this point we're only rearranging pixels
     that have already been colwerted to A8R8G8B8. The actual
     decompression is done in ColwertFromDXT[1|23|45]
 */

void LinearizeDXTTexel(LwU8 *inBuffer, LwU8 *outBuffer, LwU32 pitch)
{
    LwU32 texel_width = 4 * BMP_SOI;
    LwU32 i;

    for (i = 0; i < 4; i++)
    {
        memcpy(outBuffer, inBuffer, texel_width);
        outBuffer += pitch; // move down one row
        inBuffer += texel_width; // move right one texel
    }
}

typedef LW_STATUS (*formatColwert)(LwU8 *inBuffer, LwU8 *outBuffer, LwU32 flags);

struct FORMATCOLWERTTABLE
{
    LwU32 inTexelSize;
    LwU32 outTexelSize;
    formatColwert colwertFunc;
};

#ifndef LWCTASSERT
# define LWCTASSERT(b)        (1 / ((b) ? 1 : 0))
#endif

#ifdef WIN32
static const LwU32 g_TexelFormatColwert_Base = __COUNTER__;
#define ENUM_TABLE(FromEnum, inTexelSize, outTexelSize, colwertFunc)     \
    {(LWCTASSERT(FromEnum == (__COUNTER__-1))*inTexelSize), outTexelSize, colwertFunc}
#else
#define ENUM_TABLE(FromEnum, inTexelSize, outTexelSize, colwertFunc) \
    {inTexelSize, outTexelSize, colwertFunc}
#endif

static struct FORMATCOLWERTTABLE g_FormatColwert[] =
{
    ENUM_TABLE(TF_X8R8G8B8,         4, 4, ColwertFromARGB8             ),
    ENUM_TABLE(TF_A8R8G8B8,         4, 4, ColwertFromARGB8             ),
    ENUM_TABLE(TF_R5G6B5,           2, 4, ColwertFromR5G6B5            ),
    ENUM_TABLE(TF_A1R5G5B5,         2, 4, ColwertFromA1R5G5B5          ),
    ENUM_TABLE(TF_A16B16G16R16F,    8, 4, ColwertFromA16B16G16R16F     ),
    ENUM_TABLE(TF_R16F,             2, 4, ColwertFromR16F              ),
    ENUM_TABLE(TF_R16UN,            2, 4, ColwertFromR16UN             ),
    ENUM_TABLE(TF_R32F,             4, 4, ColwertFromR32F              ),
    ENUM_TABLE(TF_A32B32G32R32F,   16, 4, ColwertFromA32B32G32R32F     ),
    ENUM_TABLE(TF_A2R10G10B10,      4, 4, ColwertFromA2R10G10B10       ),
    ENUM_TABLE(TF_A4R4G4B4,         2, 4, ColwertFromA4R4G4B4          ),
    ENUM_TABLE(TF_S8Z24,            4, 4, ColwertFromS8Z24             ),
    ENUM_TABLE(TF_Z24S8,            4, 4, ColwertFromZ24S8             ),
    ENUM_TABLE(TF_Y8,               1, 4, ColwertFromY8                ),
    ENUM_TABLE(TF_YUY2,             2, 4, ColwertFromYUY2              ),
    ENUM_TABLE(TF_UYVY,             2, 4, ColwertFromUYVY              ),
    ENUM_TABLE(TF_A8B8G8R8,         4, 4, ColwertFromABGR8             ),
    ENUM_TABLE(TF_R11G11B10F,       4, 4, ColwertFromR11G11B10         ),
    ENUM_TABLE(TF_LW12,             1, 4, ColwertFromY8                ),
    ENUM_TABLE(TF_LW24,             1, 4, ColwertFromY8                ),
    ENUM_TABLE(TF_YV12,             1, 4, ColwertFromY8                ),
    ENUM_TABLE(TF_UV,               2, 4, ColwertFromUV                ),
    ENUM_TABLE(TF_R32G32F,          8, 4, ColwertFromR32G32F           ),
    ENUM_TABLE(TF_Z24X8_X16V8S8,    8, 4, ColwertFromZ24X8_X16V8S8     ),
    ENUM_TABLE(TF_Z32F_X16V8S8,     8, 4, ColwertFromZ32F_X16V8S8      ),
    ENUM_TABLE(TF_DXT1,             8,64, ColwertFromDXT1              ),
    ENUM_TABLE(TF_DXT23,           16,64, ColwertFromDXT23             ),
    ENUM_TABLE(TF_DXT45,           16,64, ColwertFromDXT45             ),
    ENUM_TABLE(TF_AYUV,             4, 4, ColwertFromAYUV              ),
    ENUM_TABLE(TF_P010,             2, 4, ColwertFromY8                ),
    ENUM_TABLE(TF_YY16,             2, 4, ColwertFromY16               ),
    ENUM_TABLE(TF_UV16,             4, 4, ColwertFromUV16              )
};

#undef ENUM_TABLE

LwU32 get16x2ByteOffset(LwU32 xB, LwU32 yL)
{
    LwU32 halfGobAddr  = ((xB % 64) / 32) * 256;
    LwU32 _8thGobAddr  = ((yL % 8) / 2) * 64;
    LwU32 _16thGobAddr = ((xB % 32) / 16) * 32;
    LwU32 _32thGobAddr = (yL % 2) * 16;
    LwU32 lsbAddr      = xB % 16;

    return (halfGobAddr + _8thGobAddr + _16thGobAddr + _32thGobAddr + lsbAddr);
}

void do16x2UnSwizzile(LwU8 *outBuffer, LwU32 outOffset, LwU8 *inputBuffer, LwU32 gobY, LwU32 gobWidth)
{
    LwU32 i, inOffset;
    LwU32 numStride = gobWidth/16;

    for (i = 0; i < numStride; i++)
    {
        inOffset = get16x2ByteOffset(i*16, gobY);
        memcpy(outBuffer + outOffset, inputBuffer + inOffset, 16);
        outOffset += 16;
    }
}

static LW_STATUS blockLinearToLinearCB(LwU64 va, void* buffer, LwU32 length, VCB_PARAM* pParam)
{
    BLTOLINEAR_VCB_PARAM* pBlToLinearParam;
    LwU32 numGobs;
    LwU8 *pInData, *pOutData;
    LwU32 lwrLinearYBase, lwrLinearXBase, lwrLinearZBase; //The linear position of the data we're lwrrently writing to
    LwU32 lwrGobY; //The current X/Y within the current gob

    LWCTASSERT(ARRAY_ELEMENT_COUNT(g_FormatColwert) == TF_COUNT);

    pBlToLinearParam = (BLTOLINEAR_VCB_PARAM*) pParam;

    if (!pBlToLinearParam || pBlToLinearParam->vcbParam.Id != VCB_ID_BLTOLINEAR)
    {
        dprintf("Invalid parameter to blockLinearToLinearCB - aborting\n");
        return LW_ERR_GENERIC;
    }

    pInData  = (LwU8*) buffer;
    pOutData = (LwU8*) pBlToLinearParam->pOutBuf;
    if (length & (pBlToLinearParam->gobWidth*pBlToLinearParam->gobHeight - 1))
    {
        dprintf("Length is not gob aligned - aborting\n");
        return LW_ERR_GENERIC;
    }

    numGobs = length / (pBlToLinearParam->gobWidth * pBlToLinearParam->gobHeight);

    for (;pBlToLinearParam->lwrRead.zBlock < pBlToLinearParam->blockLinearInfo.zBlocks;
         ++pBlToLinearParam->lwrRead.zBlock)
    {
        for (;pBlToLinearParam->lwrRead.yBlock < pBlToLinearParam->blockLinearInfo.yBlocks;
            ++pBlToLinearParam->lwrRead.yBlock)
        {
            for (;pBlToLinearParam->lwrRead.xBlock < pBlToLinearParam->blockLinearInfo.xBlocks;
                ++pBlToLinearParam->lwrRead.xBlock)
            {
                for (;pBlToLinearParam->lwrRead.zGob < (1ul << pBlToLinearParam->blockLinearInfo.log2GobsPerBlock.z);
                     ++pBlToLinearParam->lwrRead.zGob)
                {
                    lwrLinearZBase = ((1 << pBlToLinearParam->blockLinearInfo.log2GobsPerBlock.z) * pBlToLinearParam->lwrRead.zBlock +
                                     pBlToLinearParam->lwrRead.zGob);
                    for (;pBlToLinearParam->lwrRead.yGob < (1ul << pBlToLinearParam->blockLinearInfo.log2GobsPerBlock.y);
                         ++pBlToLinearParam->lwrRead.yGob)
                    {
                        lwrLinearYBase = ((1 << pBlToLinearParam->blockLinearInfo.log2GobsPerBlock.y) * pBlToLinearParam->lwrRead.yBlock +
                                         pBlToLinearParam->lwrRead.yGob) * pBlToLinearParam->gobHeight;
                        for (;pBlToLinearParam->lwrRead.xGob < (1ul << pBlToLinearParam->blockLinearInfo.log2GobsPerBlock.x);
                             ++pBlToLinearParam->lwrRead.xGob)
                        {
                            lwrLinearXBase = ((1 << pBlToLinearParam->blockLinearInfo.log2GobsPerBlock.x) * pBlToLinearParam->lwrRead.xBlock +
                                             pBlToLinearParam->lwrRead.xGob) * pBlToLinearParam->gobWidth;

                            for (lwrGobY = 0; lwrGobY < pBlToLinearParam->gobHeight; ++lwrGobY)
                            {
                                if (osCheckControlC())
                                    return LW_ERR_GENERIC;

                                if (pBlToLinearParam->vMemType != VMEM_TYPE_IOMMU)
                                {
                                    memcpy(pOutData +
                                        lwrLinearZBase * pBlToLinearParam->slicePitch +
                                        (lwrLinearYBase + lwrGobY) * pBlToLinearParam->pitch +
                                        (lwrLinearXBase),
                                        pInData + lwrGobY * pBlToLinearParam->gobWidth,
                                        pBlToLinearParam->gobWidth);
                                }
                                else
                                {
                                    do16x2UnSwizzile(pOutData, lwrLinearZBase * pBlToLinearParam->slicePitch +
                                                     (lwrLinearYBase + lwrGobY) * pBlToLinearParam->pitch +
                                                     (lwrLinearXBase),
                                                     pInData, lwrGobY, pBlToLinearParam->gobWidth);
                                }
                            }

                            pInData += pBlToLinearParam->gobWidth * pBlToLinearParam->gobHeight;

                            if (--numGobs == 0)
                                break;
                        }

                        if (!numGobs)
                            break;

                        pBlToLinearParam->lwrRead.xGob = 0;
                    }

                    if (!numGobs)
                        break;

                    pBlToLinearParam->lwrRead.yGob = 0;
                }
                if (!numGobs)
                    break;

                pBlToLinearParam->lwrRead.zGob = 0;
            }

            if (!numGobs)
                break;

            pBlToLinearParam->lwrRead.xBlock = 0;
        }

        if (!numGobs)
            break;

        pBlToLinearParam->lwrRead.yBlock = 0;
    }

    ++pBlToLinearParam->lwrRead.xGob;
    if (pBlToLinearParam->lwrRead.xGob == (LwU32)(1 << pBlToLinearParam->blockLinearInfo.log2GobsPerBlock.x))
    {
        pBlToLinearParam->lwrRead.xGob = 0;
        ++pBlToLinearParam->lwrRead.yGob;
    }

    if (pBlToLinearParam->lwrRead.yGob == (LwU32)(1 << pBlToLinearParam->blockLinearInfo.log2GobsPerBlock.y))
    {
        pBlToLinearParam->lwrRead.yGob = 0;
        ++pBlToLinearParam->lwrRead.xBlock;
    }

    if (pBlToLinearParam->lwrRead.xBlock == pBlToLinearParam->blockLinearInfo.xBlocks)
    {
        pBlToLinearParam->lwrRead.xBlock = 0;
        ++pBlToLinearParam->lwrRead.yBlock;
    }

    if (pBlToLinearParam->lwrRead.yBlock == pBlToLinearParam->blockLinearInfo.yBlocks)
    {
    }

    return LW_OK;
}

static LW_STATUS blockLinearToLinearCB_Turing(LwU64 va, void* buffer, LwU32 length, VCB_PARAM* pParam)
{
    BLTOLINEAR_VCB_PARAM* pBlToLinearParam;
    LwU32 numGobs;
    LwU8 *pInData, *pOutData;
    LwU32 lwrLinearYBase, lwrLinearXBase, lwrLinearZBase; //The linear position of the data we're lwrrently writing to
    LwU32 lwrGobX, lwrGobY; //The current X/Y within the current gob
    LwU32 newGobX, newGobY;

    LWCTASSERT(ARRAY_ELEMENT_COUNT(g_FormatColwert) == TF_COUNT);

    pBlToLinearParam = (BLTOLINEAR_VCB_PARAM*)pParam;

    if (!pBlToLinearParam || pBlToLinearParam->vcbParam.Id != VCB_ID_BLTOLINEAR)
    {
        dprintf("Invalid parameter to blockLinearToLinearCB_Turing - aborting\n");
        return LW_ERR_GENERIC;
    }

    pInData = (LwU8*)buffer;
    pOutData = (LwU8*)pBlToLinearParam->pOutBuf;
    if (length & (pBlToLinearParam->gobWidth*pBlToLinearParam->gobHeight - 1))
    {
        dprintf("Length is not gob aligned - aborting\n");
        return LW_ERR_GENERIC;
    }

    numGobs = length / (pBlToLinearParam->gobWidth * pBlToLinearParam->gobHeight);

    for (; pBlToLinearParam->lwrRead.zBlock < pBlToLinearParam->blockLinearInfo.zBlocks;
        ++pBlToLinearParam->lwrRead.zBlock)
    {
        for (; pBlToLinearParam->lwrRead.yBlock < pBlToLinearParam->blockLinearInfo.yBlocks;
            ++pBlToLinearParam->lwrRead.yBlock)
        {
            for (; pBlToLinearParam->lwrRead.xBlock < pBlToLinearParam->blockLinearInfo.xBlocks;
                ++pBlToLinearParam->lwrRead.xBlock)
            {
                for (; pBlToLinearParam->lwrRead.zGob < (1ul << pBlToLinearParam->blockLinearInfo.log2GobsPerBlock.z);
                    ++pBlToLinearParam->lwrRead.zGob)
                {
                    lwrLinearZBase = ((1 << pBlToLinearParam->blockLinearInfo.log2GobsPerBlock.z) * pBlToLinearParam->lwrRead.zBlock +
                        pBlToLinearParam->lwrRead.zGob);
                    for (; pBlToLinearParam->lwrRead.yGob < (1ul << pBlToLinearParam->blockLinearInfo.log2GobsPerBlock.y);
                        ++pBlToLinearParam->lwrRead.yGob)
                    {
                        lwrLinearYBase = ((1 << pBlToLinearParam->blockLinearInfo.log2GobsPerBlock.y) * pBlToLinearParam->lwrRead.yBlock +
                            pBlToLinearParam->lwrRead.yGob) * pBlToLinearParam->gobHeight;
                        for (; pBlToLinearParam->lwrRead.xGob < (1ul << pBlToLinearParam->blockLinearInfo.log2GobsPerBlock.x);
                            ++pBlToLinearParam->lwrRead.xGob)
                        {
                            lwrLinearXBase = ((1 << pBlToLinearParam->blockLinearInfo.log2GobsPerBlock.x) * pBlToLinearParam->lwrRead.xBlock +
                                pBlToLinearParam->lwrRead.xGob) * pBlToLinearParam->gobWidth;

                            for (lwrGobY = 0; lwrGobY < pBlToLinearParam->gobHeight; ++lwrGobY)
                            {
                                if (osCheckControlC())
                                    return LW_ERR_GENERIC;

                                for (lwrGobX = 0; lwrGobX < pBlToLinearParam->gobWidth; lwrGobX += pBlToLinearParam->gobWidth / 4)
                                {
                                    // From gob (x,y) position, callwlate new position to copy from.
                                    newGobX = ((lwrGobY & 4) << 3) | ((lwrGobY & 1) << 4);
                                    newGobY = ((lwrGobY & 2) << 1) | (lwrGobX >> 4);

                                    memcpy(pOutData +
                                        lwrLinearZBase * pBlToLinearParam->slicePitch +
                                        (lwrLinearYBase + newGobY) * pBlToLinearParam->pitch +
                                        (lwrLinearXBase)
                                        +newGobX,
                                        pInData + lwrGobY * pBlToLinearParam->gobWidth
                                        + lwrGobX,
                                        pBlToLinearParam->gobWidth / 4);
                                }
                            }

                            pInData += pBlToLinearParam->gobWidth * pBlToLinearParam->gobHeight;

                            if (--numGobs == 0)
                                break;
                        }

                        if (!numGobs)
                            break;

                        pBlToLinearParam->lwrRead.xGob = 0;
                    }

                    if (!numGobs)
                        break;

                    pBlToLinearParam->lwrRead.yGob = 0;
                }
                if (!numGobs)
                    break;

                pBlToLinearParam->lwrRead.zGob = 0;
            }

            if (!numGobs)
                break;

            pBlToLinearParam->lwrRead.xBlock = 0;
        }

        if (!numGobs)
            break;

        pBlToLinearParam->lwrRead.yBlock = 0;
    }

    ++pBlToLinearParam->lwrRead.xGob;
    if (pBlToLinearParam->lwrRead.xGob == (LwU32)(1 << pBlToLinearParam->blockLinearInfo.log2GobsPerBlock.x))
    {
        pBlToLinearParam->lwrRead.xGob = 0;
        ++pBlToLinearParam->lwrRead.yGob;
    }

    if (pBlToLinearParam->lwrRead.yGob == (LwU32)(1 << pBlToLinearParam->blockLinearInfo.log2GobsPerBlock.y))
    {
        pBlToLinearParam->lwrRead.yGob = 0;
        ++pBlToLinearParam->lwrRead.xBlock;
    }

    if (pBlToLinearParam->lwrRead.xBlock == pBlToLinearParam->blockLinearInfo.xBlocks)
    {
        pBlToLinearParam->lwrRead.xBlock = 0;
        ++pBlToLinearParam->lwrRead.yBlock;
    }

    return LW_OK;
}

#ifdef WIN32
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    PAINTSTRUCT ps;
    HDC hdc;

    switch (message)
    {
    case WM_PAINT:
        hdc = BeginPaint(hWnd, &ps);
        EndPaint(hWnd, &ps);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }

    return 0;
}

#endif // WIN32


LW_STATUS calwlateBlockLinearInfo(LwU32 width, LwU32 height,
                             LwU32 logBlockWidth, LwU32 logBlockHeight, LwU32 logBlockDepth,
                             LwU32 format, LwU32 gobWidth, LwU32 gobHeight,
                             LwBlockLinearImageInfo *pBlockLinearInfo,
                             LwBlockLinearTexParams *pTexParams)
{
    LwU32 formatIndex;

    if (!pBlockLinearInfo || !pTexParams)
    {
        dprintf("lw: %s: both pBlockLinearInfo and pTexParams are required.", __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    formatIndex = DRF_VAL(_DISPLAYVIRTUAL, _FORMAT, _TYPE, format);

    memset (pBlockLinearInfo, 0, sizeof(LwBlockLinearImageInfo));
    memset (pTexParams, 0, sizeof(LwBlockLinearTexParams));

    //
    // the getBlockLinearInfo function needs the size in texels to correctly compute
    // the mip level size, not in compressed blocks
    //
    pTexParams->dwBaseWidth       = width;
    pTexParams->dwBaseHeight      = height;
    pTexParams->dwBaseDepth       = 1;

    //Need to fill gob per block params
    pTexParams->dwTexelSize       = g_FormatColwert[formatIndex].inTexelSize;
    pTexParams->dwDimensionality  = 2;
    pTexParams->dwFace            = 0;
    pTexParams->dwFaceSize        = 0;
    pTexParams->dwBorderSize      = 0;
    pTexParams->dwLOD             = 0;
    pTexParams->dwBlockWidthLog2  = 0;
    pTexParams->dwBlockHeightLog2 = 0;

    pBlockLinearInfo->log2GobsPerBlock.x = logBlockWidth;
    pBlockLinearInfo->log2GobsPerBlock.y = logBlockHeight;
    pBlockLinearInfo->log2GobsPerBlock.z = logBlockDepth;

    lwGetBlockLinearTexLevelInfo(pBlockLinearInfo, pTexParams);

    return LW_OK;
}

LW_STATUS virtDisplayVirtual(VMemTypes vMemType, LwU32 chId, LwU64 va,
                        LwU32 width, LwU32 height, //No depth since it displays a 2D image
                        LwU32 logBlockWidth, LwU32 logBlockHeight, LwU32 logBlockDepth, //Still need block depth though!
                        LwU32 format, LwU32 gobWidth, LwU32 gobHeight)
{
    LW_STATUS  status = LW_OK;
    LwU32 length = 0;
    LwU8* pLinearBuf, *pLinearBuf2 = NULL;
    LwU8* pBmpBuf;
    VMemSpace vMemSpace;
    BLTOLINEAR_VCB_PARAM   BlToLinearData;
    READVIRTUAL_VCB_PARAM  linearToLinearParam;
    VCB_PARAM             *pVCBParam = (VCB_PARAM *) &BlToLinearData;

    LwU32 formatIndex = DRF_VAL(_DISPLAYVIRTUAL, _FORMAT, _TYPE, format);

    LwU32 blockWidthAlign = (1 << logBlockWidth) * gobWidth;
    LwU32 blockHeightAlign = (1 << logBlockHeight) * gobHeight;

    unsigned long inPitch;
    unsigned long inSlicePitch;

    LwU32 outPitch;
    LwU32 outSlicePitch;

    BOOL bColwerted = FALSE;
    BOOL bBlockLinear = TRUE;

    VMEM_INPUT_TYPE Id;

    virtualCallback pfnReadCB = blockLinearToLinearCB;

    // If Turing or later GPU, treat this surface as Xbar Raw view.
    if (IsTU102orLater())
    {
        pfnReadCB = blockLinearToLinearCB_Turing;
    }

    // DXT formats work on 4x4 blocks - modify width and height accordingly
    if (format == TF_DXT1 || format == TF_DXT23 || format == TF_DXT45)
    {
        width >>= 2;
        height >>= 2;
    }

    inPitch = (width * g_FormatColwert[formatIndex].inTexelSize + (blockWidthAlign-1)) & ~(blockWidthAlign-1);
    inSlicePitch = ((height + (blockHeightAlign-1)) & ~(blockHeightAlign-1)) * inPitch;

    outPitch = width * g_FormatColwert[formatIndex].outTexelSize;
    outSlicePitch = height * outPitch;

    // If logBLockWidth is greater than or equal to the default linear pitch
    // then treat this surface as pitch linear, i.e. not a block linear surface
    if ((logBlockWidth >= (width * g_FormatColwert[formatIndex].inTexelSize)) && (logBlockHeight == 0)) // pitch linear
    {
        inPitch = blockWidthAlign = logBlockWidth;
        blockHeightAlign = 2;
        length = inSlicePitch = height * inPitch;
        bBlockLinear = FALSE;
        pfnReadCB = readVirtualCB;

        linearToLinearParam.vcbParam.Id = VCB_ID_READVIRTUAL;
        linearToLinearParam.vcbParam.memType = MT_GPUVIRTUAL;
        linearToLinearParam.vcbParam.bStatus = TRUE;
        linearToLinearParam.lwrOffset = 0;
        linearToLinearParam.bufferSize = length;

        pVCBParam = (VCB_PARAM *) &linearToLinearParam;
    }
    
    memset(&Id, 0, sizeof(Id));
    Id.ch.chId = chId;

    if (vmemGet(&vMemSpace, vMemType, &Id) != LW_OK)
    {
        dprintf("lw: %s: Could not get a VMEM Space for ChId 0x%x.\n",
                __FUNCTION__, chId);
        return LW_ERR_GENERIC;
    }

    pLinearBuf = (LwU8*) malloc(inSlicePitch);
    pBmpBuf = (LwU8*) malloc(outSlicePitch);
    if (!pBmpBuf || !pLinearBuf)
    {
        dprintf("%s : allocation failed\n", __FUNCTION__);
        free(pLinearBuf);
        free(pBmpBuf);
        return LW_ERR_GENERIC;
    }

    if (bBlockLinear)
    {
        BlToLinearData.vcbParam.Id = VCB_ID_BLTOLINEAR;
        BlToLinearData.vcbParam.memType = MT_GPUVIRTUAL;
        BlToLinearData.vcbParam.bStatus = TRUE;

        BlToLinearData.gobWidth = gobWidth;
        BlToLinearData.gobHeight = gobHeight;

        BlToLinearData.pitch = inPitch;
        BlToLinearData.slicePitch = inSlicePitch;
        BlToLinearData.pOutBuf = pLinearBuf;

        BlToLinearData.lwrRead.xGob = 0;
        BlToLinearData.lwrRead.yGob = 0;
        BlToLinearData.lwrRead.zGob = 0;
        BlToLinearData.lwrRead.xBlock = 0;
        BlToLinearData.lwrRead.yBlock = 0;
        BlToLinearData.lwrRead.zBlock = 0;

        BlToLinearData.vMemType = vMemType;

        status = calwlateBlockLinearInfo(width, height,
                                         logBlockWidth, logBlockHeight, logBlockDepth,
                                         format, gobWidth, gobHeight,
                                         &BlToLinearData.blockLinearInfo,
                                         &BlToLinearData.texParams);

        length = BlToLinearData.blockLinearInfo.size;
    }
    else
    {
        linearToLinearParam.pData = pLinearBuf;
    }

    if (status == LW_OK)
    {

        status = vmemDoVirtualOp(&vMemSpace, va, length, 0,
                                 pfnReadCB,
                                 pVCBParam);
    }


    if (status == LW_OK && (format==TF_LW12 || format== TF_LW24 || format== TF_YV12 || format==TF_P010))
    {
        unsigned long chromaBlockSize;
        unsigned long chromaFormat = format;
        unsigned long chromaHeight = height;
        unsigned long chromaWidth = width >> 1;
        unsigned long chromaPitch = inPitch;

        if (format!= TF_YV12)
        {
            if (format!=TF_P010)
            {
            chromaHeight = (((height >> 2) + (blockHeightAlign-1)) & ~(blockHeightAlign-1)) << 1;
            chromaFormat = TF_UV;
        }
        else
        {
                chromaHeight = (((height >> 1) + (blockHeightAlign-1)) & ~(blockHeightAlign-1));
                chromaFormat = TF_UV16;
            }
        }
        else
        {
            BlToLinearData.pitch = chromaPitch = inPitch >> 1;
        }
        dprintf("\n");
        dprintf("chroma pitch 0x%lx\n", chromaPitch);
        dprintf("chroma height 0x%lx\n", chromaHeight);

        chromaBlockSize = chromaHeight * chromaPitch;

        pLinearBuf2 = (LwU8*) malloc(chromaBlockSize);

        if (!pLinearBuf2)
        {
            dprintf("%s : allocation failed\n", __FUNCTION__);
            free(pLinearBuf);
            free(pBmpBuf);
            return LW_ERR_GENERIC;
        }

        if (bBlockLinear)
        {

            BlToLinearData.slicePitch = chromaBlockSize;
            BlToLinearData.pOutBuf = pLinearBuf2;      // UV plane (LW12,LW24) or V plane (YV12)

            BlToLinearData.lwrRead.xGob = 0;
            BlToLinearData.lwrRead.yGob = 0;
            BlToLinearData.lwrRead.zGob = 0;
            BlToLinearData.lwrRead.xBlock = 0;
            BlToLinearData.lwrRead.yBlock = 0;
            BlToLinearData.lwrRead.zBlock = 0;

            status = calwlateBlockLinearInfo(chromaWidth, chromaHeight,
                                             logBlockWidth, logBlockHeight, logBlockDepth,
                                             chromaFormat, gobWidth, gobHeight,
                                             &BlToLinearData.blockLinearInfo,
                                             &BlToLinearData.texParams);

        }
        else
        {
            linearToLinearParam.lwrOffset = 0;
            linearToLinearParam.bufferSize = chromaBlockSize;
            linearToLinearParam.pData = pLinearBuf2;

        }

        dprintf("luma block size 0x%lx\n", inSlicePitch);
        dprintf("chroma block size 0x%lx\n", chromaBlockSize);
        // dprintf("chroma block Info size %d\n", BlToLinearData.blockLinearInfo.size);

        va += inSlicePitch;
        length = chromaBlockSize;

        // dprintf("before mapping chroma %d %d\n", *(LwU32 *)pLinearBuf2, *(LwU32 *)(pLinearBuf2 + 4));


        status = vmemDoVirtualOp(&vMemSpace, va, length, 0,
                                 pfnReadCB,
                                 pVCBParam);

        // dprintf("after mapping chroma %d %d\n", *(LwU32 *)pLinearBuf2, *(LwU32 *)(pLinearBuf2 + 4));


        if (status == LW_OK)
        {
            if (format==TF_P010)
            {
                status = ColwertFromP010(pLinearBuf, pLinearBuf2, pBmpBuf, width, height, inPitch, outPitch, format, chromaBlockSize);
            }
            else
            {
            status = ColwertFromYUV420(pLinearBuf, pLinearBuf2, pBmpBuf, width, height, inPitch, outPitch, format, chromaBlockSize);
        }
        }


        bColwerted = TRUE;
    }

    if (status == LW_OK && bColwerted == FALSE)
    {
        LwU32 i, j, k;
        LwU8* pInBuffer = (LwU8*) pLinearBuf;
        LwU8* pOutBuffer = (LwU8*) pBmpBuf;
        LwU32 inSkipPitch, inSkipSlicePitch;
        LwU32 outSkipPitch, outSkipSlicePitch;

        inSkipPitch = inPitch - width * g_FormatColwert[formatIndex].inTexelSize;
        inSkipSlicePitch = (inSlicePitch - height * inPitch);

        outSkipPitch = (outPitch - width * g_FormatColwert[formatIndex].outTexelSize);
        outSkipSlicePitch = (outSlicePitch - height * outPitch);

        for (k = 0; k < 1; ++k)
        {
            for (j = 0; j < height; ++j)
            {
                for (i = 0; i < width; ++i)
                {
                    LwU32 myFormat = format;

                    if (i&1)
                        myFormat |= DRF_DEF(_DISPLAYVIRTUAL, _FORMAT, _DISPLAY_PIXEL, _ODD);

                    status = g_FormatColwert[formatIndex].colwertFunc(pInBuffer, pOutBuffer, myFormat );
                    pInBuffer += g_FormatColwert[formatIndex].inTexelSize;
                    pOutBuffer += g_FormatColwert[formatIndex].outTexelSize;
                }
                pInBuffer += inSkipPitch;
                pOutBuffer += outSkipPitch;
            }
            pInBuffer += inSkipSlicePitch;
            pOutBuffer += outSkipSlicePitch;
        }
    }

    // DXT formats must be colwerted from texel-linear to linear
    // see comments before LinearizeDXTTexel for details
    if (status == LW_OK && (format == TF_DXT1 || format == TF_DXT23 || format == TF_DXT45))
    {
        LwU32 i, j;
        LwU8* tmpRowBuffer = (LwU8*) malloc(outPitch);  // used for processing one row of texels from the image
        LwU8* pInBuffer;
        LwU8* pOutBuffer;
        const LwU32 texelWidth = 4 * BMP_SOI; // each texel is a 4x4 block of pixels
        const LwU32 truePitch = width * texelWidth; // remember, 'width' is in texels

        for (j = 0; j < height; ++j)
        {
            pOutBuffer = tmpRowBuffer;
            pInBuffer = (LwU8*) &pBmpBuf[outPitch * j];
            for (i = 0; i < width; ++i)
            {
                LinearizeDXTTexel(pInBuffer, pOutBuffer, truePitch);
                pInBuffer += g_FormatColwert[formatIndex].outTexelSize;
                pOutBuffer += texelWidth;
            }
            memcpy(&pBmpBuf[outPitch * j], tmpRowBuffer, outPitch);
        }
        free(tmpRowBuffer);

        // texels have been colwerted to pixels
        // change these accordingly
        outPitch = truePitch;
        width <<= 2;
        height <<= 2;
    }

#if defined(WIN32)
    if (status == LW_OK) {
        BITMAPINFO bmInfo;
        WNDCLASS wndClass;
        HWND hWnd = 0;
        HDC hdcWindow = 0, hdcBmp = 0;
        HBITMAP hBmp = 0;
        RECT rectWindow, rectDesktop;
        int windowWidth, windowHeight;
        LwU32 swapLine;
        BOOL bNeedsScrollH, bNeedsScrollV;
        MSG msg;
        LwU8* pBmpBufferLine = NULL;

        pBmpBufferLine = malloc(outPitch);
        if (!pBmpBufferLine)
        {
            dprintf("Allocation of temp buffer for BMP reformatting failed");
            return LW_ERR_GENERIC;
        }

        for (swapLine = 0; swapLine < height / 2; ++swapLine)
        {
            memcpy(pBmpBufferLine, &pBmpBuf[(height - swapLine - 1) * outPitch], outPitch);
            memcpy(&pBmpBuf[(height - swapLine - 1) * outPitch], &pBmpBuf[swapLine * outPitch], outPitch);
            memcpy(&pBmpBuf[swapLine * outPitch], pBmpBufferLine, outPitch);
        }

        free(pBmpBufferLine);
        pBmpBufferLine = NULL;

        //
        // register class
        //
        memset (&wndClass,0,sizeof(wndClass));
        wndClass.cbWndExtra    = 4;
        wndClass.hLwrsor       = NULL;
        wndClass.hIcon         = NULL;
        wndClass.hInstance     = NULL;
        wndClass.lpfnWndProc   = WndProc;
        wndClass.lpszClassName = "SurfDisplay";
        wndClass.style         = CS_HREDRAW | CS_VREDRAW;
        if (!RegisterClass(&wndClass))
        {
            wndClass.lpfnWndProc = NULL;
            dprintf("RegisterClass failed");
        }
        // Setup the window rectangle
        rectWindow.top    = 0;
        rectWindow.left   = 0;
        rectWindow.bottom = height - 1;
        rectWindow.right  = width - 1;

        // Adjust the window rectangle for the client area
        AdjustWindowRect(&rectWindow, WS_THICKFRAME | WS_CAPTION | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU, FALSE);

        // Get the desktop window size and check for scroll bars needed
        GetWindowRect(GetDesktopWindow(), &rectDesktop);
        if ((rectWindow.right - rectWindow.left) > (rectDesktop.right - rectDesktop.left))
        {
            windowWidth   = rectDesktop.right - rectDesktop.left + 1;
            bNeedsScrollH = TRUE;
        }
        else
        {
            windowWidth   = rectWindow.right - rectWindow.left + 1;
            bNeedsScrollH = FALSE;
        }

        if ((rectWindow.bottom - rectWindow.top) > (rectDesktop.bottom - rectDesktop.top))
        {
            windowHeight  = rectDesktop.bottom - rectDesktop.top + 1;
            bNeedsScrollV = TRUE;
        }
        else
        {
            windowHeight  = rectWindow.bottom - rectWindow.top + 1;
            bNeedsScrollV = FALSE;
        }
        //
        // create window
        //
        hWnd = CreateWindow("SurfDisplay", "Surface Display",
                            (bNeedsScrollH ? WS_HSCROLL : 0) | (bNeedsScrollV ? WS_VSCROLL : 0) |
                            WS_THICKFRAME | WS_CAPTION | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU,
                            CW_USEDEFAULT, CW_USEDEFAULT, windowWidth, windowHeight, NULL, NULL, wndClass.hInstance, 0);
        if (!hWnd)
        {
            dprintf("CreateWindow failed");
            goto cleanup;
        }

        ShowWindow(hWnd,SW_NORMAL);
        UpdateWindow(hWnd);

        hdcWindow = GetDC(hWnd);
        if (!hdcWindow)
        {
            dprintf("Couldn't get DC fo 'DISPLAY'");
            goto cleanup;
        }

        hdcBmp = CreateCompatibleDC(hdcWindow);
        if (!hdcBmp)
        {
            dprintf("Couldn't create display compatible DC'");
            goto cleanup;
        }

        if (!GetWindowRect(hWnd, &rectWindow))
        {
            dprintf("Couldn't get desktop rect");
            goto cleanup;
        }

        hBmp = CreateBitmap(width, height, GetDeviceCaps(hdcBmp, PLANES), GetDeviceCaps(hdcBmp, BITSPIXEL), NULL);
        if (!hBmp)
        {
            dprintf("Couldn't create compatible bitmap");
            goto cleanup;
        }

        memset(&bmInfo.bmiHeader, 0, sizeof(BITMAPINFOHEADER));
        bmInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmInfo.bmiHeader.biWidth = width;
        bmInfo.bmiHeader.biHeight = height;
        bmInfo.bmiHeader.biPlanes = 1;
        bmInfo.bmiHeader.biBitCount = 32;
        bmInfo.bmiHeader.biCompression = BI_RGB;

        if (!SetDIBits(hdcWindow, hBmp, 0, height, pBmpBuf, &bmInfo, DIB_RGB_COLORS))
        {
            dprintf("Couldn't get DI bits");
            goto cleanup;
        }

        ShowWindow(hWnd, SW_NORMAL);
        PostMessage(hWnd, WM_PAINT, 0, 0);

        // Main message loop:
        while (status = GetMessage(&msg, hWnd, 0, 0))
        {
            if (status == -1)
            {
                LwU32 error = GetLastError();
                if (error == ERROR_ILWALID_WINDOW_HANDLE)
                    status = LW_OK;
                break;
            }

            if (msg.hwnd == hWnd)
            {
                switch (msg.message)
                {
                case WM_PAINT:
                    if (!SelectObject(hdcBmp, hBmp))
                    {
                        CloseWindow(hWnd);
                        dprintf("Select object failed");
                    }

                    if (!BitBlt(hdcWindow, 0, 0, width, height,
                                hdcBmp, 0, 0, SRCCOPY))
                    {
                        CloseWindow(hWnd);
                        dprintf("BitBlt failed");
                    }

                    SelectObject(hdcBmp, 0);
                    break;
                }
            }

            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

cleanup:
        if (wndClass.lpfnWndProc)
            UnregisterClass(wndClass.lpszClassName, wndClass.hInstance);

        if (hdcWindow)
            DeleteDC(hdcWindow);

        if (hdcBmp)
            DeleteDC(hdcBmp);

        if (hBmp)
            DeleteObject(hBmp);
    }

    if (status == LW_OK)
    {
        // Write out a 'out.bmp' of the surface
        FILE* pFileOut;
        BITMAPFILEHEADER bmFileHeader;
        BITMAPINFOHEADER bmInfoHeader;

        pFileOut = fopen("out.bmp", "wb");

        if (!pFileOut)
        {
            printf("lw: Failed to open output file (%s)\n", "out.bmp");
            return -1;
        }

        bmFileHeader.bfType = 'MB';
        bmFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
        bmFileHeader.bfSize = bmFileHeader.bfOffBits + width * height * BMP_SOI;
        bmFileHeader.bfReserved1 = bmFileHeader.bfReserved2 = 0;

        memset(&bmInfoHeader, 0, sizeof(BITMAPINFOHEADER));
        bmInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmInfoHeader.biWidth = width;
        bmInfoHeader.biHeight = height;
        bmInfoHeader.biPlanes = 1;
        bmInfoHeader.biBitCount = BMP_SOI << 3;
        bmInfoHeader.biCompression = BI_RGB;

        fwrite(&bmFileHeader, sizeof(BITMAPFILEHEADER), 1, pFileOut);
        fwrite(&bmInfoHeader, sizeof(BITMAPINFOHEADER), 1, pFileOut);

        fwrite(pBmpBuf, outSlicePitch, 1, pFileOut);

        fclose(pFileOut);
    }

#endif

    free(pLinearBuf);
    free(pLinearBuf2);
    free(pBmpBuf);

    return status;
}

LW_STATUS virtFillVirtual(VCBMEMTYPE memType, LwU32 chId, LwU64 va,
                     LwU32 width, LwU32 height, //No depth since it displays a 2D image
                     LwU32 logBlockWidth, LwU32 logBlockHeight, LwU32 logBlockDepth, //Still need block depth though!
                     LwU32 format, LwU32 gobWidth, LwU32 gobHeight,
                     LwU32 color)
{
    LW_STATUS status;
    VMemSpace vMemSpace;
    LwBlockLinearImageInfo BlockLinearInfo;
    LwBlockLinearTexParams TexParams;
    VMEM_INPUT_TYPE Id;
    
    memset(&Id, 0, sizeof(Id));
    Id.ch.chId = chId;

    if (vmemGet(&vMemSpace, VMEM_TYPE_CHANNEL, &Id) != LW_OK)
    {
        dprintf("lw: %s: Could not get a VMEM Space for ChId 0x%x.\n",
                __FUNCTION__, chId);
        return LW_ERR_GENERIC;
    }

    status = calwlateBlockLinearInfo(width, height,
                                     logBlockWidth, logBlockHeight, logBlockDepth,
                                     format, gobWidth, gobHeight,
                                     &BlockLinearInfo,
                                     &TexParams);

    if (status == LW_OK)
    {
        status = pVmem[indexGpu].vmemFill(&vMemSpace, va, BlockLinearInfo.size, color);
    }

    return status;
}

static void decompose3(LwU32 total, LwU32 t2, LwU32 t1, LwU32 * c2, LwU32 * c1, LwU32 * c0)
{
    LwU32 remains = 0;

    *c2 = total / t2;
    remains = total - *c2 * t2;
    *c1 = remains / t1;
    *c0 = remains - *c1 * t1;
}

LwU32 offsetBLToLinear(LwU32 offsetBL, LwU32 width, LwU32 height,
                     LwU32 logBlockWidth, LwU32 logBlockHeight, LwU32 logBlockDepth,
                     LwU32 logGobWidth, LwU32 logGobHeight, LwU32 logGobDepth, LwU32 pitch, LwU32 format)
{
    LW_STATUS status;
    LwBlockLinearImageInfo BlockLinearInfo;
    LwBlockLinearTexParams TexParams;
    LwU32 offsetLinear;
    LwU32 linearRowBytes, linearHeight, sliceBlocks, blockWidthGobs, blockSliceGobs;
    LwU32 blockWidthBytes, blockHeight, blockDepth, blockSizeBytes;

    LwU32 zBlock, yBlock, xBlock;
    LwU32 zGob, yGob, xGob;
    LwU32 z, y, x;
    LwU32 blockIdx, gobIdx, gobOffset;

    LwU32 gobWidthBytes = 1 << logGobWidth;
    LwU32 gobHeight = 1 << logGobHeight;
    LwU32 gobDepth = 1 << logGobDepth;
    LwU32 gobSliceBytes = 1 << (logGobWidth + logGobHeight);
    LwU32 gobSizeBytes = 1 << (logGobWidth + logGobHeight + logGobDepth);

    status = calwlateBlockLinearInfo(width, height,
                                     logBlockWidth, logBlockHeight, logBlockDepth,
                                     format, gobWidthBytes, gobHeight,
                                     &BlockLinearInfo,
                                     &TexParams);
    if (status != LW_OK)
    {
        return 0;
    }

    linearRowBytes = TexParams.dwTexelSize * TexParams.dwBaseWidth;
    linearHeight = TexParams.dwBaseHeight;
    if (pitch < linearRowBytes)
    {
        pitch = linearRowBytes;
    }

    sliceBlocks = BlockLinearInfo.xBlocks * BlockLinearInfo.yBlocks;

    blockWidthGobs = 1 << BlockLinearInfo.log2GobsPerBlock.x;
    blockSliceGobs = 1 << (BlockLinearInfo.log2GobsPerBlock.x + BlockLinearInfo.log2GobsPerBlock.y);

    blockWidthBytes = 1 << (logGobWidth + BlockLinearInfo.log2GobsPerBlock.x);
    blockHeight = 1 << (logGobHeight + BlockLinearInfo.log2GobsPerBlock.y);
    blockDepth = 1 << (logGobDepth + BlockLinearInfo.log2GobsPerBlock.z);
    blockSizeBytes = 1 << (logGobWidth + logGobHeight + logGobDepth +
                                 BlockLinearInfo.log2GobsPerBlock.x +
                                 BlockLinearInfo.log2GobsPerBlock.y +
                                 BlockLinearInfo.log2GobsPerBlock.z);

    decompose3(offsetBL, blockSizeBytes, gobSizeBytes, &blockIdx, &gobIdx, &gobOffset);
    decompose3(blockIdx, sliceBlocks, BlockLinearInfo.xBlocks, &zBlock, &yBlock, &xBlock);
    decompose3(gobIdx, blockSliceGobs, blockWidthGobs, &zGob, &yGob, &xGob);
    decompose3(gobOffset, gobSliceBytes, gobWidthBytes, &z, &y, &x);

    x = xBlock * blockWidthBytes + xGob * gobWidthBytes + x;
    y = yBlock * blockHeight + yGob * gobHeight + y;
    z = zBlock * blockDepth + zGob * gobDepth + z;
    offsetLinear = z * pitch * linearHeight + y * pitch + x;

    return offsetLinear;
}

LwU32 offsetLinearToBL(LwU32 offsetLinear, LwU32 width, LwU32 height,
                     LwU32 logBlockWidth, LwU32 logBlockHeight, LwU32 logBlockDepth,
                     LwU32 logGobWidth, LwU32 logGobHeight, LwU32 logGobDepth, LwU32 pitch, LwU32 format)
{
    LW_STATUS status;
    LwBlockLinearImageInfo BlockLinearInfo;
    LwBlockLinearTexParams TexParams;
    LwU32 offsetBL;

    LwU32 linearRowBytes, linearHeight;
    LwU32 sliceBlocks, blockWidthGobs, blockSliceGobs, blockWidthBytes, blockHeight, blockDepth;
    LwU32 blockSizeBytes;

    LwU32 zBlock, yBlock, xBlock;
    LwU32 zGob, yGob, xGob;
    LwU32 z, y, x;
    LwU32 blockIdx, gobIdx, gobOffset;

    LwU32 gobWidthBytes = 1 << logGobWidth;
    LwU32 gobHeight = 1 << logGobHeight;
    LwU32 gobDepth = 1 << logGobDepth;
    LwU32 gobSliceBytes = 1 << (logGobWidth + logGobHeight);
    LwU32 gobSizeBytes = 1 << (logGobWidth + logGobHeight + logGobDepth);

    status = calwlateBlockLinearInfo(width, height,
                                     logBlockWidth, logBlockHeight, logBlockDepth,
                                     format, gobWidthBytes, gobHeight,
                                     &BlockLinearInfo,
                                     &TexParams);
    if (status != LW_OK)
    {
        return 0;
    }

    linearRowBytes = TexParams.dwTexelSize * TexParams.dwBaseWidth;
    linearHeight = TexParams.dwBaseHeight;
    if (pitch < linearRowBytes)
    {
        pitch = linearRowBytes;
    }

    sliceBlocks = BlockLinearInfo.xBlocks * BlockLinearInfo.yBlocks;

    blockWidthGobs = 1 << BlockLinearInfo.log2GobsPerBlock.x;
    blockSliceGobs = 1 << (BlockLinearInfo.log2GobsPerBlock.x + BlockLinearInfo.log2GobsPerBlock.y);

    blockWidthBytes = 1 << (logGobWidth + BlockLinearInfo.log2GobsPerBlock.x);
    blockHeight = 1 << (logGobHeight + BlockLinearInfo.log2GobsPerBlock.y);
    blockDepth = 1 << (logGobDepth + BlockLinearInfo.log2GobsPerBlock.z);
    blockSizeBytes = 1 << (logGobWidth + logGobHeight + logGobDepth +
                                 BlockLinearInfo.log2GobsPerBlock.x +
                                 BlockLinearInfo.log2GobsPerBlock.y +
                                 BlockLinearInfo.log2GobsPerBlock.z);

    decompose3(offsetLinear, pitch * linearHeight, pitch, &z, &y, &x);
    decompose3(x, blockWidthBytes, gobWidthBytes, &xBlock, &xGob, &x);
    decompose3(y, blockHeight, gobHeight, &yBlock, &yGob, &y);
    decompose3(z, blockDepth, gobDepth, &zBlock, &zGob, &z);

    gobOffset = z * gobSliceBytes + y * gobWidthBytes + x;
    gobIdx = zGob * blockSliceGobs + yGob * blockWidthGobs + xGob;
    blockIdx = zBlock * sliceBlocks + yBlock * BlockLinearInfo.xBlocks + xBlock;
    offsetBL = blockIdx * blockSizeBytes + gobIdx * gobSizeBytes + gobOffset;

    return offsetBL;
}

LwU32 coordToBL(LwU32 cx, LwU32 cy, LwU32 width, LwU32 height,
                LwU32 logBlockWidth, LwU32 logBlockHeight, LwU32 logBlockDepth,
                LwU32 logGobWidth, LwU32 logGobHeight, LwU32 logGobDepth, LwU32 format)
{
    LW_STATUS status;
    LwBlockLinearImageInfo BlockLinearInfo;
    LwBlockLinearTexParams TexParams;
    LwU32 offsetBL;
    LwU32 linearOffset;

    LwU32 gobWidthBytes = 1 << logGobWidth;
    LwU32 gobHeight = 1 << logGobHeight;

    status = calwlateBlockLinearInfo(width, height,
                                     logBlockWidth, logBlockHeight, logBlockDepth,
                                     format, gobWidthBytes, gobHeight,
                                     &BlockLinearInfo,
                                     &TexParams);
    if (status != LW_OK)
    {
        return 0;
    }

    linearOffset = cy * TexParams.dwTexelSize * TexParams.dwBaseWidth + cx * TexParams.dwTexelSize;

    offsetBL = offsetLinearToBL(linearOffset, width, height,
                    logBlockWidth, logBlockHeight, logBlockDepth,
                    logGobWidth, logGobHeight, logGobDepth, 0, format);

    return offsetBL;
}

LwU32 blToCoord(LwU32 offsetBL, LwU32 * cx, LwU32 * cy, LwU32 width, LwU32 height,
                LwU32 logBlockWidth, LwU32 logBlockHeight, LwU32 logBlockDepth,
                LwU32 logGobWidth, LwU32 logGobHeight, LwU32 logGobDepth, LwU32 format)
{
    LW_STATUS status;
    LwBlockLinearImageInfo BlockLinearInfo;
    LwBlockLinearTexParams TexParams;
    LwU32 offsetLinear;
    LwU32 x, y, z;
    LwU32 gobWidthBytes = 1 << logGobWidth;
    LwU32 gobHeight = 1 << logGobHeight;

    status = calwlateBlockLinearInfo(width, height,
                                     logBlockWidth, logBlockHeight, logBlockDepth,
                                     format, gobWidthBytes, gobHeight,
                                     &BlockLinearInfo,
                                     &TexParams);
    if (status != LW_OK)
    {
        return 0;
    }

    offsetLinear = offsetBLToLinear(offsetBL, width, height,
                        logBlockWidth, logBlockHeight, logBlockDepth,
                        logGobWidth, logGobHeight, logGobDepth, 0, format);
    decompose3(offsetLinear, TexParams.dwTexelSize * TexParams.dwBaseWidth * TexParams.dwBaseHeight,
        TexParams.dwTexelSize * TexParams.dwBaseWidth, &z, &y, &x);

    if (cx)
    {
        *cx = x / TexParams.dwTexelSize;
    }
    if (cy)
    {
        *cy = y;
    }

    return (((y & 0xFFFF) << 16) | (x & 0xFFFF));
}
