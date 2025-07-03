/*
 * Copyright (c) 2021-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_VALIDATE_PLATFORM_COMMON_H
#define INCLUDED_LWSCIBUF_ATTR_VALIDATE_PLATFORM_COMMON_H

#include "lwscibuf_attr_priv.h"
#include "lwscicommon_covanalysis.h"
#include "lwscicommon_libc.h"
#include "lwscierror.h"
#include "lwscilog.h"

typedef LwSciError (*LwSciBufValidateAttrFn)(
    LwSciBufAttrList attrList,
    const void *val);

// Define backing datatype for each attribute key
// General attribute key type backing datatype
#define LW_SCI_BUF_GENKEYTYPE_TYPES LwSciBufType
#define LW_SCI_BUF_GENKEYTYPE_NEEDCPUACCESS bool
#define LW_SCI_BUF_GENKEYTYPE_REQUIREDPERM LwSciBufAttrValAccessPerm
#define LW_SCI_BUF_GENKEYTYPE_ENABLECPUCACHE bool
#define LW_SCI_BUF_GENKEYTYPE_GPUID LwSciRmGpuId
#define LW_SCI_BUF_GENKEYTYPE_CPUSWCACHECOHERENCY bool
#define LW_SCI_BUF_GENKEYTYPE_ACTUALPERM LwSciBufAttrValAccessPerm
#define LW_SCI_BUF_GENKEYTYPE_VIDMEMGPUID LwSciRmGpuId

// Rawbuffer attribute key type backing datatype
#define LW_SCI_BUF_RAWKEYTYPE_SIZE uint64_t
#define LW_SCI_BUF_RAWKEYTYPE_ALIGN uint64_t

// Image attribute key backing datatype
#define LW_SCI_BUF_IMGKEYTYPE_LAYOUT LwSciBufAttrValImageLayoutType
#define LW_SCI_BUF_IMGKEYTYPE_TOPPADDING uint64_t
#define LW_SCI_BUF_IMGKEYTYPE_BOTTOMPADDING uint64_t
#define LW_SCI_BUF_IMGKEYTYPE_LEFTPADDING uint64_t
#define LW_SCI_BUF_IMGKEYTYPE_RIGHTPADDING uint64_t
#define LW_SCI_BUF_IMGKEYTYPE_VPRFLAG bool
#define LW_SCI_BUF_IMGKEYTYPE_SIZE uint64_t
#define LW_SCI_BUF_IMGKEYTYPE_ALIGNMENT uint64_t
#define LW_SCI_BUF_IMGKEYTYPE_PLANECOUNT uint32_t
#define LW_SCI_BUF_IMGKEYTYPE_PLANECOLORFMT LwSciBufAttrValColorFmt
#define LW_SCI_BUF_IMGKEYTYPE_PLANECOLORSTD LwSciBufAttrValColorStd
#define LW_SCI_BUF_IMGKEYTYPE_PLANEADDRALIGN uint32_t
#define LW_SCI_BUF_IMGKEYTYPE_PLANEWIDTH uint32_t
#define LW_SCI_BUF_IMGKEYTYPE_PLANEHEIGHT uint32_t
#define LW_SCI_BUF_IMGKEYTYPE_SCANTYPE LwSciBufAttrValImageScanType
#define LW_SCI_BUF_IMGKEYTYPE_PLANEBITSPERPIXEL uint32_t
#define LW_SCI_BUF_IMGKEYTYPE_PLANEOFFSET uint64_t
#define LW_SCI_BUF_IMGKEYTYPE_PLANEDATATYPE LwSciBufAttrValDataType
#define LW_SCI_BUF_IMGKEYTYPE_PLANECHANNELCOUNT uint8_t
#define LW_SCI_BUF_IMGKEYTYPE_PLANESECONDOFFSET uint64_t
#define LW_SCI_BUF_IMGKEYTYPE_PLANEPITCH uint32_t
#define LW_SCI_BUF_IMGKEYTYPE_PLANEALIGNEDHEIGHT uint32_t
#define LW_SCI_BUF_IMGKEYTYPE_PLANEALIGNEDSIZE uint64_t
#define LW_SCI_BUF_IMGKEYTYPE_IMAGECOUNT uint64_t
#define LW_SCI_BUF_IMGKEYTYPE_SURFTYPE LwSciBufSurfType
#define LW_SCI_BUF_IMGKEYTYPE_SURFMEMLAYOUT LwSciBufSurfMemLayout
#define LW_SCI_BUF_IMGKEYTYPE_SURFSAMPLETYPE LwSciBufSurfSampleType
#define LW_SCI_BUF_IMGKEYTYPE_SURFBPC LwSciBufSurfBPC
#define LW_SCI_BUF_IMGKEYTYPE_SURFCOMPONENTORDER LwSciBufSurfComponentOrder
#define LW_SCI_BUF_IMGKEYTYPE_SURFWIDTHBASE uint32_t
#define LW_SCI_BUF_IMGKEYTYPE_SURFHEIGHTBASE uint32_t

// Tensor attribute key backing datatype
#define LW_SCI_BUF_TENSKEYTYPE_TYPE LwSciBufAttrValDataType
#define LW_SCI_BUF_TENSKEYTYPE_NUMDIMS uint32_t
#define LW_SCI_BUF_TENSKEYTYPE_SIZEPERDIM uint64_t
#define LW_SCI_BUF_TENSKEYTYPE_ALIGNMENTPERDIM uint32_t
#define LW_SCI_BUF_TENSKEYTYPE_STRIDESPERDIM uint64_t
#define LW_SCI_BUF_TENSKEYTYPE_PIXELFORMAT LwSciBufAttrValColorFmt
#define LW_SCI_BUF_TENSKEYTYPE_BASEADDRALIGN uint64_t
#define LW_SCI_BUF_TENSKEYTYPE_SIZE uint64_t

// Array attribute key backing datatype
#define LW_SCI_BUF_ARRKEYTYPE_TYPE LwSciBufAttrValDataType
#define LW_SCI_BUF_ARRKEYTYPE_STRIDE uint64_t
#define LW_SCI_BUF_ARRKEYTYPE_CAPACITY uint64_t
#define LW_SCI_BUF_ARRKEYTYPE_SIZE uint64_t
#define LW_SCI_BUF_ARRKEYTYPE_ALIGNMENT uint64_t

// Pyramid attribute key backing datatype
#define LW_SCI_BUF_PYRKEYTYPE_NUMLEVELS uint32_t
#define LW_SCI_BUF_PYRKEYTYPE_SCALE float
#define LW_SCI_BUF_PYRKEYTYPE_LEVELOFFSET uint64_t
#define LW_SCI_BUF_PYRKEYTYPE_LEVELSIZE uint64_t
#define LW_SCI_BUF_PYRKEYTYPE_ALIGNMENT uint64_t

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 20_10), "LwSciBuf-ADV-MISRAC2012-021")
#define RANGE_CHECK_MIN(name, min, type) \
    static inline LwSciError LwSciBufValidate##name(LwSciBufAttrList attrList, \
        const void *val) { \
        LwSciError sciErr = LwSciError_Success; \
        type value = *(const type *)val; \
        if ((value < (type)(min))) { \
            LWSCI_ERR_STR("Validation check failed\n"); \
            sciErr = LwSciError_BadParameter; \
        } \
        (void)attrList; \
        return sciErr; \
    }
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 20_10))

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 20_10), "LwSciBuf-ADV-MISRAC2012-021")
#define RANGE_CHECK(name, min, max, type) \
    static inline LwSciError LwSciBufValidate##name(LwSciBufAttrList attrList, const void *val) { \
        LwSciError sciErr = LwSciError_Success; \
        type value; \
        LwSciCommonMemcpyS(&value, sizeof(type), val, sizeof(type)); \
        if ((value < (type)(min)) || (value > (type)(max))) { \
            LWSCI_ERR_STR("Validation check failed\n"); \
            sciErr = LwSciError_BadParameter; \
        } \
        (void)attrList; \
        return sciErr; \
    }
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 20_10))

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 20_10), "LwSciBuf-ADV-MISRAC2012-021")
#define ALIGN_UINT_CHECK(bitcount) \
    static inline LwSciError LwSciBufValidateAlignmentU##bitcount( \
                            LwSciBufAttrList attrList, const void *val) { \
        LwSciError sciErr = LwSciError_Success; \
        uint##bitcount##_t value = *(const uint##bitcount##_t*)val; \
        if ((value == 0U) || ((value & (value - 1U)) != 0U)) { \
            LWSCI_ERR_STR("Alignment check failed. Must be a power of 2\n"); \
            sciErr = LwSciError_BadParameter; \
        } \
        (void)attrList; \
        return sciErr; \
    }
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 20_10))

RANGE_CHECK_MIN(BufferSize, 1, uint64_t)
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
RANGE_CHECK(BufferType, LwSciBufType_RawBuffer, LwSciBufType_Pyramid, int32_t)
RANGE_CHECK(ImageCount, 1, 1, uint64_t)
RANGE_CHECK(LayoutType, LwSciBufImage_BlockLinearType, LwSciBufImage_PitchLinearType, int32_t)
RANGE_CHECK(PlaneCount, 1, LW_SCI_BUF_IMAGE_MAX_PLANES, int32_t)
RANGE_CHECK(PlaneColorFmt, (int32_t)LwSciColor_LowerBound+1, (int32_t)LwSciColor_UpperBound-1, int32_t)
RANGE_CHECK(PlaneColorStd, LwSciColorStd_SRGB, LwSciColorStd_REQ2020PQ_ER, int32_t)
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
RANGE_CHECK(ScanType, LwSciBufScan_ProgressiveType, LwSciBufScan_InterlaceType, int32_t)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
RANGE_CHECK_MIN(ImageWidth, 1, uint32_t)
RANGE_CHECK_MIN(ImageHeight, 1, uint32_t)
RANGE_CHECK_MIN(BitsPerPixel, 1, uint32_t)
RANGE_CHECK(PlaneDataType, LwSciDataType_Int4, LwSciDataType_Bool, LwSciBufAttrValDataType)
RANGE_CHECK_MIN(PlaneChannelCount, 1, uint8_t)
RANGE_CHECK_MIN(PlanePitch, 1, uint32_t)
RANGE_CHECK_MIN(PlaneAlignedHeight, 1, uint32_t)
RANGE_CHECK(SurfType, (int32_t)LwSciSurfType_YUV, (int32_t)LwSciSurfType_YUV, int32_t)
RANGE_CHECK(SurfMemLayout, (int32_t)LwSciSurfMemLayout_SemiPlanar, (int32_t)LwSciSurfMemLayout_MaxValid - 1, int32_t)
RANGE_CHECK(SurfSampleType, (int32_t)LwSciSurfSampleType_420, (int32_t)LwSciSurfSampleType_MaxValid - 1, int32_t)
RANGE_CHECK(SurfBPC, (int32_t)LwSciSurfBPC_Layout_16_8_8, (int32_t)LwSciSurfBPC_MaxValid - 1, int32_t)
RANGE_CHECK(SurfComponentOrder, (int32_t)LwSciSurfComponentOrder_YUV, (int32_t)LwSciSurfComponentOrder_MaxValid - 1, int32_t)
RANGE_CHECK(TensorDataType, LwSciDataType_Int4, LwSciDataType_Float32, LwSciBufAttrValDataType)
RANGE_CHECK_MIN(TensorSizePerDim, 1, uint64_t)
RANGE_CHECK_MIN(TensorStridePerDim, 1, uint64_t)
RANGE_CHECK(NumDims, 1, LW_SCI_BUF_TENSOR_MAX_DIMS, int32_t)
RANGE_CHECK_MIN(ArrayStride, 1, uint64_t)
RANGE_CHECK_MIN(ArrayCapacity, 1, uint64_t)
RANGE_CHECK(ArrayDataType, LwSciDataType_Int4, LwSciDataType_Bool, int32_t)
RANGE_CHECK(PyramidLevels, 1, LW_SCI_BUF_PYRAMID_MAX_LEVELS, int32_t)
RANGE_CHECK(GpuCompressionExternal, LwSciBufCompressionType_GenericCompressible,
    LwSciBufCompressionType_GenericCompressible, LwSciBufCompressionType)
RANGE_CHECK(GpuCompressionInternal, LwSciBufCompressionType_None,
    LwSciBufCompressionType_GenericCompressible, LwSciBufCompressionType)

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
ALIGN_UINT_CHECK(64)
ALIGN_UINT_CHECK(32)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static inline LwSciError LwSciBufValidatePyramidScale(
    const LwSciBufAttrList attrList,
    const void *val)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError sciErr = LwSciError_Success;
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciBuf-ADV-MISRAC2012-006")
    float value = *(const float *)val;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    if ((value <= 0.0F) || (value > 1.0F)) {
        LWSCI_ERR_STR("Validation check failed\n");
        sciErr = LwSciError_BadParameter;
    }

    (void)attrList;

    return sciErr;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static inline LwSciError LwSciBufValidatePermission(
    const LwSciBufAttrList attrList,
    const void *val)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError sciErr = LwSciError_Success;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    int32_t value = *(const int32_t*)val;
    bool validPermission = ((value == (int32_t)LwSciBufAccessPerm_Readonly) ||
                            (value == (int32_t)LwSciBufAccessPerm_ReadWrite));
    if (!validPermission) {
        LWSCI_ERR_STR("Invalid permission attribute value.\n");
        sciErr = LwSciError_BadParameter;
    }

    (void)attrList;

    return sciErr;
}

static inline LwSciError LwSciBufValidateBool(
    LwSciBufAttrList attrList,
    const void *val)
{
    LwSciError sciErr = LwSciError_Success;
    bool isTrue = true;
    bool isFalse = false;

    (void)attrList;

    if (LwSciCommonMemcmp(val, &isTrue, sizeof(bool)) != 0 &&
        LwSciCommonMemcmp(val, &isFalse, sizeof(bool)) != 0) {
        LWSCI_ERR_STR("Invalid boolean value.");
        sciErr = LwSciError_BadParameter;
    }

    return sciErr;
}

LwSciError LwSciBufValidateGpuId(
    const LwSciBufAttrList attrList,
    const void *val);

LwSciError LwSciBufValidateAttrValGpuCache(
    LwSciBufAttrList attrList,
    const void* val);

LwSciError LwSciBufValidateAttrValGpuCompressionExternal(
    LwSciBufAttrList attrList,
    const void* val);

LwSciError LwSciBufValidateAttrValGpuCompressionInternal(
    LwSciBufAttrList attrList,
    const void* val);

#endif /* INCLUDED_LWSCIBUF_ATTR_VALIDATE_PLATFORM_COMMON_H */
