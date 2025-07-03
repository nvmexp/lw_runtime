/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <stddef.h>

#include "lwscibuf_colorcolwersion.h"
#include "lwscilog.h"
#include "lwscicommon_libc.h"

/* This macro will expand into colwersion lookup array and
 * assign values.
 * This macro will create a array of uint64_t colorColwersionArray[];
 */
#define LW_SCI_COLOR_TABLE \
    LW_SCI_COLOR_ENTRY(LwSciColor_LowerBound, LwColorFormat_Unspecified, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer8RGGB, LwColorFormat_Bayer8RGGB, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer8CCCC, LwColorFormat_Bayer8CCCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer8BGGR, LwColorFormat_Bayer8BGGR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer8GBRG, LwColorFormat_Bayer8GBRG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer8GRBG, LwColorFormat_Bayer8GRBG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16BGGR, LwColorFormat_Bayer16BGGR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16CCCC, LwColorFormat_Bayer16CCCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16GBRG, LwColorFormat_Bayer16GBRG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16GRBG, LwColorFormat_Bayer16GRBG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16RGGB, LwColorFormat_Bayer16RGGB, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16RCCB, LwColorFormat_Bayer16RCCB, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16BCCR, LwColorFormat_Bayer16BCCR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16CRBC, LwColorFormat_Bayer16CRBC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16CBRC, LwColorFormat_Bayer16CBRC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16RCCC, LwColorFormat_Bayer16RCCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16CCCR, LwColorFormat_Bayer16CCCR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16CRCC, LwColorFormat_Bayer16CRCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16CCRC, LwColorFormat_Bayer16CCRC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X2Bayer14GBRG, LwColorFormat_X2Bayer14GBRG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12GBRG, LwColorFormat_X4Bayer12GBRG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12RCCB, LwColorFormat_X4Bayer12RCCB, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12BCCR, LwColorFormat_X4Bayer12BCCR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12CRBC, LwColorFormat_X4Bayer12CRBC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12CBRC, LwColorFormat_X4Bayer12CBRC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12RCCC, LwColorFormat_X4Bayer12RCCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12CCCR, LwColorFormat_X4Bayer12CCCR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12CRCC, LwColorFormat_X4Bayer12CRCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12CCRC, LwColorFormat_X4Bayer12CCRC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X6Bayer10GBRG, LwColorFormat_X6Bayer10GBRG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X2Bayer14GRBG, LwColorFormat_X2Bayer14GRBG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12GRBG, LwColorFormat_X4Bayer12GRBG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X6Bayer10GRBG, LwColorFormat_X6Bayer10GRBG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X2Bayer14BGGR, LwColorFormat_X2Bayer14BGGR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12BGGR, LwColorFormat_X4Bayer12BGGR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X6Bayer10BGGR, LwColorFormat_X6Bayer10BGGR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X2Bayer14RGGB, LwColorFormat_X2Bayer14RGGB, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12RGGB, LwColorFormat_X4Bayer12RGGB, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X6Bayer10RGGB, LwColorFormat_X6Bayer10RGGB, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X2Bayer14CCCC, LwColorFormat_X2Bayer14CCCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12CCCC, LwColorFormat_X4Bayer12CCCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X6Bayer10CCCC, LwColorFormat_X6Bayer10CCCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Signed_X2Bayer14CCCC, LwColorFormat_Signed_X2Bayer14CCCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Signed_X4Bayer12CCCC, LwColorFormat_Signed_X4Bayer12CCCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Signed_X12Bayer20CCCC, LwColorFormat_Signed_X12Bayer20CCCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Signed_X6Bayer10CCCC, LwColorFormat_Signed_X6Bayer10CCCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Signed_Bayer16CCCC, LwColorFormat_Signed_Bayer16CCCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_FloatISP_Bayer16CCCC, LwColorFormat_FloatISP_Bayer16CCCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_FloatISP_Bayer16RGGB, LwColorFormat_FloatISP_Bayer16RGGB, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_FloatISP_Bayer16BGGR, LwColorFormat_FloatISP_Bayer16BGGR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_FloatISP_Bayer16GRBG, LwColorFormat_FloatISP_Bayer16GRBG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_FloatISP_Bayer16GBRG, LwColorFormat_FloatISP_Bayer16GBRG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_FloatISP_Bayer16RCCB, LwColorFormat_FloatISP_Bayer16RCCB, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_FloatISP_Bayer16BCCR, LwColorFormat_FloatISP_Bayer16BCCR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_FloatISP_Bayer16CRBC, LwColorFormat_FloatISP_Bayer16CRBC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_FloatISP_Bayer16CBRC, LwColorFormat_FloatISP_Bayer16CBRC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_FloatISP_Bayer16RCCC, LwColorFormat_FloatISP_Bayer16RCCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_FloatISP_Bayer16CCCR, LwColorFormat_FloatISP_Bayer16CCCR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_FloatISP_Bayer16CRCC, LwColorFormat_FloatISP_Bayer16CRCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_FloatISP_Bayer16CCRC, LwColorFormat_FloatISP_Bayer16CCRC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X12Bayer20CCCC, LwColorFormat_X12Bayer20CCCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X12Bayer20BGGR, LwColorFormat_X12Bayer20BGGR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X12Bayer20RGGB, LwColorFormat_X12Bayer20RGGB, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X12Bayer20GRBG, LwColorFormat_X12Bayer20GRBG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X12Bayer20GBRG, LwColorFormat_X12Bayer20GBRG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X12Bayer20RCCB, LwColorFormat_X12Bayer20RCCB, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X12Bayer20BCCR, LwColorFormat_X12Bayer20BCCR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X12Bayer20CRBC, LwColorFormat_X12Bayer20CRBC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X12Bayer20CBRC, LwColorFormat_X12Bayer20CBRC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X12Bayer20RCCC, LwColorFormat_X12Bayer20RCCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X12Bayer20CCCR, LwColorFormat_X12Bayer20CCCR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X12Bayer20CRCC, LwColorFormat_X12Bayer20CRCC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X12Bayer20CCRC, LwColorFormat_X12Bayer20CCRC, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_U8V8, LwColorFormat_U8V8, 2U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_U8_V8, LwColorFormat_U8_V8, 2U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_V8U8, LwColorFormat_V8U8, 2U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_V8_U8, LwColorFormat_V8_U8, 2U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_U10V10, LwColorFormat_U10V10, 2U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_V10U10, LwColorFormat_V10U10, 2U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_U12V12, LwColorFormat_U12V12, 2U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_V12U12, LwColorFormat_V12U12, 2U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_U16V16, LwColorFormat_U16V16, 2U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_V16U16, LwColorFormat_V16U16, 2U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Y8, LwColorFormat_Y8, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Y10, LwColorFormat_Y10, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Y12, LwColorFormat_Y12, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Y16, LwColorFormat_Y16, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_U8, LwColorFormat_U8, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_V8, LwColorFormat_V8, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_U10, LwColorFormat_U10, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_V10, LwColorFormat_V10, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_U12, LwColorFormat_U12, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_V12, LwColorFormat_V12, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_U16, LwColorFormat_U16, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_V16, LwColorFormat_V16, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_A8Y8U8V8, LwColorFormat_A8Y8U8V8, 4U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Y8U8Y8V8, LwColorFormat_YUYV, 4U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Y8V8Y8U8, LwColorFormat_YVYU, 4U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_U8Y8V8Y8, LwColorFormat_UYVY, 4U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_V8Y8U8Y8, LwColorFormat_VYUY, 4U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_A16Y16U16V16, LwColorFormat_A16Y16U16V16, 4U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_A8, LwColorFormat_A8, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Signed_A8, LwColorFormat_Signed_A8, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_B8G8R8A8, LwColorFormat_B8G8R8A8, 4U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_A8R8G8B8, LwColorFormat_A8R8G8B8, 4U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_A8B8G8R8, LwColorFormat_A8B8G8R8, 4U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_A2R10G10B10, LwColorFormat_A2R10G10B10, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_A16, LwColorFormat_A16, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Signed_A16, LwColorFormat_Signed_A16, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Signed_R16G16, LwColorFormat_Signed_R16G16, 2U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_A16B16G16R16, LwColorFormat_A16B16G16R16, 4U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Signed_A16B16G16R16, LwColorFormat_Signed_A16B16G16R16, 4U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Float_A16B16G16R16, LwColorFormat_Float_A16B16G16R16, 4U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_A32, LwColorFormat_A32, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Signed_A32, LwColorFormat_Signed_A32, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Signed_X12Bayer20GBRG, LwColorFormat_Unspecified, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Float_A16, LwColorFormat_Float_A16, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X6Bayer10BGGI_RGGI, LwColorFormat_X6Bayer10BGGI_RGGI, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X6Bayer10GBIG_GRIG, LwColorFormat_X6Bayer10GBIG_GRIG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X6Bayer10GIBG_GIRG, LwColorFormat_X6Bayer10GIBG_GIRG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X6Bayer10IGGB_IGGR, LwColorFormat_X6Bayer10IGGB_IGGR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X6Bayer10RGGI_BGGI, LwColorFormat_X6Bayer10RGGI_BGGI, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X6Bayer10GRIG_GBIG, LwColorFormat_X6Bayer10GRIG_GBIG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X6Bayer10GIRG_GIBG, LwColorFormat_X6Bayer10GIRG_GIBG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X6Bayer10IGGR_IGGB, LwColorFormat_X6Bayer10IGGR_IGGB, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12BGGI_RGGI, LwColorFormat_X4Bayer12BGGI_RGGI, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12GBIG_GRIG, LwColorFormat_X4Bayer12GBIG_GRIG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12GIBG_GIRG, LwColorFormat_X4Bayer12GIBG_GIRG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12IGGB_IGGR, LwColorFormat_X4Bayer12IGGB_IGGR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12RGGI_BGGI, LwColorFormat_X4Bayer12RGGI_BGGI, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12GRIG_GBIG, LwColorFormat_X4Bayer12GRIG_GBIG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12GIRG_GIBG, LwColorFormat_X4Bayer12GIRG_GIBG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X4Bayer12IGGR_IGGB, LwColorFormat_X4Bayer12IGGR_IGGB, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X2Bayer14BGGI_RGGI, LwColorFormat_X2Bayer14BGGI_RGGI, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X2Bayer14GBIG_GRIG, LwColorFormat_X2Bayer14GBIG_GRIG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X2Bayer14GIBG_GIRG, LwColorFormat_X2Bayer14GIBG_GIRG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X2Bayer14IGGB_IGGR, LwColorFormat_X2Bayer14IGGB_IGGR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X2Bayer14RGGI_BGGI, LwColorFormat_X2Bayer14RGGI_BGGI, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X2Bayer14GRIG_GBIG, LwColorFormat_X2Bayer14GRIG_GBIG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X2Bayer14GIRG_GIBG, LwColorFormat_X2Bayer14GIRG_GIBG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_X2Bayer14IGGR_IGGB, LwColorFormat_X2Bayer14IGGR_IGGB, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16BGGI_RGGI, LwColorFormat_Bayer16BGGI_RGGI, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16GBIG_GRIG, LwColorFormat_Bayer16GBIG_GRIG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16GIBG_GIRG, LwColorFormat_Bayer16GIBG_GIRG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16IGGB_IGGR, LwColorFormat_Bayer16IGGB_IGGR, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16RGGI_BGGI, LwColorFormat_Bayer16RGGI_BGGI, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16GRIG_GBIG, LwColorFormat_Bayer16GRIG_GBIG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16GIRG_GIBG, LwColorFormat_Bayer16GIRG_GIBG, 1U) \
    LW_SCI_COLOR_ENTRY(LwSciColor_Bayer16IGGR_IGGB, LwColorFormat_Bayer16IGGR_IGGB, 1U)

typedef struct {
    LwColorFormat lwColor;
    uint8_t componentCount;
} LwSciColorMap;

#define LW_SCI_COLOR_ENTRY(_lwscicolor, _lwcolor, _componentcount) \
    [(_lwscicolor)] = {.lwColor = (_lwcolor), .componentCount = (_componentcount)},

static const LwSciColorMap colorColwersionArray[LwSciColor_UpperBound] = { \
        LW_SCI_COLOR_TABLE
    };

LwSciError LwSciColorToLwColor(
    LwSciBufAttrValColorFmt lwSciColorFmt,
    LwColorFormat* lwColorFmt)
{
    LwSciError err;

    LWSCI_FNENTRY("");

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    if ((NULL == lwColorFmt) || !LWSCI_IS_VALID_COLOR(lwSciColorFmt)) {
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    *lwColorFmt = colorColwersionArray[(uint32_t)lwSciColorFmt].lwColor;

    err = LwSciError_Success;

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwColorToLwSciColor(
    LwColorFormat lwColorFmt,
    LwSciBufAttrValColorFmt* lwSciColorFmt)
{
    LwSciError err = LwSciError_Success;
    uint32_t idx = 0;

    LWSCI_FNENTRY("");

    if (NULL == lwSciColorFmt) {
        err = LwSciError_BadParameter;

        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* TODO optimize this approach for performance */
    for (idx = (uint32_t)LwSciColor_LowerBound+1U; idx < (uint32_t)LwSciColor_UpperBound;
            idx++) {
        if (lwColorFmt == colorColwersionArray[idx].lwColor) {
            LwSciCommonMemcpyS(lwSciColorFmt, sizeof(*lwSciColorFmt),
                                                            &idx, sizeof(idx));
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    /* We reached this point means there is no LwSciBufAttrValColorFmt defined
     * for input LwColorFormat. Return Error.
     */
    err = LwSciError_BadParameter;

ret:
    LWSCI_FNEXIT("");
    return err;
}

/* Get Number of compoents in LwSciColorFormat. */
LwSciError LwSciColorGetComponentCount(
    LwSciBufAttrValColorFmt lwSciColorFmt,
    uint8_t* componentCount)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    if ((NULL == componentCount) || !LWSCI_IS_VALID_COLOR(lwSciColorFmt)) {
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    *componentCount = colorColwersionArray[lwSciColorFmt].componentCount;

ret:
    LWSCI_FNEXIT("");
    return err;
}
