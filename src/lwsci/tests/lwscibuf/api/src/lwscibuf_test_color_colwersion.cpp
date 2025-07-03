/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <stdio.h>
#if defined(__x86_64__)
#include "mobile_common.h"
#endif
#include <lwscibuf_colorcolwersion.h>
//This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0
#include "gtest/gtest.h"

TEST(LwSciCommon, LwSciColorToLwColor)
{
    LwSciBufAttrValColorFmt lwsciColor;
    LwColorFormat lwColor;
    LwColorFormat lwColorDesired;
    lwsciColor = LwSciColor_LowerBound;

    EXPECT_EQ(LwSciColorToLwColor(lwsciColor, &lwColor), LwSciError_BadParameter);

    lwsciColor = LwSciColor_A32;
    lwColorDesired = LwColorFormat_A32;
    EXPECT_EQ(LwSciColorToLwColor(lwsciColor, &lwColor), LwSciError_Success);
    EXPECT_EQ(lwColor, lwColorDesired);

    lwsciColor = LwSciColor_X12Bayer20CCCC;
    lwColorDesired = LwColorFormat_X12Bayer20CCCC;
    EXPECT_EQ(LwSciColorToLwColor(lwsciColor, &lwColor), LwSciError_Success);
    EXPECT_EQ(lwColor, lwColorDesired);

    lwsciColor = LwSciColor_V16;
    lwColorDesired = LwColorFormat_V16;
    EXPECT_EQ(LwSciColorToLwColor(lwsciColor, &lwColor), LwSciError_Success);
    EXPECT_EQ(lwColor, lwColorDesired);
}

TEST(LwSciCommon, LwColorToLwSciColor)
{
    LwSciBufAttrValColorFmt lwsciColor;
    LwSciBufAttrValColorFmt lwsciColorDesired;
    LwColorFormat lwColor;

    lwColor = (LwColorFormat) (-1);
    EXPECT_EQ(LwColorToLwSciColor(lwColor, &lwsciColor), LwSciError_BadParameter);

    lwColor = LwColorFormat_X4Bayer12BGGR;
    lwsciColorDesired = LwSciColor_X4Bayer12BGGR;
    EXPECT_EQ(LwColorToLwSciColor(lwColor, &lwsciColor), LwSciError_Success) ;
    EXPECT_EQ(lwsciColor, lwsciColorDesired);

    lwColor = LwColorFormat_V10U10;
    lwsciColorDesired = LwSciColor_V10U10;
    EXPECT_EQ(LwColorToLwSciColor(lwColor, &lwsciColor), LwSciError_Success) ;
    EXPECT_EQ(lwsciColor, lwsciColorDesired);

    lwColor = LwColorFormat_YUYV;
    lwsciColorDesired = LwSciColor_Y8U8Y8V8;
    EXPECT_EQ(LwColorToLwSciColor(lwColor, &lwsciColor), LwSciError_Success) ;
    EXPECT_EQ(lwsciColor, lwsciColorDesired);
}
