/*
 * Copyright 2008 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _HELP_H_
#define _HELP_H_

#include "os.h"

#if defined(WIN32)
static char g_szGvDispUsage[] =
{
    " gvdisp ChId vAddr width height log2BlockWidth/linearPitch log2BlockHeight log2BlockDepth format\n" \
    "                           - Opens a window to display the surface at offset and outputs out.bmp\n" \
    "                           - format = 0 - X8R8G8B8\n" \
    "                           - format = 1 - A8R8G8B8\n" \
    "                           - format = 2 - R5G6B5\n" \
    "                           - format = 3 - A1R5G5B5\n" \
    "                           - format = 4 - A16B16G16R16F\n" \
    "                           - format = 5 - R16F\n" \
    "                           - format = 6 - R16UN\n" \
    "                           - format = 7 - R32F\n" \
    "                           - format = 8 - A32B32G32R32F\n" \
    "                           - format = 9 - A2R10G10B10\n" \
    "                           - format = A - A4R4G4B4\n" \
    "                           - format = B - S8Z24\n" \
    "                           - format = C - Z24S8\n" \
    "                           - format = D - Y8\n" \
    "                           - format = E - YUY2\n" \
    "                           - format = F - UYVY\n" \
    "                           - format = 10 - A8B8G8R8\n" \
    "                           - format = 11 - R11G11B10F\n" \
    "                           - format = 12 - LW12\n" \
    "                           - format = 13 - LW24\n" \
    "                           - format = 14 - YV12\n" \
    "                           - format = 15 - UV\n" \
    "                           - format = 16 - R32G32F\n" \
    "                           - format = 17 - Z24X8_X16V8S8\n" \
    "                           - format = 18 - Z32F_X16V8S8\n" \
    "                           - format = 19 - DXT1\n" \
    "                           - format = 1A - DXT23\n" \
    "                           - format = 1B - DXT45\n" \
    "                           - format = 1C - AYUV\n" \
    "                           - format = 1D - P010\n" \
    "                           - format = 1E - Y16\n" \
    "                           - format = 1F - UV16\n" \
    "                           - format bit 31 - read alpha/stencil (if available)\n" \
    "                           - format bit 30 - read vcaa (if available)\n"
};

static char g_szGCo2blUsage[] =
{
    " gco2bl <cx> <cy> <width> <height> <logBlockWidth> <logBlockHeight> <logBlockDepth> <logGobWidth> <logGobHeight> <logGobDepth> <format>\n"
};

static char g_szGBl2coUsage[] =
{
    " gbl2co <bl offset> <width> <height> <logBlockWidth> <logBlockHeight> <logBlockDepth> <logGobWidth> <logGobHeight> <logGobDepth> <format>\n"
};
#endif

//
// help routines
// 
void printHelpMenu(void);

#endif // _HELP_H_

