// lwBlockLinear.c
//
// utility functions for handling block-linear surfaces.
// see lwBlockLinear.h for a lot more gorey detail

// lwGetBlockLinearTexLevelInfo:  Return block linear information (via "pBlockLinearInfo")
// for the texture whose properties are specified in "pTexParams"

#include "lwtypes.h"
#include "lwBlockLinear.h"

#ifndef MAX
#define MAX(a, b)   (((a) > (b)) ? (a) : (b))
#endif

void lwGetBlockLinearTexLevelInfo (LwBlockLinearImageInfo *pBlockLinearInfo,    // in & out
                                   LwBlockLinearTexParams *pTexParams)          // in
{
    const LwU32 baseW = pTexParams->dwBaseWidth;
    const LwU32 baseH = pTexParams->dwBaseHeight;
    const LwU32 baseD = pTexParams->dwBaseDepth;
    const LwU32 blockWLog2 = pTexParams->dwBlockWidthLog2;
    const LwU32 blockHLog2 = pTexParams->dwBlockHeightLog2;

    LwU32 dwBorderW, dwBorderH, dwBorderD;
    LwU32 dwOffset, dwLevel;
    int   w, h ,d;

    // Compute the initial texture size parameters.
    dwBorderW = 2 * pTexParams->dwBorderSize;
    dwBorderH = (pTexParams->dwDimensionality > 1) ? dwBorderW : 0;
    dwBorderD = (pTexParams->dwDimensionality > 2) ? dwBorderW : 0;

    // Make sure we don't have a compressed, bordered texture.  Not lwrrently
    // legal for real compressed texture formats.  The code below doesn't try
    // to handle that case.
    if (!((dwBorderW == 0) || ((blockWLog2 == 0) && (blockHLog2 == 0))))
        return;

    // Loop over the levels (including the level in question), computing sizes
    // and adding to the offset.
    dwOffset = 0;
    for (dwLevel = 0; dwLevel <= pTexParams->dwLOD; dwLevel++) {
        w = MAX(1, (baseW >> dwLevel)) + dwBorderW;
        h = MAX(1, (baseH >> dwLevel)) + dwBorderH;
        d = MAX(1, (baseD >> dwLevel)) + dwBorderD;
        // Be sure to round up properly for compressed formats.
        w = (w + (1 << blockWLog2) - 1) >> blockWLog2;
        h = (h + (1 << blockHLog2) - 1) >> blockHLog2;
        lwGetBlockLinearImageInfo (pBlockLinearInfo, w, h, d, pTexParams->dwTexelSize);
        dwOffset += pBlockLinearInfo->size;
    }

    // The final offset is the end of the level in question.  Roll back to the
    // beginning of that level.
    pBlockLinearInfo->offset = dwOffset - pBlockLinearInfo->size;

    // The mipmap offset is relative to face zero.  Add in the offset to the
    // proper face.
    pBlockLinearInfo->offset += pTexParams->dwFace * pTexParams->dwFaceSize;

    // The other parameters already correspond to the level in question.  Note
    // that the gobs-per-block parameters may be different than those of the
    // level zero mipmap.
}

