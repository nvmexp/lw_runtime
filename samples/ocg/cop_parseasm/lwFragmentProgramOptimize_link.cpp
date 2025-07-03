/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2003-2017, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

//
// lwFragmentProgramOptimize_link.cpp
//
#ifdef WIN32
#include <windows.h>
#define MIN(X, Y) min(X, Y)
#define MAX(X, Y) max(X, Y)
#endif
#include <math.h>
#include <stdio.h>
#include <assert.h>

/* This bit encapsulates all lw40-specific behavior */
#define LW_OGL_GRAPHICS_CAPS_ARCH_SPECIFIC_LW40     0x00000100

#define LW40_OGL_GRAPHICS_CAPS_LWRIE \
        LW_OGL_GRAPHICS_CAPS_ARCH_SPECIFIC_LW40

#ifdef __cplusplus
extern "C" {
#endif
    char* __cdecl lwStrCat (char *szStr1, const char *szStr2);
#ifdef __cplusplus
}
#endif

// --------------------------------------------------------------
#define IS_OPENGL 1
#include "lwInst.h"
#include "Lwcm.h"
// --------------------------------------------------------------
#include "lwFragmentProgram.h" 
#include "lwFragmentProgram.c" 
#include "lwFragmentProgramOptimize.cpp"
#include "lwiworkarounds.c"


/*
 * fpPrintLwInst() - Print PS instruction format..
 *
 */
void fpPrintLwInst(const lwInst *pInsts, FILE *fOutFile, int fIsOutput32bit, lwiOptimizationProfile eProf)
{
    if (fOutFile) {
        int ii;
        const lwInst *lInst;
        char tmpBuf[512];
        fprintf(fOutFile, "BEGIN SHADER\n");
        fprintf(fOutFile, "DRAWFLAGS: OUT%c0\n", fIsOutput32bit ? 'R' : 'H');
        fprintf(fOutFile, "HASHINFO: 0x00000000\n");

        for (lInst=pInsts, ii = 0; lInst; lInst=lInst->next, ii++) {
            lwPrintInstructionStr(tmpBuf, lInst, eProf, 1);
            fprintf(fOutFile, "%s\n", tmpBuf);
        }
        fprintf(fOutFile, "END SHADER\n");
    }
} // fpPrintLwInst

/*
 * fpPrintSinglePSInst() - Print PS instruction
 *
 */
void fpPrintSinglePSInst(lwInst *pInst, FILE *fOutFile, lwiOptimizationProfile eProf)
{
    char tmpBuf[256];

    lwPrintInstructionStr(tmpBuf, pInst, eProf, 1);
    fprintf(fOutFile, "%s\n", tmpBuf);

} // fpPrintSinglePSInst
