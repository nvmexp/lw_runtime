/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2003-2020, LWPU CORPORATION.  All rights reserved.
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
// parseasmutils.cpp
//
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdarg.h>


// --------------------------------------------------------------

#include "cop_parseasm.h"
#include "parseasmfun.h"
#include "parseasm_profileoption.h"

// --------------------------------------------------------------

#include "cop_lwinst_info.h"
#include "copi_ucode.h"

#define LWI_INST_NUMARGS_ARGMASK(OPCODE, NUMARGS, SCALAR, SWIZZLEOK, ARG0MASK, ARG1MASK, \
    ARG2MASK, ISTEX, ISBLEND) NUMARGS, ARG0MASK, ARG1MASK, ARG2MASK

static struct {
    int dwNumArgs;
    lwiMask aeReadMask[3]; // a null mask means that it is a "passthrough" read--otherwise, only those selected components are needed
} fpInstOpInfo[] = {
    LWI_INST_INFO(LWI_INST_NUMARGS_ARGMASK)
};

/*
 * __fpMemAlloc() - Wrapper function to add allocate memory.
 *
 */

void *__fpMemAlloc(void *fArg, size_t size)
{
    return malloc(size);
} // __fpMemAlloc

/*
 * fpEmitError() - Wrapper function to add errors to the output.
 *
 */

void fpEmitError(void *fArg, SourceLoc *fLoc, int fNum, const char *fFmt, va_list args)
{
    if (fLoc)
        fprintf((FILE *)fArg, "Line: %d ", fLoc->line);
    vfprintf((FILE *)fArg, fFmt, args);
} // fpEmitError

/*
 * fpEmitWarning() - Wrapper function to add warnings to the output.
 *
 */

void fpEmitWarning(void *fArg, SourceLoc *fLoc, int fNum, const char *fFmt, va_list args)
{
    if (fLoc)
        fprintf((FILE *)fArg, "Line: %d ", fLoc->line);
    vfprintf((FILE *)fArg, fFmt, args);
} // fpEmitWarning

static void *lMalloc(void *t, size_t size)
{
    return malloc(size);
}

static void lFree(void *t, void *ptr)
{
    free(ptr);
}

/*
 * SetCOPBaseArgs() - Set some of the COPBaseArgs values from options in a common place.
 *
 */

void SetCOPBaseArgs(COPBaseArgs *args, lwInst **pOutInsts, lwInst **pOutStart,
                    ParamsForCOP *fParams)
{
    static OptimizerTimer theTimer;
    lwiShaderType shaderType;

    shaderType = args->shaderType;
    InitCOPBaseArgs(args, fParams);
    args->shaderType = shaderType;

    args->outInsts = pOutInsts;
    args->outStart = pOutStart;
    args->eOutputMode = lwiOutputMode(-1);
    args->eProf = lwiOptimizationProfile(o_eProf);
    args->dwOutputMask = 0;
    args->bUseDepth = 0;
    args->bUseCoverage = 0;
    args->dwNumCombiners = o_NumCombiners;
    args->pTimer = &theTimer;
    args->fTimeout = 1000;
    args->memParams.pfnAlloc = lMalloc;
    args->memParams.pfnFree = lFree;
    args->memParams.pAllocArg = NULL; // not needed
    args->memParams.pFreeArg = NULL; // not needed
    args->cycleCount.dwNumCycles = -1;
    args->cycleCount.fNumCyclesEffective = -1;
    args->cycleCount.bFasttex = 0;
    args->dwMaxPassInstrCount = -1;
    args->qwHash = 0;
    args->dwLevel = 99;
    args->bAssume0 = o_oglOpt;
    args->dwSupportsSignedRemap = o_SupportSignedRemap;
    args->dwDoSignedRemap = 0;
    args->dwTextureRange = o_TextureRange;
    args->bCanPromoteFixed =          o_oglOpt ? FALSE : FALSE;
    args->bCanPromoteHalf =           o_oglOpt ? TRUE  : TRUE;
    args->bCanReorderFixedWithCheck = o_oglOpt ? FALSE : TRUE;
    args->bCanReorderHalf =           o_oglOpt ? TRUE  : TRUE;
    args->bCanReorderFloat =          o_oglOpt ? TRUE  : TRUE;
    args->bCanIgnoreNan =             o_oglOpt ? TRUE  : TRUE;
    args->bCanIgnoreInf =             o_oglOpt ? TRUE  : TRUE;
    args->bCanIgnoreSignedZero =      o_oglOpt ? TRUE  : TRUE;
    args->bCanDemoteNonFP32Targets =  o_oglOpt ? FALSE : FALSE;
    args->bCanUseNrmhAlways =         o_oglOpt ? FALSE : FALSE;
    args->bUseDX10SAT = 0;
    args->bUseDX10AddressInRange = 0;    
#if defined(AR20)
    // FIX_ME: must define a way to define these for testing
    args->numDrawBuffers = 0;
    args->drawBufferFormat = NULL;
    args->drawBufferBlendState = NULL;
    args->colorWriteMask = NULL;
#endif // AR20

    OverrideCOPBaseArgs(args);
} // SetCOPBaseArgs

void OverrideCOPBaseArgs(COPBaseArgs *args)
{
    int ii;

    // Command-line overrides
    if (o_CanPromoteFixed != -1)
        args->bCanPromoteFixed = o_CanPromoteFixed;
    if (o_CanPromoteHalf != -1)
        args->bCanPromoteHalf = o_CanPromoteHalf;
    if (o_CanReorderFixedWithCheck != -1)
        args->bCanReorderFixedWithCheck = o_CanReorderFixedWithCheck;
    if (o_CanReorderHalf != -1)
        args->bCanReorderHalf = o_CanReorderHalf;
    if (o_CanReorderFloat != -1)
        args->bCanReorderFloat = o_CanReorderFloat;
    if (o_CanIgnoreNan != -1)
        args->bCanIgnoreNan = o_CanIgnoreNan;
    if (o_CanIgnoreInf != -1)
        args->bCanIgnoreInf = o_CanIgnoreInf;
    if (o_CanIgnoreSignedZero != -1)
        args->bCanIgnoreSignedZero = o_CanIgnoreSignedZero;
    if (o_CanDemoteNonFP32Targets != -1)
        args->bCanDemoteNonFP32Targets = o_CanDemoteNonFP32Targets;
    if (o_CanUseNrmhAlways != -1)
        args->bCanUseNrmhAlways = o_CanUseNrmhAlways;
    if (o_SupportSignedRemap != 0)
        args->dwSupportsSignedRemap = o_SupportSignedRemap;
    if (o_ConstantComputationExtraction > 0)
        args->optFlags |= LWI_OPT_CONSTANT_COMPUTATION_EXTRACTION;
    for(ii = 0; ii < 16; ii++) {
        if (o_TextureRemap[ii] != -1)
            args->wTextureRemap[ii] = o_TextureRemap[ii];
    }    
} // OverrideCOPBaseArgs

static unsigned int lGetArity(unsigned int dwOp)
{
    // The arguments for SFL and STR are irrelevant, but some of the other
    // optimizer code depends on op_arity returning 2.
    if (dwOp == opSTR ||
        dwOp == opSFL) {
        return 2;
    }

    assert(dwOp < (sizeof(fpInstOpInfo) / sizeof(fpInstOpInfo[0])));
    return fpInstOpInfo[dwOp].dwNumArgs;
}

/* lReswizzle(a, b) swizzles a by b, so
   sWZYX, sXXYY => sWWZZ
*/
static lwiSwizzle lReswizzle(lwiSwizzle a, lwiSwizzle b) {
    return (lwiSwizzle)((((a >> ((b << 1) & 0x6)) << 0) & 0x03) |
                       (((a >> ((b >> 1) & 0x6)) << 2) & 0x0c) |
                       (((a >> ((b >> 3) & 0x6)) << 4) & 0x30) |
                       (((a >> ((b >> 5) & 0x6)) << 6) & 0xc0));
}

// Returns a mask containing the components referenced by this swizzle
// sZXXZ => mX_Z_
static lwiMask lMaskFromSwizzle(lwiSwizzle eSwiz) {
    return (lwiMask)((1 << ((eSwiz & 0x03) >> 0)) |
                    (1 << ((eSwiz & 0x0c) >> 2)) |
                    (1 << ((eSwiz & 0x30) >> 4)) |
                    (1 << ((eSwiz & 0xc0) >> 6)));
}


// Returns a swizzle of the first set component in the mask (m____ returns sWWWW).
// sWZYX, m_YZ_ => sZZZZ
static lwiSwizzle lSwizzleCompFromMask(lwiSwizzle eSwiz, lwiMask eMask) {
    return ((eMask & mXY__) ? ((eMask & mX___) ? lReswizzle(eSwiz, sXXXX) : lReswizzle(eSwiz, sYYYY)) :
                              ((eMask & m__Z_) ? lReswizzle(eSwiz, sZZZZ) : lReswizzle(eSwiz, sWWWW)));
}

// Returns X or W in the swizzle based on the mask
// m_YZ_ => sXWWX
static lwiSwizzle lSwizzleMaskFromMask(lwiMask eMask) {
    return (lwiSwizzle)(((eMask & mX___) ? sWXXX : sXXXX) |
                       ((eMask & m_Y__) ? sXWXX : sXXXX) |
                       ((eMask & m__Z_) ? sXXWX : sXXXX) |
                       ((eMask & m___W) ? sXXXW : sXXXX));
}

// For each component in the mask that's set, select the appropriate component from A, else B
// m_YZ_, sXXYY, sZZWW => sZXYW
static lwiSwizzle lSwizzleInterpolate(lwiMask eMask, lwiSwizzle eA, lwiSwizzle eB) {
    lwiSwizzle eSwizMask = lSwizzleMaskFromMask(eMask);

    return (lwiSwizzle)((eA & eSwizMask) | (eB & ~eSwizMask));
}

// Replaces those components in the swizzle that are not in the mask with ones that are in the mask.
// If the mask is null, we get the original swizzle back.
// sYXZZ, mX__W => sYYYZ
static lwiSwizzle lSwizzleMaskReplace(lwiSwizzle eSwiz, lwiMask eMask) {
    return lSwizzleInterpolate(eMask, eSwiz, lSwizzleCompFromMask(eSwiz, eMask));
}

// Gets a mask from the components that are set within the swizzle,
// but masked
// sZWXY, mXY__ => m__ZW
// sXYYZ, mXY_W =? mXYZ_
static lwiMask lMaskFromSwizzleMask(lwiSwizzle eSwiz, lwiMask eMask) {
    return lMaskFromSwizzle(lSwizzleMaskReplace(eSwiz, eMask));
}

/*
 * lGetOperandMask() - return a mask of the referenced components
 */
static int lGetOperandMask(lwInst *fInst, int ix)
{
    lwiMask mask;

    mask = fpInstOpInfo[fInst->dwOp].aeReadMask[ix];
    if (mask == m____) // pass through
        mask = (lwiMask) (fInst->sDstReg.eMask);
    // swizzle the results appropriately
    mask = lMaskFromSwizzleMask(fInst->sSrcReg[ix].eSwiz, mask);
    return mask;
}

/*
 * fpFindOutputInfo() - Deduce ouput information from pList and set
 * appropriate fields in pArgs.  Should set eOutputMode, dwOutputMask,
 * and bUseDepth.
 *
 */

void fpFindOutputInfo(lwInst *pList, COPBaseArgs *pArgs)
{
    int ii, arity;
    lwInst *lInst;
    lwiReg *lReg;

    const int NUM_OUTPUT_REGS = 9;  // Require > 8 as code below indexes element 8.

    unsigned int lUnReadWrites[rcCount][NUM_OUTPUT_REGS];

    memset(lUnReadWrites, 0, sizeof(lUnReadWrites));
    
    // For each instruction, mark the sources as read and the destinations as unread.
    // Since we're looking for output info, only examine the registers that might be outputs.
    for (lInst = pList; lInst; lInst = lInst->next) {
        arity = lGetArity(lInst->dwOp);
        for (ii = 0; ii < arity; ii++) {
            lReg = &lInst->sSrcReg[ii].reg;
            if ((lReg->eClass == rcTempR || lReg->eClass == rcTempH) && lReg->iIndex < NUM_OUTPUT_REGS)
                lUnReadWrites[lReg->eClass][lReg->iIndex] &= ~(lGetOperandMask(lInst, ii));
        }
        lReg = &lInst->sDstReg.reg;
        if ((lReg->eClass == rcTempR || lReg->eClass == rcTempH) && lReg->iIndex < NUM_OUTPUT_REGS)
            lUnReadWrites[lReg->eClass][lReg->iIndex] |= lInst->sDstReg.eMask;
    }

    // Following is a guess and can be wrong at times..
    // However, seems to work in most cases...

    if (lUnReadWrites[rcTempR][0] || lUnReadWrites[rcTempR][2] ||
        lUnReadWrites[rcTempR][3] || lUnReadWrites[rcTempR][4]) {
        pArgs->eOutputMode = omR0;
        pArgs->dwOutputMask = lUnReadWrites[rcTempR][0];
        pArgs->dwOutputMask |= lUnReadWrites[rcTempR][2] << 4;
        pArgs->dwOutputMask |= lUnReadWrites[rcTempR][3] << 8;
        pArgs->dwOutputMask |= lUnReadWrites[rcTempR][4] << 12;
    } else if ((lUnReadWrites[rcTempH][0] || lUnReadWrites[rcTempH][2] ||
                lUnReadWrites[rcTempH][4] || lUnReadWrites[rcTempH][8]) &&
               !lUnReadWrites[rcTempH][1] && !lUnReadWrites[rcTempH][3]) {
        pArgs->eOutputMode = omH0;
        pArgs->dwOutputMask = lUnReadWrites[rcTempH][0];
        pArgs->dwOutputMask |= lUnReadWrites[rcTempH][4] << 4;
        pArgs->dwOutputMask |= lUnReadWrites[rcTempH][6] << 8;
        pArgs->dwOutputMask |= lUnReadWrites[rcTempH][8] << 12;
    } else {
        pArgs->eOutputMode = omCombiners;
        pArgs->dwOutputMask = lUnReadWrites[rcTempH][0];
        pArgs->dwOutputMask |= lUnReadWrites[rcTempH][1] << 4;
        pArgs->dwOutputMask |= lUnReadWrites[rcTempH][2] << 8;
        pArgs->dwOutputMask |= lUnReadWrites[rcTempH][3] << 12;
    }

    if (pArgs->eOutputMode != omCombiners &&
        (lUnReadWrites[rcTempR][1] & m__Z_) && !(lUnReadWrites[rcTempR][1] & mXY_W)) {
        pArgs->bUseDepth = 1;
    } else {
        pArgs->bUseDepth = 0;
    }
    pArgs->bUseCoverage = 0;        // until find a use
} // fpFindOutputInfo

/*
 * optimizeProgram() - Optimize the program in 'pArgs->inInsts'.
 */

int optimizeProgram(COPBaseArgs *pArgs, FILE *fOutFile, LWuCode **fpUCode)
{
    int lSuppressOutputPrint = o_Binary || (o_codeGenRuns > 0);

    lwCompileLWInstToUCode(NULL, fpUCode, pArgs, NULL, NULL, NULL, NULL);
    if (o_lwinst && *(pArgs->outInsts) && !lSuppressOutputPrint) {
        fpPrintLwInst(*(pArgs->outInsts), fOutFile,
                        pArgs->eOutputMode != omH0 && pArgs->eOutputMode != omCombiners,
                        pArgs->eProf);
    }

    return 0;
} // optimizeProgram


// --------------------------------------------------------------
