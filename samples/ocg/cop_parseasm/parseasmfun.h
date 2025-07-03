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
// parseasmfun.h
//

#ifndef __PARSEASMFUN_H_
#define __PARSEASMFUN_H_

#include "copi_sourceloc.h"
#include "parseasmdecls.h"

void fpPrintLwInst(const lwInst *pInsts, FILE *fOutFile, int fIsOutput32bit,
                    lwiOptimizationProfile eProf);
void fpPrintSinglePSInst(lwInst *pInsts, FILE *fOutFile, lwiOptimizationProfile eProf);
int optimizeProgram(COPBaseArgs *fArgs, FILE *fOutFile, LWuCode **fpUCode);
void SetCOPBaseArgs(COPBaseArgs *fArgs, lwInst **pOutInsts, lwInst **pOutStart,
                    ParamsForCOP *fParams);
void OverrideCOPBaseArgs(COPBaseArgs *fArgs);
void fpFindOutputInfo(lwInst *pList, COPBaseArgs *pArgs);

void fpEmitString(void *fArg, const char *fString);
void fpEmitError(void *fArg, SourceLoc *fLoc, int fNum, const char *fFmt, va_list args);
void fpEmitWarning(void *fArg, SourceLoc *fLoc, int fNum, const char *fFmt, va_list args);

void lCallExit(int returlwalue);


int ConstructDagFromLwInst(COPBaseArgs *fArgs, LWuCode **ppUCode, void *fOutFile, int psinst,
                           int verbosityLevel);

void *__fpMemAlloc(void *fArg, size_t size);

#endif // __PARSEASMFUN_H_
