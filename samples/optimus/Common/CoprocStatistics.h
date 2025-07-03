/***************************************************************************\
|*                                                                           *|
|*       Copyright 1993-2013 LWPU, Corporation.  All rights reserved.      *|
|*                                                                           *|
|*     NOTICE TO USER:   The source code  is copyrighted under  U.S. and     *|
|*     international laws.  Users and possessors of this source code are     *|
|*     hereby granted a nonexclusive,  royalty-free copyright license to     *|
|*     use this code in individual and commercial software.                  *|
|*                                                                           *|
|*     Any use of this source code must include,  in the user dolwmenta-     *|
|*     tion and  internal comments to the code,  notices to the end user     *|
|*     as follows:                                                           *|
|*                                                                           *|
|*       Copyright 1993-2013 LWPU, Corporation.  All rights reserved.      *|
|*                                                                           *|
|*     LWPU, CORPORATION MAKES NO REPRESENTATION ABOUT THE SUITABILITY     *|
|*     OF  THIS SOURCE  CODE  FOR ANY PURPOSE.  IT IS  PROVIDED  "AS IS"     *|
|*     WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.  LWPU, CORPOR-     *|
|*     ATION DISCLAIMS ALL WARRANTIES  WITH REGARD  TO THIS SOURCE CODE,     *|
|*     INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGE-     *|
|*     MENT,  AND FITNESS  FOR A PARTICULAR PURPOSE.   IN NO EVENT SHALL     *|
|*     LWPU, CORPORATION  BE LIABLE FOR ANY SPECIAL,  INDIRECT,  INCI-     *|
|*     DENTAL, OR CONSEQUENTIAL DAMAGES,  OR ANY DAMAGES  WHATSOEVER RE-     *|
|*     SULTING FROM LOSS OF USE,  DATA OR PROFITS,  WHETHER IN AN ACTION     *|
|*     OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF     *|
|*     OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.     *|
|*                                                                           *|
|*     U.S. Government  End  Users.   This source code  is a "commercial     *|
|*     item,"  as that  term is  defined at  48 C.F.R. 2.101 (OCT 1995),     *|
|*     consisting  of "commercial  computer  software"  and  "commercial     *|
|*     computer  software  documentation,"  as such  terms  are  used in     *|
|*     48 C.F.R. 12.212 (SEPT 1995)  and is provided to the U.S. Govern-     *|
|*     ment only as  a commercial end item.   Consistent with  48 C.F.R.     *|
|*     12.212 and  48 C.F.R. 227.7202-1 through  227.7202-4 (JUNE 1995),     *|
|*     all U.S. Government End Users  acquire the source code  with only     *|
|*     those rights set forth herein.                                        *|
|*                                                                           *|
\***************************************************************************/

#ifndef _COPROCSTATS_H
#define _COPROCSTATS_H
#include <vector>
#include <lwapi.h>

using namespace std;

typedef struct _tagGPU
{
    LwPhysicalGpuHandle hPhyGPU;
	char		        gpuName[LWAPI_SHORT_STRING_MAX];
} GPU, *PGPU;

bool DisableGC6();

bool ResetTest();
void print(bool printToDebug, bool printToFile, FILE *outFile, wchar_t *format, ...);

void dumpLastError();

// TODO: There should really be a better way of finding this out
bool isWinBlue();
void DebugDumpHistogram( LwS32 *histogram, LwU32 timeIncrement, FILE *outFile);
bool populate(PFND3DKMT_ESCAPE* ppEscape, D3DKMT_HANDLE* phAdapter);

// helper functions
wchar_t *coprocStateToString( LWL_COPROC_POWER_STATE state );
D3DKMT_HANDLE OpenFirstLwidiaAdapter( PFND3DKMT_OPENADAPTERFROMDEVICENAME pOpenAdapterFromDeviceName );

void formatTime( wchar_t *timeString, LwU64 time, size_t numChars );

static void DisplayEscape_Pause()
{
    print(true, true, NULL, L"Press any key to continue\n");
    _getch();
}

void getAndShowCoprocInfo(LwPhysicalGpuHandle hPhyGPU, bool& gc6SupportStatus, LwAPI_Status coprocInfoStatus, FILE *outFile);
bool fetchGpuList(vector<GPU> &gpuList);
bool showCoprocCycleInformation(LwPhysicalGpuHandle hPhyGPU, bool bResetCycles, LwU64 cycleCount, bool printStats, FILE *outFile);
void getAndShowGC6Statistics(LwPhysicalGpuHandle hPhyGPU, bool bAbbreviated, FILE *outFile);
void getAndShowGOLDStatistics(LwPhysicalGpuHandle hPhyGPU, bool bAbbreviated, bool bResetStats, FILE *outFile);
bool clearCoprocStats(LwPhysicalGpuHandle hPhyGPU);
#endif
