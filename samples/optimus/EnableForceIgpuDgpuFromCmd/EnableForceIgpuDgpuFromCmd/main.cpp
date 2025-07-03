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


#include "stdafx.h"
#include "CoprocStatistics.h"

int _tmain(int argc, _TCHAR* argv[])
{
    print(true, true, NULL, L"enter 0 for Forced iGPU mode\n"
                            L"enter 1 for Forced dGPU mode\n"
                            L"enter 2 for Optimus mode\n");

    int     forceIGPUMode = 0;
    bool    result = false;

    // read cmd arguments
    if(argc<=1) 
    {
        printf("You did not feed me arguments, TERMINATING......");
        exit(1);
    }

    forceIGPUMode = _ttoi(argv[1]);

    vector<GPU>                         gpuList;
    LwAPI_Status                        status = LWAPI_ERROR;
    // update gpuList
    if(!fetchGpuList(gpuList))
    {
        print(true, true, NULL, L"No GPU has been found\n");
        return 0;
    }
    
    LwU32 coprocInfoFlags;
    switch(forceIGPUMode)
    {
        case 0:
            coprocInfoFlags = LW_COPROC_FLAGS_IGPU_MODE_ONLY;
            print(true, true, NULL, L"Setiing FORCED IGPU mode\n");
            break;
        case 1:
            coprocInfoFlags = LW_COPROC_FLAGS_DGPU_MODE_ONLY;
            print(true, true, NULL, L"Setiing FORCED DGPU mode\n");
            break;
        case 2:
            coprocInfoFlags = LW_COPROC_FLAGS_FORCE_OPTIMUS;
            print(true, true, NULL, L"Setiing Optimus mode\n");
            break;
        default:
            print(true, true, NULL, L"invalid value.\n");
            exit(1);
    }

    // clear cycle and stats for all GPU's
    for (std::vector<GPU>::iterator it = gpuList.begin() ; it != gpuList.end(); ++it)
    {
        GPU gpu = *it;
        status = LwAPI_Coproc_SetCoprocInfoFlagsEx2(gpu.hPhyGPU, coprocInfoFlags);
        if(status == LWAPI_OK)
        {
            result = true;
        }
    }
    if(result)
    {
        print(true, true, NULL, L"able to set coproc flag successfully\n");
    }
    else
    {
        print(true, true, NULL, L"Somthing wrong\n");
    }
    return 0;
}