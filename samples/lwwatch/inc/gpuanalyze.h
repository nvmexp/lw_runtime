/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2008-2014 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


//*****************************************************
//
// Slavik Bryksin <sbryksin@lwpu.com>
// lwwatch gpuanalyze Extension
// gpuanalyze.h
//
//*****************************************************


#ifndef _GPUANALYZE_H_
#define _GPUANALYZE_H_

#include "hal.h"

typedef struct{

    BOOL busyGr;
    BOOL busyMsvld;
    BOOL busyMspdec;
    BOOL busyMsppp;
    BOOL busyMsenc;
    BOOL busyCe0;
    BOOL busyCe1;
    BOOL busyCe2;

    char * output;
    char * unitErr;

}gpu_state;

extern gpu_state gpuState;

void addUnitErr(const char *format, ... );
LW_STATUS gpuAnalyze( LwU32 grIdx );


#endif // _GPUANALYZE_H
