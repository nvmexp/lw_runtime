//*****************************************************
//
// lwwatch DFD Assembly Extension
// xiaohaow@lwpu.com
// dfdasm.h
//
//*****************************************************

#ifndef _DFDASM_H_
#define _DFDASM_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "hal.h"
    
void runDfdAsm(const char *asmFname, const char *logFname, const char *command, int verbose, int verboseLog, int test);

#ifdef __cplusplus
}
#endif


#endif