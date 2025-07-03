/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// vgupta@lwpu.com - July 2004
// regMods.h has code to support reading/writing chip
// registers thru Mods
// 
//*****************************************************

#ifndef _REGMODS_H_
#define _REGMODS_H_

#include "os.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ILWALID_REG32 0xffffffff
#define ILWALID_REG16 0xffff
#define ILWALID_REG08 0xff

//
// "Enum hack"
//
enum 
{
    INVALID=0, 
    VALID=1
};

//
// These strings are appended to .call commands made from windbg extension to
// MODs. Grep for FORCE_REG_OP_STRING in regMods.cpp to see what's going on.
//
const char * const FORCE_REG_OP_STRING = "g;? 1";

//
// FORCE_REG_OP_STRING2 is basically what the debugger engine spits
// out when FORCE_REG_OP_STRING is fed to it.
// WINDBG VERSION CAUTION: If you using a really new windbg, this expression may
// be different and RegRd/wr would fail.
//
const char * const FORCE_REG_OP_STRING2 = "Evaluate expression: 1 =";

//
// READ_SIZE used in ReadStruct below
//
// READ00 means 'dont read anything'
// READ32 means read a 32 bit number'
// READ16 means read a 16 bit number'
// READ08 means read a 08 bit number'
//
typedef enum {READ32, READ16, READ08, READ00} READ_SIZE;

//
// Global struct definition used for RegRd/RegWr and Output communication
// Read docs/lwwatch-windbg-workings.txt for more
//
typedef struct 
{
    READ_SIZE readSizeCmd;
    U032 readU032;
    U016 readU016;
    U008 readU008;
    int readIsValid;
} ReadStruct; 

//
// Methods associated with ReadStruct. We need C++
// classes here. Too much global functions and
// variables
//
#define SET_READ_ILWALID()     do {readStruct.readIsValid = INVALID;}  while(0)
#define IS_READ_VALID()           (readStruct.readIsValid?VALID:INVALID)

//
// Global used for RegRd/RegWr and Output communication
//
extern ReadStruct readStruct;


UINT32 RegRd (char *command, READ_SIZE _readSizeCmd, UINT32 Address);
BOOL RegWr   (char *command, UINT32 Address, UINT32 value);

#ifdef __cplusplus
}
#endif

#endif // _REGMODS_H_
