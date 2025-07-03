/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2008 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// vgupta@lwpu.com - July 2004
// 
// Include file for lwwatch.cpp. Named so because
// lwwatch.h is already taken.  Can't use lwwatch.h to
// contain the contents of this file. Reason: lwwatch.h
// is included by many .c files.  This file has C++
// constructs in it. If the contents of this file are
// moved to lwwatch.h, then the .c files which #include
// lwwatch.h would have to be renamed to .cpp which we
// don't want to do.
// 
//*****************************************************


#ifndef _LWWATCH2_H_ 
#define _LWWATCH2_H_

//
// For g_ExtOutputCallbacks variable decl below
//
#include "outputCb.h"

//
// DEBUG_OUTPUT_NONE and DEBUG_OUTPUT_NORMAL are used to set output masks.  Used
// by RegRd to express interest in output messages spit out by the debugger
// engine.
//
// DEBUG_OUTPUT_NONE is used to stop getting ANY output messages from the
// debugger engine.
// Defining here since it's not defined in dbgeng.h
//
#define DEBUG_OUTPUT_NONE            0x00000000

//
// Global variables initialized by query.
//
extern PDEBUG_CLIENT         g_ExtClient;
extern PDEBUG_CONTROL        g_ExtControl;
extern PDEBUG_SYMBOLS2       g_ExtSymbols;
extern PDEBUG_DATA_SPACES4   g_ExtMemory;
extern PDEBUG_REGISTERS2     g_ExtRegisters;

extern PLWWATCH_DEBUG_OUTPUT_CALLBACK  g_ExtOutputCallbacks;

HRESULT ExtQuery(void);
void ExtRelease(void);

#endif // _LWWATCH2_H_
