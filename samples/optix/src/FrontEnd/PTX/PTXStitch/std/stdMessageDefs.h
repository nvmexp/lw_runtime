/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
/*
 *  Module name              : stdMessageDefs.h
 *
 *  Description              :
 *
 *         This file defines the messages generated
 *         by the std library.
 */

#ifndef stdMessageDefs_INCLUDED
#define stdMessageDefs_INCLUDED

#include <misc/stdMessageDefsBegin.h>

MSG( stdMsgBailoutDueToErrors, Fatal,   "Bailing out due to earlier errors"  )
MSG( stdMsgMemoryOverFlow,     Fatal,   "Memory allocation failure"          ) 
MSG( stdMsgForkFailed,         Error,   "fork() failed with '%s'"            ) 
MSG( stdMsgWaitFailed,         Error,   "wait() failed with '%s'"            ) 
MSG( stdMsgExecFailed,         Fatal,   "exec() failed with '%s'"            ) 
MSG( stdMsgOpenInputFailed,    Fatal,   "Could not open input file '%s'"     ) 
MSG( stdMsgOpenOutputFailed,   Fatal,   "Could not open output file '%s'"    ) 
MSG( stdMsgCoreDumped,         Error,   "'%s' core dumped"                   ) 

MSG( stdMsgStrayBackSlash,     Fatal,   "Stray '\' character"  )
MSG( stdMsgStrayBracket,       Fatal,   "Stray '[' character"  )
MSG( stdMsgStrayQuote,         Fatal,   "Stray '\"' character" )

#ifdef STD_OS_win32
MSG( stdMsgSignal,             Error,   "'%s' died with status 0x%08X %s"    ) 
#else
MSG( stdMsgSignal,             Error,   "'%s' died due to signal %d %s"      ) 
#endif

#include <misc/stdMessageDefsEnd.h>

#endif

