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
 *  Module name              : compilerToolsMessageDefs.h
 *
 *  Description              :
 *
 *         This file defines the messages generated
 *         by the compilerTools utilities
 */

#ifndef compilerToolsMessageDefs_INCLUDED
#define compilerToolsMessageDefs_INCLUDED

#include <misc/stdMessageDefsBegin.h>

MSG( ctExitMessage,           Fatal,   "Internal error: exit status %d"  )
MSG( ctAbortMessage,          Fatal,   "Internal error: Aborting"        )
MSG( ctBadArchName,           Fatal,   "Unknown arch name '%s'"          )
MSG( ctMsgBadFileName,        Fatal,   "Bad file name '%s'"              )
MSG( ctMsgCannotOpenFile,     Fatal,   "Cannot open file '%s'"           )

#include <misc/stdMessageDefsEnd.h>

#endif

