/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2013-2016, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : ctMessages.c
 *
 *  Description              :
 *     
 */

#include <ctMessages.h>
#include <stdMessages.h>

FILE        *ctMessagesRedirectTo;  // Used for redirection of system() calls like in proc(V)(String)System

static void ctMessageRedirectTo ()
{
#ifdef STD_OS_win32
    ctMessagesRedirectTo = stdout;
#else
    ctMessagesRedirectTo = stderr;
#endif
}

/*
 * Function        : Redirects info messages to stdout or stderr based on platform
 * Parameters      : clientWriter  (IO) Writer to be installed and the current writer would be returned
 * Function Result : 
 * Note            : On windows, info messages are redirected to stdOut else on stdErr
 */
static void ctMessageInfoMsgRedirectInit(stdWriter_t* clientWriter)
{
    // Redirect all verbose info messages to stdout on Windows only
#ifdef STD_OS_win32
    *clientWriter  = wtrCreateFileWriter(stdout);
    // For SYSLOG* functions, override using stdSetLogFile
    stdSetLogFile(stdout);
#else
    *clientWriter  = wtrCreateFileWriter(stderr);
    stdSetLogFile(stderr);
#endif
    msgSwapMessageChannels((Pointer *)clientWriter, NULL, NULL);
}

/*
 * Function        : Resets the redirection of info messages
 * Parameters      : olderWriter  (IO)  Writer will be installed and the current writer would be freed up
 * Function Result : 
 */
static void ctMessageInfoMsgRedirectCleanup(stdWriter_t* olderWriter)
{
    // Re-install the previously present writers and free up the current writer
    msgSwapMessageChannels((Pointer *)olderWriter, NULL, NULL);
    wtrDelete(*olderWriter);
}


/*
 * Function        : Top level init function which calls other ctMessage* initializing routines
 * Parameters      : clientWriter  (IO) Writer to be installed and the current writer would be returned
 * Function Result : 
 */
void ctMessageInit (stdWriter_t* clientWriter)
{
// In online path, driver would have set stdSetLogLine to redirect messages to LW_JIT_LOG_BUFFERS
// So do not set logFile and channelWriters
#ifndef GPGPUCOMP_DRV_BUILD
    msgSetDontPrefixErrorLines(True);
    ctMessageInfoMsgRedirectInit(clientWriter);
#endif
    ctMessageRedirectTo();
}


/*
 * Function        : Top level cleanup function which calls other ctMessage* cleanup routines
 * Parameters      : olderWriter  (IO) Writer will be installed and the current writer would be freed up
 * Function Result : 
 */
void ctMessageExit(stdWriter_t* olderWriter)
{
#ifndef GPGPUCOMP_DRV_BUILD
    ctMessageInfoMsgRedirectCleanup(olderWriter);
#endif
}

/*
 * Function        : Create new msgSourcePos_t object
 * Parameters      : fileName     (I) source filename
 *                   lineNo       (I) line number
 *                   sourceStruct (I) Managing msgSourceStruct_t object
 * Function Result : Requested new msgSourcePos_t, other fields set to NULL
 * Note            : Only in JAS path, sourceStruct acts as a manager of  msgSourcePos_t.
 *                   So no explicit freeing of msgSourcePos_t objects are needed in JAS path
 */
msgSourcePos_t ctMsgCreateSourcePos( cString fileName, msgSourceStructure_t* sourceStruct, uInt lineNo )
{
    return msgPullSourcePos(*sourceStruct, lineNo);
}
