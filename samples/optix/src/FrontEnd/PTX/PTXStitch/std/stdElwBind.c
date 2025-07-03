/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2006-2021, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdElwBind.c
 *
 *  Description              :
 *     
 */

/*-------------------------------- Includes ---------------------------------*/

#include "stdLocal.h"
#include "stdProcess.h"
#include "stdString.h"

/*--------------------------- Environment Bindings --------------------------*/

static FILE *logFile;
#define LOGFILE  (logFile?logFile:stderr)


void STD_CDECL stdSetLogFile( FILE *f )
{ logFile = f; }


void STD_CDECL stdVFSYSLOG( cString format, va_list arg )
{ vfprintf(LOGFILE,format,arg); }


void STD_CDECL stdFSYSLOG( cString format, ... )
{
    va_list arg;
     
    va_start(arg,format);
    stdVFSYSLOG(format,arg);
    va_end(arg);
}


void STD_CDECL stdSYSLOG( cString format, ... )
{
    va_list arg;
     
    va_start(arg,format);
    stdVSYSLOG(format,arg);
    va_end(arg);
}


static void stdSetMsgBuff(stdString_t buff)
{
    stdGetThreadContext()->mInfo.msgBuff = (void *) buff;
}

static stdString_t stdGetMsgBuff()
{
    return (stdString_t) stdGetThreadContext()->mInfo.msgBuff;
}

static void stdClearMsgBuff()
{
    if (stdGetMsgBuff()) {
        stringDelete(stdGetMsgBuff());
        stdSetMsgBuff(NULL);
    }
}

void STD_CDECL stdVSYSLOG( cString format, va_list arg )
{
    if (!stdGetLogLine()) {
        vfprintf(LOGFILE,format,arg);
    } else {
        stdMemSpace_t savedSpace = stdSwapMemSpace( memspNativeMemSpace );
        {
            Char  *buff = stdMALLOC(100000);
            Char  *str  = buff;
            String line;

            vsprintf(buff,format,arg);

            while (*str) {
                Char *nl= strchr(str,'\n');

                if (!stdGetMsgBuff()) {
                    stdSetMsgBuff(stringNEW());
                }

                if (nl) {
                    *nl=0;
                    stringAddBuf(stdGetMsgBuff(),str);
                    line= stringStripToBuf(stdGetMsgBuff());
                    if (stdGetLogLine()) {
                        (stdGetLogLine())(line);
                    }
                    stdFREE(line);
                    stdSetMsgBuff(NULL);
                    str=nl+1;
                } else {
                    stringAddBuf(stdGetMsgBuff() ,str);
                    str="";
                }
            }

            stdFREE(buff);
        }
        stdSwapMemSpace(savedSpace);
    }
}

stdLogLineFunc STD_CDECL stdGetLogLine()
{
    return stdGetThreadContext()->mInfo.logLine;
}

stdLogLineFunc STD_CDECL stdSetLogLine( stdLogLineFunc f )
{
    stdClearMsgBuff();
    stdLogLineFunc result = stdGetThreadContext()->mInfo.logLine;
    stdGetThreadContext()->mInfo.logLine = f;
    stdGetThreadContext()->mInfo.oldLogLineFunc = result;
    return result;
}

void STD_CDECL stdRestoreLogLine(void)
{
    stdClearMsgBuff();
    stdGetThreadContext()->mInfo.logLine =
        stdGetThreadContext()->mInfo.oldLogLineFunc;
}

void* STD_CDECL stdGetLogLineData()
{
    return stdGetThreadContext()->mInfo.logLineData;
}

void* STD_CDECL stdSetLogLineData(void* data)
{
    void* result = stdGetThreadContext()->mInfo.logLineData;
    stdGetThreadContext()->mInfo.logLineData = data;
    stdGetThreadContext()->mInfo.oldLogLineData = result;
    return result;
}

void STD_CDECL stdRestoreLogLineData(void)
{
    stdGetThreadContext()->mInfo.logLineData =
        stdGetThreadContext()->mInfo.oldLogLineData;
}


#if defined STD_OS_win32 || !defined ABORT_BACKTRACE
    #define BACKTRACE()
#else
    #include "libunwind.h"
    
    static void BACKTRACE()
    {
        unw_context_t         context;
        unw_lwrsor_t          cursor;
        
        unw_getcontext(&context);
        unw_init_local(&cursor, &context);
        unw_step(&cursor);
    
        stdSYSLOG("BACKTRACE:\n");
        stdSYSLOG("=========\n");

        while ( unw_step(&cursor) > 0 ) {
            unw_word_t      offset;
            Char            buffer[1000];
        
            Int status = unw_get_proc_name(&cursor, buffer, sizeof(buffer), &offset);
                            
            if (status == UNW_ENOINFO || status == UNW_EUNSPEC) {
                unw_proc_info_t procInfo;
                unw_get_proc_info(&cursor, &procInfo);
                stdSYSLOG("        ??@0x%" stdFMT_LLX "\n", (uInt64)procInfo.start_ip);
            } else {
                stdSYSLOG("        %s\n", buffer);
            }
        }
    }
#endif



static void STD_CDECL defaultAbort(void) 
{
    BACKTRACE();
    if (procTrapOnError) { 
        abort();  
    } else { 
        stdEXIT_ERROR(); 
    }
}
//qnx/qnx660/common/target/qnx6/usr/include/process.h:76:18: note: previous declaration of '__exit' was here
// Hence changing the name
static stdABORTFunc ___abort = (stdABORTFunc)defaultAbort;
static stdEXITFunc  ___exit  = (stdEXITFunc )exit;
    
void STD_CDECL stdGetTerminators( stdABORTFunc *abort, stdEXITFunc *exit )
{
    if (abort) { *abort = ___abort; }
    if (exit)  { *exit  = ___exit ; }
}

void STD_CDECL stdSetTerminators( stdABORTFunc abort, stdEXITFunc exit )
{
    if (abort) { ___abort= abort; }
    if (exit ) { ___exit = exit ; }
}

void STD_CDECL stdABORT( ) 
{
    ___abort();
      abort();
}

void STD_CDECL stdEXIT( Int status ) 
{
     /*
      * According to the Posix spec, only the 8 lower 
      * bits are returned to the parent process.
      *
      * Explicitly truncate 'status' accordingly in order
      * to avoid the bash problem as described in:
      *
      * http://lists-archives.com/mingw-users/16090-the-exit-value-of-statement-return-1.html
      * http://sourceforge.net/p/mingw/bugs/747/
      */
    ___exit(status & 0xff);
      exit(status & 0xff);
}

    
    
