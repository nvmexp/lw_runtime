/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2006-2020, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdProcess.h
 *
 *  Description              :
 *
 *        This module provides for .
 */

#ifndef stdProcess_INCLUDED
#define stdProcess_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include "stdLocal.h"
#include "stdString.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------- Functions --------------------------------*/


/*
 * Function        : Perform all OS specific process setup
 * Parameters      : argv0        (I) Name by which the current 
 *                                    exelwtable was ilwoked.
 *                                    This is either an absolute file name,
 *                                    or a relative name found via $PATH
 */
void STD_CDECL procProcessSetup( cString argv0 );


/*----------------------------- Process Creation -----------------------------*/

/*
 * Function        : Run an exelwtable in a subprocess, and wait for completion.
 * Parameters      : argv         (I) path of exelwtable, plus command line options.
 *                                    The exelwtable is located in the PATH environment 
 *                                    variable.
 *                   inputFile,
 *                   outputFile,
 *                   errorFile    (I) Names for stdin, stdout or stderr redirection files,
 *                                    or Nil to inherit parent's standard i/o files.
 *                   appendOutput (I) True iff. stdout of the subprocess has to be
 *                                    appended to outputFile
 *                   reportCrash  (I) True iff. a message is required when the exelwted
 *                                    subprocess crashes.
 * Function Result : Exit status of ilwoked command.
 */
Int STD_CDECL procRunCommand( String argv[], cString inputFile, cString outputFile, cString errorFile, Bool appendOutput, Bool reportCrash );

#ifdef STD_OS_win32 


/*
 * Function        : Run an exelwtable in a subprocess, and wait for completion.
 * Parameters      : argv         (I) path of exelwtable, plus command line options.
 *                                    The exelwtable is located in the PATH environment 
 *                                    variable.
 *                   inputFile,
 *                   outputFile,
 *                   errorFile    (I) Names for stdin, stdout or stderr redirection files,
 *                                    or Nil to inherit parent's standard i/o files.
 *                   appendOutput (I) True iff. stdout of the subprocess has to be
 *                                    appended to outputFile
 *                   reportCrash  (I) True iff. a message is required when the exelwted
 *                                    subprocess crashes.
 * Function Result : Exit status of ilwoked command.
 */
Int STD_CDECL procRunCommand( String argv[], cString inputFile, cString outputFile, cString errorFile, Bool appendOutput, Bool reportCrash );
/*
 * Function        : Debug an exelwtable in a subprocess, lauch asynchrously so
 *                   the main process can attach.
 * Parameters      : argv         (I) path of exelwtable, plus command line options.
 *                                    The exelwtable is located in the PATH environment 
 *                                    variable.
 *                   inputFile,
 *                   outputFile,
 *                   errorFile    (I) Names for stdin, stdout or stderr redirection files,
 *                                    or Nil to inherit parent's standard i/o files.
 *                   appendOutput (I) True iff. stdout of the subprocess has to be
 *                                    appended to outputFile
 *                   reportCrash  (I) True iff. a message is required when the exelwted
 *                                    subprocess crashes.
 *                   pi           (O) structure to hold the debgugge process
 *                                    information.
 */
void STD_CDECL procDebugCommand( String argv[], cString inputFile, cString outputFile, 
cString errorFile, Bool appendOutput, Bool reportCrash, PROCESS_INFORMATION *pi );

/*
 * Function        : Execute an exelwtable in a subprocess but first inject a
 * dll in the load sequence.
 * Parameters      : argv         (I) path of exelwtable, plus command line options.
 *                                    The exelwtable is located in the PATH environment
 *                                    variable.
 *                   inputFile,
 *                   outputFile,
 *                   errorFile    (I) Names for stdin, stdout or stderr redirection files,
 *                                    or Nil to inherit parent's standard i/o files.
 *                   appendOutput (I) True iff. stdout of the subprocess has to be
 *                                    appended to outputFile
 *                   reportCrash  (I) True iff. a message is required when the exelwted
 *                                    subprocess crashes.
 *                   DllPaths     (I) Full paths to the DLLs that are to be
 *                   injected.
 *                   numDlls      (I) Number of DLLs injected
 */
Int STD_CDECL procRunCommandWithDll( String argv[], cString inputFile, cString outputFile, 
cString errorFile, Bool appendOutput, Bool reportCrash, cString DllPaths[], uInt numDlls );
#endif

/*-------------------------------- 'System' ---------------------------------*/


Int STD_CDECL procSystem      ( FILE* traceFile, Bool verbose, Bool dryrun, Bool force, Bool exitOnError, String x );
Int STD_CDECL procVSystem     ( FILE* traceFile, Bool verbose, Bool dryrun, Bool force, Bool exitOnError, cString format, ... );
Int STD_CDECL procStringSystem( FILE* traceFile, Bool verbose, Bool dryrun, Bool force, Bool exitOnError, stdString_t command );

FILE* STD_CDECL procFOPEN( cString name, cString mode );
void STD_CDECL procFCLOSE( FILE* f );

extern Bool procTrapOnError;
void STD_CDECL procSetTrapOnError(void);


/*------------------------------ Platform Info ------------------------------*/

/*
 * Function        : Return True iff the current platform is 64 bit.
 * Function Result : Sic.
 */
Bool STD_CDECL procIs64Bit(void);

/*
 * Function        : Check the machine bitness
 * Function Result : If the machine is 64-bit
 */
Bool STD_CDECL procIsMachine64Bit(void);

/*------------------------------- File Names --------------------------------*/

/*
 * Function        : Return absolute pathname of the current working directory.
 * Function Result : Absolute pathname of the current working directory.
 */
String STD_CDECL procGetPwd(void);


/*
 * Function        : Decide if specified file name has been returned by
 *                   function procGetTempName or procGetTempDirName.
 * Parameters      : name  (I) File name to inspect.
 * Function Result : True iff name is the name of a temp file.
 */
Bool STD_CDECL procIsTempName( cString name );


/*
 * Function        : Return a fresh temporary file name.
 * Function Result : Name of file.
 */
String STD_CDECL procGetTempName(void);


/*
 * Function        : Return the base name of temporary file names
 * Function Result : Name of file.
 */
String STD_CDECL procGetTempNameBase(void);

/*
 * Function        : Return a fresh temporary directory name.
 * Function Result : Name of directory.
 */
String STD_CDECL procGetTempDirName(void);


/*
 * Function        : Return absolute pathname of the current exelwtable.
 * Function Result : Name of exelwtable running current process.
 */
String STD_CDECL procLwrrentExelwtableName(void);


/*
 * Function        : Return the hostname
 */
String STD_CDECL procGetHostName(void);


/*---------------------------- Native Threads ----------------------------*/

#if !defined(__ANDROID__)
    #if !defined(STD_OS_win32) && !defined(STD_OS_CygWin)

        typedef pthread_t         procThreadId_t;

        #include <pthread.h>
        #ifdef STD_OS_Darwin
          #include <signal.h> // pthread_kill
        #endif

        static inline Bool procThreadKill( procThreadId_t tid, Int signal )
        {
            return pthread_kill(tid, signal) == 0;
        }

        static inline procThreadId_t procGetThreadId( void )
        {
            return pthread_self();
        }
    #else
        typedef DWORD             procThreadId_t; 

        //TODO: Implement for Windows. 
        static inline Bool procThreadKill( procThreadId_t tid, Int signal )
        {
            return False;
        }
        static inline procThreadId_t procGetThreadId( void )
        {
            return 0;
        }
    #endif



    /*
     * Function        : Hash/Equal pair for ThreadIds
     */
    Bool STD_CDECL procpThreadIdHash ( procThreadId_t *l );
    Bool STD_CDECL procpThreadIdEqual( procThreadId_t *l, procThreadId_t *r );

#endif

/*
 * Function        : Set environment variable
 * Parameters      : elw (I) name of environment variable
 *                   val (I) value to set
 */
void STD_CDECL stdSetElw(const Char *elw, const Char *val);


/*
 * Function        : Get environment variable
 * Parameters      : elw  (I) name of environment variable
 * Function Result : New String holding environment variable or Nil
 */
String STD_CDECL stdGetElw(const Char *elw);

#ifdef __cplusplus
}
#endif

#endif
