/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2015-2017, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdProcess_detours.c
 *
 *  Description              :
 *
 */

/*--------------------------------- Includes ---------------------------------*/

#include "stdLocal.h"
#include "stdString.h"
#include "stdSet.h"
#include "stdFileNames.h"
#include "stdProcess.h"
#include "stdRangeMap.h"

#if (defined STD_OS_win32) && (defined BUILD_WITH_DETOURS)
    #include "detours.h"
#endif

/*----------------------------- Process Creation -----------------------------*/

#if (defined STD_OS_win32) && (defined BUILD_WITH_DETOURS)

    static Int reportIfCrashed(String command, Int status)
    {
        #define WIFSIGNALED(status)  (((status)&0xF0000000) == 0xC0000000)
        #define WTERMSIG(status)       (status)
        if (WIFSIGNALED(status)) {
            Int termsig = WTERMSIG(status);
            String sigtext;

            switch (status) {
                #define STATUSCASE(s) case STATUS_##s : sigtext="("#s")" ; break;
                 STATUSCASE(ACCESS_VIOLATION)
                 STATUSCASE(IN_PAGE_ERROR)
                 STATUSCASE(ILWALID_HANDLE)
                 STATUSCASE(NO_MEMORY)
                 STATUSCASE(ILLEGAL_INSTRUCTION)
                 STATUSCASE(NONCONTINUABLE_EXCEPTION)
                 STATUSCASE(ILWALID_DISPOSITION)
                 STATUSCASE(ARRAY_BOUNDS_EXCEEDED)
                 STATUSCASE(FLOAT_DENORMAL_OPERAND)
                 STATUSCASE(FLOAT_DIVIDE_BY_ZERO)
                 STATUSCASE(FLOAT_INEXACT_RESULT)
                 STATUSCASE(FLOAT_ILWALID_OPERATION)
                 STATUSCASE(FLOAT_OVERFLOW)
                 STATUSCASE(FLOAT_STACK_CHECK)
                 STATUSCASE(FLOAT_UNDERFLOW)
                 STATUSCASE(INTEGER_DIVIDE_BY_ZERO)
                 STATUSCASE(INTEGER_OVERFLOW)
                 STATUSCASE(PRIVILEGED_INSTRUCTION)
                 STATUSCASE(STACK_OVERFLOW)
                 STATUSCASE(CONTROL_C_EXIT)
                 default: sigtext=""; break;
             }

             stdCHECK( False, (stdMsgSignal,command,status,sigtext) );
        }
        return status;
    }

    Int STD_CDECL procRunCommandWithDll( String argv[], String inputFile, String
    outputFile, String errorFile, Bool appendOutput, Bool reportCrash, String
    DllPaths[], uInt numDlls )
    {
        Int status;
        String cmdLine;
        Bool   outIsError= False;
        SELWRITY_ATTRIBUTES sa;
        HANDLE old_stdin, old_stdout, old_stderr;
        HANDLE new_stdin, new_stdout, new_stderr;

        sa.nLength = sizeof(sa);
        sa.lpSelwrityDescriptor = Nil;
        sa.bInheritHandle = TRUE;

        if (inputFile) {
            new_stdin = CreateFile(inputFile, GENERIC_READ, FILE_SHARE_READ, &sa, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, Nil);
            old_stdin = GetStdHandle (STD_INPUT_HANDLE);
            SetStdHandle (STD_INPUT_HANDLE, new_stdin);
        }

        if (outputFile) {
            uInt disp= appendOutput ? OPEN_ALWAYS : CREATE_ALWAYS;
            new_stdout = CreateFile (outputFile, GENERIC_WRITE, 0, &sa, disp, FILE_ATTRIBUTE_NORMAL, Nil);
            old_stdout = GetStdHandle (STD_OUTPUT_HANDLE);
            SetStdHandle (STD_OUTPUT_HANDLE, new_stdout);
            if (appendOutput) { SetFilePointer(new_stdout, 0, Nil, FILE_END); }
        }

        if (errorFile) {
            outIsError= outputFile && stdEQSTRING(outputFile,errorFile);
            old_stderr = GetStdHandle (STD_ERROR_HANDLE);
            if (outIsError) {
                SetStdHandle (STD_ERROR_HANDLE, new_stdout);
            } else {
                new_stderr = CreateFile (outputFile, GENERIC_WRITE, 0, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, Nil);
                SetStdHandle (STD_ERROR_HANDLE, new_stderr);
            }
        }

        {
           stdString_t buffer = stringNEW();
           String      *arg   = argv;

           while (*arg) {
               Char *pc= *(arg++);
               stringAddChar(buffer,'"');
               while (*pc) {
                   Char c= *(pc++);
                   if (c=='"' || c=='\\') { stringAddChar(buffer,'\\'); }
                   stringAddChar(buffer,c);
               }
               stringAddBuf (buffer,"\" ");
           }

           cmdLine= stringStripToBuf(buffer);
        }

        {
            STARTUPINFO si;
            PROCESS_INFORMATION pi;

            ZeroMemory( &si, sizeof(si) );
            ZeroMemory( &pi, sizeof(pi) );
            si.cb     = sizeof(si);

            // Start the child process. 
            if( !DetourCreateProcessWithDll( Nil,  // no module just use command line
                cmdLine,               // Command line  i.e. "c:\Program Files\my dir\cl.exe" -c "my file.c"
                Nil,                  // Process handle not inheritable
                Nil,                  // Thread handle not inheritable
                TRUE,                  // Set handle inheritance
                0,                     // No creation flags
                Nil,                  // Use parent's environment block
                Nil,                  // Use parent's starting directory 
                &si,                   // Pointer to STARTUPINFO structure
                &pi,                   // Pointer to PROCESS_INFORMATION structure
                DllPaths,
                numDlls,
                NULL)
            ) 
            {
                LPVOID lpMsgBuf;
                FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM, 
                               Nil, GetLastError(), MAKELANGID(LANG_NEUTRAL,SUBLANG_DEFAULT),
                               (LPTSTR)&lpMsgBuf, 0, Nil);
            
                stdSYSLOG( "Failed to run %s (%s).\n", argv[0], lpMsgBuf );
                status= -1;
            } else {
                DWORD exitCode;
                WaitForSingleObject( pi.hProcess, INFINITE  );
                GetExitCodeProcess ( pi.hProcess, &exitCode );
                status= exitCode;

                if (reportCrash) { reportIfCrashed(argv[0],status); }

                CloseHandle( pi.hProcess );
                CloseHandle( pi.hThread );
            }
        }

        stdFREE(cmdLine);

        if (inputFile) {
            SetStdHandle (STD_INPUT_HANDLE, old_stdin);
            CloseHandle( new_stdin );
        }

        if (outputFile) {
            SetStdHandle (STD_OUTPUT_HANDLE, old_stdout);
            CloseHandle( new_stdout );
        }

        if (errorFile) {
            SetStdHandle (STD_ERROR_HANDLE, old_stderr);
            if (!outIsError) {
                CloseHandle( new_stderr );
            }
        }

        return status;
   
    }
    
#else
   // prevent complaints on empty object on Darwin
      int ______gg10fbc44;
#endif // (defined STD_OS_win32) && (defined BUILD_WITH_DETOURS)
