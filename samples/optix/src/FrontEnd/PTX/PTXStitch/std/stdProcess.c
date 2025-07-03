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
 *  Module name              : stdProcess.c
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
#include "stdMessageDefs.h"

#if (defined STD_OS_win32) && (defined BUILD_WITH_DETOURS)
    #include "detours.h"
#endif

#if defined(LWVM_WSL) && !defined(LWVM_WSA)
// FIXME: Remove this hack once DX moves to a sane linking model for Linux/WSL
int __xstat(int ver, const char* path, struct stat* stat_buf);
int __fxstat(int ver, int fildes, struct stat* stat_buf);

#define stat(x,y)       __xstat(3, x, y)
#define fstat(x,y)      __fxstat(3, x, y)
#endif

/*----------------------------- Process Creation -----------------------------*/

Bool procTrapOnError = False;

#if defined(STD_OS_win32) || defined(STD_OS_MinGW)

    #include "windows.h"
    #include "psapi.h"
#ifdef STD_OS_win32
    #include "crtdbg.h"
#endif

    // rmdir and getcwd are deprecated
    #define rmdir _rmdir
    #define getcwd _getcwd

    #ifdef NO_VXL_EXCEPTION_FILTER
        #define INSTALL_VXL_EXCEPTION_FILTER()

    #else

        #define INSTALL_VXL_EXCEPTION_FILTER() \
            SetUnhandledExceptionFilter( vxl_exception_filter )

        static LONG WINAPI vxl_exception_filter( struct _EXCEPTION_POINTERS *ExceptionInfo )
        {
            // Default action is to abort
            stdSYSLOG( "Internal error\n");

            #ifndef RELEASE
            {
                if (getelw("TRAP_INTO_DEBUGGER")) {
                    stdSYSLOG("Started the debugger because environment variable TRAP_INTO_DEBUGGER is set\n");
                    return EXCEPTION_CONTINUE_SEARCH;
                } else {
                    stdSYSLOG("Set the environment variable TRAP_INTO_DEBUGGER to break into the debugger next time\n");
                }
            }
            #endif

            return EXCEPTION_EXELWTE_HANDLER;
        }
    #endif


    /*
     * Function        : Perform all OS specific process setup
     * Parameters      : argv0        (I) Name by which the current
     *                                    exelwtable was ilwoked.
     *                                    This is either an absolute file name,
     *                                    or a relative name found via $PATH
     */
    void STD_CDECL procProcessSetup( cString argv0 )
    {
        // Make C runtime check errors go to stderr instead of popping up a window.
#ifdef STD_OS_win32
        _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE  | _CRTDBG_MODE_DEBUG);
        _CrtSetReportMode(_CRT_ERROR,  _CRTDBG_MODE_FILE  | _CRTDBG_MODE_DEBUG);
        _CrtSetReportMode(_CRT_WARN,   _CRTDBG_MODE_FILE  | _CRTDBG_MODE_DEBUG);

        _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
        _CrtSetReportFile(_CRT_ERROR,  _CRTDBG_FILE_STDERR);
        _CrtSetReportFile(_CRT_WARN,   _CRTDBG_FILE_STDERR);
#endif
        // set up filter
        INSTALL_VXL_EXCEPTION_FILTER();

        // Set stdout and stderr to not be buffered so we get all of our debug output.
        setvbuf(stderr, Nil, _IONBF, 0);

        // Winsock initialization
        #if !defined(STD_NOWINSOCK2)
        {
          WSADATA wsaData;
#ifdef STD_OS_win32
          WSAStartup(WINSOCK_VERSION,&wsaData);
#endif
        }
        #endif
    }


    static Int reportIfCrashed(cString command, Int status)
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
            default : sigtext=""; break;
            }

            stdCHECK( False, (stdMsgSignal,command,status,sigtext) );
        }

        return status;
    }

void STD_CDECL procDebugCommand( String argv[], cString inputFile, 
                                 cString outputFile, cString errorFile, 
                                 Bool appendOutput, Bool reportCrash, 
                                 PROCESS_INFORMATION *pi )
        {
        Int status = 0;
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
           cString      *arg   = argv;

           while (*arg) {
               cString pc= *(arg++);
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

            ZeroMemory( &si, sizeof(si) );
            ZeroMemory( pi, sizeof(*pi) );
            si.cb     = sizeof(si);

            // Start the child process.
            if( !CreateProcess( Nil,  // no module just use command line
                cmdLine,               // Command line  i.e. "c:\Program Files\my dir\cl.exe" -c "my file.c"
                Nil,                  // Process handle not inheritable
                Nil,                  // Thread handle not inheritable
                TRUE,                  // Set handle inheritance
                DEBUG_ONLY_THIS_PROCESS,// Enable debugging of this process
                Nil,                  // Use parent's environment block
                Nil,                  // Use parent's starting directory
                &si,                   // Pointer to STARTUPINFO structure
                pi)                  // Pointer to PROCESS_INFORMATION structure
            )
            {
                LPVOID lpMsgBuf;
                FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
                               Nil, GetLastError(), MAKELANGID(LANG_NEUTRAL,SUBLANG_DEFAULT),
                               (LPTSTR)&lpMsgBuf, 0, Nil);

                stdSYSLOG( "Failed to run %s (%s).\n", argv[0], lpMsgBuf );
                LocalFree(lpMsgBuf);
                status= -1;
            } else {
                if (reportCrash) { reportIfCrashed(argv[0],status); }
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

    }


    Int STD_CDECL procRunCommand( String argv[], cString inputFile, cString outputFile, cString errorFile, Bool appendOutput, Bool reportCrash )
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
                new_stderr = CreateFile (errorFile, GENERIC_WRITE, 0, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, Nil);
                SetStdHandle (STD_ERROR_HANDLE, new_stderr);
            }
        }

        {
           stdString_t buffer = stringNEW();
           cString     *arg   = argv;

           while (*arg) {
               cString pc= *(arg++);
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
            if( !CreateProcess( Nil,  // no module just use command line
                cmdLine,               // Command line  i.e. "c:\Program Files\my dir\cl.exe" -c "my file.c"
                Nil,                  // Process handle not inheritable
                Nil,                  // Thread handle not inheritable
                TRUE,                  // Set handle inheritance
                0,                     // No creation flags
                Nil,                  // Use parent's environment block
                Nil,                  // Use parent's starting directory
                &si,                   // Pointer to STARTUPINFO structure
                &pi )                  // Pointer to PROCESS_INFORMATION structure
            )
            {
                LPVOID lpMsgBuf;
                FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
                               Nil, GetLastError(), MAKELANGID(LANG_NEUTRAL,SUBLANG_DEFAULT),
                               (LPTSTR)&lpMsgBuf, 0, Nil);

                stdSYSLOG( "Failed to run %s (%s).\n", argv[0], lpMsgBuf );
                LocalFree(lpMsgBuf);
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

    typedef BOOL (WINAPI *LPFN_ISWOW64PROCESS)(HANDLE, PBOOL);

    /*
     * Function        : Check the machine bitness
     * Function Result : If the machine is 64-bit
     */
    Bool STD_CDECL procIsMachine64Bit(void)
    {
    #if defined(STD_ARCH_x86_64)
        return True;  // 64-bit programs run only on Win64
    #elif (defined(STD_ARCH_i686) || defined(STD_ARCH_i386))
        // 32-bit programs run on both 32-bit and 64-bit Windows
        // so must sniff
        BOOL f64 = FALSE;
        LPFN_ISWOW64PROCESS fnIsWow64Process;
        fnIsWow64Process = (LPFN_ISWOW64PROCESS)GetProcAddress(GetModuleHandle(TEXT("kernel32")), "IsWow64Process");
        if (NULL != fnIsWow64Process)
            if (!fnIsWow64Process(GetLwrrentProcess(), &f64))
                return False;

        return f64;
    #else
        return False; // Win64 does not support Win16
    #endif
    }

#else

    #include "sys/utsname.h"

    /*
     * Function        : Perform all OS specific process setup
     * Parameters      : argv0        (I) Name by which the current
     *                                    exelwtable was ilwoked.
     *                                    This is either an absolute file name,
     *                                    or a relative name found via $PATH
     */

    void STD_CDECL procProcessSetup( cString argv0 )
    {
    }


    static Int reportIfCrashed(cString command, Int status)
    {
        if (!WIFSIGNALED(status)) {
            return WEXITSTATUS(status);
        } else {
            int termsig = WTERMSIG(status);
            String sigtext= "";

            #ifdef SIGKILL
            switch (termsig) {
            case SIGKILL : sigtext="(Kill signal)";              break;
            case SIGFPE  : sigtext="(Floating point exception)"; break;
            case SIGILL  : sigtext="(Illegal Instruction)";      break;
            case SIGBUS  : sigtext="(Bus error)";                break;
            case SIGSEGV : sigtext="(Invalid memory reference)"; break;
            default      :                                       break;
            }
            #endif

            stdCHECK( False, (stdMsgSignal,command,termsig,sigtext) );

            if (WCOREDUMP(status)) {
                stdCHECK( False, (stdMsgCoreDumped,command) );
            }

            return status;
        }
    }

    Int STD_CDECL procRunCommand( String argv[], cString inputFile, cString outputFile, cString errorFile, Bool appendOutput, Bool reportCrash )
    {
    #if !defined(STD_OS_Hos)
        uInt pid = fork();

        if (pid) {
            /* parent process */
            int status = 0;

            do {
                if(waitpid(pid, &status, 0) == -1) {
                    if (errno == EINTR) {
                        continue;
                    } else {
                        /* return directly as status might not be set correctly on error */
                        return -errno;
                    }
                }
            } while(!WIFEXITED(status) && !WIFSIGNALED(status));

            if (reportCrash) { reportIfCrashed(argv[0], status); }

            return WEXITSTATUS(status);

        } else {
            FILE *f;

            if (inputFile) {
                close(0);
                f= fopen(inputFile,  "r");
                stdCHECK( f, (stdMsgOpenInputFailed, inputFile) );
            }
            if (outputFile) {
                close(1);
                f= fopen(outputFile, appendOutput?"a":"w");
                stdCHECK( f, (stdMsgOpenOutputFailed, outputFile) );
                if (errorFile && stdEQSTRING(outputFile,errorFile)) {
                    close(2);
                    dup(1);
                    errorFile= Nil;
                }
            }
            if (errorFile) {
                close(2);
                f= fopen(errorFile,  "w");
                stdCHECK( f, (stdMsgOpenOutputFailed, outputFile) );
            }

            execvp(argv[0],argv);
            perror(argv[0]);

            stdEXIT_ERROR();
        }
    #else
        return 0;
    #endif
    }


    /*
     * Function        : Check the machine bitness
     * Function Result : If the machine is 64-bit
     */
    Bool STD_CDECL procIsMachine64Bit(void)
    {
        #if !defined(STD_OS_Darwin) && defined(STD_64_BIT_ARCH)
            return True;
        #elif defined(STD_OS_Hos)
            // 64-bit HOS is caught in the above check, so we must be 32-bit if
            // if we get here. We cannot use uname() below on HOS.
            return False;
        #else
            struct utsname name;
            int result;

            result = uname(&name);
            if (result != 0)
                return False;

            if (strcmp(name.machine, "x86_64") == 0)
                return True;

            return False;
        #endif
    }

#endif


void STD_CDECL procSetTrapOnError(void)
{
#if defined(STD_OS_win32) || defined(STD_OS_MinGW)
    putelw( "TRAP_INTO_DEBUGGER=1" );
    _set_error_mode(_OUT_TO_MSGBOX);
#ifdef STD_OS_win32
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_DEBUG | _CRTDBG_MODE_WNDW);
#endif
#endif
    procTrapOnError= True;
}



/*-------------------------------- 'System' ---------------------------------*/


Int STD_CDECL procSystem( FILE* traceFile, Bool verbose, Bool dryrun, Bool force, Bool exitOnError, String x )
{
    Int err= 0;

#if !defined(STD_OS_Hos)
#ifdef STD_OS_win32
    Char *c=x;

   /*
    * Exelwtable path should not contain forward slashes,
    * since the DOS shell will confuse these with options:
    */

    while (*c && *c!=' ') {
        if (*c=='/') { *c = '\\'; }
        c++;
    }
#endif

    if (traceFile && (verbose || dryrun)) {
        fprintf(traceFile,"#$ %s\n",x);
        fflush (traceFile);
    }

    if (force || !dryrun) {
        err= system(x);
        if (err) {
            String      cmd = x;
            stdString_t name= stringNEW();

            while (cmd[0] && cmd[0] != ' ') { stringAddChar(name,*(cmd++)); }
            cmd = stringStripToBuf(name);

            err = reportIfCrashed(cmd, err);
            if (verbose || dryrun) { stdSYSLOG("# --error 0x%x --\n", err); }
            if (exitOnError      ) { stdEXIT(err);                          }

            stdFREE(cmd);
        }
    }
#endif

    return err;
}

Int STD_CDECL procVSystem( FILE* traceFile, Bool verbose, Bool dryrun, Bool force, Bool exitOnError, cString format, ... )
{
    Char buffer[100000];
    va_list arg;

    va_start(arg,format);
    vsprintf(buffer,format,arg);
    va_end(arg);

    return procSystem(traceFile,verbose,dryrun,force,exitOnError,buffer);
}

Int STD_CDECL procStringSystem( FILE* traceFile, Bool verbose, Bool dryrun, Bool force, Bool exitOnError, stdString_t command )
{
    Int result;
    String cmdline = stringStripToBuf(command);

    result= procSystem(traceFile,verbose,dryrun,force,exitOnError,cmdline);

    stdFREE(cmdline);

    return result;
}


/*--------------------------- Current Working Dir ---------------------------*/

/*
 * Function        : Return absolute pathname of the current working directory.
 * Function Result : Absolute pathname of the current working directory.
 */
String STD_CDECL procGetPwd(void)
{
   Char *buffer;
   uInt size = 100;

   do {
       size  *= 2;
       buffer = (Char*)alloca(size);
   } while (!getcwd(buffer,size));

#ifdef STD_OS_win32
   {
       Char *c=buffer;
       while (*c) {
           if (*c=='\\') { *c= '/'; }
           c++;
       }
   }
#endif

   return stdCOPYSTRING(buffer);
}


/*------------------------------ Platform Info ------------------------------*/

/*
 * Function        : Return True iff the current platform is 64 bit.
 * Function Result : Sic.
 */
Bool STD_CDECL procIs64Bit(void)
{
#ifdef STD_OS_win32_DISABLED
    #if defined(_WIN64)
        return True;  // 64-bit programs run only on Win64
    #elif defined(_WIN32)
#if defined(GPU_DRIVER_SASSLIB)
        return False; // sasslib does not care
#else
        // 32-bit programs run on both 32-bit and 64-bit Windows
        // so must sniff
        BOOL f64 = FALSE;
        return IsWow64Process(GetLwrrentProcess(), &f64) && f64;
#endif
    #else
        return False; // Win64 does not support Win16
    #endif
#else
    #ifdef STD_64_BIT_ARCH
        return True;
    #else
        return False;
    #endif
#endif
}


/*----------------------------- Temp File Names -----------------------------*/

    /*
     * The following guarantees that temporary files can be safely deleted by
     * the atexit handler 'rmtemps'. If we do not do anything about it, then
     * upon a fatal error, files might still be open when rmtemps is exelwted.
     * This definitely oclwrs on Windows:
     */

    static stdSet_t openFiles;

    FILE* STD_CDECL procFOPEN( cString name, cString mode )
    {
        FILE *result= fopen(name,mode);

        if (result) {
            stdMemSpace_t savedSpace = stdSwapMemSpace( memspNativeMemSpace );
            if (!openFiles) { openFiles= setNEW(Pointer,32); }
            setInsert(openFiles,result);
            stdSwapMemSpace( savedSpace );
        }

        return result;
    }

    void STD_CDECL procFCLOSE( FILE* f )
    {
        if (openFiles) { setRemove(openFiles,f); }
        fclose(f);
    }

    static String tempName    = Nil; /* tmp files, common part including full path */
    static String tempDirName = Nil; /* tmp dirs, common part including full path */

    static void STD_CDECL removeFile( cString name )
    {
    #if !defined(STD_OS_Hos)
        if (fnamIsDirectory(name)) {
            fnamTraverseDirectory( name, Nil, False, True, (stdEltFun)removeFile, Nil );
            rmdir(name);
        } else {
            unlink(name);
        }
    #endif
    }

    static void STD_CDECL rmtemps(void)
    {
         String      tn  = stdCOPYSTRING(tempName);
         stdString_t buf = stringNEW();

         String      tempSysDir, tempPrefix, tempPattern;

       #ifdef STD_OS_win32
        /*
         * DOS command 'erase' (stdRM) does not like forward slashes,
         * so colwert them here:
         */
         Char *s= strchr(tn,'/');
         while (s) { *s= '\\'; s= strchr(s,'/'); }
       #endif

        /*
         * Close all registered open files:
         */
         if (openFiles) {
             setTraverse( openFiles, (stdEltFun)fclose, Nil );
             setDelete(openFiles);
         }


        /*
         * Remove all temporary files created by this process:
         */
         fnamDecomposePath(tn, &tempSysDir, &tempPrefix, Nil);
         stringAddFormat(buf, "%s*", tempPrefix);
         tempPattern= stringStripToBuf(buf);

         fnamTraverseDirectory( tempSysDir, tempPattern, False, True, (stdEltFun)removeFile, Nil );


        /*
         * Cleanup:
         */
         stdFREE(tempPattern);
         stdFREE(tempPrefix);
         stdFREE(tempName);
         stdFREE(tempSysDir);
         stdFREE(tn);
    }


    /*
     * Remove all files in a directory and the directory itself
     */
    static void STD_CDECL cleanDir( cString dir )
    {
    #if !defined(STD_OS_Hos)
        /* rmdir only works on empty directories, so unlink all files first */
        fnamTraverseDirectory(dir, "*", False, True, (stdEltFun)unlink, Nil);
        rmdir(dir);
    #endif
    }

    static void STD_CDECL rmtempDirs(void)
    {
         String      tn  = stdCOPYSTRING(tempDirName);
         stdString_t buf = stringNEW();

         String      tempSysDir, tempPrefix, tempPattern;

       #ifdef STD_OS_win32
        /*
         * DOS command 'erase' (stdRM) does not like forward slashes,
         * so colwert them here:
         */
         Char *s= strchr(tn,'/');
         while (s) { *s= '\\'; s= strchr(s,'/'); }
       #endif

        /*
         * Remove all temporary directories and their files created by this process:
         */

         /* decompose into sys-dir/tmp-dir/.d to get tmp-dir */
         fnamDecomposePath(tn, &tempSysDir, &tempPrefix, Nil);
         stringAddFormat(buf, "%s*", tempPrefix);
         tempPattern = stringStripToBuf(buf);

         fnamTraverseDirectory( tempSysDir, tempPattern, False, True, (stdEltFun)cleanDir, Nil );

        /*
         * Cleanup:
         */
         stdFREE(tempPattern);
         stdFREE(tempPrefix);
         stdFREE(tempDirName);
         stdFREE(tempSysDir);
         stdFREE(tn);
    }


    static String tempDir()
    {
#if defined(__ANDROID__)
        struct stat st;
        static String result = NULL;
        static String command = "mkdir -p /data/app/lwpu";
        result = command + 9;

        if (stat(result, &st)) {
            int err = system(command);
            if (err) {
                stdSYSLOG("Failed to create temp directory '%s'.\n", result);
                stdEXIT(err);
            }
        } else {
            if (!S_ISDIR(st.st_mode)) {
                stdSYSLOG("Failed to create temp directory '%s' as a file with the same name exists.\n", result);
                stdEXIT(-1);
            }
        }

        return stdCOPYSTRING(result);
#else
        static String result = NULL;

        if (!result) {
#if defined(STD_OS_win32)
            result= stdGetElw("TEMP");
#else
            result= stdGetElw("TMPDIR");
#endif

            if (!result) {
#if defined(STD_OS_win32)
                result= "c:/windows/temp";
#else
                result= "/tmp";
#endif
            }
        }

        /* result is static, so return copy of result, not result itself */
        return stdCOPYSTRING(result);
#endif
    }


/*
 * Function        : Return a fresh temporary file name.
 * Function Result : Name of file.
 */
String STD_CDECL procGetTempName()
{
    static uInt instanceNr   = 0;
           uInt generationNr = 0;
    stdString_t name= stringNEW();

    stdMemSpace_t savedSpace = stdSwapMemSpace( memspNativeMemSpace );
    {
        int writeAttempts = 0;

        /* generate a common temp file name */
        while (!tempName) {
            FILE *f;
            Char buffer[40];
            String tmpdir = tempDir();
            stdCHECK(tmpdir, (stdMsgBailoutDueToErrors));
            sprintf(buffer,"/tmpxft_%08x_%08x", getpid(), generationNr++ );
            tempName = stdCONCATSTRING(tmpdir, buffer);
            stdFREE(tmpdir);
            f= fopen(tempName,  "r");

            if (f) {
                fclose(f);
                stdFREE(tempName);
                tempName=Nil;
            } else {
                static Bool first= True;

                if (first) { stdSetCleanupHandler((stdDataFun)rmtemps,Nil); }
                first= False;

                f= procFOPEN(tempName,  "w");
                // try 10 times before issuing an error
                if (!f && (writeAttempts < 10)) {
                  writeAttempts++;
                  stdFREE(tempName);
                  tempName = Nil;
                } else {
                  stdCHECK( f, (stdMsgOpenOutputFailed, tempName) ) {
                    procFCLOSE(f);
                  }
                }
            }
        }
    }
    stdSwapMemSpace(savedSpace);

    stringAddBuf    (name,tempName);
    /* make the returned name unique */
    stringAddFormat (name,"-%d",stdAtomicFetchAndAdd(&instanceNr, 1));

    return stringStripToBuf(name);
}

String STD_CDECL procGetTempNameBase(void)
{
  return tempName;
}

/*
 * Function        : Return a fresh temporary directory name.
 * Function Result : Name of directory.
 */
String STD_CDECL procGetTempDirName()
{
    static Int  instanceNr   = 0;
           Int  generationNr = 0;
    stdString_t name= stringNEW();

    stdMemSpace_t savedSpace = stdSwapMemSpace( memspNativeMemSpace );
    {
        /* generate a common temp dir name */
        while (!tempDirName) {
            Char buffer[40];
            String tmpdir = tempDir();
            stdCHECK(tmpdir, (stdMsgBailoutDueToErrors));
            /* create new temporary directory below system-wide tempDir */
            sprintf(buffer,"/tmpxft_%08x_%08x", getpid(), generationNr++ );
            tempDirName = stdCONCATSTRING(tmpdir, buffer);
            stdFREE(tmpdir);

            if (tempDirName) {
                //validate directory name with comparison to blacklist
                int len = strlen(tempDirName);
                int i;
                for (i = 0; i < len; ++i) {
                    if ( '*' == *(tempDirName+i)
                      || '\?' == *(tempDirName+i)
                      || '\"' == *(tempDirName+i)
                      || '<' == *(tempDirName+i)
                      || '>' == *(tempDirName+i)
                      || '|' == *(tempDirName+i))
                    {
                        stdCHECK(False, (stdMsgOpenOutputFailed, tempDirName));
                        stdFREE(tempDirName);
                        tempDirName = Nil;
                        return NULL;
                    }
                }
            }
            if (stdMkdir(tempDirName)) {
                static Bool first = True;

                if (first) {
                    stdSetCleanupHandler((stdDataFun)rmtempDirs, Nil);
                }
                first = False;
            } else {
                stdFREE(tempDirName);
                tempDirName = Nil;
            }
        }
    }
    stdSwapMemSpace(savedSpace);

    stringAddBuf(name, tempDirName);
    /* make the returned name unique */
    stringAddFormat(name, "-%d", instanceNr++);

    return stringStripToBuf(name);
}


/*
 * Function        : Decide if specified file name has been returned by
 *                   function procGetTempName.
 * Parameters      : name  (I) File name to inspect.
 * Function Result : True iff name is the name of a temp file.
 */
Bool STD_CDECL procIsTempName( cString name )
{
    return tempName
        && stdIS_PREFIX(tempName,name);
}


/*
 * Function        : Decide if specified directory name has been returned by
 *                   function procGetTempDirName.
 * Parameters      : name  (I) Directory name to inspect.
 * Function Result : True iff name is the name of a temp directory.
 */
Bool STD_CDECL procIsTempDirName( cString name )
{
    return tempDirName
        && stdIS_PREFIX(tempDirName, name);
}


/*
 * Function        : Return absolute pathname of the current exelwtable.
 * Function Result : Name of exelwtable running current process.
 */

#ifdef STD_OS_FAMILY_Unix
    char *get_lwrrent_dir_name(void);

    String STD_CDECL procLwrrentExelwtableName(void)
    {
        Char *result = Nil;
    #if !defined(STD_OS_Hos)
        Char buffer1[64];
        Int len, size = 64;

        sprintf(buffer1,"/proc/%d/exe",getpid());

        do {
          if(result) stdFREE(result);
          size <<= 1;
          result = (Char*) stdMALLOC(size);
          len = readlink(buffer1, result, size);

          if (len == -1) {
              stdFREE(result);
              return Nil;
          }

        } while (len == size);

        result[len] = '\0';
    #endif
        return result;
    }

#elif defined(STD_OS_Darwin)
    #include "mach-o/dyld.h"

    String STD_CDECL procLwrrentExelwtableName(void)
    {
        Char result[10000];
        uint32_t buflen= sizeof(result);

        if (!_NSGetExelwtablePath(result,&buflen)) {
            return stdCOPYSTRING(result);
        } else {
            String result1= stdMALLOC(buflen);
            _NSGetExelwtablePath(result1,&buflen);
            return result1;
        }
    }

#else
    String STD_CDECL procLwrrentExelwtableName(void)
    {
        Char buffer[10000];

        if (GetModuleFileName( GetModuleHandle(0), buffer, sizeof(buffer) ) == sizeof(buffer)) {
            return Nil;
        } else {
            Char *f= buffer;
            while ( (f=strchr(f,'\\') ) ) { *f= '/'; }

            return stdCOPYSTRING(buffer);
        }
    }
#endif


#ifdef STD_OS_win32
    String STD_CDECL procGetHostName(void)
    {
    #if !defined(STD_NOWINSOCK2)
        WORD wVersionRequested;
        WSADATA wsaData;
        int res;
        Char buffer[256]; // 256 is always enough, see
                          // https://msdn.microsoft.com/en-us/library/windows/desktop/ms738527%28v=vs.85%29.aspx

        // Use the MAKEWORD(lowbyte, highbyte) macro declared in Windef.h
        wVersionRequested = MAKEWORD(2, 2);

        res = WSAStartup(wVersionRequested, &wsaData);
        if (res != 0) {
            return Nil;
        }

        res = gethostname(buffer, sizeof(buffer));

        WSACleanup();

        return stdCOPYSTRING(buffer);
    #else //STD_NOWINSOCK2
        return Nil;
    #endif
    }
#else
    String STD_CDECL procGetHostName(void)
    {
    #if !defined(STD_OS_Hos)
        #ifndef MAXHOSTNAMELEN
        #define MAXHOSTNAMELEN   10000
        #endif

        #ifndef HOST_NAME_MAX
        #define HOST_NAME_MAX    MAXHOSTNAMELEN
        #endif

        Char buffer[HOST_NAME_MAX];

        if (gethostname(buffer, sizeof(buffer)) != 0) {
            return Nil;
        } else {
            return stdCOPYSTRING(buffer);
        }
    #else
        return Nil;
    #endif
    }
#endif


#ifndef STD_OS_win32
    void STD_CDECL stdSetElw(const Char *elw, const Char *val) {
        // setelw() is not available on HOS, nor do we require it
#if !defined(STD_OS_Hos)
        setelw(elw, val, 1);
#endif
    }

    String STD_CDECL stdGetElw(const Char *elw) {
        String ret = Nil;
        String val = Nil;

        ret = getelw(elw);
        if (ret) {
            val = stdCOPYSTRING(ret);
        }

        return val;
    }

#else
    void STD_CDECL stdSetElw(const Char *elw, const Char *val) {
        SetElwironmentVariable(elw, val);
    }

    String STD_CDECL stdGetElw(const Char *elw) {
        const SizeT size = 256;
        String      val = Nil;
        DWORD       ret;

        val = stdCALLOC(size, 1);
        ret = GetElwironmentVariable(elw, val, (DWORD)size);

        if (0 == ret) {
            stdFREE(val);
            val = Nil;
        } else if(ret > size) {
             val = stdREALLOC(val, ret + 1);
             GetElwironmentVariable(elw, val, (DWORD)ret);
        }

        return val;
    }

#endif


#if !defined(__ANDROID__)
    Bool STD_CDECL procpThreadIdHash ( procThreadId_t *l )
    { return STD_POINTER_HASH((Address)*l); }

    Bool STD_CDECL procpThreadIdEqual( procThreadId_t *l, procThreadId_t *r )
    { return *l == *r; }
#endif
