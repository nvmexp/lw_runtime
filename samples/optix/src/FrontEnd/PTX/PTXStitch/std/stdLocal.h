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
 *  Module name              : stdLocal.h
 *
 *  Description              :
 *
 *     This is the basic macro file and portability file for 
 *     use in developed software. General purpose macros should
 *     be taken from this file rather than defined elsewhere, and 
 *     developers are encouraged to provide new definitions for this file
 *     rather than to define local (and probably redundant) definitions
 *     in their software components.
 *     The rationale for this file is as follows:
 *
 *        - Suppressing the need for developer specific general purpose
 *          macros, enhancing overall source code consistency
 *          of the software repertoire
 *        - The basic macro names follow the naming convention of the
 *          coding style, hence enhancing source code uniformity
 *        - Facilities are defined for system printing, memory allocation,
 *          endian colwersion, alignment, which means that all this can
 *          be redirected to a different implementation by a change in
 *          a central place, that is, here.
 */

#ifndef stdLocal_INCLUDED
#define stdLocal_INCLUDED

/*------------------------------- Includes -----------------------------------*/

#include "stdTypes.h"
#include "stdStdFun.h"

#ifdef _MSC_VER
#pragma warning (disable : 4127)
#endif

// When trying to include C header file in C++ Code extern "C" is required
// But the Standard QNX headers already have ifdef extern in them when compiling C++ Code
// extern "C" cannot be nested
// Hence keep the header out of extern "C" block
#include "math.h"

#ifdef __cplusplus
extern "C" {
#endif

/*-------------------------------- Constants -------------------------------*/

#if defined(STD_ARCH_i686) || defined(STD_ARCH_i386)
    #define stdALIGN_SHIFT                     3  /*  8 bytes alignment */
    #define __LITTLE_ENDIAN_OS__


#elif defined(STD_ARCH_x86_64)
    #define stdALIGN_SHIFT                     3  /*  8 bytes alignment */
    #define __LITTLE_ENDIAN_OS__


#elif defined(STD_ARCH_Elrond)
    #define stdALIGN_SHIFT                     3  /*  8 bytes alignment */
    #define __KERNEL__
    #define __LITTLE_ENDIAN_OS__


#elif defined(STD_ARCH_ARMv7)
    #define stdALIGN_SHIFT                     3  /*  8 bytes alignment */
    #define __LITTLE_ENDIAN_OS__


#elif defined(STD_ARCH_aarch64)
    #define stdALIGN_SHIFT                     3  /*  8 bytes alignment */
    #define __LITTLE_ENDIAN_OS__


#elif defined(STD_ARCH_ppc64le)
    #define stdALIGN_SHIFT                     3  /*  8 bytes alignment */
    #define __LITTLE_ENDIAN_OS__


#else
    #error "Platform not supported"    
#endif

#if   defined(__LITTLE_ENDIAN_OS__)
            #define stdLWRRENTENDIAN          stdLittleEndian
#elif defined(__BIG_ENDIAN_OS__)
            #define stdLWRRENTENDIAN          stdBigEndian
#endif


#define stdALIGNMENT   (1<<stdALIGN_SHIFT)


/*--------------------------------- Macros ---------------------------------*/

#if defined(STD_OS_win32) || defined (STD_OS_MinGW)

   /*
    * Change _WIN32_WINNT to windows XP.
    * This allows the components to use features like
    * CreateWaitableTimer.
    * See https://msdn.microsoft.com/en-us/library/windows/desktop/aa383745(v=vs.85).aspx
    */
    #undef _WIN32_WINNT
    #ifdef WIN7
    #define _WIN32_WINNT 0x0601
    #else
    #define _WIN32_WINNT 0x0501
    #endif
 
  #if !defined(STD_NOWINSOCK2)
    #include "winsock2.h"
  #endif
  
    #include "windows.h"
    #include "setjmp.h"
    #include "stdlib.h"
    #include "string.h"
    #include "ctype.h"
    #include "stdio.h"
    #include "limits.h"
    #include "malloc.h"
    #include "errno.h"
    #include "sys/types.h"
    #include "sys/stat.h"
    #include "time.h"
    #include "fcntl.h"
    #include "sys/stat.h"
    #include "stdarg.h"
    #include "process.h"
    #include "io.h"
    #include "direct.h"
#if defined(STD_OS_win32)
    #include "crtdefs.h"
#endif

// inline is a keyword in C++ and at least msvc140u3+ complains about it being redefined
// fatal error C1189: #error:  The C++ Standard Library forbids macroizing keywords.
#ifndef __cplusplus
    #define inline         __inline
#endif

    #define getpid         _getpid
    #define putelw         _putelw
    #define unlink         _unlink 
    #define mkdir(d,m)     _mkdir(d) 
    #define memalign(a,s)  _aligned_malloc(s,a)
    #define snprintf       _snprintf


#ifdef STD_OS_MinGW
    int random(void);
#endif

#elif defined(STD_OS_NewLib)

    #include "stdlib.h"
    #include "string.h"
    #include "ctype.h"
    #include "stdio.h"
    #include "limits.h"
    #include "malloc.h"
    #include "errno.h"
    #include "sys/types.h"
    #include "sys/stat.h"
    #include "unistd.h"
    #include "sys/time.h"
    #include "sys/fcntl.h"
    #include "sys/wait.h"
    #include "sys/stat.h"
    #include "stdarg.h"

#elif defined(STD_OS_FAMILY_Unix) || defined(STD_OS_CygWin) || defined(STD_OS_Darwin)

  #if defined(STD_OS_Linux)
    #include "syscall.h"
  #elif defined(STD_OS_QNX)
    #include "sys/mman.h"
  #endif

  #if defined(STD_OS_CygWin)
    #include "windows.h"
  #endif
  
    #include "setjmp.h"
    #include "stdlib.h"
    #include "string.h"
    #include "ctype.h"
    #include "stdio.h"
    #include "limits.h"
    #include "errno.h"
    #include "sys/types.h"
    #include "sys/stat.h"
    #include "unistd.h"
    #include "sys/time.h"
    #include "sys/socket.h"
    #include "netinet/in.h"
    #include "netinet/tcp.h"

  #if defined(STD_OS_Android) || defined(STD_OS_QNX) || defined(STD_OS_Hos)
    #include "fcntl.h"
  #else
    #include "sys/fcntl.h"
  #endif

    #include "sys/ioctl.h"
#ifndef STD_OS_Hos
    #include "netdb.h"
#endif
    #include "sys/wait.h"
    #include "sys/stat.h"
    #include "stdarg.h"
    #include "pthread.h"
#if defined(STD_OS_FreeBSD)
    #include  "signal.h"
#endif
    #include "semaphore.h"
    #include "sys/time.h"
    #include "dlfcn.h"
    
    
    #include "dirent.h"
    // This macro conflicts with Darwin sys/dirent.h  
    #undef DT_UNKNOWN
    
    
  #ifdef STD_OS_Darwin
    
    #include "mach/host_info.h"
    #include "mach/mach_host.h"
    #include "mach/mach_time.h"
    #include "sys/types.h"
    #include "sys/sysctl.h"
    #include "sys/mman.h"
    #include "libkern/OSAtomic.h"

    #include "string.h"
    #include "time.h"
    #include "unistd.h"
    #include "errno.h"
    #include "stdio.h"
    #include "stdlib.h"
    #include "sys/time.h"
    #include "unistd.h"
  #endif

#endif

// zlib happens to be not installed on all DVS 
// build elwironments.
#ifdef NEEDS_ZLIB
#include "zlib.h"
#endif

/*-------------------------- Exelwtion Context -----------------------*/

typedef struct stdThreadContext  *stdThreadContext_t;

#include "stdMemSpace.h"
#include "stdThreads.h"

typedef void (STD_CDECL *stdLogLineFunc)( String line );

// client specific attributes of message statement
typedef struct msgStateInfo {
    cString        msgToolName;
    String         msgSuffix;
    Bool           msgIgnoreWarnings;
    Bool           msgWarnAsError;
    Bool           msgAddErrorClassPrefixes;
    Bool           msgDontPrefixErrorLines;
    Bool           oldMsgAddErrorClassPrefixes;
    Bool           oldMsgDontPrefixErrorLines;
    stdLogLineFunc logLine;
    void          *logLineData;
    stdLogLineFunc oldLogLineFunc;
    void          *oldLogLineData;
    void /*stdString_t*/ *msgBuff; /* Not able to include stdString.h due to include cycle*/
} msgStateInfo;

typedef struct memStateInfo {
    void /*stdRangeMap_t*/ *blockToSpacePage;   /* Pointer --> Page */
} memStateInfo;

/*
 * Thread exelwtion context:
 */
typedef struct stdThreadContext {
    Bool              warningsFound;
    Bool              errorsFound;
    jmp_buf          *lwrrentContext;
    msgMessage        raisedException;
    stdMemSpace_t     lwrrentMemspace;
    msgStateInfo      mInfo;
    memStateInfo      memInfo;
#if defined(STD_OS_win32) && defined(USE_NATIVE_HEAP)
    HANDLE            winHeapHandle;
#endif
} stdThreadContext;


/*---------------------------- System Timers -------------------------*/

#if defined(STD_OS_win32)
    typedef struct stdTimer {
        LARGE_INTEGER t;
    } stdTimer;
    
#elif defined(STD_OS_FAMILY_Unix)
    typedef struct stdTimer {
        struct timeval t;
    } stdTimer;
    
#elif defined(STD_OS_CygWin)
    typedef struct stdTimer {
        struct timeval t;
    } stdTimer;
    
#elif defined(STD_OS_Darwin)
    typedef struct stdTimer {
    uint64_t t;
    } stdTimer;
    
#else
   #define __NO_stdTimer__
#endif


#ifndef __NO_stdTimer__
    void  STD_CDECL stdTimerInit(void);
    void  STD_CDECL stdResetTimer(stdTimer *timer);
    Float STD_CDECL stdGetTimer(stdTimer *timer);
#endif

/*-------------------------- Interrupts En/Disabling -----------------------*/

#if defined(STD_ARCH_XTensa)
    #define stdDisableInterrupts(a)  /* To be defined */
    #define stdRestoreInterrupts(a)  /* To be defined */
#endif

/*--------------------------- Section Boundaries ---------------------------*/

#if defined (STD_OS_win32)
  void STD_CDECL stdGetwin32SectionBoundaries(cString section, Pointer *start, Pointer *end);

  #define stdSectionBoundaries(section,start,end) \
      stdGetwin32SectionBoundaries(#section,&start,&end);
      
#elif defined (STD_ARCH_XTensa)
  #define stdSectionBoundaries(section,start,end) \
  {                                    \
      extern char section##_start[];   \
      extern char section##_end[];     \
      start= (Pointer)section##_start; \
      end  = (Pointer)section##_end;   \
  }
  
#elif defined (STD_OS_CygWin)

 /*
  * In my opinion a bug in the CygWin linker:
  * the __start_xx symbol points to *before*
  * the inter-segment padding:
  */
  #define stdSEGMENT_ALIGNMENT  0x1000

  #define stdSectionBoundaries(section,start,end) \
  {                                      \
      extern char __start_##section[];   \
      extern char __stop_##section[];    \
      start= (Pointer)stdROUNDUP64(__start_##section, stdSEGMENT_ALIGNMENT); \
      end  = (Pointer)__stop_##section;  \
  }
  
#else
  #define stdSectionBoundaries(section,start,end) \
  {                                      \
      extern char __start_##section[];   \
      extern char __stop_##section[];    \
      start= (Pointer)__start_##section; \
      end  = (Pointer)__stop_##section;  \
  }
#endif

/*--------------------------------- Macros ---------------------------------*/

#define _U16_(x)               ((uInt16)(x))
#define _U32_(x)               ((uInt32)(x))
#define _U64_(x)               ((uInt64)(x))
#define _S_(s)                 #s
#define _CC2_(s1,s2)           s1##s2
#define _CC3_(s1,s2,s3)        s1##s2##s3

#ifdef STD_64_BIT_ARCH
#define _A_(x)                 _U64_(x)
#else
#define _A_(x)                 _U32_(x)
#endif

#define stdINRANGE32(x,b,s)      ( ((s) > 0) && ( (_U32_(x)-_U32_(b)) < _U32_(s) ) )
#define stdINRANGE64(x,b,s)      ( ((s) > 0) && ( (_U64_(x)-_U64_(b)) < _U64_(s) ) )
#define stdINRANGE(x,b,s)        stdINRANGE32(x,b,s)

#define stdOVERLAP(b1,s1,b2,s2) \
                               ( (_U32_(b1) < _U32_((b2)+(s2))) \
                              && (_U32_(b2) < _U32_((b1)+(s1))) \
                               )

#define stdOVERLAP64(b1,s1,b2,s2) \
                               ( (_U64_(b1) < _U64_((b2)+(s2))) \
                              && (_U64_(b2) < _U64_((b1)+(s1))) \
                               )


#define stdINCLUDED(b1,s1,b2,s2) \
                               ( ((s1)>0) && stdINRANGE(b1,b2,s2) && stdINRANGE((b1)+(s1)-1,b2,s2) )

#define stdINCLUDED64(b1,s1,b2,s2) \
                               ( ((s1)>0) && stdINRANGE64(b1,b2,s2) && stdINRANGE64((b1)+(s1)-1,b2,s2) )

#define stdUPPERWORD64(x)      (_U32_(_U64_(x) >> 32))
#define stdLOWERWORD64(x)      (_U32_(_U64_(x) & 0xffffffff))

#define stdSTRING(s)          _S_(s)
#define stdCONCAT(s1,s2)      _CC2_(s1,s2)
#define stdCONCAT3(s1,s2,s3)  _CC3_(s1,s2,s3)

#define stdMEMBER(s)           s,
#define stdMEMBER_STRING(s)   _S_(s),

#define stdBITMASK32(b,s)      ( ( (s) == 32 ? ((uInt32)-1) : ((((uInt32)1)<<(s))-1) ) << (b) )
#define stdBITMASK64(b,s)      ( ( (s) == 64 ? ((uInt64)-1) : ((((uInt64)1)<<(s))-1) ) << (b) )
#define stdBITMASK(b,s)        stdBITMASK32(b,s)

#define stdSIGN_EXTEND32(value,nrofBits) \
                              ( (  ((Int32)(value)) << (32-(nrofBits))  ) >> (32-(nrofBits)) )
#define stdSIGN_EXTEND64(value,nrofBits) \
                              ( (  ((Int64)(value)) << (64-(nrofBits))  ) >> (64-(nrofBits)) )
#define stdSIGN_EXTEND(value,nrofBits) \
                              stdSIGN_EXTEND32(value,nrofBits)


#if   defined STD_OS_win32
    #define __ALWAYS_INLINE__ 
    #define __CHECK_FORMAT__(formatType, stringArg, varArg) 
#elif defined STD_ARCH_XTensa
    #define __ALWAYS_INLINE__  __attribute__((always_inline))
    #define __CHECK_FORMAT__(formatType, stringArg, varArg) __attribute__((format (formatType, stringArg, varArg)))
#elif __GNUC__ == 2
    #define __ALWAYS_INLINE__ 
    #define __CHECK_FORMAT__(formatType, stringArg, varArg) 
#else
    #define __ALWAYS_INLINE__  __attribute__((always_inline))
    #define __CHECK_FORMAT__(formatType, stringArg, varArg) __attribute__((format (formatType, stringArg, varArg)))
#endif



#if defined(STD_OS_win32)
  //#define stdTHREADLOCAL(decl)     __declspec(thread)       decl
    #define stdNORETURN(decl)        __declspec(noreturn)     decl
    #define stdALIGNED(decl,value)   __declspec(align(value)) decl
    #define stdUSED(decl)                                     decl
    #define stdUNUSED(decl)                                   decl
    #define stdSECTION(decl,s)       __declspec(allocate(s))  decl
    #define stdDEPRECATED(decl)                               decl
#else
    #define stdTHREADLOCAL(decl)                     __thread decl
    #define stdNORETURN(decl)                                 decl  __attribute__ ((noreturn))
    #define stdSECTION(decl,s)                                decl  __attribute__ ((section (s)))
    #define stdALIGNED(decl,value)                            decl  __attribute__ ((aligned (value)))
    #define stdUSED(decl)                                     decl  __attribute__ ((used))
    #define stdUNUSED(decl)                                   decl  __attribute__ ((unused))
    #define stdDEPRECATED(decl)                               decl  __attribute__ ((deprecated))
#endif


/*--------------------------- Memory Management ----------------------------*/

stdMemSpace_t STD_CDECL stdSwapMemSpace( stdMemSpace_t newSpace );

#ifdef _USE_NATIVE_MALLOC_
    #define stdLwrrentMemspace Nil

    #define __MALLOC(s)       malloc(s)
    #define __MALLOC_M(p,s)   malloc(s)
    #define __FREE(p)         free(p)
    #define __REALLOC(p,s)    realloc(p,s)
    #define __MEMALIGN(a,s)   memalign(a,s)
#else
    #define stdLwrrentMemspace (stdGetThreadContext()->lwrrentMemspace)

    #define __MALLOC(s)       memspMalloc(stdLwrrentMemspace,(SizeT)(s))
    #define __MALLOC_M(p,s)   memspMalloc(p,                 (SizeT)(s))
    #define __FREE(p)         memspFree(p)
    #define __REALLOC(p,s)    memspRealloc(p,(SizeT)(s))
    #define __MEMALIGN(a,s)   memspMemalign(stdLwrrentMemspace,a,(SizeT)(s))
    
    #ifdef STD_OS_win32
       void STD_CDECL allocateHeap();
       void STD_CDECL deallocateHeap();
       HANDLE STD_CDECL resetHeap(HANDLE newHeapHandle);
    #endif
 
#endif

void STD_CDECL stdOutOfMemory(void);
Bool STD_CDECL stdIsOOMReported(void);
String STD_CDECL stdGetOOMErrorMessage();

static inline Pointer STD_CDECL stdMALLOC(size_t size)
{
    Pointer  result= (Pointer)__MALLOC(size);
    if (!result) { stdOutOfMemory(); }
    return result;
}

static inline Pointer STD_CDECL stdMALLOC_M( stdMemSpace_t space, size_t size )
{
    Pointer  result= (Pointer)__MALLOC_M(space,size);
    if (!result) { stdOutOfMemory(); }
    return result;
}

static inline Pointer STD_CDECL stdREALLOC(Pointer p, size_t size)
{
    Pointer  result= (Pointer)__REALLOC(p, size);
    if (!result) { stdOutOfMemory(); }
    return result;
}

#if defined(SUPPORT_MEMALIGN)
static inline Pointer STD_CDECL stdMEMALIGN(uInt align, size_t size)
{
    Pointer  result= (Pointer)__MEMALIGN(align, size);
    if (!result) { stdOutOfMemory(); }
    return result;
}
#endif

#define stdFREE(p) __FREE(p)

#define stdFREE2(p1,p2)          { stdFREE(p1); stdFREE(p2); }
#define stdFREE3(p1,p2,p3)       { stdFREE(p1); stdFREE(p2); stdFREE(p3); }
#define stdFREE4(p1,p2,p3,p4)    { stdFREE(p1); stdFREE(p2); stdFREE(p3); stdFREE(p4); }
#define stdFREE5(p1,p2,p3,p4,p5) { stdFREE(p1); stdFREE(p2); stdFREE(p3); stdFREE(p4); stdFREE(p5); }

/*------------------------------- More Macros ------------------------------*/

#if defined (STD_OS_win32)
       int STD_CDECL strcasecmp (const char *s1, const char *s2);
       int STD_CDECL strncasecmp(const char *s1, const char *s2, size_t n);
#endif


#if defined (STD_OS_win32)
    #define stdNEW(x)              { *((Pointer*)&(x)) = stdMALLOC   (          sizeof(*(x))); stdMEMCLEAR  ((x)                  ); }
    #define stdNEW_N(x,n)          { *((Pointer*)&(x)) = stdMALLOC   (      (n)*sizeof(*(x))); stdMEMCLEAR_S((x), (n)*sizeof(*(x))); }

    #define stdNEW_M(m,x)          { *((Pointer*)&(x)) = stdMALLOC_M ( (m),     sizeof(*(x))); stdMEMCLEAR  ((x)                  ); }
    #define stdNEW_MN(m,x,n)       { *((Pointer*)&(x)) = stdMALLOC_M ( (m), (n)*sizeof(*(x))); stdMEMCLEAR_S((x), (n)*sizeof(*(x))); }
#else
    // Gnu compiler assumed here:
    #define stdNEW(x)              { (x) = (__typeof(x)) stdMALLOC   (          sizeof(*(x))); stdMEMCLEAR  ((x)                  ); }
    #define stdNEW_N(x,n)          { (x) = (__typeof(x)) stdMALLOC   (      (n)*sizeof(*(x))); stdMEMCLEAR_S((x), (n)*sizeof(*(x))); }
    
    #define stdNEW_M(m,x)          { (x) = (__typeof(x)) stdMALLOC_M ( (m),     sizeof(*(x))); stdMEMCLEAR  ((x)                  ); }
    #define stdNEW_MN(m,x,n)       { (x) = (__typeof(x)) stdMALLOC_M ( (m), (n)*sizeof(*(x))); stdMEMCLEAR_S((x), (n)*sizeof(*(x))); }
#endif

#define stdCOPY(p)                 stdCOPY_S(p,sizeof(*(p)))
#define stdCOPYSTRING(p)           strcpy((String) stdMALLOC(strlen(p)+1), p)
#define stdCONCATSTRING(p, q)      strcat( strcpy((String) stdMALLOC(strlen(p)+strlen(q)+1), p), q)
#define stdCOPYSTRING_NWLN(p)      strcat( strcpy((String) stdMALLOC(strlen(p)+2), p), "\n")
#define stdCONCATSTRING_NWLN(p, q) strcat( strcat( strcpy((String) stdMALLOC(strlen(p)+strlen(q)+2), p), q), "\n")
#define stdCOPY_S(p,s)             ((Pointer)stdMEMCOPY_S( stdMALLOC(s), p, s))
#define stdCOPY_N(p,n)             ((Pointer)stdMEMCOPY_S( stdMALLOC((n)*sizeof(*(p))), p, (n)*sizeof(*(p))))

#define stdCOPY_M(m,p)             stdCOPY_MS(m,p,sizeof(*(p)))
#define stdCOPYSTRING_M(m,p)       strcpy((String) stdMALLOC_M(m,strlen(p)+1), p)
#define stdCONCATSTRING_M(m,p, q)  strcat( strcpy((String) stdMALLOC_M(m,strlen(p)+strlen(q)+1), p), q)
#define stdCOPY_MS(m,p,s)          ((Pointer)stdMEMCOPY_S( stdMALLOC_M(m,s), p, s))
#define stdCOPY_MN(m,p,n)          ((Pointer)stdMEMCOPY_S( stdMALLOC_M(m,(n)*sizeof(*(p))), p, (n)*sizeof(*(p))))




#define stdIS_SUFFIX(suffix,s)  (strlen(suffix)<=strlen(s) && strcmp(suffix,(s)+strlen(s)-strlen(suffix))==0)
#define stdIS_PREFIX(prefix,s)  (stdStringIsPrefix(prefix,s)!=Nil)
#define stdEQSTRING(s1,s2)      (strcmp((s1),(s2))==0)
#define stdLEQSTRING(s1,s2)     (strcmp((s1),(s2))<=0)
#define stdGEQSTRING(s1,s2)     (strcmp((s1),(s2))>=0)
#define stdEQSTRINGN(s1,s2,n)   (strncmp((s1),(s2),(n))==0)
#define stdSUBSTRING(s1,s2)     (strstr((s1),(s2)))

/*
 * stdSTRTOK : Thread safe alternative for strtok
 */
#if defined(STD_OS_win32)
    #define stdSTRTOK(str, delim, savedStrPtr)       strtok_s(str, delim, savedStrPtr)
#else
    #define stdSTRTOK(str, delim, savedStrPtr)       strtok_r(str, delim, savedStrPtr)
#endif

#define stdIS_PREFIX_C(p,s)     (strncasecmp(p,s,strlen(p))==0)
#define stdEQSTRING_C(s1,s2)    (strcasecmp((s1),(s2))==0)
#define stdLEQSTRING_C(s1,s2)   (strcasecmp((s1),(s2))<=0)
#define stdEQSTRINGN_C(s1,s2,n) (strncasecmp((s1),(s2),(n))==0)

#define stdISEMPTYSTRING(s)   (!(s)[0])

#define stdMEMCOPY(t,f)       memcpy((Pointer)(t),(Pointer)(f),sizeof(*(t)))
#define stdMEMSET(t,c)        memset((Pointer)(t),c,sizeof(*(t)))
#define stdMEMCLEAR(t)        memset((Pointer)(t),0,sizeof(*(t)))

#define stdMEMCOPY_S(t,f,s)   memcpy((Pointer)(t),(Pointer)(f),(size_t)(s))
#define stdMEMSET_S(t,c,s)    memset((Pointer)(t),c,(size_t)(s))
#define stdMEMCLEAR_S(t,s)    memset((Pointer)(t),0,(size_t)(s))

#define stdMEMCOPY_N(t,f,n)   memcpy((Pointer)(t),(Pointer)(f),(n)*sizeof(*(t)))
#define stdMEMSET_N(t,c,n)    memset((Pointer)(t),c,(n)*sizeof(*(t)))
#define stdMEMCLEAR_N(t,n)    memset((Pointer)(t),0,(n)*sizeof(*(t)))

#define stdCALLOC(n,s)        ((Pointer)stdMEMCLEAR_S(stdMALLOC((n)*(s)),((n)*(s))))

#define stdOFFSETOF(p,f)      ( _A_(&((p)->f)) - _A_(p) )
#define stdMULTIPLEOF(n,d)    ( !((n)%(d)) )
#define stdISALIGNED(a,n)     stdMULTIPLEOF((Address)(a), n)

#define stdIMPLIES(a,b)       ( (!(a)) || (b) )
#define stdEQUIV(a,b)         ( ((a)==False) == ((b)==False) )

#define stdMAX(a,b)           ( (a)>(b) ? (a) : (b) )
#define stdMIN(a,b)           ( (a)<(b) ? (a) : (b) )
#define stdABS(a)             ( ((Int)(a))<0   ? -(a) : +(a) )

#define stdODD(a)             ((a)&1)
#define stdEVEN(a)            (!stdODD(a))

#define stdISPOW2(a)          ( ((a)&-(a)) == (a) )
#define stdISBOOL(a)          ( !!(a) == (a) )

#define stdNELTS(arr)         ( sizeof(arr) / sizeof(*(arr)) )


#define stdROUNDDOWN32(a,b)     ((_U32_(a)/_U32_(b))*(_U32_(b)))
#define stdROUNDUP32(a,b)       stdROUNDDOWN32(_U32_(a)+_U32_(b)-1,_U32_(b))
#define stdROUND32(a,b)         stdROUNDDOWN32(_U32_(a)+_U32_(b)/2,_U32_(b))

#define stdROUNDDOWN64(a,b)     ((_U64_(a)/_U64_(b))*(_U64_(b)))
#define stdROUNDUP64(a,b)       stdROUNDDOWN64(_U64_(a)+_U64_(b)-1,_U64_(b))
#define stdROUND64(a,b)         stdROUNDDOWN64(_U64_(a)+_U64_(b)/2,_U64_(b))

#ifdef STD_64_BIT_ARCH
    #define stdROUNDDOWN(a,b)  stdROUNDDOWN64(a,b)
    #define stdROUNDUP(a,b)    stdROUNDUP64(a,b)
    #define stdROUND(a,b)      stdROUND64(a,b)
#else
    #define stdROUNDDOWN(a,b)  stdROUNDDOWN32(a,b)
    #define stdROUNDUP(a,b)    stdROUNDUP32(a,b)
    #define stdROUND(a,b)      stdROUND32(a,b)
#endif


#define stdENDIANSWAP16(x)    ( _U16_((_U16_(x) >> 8)) \
                              | _U16_((_U16_(x) << 8)) \
                              )

#define stdENDIANSWAP32(x)    ( _U32_( stdENDIANSWAP16(_U32_(x) >>16 )       ) \
                              | _U32_( stdENDIANSWAP16(_U32_(x)      ) << 16 ) \
                              )

#define stdENDIANSWAP64(x)    ( _U64_( stdENDIANSWAP32(_U64_(x) >> 32 )       ) \
                              | _U64_( _U64_(stdENDIANSWAP32(_U64_(x))) << 32 ) \
                              )

#define stdPLURAL(n)          ( (n)==1 ? "" : "s" ) 


#define stdENDIANSWAPA(x)     ((Address)stdENDIANSWAP32(x))        
#define stdENDIANSWAPP(x)     ((Pointer)stdENDIANSWAP32(x))        


#ifdef __LITTLE_ENDIAN_OS__

    #define stdBIGENDIAN16(x)       stdENDIANSWAP16(x)
    #define stdBIGENDIAN32(x)       stdENDIANSWAP32(x)
    #define stdBIGENDIAN64(x)       stdENDIANSWAP64(x)
    #define stdBIGENDIANA(x)        stdENDIANSWAPA(x)
    #define stdBIGENDIANP(x)        stdENDIANSWAPP(x)
    
    #define stdLITTLEENDIAN16(x)   _U16_(x)
    #define stdLITTLEENDIAN32(x)   _U32_(x)
    #define stdLITTLEENDIAN64(x)   _U64_(x)
    #define stdLITTLEENDIANA(x)    ((Address)(x))
    #define stdLITTLEENDIANP(x)    ((Pointer)(x))

#else

    #define stdLITTLEENDIAN16(x)    stdENDIANSWAP16(x)
    #define stdLITTLEENDIAN32(x)    stdENDIANSWAP32(x)
    #define stdLITTLEENDIAN64(x)    stdENDIANSWAP64(x)
    #define stdLITTLEENDIANA(x)     stdENDIANSWAPA(x)
    #define stdLITTLEENDIANP(x)     stdENDIANSWAPP(x)
    
    #define stdBIGENDIAN16(x)      _U16_(x)
    #define stdBIGENDIAN32(x)      _U32_(x)
    #define stdBIGENDIAN64(x)      _U64_(x)
    #define stdBIGENDIANA(x)       ((Address)(x))
    #define stdBIGENDIANP(x)       ((Pointer)(x))

#endif


#define stdSIGN_BIT32 0x80000000
#define stdSIGN_BIT64 U64_CONST(0x8000000000000000)


#define stdBITSPERNIBBLE  4
#define stdBITSPERBYTE    8
#define stdBITSIZEOF(T)     (stdBITSPERBYTE*sizeof(T)) // Type size in bits

#define stdBITSPERADDRESS stdBITSIZEOF(Pointer)
#define stdBITSPERINT     stdBITSIZEOF(uInt32)
#define stdBITSPERINT64   stdBITSIZEOF(uInt64)


#define stdCHECK(c,x)             if (!(c)) { msgReport x;        } else
#define stdCHECK_WITH_POS(c,x)    if (!(c)) { msgReportWithPos x; } else
// OPTIX_HAND_EDIT added these macros to avoid doing work unless the check fails
#define stdCHECK_WITH_POS_LAZY(c,code,x)    if (!(c)) { code; msgReportWithPos x; } else


#define stdSWAP(a,b,__type) \
     {          \
        __type ___h;        \
        ___h=a; a=b; b=___h;    \
     }


cString STD_CDECL stdIDENT(void);

#if (defined(STD_OS_win32) || defined(STD_OS_MinGW))
    #define stdFMT_LLD    "I64d"
    #define stdFMT_LLX    "I64x"
    #define stdFMT_LLU    "I64u"
#else
    #define stdFMT_LLD    "lld"
    #define stdFMT_LLX    "llx"
    #define stdFMT_LLU    "llu"
#endif

#ifdef STD_64_BIT_ARCH
    #define stdFMT_ADDR   stdFMT_LLX
#else
    #define stdFMT_ADDR   "x"
#endif


#define stdTRACE() \
    stdSYSLOG("## " __FILE__ ", line %d\n", __LINE__)


#define stdPRINT(x)     stdSYSLOG("# %s \t= %d\n",                 #x,x)
#define stdPRHEX(x)     stdSYSLOG("# %s \t= 0x%08x\n",             #x,x)
#define stdPRSTR(x)     stdSYSLOG("# %s \t= %s\n",                 #x,x)
#define stdPRINTLL(x)   stdSYSLOG("# %s \t= %" stdFMT_LLD "\n",      #x,x)
#define stdPRHEXLL(x)   stdSYSLOG("# %s \t= 0x%016" stdFMT_LLX "\n", #x,x)
#define stdPRPTR(x)     stdSYSLOG("# %s \t= %p\n",                 #x,x)

/*--------------------------- System Logging Output ------------------------*/

stdNORETURN ( void STD_CDECL stdABORT  ( void )       );
stdNORETURN ( void STD_CDECL stdEXIT   ( Int status ) );



stdLogLineFunc STD_CDECL stdGetLogLine();
stdLogLineFunc STD_CDECL stdSetLogLine( stdLogLineFunc f );
void           STD_CDECL stdRestoreLogLine (void);
void*          STD_CDECL stdSetLogLineData(void*);
void           STD_CDECL stdRestoreLogLineData (void);
void*          STD_CDECL stdGetLogLineData();
void           STD_CDECL stdSetLogFile( FILE *f );

void STD_CDECL stdSYSLOG  ( cString format, ... );
void STD_CDECL stdVSYSLOG ( cString format, va_list arg );

void STD_CDECL stdFSYSLOG ( cString format, ... );
void STD_CDECL stdVFSYSLOG( cString format, va_list arg );

void STD_CDECL stdASSERTLOGSETPOS( cString condition, cString fileName, uInt lineNo );
stdNORETURN ( void STD_CDECL stdASSERTLOGFAIL  ( cString format, ... ) );


#ifdef STD_OS_win32
    void STD_CDECL _assert (const char *,const char *,unsigned);
#endif

void STD_CDECL stdASSERTReport(const char *fileName, int lineNum, const char *p);
const char * STD_CDECL stdASSERTAssembleString(const char *p, ...);

extern Bool stdEnableAsserts;
void STD_CDECL stdSetAssertState(Bool);

#if defined(NDEBUG) || defined(STD_OS_CygWin)
    #ifdef COMPILER_AUTO_SAFETY_BUILD
        #if defined(GPU_DRIVER_SASSLIB)
            #define stdASSERT(c,p)                                                 \
            if (stdEnableAsserts) {                                                \
                if (!(c)) {                                                        \
                    stdASSERTReport(__FILE__, __LINE__, "Compiler internal error");\
                }                                                                  \
            }
        #else
            #define stdASSERT(c,p)                                                   \
            if (stdEnableAsserts) {                                                  \
                if (!(c)) {                                                          \
                    stdASSERTLOGSETPOS("Compiler internal error",__FILE__, __LINE__);\
                    stdASSERTLOGFAIL  ("Compiler internal error");                   \
                }                                                                    \
            }
        #endif
    #else
        #define stdASSERT(c,p)
    #endif // COMPILER_AUTO_SAFETY_BUILD
#elif defined(GPU_DRIVER_SASSLIB)
    #define stdASSERT(c,p)                                                  \
        if (!(c)) {                                                         \
            stdASSERTReport(__FILE__, __LINE__, stdASSERTAssembleString p); \
            }                                          
#else
    #define stdASSERT(c,p)                             \
            if (!(c)) {                                \
            stdASSERTLOGSETPOS(#c,__FILE__, __LINE__); \
            stdASSERTLOGFAIL   p;                      \
            }
#endif

/*----------------------- Exit, Abort, Error Handling ----------------------*/

typedef void (STD_CDECL *stdABORTFunc          )( void       );
typedef void (STD_CDECL *stdEXITFunc           )( Int status );
typedef void (STD_CDECL *stdASSERTLOGSETPOSFunc)( cString condition, cString fileName, uInt lineNo );
typedef void (STD_CDECL *stdASSERTLOGFAILFunc  )( cString format,  va_list arg );

void STD_CDECL stdGetTerminators   ( stdABORTFunc          *abort,           stdEXITFunc         *exit          );
void STD_CDECL stdSetTerminators   ( stdABORTFunc           abort,           stdEXITFunc          exit          );
void STD_CDECL stdSetAssertHandlers( stdASSERTLOGSETPOSFunc assertLogSetPos, stdASSERTLOGFAILFunc assertLogFail );


#define stdEXIT_ERROR()   stdEXIT(EXIT_FAILURE)

/*------------------------ MSVS Stdlib Portability -------------------------*/

#ifndef STD_OS_win32
    #define closesocket  close
#else
    #define strtoull    _strtoui64 
    #define strtoll     _strtoi64 
  #if _MSC_VER >= 1400
    #define stat        _stat64 
  #else
    #define stat        _stati64
  #endif
    #define fstat       _fstat64 
    #define vsnprintf   _vsnprintf 
    #define fileno      _fileno
    #define isatty      _isatty
    #define ftell       _ftelli64
    #define fseek       _fseeki64
#endif

/*--------------------------- Shell Command Names---------------------------*/

#ifdef STD_OS_win32
  #define stdRM      "erase"
  #define stdCP      "copy"
  #define stdCAT     "type"
  #define stdCREATE  "echo . >"
#else
  #define stdRM      "rm"
  #define stdCP      "cp"
  #define stdCAT     "cat"
  #define stdCREATE  "touch"
#endif
    
/*----------------------------- String Parsing -----------------------------*/

#ifdef STD_OS_win32
  #define stdATOUI64(s) _atoi64(s)                                      
#else
  #define stdATOUI64(s) atoll(s)                                      
#endif

/*---------------------- Windows Calling Colwentions -----------------------*/

/*
 * This source base tries to avoid sprinkling the code with __cdecl,
 * but sometimes there simply is no other way:
 */
#ifdef STD_OS_win32

    #ifdef Windows_STDCALL

       /*
        * The following definitions are needed to avoid build errors
        * in yacc- generated files. For some reason these contain
        * prototype definitions for malloc, free and isatty, but with the
        * missing __cdecl stuff. Unfortunately, these flags do not add up,
        * but generate an clash with the 'real' definitions in stdlib.h
        */

        static inline void* STD_CDECL stdXmalloc( size_t s ) { return  malloc(s); }
        static inline void  STD_CDECL stdXfree  ( void* p  ) {           free(p); }
        static inline int   STD_CDECL stdXisatty( int f    ) { return _isatty(f); }

        #define malloc stdXmalloc
        #define free   stdXfree
        #define isatty stdXisatty
    #endif
#endif

/*------------------------------- Includes -----------------------------------*/

#include "stdMessages.h"

/*---------------------------------- Macros ----------------------------------*/

/*
 * The following macros support variable sized arrays:
 */

#define stdXArray(t,x) \
         t *x; Bool x##isStatic; uInt x##Capacity

#define stdXArrayInit(x) \
        {                           \
            stdNEW_N(x,1);          \
            x##Capacity= 1;         \
            x##isStatic= False;     \
        }

#define stdXArrayInit_B(x,_initialBuffer,_capacity,_isStatic) \
        {                    \
            x = _initialBuffer;      \
            x##Capacity= _capacity;  \
            x##isStatic= _isStatic;  \
        }

#define stdXArrayTerm(x) \
        if (!(x##isStatic)) {       \
            stdFREE(x);             \
        }

#define stdXArrayCheck(x,i) \
        {                                                     \
          uInt _i1= i;                                        \
                                                              \
          if (_i1 >= x##Capacity) {                           \
              uInt _oc_= x##Capacity;                         \
                                                              \
              do {                                            \
                 x##Capacity *= 2;                            \
              } while (_i1 >= x##Capacity);                   \
                                                              \
              if (x##isStatic) {                              \
                  Pointer _y_= x;                             \
                  x= stdMALLOC(sizeof(*x)*(x##Capacity));     \
                  stdMEMCOPY_N(x,_y_,_oc_);                   \
                  (x##isStatic)= False;                       \
              } else {                                        \
                  x = stdREALLOC(x,sizeof(*x)*(x##Capacity)); \
              }                                               \
              stdMEMCLEAR_N(&(x)[_oc_],x##Capacity-_oc_);     \
          }                                                   \
        }

#define stdXArrayAssign(x,i,y) \
        {                                                     \
          uInt _i2= i;                                        \
          stdXArrayCheck(x,_i2);                              \
          (x)[_i2]= (y);                                      \
        }

/*-------------------------------- Functions -------------------------------*/

/*
 * Function        : Check if character is a white space character
 * Parameters      : c        (I) Char to test
 *                   newLine  (I) a newline is considered white space iff 
 *                                 the value of this parameter is True.
 * Function Result : True iff. char is white space.
 *
 */
static inline Bool STD_CDECL stdIsSpace( Char c, Bool newLine )
{
    return (c==' ') || (c=='\t') || (newLine && ((c=='\n') || (c=='\r')));
}


/*
 * Function        : Skips initial comments if any from string
 * Parameters      : str (I) String to skip
 * Note            : This function modifies the original string
 */
void STD_CDECL stdSkipComments(String *str);


/*
 * Function        : Skip initial white space and trunc trailing white space from string
 * Parameters      : str (I) String to shrinkwrap
 * Function Result : Pointer to first nonwhite space character in str
 * Note            : This function modifies the original string
 */
String STD_CDECL stdShrinkWrap( String s );


/*
 * Function        : Base 2 logarithm, rounded UPWARDS 
 *                   to the nearest integer.
 *                   NOTE: LOG(0) will return -1
 * Parameters      : x (I) Unsigned integer.
 * Function Result :
 *
 */
Int STD_CDECL stdLOG2U   ( uInt x );
Int STD_CDECL stdLOG2U_64( uInt x );


/*
 * Function        : Base 2 logarithm, rounded DOWNWARDS 
 *                   to the nearest integer.
 *                   NOTE: LOG(0) will return -1
 * Parameters      : x (I) Unsigned integer.
 * Function Result :
 *
 */
Int STD_CDECL stdLOG2   ( uInt   x );
Int STD_CDECL stdLOG2_64( uInt64 x );


/*
 * Function        : Least common multiple.
 * Parameters      : x (I) Unsigned integer.
 *                   y (I) Unsigned integer.
 * Function Result :
 */
uInt STD_CDECL stdLCM( uInt x, uInt y );


/*
 * Function        : Largest common denominator.
 * Parameters      : x (I) Unsigned integer.
 *                   y (I) Unsigned integer.
 * Function Result :
 */
uInt STD_CDECL stdGCD( uInt x, uInt y );


/*
 * Function        : Determine how bit masks overlap
 * Parameters      : l,r  (I) bits to test
 * Function Result :
 */
stdOverlapKind STD_CDECL stdBitsOverlapsHow32( uInt32 l, uInt32 r );
stdOverlapKind STD_CDECL stdBitsOverlapsHow64( uInt64 l, uInt64 r );


/*
 * Function        : Determine if the specified 32/64 bit integer value can be represented 
 *                   by a signed/unsigned word of the specified amount of representation bits,
 *                   under the condition that this value will further be used in two complement
 *                   arithmetic in the specified amount of computation bits.
 *                   The latter will e.g. allow the value of 16 be represented as a signed, 4 bit
 *                   quantity, provided that it is further used in 4 bit arithmetic. In contrast,
 *                   this would not be possible if e.g. 8 bit arithmetic is further used, since 
 *                   these 8 bits then would be interpreted as -16.
 * Parameters      : value         (I)  integer value.                      
 *                   nrofReprBits  (I)  Number of bits to represent value in.
 *                   nrofCompBits  (I)  Number of bits for further computations of represented value,
 *                                      or 0 if unspecified.
 *                   isSigned      (I)  If "value" is signed.               
 *                   isSSigned     (I)  If true, then value 0x80000000      
 *                                      is considered not in range.         
 *                                      This is used for checking whether   
 *                                      the number can be represented using 
 *                                      a sign bit plus absolute value      
 * Function Result : True iff. the value can be represented as described
 */
Bool STD_CDECL stdInRange32( Int32 value, uInt nrofReprBits, uInt nrofCompBits, Bool isSigned, Bool isSSigned );
Bool STD_CDECL stdInRange64( Int64 value, uInt nrofReprBits, uInt nrofCompBits, Bool isSigned, Bool isSSigned );


/*
 * Function : Get specified number of bits from a given location. 
 *
 * Parameters: startInCode      (I) Start location in code to start getting bits.
 *             length           (I) Number of bits.
 *             code             (I) Array of unsigned integers, from which bits 
 *                                  will be picked.
 * Function Result : Returned bits.
 */
uInt64 STD_CDECL stdGetBits( uInt startInCode, uInt length, uInt64 *code);


/*
 * Function : Put specified number of bits (value) at a given location.
 *
 * Parameters: value            (I) Value to be assigned for bits.
 *             startInCode      (I) Start location in code to start getting bits.
 *             length           (I) Number of bits.
 *             code             (O) Array of unsigned integers, from which bits 
 *                                  will be picked.
 * Function Result : Modified bits.
 */
void STD_CDECL stdPutBits( uInt64 value, uInt startInCode, uInt length, uInt64 *code);


/*
 * Function        :  POSIX.1-2001 Pseudo random number generator.
 * Parameters      : state (IO) Seed and history of generator
 * Function Result : new pseudo random number
 */
static inline uInt32 STD_CDECL stdRandom( uInt32 *state ) {
    *state = *state * 1103515245 + 12345;
    return((uInt32)(*state/65536) % 32768);
}


static inline uInt32 STD_CDECL stdROUNDUP_RESIDUE32( uInt32 value, uInt32 alignment, uInt32 residue )
{
    if ( (value % alignment) == residue ) { return value;                                 }
                                     else { return stdROUNDUP32(value,alignment)+residue; }
}

static inline uInt64 STD_CDECL stdROUNDUP_RESIDUE64( uInt64 value, uInt64 alignment, uInt64 residue )
{
    if ( (value % alignment) == residue ) { return value;                                 }
                                     else { return stdROUNDUP64(value,alignment)+residue; }
}

/*
 * Do not use ctype.h/toupper here, in order to avoid
 * compatibility problems with older versions of glibc:
 */
static inline Bool STD_CDECL stdIsWhiteSpace( Char c )
{ return (c==' ' || c=='\n' || c=='\t'); }

static inline Bool STD_CDECL stdToUpperCase( Char c )
{
    if ('a' <= c && c <= 'z') { return c - 'a' + 'A'; }
                         else { return c;             }
}

static inline Bool STD_CDECL stdToLowerCase( Char c )
{
    if ('A' <= c && c <= 'Z') { return c - 'A' + 'a'; }
                         else { return c;             }
}



/*
 * Function        : Translate non C identifier characters in string to canonical characters
 * Parameters      : from (I) Source string buffer
 *                   to   (I) Destination string buffer.
 *                            May be identical to 'from'
 */
void STD_CDECL stdTranslateNonIdChars( Char *from, Char *to);



void    STD_CDECL stdFreeFun    ( Pointer p );
Pointer STD_CDECL stdMallocFun  ( size_t size );
Pointer STD_CDECL stdMemAlignFun( uInt align, size_t size );
Pointer STD_CDECL stdReallocFun ( Pointer p,  size_t size );

void STD_CDECL stdProcessCleanupHandlers(void);
void STD_CDECL stdSetCleanupHandler( stdDataFun cleanup, Pointer data );


/*
 * For debugging
 */
void STD_CDECL STOP( Pointer i );

void STD_CDECL stdPrintInt    ( uInt    i );
void STD_CDECL stdPrintInt64  ( uInt64  i );
void STD_CDECL stdPrintuInt   ( uInt    i );
void STD_CDECL stdPrintuInt64 ( uInt64  i );
void STD_CDECL stdPrintHex    ( uInt    i );
void STD_CDECL stdPrintHex64  ( uInt64  i );
void STD_CDECL stdPrintPInt64 ( uInt64 *i );
void STD_CDECL stdPrintPuInt64( uInt64 *i );
void STD_CDECL stdPrintPHex64 ( uInt64 *i );
void STD_CDECL stdPrintString ( cString  s );


/*
 * Function        : Return index of lowest order bit in bit mask
 * Parameters      : mask    (I) bit mask to inspect
 * Function Result : Requested bit index
 * Precondition    : mask != 0
 */
 #if !defined(STD_OS_win32) && (defined(STD_ARCH_x86_64) || defined(STD_ARCH_i386) || defined(STD_ARCH_i686))

     #if defined(STD_OS_win32)

        #define ___STD_BITOPS_32
        static inline uInt STD_CDECL stdFirstBit32(uInt32 mask)
        {
            __asm bsf eax,  mask
            __asm mov mask, eax
            return mask;
        }

        #if defined(STD_ARCH_x86_64_DISABLED)
        #define ___STD_BITOPS_64
        static inline uInt STD_CDECL stdFirstBit64(uInt64 mask)
        {
            __asm bsfq eax,  mask
            __asm movq mask, eax
            return mask;
        }
        #endif

     #else
        #define ___STD_BITOPS_32
        static inline uInt STD_CDECL stdFirstBit32(uInt32 mask)
        {
            __asm__("bsf %1,%0"
                  :"=r" (mask)
                  :"rm" (mask));
            return mask;
        }

        #if defined(STD_ARCH_x86_64)
        #define ___STD_BITOPS_64
        static inline uInt STD_CDECL stdFirstBit64(uInt64 mask)
        {
            __asm__("bsfq %q1,%q0"
                  :"=r" (mask)
                  :"rm" (mask));
            return mask;
        }
        #endif
    #endif
#endif


#if !defined(___STD_BITOPS_32)
    static inline uInt STD_CDECL stdFirstBit32(uInt32 mask)
    {
       uInt i;

       for (i=0; i<32; i++) {
           if (mask&1) { return i; }
           mask >>= 1;
       }

       stdASSERT(False,("zero argument to stdFirstBit32"));
       return ((uInt)-1);
    }
#endif


#if !defined(___STD_BITOPS_64)
    static inline uInt STD_CDECL stdFirstBit64(uInt64 mask)
    {
        if ((uInt32)mask) {
            return stdFirstBit32((uInt32)mask);
        } else {
            return stdFirstBit32((uInt32)(mask >> 32)) + 32;
        }
    }
#endif


#if !defined(ENABLE_SSE4_SUPPORT)
    static inline uInt STD_CDECL stdNrofBits32( uInt32 word )
    {
        word = word - ((word >> 1) & 0x55555555);
        word = (word & 0x33333333) + ((word >> 2) & 0x33333333);
        word = (word + (word >> 4)) & 0x0F0F0F0F;
        return (word * 0x01010101) >> 24;
    }

    static inline uInt STD_CDECL stdNrofBits64( uInt64 word )
    {
        word = word - ((word >> 1) & 0x5555555555555555ULL);
        word = (word & 0x3333333333333333ULL) + ((word >> 2) & 0x3333333333333333ULL);
        word = (word + (word >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
        return (uInt)((word * 0x0101010101010101ULL) >> 56);
    }
#else
    static inline uInt STD_CDECL stdNrofBits32( uInt32 word )
    {
        __asm__("popcnt %1,%0"
              :"=r" (word)
              :"rm" (word));
        return word;
    }
    static inline uInt STD_CDECL stdNrofBits64( uInt64 word )
    {
        __asm__("popcntq %1,%0"
              :"=r" (word)
              :"rm" (word));
        return word;
    }
#endif /*ENABLE_SSE4_SUPPORT*/



/*
 * Atomically increment the value pointed to by mem by value 
 * and return the previous value.
 */ 
static inline uInt STD_CDECL stdAtomicFetchAndAdd( uInt *mem, uInt value )
{
  #if defined STD_OS_win32 
    return InterlockedExchangeAdd((volatile LONG*)mem,value);
  #else
    return __sync_fetch_and_add(mem,value);
  #endif
}

#if defined STD_ARCH_x86_64
/*
 * Atomically increment the value pointed to by mem by value 
 * and return the previous value.
 */
#if defined STD_OS_win32
#pragma intrinsic (_InterlockedExchangeAdd64)
#endif

static inline uInt64 STD_CDECL stdAtomicFetchAndAdd64( uInt64 *mem, uInt64 value )
{
  #if defined STD_OS_win32
    // use compiler intrinsic _InterlockedExchangeAdd64 than function
    // InterlockedExchangeAdd64, as former is also supported on older
    // Windows versions like WinXP
    return _InterlockedExchangeAdd64((volatile LONGLONG*)mem,value);
  #else
    return __sync_fetch_and_add(mem,value);
  #endif
}
#endif

/* 
 * Swap the pointer pointed to by mem with the specified new value. 
 * Return the old pointer pointed to by mem. 
 */
static inline Pointer STD_CDECL stdAtomicSwap( Pointer *mem, Pointer newVal )
{
  #if defined STD_OS_win32 
      return InterlockedExchangePointer(mem,newVal);
  #else
      return __sync_fetch_and_and((uInt**)mem, (uInt*)newVal);
  #endif
}

/* Check if the value pointed to by mem is equal to oldVal. If it is, atomically write
 * newVal in the location pointed to by mem. Return the old value of the memory location.
 */
static inline Pointer STD_CDECL stdAtomicCompareAndSwap( Pointer *mem, Pointer oldVal, Pointer newVal )
{
  #if defined STD_OS_win32 
    return InterlockedCompareExchangePointer(mem,newVal,oldVal);
  #else
    return __sync_val_compare_and_swap((uInt**)mem,(uInt*)oldVal,(uInt*)newVal);
  #endif
}

/* Check if the value pointed to by mem is equal to oldVal. If it is, atomically write
 * newVal in the location pointed to by mem. Return if the write oclwred.
 * This is not a general function, because not implemented on Windows
 */
static inline Bool STD_CDECL __stdAtomicCompareAndSwapInt( uInt *mem, uInt oldVal, uInt newVal )
{
  #if defined STD_OS_win32 
    stdASSERT(False,("Implement atomic functions for Windows"));
  #else
    return __sync_bool_compare_and_swap(mem, oldVal, newVal);
  #endif
}


#define stdAtomicFetch(x) *(x)
#define stdAtomicSet(x,y) *(x)=y

/* 
 * This is not guaranteed on 32 bit platforms,
 * but these seem to be dying out:
 */
#define stdAtomicFetch64(x) *(x)
#define stdAtomicSet64(x,y) *(x)=y
       
#define stdAtomicFetchP(x) *(x)
#define stdAtomicSetP(x,y) *(x)=y
       

       
/*
 * Stream hash function; see the following website on the modulo value.
 * Seems little difference, but 0xffff is faster.
 *
 *      http://stackoverflow.com/questions/927277/why-modulo-65521-in-adler-32-checksum-algorithm
 */
static inline uInt32  STD_CDECL stdStreamHash( uInt32 seed, uInt16 val )
{
    uInt multiplier = 31;
  //uInt modulo     = 65521;
    uInt modulo     = 0xffff;
    
    return (multiplier * seed + val) % modulo;
}



/*
 * Create a new directory with default permissions.
 * Return True iff. the directory could be created.
 */
static inline Bool STD_CDECL stdMkdir( cString path )
{
  #if defined STD_OS_win32
    LPSELWRITY_ATTRIBUTES lpSelwrityAttributes = NULL;
    return (0 != CreateDirectory((LPCTSTR) path, lpSelwrityAttributes));
  #else
    return (0 == mkdir(path, 0700));
  #endif
}

#ifdef __cplusplus
}
#endif

#endif

