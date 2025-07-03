/*
 * Copyright (c) 1999 - 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __OGTEST_H
#define __OGTEST_H

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// For debug Windows builds, use <crtdbg.h> and _CRTDBG_MAP_ALLOC to redirect
// standard memory allocation functions to debug versions taking __FILE__ and
// __LINE_.  Also use a macro to point new at an overload taking __FILE__ and
// __LINE__ as well.
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#define new            new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

#endif // #ifdef _WIN32

#include "ossymbols.h"

#include <assert.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>

#include <stddef.h>

// Clients using older MSVC versions without support for stdint.h need
// to supply their own versions.
#if (defined(_MSC_VER) && (_MSC_VER >= 1600)) || defined(__GNUC__)
#include <stdint.h>
#endif

// Definitions of old LWN types that are no longer in the official API, but
// are still in the driver code base.
typedef uint32_t LWNbitfield;
typedef int LWNint;
typedef int LWNsizei;
typedef uint32_t LWNuint;
typedef float LWNfloat;
typedef uintptr_t LWNuintptr;
typedef intptr_t LWNsizeiptr;
typedef int64_t LWNint64;
typedef uint64_t LWNuint64;
typedef const char * LWNstring;

// lwntest has some utility code originally written for OpenGL that still uses
// OpenGL #defines.  Rather than including OpenGL headers or rewriting all
// this code to be non-shareable with OpenGL, we just define some dummy types
// and #defines.
typedef unsigned int    GLenum;
typedef unsigned int    GLuint;
typedef unsigned char   GLboolean;
typedef int             GLint;

#define GL_TRUE                     1
#define GL_FALSE                    0
#define GL_ILWALID_ENUM             0x0500

// Primitive types for geometry shaders.
#define GL_POINTS                   0x0000
#define GL_LINES                    0x0001
#define GL_LINE_STRIP               0x0003
#define GL_TRIANGLES                0x0004
#define GL_TRIANGLE_STRIP           0x0005
#define GL_QUADS                    0x0007
#define GL_LINES_ADJACENCY          0x000A
#define GL_TRIANGLES_ADJACENCY      0x000C
#define GL_PATCHES                  0x000E

// Tessellation shader parameters.
#define GL_CW                       0x0900
#define GL_CCW                      0x0901
#define GL_ISOLINES                 0x8E7A
#define GL_EQUAL                    0x0202
#define GL_FRACTIONAL_ODD           0x8E7B
#define GL_FRACTIONAL_EVEN          0x8E7C


#include <stdint.h>
#define LW_STDINT_INCLUDED

#include "lwctassert.h"

#define lwogMalloc(x,y,z) malloc(x)
#define lwogFree(x,y,z) free(x)
#define __LWOG_FREE free
#define __LWOG_MALLOC malloc
#define __GL_ARRAYSIZE(x)   (sizeof(x)/sizeof(*(x)))

extern int lwrrentWindowWidth, lwrrentWindowHeight;

#if defined(_MSC_VER)
#pragma warning (disable:4305)
#pragma warning (disable:4244)
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum _lwogExitStatus{
    EXIT_STATUS_NORMAL=0,
    EXIT_STATUS_NOT_NORMAL,       // Runtime error - malloc failures, missing files, etc.
    EXIT_STATUS_USER_ERROR,       // Commandline argument errors 
    EXIT_STATUS_USER_QUIT,        // User-requested exit (such as by hitting escape)
    EXIT_STATUS_TRAPPED_EXIT,     // Window close (by pressing the "X" button), or rogue exit()s
    EXIT_STATUS_FORCE_NO_CLEANUP, // Exit with minimal cleanup so lwogtests can test unclean exits
    EXIT_STATUS_NORMAL_NO_RESULTS,// Exit normally, but don't print any results summary (somewhat magic)
    EXIT_STATUS_EXCEPTION,        // Runtime error - unhandled exception caught (with -catch cmdline argument)
} lwogExitStatus;

extern void Terminate(lwogExitStatus exitStatus);


#ifdef __cplusplus
#define CPP_PREFIX extern "C"
#else
#define CPP_PREFIX
#endif

#define OGTEST_GetDescriptionProto(a)   CPP_PREFIX void ogtest_##a##_GetDescription(char *str, const char *testName)
#define OGTEST_IsSupportedProto(a)      CPP_PREFIX int ogtest_##a##_IsSupported(const char *testName)
#define OGTEST_InitGraphicsProto(a)     CPP_PREFIX void ogtest_##a##_InitGraphics(float lwScale, const char *testName, int path)
#define OGTEST_DoGraphicsProto(a)       CPP_PREFIX void ogtest_##a##_DoGraphics(float lwScale, const char *testName, int path)
#define OGTEST_ExitGraphicsProto(a)     CPP_PREFIX void ogtest_##a##_ExitGraphics(const char *testName, int path)

#define OGTEST_DoBenchmarkProto(a)      CPP_PREFIX void ogtest_##a##_DoBenchmark(const char *testName, int subtest, int cycles, int path)
#define OGTEST_GetBenchmarkProto(a)     CPP_PREFIX void ogtest_##a##_GetBenchmark(BENCHMARKSUMMARY *info, const char *testName, int subtest, int cycles, int path)
#define OGTEST_GetNumSubtestsProto(a)   CPP_PREFIX int ogtest_##a##_GetNumSubtests(const char *testName, int path)


// So every test "knows its name", the OGTEST_IsSupported macro also creates an ogtest_##name##_FileName string.
# define OGTEST_FileName(a)          ogtest_##a##_FileName
# define OGTEST_FileNameComma(a)     OGTEST_FileName(a),
# define OGTEST_FileNameProto(a)     extern const char OGTEST_FileName(a)[]
# define OGTEST_FileNameDefine(a)    const char OGTEST_FileName(a)[] = __FILE__;
# define OGTEST_Stats(a)             {NULL}
# define OGTEST_StatsComma(a)        OGTEST_Stats(a),


#define OGTEST_GetDescription(a) \
    OGTEST_GetDescriptionProto(a) ; \
    OGTEST_GetDescriptionProto(a)
#define OGTEST_IsSupported(a) \
    OGTEST_IsSupportedProto(a); \
    OGTEST_FileNameDefine(a) \
    OGTEST_IsSupportedProto(a)
#define OGTEST_InitGraphics(a) \
    OGTEST_InitGraphicsProto(a) ; \
    OGTEST_InitGraphicsProto(a)
#define OGTEST_DoGraphics(a) \
    OGTEST_DoGraphicsProto(a); \
    OGTEST_DoGraphicsProto(a)
#define OGTEST_ExitGraphics(a) \
    OGTEST_ExitGraphicsProto(a); \
    OGTEST_ExitGraphicsProto(a)

#define OGTEST_DoBenchmark(a) \
    OGTEST_DoBenchmarkProto(a) ; \
    OGTEST_DoBenchmarkProto(a)
#define OGTEST_GetBenchmark(a) \
    OGTEST_GetBenchmarkProto(a); \
    OGTEST_GetBenchmarkProto(a)
#define OGTEST_GetNumSubtests(a) \
    OGTEST_GetNumSubtestsProto(a); \
    OGTEST_GetNumSubtestsProto(a)

#define OGTEST_GetDescriptionBA(b, a) \
    OGTEST_GetDescriptionProto(a);\
    OGTEST_GetDescriptionProto(b) \
    {\
        ogtest_##a##_GetDescription(str, testName);\
    }

#define OGTEST_GetDescription2(a, b) \
    OGTEST_GetDescriptionProto(b);\
    OGTEST_GetDescriptionBA(b, a) \
    OGTEST_GetDescriptionProto(a)

#define OGTEST_GetDescription3(a, b, c) \
    OGTEST_GetDescriptionProto(c);\
    OGTEST_GetDescriptionBA(c, b) \
    OGTEST_GetDescriptionBA(b, a) \
    OGTEST_GetDescriptionProto(a)

#define OGTEST_GetDescription4(a, b, c, d) \
    OGTEST_GetDescriptionProto(d);\
    OGTEST_GetDescriptionBA(d, c) \
    OGTEST_GetDescriptionBA(c, b) \
    OGTEST_GetDescriptionBA(b, a) \
    OGTEST_GetDescriptionProto(a)



#define OGTEST_IsSupportedBA(b, a) \
    OGTEST_IsSupportedProto(a);\
    OGTEST_FileNameDefine(b) \
    OGTEST_IsSupportedProto(b) \
    {\
        return ogtest_##a##_IsSupported(testName);\
    }

#define OGTEST_IsSupported2(a, b) \
    OGTEST_IsSupportedProto(b);\
    OGTEST_IsSupportedBA(b, a) \
    OGTEST_FileNameDefine(a) \
    OGTEST_IsSupportedProto(a)

#define OGTEST_IsSupported3(a, b, c) \
    OGTEST_IsSupportedProto(c);\
    OGTEST_IsSupportedBA(c, b) \
    OGTEST_IsSupportedBA(b, a) \
    OGTEST_FileNameDefine(a) \
    OGTEST_IsSupportedProto(a)

#define OGTEST_IsSupported4(a, b, c, d) \
    OGTEST_IsSupportedProto(d);\
    OGTEST_IsSupportedBA(d, c) \
    OGTEST_IsSupportedBA(c, b) \
    OGTEST_IsSupportedBA(b, a) \
    OGTEST_FileNameDefine(a) \
    OGTEST_IsSupportedProto(a)




#define OGTEST_InitGraphicsBA(b, a) \
    OGTEST_InitGraphicsProto(a);\
    OGTEST_InitGraphicsProto(b) \
    {\
        ogtest_##a##_InitGraphics(lwScale, testName, path);\
    }

#define OGTEST_InitGraphics2(a, b) \
    OGTEST_InitGraphicsProto(b);\
    OGTEST_InitGraphicsBA(b, a) \
    OGTEST_InitGraphicsProto(a)

#define OGTEST_InitGraphics3(a, b, c) \
    OGTEST_InitGraphicsProto(c);\
    OGTEST_InitGraphicsBA(c, b) \
    OGTEST_InitGraphicsBA(b, a) \
    OGTEST_InitGraphicsProto(a)

#define OGTEST_InitGraphics4(a, b, c, d) \
    OGTEST_InitGraphicsProto(d);\
    OGTEST_InitGraphicsBA(d, c) \
    OGTEST_InitGraphicsBA(c, b) \
    OGTEST_InitGraphicsBA(b, a) \
    OGTEST_InitGraphicsProto(a)



#define OGTEST_DoGraphicsBA(b, a) \
    OGTEST_DoGraphicsProto(a);\
    OGTEST_DoGraphicsProto(b) \
    {\
        ogtest_##a##_DoGraphics(lwScale, testName, path);\
    }

#define OGTEST_DoGraphics2(a, b) \
    OGTEST_DoGraphicsProto(b);\
    OGTEST_DoGraphicsBA(b, a) \
    OGTEST_DoGraphicsProto(a)

#define OGTEST_DoGraphics3(a, b, c) \
    OGTEST_DoGraphicsProto(c);\
    OGTEST_DoGraphicsBA(c, b) \
    OGTEST_DoGraphicsBA(b, a) \
    OGTEST_DoGraphicsProto(a)

#define OGTEST_DoGraphics4(a, b, c, d) \
    OGTEST_DoGraphicsProto(d);\
    OGTEST_DoGraphicsBA(d, c) \
    OGTEST_DoGraphicsBA(c, b) \
    OGTEST_DoGraphicsBA(b, a) \
    OGTEST_DoGraphicsProto(a)


#define OGTEST_ExitGraphicsBA(b, a) \
    OGTEST_ExitGraphicsProto(a);\
    OGTEST_ExitGraphicsProto(b) \
    {\
        ogtest_##a##_ExitGraphics(testName, path);\
    }

#define OGTEST_ExitGraphics2(a, b) \
    OGTEST_ExitGraphicsProto(b);\
    OGTEST_ExitGraphicsBA(b, a) \
    OGTEST_ExitGraphicsProto(a)

#define OGTEST_ExitGraphics3(a, b, c) \
    OGTEST_ExitGraphicsProto(c);\
    OGTEST_ExitGraphicsBA(c, b) \
    OGTEST_ExitGraphicsBA(b, a) \
    OGTEST_ExitGraphicsProto(a)

#define OGTEST_ExitGraphics4(a, b, c, d) \
    OGTEST_ExitGraphicsProto(d);\
    OGTEST_ExitGraphicsBA(d, c) \
    OGTEST_ExitGraphicsBA(c, b) \
    OGTEST_ExitGraphicsBA(b, a) \
    OGTEST_ExitGraphicsProto(a)




#define MAX_TEST_DESCRIPTION_LENGTH 8000
#define RETURN_TEST_DESCRIPTION_BUFFER_LENGTH (2*(MAX_TEST_DESCRIPTION_LENGTH))

#define OPROTO_I(a)            \
OGTEST_FileNameProto(a);       \
OGTEST_GetDescriptionProto(a); \
OGTEST_IsSupportedProto(a);    \
OGTEST_InitGraphicsProto(a);   \
OGTEST_DoGraphicsProto(a);     \
OGTEST_ExitGraphicsProto(a)

#define OPROTO_P(a)            \
OGTEST_FileNameProto(a);       \
OGTEST_GetDescriptionProto(a); \
OGTEST_IsSupportedProto(a);    \
OGTEST_InitGraphicsProto(a);   \
OGTEST_ExitGraphicsProto(a);   \
OGTEST_DoBenchmarkProto(a);    \
OGTEST_GetBenchmarkProto(a);   \
OGTEST_GetNumSubtestsProto(a)

#define OPROTO_IP(a)           \
OGTEST_FileNameProto(a);       \
OGTEST_GetDescriptionProto(a); \
OGTEST_IsSupportedProto(a);    \
OGTEST_InitGraphicsProto(a);   \
OGTEST_DoGraphicsProto(a);     \
OGTEST_ExitGraphicsProto(a);   \
OGTEST_DoBenchmarkProto(a);    \
OGTEST_GetBenchmarkProto(a);   \
OGTEST_GetNumSubtestsProto(a)


#ifndef LW_MIN
#define LW_MIN(x, y)            ((x) <= (y) ? (x)  : (y))
#endif
#ifndef LW_MAX
#define LW_MAX(x, y)            ((x) >= (y) ? (x)  : (y))
#endif
#define LW_MAX3(x, y, z)        ((x) >= (y) ? ((x) >= (z) ? (x) : (z)) : \
                                              ((y) >= (z) ? (y) : (z))  )
#define LW_MAX4(x, y, z, w)     ((x) >= (y) ? LW_MAX3((x), (z), (w)) :  \
                                              LW_MAX3((y), (z), (w))   )

#define LW_ISINF(x) ((2*x == x) && (x != 0))
#define LW_ISNAN(x) (x != x)

/* rand.c */
extern unsigned int lwBitRand(int);
extern unsigned int lwGetSeed(void);
extern int     lwIntRand(int,int);
extern float   lwFloatRand(float,float);
extern void    lwSRand(unsigned int n);
extern void    lwSetRandFunc(int (*f)(void));
extern int     lwRandNumber(void);
extern void    lwRandColor(float rgba[4]);

/* Use these two functions only for passing to lwSetRandFunc.
  Use lwRandNumber instead of calling either function below
  directly. */
int     lwRand(void);
int     lwAlternateRand(void);


/* glutwind.c */
#if !defined(LW_WINDOWS)
int stricmp(const char *s1, const char *s2);
#endif
int strincmp(const char *s1, const char *s2, size_t n);


/* float_util.c */
int FloorLog2(float x);

#include "cells.h"
#include "str_util.h"
extern FILE *results;          // Global results file pointer.

#ifdef _WIN32
static __inline uint32_t float_as_uint32(float f)
#else
static inline uint32_t float_as_uint32(float f)
#endif
{
    union { uint32_t u; float f; } x;
    x.f = f;
    return x.u;
}

#ifdef _WIN32
static __inline float uint32_as_float(uint32_t u)
#else
static inline float uint32_as_float(uint32_t u)
#endif
{
    union { uint32_t u; float f; } x;
    x.u = u;
    return x.f;
}


// Operating system-specific entry points, implemented in tegra_main.cpp and
// windows_main.cpp:

extern void lwogSwapBuffers(void);
extern int lwogCheckLWNAPIVersion(int32_t neededMajor, int32_t neededMinor);
extern int lwogCheckLWNGLSLCGpuVersion(uint32_t neededMajor, uint32_t neededMinor);
extern int lwogCheckLWNGLSLCPackageVersion(uint32_t neededVer);
extern void lwogTerminate(lwogExitStatus exitStatus);
extern void lwogHandleWindowEvents(void);
extern void lwogSetWindowTitle(const char *title);
extern uint64_t lwogGetTimerValue(void);
extern uint64_t lwogGetTimerFrequency(void);
extern void lwogSetupGLContext(int enable);
extern void lwogClearWindow(void);
extern void lwogDeleteDonorContext(void);
extern void lwogSetNativeWindow(void);

// Gets the refresh rate (in Hz) of the current monitor - returns 0 upon failure.
extern uint32_t lwogGetRefreshRate(void);

// Facilities for creating and finalizing worker threads.
typedef struct LWOGthread LWOGthread;
extern LWOGthread *lwogThreadCreate(void (*threadFunc)(void*), void *args, size_t stackSize /* 0 = default */);
extern void lwogThreadWait(LWOGthread *thread);
extern void lwogThreadYield(void);
extern void lwogRunOnWorkerThread(void(*threadFunc)(void*), void *args, size_t stackSize /* 0 = default */);

#if defined(LW_HOS)
// Additional HOS support for CPU core selection on worker threads.

extern uint64_t lwogThreadGetAvailableCoreMask(void);
extern int lwogThreadGetLwrrentCoreNumber(void);
extern void lwogThreadSetCoreMask(int idealCore, uint64_t coreMask);

// Select a core to use for a specific "worker" thread.  The algorithm here
// just loops through the bits in <coreMask> repeatedly and picks the core
// corresponding to the <thread>+1'th one bit.  If the caller assigns
// conselwtive indices to <threadID>, this will spread worker threads across
// all cores enabled in the mask.  This function just selects a core number;
// it doesn't actually assign the current thread to that core.
extern int lwogThreadSelectCoreRoundRobin(int threadID, uint64_t coreMask);

// Create a thread to run on a specific CPU core.
extern LWOGthread *lwogThreadCreateOnCore(void(*threadFunc)(void*), void *args, size_t stackSize /* 0 = default */,
                                          int idealCore);

#endif

#ifdef LW_HOS
// Avoids conflicts with incomplete standard library implementation on HOS
int int_printf(const char *format, ...);
int int_vprintf(const char *format, va_list ap);
int int_fprintf(FILE *stream, const char *format, ...);
FILE * int_fopen(const char *path, const char *mode);
int int_fclose(FILE *file);
size_t int_fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
size_t int_fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);
int int_feof(FILE *file);
int int_fflush(FILE *stream);
char *int_fgets(char *s, int size, FILE *stream);
int int_fgetc(FILE *stream);
int int_fseek(FILE *file, long int offset, int origin);
long int int_ftell(FILE *file);
#define printf int_printf
#define vprintf int_vprintf
#define fprintf int_fprintf
#define fopen int_fopen
#define fclose int_fclose
#define fread int_fread
#define fwrite int_fwrite
#define feof int_feof
#define fflush int_fflush
#define fgets int_fgets
#define fgetc int_fgetc
#define fseek int_fseek
#define ftell int_ftell
#endif

#ifdef __cplusplus
}
#endif

#endif // __OGTEST_H

