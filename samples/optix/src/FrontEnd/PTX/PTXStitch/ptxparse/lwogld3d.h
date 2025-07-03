
/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 *
 * THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
 * LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
 * IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
 */

// Define names for various macros and routines which can be used in
// common/src code shared by OpenGL and D3D driver components.

// For the non-trace macros, the D3D names like lwAssert and lwStrCat
// are a better choice (more generic).

// For the trace macros, the lwTraceOgl.h macros are more general, so use
// that interface.
// For D3D, you can get "level" tracing by redefining the TR_* bits.
// (D3D's "level" tracing is really done with a mask).
// Think of the level in lwTraceOgl.h as a verbosity control.
// The higher the level, the more verbose. See lwTraceOgl.h
// D3D code doesn't have a level, so the D3D flavor of the trace macros
// ignore the level.

#ifndef _lwogld3d_h
#define _lwogld3d_h

/*****************************************************************************/

#if defined(LW_PARSEASM) || defined(LW_GLSLC)

    #include <stdio.h>
    #include <stdlib.h>
    #include <memory.h>
    #include <assert.h>
    #include <string.h>
    #include <math.h>

//    #include "lwos.h"
    #include "lwTraceOgl.h"

    #define AllocIPM(X)                 malloc((X))
    #define ReallocIPM(OLD,SIZE,PNEW)   (*(PNEW) = realloc((OLD), (SIZE)))
    #define FreeIPM(X)                  free((X))
    #define lwMemCmp(X, Y, Z)           memcmp((X), (Y), (Z))
    #define lwMemCpy(X, Y, Z)           memcpy((X), (Y), (Z))
    #define lwMemSet(X, Y, Z)           memset((X), (Y), (Z))
    #ifndef lwAssert
    #define lwAssert(X)                 assert((X))
    #endif
    // Note: care should be taken if setjmp/longjmp are changed not to point to system macros/functions
    #define lwSetJmp(X)                 setjmp((X))
    #define lwLongJmp(X, Y)             longjmp((X), (Y))
    #define tprintf printf
    #define tprintString(str) { \
                int ii; \
                int bytes = strlen(str); \
                for (ii = 0;  ii < bytes;  ii++) { \
                    TPRINTF(("%c", (str)[ii])); \
                } \
            } \

    #define tprintProgram(str) tprintString(str)

    #define relprintf printf
    #define lwTracePrintArgs (void)
    #define lwTracePrintFunName(X)
    #define lwTraceFuncEnter(X) 0
    #define lwTraceFuncExit(X, Y, line)
    #define lwTraceFunCount(X)

    #define lwLog(X)                   log((X))
    #define lwPow(X, Y)                pow((X), (Y))

#elif defined(IS_OPENGL)

    #include <stdio.h>

//    #include "lwos.h"
    #if defined(LW_MACOSX_OPENGL)
    #include <OpenGL/gld.h>
    #include <OpenGL/gl.h>
    #include <OpenGL/glext.h>
    #endif

    #include "lwassert.h"
    #include "gltypescore.h"
    #include "GL/gl.h"
    #include "lwTraceOgl.h"
    #include "imports.h"

    #define AllocIPM(X)                 __GL_MALLOC(NULL, (X))
    #define ReallocIPM(OLD,SIZE,PNEW)   (*(PNEW) = __GL_REALLOC(NULL, (OLD), (SIZE)))
    #define FreeIPM(X)                  __GL_FREE(NULL, (X))
    #define lwMemCmp(X, Y, Z)           __GL_MEMCMP((X), (Y), (Z))
    #define lwMemCpy(X, Y, Z)           __GL_MEMCOPY((X), (Y), (Z))
    #define lwMemSet(X, Y, Z)           __GL_MEMSET((X), (Y), (Z))
    #ifndef lwAssert
    #define lwAssert(X)                 assert(X)
    #endif

    // Note: care should be taken if setjmp/longjmp are changed not to point to system macros/functions
    #ifdef _WIN32
    #  define lwSetJmp(X)               setjmp((X))
    #  define lwLongJmp(X, Y)           longjmp((X), (Y))
    #else
    // defining setjmp/longjmp away
    #  define lwSetJmp(X)               (0)
    #  define lwLongJmp(X, Y)           lwAssert(0)
    #endif

    // Math functions
    #define lwLog(X)                   __GL_LOGF((X))
    #define lwPow(X, Y)                __GL_POWF((X), (Y))

#else // D3D || D3D10

    #if IS_D3D10
        #define AllocIPM(X)             malloc(X);
        #define FreeIPM(X)              free(X);
    #endif

    #define lwMemCmp(X, Y, Z)           memcmp((X), (Y), (Z))
    #define lwMemCpy(X, Y, Z)           memcpy((X), (Y), (Z))
    #define lwMemSet(X, Y, Z)           memset((X), (Y), (Z))
    // Note: care should be taken if setjmp/longjmp are changed not to point to system macros/functions
    #define lwSetJmp(X)                 setjmp((X))
    #define lwLongJmp(X, Y)             longjmp((X), (Y))

    // kill off some macros that only make sense to OGL
    #undef LW_FUN_REC
    #undef LW_FUN_NAME
    #undef LW_TRACE_FUNC
    #undef LW_TRACE_PUSH
    #undef LW_TRACE_POP
    #define LW_FUN_REC(X, Y)           extern void lwNullDeclaration(void)
    #define LW_FUN_NAME(X, Y, Z)       extern void lwNullDeclaration(void)
    #define LW_TRACE_FUNC(X)
    #define LW_TRACE_PUSH()
    #define LW_TRACE_POP()

    // Math functions
    #define lwLog(X)                   log((X))
    #define lwPow(X, Y)                pow((X), (Y))

#endif // parseasm/ogl/d3d

#endif // _lwogld3d_h

