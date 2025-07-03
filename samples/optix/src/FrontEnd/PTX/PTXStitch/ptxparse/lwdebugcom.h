
#ifndef __LW_DEBUGCOM_H
#define __LW_DEBUGCOM_H

// common definitions for trace/debug code.
// Mostly for OGL code.
// Ensure display driver code and OGL lw*shared* code compiled w/same options

/*****************************************************************************/
// options for lwTrace.h and lwDebug.h

// tracing implies enable all the other debug code
#if defined(DEBUG) || defined(ENABLE_TRACE_CODE)
#   undef  ENABLE_DEBUG_CODE
#   define ENABLE_DEBUG_CODE 1
#endif

// If full debug control is on, then make sure this one on too.
// It's OK to just set ENABLE_FORCE_CLEAR_COLOR without anything else.
#if defined(ENABLE_DEBUG_CODE)
#   undef  ENABLE_FORCE_CLEAR_COLOR
#   define ENABLE_FORCE_CLEAR_COLOR 1
#endif

// For GLrandom paths, output trace information for TR_RASTER level zero.

// about anything set implies to compile the supporting code in lwDebug.c, etc
#if defined(ENABLE_DEBUG_CODE) || defined(ENABLE_FORCE_CLEAR_COLOR) || defined(COMPILE_DEBUG_LIBRARY) || defined(COMPILE_TRACE_LIBRARY) || defined(COMPILE_DEBUG_CONTROL) || defined(GLRANDOM) || defined(ENABLE_ASSERT)
    // debug library is for supporting debug code, mostly in lwDebug.c
#   undef  COMPILE_DEBUG_LIBRARY
#   define COMPILE_DEBUG_LIBRARY 1
    // trace library is for supporting trace code, mostly in lwTrace.c
#   undef  COMPILE_TRACE_LIBRARY
#   define COMPILE_TRACE_LIBRARY 1
    // debug control is just for defining some debug variables and setting
    // them using registry entries
#   undef  COMPILE_DEBUG_CONTROL
#   define COMPILE_DEBUG_CONTROL 1
#endif

#endif // __LW_DEBUGCOM_H

/*****************************************************************************/
// only include the following after lwTrace.h has been included (or from it).

#ifdef _LWTRACE_H

#ifndef __LW_DEBUGCOM_H2
#define __LW_DEBUGCOM_H2

/*****************************************************************************/
#if defined(COMPILE_DEBUG_LIBRARY) || defined(COMPILE_TRACE_LIBRARY) || defined(COMPILE_EXPERT_LIBRARY)

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

extern int lwDebugOptions;
extern int lwDebugStatus;           // for internal tracing info and control
extern int lwControlOptions;

extern int lwitemp0;                // for temp hacks
extern int lwitemp1;
extern int lwitemp2;
extern int lwitemp3;
extern int lwitemp4;
extern int lwitagtest;

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // defined(COMPILE_DEBUG_LIBRARY) || defined(COMPILE_TRACE_LIBRARY) || defined(COMPILE_EXPERT_LIBRARY)


#endif // __LW_DEBUGCOM_H2

#endif // _LWTRACE_H

