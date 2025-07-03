// **************************************************************************
//
//       Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
//
//     NOTICE TO USER:   The source code  is copyrighted under  U.S. and
//     international laws.  Users and possessors of this source code are
//     hereby granted a nonexclusive,  royalty-free copyright license to
//     use this code in individual and commercial software.
//
//     Any use of this source code must include,  in the user dolwmenta-
//     tion and  internal comments to the code,  notices to the end user
//     as follows:
//
//       Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
//
//     LWPU, CORPORATION MAKES NO REPRESENTATION ABOUT THE SUITABILITY
//     OF  THIS SOURCE  CODE  FOR ANY PURPOSE.  IT IS  PROVIDED  "AS IS"
//     WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.  LWPU, CORPOR-
//     ATION DISCLAIMS ALL WARRANTIES  WITH REGARD  TO THIS SOURCE CODE,
//     INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGE-
//     MENT,  AND FITNESS  FOR A PARTICULAR PURPOSE.   IN NO EVENT SHALL
//     LWPU, CORPORATION  BE LIABLE FOR ANY SPECIAL,  INDIRECT,  INCI-
//     DENTAL, OR CONSEQUENTIAL DAMAGES,  OR ANY DAMAGES  WHATSOEVER RE-
//     SULTING FROM LOSS OF USE,  DATA OR PROFITS,  WHETHER IN AN ACTION
//     OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF
//     OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.
//
//     U.S. Government  End  Users.   This source code  is a "commercial
//     item,"  as that  term is  defined at  48 C.F.R. 2.101 (OCT 1995),
//     consisting  of "commercial  computer  software"  and  "commercial
//     computer  software  documentation,"  as such  terms  are  used in
//     48 C.F.R. 12.212 (SEPT 1995)  and is provided to the U.S. Govern-
//     ment only as  a commercial end item.   Consistent with  48 C.F.R.
//     12.212 and  48 C.F.R. 227.7202-1 through  227.7202-4 (JUNE 1995),
//     all U.S. Government End Users  acquire the source code  with only
//     those rights set forth herein.
//
// **************************************************************************
//
//  Module: lwTrace.h
//     header file for basic library of tracing routines
//     common to all LW drivers
//
// **************************************************************************
//
//  History:
//       Craig Duttweiler    February 2005   debug spew cleanup
//
// **************************************************************************

#ifndef _LWTRACE_H
#define _LWTRACE_H

#include <stdarg.h>      // needed for va_list
#include <lwdebugcom.h>
#include "lwtypes.h"     // LW_CDECLCALL

#if !defined (LWCFG)
#define LWCFG(x)       1
#define LWCFG_OVERRIDE 1
#endif // !defined (LWCFG)

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 * lwTrace code is a set of macros and routines to make code for tracing,
 *      including controls for selecting which trace code is active.
 *      It is compiled with COMPILE_TRACE_LIBRARY and ENABLE_TRACE_CODE.
 *      For info on the distinction, see below where they're defined.
 *
 * Rules for using TRACE macros:
 *
 * lwTrace.masks[n] - Bitmask array used to select what gets traced.
 *     Categories are of the form TR_XXX, such as TR_CLEAR.
 *     Defined as an array of bitfields so we can have more than 32 bits.
 *     The TR_XXX values represent a bit number in this array, and will be
 *     turned into a word/bit value pair as necessary.
 *
 * lwTraceLevel - Level of detail to trace:
 *
 * < -1  - less than -1 are other levels for user hacks
 *
 *   -1  - Nothing will ever print.
 *         This is the only special level that will disable all printing.
 *
 *  0-1  - Level 0 and 1 are not used by the checked in driver.
 *         Level 1 is the default so TPRINTF() works by default.
 *         Level 0 behaves the same for backward compatibility.
 *         No code should be checked in which will default to printing
 *         with the level <= 1. These levels are for user hacks.
 *         Asserts should print at any level >= 0.
 *
 *    2  - Errors may print.
 *         For errors, driver code should use LW_TRACE*(TR_ERRORS, 2, ...)
 *
 *    3  - Warnings may print. Summary reports may print.
 *         For warnings, driver code should use LW_TRACE*(TR_ERRORS, 3, ...)
 *
 *    4  - Function counts are updated and printed on exit.
 *         LW_TRACE(TR_ERRORS, 4, ...) is used in __glSetError
 *
 *    5  - Trace ONLY actual user function calls, args, and return values.
 *         Keep this terse, it's not for anything but capturing what the
 *         user is calling!
 *    15 - Trace ONLY user calls, but print more info about them.  Must print
 *         as some form of C code.  Put "//" style comments before any non-C.
 *         Use 10 for very simple stuff, and 15 for a little more, but no
 *         firehoses here.
 *    20 - Lowest level to trace internal things, such as which functions are
 *         called, what kinds of optimizations are made.  Should be simple,
 *         one line items only.
 *    25 - Print more details
 *    30 - Even more
 *    50 - Bunches more, like dumping the entire push buffer
 *    100 - Lookout, here it comes!  Maybe even dump all textures, etc.
 *
 * LW_TRACE(tmask, level, args) - Calls "tprintf" to trace "args" when
 *    "tmask" is set in lwTrace.masks and lwTraceLevel is >= "level"
 *    The value of level should comply with the previously listed rules.
 *
 * LW_TRACE_CODE_COND(tmask, level, code) - like LW_TRACE, but any C code is
 *    allowed as the third argument, except lists of parameters at the top
 *    level.  For example:
 *    LW_TRACE_CODE_COND(tmask, level,
 *        int ii; int jj;          // not "int ii, jj"
 *        jj = someFunction(gc);
 *        for ( ii = 0; ii < jj; ii++ ) {
 *            printItem(gc, ii);
 *        }
 *    );
 *
 * LW_TRACE_ENABLED(tmaskIdx, level) -- Determines if tracing of <tmaskIdx> (of
 *    the form TR_XXX) at level <level> is enabled.  Returns non-zero if
 *    enabled and zero otherwise.
 */

/*****************************************************************************/
// COMPILE_TRACE_LIBRARY is for compiling code in lwTrace.c to support
// the use of tracing code throughout the other parts of the driver.
//
// Code compiled due to COMPILE_TRACE_LIBRARY should not be in the normal
// exelwtion paths of the driver (with the exception of registry reads on init).
// In other words, this compiles in support code, but should not effect
// performance by itself.
//
// This is separate from ENABLE_TRACE_CODE so that it may be set up
// for a release build, and other modules can
// selectively enable ENABLE_TRACE_CODE.
// For example, you can define COMPILE_TRACE_LIBRARY in a release build;
// performance should be identical; then you can hack in code using tprintf,
// use lwTraceLevel, masks, etc., and it should work as usual.

// ENABLE_TRACE_CODE is the typical control to insert code throughout the
// driver for trace purposes. This is _not_ for general debug code.
// For debug code you should be using macros/defines from lwDebug.h in OGL,
// or whatever in other modules.
//
// The trace macro's are "off" or <null> in release builds, and always enabled
// for DEBUG builds. They can also be enabled by defining
// defined(ENABLE_TRACE_CODE) (useful to get trace capability in release build).

// These are controlled here so lwTrace.h can stand on it's own.
// However, to modify for the OGL driver build, you should define in lwfirst.h
// For D3D ...
// For RM ...
// For MODS ...

#if !defined(ENABLE_TRACE_CODE)
    #if defined(DEBUG)
        #define ENABLE_TRACE_CODE 1
    #endif
#endif

#if defined(ENABLE_TRACE_CODE)
    #undef COMPILE_TRACE_LIBRARY
    #define COMPILE_TRACE_LIBRARY 1
#endif

/*****************************************************************************/
// Used to print out strings in release builds.  No code
// that uses RPRINTF should be checked in.  This is only
// for temporary use in debugging.

#define LW_TRACE_FUNC_STACK_DEPTH 128   // For OGL's PUSH/POP macros
#define LW_TRACE_MAX_LINE_SIZE 4096
#define LW_TRACE_PREFIX_PAD 65

#define LW_EXPERT_STRING_WRAP_LENGTH 72 // Length of an LW_EXPERT output line (for wrapping)

#if !defined(MAX_PATH)
#   define MAX_PATH 260   // match window's #define
#endif

#define RPRINTF(ARGS) relprintf ARGS

void LW_CDECLCALL relprintf(const char *format, ...);

void lwTraceProcessCleanup(void); // Always called, even in release builds.
void lwTraceProcessDetach(void);  // Called when the system calls process detach

extern int lwTraceLockEnable;     // enable the use of lock/unlock per print

#define LW_TRACE_MODULE_OGL     0
#define LW_TRACE_MODULE_OGLE    1
#define LW_TRACE_MODULE_D3D     2
#define LW_TRACE_MODULE_RM      3
#define LW_TRACE_MODULE_MODS    4
#define LW_TRACE_MODULE_COM     5       // can we put this first ?
#define LW_TRACE_MODULE_VIDEO   6       // can we put this first ?
#define LW_TRACE_MODULE_DX10    7
// Define GLK to be the same as DX10, since they will never
// co-exist in the same OS (pretty unlikely anyway)
// This lame trick is needed because LW_TRACE_MASK_NUM_WORDS is
// fixed at 32
#define LW_TRACE_MODULE_GLK     LW_TRACE_MODULE_DX10
#define LW_TRACE_NUM_MODULES    8       // OGL, D3D, RM, MODS

#define LW_TRACE_MASK_WORDS_PER_MODULE  4

#define LW_TRACE_MASK_NUM_WORDS \
        (LW_TRACE_MASK_WORDS_PER_MODULE * LW_TRACE_NUM_MODULES)

// For easier use in debugger,
// trace variables are kept in globals instead of a structure.
// Yet, sometimes there's a need to copy/push/pop the entire set of variables.
// struct always defined for tables in escape.c
typedef struct LW_TRACE_INFO_REC
{
    int lwTraceOptions;
    int lwTraceLevel;
    int lwTraceMask[LW_TRACE_MASK_NUM_WORDS];
    void *LW_PTR lwTraceFileHandle;     // for lwTrace
    const char *LW_PTR lwTraceReleasePrefix;
    const char *LW_PTR lwTracePrefix;

    // Debug controls packaged here too.
    // This adds a few non-trace items to lwTrace.h
    // If this gets large, add a separate structure, in a different .h, etc.
    // For now, this is simpler.
    int lwDebugOptions;
    int lwControlOptions;

    int lwitemp0;                       // for temp hacks
    int lwitemp1;
    int lwitemp2;
    int lwitemp3;
    int lwitemp4;
} LW_TRACE_INFO;

/*****************************************************************************/
#if defined(COMPILE_TRACE_LIBRARY) || defined(COMPILE_EXPERT_LIBRARY)

#define LW_TRACE_VERSION        2       // a simple version number to increment

// mask indices for each module
#define LW_TRACE_OGL_BASE_INDEX     (LW_TRACE_MODULE_OGL  *32*LW_TRACE_MASK_WORDS_PER_MODULE)
#define LW_TRACE_OGLE_BASE_INDEX    (LW_TRACE_MODULE_OGLE *32*LW_TRACE_MASK_WORDS_PER_MODULE)
#define LW_TRACE_D3D_BASE_INDEX     (LW_TRACE_MODULE_D3D  *32*LW_TRACE_MASK_WORDS_PER_MODULE)
#define LW_TRACE_RM_BASE_INDEX      (LW_TRACE_MODULE_RM   *32*LW_TRACE_MASK_WORDS_PER_MODULE)
#define LW_TRACE_MODS_BASE_INDEX    (LW_TRACE_MODULE_MODS *32*LW_TRACE_MASK_WORDS_PER_MODULE)
#define LW_TRACE_VIDEO_BASE_INDEX   (LW_TRACE_MODULE_VIDEO*32*LW_TRACE_MASK_WORDS_PER_MODULE)
#define LW_TRACE_DX10_BASE_INDEX    (LW_TRACE_MODULE_DX10 *32*LW_TRACE_MASK_WORDS_PER_MODULE)
#define LW_TRACE_GLK_BASE_INDEX     (LW_TRACE_MODULE_GLK  *32*LW_TRACE_MASK_WORDS_PER_MODULE)

/*****************************************************************************/
// define core lwtrace macros and declarations

extern char lwTraceFilename[MAX_PATH];

#define LW_MAX_CPB_IGNORE   10

////////////////////////////////////////////////////////
// Now we get started defining the different trace bits.
// To keep everything in a single location, we have the file
// lwTraceBits.h, which uses a simple macro language to
// define the bits.  lwTraceBits.h is then included into this
// file twice, the macros interpreted in different ways each
// time.
//
// Note that the OGL bits ended up being named like TR_CLEAR,
// whereas other modules have the module name embedded at the
// beginning, such as with LW_D3D_INFO.  Really, OGL should
// name their bits as other modules have (e.g., TR_OGL_CLEAR),
// but until someone feels like changing a whole lot of files,
// we'll declare different macros for OGL and everyone else.

/*****************************************************************************/
// TR_MASK_GET_{WORD,BITS}:  Given a TR_XXX value, the _WORD macro
// identifies the dword in the trace mask array and the _BIT macro identifies
// the specific bit in that word.
#define TR_MASK_GET_WORD(tmaskIdx)        ((tmaskIdx) >> 5)
#define TR_MASK_GET_BIT(tmaskIdx)         (1 << ((tmaskIdx) & 0x1F))

#define TR_TURN_MASK_ON(tmaskIdx) \
        lwTrace.masks[TR_MASK_GET_WORD(tmaskIdx)] |= \
                      TR_MASK_GET_BIT(tmaskIdx)

#define TR_TURN_MASK_OFF(tmaskIdx) \
        lwTrace.masks[TR_MASK_GET_WORD(tmaskIdx)] &= \
                     ~TR_MASK_GET_BIT(tmaskIdx)

/*****************************************************************************/
extern int lwTraceLevel;

// by default, the tracing level is the global 'lwTraceLevel'.
// You may change this by defining LW_TRACE_LEVEL before including lwTrace.h
// Some ideas:
//      1) you could use a different level in some modules
//         (if shared code, could be different per build)
//      2) you could have the level be a function of the traceIdx
//      3) ...
// Whatever you do, don't add the scheme to this file.
// Keep it somewhere "above".

#if !defined(LW_TRACE_LEVEL)
#define LW_TRACE_LEVEL(maskIdx) lwTraceLevel
#endif

// names for levels
enum {
    LW_LEVEL_NO_PRINTING = -1,      // The only level which disables printing
    LW_LEVEL_USER0 = 0,             // levels <= -2, ==0, and ==1 are ...
    LW_LEVEL_USER1 = 1,             // ... totally unused by the driver.
    LW_LEVEL_ERROR = 2,             // For serious error messages
    LW_LEVEL_WARNING = 3,           // Informative, but probably OK.
    LW_LEVEL_SUMMARY_INFO = 3,      // Results from timing or stats
    LW_LEVEL_FUNCTION_COUNTS = 4,   // func info also collected and reported
    LW_LEVEL_API = 5,               // user API traced
    LW_LEVEL_DRIVER_API = 10,       // Entry points from Apple's layer into our driver, or Apple code in Common/...
    LW_LEVEL_INTERNAL_BASE = 20,    // first level for general tracing
    LW_LEVEL_MODERATE = 25,         // Used to describe paths taken, explain decisions made, etc.
    LW_LEVEL_HEAVY = 30,            // Used to log values of structures used, timestamps, bitmasks, surface params, blit params, etc.
    LW_LEVEL_SEVERE = 50,           // Log everything without completely flooding the logging system.
                                    // No disassembly, no buffer dumps, but pretty much everything else
    LW_LEVEL_INSANE = 100           // Everything must go
};

// trace levels for TR_VP_CREATE
//      20
//      35 summary
//      40 final uCode
//      43 OG disasm in vp2CompileInternal()
//      50 intermediate uCode
//      55 re-pick tracing
//      60 initial uCode

// trace levels for TR_PROGRAM
//      50 live/dead optimization in vp2optimize.c

/*****************************************************************************/

extern int lwTraceOptions;

/*****************************************************************************/
// ***** Bits for lwTraceOptions

// Options only available in trace mode.
#if defined(COMPILE_TRACE_LIBRARY)

#define TRO_ENABLE_REREAD       0x00000100 // re-read in readDebugRegistry()
#define TRO_DISABLE_TRACE_COUNT 0x00000400 // set to disable tracing counter
#define TRO_ENABLE_USER_NAME    0x00000800 // OGL: print alt function name

#define TRO_PRINT_PUSH_POP      0x00010000 // OGL: see lwTraceOgl.h
#define TRO_TRACE_DL_EXELWTION  0x00020000 // trace during display list exec

#define TRO_TEMP1               0x01000000 // for hacks.
#define TRO_TEMP2               0x02000000 // Do not check in usage of TEMPs
#define TRO_TEMP3               0x04000000
#define TRO_TEMP4               0x08000000

#define TRO_DISABLE_ASSERT_BRK  0x80000000 // do not 'int 3' on assert

#endif // defined(COMPILE_TRACE_LIBRARY)

// Options working in both TRACE and EXPERT mode. 
// The distinction is only done to make sure expert mode 
// doesn't reveal driver internal tracing.
#define TRO_LOG_TO_CONSOLE      0x00000001 // trace output to a console window
#define TRO_LOG_TO_FILE         0x00000002 // trace output to a file
#define TRO_LOG_TO_DEBUGGER     0x00000004 // trace output to debugger window
#define TRO_LOG_TO_CALLBACK     0x00000008 // trace output to a user-specified callback
#define TRO_LOG_TO_STDOUT       0x00000010 // trace output to stdout
#define TRO_LOG_TO_ANY \
        (TRO_LOG_TO_CONSOLE | TRO_LOG_TO_STDOUT | TRO_LOG_TO_FILE | TRO_LOG_TO_DEBUGGER | TRO_LOG_TO_CALLBACK)

// DAR These two are not working with LW_EXPERT, yet, but are needed to compile.
#define TRO_PER_FRAME_LOGFILE   0x00000200 // w/LOG_TO_FILE, new name each frame
#define TRO_PRINT_FRAME_COUNT   0x00001000 // prepend each line with frame count

#define TRO_PRINT_THREAD_ID     0x00002000 // prepend each line with thread ID
#define TRO_PRINT_THREAD_NUM    0x00004000 // ... with thread number
#define TRO_PRINT_INDENT        0x00008000 // indent per stack level
#define TRO_PRINT_FUNCTION      0x00040000 // prepend each line with the function name (class::function for C++)
#define TRO_PRINT_THIS          0x00080000 // Add the this ptr for C++ (i.e. class(ptr)::function)
#define TRO_PRINT_PREFIX_PAD    0x00100000 // pad the prefix out to LW_TRACE_PREFIX_PAD characters
#define TRO_PRINT_THREAD_NAME   0x00200000 // Print the owning process name

#define TRO_FLUSHFILE_PER_WRITE 0x40000000 // flush log file after each write

/*****************************************************************************/
// note: variables and prototypes are only available when compiling with
// COMPILE_TRACE_LIBRARY and/or ENABLE_TRACE_CODE
// This provides for getting a compile warning with a specific line number
// when these names are used out of context.

int LW_CDECLCALL tprintf(const char *format, ...);
int LW_CDECLCALL xtprintf(const char *pretty_func_name, const void *class_ptr, const char *format, ...);
int LW_CDECLCALL vtprintf(const char *format, va_list args);
int LW_CDECLCALL tprintfOptions(int options, const char *format, ...);
int LW_CDECLCALL vtprintfOptions(int options, const char *format, va_list args);
int LW_CDECLCALL vxtprintfOptions(int options, const char *pretty_func_name, const void *class_ptr, const char *format, va_list args);

void LW_CDECLCALL lwTraceFlushFile(void);
void *lwTraceGetFileHandle(void);           // for log file tracing

void lwTraceGetInfo(LW_TRACE_INFO *pInfo);
void lwTraceSetInfo(LW_TRACE_INFO *pInfo);
void lwTracePushInfo(int setOption); // setOption == what to set after push
void lwTracePopInfo(void);
#if defined(LW_MACOSX_OPENGL)
int  lwTraceBuildPrefix(int options, const char *pretty_func_name, const void *class_ptr);
#endif

void tprintString(const char *str);         // for arbitrarily long strings
void tprintProgram(const char *str);        // for VP/FP/GLSL/XX programs

// LW_TPRINTF_FUNC for callback into alternative printf function.
// return non-zero if printing was handled.
typedef int LW_TPRINTF_FUNC(int options, const char *fStr, 
                            void (* printFunc)(const char *pPrefix, int options, const char *));

LW_TPRINTF_FUNC *lwTraceSetPrintCallback(LW_TPRINTF_FUNC *pNewFunc);

LW_TPRINTF_FUNC tprinttid;    // callback function for extended print options

////////////////////////////////////////////////////////
// First thing is to declare a structure that contains
// our trace debug bits, named appropriately, and unioned
// with a mask array.

#define TR_DEFINE_BIT(module, name, index) unsigned int b##module##_##name : 1;
#define TR_DEFINE_BIT_OGL(module, name, index) TR_DEFINE_BIT(module, name, index)

#define TR_START_MODULE(module) \
union { \
    unsigned int masks[LW_TRACE_MASK_WORDS_PER_MODULE]; \
    struct { \

#define TR_END_MODULE(module) \
    } bits; \
} module##Bits;

#define TR_SKIP_BITS(start, end) unsigned int bIlwalid##start##_##end : (end-start+1);

typedef union _lwTraceType {
    unsigned int masks[LW_TRACE_MASK_NUM_WORDS];
    struct {
#include "lwTraceBits.h"
    } namedBits;
} lwTraceType;

extern lwTraceType lwTrace;

/////////////////////////////////////////////////////////
// Now, redefine our macros to define a set of enums, again
// named appropriately and having values equal to the index
// plus the base index for that module.

#undef TR_START_MODULE
#undef TR_END_MODULE
#undef TR_DEFINE_BIT
#undef TR_DEFINE_BIT_OGL
#undef TR_SKIP_BITS

#define TR_START_MODULE(module)
#define TR_END_MODULE(module)
#define TR_DEFINE_BIT(module, name, index) TR_##module##_##name = ((index) +  LW_TRACE_##module##_BASE_INDEX),
#define TR_DEFINE_BIT_OGL(module, name, index) TR_##name = ((index) +  LW_TRACE_##module##_BASE_INDEX),
#define TR_SKIP_BITS(start, end)

enum {
#include "lwTraceBits.h"
};

#define TR_GLX          TR_ICD_CMDS     // glx
#define TR_WIN          TR_ICD_CMDS     // window interfacing stuff
#define TR_FENCE        TR_PUSHBUFFER   // Trace as buffers

// TR_ALWAYS:  When we used a single bitfield, TR_ALWAYS was set to 0xFFFFFFFF
// -- trace if ANY tracing bit is set.  This behavior is preserved by setting
// this bit in the mask array iff any other tracing bit is set.  This is done
// by the macro LW_TRACE_SET_ALWAYS_BIT below.
#define TR_ALWAYS               127

#endif // defined(COMPILE_TRACE_LIBRARY) || defined(COMPILE_EXPERT_LIBRARY)

/*****************************************************************************/
/*****************************************************************************/
// Shared macros between trace and expert mode.

#if defined(ENABLE_TRACE_CODE) || defined(ENABLE_EXPERT_CODE)

// Extended version of tprintf that will add function/threadid/this prefix data
// Requires extra data to be passed by macro
#if defined(LW_MACOSX_OPENGL)
#define XTPRINTF_CLASS(fmt, ...) xtprintf(__PRETTY_FUNCTION__, this, fmt, ##__VA_ARGS__) 
#define XTPRINTF(fmt, ...)       xtprintf(__PRETTY_FUNCTION__, NULL, fmt, ##__VA_ARGS__) 
#endif
#define TPRINTF(X) tprintf X
#define VTPRINTF(X) vtprintf X

#define TR_IS_MASK_ON(tmaskIdx) \
        (lwTrace.masks[TR_MASK_GET_WORD(tmaskIdx)] & TR_MASK_GET_BIT(tmaskIdx))

#define LW_TRACE_ENABLED(tmaskIdx, level) \
        ((LW_TRACE_LEVEL(tmaskIdx) >= (level)) && TR_IS_MASK_ON(tmaskIdx))

#define LW_TRACE_FLUSHFILE() lwTraceFlushFile();

#else   // defined(ENABLE_TRACE_CODE) || defined(ENABLE_EXPERT_CODE)

#define XTPRINTF_CLASS(X)                       ((void)0)
#define XTPRINTF(X)                             ((void)0)
#define TPRINTF(X)                              ((void)0)
#define VTPRINTF(X)                             ((void)0)
#define TR_IS_MASK_ON(tmaskIdx)                 0
#define LW_TRACE_ENABLED(tmaskIdx, level)       0
#define LW_TRACE_FLUSHFILE()                    ((void)0)

#endif  // defined(ENABLE_TRACE_CODE) || defined(ENABLE_EXPERT_CODE)


/*****************************************************************************/
// These macros are the backbone of the OpenGL expert functionality:

#if defined(COMPILE_EXPERT_LIBRARY)

// The essential variables (ones not shared with LW_TRACE)
extern int lwExpertDetailMask;
extern char lwExpertDetailedErrorString[LW_TRACE_MAX_LINE_SIZE];
extern const char* lwExpertBufferObjectTypes[];
extern int lwExpertObjectId;

// by default, the expert level mask is the global 'lwExpertDetailMask' You may
// change this by defining LW_EXPERT_DETAIL_MASK before including lwTrace.h 
// Some ideas:
//      1) you could use a different mask in some modules
//         (if shared code, could be different per build)
//      2) you could have the mask be a function of the expertIndex
//      3) ...
// Whatever you do, don't add the scheme to this file.
// Keep it somewhere "above".
#if !defined(LW_EXPERT_DETAIL_MASK)
#define LW_EXPERT_DETAIL_MASK(expertIndex) lwExpertDetailMask
#endif

typedef void (* LW_EXPERT_CALLBACK) (unsigned int categoryId, unsigned int messageId, unsigned int level, int objectID, const char *messageStr);
extern LW_EXPERT_CALLBACK lwExpertCallback;

// A pass to define an array of unique indentifiers for all available messages.
// This value will be passed to the application with the message.
#define OGLE_CATEGORY_SHIFT 16

#define OGLE_BEGIN_CATEGORY(category, description) \
    OGLE_##category##_BASE_UID = (TR_OGLE_##category << OGLE_CATEGORY_SHIFT),

#define OGLE_MESSAGE(category, index, name, level, message) \
    OGLE_##category##_##name##_UID = (OGLE_##category##_BASE_UID + index),

#define OGLE_END_CATEGORY()

enum {
#include "lwExpertMessages.h"
};

#undef OGLE_BEGIN_CATEGORY
#undef OGLE_MESSAGE
#undef OGLE_END_CATEGORY

// A pass to define an enum of all available messages, with contiguous
// range of indices. 
#define OGLE_BEGIN_CATEGORY(category, description)
#define OGLE_MESSAGE(category, index, name, level, message) OGLE_##category##_##name,
#define OGLE_END_CATEGORY()

enum {
#include "lwExpertMessages.h"
};

#undef OGLE_BEGIN_CATEGORY
#undef OGLE_MESSAGE
#undef OGLE_END_CATEGORY

// Finally, we'll define an array of structures, which will be indexed with the
// enum above, which defines the level and base message for each of the Expert
// Driver messages.
typedef struct OGLE_MESSAGES_REC {
    unsigned int categoryInt;   // Mask used internally by the trace routines
    unsigned int categoryExt;   // Mask used externally by the user
    unsigned int messageID;     // Unique ID for this message
    const char  *message;       // Base text for this message
    int detailLevel;            // Bit that corresponds to the detail mask
} OGLE_MESSAGES;

extern OGLE_MESSAGES OGLEMessages[];

#endif // defined(COMPILE_EXPERT_LIBRARY)

/*****************************************************************************/
// Expert only code:

#if defined(ENABLE_EXPERT_CODE)

#define LW_EXPERT_ENABLED(expertIndex) \
    ((LW_EXPERT_DETAIL_MASK(expertIndex) & (OGLEMessages[expertIndex].detailLevel)) && \
     (TR_IS_MASK_ON(OGLEMessages[expertIndex].categoryInt)))

void LW_CDECLCALL lwExpertPrint(unsigned int expertIndex, const char *userFormat, ...);
void LW_CDECLCALL lwExpertPrintArgs(unsigned int expertIndex, const char *userFormat, va_list args);

#define LW_EXPERT_CODE(code)                                                            \
    if (__glProcessGlobalData.lwpmInstrumentationEnabled) {                             \
        code                                                                            \
    }

#define LW_EXPERT(printParameters)                                                      \
    if (__glProcessGlobalData.lwpmInstrumentationEnabled) {                             \
        lwExpertObjectId = -1;                                                          \
        lwExpertPrint printParameters;                                                  \
    }

#define LW_EXPERT_ARGS(expertIndex, userFormat, args)                                   \
    if (__glProcessGlobalData.lwpmInstrumentationEnabled) {                             \
        lwExpertObjectId = -1;                                                          \
        lwExpertPrintArgs(expertIndex, userFormat, args);                               \
    }

#define LW_EXPERT_OBJECT(objectId, printParameters)                                     \
    if (__glProcessGlobalData.lwpmInstrumentationEnabled) {                             \
        lwExpertObjectId = objectId;                                                    \
        lwExpertPrint printParameters;                                                  \
    }

#define LW_EXPERT_OBJECT_ARGS(objectId, expertIndex, userFormat, args)                  \
    if (__glProcessGlobalData.lwpmInstrumentationEnabled) {                             \
        lwExpertObjectId = objectId;                                                    \
        lwExpertPrintArgs(expertIndex, userFormat, args);                               \
    }

#define LW_EXPERT_ERROR(errorString)                                                    \
    if (__glProcessGlobalData.lwpmInstrumentationEnabled) {                             \
        if (lwStrLen(errorString) < LW_TRACE_MAX_LINE_SIZE - 9) {                       \
            lwSprintf(lwExpertDetailedErrorString, "\n      %s", errorString);          \
        } else {                                                                        \
            char truncatedString[LW_TRACE_MAX_LINE_SIZE];                               \
            lwMemCpy(truncatedString, errorString, LW_TRACE_MAX_LINE_SIZE - 9);         \
            truncatedString[LW_TRACE_MAX_LINE_SIZE-8] = 0;                              \
            lwSprintf(lwExpertDetailedErrorString, "\n      %s", truncatedString);      \
        }                                                                               \
    }

#else // defined(ENABLE_EXPERT_CODE)

#define LW_EXPERT_CODE(code)
#define LW_EXPERT(code) ((void)0)
#define LW_EXPERT_OBJECT(objectId, printParameters)
#define LW_EXPERT_ERROR(code)

#endif // defined(ENABLE_EXPERT_CODE)

/*****************************************************************************/
// Trace only code:

#if defined(ENABLE_TRACE_CODE)

#define LW_TRACE_CODE(code) code

#if defined(LW_MACOSX_OPENGL)

#define LW_TRACE_CODE_COND(tmaskIdx, level, code) \
        do { if (LW_TRACE_ENABLED(tmaskIdx, level)) { code; } } while(0)

#define LW_TRACE(tmaskIdx, level, args) \
        do { if (LW_TRACE_ENABLED(tmaskIdx, level)) { XTPRINTF args; } } while(0)

#define LW_TRACE_CLASS(tmaskIdx, level, args) \
        do { if (LW_TRACE_ENABLED(tmaskIdx, level)) { XTPRINTF_CLASS args; } } while(0)

#else

#define LW_TRACE_CODE_COND(tmaskIdx, level, code) \
        { if (LW_TRACE_ENABLED(tmaskIdx, level)) { code; } }

#define LW_TRACE(tmaskIdx, level, args) \
        { if (LW_TRACE_ENABLED(tmaskIdx, level)) { TPRINTF(args); } }

#endif //defined(LW_MACOSX_OPENGL)

// #define LW_TRACE_FLUSHFILE() lwTraceFlushFile();

#else  // defined(ENABLE_TRACE_CODE)

#define LW_TRACE_CODE(code)
#define LW_TRACE_CODE_COND(tmaskIdx, level, code)    ((void)0)
#define LW_TRACE(tmaskIdx, level, args)              ((void)0)
#define LW_TRACE_CLASS(tmaskIdx, level, args)        ((void)0)

#endif  // defined(ENABLE_TRACE_CODE)

// Enabling IPRINTF in instrumented driver
// Only for internal use
#ifdef DX_INSTRUMENTATION
#if(DX_INSTRUMENTATION==1)
    #define IPRINTF(X) relprintf X
    int LW_CDECLCALL iprintf(const char *format, ...);
#endif //(DX_INSTRUMENTATION==1)
#endif

/*****************************************************************************/

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#if defined (LWCFG_OVERRIDE)
#undef LWCFG
#endif // defined (LWCFG_OVERRIDE)

#endif // _LWTRACE_H
