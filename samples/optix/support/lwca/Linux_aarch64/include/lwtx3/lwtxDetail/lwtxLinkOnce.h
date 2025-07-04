#ifndef __LWTX_LINKONCE_H__
#define __LWTX_LINKONCE_H__

/* This header defines macros to permit making definitions of global variables
 * and functions in C/C++ header files which may be included multiple times in
 * a translation unit or linkage unit.  It allows authoring header-only libraries
 * which can be used by multiple other header-only libraries (either as the same
 * copy or multiple copies), and does not require any build changes, such as
 * adding another .c file, linking a static library, or deploying a dynamic
 * library.  Globals defined with these macros have the property that they have
 * the same address, pointing to a single instance, for the entire linkage unit.
 * It is expected but not guaranteed that each linkage unit will have a separate
 * instance.
 *
 * In some situations it is desirable to declare a variable without initializing
 * it, refer to it in code or other variables' initializers, and then initialize
 * it later.  Similarly, functions can be prototyped, have their address taken,
 * and then have their body defined later.  In such cases, use the FWDDECL macros 
 * when forward-declaring LINKONCE global variables without initializers and
 * function prototypes, and then use the DEFINE macros when later defining them.
 * Although in many cases the FWDDECL macro is equivalent to the DEFINE macro,
 * following this pattern makes code maximally portable.
 */

#if defined(__MINGW32__) /* MinGW */
    #define LWTX_LINKONCE_WEAK __attribute__((section(".gnu.linkonce.0.")))
    #if defined(__cplusplus)
        #define LWTX_LINKONCE_DEFINE_GLOBAL   __declspec(selectany)
        #define LWTX_LINKONCE_DEFINE_FUNCTION extern "C" inline LWTX_LINKONCE_WEAK
    #else
        #define LWTX_LINKONCE_DEFINE_GLOBAL   __declspec(selectany)
        #define LWTX_LINKONCE_DEFINE_FUNCTION LWTX_LINKONCE_WEAK
    #endif
#elif defined(_MSC_VER) /* MSVC */
    #if defined(__cplusplus)
        #define LWTX_LINKONCE_DEFINE_GLOBAL   extern "C" __declspec(selectany)
        #define LWTX_LINKONCE_DEFINE_FUNCTION extern "C" inline
    #else
        #define LWTX_LINKONCE_DEFINE_GLOBAL   __declspec(selectany)
        #define LWTX_LINKONCE_DEFINE_FUNCTION __inline
    #endif
#elif defined(__CYGWIN__) && defined(__clang__) /* Clang on Cygwin */
    #define LWTX_LINKONCE_WEAK __attribute__((section(".gnu.linkonce.0.")))
    #if defined(__cplusplus)
        #define LWTX_LINKONCE_DEFINE_GLOBAL   LWTX_LINKONCE_WEAK
        #define LWTX_LINKONCE_DEFINE_FUNCTION extern "C" LWTX_LINKONCE_WEAK
    #else
        #define LWTX_LINKONCE_DEFINE_GLOBAL   LWTX_LINKONCE_WEAK
        #define LWTX_LINKONCE_DEFINE_FUNCTION LWTX_LINKONCE_WEAK
    #endif
#elif defined(__CYGWIN__) /* Assume GCC or compatible */
    #define LWTX_LINKONCE_WEAK __attribute__((weak))
    #if defined(__cplusplus)
        #define LWTX_LINKONCE_DEFINE_GLOBAL   __declspec(selectany)
        #define LWTX_LINKONCE_DEFINE_FUNCTION extern "C" inline
    #else
        #define LWTX_LINKONCE_DEFINE_GLOBAL   LWTX_LINKONCE_WEAK
        #define LWTX_LINKONCE_DEFINE_FUNCTION LWTX_LINKONCE_WEAK
    #endif
#else /* All others: Assume GCC, clang, or compatible */
    #define LWTX_LINKONCE_WEAK   __attribute__((weak))
    #define LWTX_LINKONCE_HIDDEN __attribute__((visibility("hidden")))
    #if defined(__cplusplus)
        #define LWTX_LINKONCE_DEFINE_GLOBAL   LWTX_LINKONCE_HIDDEN LWTX_LINKONCE_WEAK
        #define LWTX_LINKONCE_DEFINE_FUNCTION extern "C" LWTX_LINKONCE_HIDDEN inline
    #else
        #define LWTX_LINKONCE_DEFINE_GLOBAL   LWTX_LINKONCE_HIDDEN LWTX_LINKONCE_WEAK
        #define LWTX_LINKONCE_DEFINE_FUNCTION LWTX_LINKONCE_HIDDEN LWTX_LINKONCE_WEAK
    #endif
#endif

#define LWTX_LINKONCE_FWDDECL_GLOBAL   LWTX_LINKONCE_DEFINE_GLOBAL   extern
#define LWTX_LINKONCE_FWDDECL_FUNCTION LWTX_LINKONCE_DEFINE_FUNCTION

#endif /* __LWTX_LINKONCE_H__ */
