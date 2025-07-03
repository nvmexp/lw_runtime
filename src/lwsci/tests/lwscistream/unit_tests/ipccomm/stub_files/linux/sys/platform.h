#ifndef __PLATFORM_H_INCLUDED
#define __PLATFORM_H_INCLUDED


#define __QNXNTO__

#if defined(__MWERKS__)
    #include "sys/compiler_mwerks.h"
#elif defined(__GNUC__)
    #include "sys/compiler_gnu.h"
#elif defined(__INTEL_COMPILER)
    #include "sys/compiler_intel.h"
#else
    #error not configured for compiler
#endif

#if __INT_BITS__ == 32
#define _INT32        int
#define _UINT32       unsigned
#else
#define _INT32        _Int32t
#define _UINT32       _Uint32t
#endif

#if defined(__QNXNTO__)
    #include "sys/target_nto.h"
#elif defined(__QNX__)
    #include "sys/target_qnx.h"
#elif defined(__SOLARIS__) || defined(__NT__) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__LINUX__) || defined(__APPLE__)
    /* partial support only, solaris/win32/linux/darwin hosted targetting qnx6 */
#else
    #error not configured for target
#endif


#endif /* __PLATFORM_H_INCLUDED */
