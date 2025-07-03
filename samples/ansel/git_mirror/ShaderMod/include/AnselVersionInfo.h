// This file is only consumed by MS Resource compiler
#include "Ansel.h"
#include "AnselBuildNumber.h"

#define STRINGIZE_(x) #x
#define STRINGIZE(x) STRINGIZE_(x)

#define ANSEL_FILEVERSION \
    ANSEL_VERSION_MAJOR, ANSEL_VERSION_MINOR, ANSEL_BUILD_NUMBER, 0
#define ANSEL_FILEVERSION_STRING \
    STRINGIZE(ANSEL_VERSION_MAJOR.ANSEL_VERSION_MINOR.ANSEL_BUILD_NUMBER.0)

#define ANSEL_PRODUCTVERSION  \
    ANSEL_VERSION_MAJOR, ANSEL_VERSION_MINOR, ANSEL_BUILD_NUMBER, 0
#define ANSEL_PRODUCTVERSION_STRING \
    STRINGIZE(ANSEL_VERSION_MAJOR.ANSEL_VERSION_MINOR)

// Silly workaround for crippled Resource compiler. This never fires in a real build because
// the macro is set at compile time.  However, the RC compiler does static parsing of these
// files so this makes the IDE work ... <halldor>
#ifndef ANSEL_TARGET_NAME 
#define ANSEL_TARGET_NAME "LwCamera.dll"
#endif
