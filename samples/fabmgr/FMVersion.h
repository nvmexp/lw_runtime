#ifdef __linux__
#include "lwUnixVersion.h"
#define FM_VERSION_STRING LW_VERSION_STRING
#else
#include "lwVer.h"
#define FM_VERSION_STRING LW_VERSION_STRING
#endif
