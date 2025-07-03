#include "stdafx.h"
#include "Utility.h"

// This is useful in debugging which command was the last one exelwted before swak died.
// Enabled by -dbglog option.
void LwsiErrorMsg(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    fprintf(stderr,fmt,args);
    va_end(args);
}
