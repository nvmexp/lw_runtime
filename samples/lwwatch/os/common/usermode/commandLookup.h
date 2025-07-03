//****************************************************************
//
// commandLookup.h - routes LwWatch commands to their implementations
//
//****************************************************************

#ifndef COMMAND_LOOKUP_H
#define COMMAND_LOOKUP_H

#include "lwwatch.h"
#include "lwtypes.h"

#if !LWWATCHCFG_FEATURE_ENABLED(WINDOWS_STANDALONE)
#include "usermode.h"
#endif // WINDOWS_STANDALONE
#include "exts.h"

int strcasecmp(const char * s0, const char * s1);

struct cmd
{
    char *handle;
    char *short_handle;
    void (*function)(void);
};

extern void quit(void);


//-------------------------------------------------------------------------------------------------------
// Api[] is the routing table for LwWatch. Appropriate routes must beadded if you add new
// functionality to LwWatch. Without the routeyour new API will just be dead code :).
// To add a new functionality that you want to support in LwWatch, implement the routine X in exts.c.
// Then add a 'COMMAND(X)' below. Also add an 'extern void X();' in the list above so that the compiler
// doesnt complain.
//-------------------------------------------------------------------------------------------------------

extern struct cmd Api[];

void commandLookup( const char *Cmd );

#endif // COMMAND_LOOKUP_H
