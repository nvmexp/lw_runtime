//****************************************************************
//
// commandLookup.c - routes LwWatch commands to their implementations
//
//****************************************************************
#include "lwtypes.h"
#include <string.h>
#include <stdio.h>
#include "lwwatch-config.h"

#if !LWWATCHCFG_FEATURE_ENABLED(WINDOWS_STANDALONE)
#include "usermode.h"
#endif // WINDOWS_STANDALONE
#include "commandLookup.h"

#define LWWATCH_API(NAME)  {"!lw." #NAME, #NAME, (void (*)(void))NAME },
struct cmd Api[] =    
{
#include "exts.h"
    LWWATCH_API(quit)
};
#undef LWWATCH_API

int NUMELEMS() { return sizeof(Api)/sizeof( struct cmd); }

void commandLookup( const char *Cmd )
{
    int Index;
    BOOL Success;

    Success = FALSE;
   
    for( Index =0; Index < NUMELEMS(); Index++)
    {
        if( !strcasecmp(Cmd, Api[Index].handle)  || !strcasecmp(Cmd,Api[Index].short_handle) )
        {
            (*(Api[Index].function))();
            Success = TRUE;
            break;
        }
    }

    if ( !Success)
        dprintf("lw: Unsupported Command.\n");
        
}
