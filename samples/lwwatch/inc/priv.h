//*****************************************************
//
// lwwatch WinDbg Extension
// priv.h
//
//*****************************************************

#ifndef _PRIV_H_
#define _PRIV_H_

#define PRIV_DUMP_REGISTER_FLAGS_DEFAULT                0x0
#define PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES            0x1

void priv_dump(const char *params);
void priv_dump_register( const char *params, LwU32 skip_zeroes );
BOOL getManualRegName(LwU32 address, char *szRegisterName, LwU32 nNameLength);

#include "g_priv_hal.h"     // (rmconfig)  public interfaces

#endif // _PRIV_H_
