 /************************ BEGIN COPYRIGHT NOTICE ***************************\
|*                                                                           *|
|* Copyright 2003-2015 by LWPU Corporation.  All rights reserved.  All     *|
|* information contained herein is proprietary and confidential to LWPU    *|
|* Corporation.  Any use, reproduction, or disclosure without the written    *|
|* permission of LWPU Corporation is prohibited.                           *|
|*                                                                           *|
 \************************** END COPYRIGHT NOTICE ***************************/

#if !defined(_H_LWCYCLESTATSCONSOLE_H)
#define _H_LWCYCLESTATSCONSOLE_H

// exports
#define CONSOLE_DEVICE_HANDLE_ID                 0xbeef0000
#define CONSOLE_MAPPING_ID                       0xbeef0001

extern LwU32 g_hClient;
extern void *g_bar0;
extern LwU32 g_accessBAR0ViaRM;

extern void log(const char *format, ...);

typedef unsigned __int64 uint64;

// Macros to make lw32.h behave more like we want it to be
#include "lwmisc.h"

#define REG_WR_DRF_NUM(d,r,f,n) REG_WR32(LW ## d ## r, DRF_NUM(d,r,f,n))
#define REG_WR_DRF_DEF(d,r,f,c) REG_WR32(LW ## d ## r, DRF_DEF(d,r,f,c))
#define FLD_WR_DRF_NUM(d,r,f,n) REG_WR32(LW##d##r,(REG_RD32(LW##d##r)&~(DRF_MASK(LW##d##r##f)<<DRF_SHIFT(LW##d##r##f)))|DRF_NUM(d,r,f,n))
#define FLD_WR_DRF_DEF(d,r,f,c) REG_WR32(LW##d##r,(REG_RD32(LW##d##r)&~(DRF_MASK(LW##d##r##f)<<DRF_SHIFT(LW##d##r##f)))|DRF_DEF(d,r,f,c))
#define REG_RD_DRF(d,r,f)       (((REG_RD32(LW ## d ## r))>>DRF_SHIFT(LW ## d ## r ## f))&DRF_MASK(LW ## d ## r ## f))
#define VAR_WR32(var,value)     var = (value)
#define VAR_RD32(var)           (var)
#define VAR_FLD_WR_DRF_NUM(v,d,r,f,n) VAR_WR32(v,(VAR_RD32(v)&~(DRF_MASK(LW##d##r##f)<<DRF_SHIFT(LW##d##r##f)))|DRF_NUM(d,r,f,n))
#define VAR_FLD_WR_DRF_DEF(v,d,r,f,c) VAR_WR32(v,(VAR_RD32(v)&~(DRF_MASK(LW##d##r##f)<<DRF_SHIFT(LW##d##r##f)))|DRF_DEF(d,r,f,c))
#define VAR_RD_DRF(v,d,r,f)       (((VAR_RD32(v))>>DRF_SHIFT(LW ## d ## r ## f))&DRF_MASK(LW ## d ## r ## f))
#define SHIFTBIT(MASK)                  (0 ? MASK)

#define HWCONST(d, r, f, c)     DRF_DEF( d, _ ## r, _ ## f, _ ## c)
#define HWVALUE(d, r, f, v)     DRF_NUM( d, _ ## r, _ ## f, v )

#endif // defined(_H_LWCYCLESTATSCONSOLE_H)
