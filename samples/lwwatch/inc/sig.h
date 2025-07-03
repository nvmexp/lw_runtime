/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2006-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// gr.h
//
//*****************************************************

#ifndef _SIG_H_
#define _SIG_H_

#include "os.h"
#include "hal.h"

//#define SIGDUMP_ENABLE
#define COMMON_DUMP 0
#define XBAR_DUMP   1

typedef struct
{
    const char* instanceName;
    LwU32 chipletLimit;
    LwU32 instanceLimit;
    BOOL  bValid;
}InstanceInfo;

// NOTE: When adding a new instance; please also update the 
// string array "instanceNames" in sigdump.c. Both should match order wise.
typedef enum
{
    gpc,
    gpc_tpc,
    fbp,
    sys,
    sys_mxbar_cq_daisy,
    sys_wxbar_cq_daisy,
    sys_mxbar_cq_daisy_unrolled,
    gpc_dtpc,   //added in gf117
    gpc_ppc,
    sys_mxbar_cs_daisy,
    sys_wxbar_cs_daisy,
    NUMINSTANCES
}InstanceClass;

#define CHIPLET_NAME_MAX_LENGTH     64
#define DOMAIN_NAME_MAX_LENGTH      64
#define DOMAIN_NUM_MAX_LENGTH       4
#define SIG_STR_MAX                 256

typedef struct
{
    LwU32 addr;
    LwU32 value;
    LwU32 mask;
} reg_write_GK104_t;

typedef struct
{
    char str[SIG_STR_MAX];
    LwU32 addr;
    LwU32 lsb;
    LwU32 msb;
    LwU32 num_writes;
    LwU32 instanceClass;
    LwU32 chiplet;
    LwU32 instance;
} sigdump_GK104_t;


// Generic functions
void RegWrite(LwU32, LwU32, LwU32);
LwU32 RegBitRead(LwU32, LwU32);
LwBool RegRead(LwU32 *, LwU32, LwBool);
LwBool RegBitfieldRead(LwU32 *, LwU32, LwU32, LwU32, LwBool);
void OutputSignal(FILE *, char *, LwU32);
const char* sigGetInstanceName(LwU32 instance);

#include "g_sig_hal.h"     // (rmconfig)  public interfaces


#endif
