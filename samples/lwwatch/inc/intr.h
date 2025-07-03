/* Copyright 2019-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 *_LWRM_COPYRIGHT_END_
 */

#ifndef __INTR_H
#define __INTR_H

#include "fifo.h"
#include "lwtypes.h"
#include "lwwatch.h"

#include "g_intr_hal.h"

#define INTR_NAME_MAXLEN 50     // maximum length of interrupt name
#define REG_NUM_BITS 32         // number of bits per register
#define INTR_MAX_PHYSICAL 4096  // maximum number of interrupts per gfid allowed by hardware
#define MAX_GFIDS 64            // maximum number of gfids
#define INTR_ILWALID_GFID -1
#define INTR_DEFAULT_GFID 0
#define LW_CTRL_LEAF_REG_PER_TOP_BIT 2

typedef struct intr
{
    LwU32 gpuIV;         // interrupt vector number
    LwBool bPulse;       // pulse or level based
    char *name;          // name of unit / engine

    LwU32 leafReg;       // index of leaf register
    LwU32 leafBit;       // index of leaf register bit

    LwU32 topReg;        // index of top register
    LwU32 topBit;        // index of top register bit

    LwBool bGspTree;     // whether or not interrupt is in gsp tree
    LwBool bVirtualized; // virtual function

    LwU32 retriggerAddress;
    char *retriggerName;

    LwBool bTag;         // If interrupt entry is in use
    LwBool bDiscovery;   // Indicates if interrupt is nonstatic.
                         // If disc is on, then after reading, turn tag off

    LwBool bFound;       // Indicates if interrupt is pending

    struct intr *next;  // pointer to intr_type with same gpuIV
} intr_type;

// We use intr_disc_type to populate intr_type before printing
typedef struct
{
    LwU32 ctrlReg;                          // control register address
    char name[INTR_NAME_MAXLEN];            // name of unit / engine

    LwU32 retriggerAddress;
    char retriggerName[INTR_NAME_MAXLEN];
} intr_disc_type;

typedef struct
{
    LwU32 vector;
    LwBool bPulse;
    char *name;
    LwBool bGspTree;
    LwBool bVirtualized;
    LwU32 retriggerAddress;
    char *retriggerName;
} intr_info;

typedef struct
{
    char  name[INTR_NAME_MAXLEN];               // i.e. "ce"
    LwU32 ctrlReg;                              // LW_..._INTR_CTRL
    char  ctrlRegName[INTR_NAME_MAXLEN];        // "LW_..._INTR_CTRL"
    LwU32 retriggerReg;                         // LW_..._INTR_RETRIGGER
    char  retriggerName[INTR_NAME_MAXLEN];      // "LW_..._INTR_RETRIGGER"
    LwU32 ctrlNotificationReg;                  // LW_...NOTIFY_INTR_CTRL or LW_..._INTR_CTRL(1)
    char  ctrlNotificationRegName[INTR_NAME_MAXLEN]; // "LW_...NOTIFY_INTR_CTRL" or "LW_..._INTR_CTRL(1)"
    /* Retriggers are not needed for notification interrupts */
    LwU32 stride;                               // Stride of the engine PRIs
                                                // Hack since we're not using pointers to the objects themselves
    LwU32 numArgs;                              // Number of args in register indexing. Valid values are 0, 1, 2
    LwU32 count;                                // Number of entries in deviceinfo that match this engine type
    LwBool bValid;                              // Whether or not engine has an associated interrupt
} engine_disc_info;

typedef struct
{
    intr_type interrupts[INTR_MAX_PHYSICAL];
    intr_disc_type intrDisc[INTR_MAX_PHYSICAL];
    engine_disc_info discInfo[ENGINE_TAG_ILWALID];

    LwU32 intrCtr;
    LwU32 discCtr;

    LwBool bInterruptTableInit;
} enumeration_table;

enumeration_table intrEnumTable;

void intrPrintHelp(void);
void intrRegisterDiscWithRetrigger(LwU32 ctrlReg, char *name, LwU32 retrigger,
                      char *retriggerName);
LwBool intrUpdateDisc(intr_disc_type *disc);
LwBool intrUpdateAll(void);
void addEngineInfo(LwU32 tag, char *name, LwU32 ctrlReg, char *ctrlRegName,
                   LwU32 retriggerReg, char *retriggerName, LwU32 ctrlNotificationReg,
                   char *ctrlNotificationRegName, LwU32 stride, LwU32 numArgs, LwBool bValid);
void intrEnum(void);
void intrEnumPending(LwU32 gfid, LwBool bGsp);

#endif
