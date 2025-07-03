/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch l2ila Plugin
// xiaohaow@lwpu.com - 11.25.2020
// l2ila.h
//
//*****************************************************

// ****************************************************
//
// Plugin for l2ila: Level 2 cache internal logic analyzer
//
// l2ila is used to assist post silicon debugging by providing the ability to monitor l2 traffic, sample
// request information or generate global trigger.
//
// Confluence page for l2ila: https://confluence.lwpu.com/display/GPUPSIDBG/L2-iLA+Config+Tool+V2
//
// *****************************************************

#ifndef _L2ILA_H_
#define _L2ILA_H_

#include "os.h"
#include "hal.h"

#define COMMAND_UNDEFINED   0
#define COMMAND_CONFIG      1
#define COMMAND_ARM         2
#define COMMAND_DISARM      3
#define COMMAND_STATUS      4
#define COMMAND_CAPTURE     5

#define CMD_READ            0
#define CMD_WRITE           1
#define CMD_CHECK           2

#define L2ILA_MAX_LINESIZE  1024
#define PROJECT_LENGTH      5

typedef struct
{
    // Parse from input script
    char project[6];
    LwU32 configRegValues[128];
    int configRegValuesSize;
    LwU32 armFieldRegIndex;
    LwU32 armFieldMask;
    LwU32 armFieldArmValue;
    LwU32 armFieldDisarmValue;
    LwU32 statusRegIndices[20];
    int statusRegIndicesSize;
    LwU32 sampleCntFieldRegIndex;
    LwU32 sampleCntFieldMask;
    LwU32 sampleSizeFieldRegIndex;
    LwU32 sampleSizeFieldMask;
    LwU32 sampleSizeField128Bit;
    LwU32 sampleSizeField256Bit;
    LwU32 readRegIndex;
    LwU32 LTCPriBase;
    LwU32 LTCPriSharedBase;
    LwU32 LTCPriStride;
    LwU32 LTSPriInLTCBase;
    LwU32 LTSPriInLTCSharedBase;
    LwU32 LTSPriStride;
    LwU32 ctrlAddrShift;
    LwU32 apertureAddrShift;
    LwU32 LTCPerChip;
    LwU32 LTSPerLTC;
    // callwlated from values above
    LwU32 broadcastBaseAddr;
    LwU32 broadcastCtrlAddr;
    LwU32 broadcastApertureAddr;
} L2ILAConfig;

typedef struct
{
    int verbose;
    int keep;
    int command;
    char inFname[128];
    char outFname[128];
    char logFname[128];
    FILE *inFile;
    FILE *outFile;
    FILE *logFile;
} L2ILAArguments;

typedef struct
{
    LwU32 addr;
    int cmd;
    LwU32 value;
    LwU32 mask;
    LwU32 readback;
} L2ILACommand;

// Helper functions
int ParseConfigFromFile(L2ILAConfig *config, L2ILAArguments *args);
int ExelwteConfig(L2ILAConfig *config, L2ILAArguments *args);
int ExelwteArm(L2ILAConfig *config, L2ILAArguments *args);
int ExelwteDisarm(L2ILAConfig *config, L2ILAArguments *args);
int ExelwteStatus(L2ILAConfig *config, L2ILAArguments *args);
int ExelwteCapture(L2ILAConfig *config, L2ILAArguments *args);
void L2ILACleanup(L2ILAArguments *args);

#endif
