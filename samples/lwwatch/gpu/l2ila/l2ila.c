/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "l2ila.h"

// helper macros

#define PARSE_NEXT_FIELD_OR_RETURN(data, fp, name, val_ptr, base) \
    do{ \
        if (!GetNextLine(data, fp)) \
        { \
            return 0; \
        } \
        if (!ParseSingle(data, name, val_ptr, base)) \
        { \
            return 0; \
        } \
    } while (0)

#define PARSE_NEXT_ARRAY_OR_RETURN(data, fp, name, val_ptr, count, base) \
    do { \
        if (!GetNextLine(data, fp)) \
        { \
            return 0; \
        } \
        if (!ParseArray(data, name, val_ptr, count, base)) \
        { \
            return 0; \
        } \
    } while (0)



LwU32 CallwlateAddr(L2ILAConfig *config, int ltc, int lts, int shift)
{
    return config->LTCPriBase + ltc * config->LTCPriStride + config->LTSPriInLTCBase 
           + lts * config->LTSPriStride + shift;
}

int IsValidHex(char* in)
{
    return ((*in >= '0') && (*in < '9')) || ((*in >= 'a') && (*in <= 'f')) || ((*in >= 'A') && (*in <= 'F'));
}

// Helper function to get next valid line, will skip empty lines or comment lines
// also colwert the line to lower case
int GetNextLine(char* buf, FILE* fp) 
{
    int skip = 0;
    int i = 0;
    while (fgets(buf, L2ILA_MAX_LINESIZE, fp) != NULL)
    {
        skip = 1;
        for (i = 0; i < L2ILA_MAX_LINESIZE; i++) 
        {
            // skip spaces
            if (buf[i] == ' ') 
            {
                continue;
            }
            // stop processing this line when we see a newline or comment or empty line
            else if ((buf[i] == '\n') || (buf[i] == '\r') || (buf[i] == '#') || (buf[i] == '\0'))
            {
                break;
            }
            // have some valid content
            else 
            {
                skip = 0;
                buf[i] = tolower(buf[i]);
            }
        }
        if (!skip) 
        {
            return 1;
        }
    }
    // if reach here means no valid line till EOF
    return 0;
}

int ParseSingle(char *line, char *name, LwU32 *dst, int base) {
    char* needle;

    // try to find name in line
    needle = strstr(line, name);
    if (needle == NULL) 
    {
        return 0;
    }

    // skip to the value
    needle += strlen(name);
    while (*needle != '\0') 
    {
        if (IsValidHex(needle))
        {
            *dst = (LwU32)strtol(needle, NULL, base);
            return 1;
        }
        needle += 1;
    }
    // no value found if reach here
    return 0;

}

int ParseProjectString(char *line, char *name, char *dst) 
{
    char *needle;

    // try to find name in line
    needle = strstr(line, name);
    if (needle == NULL)
    {
        return 0;
    }

    // skip to the value
    needle += strlen(name);
    while (*needle != '\0')
    {
        if ((*needle != ' ') && (*needle != '=') && (*needle != '\"'))
        {
            strncpy(dst, needle, PROJECT_LENGTH);
            return 1;
        }
        needle += 1;
    }
    // no value found if reach here
    return 0;
}

int ParseArray(char *line, char *name, LwU32 *dst, int *count, int base)
{
    char *needle;
    char *val_end;
    int local_count = 0;

    // try to find name in line
    needle = strstr(line, name);
    if (needle == NULL)
    {
        return 0;
    }

    // skip to the value
    needle += strlen(name);
    while (*needle != '\0')
    {
        if (IsValidHex(needle))
        {
            // first parse this value
            *dst = (LwU32)strtol(needle, &val_end, base);
            // skip parsed value
            needle = val_end;
            // incr local count and dst
            local_count += 1;
            dst += 1;
        }
        needle += 1;
    }
    // always return success if entered while loop
    *count = local_count;
    return 1;
}

int ParseConfigFromFile(L2ILAConfig *config, L2ILAArguments *args)
{
    int i;
    char buf[L2ILA_MAX_LINESIZE];

    // parse project
    config->project[0] = '\0';
    while (GetNextLine(buf, args->inFile))
    {
        if (ParseProjectString(buf, "project", config->project)) 
        {
            config->project[5] = '\0';
            break;
        }
    }
    // check if project is parsed
    if (config->project[0] == '\0') {
        return 0;
    }

    // parse fields
    PARSE_NEXT_ARRAY_OR_RETURN(buf, args->inFile, "config_reg_values", &config->configRegValues[0], &config->configRegValuesSize, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "arm_field_reg_index", &config->armFieldRegIndex, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "arm_field_mask", &config->armFieldMask, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "arm_field_arm_value", &config->armFieldArmValue, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "arm_field_disarm_value", &config->armFieldDisarmValue, 16);
    PARSE_NEXT_ARRAY_OR_RETURN(buf, args->inFile, "status_reg_indices", &(config->statusRegIndices[0]), &config->statusRegIndicesSize, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "sample_cnt_field_reg_index", &config->sampleCntFieldRegIndex, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "sample_cnt_field_mask", &config->sampleCntFieldMask, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "sample_size_field_reg_index", &config->sampleSizeFieldRegIndex, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "sample_size_field_mask", &config->sampleSizeFieldMask, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "sample_size_field_128bit", &config->sampleSizeField128Bit, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "sample_size_field_256bit", &config->sampleSizeField256Bit, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "read_reg_index", &config->readRegIndex, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "ltc_pri_base", &config->LTCPriBase, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "ltc_pri_shared_base", &config->LTCPriSharedBase, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "ltc_pri_stride", &config->LTCPriStride, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "lts_pri_in_ltc_base", &config->LTSPriInLTCBase, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "lts_pri_in_ltc_shared_base", &config->LTSPriInLTCSharedBase, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "lts_pri_stride", &config->LTSPriStride, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "ctrl_addr_shift", &config->ctrlAddrShift, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "aperture_addr_shift", &config->apertureAddrShift, 16);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "ltc_per_chip", &config->LTCPerChip, 10);
    PARSE_NEXT_FIELD_OR_RETURN(buf, args->inFile, "lts_per_ltc", &config->LTSPerLTC, 10);

    // callwlate some other configs
    config->broadcastBaseAddr = config->LTCPriSharedBase + config->LTSPriInLTCSharedBase;
    config->broadcastCtrlAddr = config->broadcastBaseAddr + config->ctrlAddrShift;
    config->broadcastApertureAddr = config->broadcastBaseAddr + config->apertureAddrShift;

    // for debugging purpose print out parsed config
    if (args->verbose) {
        dprintf("==============================================================\n");
        dprintf("project: %s\n", config->project);
        dprintf("configRegValues:\n");
        for (i = 0; i < config->configRegValuesSize; i++) {
            dprintf("0x%08x ", config->configRegValues[i]);
        }
        dprintf("\n");
        dprintf("armFieldRegIndex: 0x%08x\n", config->armFieldRegIndex);
        dprintf("armFieldMask: 0x%08x\n", config->armFieldMask);
        dprintf("armFieldArmValue: 0x%08x\n", config->armFieldArmValue);
        dprintf("armFieldDisarmValue: 0x%08x\n", config->armFieldDisarmValue);
        dprintf("statusRegIndices:\n");
        for (i = 0; i < config->statusRegIndicesSize; i++)
        {
            dprintf("0x%08x ", config->statusRegIndices[i]);
        }
        dprintf("\n");
        dprintf("sampleCntFieldRegIndex: 0x%08x\n", config->sampleCntFieldRegIndex);
        dprintf("sampleCntFieldMask: 0x%08x\n", config->sampleCntFieldMask);
        dprintf("sampleSizeFieldRegIndex: 0x%08x\n", config->sampleSizeFieldRegIndex);
        dprintf("sampleSizeFieldMask: 0x%08x\n", config->sampleSizeFieldMask);
        dprintf("sampleSizeField128Bit: 0x%08x\n", config->sampleSizeField128Bit);
        dprintf("sampleSizeField256Bit: 0x%08x\n", config->sampleSizeField256Bit);
        dprintf("readRegIndex: 0x%08x\n", config->readRegIndex);
        dprintf("LTCPriBase: 0x%08x\n", config->LTCPriBase);
        dprintf("LTCPriSharedBase: 0x%08x\n", config->LTCPriSharedBase);
        dprintf("LTCPriStride: 0x%08x\n", config->LTCPriStride);
        dprintf("LTSPriInLTCBase: 0x%08x\n", config->LTSPriInLTCBase);
        dprintf("LTSPriInLTCSharedBase: 0x%08x\n", config->LTSPriInLTCSharedBase);
        dprintf("LTSPriStride: 0x%08x\n", config->LTSPriStride);
        dprintf("ctrlAddrShift: 0x%08x\n", config->ctrlAddrShift);
        dprintf("apertureAddrShift: 0x%08x\n", config->apertureAddrShift);
        dprintf("LTCPerChip: %d\n", config->LTCPerChip);
        dprintf("LTSPerLTC: %d\n", config->LTSPerLTC);
        dprintf("broadcastBaseAddr: 0x%08x\n", config->broadcastBaseAddr);
        dprintf("broadcastCtrlAddr: 0x%08x\n", config->broadcastCtrlAddr);
        dprintf("broadcastApertureAddr: 0x%08x\n", config->broadcastApertureAddr);
        dprintf("==============================================================\n");
    }
    return 1;
}

int ExelwteL2ILACommands(L2ILAConfig *config, L2ILAArguments *args, L2ILACommand *cmds, int numCmd)
{
    int i;
    for (i = 0; i < numCmd; i++) 
    {
        switch (cmds[i].cmd) 
        {
            case CMD_READ:
                cmds[i].value = GPU_REG_RD32(cmds[i].addr);
                if (args->verbose) 
                {
                    dprintf("R:0x%08x:0x%08x\n", cmds[i].addr, cmds[i].value);
                }
                if (args->keep)
                {
                    fprintf(args->logFile, "R:0x%08x\n", cmds[i].addr);
                }
                break;
            case CMD_WRITE:
                // write ignores mask value
                GPU_REG_WR32(cmds[i].addr, cmds[i].value);
                if (args->verbose)
                {
                    dprintf("W:0x%08x:0x%08x\n", cmds[i].addr, cmds[i].value);
                }
                if (args->keep)
                {
                    fprintf(args->logFile, "W:0x%08x:0x%08x\n", cmds[i].addr, cmds[i].value);
                }
                break;
            case CMD_CHECK:
                cmds[i].readback = GPU_REG_RD32(cmds[i].addr);
                if (args->verbose)
                {
                    dprintf("C:0x%08x:0x%08x:0x%08x\n", cmds[i].addr, cmds[i].value, cmds[i].readback);
                }
                if (args->keep)
                {
                    fprintf(args->logFile, "C:0x%08x:0x%08x\n", cmds[i].addr, cmds[i].value);
                }
                break;
            default: break;
        }
    }
    // always success
    return 1;
}

int VerifyCheckResult(L2ILAArguments *args, L2ILACommand *cmds, int numCmd)
{
    int numFailure = 0;
    int i;
    for (i = 0; i < numCmd; i++) 
    {
        if (cmds[i].cmd == CMD_CHECK) 
        {
            if ((cmds[i].value & cmds[i].mask) != (cmds[i].readback & cmds[i].mask)) {
                numFailure += 1;
                if (args->verbose) {
                    dprintf("MISMATCH! Addr: 0x%08x Expected: 0x%08x Read: 0x%08x\n", cmds[i].addr,
                            cmds[i].value, cmds[i].readback);
                }
            }
        }
    }
    return numFailure;
}

int CheckAccess(L2ILAConfig *config, L2ILAArguments *args)
{
    int numFailures = 0;
    L2ILACommand accessCmds[6] = {
        {0x0, CMD_WRITE, 0x0, 0xFFFFFFFF, 0x0},
        {0x0, CMD_CHECK, 0x0, 0xFFFFFFFF, 0x0},
        {0x0, CMD_WRITE, 0x00030001, 0xFFFFFFFF, 0x0},
        {0x0, CMD_CHECK, 0x00030001, 0xFFFFFFFF, 0x0},
        {0x0, CMD_READ, 0x0, 0xFFFFFFFF, 0x0},
        {0x0, CMD_CHECK, 0x00030002, 0xFFFFFFFF, 0x0}};

    // TODO: print msg
    accessCmds[0].addr = config->broadcastCtrlAddr;
    accessCmds[1].addr = config->broadcastCtrlAddr;
    accessCmds[2].addr = config->broadcastCtrlAddr;
    accessCmds[3].addr = config->broadcastCtrlAddr;
    accessCmds[4].addr = config->broadcastApertureAddr;
    accessCmds[5].addr = config->broadcastCtrlAddr;
    ExelwteL2ILACommands(config, args, accessCmds, 6);
    numFailures = VerifyCheckResult(args, accessCmds, 6);
    return numFailures;
}

int VerifyConfig(L2ILAConfig *config, L2ILAArguments *args)
{
    int size = sizeof(L2ILACommand) * (config->configRegValuesSize + 1);
    int numFailure = 0;
    int i;
    L2ILACommand *cmds = (L2ILACommand *)malloc(size);
    memset(cmds, 0, size);
    cmds[0].addr = config->broadcastCtrlAddr;
    cmds[0].cmd = CMD_WRITE;
    cmds[0].value = 0x10000;
    for (i = 0; i < config->configRegValuesSize; i++) 
    {
        cmds[i + 1].addr = config->broadcastApertureAddr;
        cmds[i + 1].cmd = CMD_CHECK;
        cmds[i + 1].value = config->configRegValues[i];
        cmds[i + 1].mask = 0xFFFFFFFF;
    }
    ExelwteL2ILACommands(config, args, cmds, config->configRegValuesSize + 1);
    numFailure = VerifyCheckResult(args, cmds, config->configRegValuesSize + 1);
    free(cmds);
    return numFailure;
}

int VerifyArmRegister(L2ILAConfig *config, L2ILAArguments *args, int value)
{
    int numFailure = 0;
    L2ILACommand cmds[2] = {
        {0x0, CMD_WRITE, 0x0, 0xFFFFFFFF, 0x0},
        {0x0, CMD_CHECK, 0x0, 0xFFFFFFFF, 0x0}};
    cmds[0].addr = config->broadcastCtrlAddr;
    cmds[0].value = config->armFieldRegIndex;
    cmds[1].addr = config->broadcastApertureAddr;
    cmds[1].value = value;
    cmds[1].mask = config->armFieldMask;
    ExelwteL2ILACommands(config, args, cmds, 2);
    numFailure = VerifyCheckResult(args, cmds, 2);
    return numFailure;
}

int ExelwteConfig(L2ILAConfig *config, L2ILAArguments *args)
{
    int size = sizeof(L2ILACommand) * (config->configRegValuesSize + 1);
    int numFailure = 0;
    int i;
    L2ILACommand* cmds = (L2ILACommand*)malloc(size);
    dprintf("=======================================================================\n");
    dprintf("Verifying L2-iLA is disarmed...\n");
    if (VerifyArmRegister(config, args, config->armFieldDisarmValue)) {
        dprintf("ERROR: L2 - iLA is lwrrently ARMED.Please disarm it before config.\n");
        return 0;
    }
    dprintf("Verifying L2-iLA is disarmed passed.\n");
    dprintf("=======================================================================\n");
    dprintf("Programing config values to hardware...\n");
    memset(cmds, 0, size);
    cmds[0].addr = config->broadcastCtrlAddr;
    cmds[0].cmd = CMD_WRITE;
    cmds[0].value = 0x20000;
    for (i = 0; i < config->configRegValuesSize; i++)
    {
        cmds[i + 1].addr = config->broadcastApertureAddr;
        cmds[i + 1].cmd = CMD_WRITE;
        cmds[i + 1].value = config->configRegValues[i];
        cmds[i + 1].mask = 0xFFFFFFFF;
    }
    ExelwteL2ILACommands(config, args, cmds, config->configRegValuesSize + 1);
    free(cmds);
    dprintf("Programing config values to hardware done.\n");

    dprintf("=======================================================================\n");
    dprintf("Verifying hardware config values...\n");
    if (VerifyConfig(config, args)) {
        dprintf("ERROR: Config values were not properly programmed to hardware.\n");
        return 0;
    }
    dprintf("Verifying hardware config values passed.\n");
    return 1;
}

int ExelwteArm(L2ILAConfig *config, L2ILAArguments *args)
{
    L2ILACommand cmds[2] = {
        {0x0, CMD_WRITE, 0x0, 0xFFFFFFFF, 0x0},
        {0x0, CMD_WRITE, 0x0, 0xFFFFFFFF, 0x0}};
    dprintf("=======================================================================\n");
    dprintf("Verifying L2-iLA is disarmed...\n");
    if (VerifyArmRegister(config, args, config->armFieldDisarmValue))
    {
        dprintf("ERROR: L2-iLA is lwrrently ARMED. Please disarm it before arming again.\n");
        return 0;
    }
    dprintf("Verifying L2-iLA is disarmed passed.\n");

    dprintf("=======================================================================\n");
    dprintf("Verifying hardware config values...\n");
    if (VerifyConfig(config, args)) {
        dprintf("ERROR: L2-iLA is lwrrently config to a different setting. Please config L2-iLA before arming it.\n");
        return 0;
    }
    dprintf("Verifying hardware config values passed.\n");

    dprintf("=======================================================================\n");
    dprintf("Arming L2-iLA on all L2 slices...\n");
    cmds[0].addr = config->broadcastCtrlAddr;
    cmds[0].value = config->armFieldRegIndex;
    cmds[1].addr = config->broadcastApertureAddr;
    cmds[1].value = config->armFieldArmValue;
    ExelwteL2ILACommands(config, args, cmds, 2);
    dprintf("Arming L2-iLA on all L2 slices done.\n");

    dprintf("=======================================================================\n");
    dprintf("Verifying L2-iLA is armed...\n");
    if (VerifyArmRegister(config, args, config->armFieldArmValue))
    {
        dprintf("ERROR: L2-iLA is NOT armed. Something went wrong.\n");
        return 0;
    }
    dprintf("Verifying L2-iLA is armed passed.\n");
    return 1;
}

int ExelwteDisarm(L2ILAConfig *config, L2ILAArguments *args)
{
    L2ILACommand cmds[2] = {
        {0x0, CMD_WRITE, 0x0, 0xFFFFFFFF, 0x0},
        {0x0, CMD_WRITE, 0x0, 0xFFFFFFFF, 0x0}};
    dprintf("=======================================================================\n");
    dprintf("Verifying L2-iLA is armed...\n");
    if (VerifyArmRegister(config, args, config->armFieldArmValue))
    {
        dprintf("ERROR: L2-iLA is lwrrently already DISARMED.\n");
        return 0;
    }
    dprintf("Verifying L2-iLA is armed passed.\n");

    dprintf("=======================================================================\n");
    dprintf("Disarming L2-iLA on all L2 slices...\n");
    cmds[0].addr = config->broadcastCtrlAddr;
    cmds[0].value = config->armFieldRegIndex;
    cmds[1].addr = config->broadcastApertureAddr;
    cmds[1].value = config->armFieldDisarmValue;
    ExelwteL2ILACommands(config, args, cmds, 2);
    dprintf("Disarming L2-iLA on all L2 slices done.\n");

    dprintf("=======================================================================\n");
    dprintf("Verifying L2-iLA is disarmed...\n");
    if (VerifyArmRegister(config, args, config->armFieldDisarmValue))
    {
        dprintf("ERROR: L2-iLA is still ARMED. Something went wrong.\n");
        return 0;
    }
    dprintf("Verifying L2-iLA is disarmed passed.\n");
    return 1;
}

int ExelwteStatus(L2ILAConfig *config, L2ILAArguments *args)
{
    int idx;
    LwU32 ltc, lts;
    L2ILACommand cmds[2] = {
        {0x0, CMD_WRITE, 0x0, 0xFFFFFFFF, 0x0},
        {0x0, CMD_READ, 0x0, 0xFFFFFFFF, 0x0}};
    dprintf("=======================================================================\n");
    dprintf("Dumping out all L2 slice status\n");
    // loop through all LTS in all LTC
    for (ltc = 0; ltc < config->LTCPerChip; ltc++) 
    {
        for (lts = 0; lts < config->LTSPerLTC; lts++) 
        {
            LwU32 ctrl_addr = CallwlateAddr(config, ltc, lts, config->ctrlAddrShift);
            LwU32 aperture_addr = CallwlateAddr(config, ltc, lts, config->apertureAddrShift);
            for (idx = 0; idx < config->statusRegIndicesSize; idx++) 
            {
                cmds[0].addr = ctrl_addr;
                cmds[0].value = config->statusRegIndices[idx];
                cmds[1].addr = aperture_addr;
                ExelwteL2ILACommands(config, args, cmds, 2);
                dprintf("LTC%02u LTS%u: {0x%08X :m 0x%08X}\n", ltc, lts, cmds[1].addr, cmds[1].value);
            }
        }
    }
    dprintf("Dump completed\n");
    return 1;
}

int ExelwteCapture(L2ILAConfig *config, L2ILAArguments *args)
{
    L2ILACommand cmd_status[2] = {
        {0x0, CMD_WRITE, 0x0, 0xFFFFFFFF, 0x0},
        {0x0, CMD_READ, 0x0, 0xFFFFFFFF, 0x0}};
    
    L2ILACommand cmd_sample[4] = {
        {0x0, CMD_WRITE, 0x0, 0xFFFFFFFF, 0x0},
        {0x0, CMD_READ, 0x0, 0xFFFFFFFF, 0x0},
        {0x0, CMD_WRITE, 0x0, 0xFFFFFFFF, 0x0},
        {0x0, CMD_READ, 0x0, 0xFFFFFFFF, 0x0}};

    L2ILACommand *cmd_read;
    int num_words, sample_cnt, words_per_sample, i, idx;
    LwU32 ltc, lts, ctrl_addr, aperture_addr;
    char *last;

    dprintf("=======================================================================\n");
    dprintf("Verifying L2-iLA is disarmed...\n");
    if (VerifyArmRegister(config, args, config->armFieldDisarmValue))
    {
        dprintf("ERROR: L2-iLA is lwrrently ARMED. Please disarm it before trying to caputure data.\n");
        return 0;
    }
    dprintf("Verifying L2-iLA is disarmed passed.\n");

    dprintf("=======================================================================\n");
    dprintf("Verifying hardware config values...\n");
    if (VerifyConfig(config, args))
    {
        dprintf("ERROR: L2-iLA is lwrrently config to a different setting. Please config L2-iLA before arming or capturing data.\n");
        return 0;
    }
    dprintf("Verifying hardware config values passed.\n");

    dprintf("=======================================================================\n");
    dprintf("Capturing data to %s ...\n", args->outFname);

    // print header
    fprintf(args->outFile, "{\n  \"config\": {\n    \"project\": \"%s\",\n    \"config_reg_values\": [\n", config->project);
    for (i = 0; i < config->configRegValuesSize; i++)
    {
        last = (i == (config->configRegValuesSize - 1)) ? "" : ",";
        fprintf(args->outFile, "      \"0x%08x\"%s\n", config->configRegValues[i], last);
    }
    fprintf(args->outFile, "    ],\n    \"status_reg_indices\": [\n");
    for (i = 0; i < config->statusRegIndicesSize; i++) 
    {
        last = (i == (config->statusRegIndicesSize - 1)) ? "" : ",";
        fprintf(args->outFile, "      \"0x%x\"%s\n", config->statusRegIndices[i], last);
    }
    fprintf(args->outFile, "    ],\n    \"LTC_PER_CHIP\": %u,\n", config->LTCPerChip);
    fprintf(args->outFile, "    \"LTS_PER_LTC\": %u\n  },\n", config->LTSPerLTC);
    fprintf(args->outFile, "  \"data\": {\n");

    // main read loop
    for (ltc = 0; ltc < config->LTCPerChip; ltc++)
    {
        fprintf(args->outFile, "    \"LTC%u\": {\n", ltc);
        for (lts = 0; lts < config->LTSPerLTC; lts++) 
        {
            // LTS header
            fprintf(args->outFile, "      \"LTS%u\": {\n        \"status\": [\n", lts);

            ctrl_addr = CallwlateAddr(config, ltc, lts, config->ctrlAddrShift);
            aperture_addr = CallwlateAddr(config, ltc, lts, config->apertureAddrShift);

            // LTS status
            for (idx = 0; idx < config->statusRegIndicesSize; idx++)
            {
                cmd_status[0].addr = ctrl_addr;
                cmd_status[0].value = config->statusRegIndices[idx];
                cmd_status[1].addr = aperture_addr;
                ExelwteL2ILACommands(config, args, cmd_status, 2);
                fprintf(args->outFile, "        \"0x%08x\"\n", cmd_status[1].value);
                dprintf("LTC%02u LTS%u: {0x%08X :m 0x%08X}\n", ltc, lts, cmd_status[1].addr, cmd_status[1].value);
            }
            fprintf(args->outFile, "      ],\n");

            // Read maximum number of sample count and sample count
            cmd_sample[0].addr = ctrl_addr;
            cmd_sample[0].value = config->sampleCntFieldRegIndex;
            cmd_sample[1].addr = aperture_addr;
            cmd_sample[2].addr = ctrl_addr;
            cmd_sample[2].value = config->sampleSizeFieldRegIndex;
            cmd_sample[3].addr = aperture_addr;
            ExelwteL2ILACommands(config, args, cmd_sample, 4);
            // check if sample cnt is valid
            if ((cmd_sample[1].value & 0xFFFF0000) == 0xBADF0000)
            {
                sample_cnt = 0;
                words_per_sample = 0;
            }
            else
            {
                sample_cnt = cmd_sample[1].value & config->sampleCntFieldMask;
                if ((cmd_sample[3].value & config->sampleSizeField256Bit) == config->sampleSizeField256Bit)
                {
                    words_per_sample = 256 / 32;
                }
                else
                {
                    words_per_sample = 128 / 32;
                }
                        }
            
            num_words = sample_cnt * words_per_sample;
            dprintf( "    sample_cnt: %u, words_per_sample: %u,\n", sample_cnt, words_per_sample);
            fprintf(args->outFile, "      \"sample_cnt\": %u,\n      \"words_per_sample\": %u,\n",
                    sample_cnt, words_per_sample);

            // Read data
            cmd_read = (L2ILACommand*)malloc(sizeof(L2ILACommand) * (num_words + 2));
            memset(cmd_read, 0, sizeof(L2ILACommand) * (num_words + 2));
            cmd_read[0].addr = ctrl_addr;
            cmd_read[0].value = config->readRegIndex;
            cmd_read[0].cmd = CMD_WRITE;
            cmd_read[1].addr = aperture_addr;
            cmd_read[1].cmd = CMD_WRITE;
            for (i = 0; i < num_words; i++) {
                cmd_read[i + 2].addr = aperture_addr;
                cmd_read[i + 2].cmd = CMD_READ;
            }
            ExelwteL2ILACommands(config, args, cmd_read, num_words + 2);
            fprintf(args->outFile, "        \"data\": [\n");
            for (i = 0; i < num_words; i++) {
                last = (i == (num_words - 1)) ? "" : ",";
                fprintf(args->outFile, "          \"0x%08x\"%s\n", cmd_read[i + 2].value, last);
            }
            last = (lts == (config->LTSPerLTC - 1)) ? "" : ",";
            fprintf(args->outFile, "        ]\n      }%s\n", last);
        }
        last = (ltc == (config->LTCPerChip - 1)) ? "" : ",";
        fprintf(args->outFile, "    }%s\n", last);
    }
    fprintf(args->outFile, "  }\n}\n");

    dprintf("Capture completed!\n");

    return 1;
}

void L2ILACleanup(L2ILAArguments *args) {
    if (args->inFile) 
    {
        fclose(args->inFile);
    }
    if (args->outFile) 
    {
        fclose(args->outFile);
    }
    if (args->logFile)
    {
        fclose(args->logFile);
    }
}
