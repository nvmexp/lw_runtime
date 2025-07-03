#pragma once

#include <stdio.h>
#ifdef __linux__
#include "GlobalFabricManager.h"
#include "LocalFabricManager.h"
#endif
#include "errno.h"
#include "fm_cmd_parser.h"
#include "FMVersion.h"

#define TIMETOSLEEP 100000      //this is in microseconds

typedef struct {
    int stopLoop;
    fabricManagerCmdParser_t *pCmdParser;
//TODO: temporary purpose. Will be removed when gfm and lfm changes are uploaded
#ifdef __linux__
    LocalFabricManagerControl *pLocalFM;
    GlobalFabricManager       *pGlobalFM;
#endif
} fmCommonCtxInfo_t;

static struct all_args fabricManagerArgs[] = {
        {
                FM_CMD_HELP,
                "-h",
                "--help",
                "\t\tDisplays help information.",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                FM_CMD_VERSION,
                "-v",
                "--version",
                "\t\tDisplays the Fabric Manager version and exit.",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                FM_CMD_CONFIG_FILE,
                "-c",
                "--config",
                "\t\tProvides Fabric Manager config file path/name which controls all the config options.",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                FM_CMD_RESTART,
                "-r",
                "--restart",
                "\t\tRestart Fabric Manager after exit in Shared LWSwitch or vGPU multi-tenancy mode.",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
};

void displayFabricManagerUsage(void* pCmdLine);
void displayFabricManagerHelpMsg(void* pCmdLine);
void displayFabricManagerVersionInfo(void);
void dumpLwrrentConfigOptions(void);
//TODO: temporary purpose. Will be removed when gfm and lfm changes are uploaded
#ifdef __linux__
int fmCommonCleanup(int status, fmCommonCtxInfo_t *gFmCommonCtxInfo);
int enableLocalFM(fmCommonCtxInfo_t &gFmCommonCtxInfo); 
int enableGlobalFM(fmCommonCtxInfo_t &gFmCommonCtxInfo); 
#endif
