/*
 *  Copyright 2018-2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>
#include "commandline/commandline.h"

#define MAX_PATH_LEN 256

#ifdef __linux__
#define FM_DEFAULT_CONFIG_FILE_LOCATION "/usr/share/lwpu/lwswitch/fabricmanager.cfg"
#else
#define FM_DEFAULT_CONFIG_FILE_LOCATION "C:\\Program Files (x86)\\"
#endif

typedef enum {
    CMD_PARSE_ST_OK = 0,
    CMD_PARSE_ST_BADPARAM = -1,
    CMD_PARSE_ST_GENERIC_ERROR = -2
} fabricManagerCmdParseReturn_t;

/**
 * Structure to store the properties read from the command-line
 */    

typedef struct {
    void (*pfCmdUsage)(void * pCmdLine);/* Callback to ilwoke Command Usage */
    void (*pfCmdHelp)(void * pCmdLine); /* Callback to ilwoke Command Help */
    void * pCmdLine;                    /* Pointer to command line */
    unsigned int printVersion;         /* Should we print out our version and exit? 1=yes. 0=no. */
    char configFilename[MAX_PATH_LEN];          /* FM config file name/path */
    bool restart;
} fabricManagerCmdParser_t;

/* Enum to represent commandline for Fabric Manager */
typedef enum {
    // This list must start at zero, and be in the same order
    // as the struct all_args list, below:
    FM_CMD_HELP = 0,       /* Shows help output */
    FM_CMD_VERSION,        /* Print out Fabric Manager version and exit */ 
    FM_CMD_CONFIG_FILE,    /* Command line to provide Fabric Manager config file path/name */
    FM_CMD_RESTART,        /* Restart Fabric Manager */
    FM_CMD_COUNT           /* Always keep this last */
} fabricManagerCmdEnum_t;

/*****************************************************************************
 Method to Initialize the command parser
*****************************************************************************/

fabricManagerCmdParser_t*
fabricManagerCmdParserInit(int argc, char** rgv,
                           struct all_args* allArgs,
                           int numberElementsInAllArgs,
                           void (*pfCmdUsage)(void *),
                           void (*pfCmdHelp)(void *));

/*****************************************************************************
 Method to destroy Command Line Parser 
*****************************************************************************/
void
fabricManagerCmdParserDestroy(fabricManagerCmdParser_t* pCmdParser);

/*****************************************************************************
 Method to Parse Command Line
*****************************************************************************/
fabricManagerCmdParseReturn_t
fabricManagerCmdProcessing(fabricManagerCmdParser_t* pCmdParser);
#ifdef __cplusplus
}
#endif

