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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fm_cmd_parser.h"

/*****************************************************************************
 Method to Initialize the command parser
 *****************************************************************************/

fabricManagerCmdParser_t*
fabricManagerCmdParserInit(int argc, char** argv,
                           struct all_args* pAllArgs,
                           int numberElementsInAllArgs,
                           void (*pfCmdUsage)(void *),
                           void (*pfCmdHelp)(void *))
{
    fabricManagerCmdParser_t* pCmdParser;
    int cmdLineResult = 0;
    if ((NULL == pAllArgs) || (NULL == pfCmdUsage) || (NULL == pfCmdHelp)) {
        return NULL;
    }
    pCmdParser = (fabricManagerCmdParser_t *)malloc(sizeof (*pCmdParser));
    if (NULL == pCmdParser) {
        return NULL;
    }
    // memset to 0
    memset(pCmdParser, 0, sizeof (*pCmdParser));
    // set all default arguments
    // default config file path
    strncpy(pCmdParser->configFilename, 
            FM_DEFAULT_CONFIG_FILE_LOCATION, sizeof(pCmdParser->configFilename)-1);
    // store the callbacks
    pCmdParser->pfCmdUsage = pfCmdUsage;
    pCmdParser->pfCmdHelp = pfCmdHelp;
    cmdLineResult = cmdline_init(argc, argv, pAllArgs, numberElementsInAllArgs, &pCmdParser->pCmdLine);
    if (0 != cmdLineResult) {
        free(pCmdParser);
        pCmdParser = NULL;
        return NULL;
    }
    return pCmdParser;
}


/*****************************************************************************
 Method to destroy Command Line Parser
 *****************************************************************************/
void
fabricManagerCmdParserDestroy(fabricManagerCmdParser_t* pCmdParser)
{
    if (NULL != pCmdParser) {
        if (NULL != pCmdParser->pCmdLine) {
            cmdline_destroy(pCmdParser->pCmdLine);
        }
        free(pCmdParser);
    }
}

/*****************************************************************************
 Internal Method to verify basic correctness for command line
 *****************************************************************************/
static fabricManagerCmdParseReturn_t
fabricManagerVerifyOptions(fabricManagerCmdParser_t* pCmdParser)
{
    int cmdLineResult;
    char * firstUnknownOptionName = NULL, *firstArgWithMissingOption = NULL;
    cmdLineResult = cmdline_checkOptions(pCmdParser->pCmdLine,
            (const char **) &firstUnknownOptionName,
            (const char **) &firstArgWithMissingOption);
    switch (cmdLineResult) {
        case CMDLINE_CHECK_OPTIONS_UNKNOWN_OPTION_FOUND:
            printf("Option \"%s\" is not recognized.\n", firstUnknownOptionName);
            return CMD_PARSE_ST_GENERIC_ERROR;
        case CMDLINE_CHECK_OPTIONS_MISSING_REQUIRED_VALUE:
            printf( "Option \"%s\" is missing its value.\n", firstArgWithMissingOption);
            return CMD_PARSE_ST_GENERIC_ERROR;
        case CMDLINE_CHECK_OPTIONS_SUCCESS:
            break;
    }
    return CMD_PARSE_ST_OK;
}

/*****************************************************************************
 Method to Parse Command Line
 *****************************************************************************/
fabricManagerCmdParseReturn_t
fabricManagerCmdProcessing(fabricManagerCmdParser_t* pCmdParser)
{
    fabricManagerCmdParseReturn_t cmdParseRet;
    if (NULL == pCmdParser) {
        return CMD_PARSE_ST_BADPARAM;
    }
    // check if there is a help option specified for the command line
    if (cmdline_exists(pCmdParser->pCmdLine, FM_CMD_HELP)) {
        pCmdParser->pfCmdHelp(pCmdParser->pCmdLine);
        return CMD_PARSE_ST_OK;
    }
    // check if version is specified
    if (cmdline_exists(pCmdParser->pCmdLine, FM_CMD_VERSION)) {
        pCmdParser->printVersion = 1;
        return CMD_PARSE_ST_OK;
    }
    // check if config file path is specified
    if (cmdline_exists(pCmdParser->pCmdLine, FM_CMD_CONFIG_FILE)) {
        char *fileName = NULL;
        if (cmdline_getStringVal(pCmdParser->pCmdLine, FM_CMD_CONFIG_FILE, (const char **) &fileName)) {
            strncpy(pCmdParser->configFilename, fileName, sizeof(pCmdParser->configFilename)-1);
        }
    }
    // check if restart is specified
    if (cmdline_exists(pCmdParser->pCmdLine, FM_CMD_RESTART)) {
        pCmdParser->restart = true;
        return CMD_PARSE_ST_OK;
    }
    // do basic sanity
    cmdParseRet = fabricManagerVerifyOptions(pCmdParser);
    if (CMD_PARSE_ST_OK != cmdParseRet) {
        return CMD_PARSE_ST_GENERIC_ERROR;
    }
    return CMD_PARSE_ST_OK;
}

