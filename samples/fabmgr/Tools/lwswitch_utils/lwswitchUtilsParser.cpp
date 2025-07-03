#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include "lwswitchUtilsParser.h"

/*****************************************************************************
 Method to Initialize the command parser
 *****************************************************************************/
LwswitchUtilsCmdParser_t * lwswitchUtilsCmdParserInit(int argc, char **argv,
        struct all_args * pAllArgs,
        int numberElementsInAllArgs,
        void (*pfCmdUsage)(void *),
        void (*pfCmdHelp)(void *))
{
    LwswitchUtilsCmdParser_t *pCmdParser;
    int cmdLineResult = 0;

    if ((NULL == pAllArgs) || (NULL == pfCmdUsage) || (NULL == pfCmdHelp)) {
        return NULL;
    }

    pCmdParser = (LwswitchUtilsCmdParser_t *)malloc(sizeof (*pCmdParser));
    if (NULL == pCmdParser) {
        return NULL;
    }

    /* memset to 0 */
    memset(pCmdParser, 0, sizeof (*pCmdParser));

    /* Store the callbacks */
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
void lwswitchUtilsCmdParserDestroy(LwswitchUtilsCmdParser_t *pCmdParser)
{
    if (NULL != pCmdParser) 
    {
        if (NULL != pCmdParser->pCmdLine) 
        {
            cmdline_destroy(pCmdParser->pCmdLine);
        }

        free(pCmdParser);
    }
}

/*****************************************************************************
 Internal Method to verify basic correctness for command line
 *****************************************************************************/
static int lwswitchUtilsCmdVerifyOptions(LwswitchUtilsCmdParser_t *pCmdParser)
{
    int cmdLineResult;
    char * firstUnknownOptionName = NULL, *firstArgWithMissingOption = NULL;

    cmdLineResult = cmdline_checkOptions(pCmdParser->pCmdLine,
            (const char **) &firstUnknownOptionName,
            (const char **) &firstArgWithMissingOption);

    switch (cmdLineResult) {
        case CMDLINE_CHECK_OPTIONS_UNKNOWN_OPTION_FOUND:
            printf("Option \"%s\" is not recognized.\n", firstUnknownOptionName);
            return FM_INT_ST_GENERIC_ERROR;

        case CMDLINE_CHECK_OPTIONS_MISSING_REQUIRED_VALUE:
            printf( "Option \"%s\" is missing its value.\n", firstArgWithMissingOption);
            return FM_INT_ST_GENERIC_ERROR;

        case CMDLINE_CHECK_OPTIONS_SUCCESS:
            break;
    }

    return FM_INT_ST_OK;
}


/*****************************************************************************
 Method to Parse Command Line
 *****************************************************************************/
FMIntReturn_t lwswitchUtilsCmdProcessing(LwswitchUtilsCmdParser_t *pCmdParser)
{
    if (NULL == pCmdParser) {
        return FM_INT_ST_BADPARAM;
    }

    /* Check if there is a help option specified for the command line */
    if (cmdline_exists(pCmdParser->pCmdLine, LWSWITCH_UTILS_CMD_HELP)) {
        pCmdParser->pfCmdHelp(pCmdParser->pCmdLine);
        return FM_INT_ST_OK;
    }

    if (cmdline_exists(pCmdParser->pCmdLine, LWSWITCH_UTILS_CMD_LIST_LWSWITCH)) {
        pCmdParser->mListSwitches = true;
    }

    if (cmdline_exists(pCmdParser->pCmdLine, LWSWITCH_UTILS_CMD_GET_FATAL_ERROR_SCOPE)) {
        pCmdParser->mGetSwitchErrorScope = true;
    }

    if (cmdline_exists(pCmdParser->pCmdLine, LWSWITCH_UTILS_CMD_INJECT_NON_FATAL_ERROR)) {
        pCmdParser->mInjectSwitchError = true;
        pCmdParser->mFatalScope = NON_FATAL;
    }

    if (cmdline_exists(pCmdParser->pCmdLine, LWSWITCH_UTILS_CMD_INJECT_FATAL_DEVICE_ERROR)) {
        pCmdParser->mInjectSwitchError = true;
        pCmdParser->mFatalScope = FATAL_DEVICE;
    }

    if (cmdline_exists(pCmdParser->pCmdLine, LWSWITCH_UTILS_CMD_INJECT_FATAL_PORT_ERROR)) {
        pCmdParser->mInjectSwitchError = true;
        pCmdParser->mFatalScope = FATAL_PORT;
    }

    if (cmdline_exists(pCmdParser->pCmdLine, LWSWITCH_UTILS_CMD_SWITCH_INSTANCE)) {

        unsigned long long switchInstance = 0;
        /* Get the partition id */
        if (!cmdline_getIntegerVal(pCmdParser->pCmdLine, LWSWITCH_UTILS_CMD_SWITCH_INSTANCE,
                                   &switchInstance)) {
            return FM_INT_ST_BADPARAM;
        }

        pCmdParser->mSwitchDevInstance = switchInstance;
    }

    if (cmdline_exists(pCmdParser->pCmdLine, LWSWITCH_UTILS_CMD_SWITCH_PORT_NUM)) {

        unsigned long long switchPort = 0;
        /* Get the partition id */
        if (!cmdline_getIntegerVal(pCmdParser->pCmdLine, LWSWITCH_UTILS_CMD_SWITCH_PORT_NUM,
                                   &switchPort)) {
            return FM_INT_ST_BADPARAM;
        }

        pCmdParser->mSwitchPortNum = switchPort;
    }

    return FM_INT_ST_OK;
}
