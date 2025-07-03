#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lwlink_train_cmd_parser.h"
#include <errno.h>
#include <string.h>

// Method to Initialize the command parser
ntCmdParser_t * ntCmdParserInit(int argc, char **argv,
        struct all_args * pAllArgs,
        int numberElementsInAllArgs,
        void (*pfCmdUsage)(void *),
        void (*pfCmdHelp)(void *))
{
    ntCmdParser_t *pCmdParser;
    int cmdLineResult = 0;

    if ((NULL == pAllArgs) || (NULL == pfCmdUsage) || (NULL == pfCmdHelp)) 
    {
        return NULL;
    }

    pCmdParser = (ntCmdParser_t *)malloc(sizeof (*pCmdParser));
    if (NULL == pCmdParser) 
    {
        return NULL;
    }

    // memset to 0 
    memset(pCmdParser, 0, sizeof (*pCmdParser));

    // Store the callbacks
    pCmdParser->pfCmdUsage = pfCmdUsage;
    pCmdParser->pfCmdHelp = pfCmdHelp;
    pCmdParser->mPrintVerbose = false;
    pCmdParser->mBatchMode = false;

    cmdLineResult = cmdline_init(argc, argv, pAllArgs, numberElementsInAllArgs, &pCmdParser->pCmdLine);
    if (0 != cmdLineResult) 
    {
        free(pCmdParser);
        pCmdParser = NULL;
        return NULL;
    }

    return pCmdParser;
}

// Method to destroy Command Line Parser
void ntCmdParserDestroy(ntCmdParser_t *pCmdParser)
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

// Internal Method to verify basic correctness for command line
static int naVerifyOptions(ntCmdParser_t *pCmdParser)
{
    int cmdLineResult;
    char * firstUnknownOptionName = NULL, *firstArgWithMissingOption = NULL;

    cmdLineResult = cmdline_checkOptions(pCmdParser->pCmdLine,
            (const char **) &firstUnknownOptionName,
            (const char **) &firstArgWithMissingOption);
    switch (cmdLineResult) 
    {
        case CMDLINE_CHECK_OPTIONS_UNKNOWN_OPTION_FOUND:
            fprintf(stderr, "Option \"%s\" is not recognized.\n", firstUnknownOptionName);
            return NT_ST_GENERIC_ERROR;

        case CMDLINE_CHECK_OPTIONS_MISSING_REQUIRED_VALUE:
            fprintf(stderr, "Option \"%s\" is missing its value.\n", firstArgWithMissingOption);
            return NT_ST_GENERIC_ERROR;

        case CMDLINE_CHECK_OPTIONS_SUCCESS:
            return NT_ST_OK;
                
        default:
            fprintf(stderr, "Unknown return value when checking command line options\n");
            return NT_ST_GENERIC_ERROR;
    }

}

// Method to Parse Command Line
NTReturn_t ntCmdProcessing(ntCmdParser_t *pCmdParser)
{
    if (NULL == pCmdParser) 
    {
        return NT_ST_BADPARAM;
    }

    // Check if there is a help option specified for the command line
    if (cmdline_exists(pCmdParser->pCmdLine, NT_CMD_HELP)) 
    {
        pCmdParser->pfCmdHelp(pCmdParser->pCmdLine);
        return NT_ST_OK;
    }

    if (NT_ST_OK != naVerifyOptions(pCmdParser)) 
    {
        return NT_ST_GENERIC_ERROR;
    }

    if (cmdline_exists(pCmdParser->pCmdLine, NT_CMD_VERBOSE))
    {
        pCmdParser->mPrintVerbose = true;
    }

    if (cmdline_exists(pCmdParser->pCmdLine, NT_CMD_BATCH_MODE))
    {
        pCmdParser->mBatchMode = true;
    }


    return NT_ST_OK;
}
