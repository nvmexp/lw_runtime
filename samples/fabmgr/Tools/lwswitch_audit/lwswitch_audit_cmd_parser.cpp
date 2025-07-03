#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lwswitch_audit_cmd_parser.h"
#include "lwswitch_audit_node.h"
#include <errno.h>
#include <string.h>

// Method to Initialize the command parser
naCmdParser_t * naCmdParserInit(int argc, char **argv,
        struct all_args * pAllArgs,
        int numberElementsInAllArgs,
        void (*pfCmdUsage)(void *),
        void (*pfCmdHelp)(void *))
{
    naCmdParser_t *pCmdParser;
    int cmdLineResult = 0;

    if ((NULL == pAllArgs) || (NULL == pfCmdUsage) || (NULL == pfCmdHelp)) 
    {
        return NULL;
    }

    pCmdParser = (naCmdParser_t *)malloc(sizeof (*pCmdParser));
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
    pCmdParser->mPrintFullMatrix = false;
    pCmdParser->mPrintCSV = false;
    pCmdParser->mSrcGPU = -1;       //Invalid GPU ID
    pCmdParser->mDestGPU = -1;      //Invalid GPU ID
#ifdef DEBUG
    pCmdParser->mSwitch = -1;       //Invalid Switch ID
    pCmdParser->mSwitchPort = -1;   //Invalid Switch port
    pCmdParser->mSetDestGPU = -1;   //Invalid Destination GPU ID
    pCmdParser->mSetDestRLID = -1;  //Invalid Destination Requestor Link ID
    pCmdParser->mSetRLID = -1;      //Invalid Requestor Link ID)
    pCmdParser->mValid = -1;        //Invalid entry
    pCmdParser->mEgressPort = -1;   //Invalid egress port
    pCmdParser->mQuick = false;     //Only read valid GOU range entries from req/resp tables
#endif

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
void naCmdParserDestroy(naCmdParser_t *pCmdParser)
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
static int naVerifyOptions(naCmdParser_t *pCmdParser)
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
            return NA_ST_GENERIC_ERROR;

        case CMDLINE_CHECK_OPTIONS_MISSING_REQUIRED_VALUE:
            fprintf(stderr, "Option \"%s\" is missing its value.\n", firstArgWithMissingOption);
            return NA_ST_GENERIC_ERROR;

        case CMDLINE_CHECK_OPTIONS_SUCCESS:
            return NA_ST_OK;
                
        default:
            fprintf(stderr, "Unknown return value when checking command line options\n");
            return NA_ST_GENERIC_ERROR;
    }

}

#ifdef DEBUG
//get next integer value from delimiter separated string. Used 
//for parsing request/response/rlid entries 
static bool getNextInt(char *str, const char *delim, int &val)
{
    char *sub_str;
    char *endPtr;
    if(str != NULL)
        sub_str = strtok(str, delim);
    else
        sub_str = strtok(NULL, delim);

    //if((sub_str == NULL) || (*sub_str == '\0'))
    if(sub_str == NULL)
        return false;


    errno = 0;
    endPtr = NULL;

    val = strtoul(sub_str, &endPtr, 10);
    if ((val == 0) && ((errno != 0) || (endPtr == sub_str)))
        return false;
    else
        return true;
}

bool parseRLID(naCmdParser_t *pCmdParser)
{
    char *rlid=NULL;
    if(cmdline_getStringVal(pCmdParser->pCmdLine, NA_CMD_RLID,(const char **) &rlid) == 0)
    {
        fprintf(stderr, "Incorrect value for Request link ID\n");
        return false;
    }

    if(!getNextInt(rlid, ":", pCmdParser->mSwitch))
    {
        fprintf(stderr, "Incorrect LWSwitch id\n");
        return false;
    }

    if(!getNextInt(NULL, ":", pCmdParser->mSwitchPort))
    {
        fprintf(stderr, "Incorrect LWSwitch port\n");
        return false;
    }

    if(!getNextInt(NULL, ":", pCmdParser->mSetRLID))
    {
        fprintf(stderr, "Incorrect Requestor Link ID\n");
        return false;
    }
    return true;
}
bool parseResEntry(naCmdParser_t *pCmdParser)
{
    char *resEntry=NULL;
    if(cmdline_getStringVal(pCmdParser->pCmdLine, NA_CMD_RES_ENTRY,(const char **) &resEntry) == 0)
    {
        fprintf(stderr, "Incorrect value for response entry\n");
        return false;
    }

    if(!getNextInt(resEntry, ":", pCmdParser->mSwitch))
    {
        fprintf(stderr, "Incorrect LWSwitch id\n");
        return false;
    }

    if(!getNextInt(NULL, ":", pCmdParser->mSwitchPort))
    {
        fprintf(stderr, "Incorrect LWSwitch port\n");
        return false;
    }

    if(!getNextInt(NULL, ":", pCmdParser->mSetDestRLID))
    {
        fprintf(stderr, "Incorrect destination Requestor Link ID\n");
        return false;
    }

    if(!getNextInt(NULL, ":", pCmdParser->mValid))
    {
        fprintf(stderr, "Incorrect value for valid bit\n");
        pCmdParser->mSetDestRLID = -1;
        return false;
    }

    if((pCmdParser->mValid != 0) && (pCmdParser->mValid != 1))
    {
        fprintf(stderr, "Incorrect value for valid bit\n");
        pCmdParser->mSetDestRLID = -1;
        return false;
    }

    if(!getNextInt(NULL, ":", pCmdParser->mEgressPort))
    {
        fprintf(stderr, "Incorrect egress port\n");
        pCmdParser->mSetDestRLID = -1;
        return false;
    }
    return true;
}
bool parseReqEntry(naCmdParser_t *pCmdParser)
{
    char *reqEntry=NULL;
    if(cmdline_getStringVal(pCmdParser->pCmdLine, NA_CMD_REQ_ENTRY,(const char **) &reqEntry) == 0)
    {
        fprintf(stderr, "Incorrect value for request entry\n");
        return false;
    }

    if(!getNextInt(reqEntry, ":", pCmdParser->mSwitch))
    {
        fprintf(stderr, "Incorrect LWSwitch id\n");
        return false;
    }

    if(!getNextInt(NULL, ":", pCmdParser->mSwitchPort))
    {
        fprintf(stderr, "Incorrect LWSwitch port\n");
        return false;
    }

    if(!getNextInt(NULL, ":", pCmdParser->mSetDestGPU))
    {
        fprintf(stderr, "Incorrect destination GPU\n");
        return false;
    }

    if(!getNextInt(NULL, ":", pCmdParser->mValid))
    {
        fprintf(stderr, "Incorrect value for valid bit\n");
        pCmdParser->mSetDestGPU = -1;
        return false;
    }

    if((pCmdParser->mValid != 0) && (pCmdParser->mValid != 1))
    {
        fprintf(stderr, "Incorrect value for valid bit\n");
        pCmdParser->mSetDestGPU = -1;
        return false;
    }

    if(!getNextInt(NULL, ":", pCmdParser->mEgressPort))
    {
        fprintf(stderr, "Incorrect egress port\n");
        pCmdParser->mSetDestGPU = -1;
        return false;
    }
    return true;
}
#endif
// Method to Parse Command Line
NAReturn_t naCmdProcessing(naCmdParser_t *pCmdParser)
{
    if (NULL == pCmdParser) 
    {
        return NA_ST_BADPARAM;
    }

    // Check if there is a help option specified for the command line
    if (cmdline_exists(pCmdParser->pCmdLine, NA_CMD_HELP)) 
    {
        pCmdParser->pfCmdHelp(pCmdParser->pCmdLine);
        return NA_ST_OK;
    }

    if (NA_ST_OK != naVerifyOptions(pCmdParser)) 
    {
        return NA_ST_GENERIC_ERROR;
    }

    if (cmdline_exists(pCmdParser->pCmdLine, NA_CMD_VERBOSE))
    {
        pCmdParser->mPrintVerbose = true;
    }

    if (cmdline_exists(pCmdParser->pCmdLine, NA_CMD_FULL_MATRIX))
    {
        pCmdParser->mPrintFullMatrix = true;
    }

    if (cmdline_exists(pCmdParser->pCmdLine, NA_CMD_CSV))
    {
        pCmdParser->mPrintCSV = true;
    }
    
    if (cmdline_exists(pCmdParser->pCmdLine, NA_CMD_SOURCE_GPU))
    {
        // Get the GPU source ID 
        unsigned long long srcGpuId;
        if(!cmdline_getIntegerVal(pCmdParser->pCmdLine, NA_CMD_SOURCE_GPU, &srcGpuId))
        {
            return NA_ST_BADPARAM;
        }
        
        pCmdParser->mSrcGPU = (int)srcGpuId - 1;
    }

    if (cmdline_exists(pCmdParser->pCmdLine, NA_CMD_DEST_GPU))
    {
        // Get the Destination GPU ID
        unsigned long long destGpuId;
        if(!cmdline_getIntegerVal(pCmdParser->pCmdLine, NA_CMD_DEST_GPU, &destGpuId))
        {
            return NA_ST_BADPARAM;
        }
        
        pCmdParser->mDestGPU = (int)destGpuId - 1;
    }

    if (pCmdParser->mPrintVerbose == true) 
    {
        if (pCmdParser->mPrintCSV == true) 
        {
            fprintf(stderr, "Verbose option not valid when printing as CSV\n");
            return NA_ST_BADPARAM;
        }
        
        if((pCmdParser->mSrcGPU > 0) || (pCmdParser->mDestGPU > 0)) 
        {
            fprintf(stderr, "Verbose option not valid when specifiying Source/Destination GPU\n");
            return NA_ST_BADPARAM;
        }
    }
    
    if (pCmdParser->mPrintCSV == true) 
    {
        if((pCmdParser->mSrcGPU > -1) || (pCmdParser->mDestGPU > -1)) 
        {
            fprintf(stderr, "CSV option not valid when specifiying Source/Destination GPU\n");
            return NA_ST_BADPARAM;
        }
    }

    //Either both src/dst GPU should be specified or neither
    if((pCmdParser->mSrcGPU > -1) != (pCmdParser->mDestGPU > -1)) 
    {
        fprintf(stderr, "Both Source and Destination GPU should be provided\n");
        return NA_ST_BADPARAM;
    }
#ifdef DEBUG
    bool res=false, req=false, rlid=false;
    if(cmdline_exists(pCmdParser->pCmdLine, NA_CMD_REQ_ENTRY))
    {
        if(parseReqEntry(pCmdParser) == false) 
        {        
            fprintf(stderr, "Incorrect value for request entry\n");
            return NA_ST_BADPARAM;
        }
        res = true;
    }

    if(cmdline_exists(pCmdParser->pCmdLine, NA_CMD_RES_ENTRY))
    {
        if(parseResEntry(pCmdParser) == false) 
        {        
            fprintf(stderr, "Incorrect value for response entry\n");
            return NA_ST_BADPARAM;
        }
        res = true;
    }

    if(cmdline_exists(pCmdParser->pCmdLine, NA_CMD_RLID))
    {
        if(parseRLID(pCmdParser) == false) 
        {        
            fprintf(stderr, "Incorrect value for Requestor Link ID\n");
            return NA_ST_BADPARAM;
        }
        rlid = true;
    }
    if((res && req) || (res && rlid) ||(req && rlid))
    {
        fprintf(stderr, "Spefify only one entry to set Req/Res/RLID\n");
        return NA_ST_BADPARAM;
    }
    if(cmdline_exists(pCmdParser->pCmdLine, NA_CMD_QUICK))
    {
        pCmdParser->mQuick = true;
    }
#endif
    return NA_ST_OK;
}
