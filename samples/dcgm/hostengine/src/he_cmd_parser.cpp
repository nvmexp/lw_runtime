#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DcgmStringTokenize.h"
#include "logging.h"
#include "he_cmd_parser.h"

/*****************************************************************************
 Method to Initialize the command parser
 *****************************************************************************/
heCmdParser_t * hostEngineCmdParserInit(int argc, char **argv,
        struct all_args * pAllArgs,
        int numberElementsInAllArgs,
        void (*pfCmdUsage)(void *),
        void (*pfCmdHelp)(void *))
{
    heCmdParser_t *pCmdParser;
    int cmdLineResult = 0;

    if ((NULL == pAllArgs) || (NULL == pfCmdUsage) || (NULL == pfCmdHelp)) {
        return NULL;
    }

    pCmdParser = (heCmdParser_t *)malloc(sizeof (*pCmdParser));
    if (NULL == pCmdParser) {
        return NULL;
    }

    /* memset to 0 */
    memset(pCmdParser, 0, sizeof (*pCmdParser));

    /* set connection type to TCP */
    pCmdParser->mHostEngineConnTCP = 1;

    /* set port number to default */
    pCmdParser->mHostEnginePort = 5555; // (update this if changing DCGM_HE_PORT_NUMBER)

    /* set Unix domain socket path to default */
    snprintf(pCmdParser->mHostEngineSockPath, sizeof(pCmdParser->mHostEngineSockPath)-1,
             "/tmp/lw-hostengine");

    /* Default to listening on localhost only */
    strncpy(pCmdParser->mHostEngineBindInterfaceIp, "127.0.0.1", sizeof(pCmdParser->mHostEngineBindInterfaceIp)-1);

    /* Default pid path */
    strncpy(pCmdParser->mPidFilePath, "/var/run/lwhostengine.pid", sizeof(pCmdParser->mPidFilePath)-1);

    /* Store the callbacks */
    pCmdParser->pfCmdUsage = pfCmdUsage;
    pCmdParser->pfCmdHelp = pfCmdHelp;

    cmdLineResult = cmdline_init(argc, argv, pAllArgs, numberElementsInAllArgs, &pCmdParser->pCmdLine);
    if (0 != cmdLineResult) {
        free(pCmdParser);
        pCmdParser = NULL;
        return NULL;
    }

#ifdef DCGM_BUILD_LWSWITCH_MODULE
    /* Both global and local fabric manager are disabled by default */
    pCmdParser->mEnableGFM = 0;
    pCmdParser->mEnableLFM = 0;
    pCmdParser->mFMStartingPort = 0;
    pCmdParser->mFMSharedFabric = 0;
    pCmdParser->mFMRestart = 0;
    memset(pCmdParser->mFMUnixSockPath, 0, sizeof(pCmdParser->mFMUnixSockPath));
    memset(pCmdParser->mFMStateFilename, 0, sizeof(pCmdParser->mFMStateFilename));
#endif

    return pCmdParser;
}

/*****************************************************************************
 Method to destroy Command Line Parser
 *****************************************************************************/
void hostEngineCmdParserDestroy(heCmdParser_t *pCmdParser)
{
    if (NULL != pCmdParser) 
    {
        if (NULL != pCmdParser->pCmdLine) 
        {
            cmdline_destroy(pCmdParser->pCmdLine);
        }

        free(pCmdParser);
        pCmdParser = NULL;
    }
}

/*****************************************************************************
 Internal Method to verify basic correctness for command line
 *****************************************************************************/
static int hostEngineVerifyOptions(heCmdParser_t *pCmdParser)
{
    int cmdLineResult;
    char * firstUnknownOptionName = NULL, *firstArgWithMissingOption = NULL;

    cmdLineResult = cmdline_checkOptions(pCmdParser->pCmdLine,
            (const char **) &firstUnknownOptionName,
            (const char **) &firstArgWithMissingOption);
    switch (cmdLineResult) {
        case CMDLINE_CHECK_OPTIONS_UNKNOWN_OPTION_FOUND:
            printf("Option \"%s\" is not recognized.\n", firstUnknownOptionName);
            return HE_ST_GENERIC_ERROR;

        case CMDLINE_CHECK_OPTIONS_MISSING_REQUIRED_VALUE:
            printf( "Option \"%s\" is missing its value.\n", firstArgWithMissingOption);
            return HE_ST_GENERIC_ERROR;

        case CMDLINE_CHECK_OPTIONS_SUCCESS:
            break;
    }

    return HE_ST_OK;
}

/*****************************************************************************
 Method to Parse the module blacklist part of the command line
 *****************************************************************************/
static HEReturn_t hostEngineParseBlacklistString(std::string blacklistStr, heCmdParser_t *pCmdParser)
{
    std::vector<std::string>tokens;
    std::vector<std::string>::iterator tokenIt;
    int haveSeen[DcgmModuleIdCount] = {0};
    
    tokenizeString(blacklistStr, ",", tokens);

    if(!tokens.size())
    {
        printf("No values provided for --blacklist-modules\n");
        return HE_ST_BADPARAM;
    }

    pCmdParser->numBlacklistModules = 0;

    for (tokenIt = tokens.begin(); tokenIt != tokens.end(); ++tokenIt)
    {
        int moduleId = atoi((*tokenIt).c_str());
        if(moduleId <= 0 || moduleId >= DcgmModuleIdCount)
        {
            printf("Invalid --blacklist-modules ID %d was given.\n", moduleId);
            return HE_ST_BADPARAM;
        }

        if(haveSeen[moduleId])
            continue; /* Just ignore it and move on */
        haveSeen[moduleId] = 1;

        pCmdParser->blacklistModules[pCmdParser->numBlacklistModules] = (dcgmModuleId_t)moduleId;
        pCmdParser->numBlacklistModules++;
    }

    return HE_ST_OK;
}

/*****************************************************************************
 Method to Parse Command Line
 *****************************************************************************/
HEReturn_t hostEngineCmdProcessing(heCmdParser_t *pCmdParser)
{
    if (NULL == pCmdParser) {
        return HE_ST_BADPARAM;
    }

    /* Check if there is a help option specified for the command line */
    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_HELP)) {
        pCmdParser->pfCmdHelp(pCmdParser->pCmdLine);
        return HE_ST_OK;
    }

    /* Check if version is specified  */
    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_VERSION)) {
        pCmdParser->mPrintVersion = 1;
        return HE_ST_OK;
    }

    /* Check if terminate is specified  */
    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_TERM)) {
        pCmdParser->mTermHostEngine = 1;
        /* Keep running in case PID filename is present */
    }

    /* Check if port is specified  */
    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_PORT)){
        unsigned long long portBuffer = 0;

        /* Get the port number */
        if(!cmdline_getIntegerVal(pCmdParser->pCmdLine, HE_CMD_PORT, &portBuffer)){
            return HE_ST_BADPARAM;
        }

        // Check if valid
        if((portBuffer <= 0) || (portBuffer >= 65535)) // 65535 = 2 ^ 16 -1 which is largest possible port number
        {
            fprintf(stderr, "Invalid port number: %d\n", (int)portBuffer);
            return HE_ST_BADPARAM;
        }

        pCmdParser->mHostEnginePort = (unsigned short) portBuffer;
    }

    /* Check if listen interface is specified */
    if(cmdline_exists(pCmdParser->pCmdLine, HE_CMD_BIND_INTERFACE)){
        char *interfaceBuffer = NULL;

        if(cmdline_getStringVal(pCmdParser->pCmdLine, HE_CMD_BIND_INTERFACE, (const char **) &interfaceBuffer))
        {
            if(!strcmp(interfaceBuffer, "all") || !strcmp(interfaceBuffer, "ALL"))
            {
                /* Set to all interfaces */
                pCmdParser->mHostEngineBindInterfaceIp[0] = 0;
            }
            else
            {
                strncpy(pCmdParser->mHostEngineBindInterfaceIp, interfaceBuffer,
                        sizeof(pCmdParser->mHostEngineBindInterfaceIp)-1);
            }
        }
    }

    /* Check if PID path is specified */
    if(cmdline_exists(pCmdParser->pCmdLine, HE_CMD_PID_FILENAME))
    {
        char *pidPathBuffer = NULL;
        if(cmdline_getStringVal(pCmdParser->pCmdLine, HE_CMD_PID_FILENAME, (const char **) &pidPathBuffer))
        {
            strncpy(pCmdParser->mPidFilePath, pidPathBuffer, sizeof(pCmdParser->mPidFilePath)-1);
        }
    }

    /* Check terminate is specified  */
    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_SOCK_PATH)){
        char *pathBuffer = NULL;

        /* Set connection type to Unix */
        pCmdParser->mHostEngineConnTCP = 0;

        /* Get the socket path */
        if(cmdline_getStringVal(pCmdParser->pCmdLine, HE_CMD_SOCK_PATH, (const char **) &pathBuffer)){
            strncpy(pCmdParser->mHostEngineSockPath, pathBuffer, sizeof(pCmdParser->mHostEngineSockPath)-1);
        }
    }

    /* Check if requested not to daemonize */
    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_NO_DAEMON)){
        pCmdParser->shouldNotDaemonize = 1;
    }

    /* Check for log level */
    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_LOG_LEVEL)){
        unsigned long long logLevel = 0;

        /* Get the log level value */
        if(!cmdline_getIntegerVal(pCmdParser->pCmdLine, HE_CMD_LOG_LEVEL, &logLevel)){
            return HE_ST_BADPARAM;
        }

        // Check if valid
        if((logLevel <= LWML_DBG_DISABLED) || (logLevel > LWML_DBG_DEBUG))
        {
            fprintf(stderr, "Invalid logging level specified: %d\n", (int)logLevel);
            return HE_ST_BADPARAM;
        }

        pCmdParser->mLogLevel = (unsigned short) logLevel;
    }

    /* Check if log path is specified */
    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_LOG_FILENAME))
    {
        char *fileName = NULL;
        if (cmdline_getStringVal(pCmdParser->pCmdLine, HE_CMD_LOG_FILENAME, (const char **) &fileName))
        {
            strncpy(pCmdParser->mLogFileName, fileName, sizeof(pCmdParser->mLogFileName)-1);
        }
    }

    /* Check if log should be rotated */
    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_LOG_FILE_ROTATE)){
        pCmdParser->mLogRotate = true;
    }

    /* Check if any modules are blacklisted */
    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_MODULE_BLACKLIST))
    {
        char *blacklistStr = NULL;
        if (cmdline_getStringVal(pCmdParser->pCmdLine, HE_CMD_MODULE_BLACKLIST, (const char **) &blacklistStr))
        {
            HEReturn_t retSt = hostEngineParseBlacklistString(std::string(blacklistStr), pCmdParser);
            if(retSt != HE_ST_OK)
                return retSt;
        }
    }

#ifdef DCGM_BUILD_LWSWITCH_MODULE
    /* Check if global or local fabric manager is enabled */
    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_EN_GFM)){
        pCmdParser->mEnableGFM = 1;
    }

    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_EN_LFM)){
        pCmdParser->mEnableLFM = 1;
    }

    /* Check for fabric manager starting TCP port number */
    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_FM_PORT)){
        unsigned long long startingPort = 0;

        /* Get the port number */
        if(!cmdline_getIntegerVal(pCmdParser->pCmdLine, HE_CMD_FM_PORT, &startingPort)){
            return HE_ST_BADPARAM;
        }

        // Check if valid
        // fabric manager starting port ranges from 1024 to 65534
        if((startingPort <= 1023) || (startingPort >= 65535))
        {
            fprintf(stderr, "Invalid port number: %d\n", (int)startingPort);
            return HE_ST_BADPARAM;
        }

        pCmdParser->mFMStartingPort = (unsigned short) startingPort;
    }

    /* Check if fabric manager domain socket path is specified  */
    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_FM_SOCK_PATH)){
        char *pathBuffer = NULL;

        if (pCmdParser->mFMStartingPort != 0 ){
            fprintf(stderr, "Can't combine Fabric Manager TCP I/P port number and Unix domain socket options.\n");
            return HE_ST_BADPARAM;
        }

        /* Get the socket path */
        if(cmdline_getStringVal(pCmdParser->pCmdLine, HE_CMD_FM_SOCK_PATH, (const char **) &pathBuffer)){
           strncpy(pCmdParser->mFMUnixSockPath, pathBuffer, sizeof(pCmdParser->mFMUnixSockPath)-1);
        }
    }

    /* Check for fabric manager shared LWSwitch multitenancy mode */
    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_FM_SHARED_FABRIC)){
        pCmdParser->mFMSharedFabric = 1;
    }

    /* Check for fabric manager is restarted in shared LWSwitch multitenancy mode */
    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_FM_RESTART)){
        if (pCmdParser->mFMSharedFabric == 0 ){
            fprintf(stderr, "Fabric Manager can only be restarted in shared LWSwitch multitenancy mode.\n");
            return HE_ST_BADPARAM;
        }

        pCmdParser->mFMRestart = 1;
    }

    /* Check if fabric manager state filename is specified in shared LWSwitch multitenancy mode  */
    if (cmdline_exists(pCmdParser->pCmdLine, HE_CMD_FM_STATE_FILENAME)){
        char *pathBuffer = NULL;

        if (pCmdParser->mFMSharedFabric == 0 ){
            fprintf(stderr, "Fabric Manager state filename can only be specified in shared LWSwitch multitenancy mode.\n");
            return HE_ST_BADPARAM;
        }

        /* Get the filename */
        if(cmdline_getStringVal(pCmdParser->pCmdLine, HE_CMD_FM_STATE_FILENAME, (const char **) &pathBuffer)){
           strncpy(pCmdParser->mFMStateFilename, pathBuffer, sizeof(pCmdParser->mFMStateFilename)-1);
        }
    }

#endif

    if (HE_ST_OK != hostEngineVerifyOptions(pCmdParser)) {
        return HE_ST_GENERIC_ERROR;
    }

    return HE_ST_OK;
}
