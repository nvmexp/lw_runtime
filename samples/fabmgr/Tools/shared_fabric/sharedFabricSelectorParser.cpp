#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include "sharedFabricSelectorParser.h"

/*****************************************************************************
 Method to Initialize the command parser
 *****************************************************************************/
SharedFabricCmdParser_t * sharedFabricCmdParserInit(int argc, char **argv,
        struct all_args * pAllArgs,
        int numberElementsInAllArgs,
        void (*pfCmdUsage)(void *),
        void (*pfCmdHelp)(void *))
{
    SharedFabricCmdParser_t *pCmdParser;
    int cmdLineResult = 0;

    if ((NULL == pAllArgs) || (NULL == pfCmdUsage) || (NULL == pfCmdHelp)) {
        return NULL;
    }

    pCmdParser = (SharedFabricCmdParser_t *)malloc(sizeof (*pCmdParser));
    if (NULL == pCmdParser) {
        return NULL;
    }

    /* memset to 0 */
    memset(pCmdParser, 0, sizeof (*pCmdParser));

    /* Default to connect to localhost */
    strncpy(pCmdParser->mHostname, "127.0.0.1", sizeof(pCmdParser->mHostname)-1);

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
void sharedFabricCmdParserDestroy(SharedFabricCmdParser_t *pCmdParser)
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
static int sharedFabricCmdVerifyOptions(SharedFabricCmdParser_t *pCmdParser)
{
    int cmdLineResult;
    char * firstUnknownOptionName = NULL, *firstArgWithMissingOption = NULL;

    cmdLineResult = cmdline_checkOptions(pCmdParser->pCmdLine,
            (const char **) &firstUnknownOptionName,
            (const char **) &firstArgWithMissingOption);

    switch (cmdLineResult) {
        case CMDLINE_CHECK_OPTIONS_UNKNOWN_OPTION_FOUND:
            printf("Option \"%s\" is not recognized.\n", firstUnknownOptionName);
            return FM_ST_GENERIC_ERROR;

        case CMDLINE_CHECK_OPTIONS_MISSING_REQUIRED_VALUE:
            printf( "Option \"%s\" is missing its value.\n", firstArgWithMissingOption);
            return FM_ST_GENERIC_ERROR;

        case CMDLINE_CHECK_OPTIONS_SUCCESS:
            break;
    }

    return FM_ST_SUCCESS;
}

fmReturn_t parsePartitionIdlistString(std::string &partitionListStr,
                                      SharedFabricCmdParser_t *pCmdParser)
{
    char* token = strtok((char *)partitionListStr.c_str(), ",");
    int haveSeen[FM_MAX_FABRIC_PARTITIONS] = {0};

    pCmdParser->mNumPartitions = 0;

    while (token != NULL) {
        int partitionId = (int)atoi(token);

        if(partitionId < 0 || partitionId >= FM_MAX_FABRIC_PARTITIONS)
        {
            printf("Invalid partition Id %u was given.\n", partitionId);
            return FM_ST_BADPARAM;
        }

        if (haveSeen[partitionId])
            continue; /* Just ignore it and move on */
        haveSeen[partitionId] = 1;

        pCmdParser->mPartitionIds[pCmdParser->mNumPartitions] = partitionId;
        pCmdParser->mNumPartitions++;

        token = strtok(NULL, ",");
    }

    return FM_ST_SUCCESS;
}

/*****************************************************************************
 Method to Parse Command Line
 *****************************************************************************/
fmReturn_t sharedFabricCmdProcessing(SharedFabricCmdParser_t *pCmdParser)
{
    if (NULL == pCmdParser) {
        return FM_ST_BADPARAM;
    }

    /* Check if there is a help option specified for the command line */
    if (cmdline_exists(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_HELP)) {
        pCmdParser->pfCmdHelp(pCmdParser->pCmdLine);
        return FM_ST_SUCCESS;
    }

    /* Check if listen interface is specified */
    if (cmdline_exists(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_HOSTNAME)) {
        char *hostname = NULL;

        if (cmdline_getStringVal(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_HOSTNAME,
                                 (const char **) &hostname))
        {
            strncpy(pCmdParser->mHostname, hostname, MAX_PATH_LEN);
        }
    }

    /* Check if unix domain socket path is specified */
    if (cmdline_exists(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_UNIX_DOMAIN_SOCKET)) {
        char *unixSockPath = NULL;

        if (cmdline_getStringVal(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_UNIX_DOMAIN_SOCKET,
                                 (const char **) &unixSockPath))
        {
            strncpy(pCmdParser->mUnixSockPath, unixSockPath, MAX_PATH_LEN);
        }
    }

    if (cmdline_exists(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_LIST_PARTITION)) {
        pCmdParser->mListPartition = 1;
        return FM_ST_SUCCESS;
    }

    if (cmdline_exists(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_LIST_UNSUPPORTED_PARTITION)) {
        pCmdParser->mListUnsupportedPartition = 1;
        return FM_ST_SUCCESS;
    }

    /* Check if partitionId is specified  */
    if (cmdline_exists(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_ACTIVATE_PARTITION)) {

        unsigned long long tempPartitionId = 0;
        /* Get the partition id */
        if (!cmdline_getIntegerVal(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_ACTIVATE_PARTITION,
                                   &tempPartitionId)) {
            return FM_ST_BADPARAM;
        }

        // Check if partition ID is valid
        if ((tempPartitionId < SHARED_FABRIC_SELECTOR_PARTITION_ID_MIN) ||
            (tempPartitionId > SHARED_FABRIC_SELECTOR_PARTITION_ID_MAX)) {
            fprintf(stderr, "Invalid Partition ID: %llu. Valid Partition ID ranges from %d to %d.\n",
                    tempPartitionId, SHARED_FABRIC_SELECTOR_PARTITION_ID_MIN,
                    SHARED_FABRIC_SELECTOR_PARTITION_ID_MAX);
            return FM_ST_BADPARAM;
        }

        pCmdParser->mActivatePartition = 1;
        pCmdParser->mPartitionId = tempPartitionId;
    }

    if (cmdline_exists(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_DEACTIVATE_PARTITION)) {

        unsigned long long tempPartitionId = 0;
        /* Get the partition id */
        if (!cmdline_getIntegerVal(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_DEACTIVATE_PARTITION,
                                   &tempPartitionId)) {
            return FM_ST_BADPARAM;
        }

        // Check if partition ID is valid
        if ((tempPartitionId < SHARED_FABRIC_SELECTOR_PARTITION_ID_MIN) ||
            (tempPartitionId > SHARED_FABRIC_SELECTOR_PARTITION_ID_MAX)) {
            fprintf(stderr, "Invalid Partition ID: %llu. Valid Partition ID ranges from %d to %d.\n",
                    tempPartitionId, SHARED_FABRIC_SELECTOR_PARTITION_ID_MIN,
                    SHARED_FABRIC_SELECTOR_PARTITION_ID_MAX);
            return FM_ST_BADPARAM;
        }

        pCmdParser->mDeactivatePartition = 1;
        pCmdParser->mPartitionId = tempPartitionId;
    }

    /* Check if activated partition Ids is specified */
    if (cmdline_exists(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_SET_ACTIVATED_PARTITION_LIST)) {
        char *partitionList = NULL;
        pCmdParser->mSetActivatedPartitions = 1;
        pCmdParser->mNumPartitions = 0;

        if (cmdline_getStringVal(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_SET_ACTIVATED_PARTITION_LIST,
                                 (const char **) &partitionList))
        {
            std::string partitionListStr = partitionList;
            fmReturn_t ret = parsePartitionIdlistString(partitionListStr, pCmdParser);
            if (ret != FM_ST_SUCCESS) {
                return ret;
            }
        }
    }

    if (cmdline_exists(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_GET_LWLINK_FAILED_DEVICES)) {
        pCmdParser->mGetLwlinkFailedDevices = 1;
        return FM_ST_SUCCESS;
    }

    /* Check if a binary state file to be colwerted to text file */
    if (cmdline_exists(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_COLWERT_BIN_TO_TXT)) {
        char *filename = NULL;

        if (cmdline_getStringVal(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_COLWERT_BIN_TO_TXT,
                                 (const char **) &filename))
        {
            pCmdParser->mBinToTxt = true;
            strncpy(pCmdParser->mInFileName, filename,
                    sizeof(pCmdParser->mInFileName)-1);

            if (strlen(filename) < (256 - 5)) {
                strncpy(pCmdParser->mOutFileName, filename,
                        sizeof(pCmdParser->mOutFileName)-1);
            } else {
                strncpy(pCmdParser->mOutFileName, filename,
                        (256 - 5));
            }

            strncat(pCmdParser->mOutFileName, ".txt",
                    sizeof(pCmdParser->mOutFileName)-1);
        }
    }

    /* Check if a text state file to be colwerted to binary file */
    if (cmdline_exists(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_COLWERT_TXT_TO_BIN)) {
        char *filename = NULL;

        if (cmdline_getStringVal(pCmdParser->pCmdLine, SHARED_FABRIC_CMD_COLWERT_TXT_TO_BIN,
                                 (const char **) &filename))
        {
            pCmdParser->mTxtToBin = true;
            strncpy(pCmdParser->mInFileName, filename,
                    sizeof(pCmdParser->mInFileName)-1);

            if (strlen(filename) < (256 - 5)) {
                strncpy(pCmdParser->mOutFileName, filename,
                        sizeof(pCmdParser->mOutFileName)-1);
            } else {
                strncpy(pCmdParser->mOutFileName, filename,
                        (256 - 5));
            }

            strncat(pCmdParser->mOutFileName, ".bin",
                    sizeof(pCmdParser->mOutFileName)-1);
        }
    }

    if (FM_ST_SUCCESS != sharedFabricCmdVerifyOptions(pCmdParser)) {
        return FM_ST_GENERIC_ERROR;
    }

    return FM_ST_SUCCESS;
}
