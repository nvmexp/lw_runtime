
#include <stdio.h>
#include <stdlib.h>
#include <commandline.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include "errno.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm> 
#include <chrono> 
#include <iostream> 
#include <vector>
#include <string>
#include <cstring>

#include "sharedFabricTestParser.h"
#include "platformModelDelta.h"

using namespace std; 
using namespace std::chrono; 
unsigned int fabricMode = 0;    /* FM operating mode */

struct all_args sharedFabricTestArgs[] = {
        {
                SHARED_FABRIC_TEST_CMD_HELP,
                "-h",
                "--help",
                "\t\tDisplays help information",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                SHARED_FABRIC_TEST_CMD_LIST_PARTITION,
                "-l",
                "--list",
                "\t\tQuery all the supported fabric partitions",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                SHARED_FABRIC_TEST_CMD_ACTIVATE_PARTITION,
                "-a",
                "--activate",
                "\t\tActivate a supported fabric partition",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                SHARED_FABRIC_TEST_CMD_DEACTIVATE_PARTITION,
                "-d",
                "--deactivate",
                "\tDeactivate a previously activated fabric partition",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                SHARED_FABRIC_TEST_CMD_SET_ACTIVATED_PARTITION_LIST,
                "",
                "--set-activated-list",
                "\tSet a list of lwrrently activated fabric partitions",
                "\n\t",
                CMDLINE_OPTION_VALUE_OPTIONAL
        },
        {
                SHARED_FABRIC_TEST_CMD_ACTIVATE_PARTITION_STRESS,
                "",
                "--activate-stress-test",
                "\t\tTest Fabric Manager partition activation deactivation for few times",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
};

/*****************************************************************************
 Method to Display Usage Info
 *****************************************************************************/
static void sharedFabricSelectorUsage(void * pCmdLine) {
    printf("\n\n");
    printf("    Usage: sharedfabrictest [options]\n");
    printf("\n");
    printf("    Options include:\n");
    cmdline_printOptionsSummary(pCmdLine, 0);
    exit(0);
}

/*****************************************************************************
 Method to Display Help Message
 *****************************************************************************/
static void sharedFabricSelectorDisplayHelpMessage(void * pCmdLine) {
    printf("\n");
    printf("    Shared Fabric Partition Selector test utility\n");

    sharedFabricSelectorUsage(pCmdLine);
}

static fmHandle_t gFmHandle = NULL;

void
initializeFMLibInterface(void)
{
    fmReturn_t fmReturn;
    int retryCount;

    if (gFmHandle != NULL) {
        fprintf(stderr, "Error: fabric manager interface is already initialized");
        exit(0);        
    }

    // initialize FM lib
    fmReturn = fmLibInit();
    if (FM_ST_SUCCESS != fmReturn) {
        fprintf(stderr, "Error: failed to initialize fabric manager interface library");
        exit(0);
    }

    // connect to fabric manager
    fmConnectParams_t connectParams;
    strncpy(connectParams.addressInfo, "127.0.0.1", FM_MAX_STR_LENGTH);
    connectParams.timeoutMs = 1000; // in milliseconds
    connectParams.version = fmConnectParams_version;
    connectParams.addressIsUnixSocket = 0;

    fprintf(stderr, "trying fmConnect \n");
    retryCount = 5;
    while (true) {
        fmReturn = fmConnect(&connectParams, &gFmHandle);
        if (fmReturn == FM_ST_SUCCESS) {
            fprintf(stderr, "fmConnect completed\n");
            break;
        }
        if (retryCount == 0) {
            fprintf(stderr, "Error connecting to Fabric Manager. Return: %d \n", fmReturn);
            fmLibShutdown();
            exit(0);
        }

        retryCount--;
        usleep(50000);
    }
}


int main(int argc, char **argv)
{
    fmReturn_t ret;
    SharedFabricCmdParser_t *pCmdParser;
    FILE *fp;
    char str[8];

    /* Check current FM operating mode. Return failure if FM is not running either in Shared LWSwitch or vGPU mode */
    fp = popen("grep FABRIC_MODE= /usr/share/lwpu/lwswitch/fabricmanager.cfg | cut -d'=' -f 2", "r");
    if (!fp) {
        fprintf(stderr, "popen() failed \n");
        return FM_ST_NOT_SUPPORTED;
    }

    fgets(str, sizeof(str), fp);
    fabricMode = std::stoul(str,nullptr,0);

    if ((fabricMode != 0x1) && (fabricMode != 0x2)) {
        fprintf(stderr, "FM is not running in Shared LWSwitch or vGPU based multitenancy mode");
        return FM_ST_NOT_SUPPORTED;
    }

    printf("Current FM Operating mode is %d\n", fabricMode);

    pCmdParser = sharedFabricCmdParserInit(argc, argv, sharedFabricTestArgs,
                                           SHARED_FABRIC_CMD_COUNT, sharedFabricSelectorUsage,
                                           sharedFabricSelectorDisplayHelpMessage);
    if (NULL == pCmdParser) {
        return FM_ST_BADPARAM;
    }

    ret = sharedFabricCmdProcessing(pCmdParser);
    if (FM_ST_SUCCESS != ret) {
        if (ret == FM_ST_BADPARAM){
            fprintf(stderr, "Unable to process command: bad command line parameter. \n");
        } else {
            fprintf(stderr, "Unable to process command: generic error. \n");
        }
        sharedFabricCmdParserDestroy(pCmdParser);
        return ret;
    }

    initializeFMLibInterface();

    // lwrrently hardcoded for Delta
    if (pCmdParser->mListPartition) {
        platformModelDelta::doPartitionList(gFmHandle);
    }

    if (pCmdParser->mActivationStreeTest) {
        platformModelDelta::doPartitionActivationStreeTest(gFmHandle);
    }

    if (pCmdParser->mActivatePartition) {
        platformModelDelta::doPartitionActivation(gFmHandle, pCmdParser->mPartitionId);
    }

    if (pCmdParser->mDeactivatePartition) {
        platformModelDelta::doPartitionDeactivation(gFmHandle, pCmdParser->mPartitionId);
    }

    if (pCmdParser->mSetActivatedPartitions) {
        platformModelDelta::setActivatedPartitionList(gFmHandle, pCmdParser);
    }
    
    return ret;
}


