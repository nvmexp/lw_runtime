#include <fcntl.h>
#include <limits.h>
#include <map>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <errno.h>
#include <unistd.h>
#include <vector>
#include <iostream>

#include "commandline/commandline.h"
#include "lwos.h"
#include "logging.h"
#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"
#include "lwswitch_audit_cmd_parser.h"
#include "lwswitch_audit_paths.h"
#include "lwswitch_audit_logging.h"
#include "lwswitch_audit_explorer16_juno.h"
#include "lwswitch_audit_explorer16_delta.h"
#include "lwswitch_audit_lwswitch.h"
#include "lwswitch_audit_node.h"

#include "FMGpuDriverVersionCheck.h"
#include "FMVersion.h"

extern "C"
{
#include "lwswitch_user_api.h"
}


bool verbose=false;
bool verboseErrors=true;

struct all_args naArgs[] = {
        {
                NA_CMD_HELP,
                "-h",
                "--help",
                "\t\tDisplays help information",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                NA_CMD_VERBOSE,
                "-v",
                "--verbose",
                "\t\tVerbose output including all Request and Reponse table entries",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                NA_CMD_FULL_MATRIX,
                "-f",
                "--full-matrix",
                "\t\tDisplay All possible GPUs including those with no connecting paths",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                NA_CMD_CSV,
                "-c",
                "--csv",
                "\t\tOutput the GPU Reachability Matrix as Comma Separated Values",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                NA_CMD_SOURCE_GPU,
                "-s",
                "--src",
                "\t\tSource GPU for displaying number of unidirectional connections",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                NA_CMD_DEST_GPU,
                "-d",
                "--dst",
                "\t\tDestination GPU for displaying number of unidirectional connections",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
#ifdef DEBUG
        {
                NA_CMD_REQ_ENTRY,
                "",
                "--req",
                "\t\tSet Request entry for swicth_id:switchPort:destGpuId:entry_valid:egressPort",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                NA_CMD_RES_ENTRY,
                "",
                "--res",
                "\t\tSet Response entry for swicth_id:switchPort:req_link_id:entry_valid:egressPort",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                NA_CMD_RLID,
                "",
                "--rlid",
                "\t\tSet Requestor Link ID for swicth_id:switchPort:req_link_id",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                NA_CMD_QUICK,
                "-q",
                "--quick",
                "\t\tOnly read Request/Response table entries for valid GPU range",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },

#endif
};

/*****************************************************************************
 Method to Display Usage Info for lwswitch-audit
 *****************************************************************************/
static void naUsage(void * pCmdLine) {
    printf("\n\n");
    printf("    Usage: lwswitch-audit [options]\n");
    printf("\n");
    printf("    Options include:\n");
    cmdline_printOptionsSummary(pCmdLine, 0);
    printf("\n");
    printf("\n\n");
    exit(0);
}

/*****************************************************************************
 Method to Display Help Message for lwswitch-audit
 *****************************************************************************/
static void naDisplayHelpMessage(void * pCmdLine) {
    printf("\n");
    printf("    LWPU LWSwitch audit tool version %s\n"
           "    Reads LWSwitch hardware tables and outputs the current number of\n"
           "    LWlink connections between each pair of GPUs\n",
           FM_VERSION_STRING);

    naUsage(pCmdLine);
}

static int cleanup(naCmdParser_t *pCmdParser, int status)
{
    naCmdParserDestroy(pCmdParser);
    return status;
}

//get the arch type of the first switch device
static int getArch()
{
    LWSWITCH_GET_DEVICES_V2_PARAMS deviceInfo;
    int numSwitches = node::checkVersionGetNumSwitches( &deviceInfo );

    if (numSwitches == 0)
    {
        fprintf(stderr, "No switches found\n");
        exit ( -1);
    }

    lwswitch_device *pLWSwitchDev = NULL;
    int arch;

    for (int i = 0; i < numSwitches; i++) {
        pLWSwitchDev = lwswitch::openSwitchDev( deviceInfo.info[ i ]);

        // the switch could be failed to open, due to the reason
        // that the switch is excluded or degraded.
        if (pLWSwitchDev != NULL) {
            arch = lwswitch::getSwitchArchInfo( pLWSwitchDev );
            break;
        }
    }

    if (pLWSwitchDev) {
        lwswitch_api_free_device( &pLWSwitchDev );
        return arch;
    } else {
        fprintf(stderr, "No switches can be opened\n");
        exit ( -1);
    }
}

int main(int argc, char **argv)
{
    int numSwitches;
    naCmdParser_t *pCmdParser;
    NAReturn_t naRet;

    pCmdParser = naCmdParserInit(argc, argv, naArgs, NA_CMD_COUNT, naUsage, naDisplayHelpMessage);
    if (NULL == pCmdParser)
    {
        return -1;
    }

    if (geteuid() != 0)
    {
        PRINT_VERBOSE("lwswitch-audit is running as non-root\n");
    }

    if (NA_ST_OK != (naRet = naCmdProcessing(pCmdParser)))
    {
        if (naRet == NA_ST_BADPARAM)
        {
            fprintf(stderr, "Unable to run lwswitch-audit: bad command line parameter. \n");
        }
        else
        {
            fprintf(stderr, "Unable to run lwswitch-audit: generic error. \n");
        }
        return cleanup(pCmdParser, -1);
    }

    verbose = pCmdParser->mPrintVerbose;
    if((pCmdParser->mPrintCSV) || (pCmdParser->mSrcGPU >= 0))
    {
        verboseErrors = false;
    }

    //
    // LWSwitch Audit Tool needs to communicate with LWSwitch Driver via IOCTL calls and 
    // it must maintain the application binary interface (ABI) compatibility with the Driver.
    //
    // However, certain customer wants to update only the Fabric Manager package in order to get a very 
    // targeted FM only fix and don't update their Driver (due to longer driver qualification time).
    // Since LWSwitch Audit Tool is part of FM package, it should also work with those Driver versions.
    //
    // Lwrrently all the other drivers (LWSwitch and LWLinkCoreLib) uses the RM version itself. So the
    // version is validated explicitly using the FMGpuDriverVersionCheck interface class and other 
    // driver interfaces will skip the version check. In future, if these
    // drivers are broken into individual drivers, then their respective abstraction class should
    // verify each driver versions.

    //This will throw exception if the version doesn’t match or whitelisted
    FMGpuDriverVersionCheck gpuDrvVersionChk;
    try {
        gpuDrvVersionChk.checkGpuDriverVersionCompatibility("lwswitch-audit");
    } catch (const std::runtime_error &e) {
        fprintf(stderr, "%s\n", e.what());
        exit (-1);
    }

    int switchArch = getArch();

    PRINT_VERBOSE("Switch Arch = %d\n", switchArch);

    node *np=nullptr;
    if (switchArch == LWSWITCH_GET_INFO_INDEX_ARCH_LR10)
    {
        np = new explorer16Delta;
    } 
    else if (switchArch == LWSWITCH_GET_INFO_INDEX_ARCH_SV10)
    {
        np = new explorer16Juno;
    }
    else
    {
        fprintf(stderr, "Unknown switch architecture %d\n", switchArch);
        exit (-1);
    }
    np->openSwitchDevices();
    numSwitches = np->getMaxSwitch();

#if DEBUG
    if(pCmdParser->mSetDestGPU >= 0) 
    {
        if(!np->setRequestEntry(pCmdParser->mSwitch, pCmdParser->mSwitchPort, 
                                pCmdParser->mSetDestGPU, pCmdParser->mValid, pCmdParser->mEgressPort))
        {
            fprintf(stderr, "Unable to set request entry\n");
            return -1;
        }
        return 0;
    }

    if(pCmdParser->mSetDestRLID >= 0) 
    {
        if(!np->setResponseEntry(pCmdParser->mSwitch, pCmdParser->mSwitchPort, 
                                pCmdParser->mSetDestRLID, pCmdParser->mValid, pCmdParser->mEgressPort))
        {
            fprintf(stderr, "Unable to set response entry\n");
            return -1;
        }
        return 0;
    }

    if(pCmdParser->mSetRLID >= 0) 
    {
        naSetRLID(pCmdParser->mSwitch, pCmdParser->mSwitchPort, pCmdParser->mSetRLID);
    }

#endif
    //Initialize the request and response tables
    int numReqLinkIds = np->getMaxGpu() * np->getMaxSwitchPerBaseboard();
    naNodeTables_t responseTables(numSwitches, 
                                            naSwitchTables_t(np->getNumSwitchPorts(),
                                                                     naPortTable_t(np->getNumResIds(),
                                                                                           DEST_UNREACHABLE)));
    naNodeTables_t requestTables(numSwitches, naSwitchTables_t(np->getNumSwitchPorts(), 
                                                                               naPortTable_t(np->getNumReqIds(),
                                                                                                    DEST_UNREACHABLE)));
    uint32_t maxTableEntries = UINT_MAX;
#ifdef DEBUG
    if(pCmdParser->mQuick)
        maxTableEntries = 256;
#endif
    //Read all request tables from switch
    if(!np->readTables(requestTables, numSwitches, maxTableEntries, true))
    {
        fprintf(stderr, "Unable to read Request Tables\n");
        return cleanup(pCmdParser, -1);
    }

    //read all response tables from switch
    if(!np->readTables(responseTables, numSwitches, maxTableEntries, false))
    {
        fprintf(stderr, "Unable to read Response Tables\n");
        return cleanup(pCmdParser, -1);
    }
    //Compute the number of paths between each pair of GPUs
    naPathsMatrix_t pathsMatrix(np->getMaxGpu(), std::vector<int>(np->getMaxGpu(), 0));

    if(!naComputePaths(requestTables, responseTables, pathsMatrix, numSwitches, np))
    {
        fprintf(stderr, "Unable to compute paths matrix\n");
        return cleanup(pCmdParser, -1);
    }

    np->closeSwitchDevices();
    //Print full matrix in CSV/Table format or just the number  of connection between 
    //src/dst GPU pair
    if(pCmdParser->mSrcGPU < 0)
        naPrintPaths("GPU Reachability Matrix", pathsMatrix, pCmdParser->mPrintCSV, np, pCmdParser->mPrintFullMatrix);
    else
    {
        if((pCmdParser->mSrcGPU == pCmdParser->mDestGPU) && (pathsMatrix[pCmdParser->mSrcGPU][pCmdParser->mDestGPU] == 0))
            printf("X\n");
        else
            printf("%d\n", pathsMatrix[pCmdParser->mSrcGPU][pCmdParser->mDestGPU]);
    }
    return cleanup(pCmdParser, 0);
}
