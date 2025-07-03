#include <fcntl.h>
#include <limits.h>
#include <map>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <syslog.h>
#include <unistd.h>
#include <vector>

#include "commandline/commandline.h"
#include "lwos.h"
#include "lwswitch_audit_node.h"
#include "logging.h"
#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"
#include "lwswitch_audit_cmd_parser.h"
#include "lwswitch_audit_tables.h"
#include "lwswitch_audit_paths.h"
#include "lwswitch_audit_dev.h"
#include "lwswitch_audit_logging.h"
extern "C"
{
#include "lwswitch_user_linux.h"
}


bool verbose=false;
bool verbose_errors=true;

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
                "\t\tSet Request entry for swicth_id:switch_port:dest_gpu_id:entry_valid:egress_port",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                NA_CMD_RES_ENTRY,
                "",
                "--res",
                "\t\tSet Response entry for swicth_id:switch_port:req_link_id:entry_valid:egress_port",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                NA_CMD_RLID,
                "",
                "--rlid",
                "\t\tSet Requestor Link ID for swicth_id:switch_port:req_link_id",
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
    printf("    LWPU LWSwitch audit tool\n"
           "    Reads LWSwitch hardware tables and outputs the current number of\n"
           "    LWlink connections between each pair of GPUs\n");

    naUsage(pCmdLine);
}

static int cleanup(naCmdParser_t *pCmdParser, int status)
{
    naCmdParserDestroy(pCmdParser);
    return status;
}
//check version and return number of LWSwitches found 
static int checkVersion()
{
    LWSWITCH_GET_DEVICES_PARAMS params;
    LW_STATUS status;
    status = lwswitch_api_get_devices(&params);
    if (status != LW_OK)
    {
        if (status == LW_ERR_LIB_RM_VERSION_MISMATCH)
        {
            fprintf(stderr, "lwswitch-audit version is incompatible with LWSwitch driver. Please update with matching LWPU driver package");
            exit(-1);
        }
        // all other errors, log the error code and bail out
        fprintf(stderr, "lwswitch-audit:failed to query device information from LWSwitch driver, return status: %d\n", status);
        exit(-1);
    }
    if (params.deviceCount <= 0)
    {
        fprintf(stderr, "No LWSwitches found\n");
        exit(-1);
    }
    return params.deviceCount;
}

int main(int argc, char **argv)
{
    int num_switches;
    naCmdParser_t *pCmdParser;
    NAReturn_t naRet;

    pCmdParser = naCmdParserInit(argc, argv, naArgs, NA_CMD_COUNT, naUsage, naDisplayHelpMessage);
    if (NULL == pCmdParser)
    {
        return -1;
    }

    /* Check if the user is not root. Return if not */
    if (geteuid() != 0)
    {
        fprintf(stderr, "lwswitch-audit is running as non-root. Only usage/help available.\n");
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
        verbose_errors = false;
    }

    num_switches = checkVersion();

    //read switch instance to switch id maps
    if(naReadSwitchIDs(num_switches) == false)
    {
        fprintf(stderr, "Unable to access LWSwitches\n");
        return cleanup(pCmdParser, -1);
    }
#ifdef DEBUG
    if(pCmdParser->mSetDestGPU >= 0) 
    {
        if(!naSetRequestEntry(pCmdParser->mSwitch, pCmdParser->mSwitchPort, 
                                pCmdParser->mSetDestGPU, pCmdParser->mValid, pCmdParser->mEgressPort))
        {
            fprintf(stderr, "Unable to set request entry\n");
            return -1;
        }
        return 0;
    }

    if(pCmdParser->mSetDestRLID >= 0) 
    {
        if(!naSetResponseEntry(pCmdParser->mSwitch, pCmdParser->mSwitchPort, 
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
    int num_req_link_ids = MAX_GPU * MAX_SWITCH_PER_BASEBOARD;
    naNodeResponseTables_t response_tables(num_switches, 
                                            naSwitchResponseTables_t(NUM_SWITCH_PORTS,
                                                                     naPortResponseTable_t(num_req_link_ids,
                                                                                           DEST_UNREACHABLE)));
    naNodeRequestTables_t request_tables(num_switches, naSwitchRequestTables_t(NUM_SWITCH_PORTS, 
                                                                               naPortRequestTable_t(MAX_GPU)));
    for (int i =0; i < num_switches; i ++)
        for (int j = 0 ; j < NUM_SWITCH_PORTS; j ++)
            for( int k =0; k < MAX_GPU; k++)
            {
                request_tables[i][j][k].egress_port_id = DEST_UNREACHABLE;
                request_tables[i][j][k].count = 0;
            }

    uint32_t num_table_entries = NUM_TABLE_ENTRIES;
#ifdef DEBUG
    if(pCmdParser->mQuick)
        num_table_entries = MAX_GPU * REQ_ENTRIES_PER_GPU;
#endif
    //Read all request tables from switch
    if(!naReadRequestTables(request_tables, num_switches, num_table_entries))
    {
        fprintf(stderr, "Unable to read Request Tables\n");
        return cleanup(pCmdParser, -1);
    }

#ifdef DEBUG
    if(pCmdParser->mQuick)
        num_table_entries = MAX_GPU * MAX_SWITCH_PER_BASEBOARD;
#endif
    //read all response tables from switch
    if(!naReadResponseTables(response_tables, num_switches, num_table_entries))
    {
        fprintf(stderr, "Unable to read Response Tables\n");
        return cleanup(pCmdParser, -1);
    }

    //Compute the number of paths between each pair of GPUs
    int paths_matrix[MAX_GPU][MAX_GPU];
    memset(paths_matrix, 0, sizeof(paths_matrix));
    if(!naComputePaths(request_tables, response_tables, paths_matrix, num_switches))
    {
        fprintf(stderr, "Unable to compute paths matrix\n");
        return cleanup(pCmdParser, -1);
    }

    int num_bad_link_ids=0;
    num_bad_link_ids = naCheckReqLinkIDs(num_switches, paths_matrix);
    //Print the number of incorrectly programmed Requestor Link IDs found
    if(num_bad_link_ids > 0)
    {
        PRINT_ERROR_VERBOSE("%d Bad Link IDs found\n", num_bad_link_ids);
    }
  
    //Print full matrix in CSV/Table format or just the number  of connection between 
    //src/dst GPU pair
    if(pCmdParser->mSrcGPU < 0)
        naPrintPaths("GPU Reachability Matrix", paths_matrix, pCmdParser->mPrintCSV);
    else
    {
        if((pCmdParser->mSrcGPU == pCmdParser->mDestGPU) && (paths_matrix[pCmdParser->mSrcGPU][pCmdParser->mDestGPU] == 0))
            printf("X\n");
        else
            printf("%d\n", paths_matrix[pCmdParser->mSrcGPU][pCmdParser->mDestGPU]);
    }
 
    return cleanup(pCmdParser, 0);
}



