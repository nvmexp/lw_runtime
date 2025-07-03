
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <pthread.h>
#include <string>
#include <stdint.h>
#include "lwlink_train_cmd_parser.h"
#include <mpi.h>

#include "LocalFMGpuMgr.h"
#include "helper.h"
#include "json/json.h"
#include "master.h"
#include "slave.h"

extern "C" 
{
    #include "lwpu-modprobe-utils.h"
}

using namespace std;

bool verbose = false;
bool verboseErrors = true;
bool verboseDebug = false;

struct all_args naArgs[] = {
        {
                NT_CMD_HELP,
                "-h",
                "--help",
                "\t\tDisplays help information",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                NT_CMD_VERBOSE,
                "-v",
                "--verbose",
                "\t\tVerbose output",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                NT_CMD_BATCH_MODE,
                "-b",
                "--batch",
                "\t\tBatch mode for running list of steps instead of interactive menu",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        }
};

/*****************************************************************************
 Method to Display Usage Info for lwswitch-audit
 *****************************************************************************/
static void naUsage(void * pCmdLine) {
    printf("\n\n");
    printf("    Usage: lwlink-train [options]\n");
    printf("\n");
    printf("    Options include:\n");
    cmdline_printOptionsSummary(pCmdLine, 0);

    // printf("\n\nTraining steps are:\n");
    // list_steps();
    // printf("\n");
    // printf("\n\n");
    exit(0);
}

/*****************************************************************************
 Method to Display Help Message for lwswitch-audit
 *****************************************************************************/
static void naDisplayHelpMessage(void * pCmdLine) {
    printf("\n");
    printf("    LWPU LWLink training tool\n"
           "    Allows running various training steps in interactive or batch mode\n"
           "    Runs as either client/server pair or standalone\n");

    naUsage(pCmdLine);
}

static int cleanup(ntCmdParser_t *pCmdParser, int status)
{
    ntCmdParserDestroy(pCmdParser);
    return status;
}

int main(int argc, char **argv)
{
    ntCmdParser_t *pCmdParser;
    NTReturn_t naRet;

    pCmdParser = ntCmdParserInit(argc, argv, naArgs, NT_CMD_COUNT, naUsage, naDisplayHelpMessage);
    if (NULL == pCmdParser)
    {
        return -1;
    }

    if (NT_ST_OK != (naRet = ntCmdProcessing(pCmdParser)))
    {
        if (naRet == NT_ST_BADPARAM)
        {
            fprintf(stderr, "Unable to run lwlink-train: bad command line parameter. \n");
        }
        else
        {
            fprintf(stderr, "Unable to run lwlink-train: generic error. \n");
        }
        return cleanup(pCmdParser, -1);
    }

    verbose = pCmdParser->mPrintVerbose;

    std::clog.rdbuf(NULL);

    if (!lwidia_lwlink_mknod()) 
    {
        printf("failed to create node for /dev/lwpu-lwlink\n");
    }

    LocalFMGpuMgr *p = new LocalFMGpuMgr();
    p->initializeAllGpus();

    int rank, size;
    MPI_Comm new_comm;

    MPI_Init( &argc, &argv );
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    if (open_lwlinklib_driver(rank)) 
    {
       printf(" failed to open LWLinkLib driver\n");
       return 1;
    }

    if (rank == 0) {
        if(pCmdParser->mBatchMode == true) 
            startMaster(size);
        else 
            show_menu(size);
    }
    else {
        startSlave(size, rank);
    }

    MPI_Finalize();
    return 0;
}
