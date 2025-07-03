
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

#include "socket_interface.h"
#include "lwlink_lib_ioctl.h"
#include "helper.h"
#include "LocalFMGpuMgr.h"
#include "commandline/commandline.h"
#include "lwlink_train_cmd_parser.h"


extern "C"
{
#include "lwpu-modprobe-utils.h"
}

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
        },
        {
                NT_CMD_TRAIN,
                "-t",
                "--train",
                "\t\tList of training steps to be run in batch mode",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {

                NT_CMD_IP_ADDRESS,
                "-a",
                "--address",
                "\t\tIP address of server",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                NT_CMD_CLIENT,
                "-c",
                "--client",
                "\t\tRun as client(IP address of server needed)",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                NT_CMD_SERVER,
                "-s",
                "--server",
                "\t\tRun as server",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                NT_CMD_NODE_ID,
                "-n",
                "--nodeid",
                "\t\tNode ID",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
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

    printf("\n\nTraining steps are:\n");
    list_steps();
    printf("\n");
    printf("\n\n");
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
    //std::cout <<"starting LWLinkLib driver test program\n";
    //std::cout <<"argc="<< argc << "\n";
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

    if (!lwidia_lwlink_mknod()) 
    {
        printf("failed to create node for /dev/lwpu-lwlink\n");
    }

    printf("before initializeAllGpus\n");
    LocalFMGpuMgr *p = new LocalFMGpuMgr();
    p->initializeAllGpus();
    printf("after initializeAllGpus\n");

    if (open_lwlinklib_driver(pCmdParser->mNodeId)) 
    {
        printf(" failed to open LWLinkLib driver\n");
        return 1;
    }

    get_device_information();

    if(pCmdParser->mBatchMode == false) 
    {
        show_train_menu();
        return 0;
    }
    else
    {
        run_training_steps(pCmdParser);
        return 0;
    }


    //discover_intra_connections();

    discover_intra_node_connections_all_steps();
    //show_train_menu();
    if(argc > 2) {
        std::string ipaddr(argv[2]);
        run_multi_node_client(argv[2]);

    } else {
        run_multi_node_server();
    }
    return 0;
    show_train_menu();
}
