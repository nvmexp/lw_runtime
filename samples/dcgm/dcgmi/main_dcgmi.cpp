#include <iostream>
#include <stdexcept>
#include <sstream>
#include "dcgm_agent.h"
#include "dcgm_agent_internal.h"
#include "dcgm_client_internal.h"
#include "timelib.h"
#include <signal.h>
#include "CommandLineParser.h"

const etblDCGMClientInternal *g_pEtblClient = NULL;
const etblDCGMEngineInternal *g_pEtblAgent = NULL;
CommandLineParser * g_cl;

void processCommandLine(int argc, char *argv[]);

/*****************************************************************************/
void sig_handler(int signum)
{
    delete g_cl;
    exit(128 + signum);  // Exit with UNIX fatal error signal code for the received signal
}


/*****************************************************************************
* This method provides mechanism to register Sighandler callbacks for 
* SIGHUP, SIGINT, SIGQUIT, and SIGTERM
*****************************************************************************/
int InstallCtrlHandler()
{
    if (signal(SIGHUP, sig_handler) == SIG_ERR)
    {    
        return -1; 
    }
    if (signal(SIGINT, sig_handler) == SIG_ERR)
    {    
        return -1; 
    }
    if (signal(SIGQUIT, sig_handler) == SIG_ERR)
    {    
        return -1; 
    }
    if (signal(SIGTERM, sig_handler) == SIG_ERR)
    {    
        return -1; 
    }

    return 0;    
}


int main(int argc, char *argv[])
{
    int ret = 0;
    dcgmReturn_t result;
    dcgmHandle_t pLwcmHandle;

    g_cl = new CommandLineParser();

    // Obtain a pointer to Client's internal table
    result = dcgmInternalGetExportTable((const void**)&g_pEtblClient,
                                        &ETID_DCGMClientInternal);
    if (DCGM_ST_OK  != result) 
    {
        printf("Error: Can't get the export table. Return: %d\n", result);
        ret = -1;
        goto cleanup;
    }
    
    result = dcgmInternalGetExportTable((const void**)&g_pEtblAgent,
                                        &ETID_DCGMEngineInternal);
    if (DCGM_ST_OK  != result) 
    {
        printf("Error: Can't get the export table. Return: %d\n", result);
        ret = -1;
        goto cleanup;
    }

    // Install the signal handler
    InstallCtrlHandler();

    try
    {
        result = (dcgmReturn_t) g_cl->processCommandLine(argc, argv);
    } 
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        ret = -2;
        goto cleanup;
    }

    // Check if any errors thrown in dcgmi
    if (DCGM_ST_OK != result)
    {
        ret = result;
        goto cleanup;
    }


cleanup:
    delete g_cl;
    dcgmShutdown();
    return ret;
}

