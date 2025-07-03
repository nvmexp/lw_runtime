#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include "dcgm_structs.h"
#include "dcgm_agent.h"


/*********************************************************************
 * Test for DCGM Stub Library
 *
 * This program calls the following DCGM APIs:
 *   dcgmInit
 *   dcgmStartEmbedded
 *   dcgmGetAllDevices
 *   dcgmShutdown 
 *
 * If libdcgm.so.1 is present in the current environment, api calls
 * shoud succeed, otherwise the stub (static) library is used.
 * 
 ********************************************************************/

/* Create function pointer variables init, get and shut */
int (*init)(void);
int (*startEmbedded)(dcgmOperationMode_t opMpde, dcgmHandle_t *dcgmHandle);
int (*get)(dcgmHandle_t dcgmHandle, unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES], int *count);
int (*shut)(void);

int main(void)
{
    dcgmHandle_t dcgmHandle;
    dcgmOperationMode_t opMode;
    unsigned int devices = DCGM_MAX_NUM_DEVICES;
    int count;
    char IP[] = "127.0.0.1";
    int ret;

    // Calling APIs, if no shared library is present, use symbol from stub library
    void *handle = dlopen("libdcgm.so.1", RTLD_NOW);

    if (handle)
    {
        init = dlsym(handle, "dcgmInit");
    }

    if (!init)
    {
        init = dcgmInit;
    }

    ret = init();
    printf("dcgmInit() returned: %d\n", ret);

    if (handle)
    {
        startEmbedded = dlsym(handle, "dcgmStartEmbedded");
    }

    if (!startEmbedded)
    {
        startEmbedded = dcgmStartEmbedded;
    }
    
    ret = startEmbedded(DCGM_OPERATION_MODE_AUTO, &dcgmHandle);
    printf("startEmbedded returned %d\n", ret);


    if (handle)
    {
        get = dlsym(handle, "dcgmGetAllDevices");
    }

    if (!get)
    {
        get = dcgmGetAllDevices;
    }

    ret = get(dcgmHandle, &devices, &count);
    printf("dcgmGetAllDevices() returned: %d\n", ret);

    if (handle)
    {
        shut = dlsym(handle, "dcgmShutdown");
    }
    
    if(!shut)
    {
        shut = dcgmShutdown;
    }
    
    ret = shut();
    printf("dcgmShutdown() returned: %d\n", ret);
 
    return 0;
}

