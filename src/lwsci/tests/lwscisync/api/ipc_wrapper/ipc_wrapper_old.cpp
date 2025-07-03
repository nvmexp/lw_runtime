/*
 * Copyright (c) 2019-2020 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <signal.h>
#include "ipc_wrapper_old.h"
#include "ipc_wrapper_old_core.h"

/* change this to
 * a list or a reallocable array
 * for adjustable numebr of IpcWrappers
 */
#define MAX_IPC_WRAPPERS (10)
static IpcWrapperOld globalIpcWrappers[MAX_IPC_WRAPPERS];
static size_t numWrappers;

void sigHandler(int sigNum)
{
    int i;
    for (i = 0; i < numWrappers; ++i) {
        ipcDeinit(globalIpcWrappers[i]);
    }
}

void setupTerminationHandlers(void)
{
    signal(SIGINT, sigHandler);
    signal(SIGTERM, sigHandler);
    signal(SIGHUP, sigHandler);
    signal(SIGQUIT, sigHandler);
    signal(SIGABRT, sigHandler);
}

LwSciIpcEndpoint ipcWrapperGetEndpoint(IpcWrapperOld ipcWrapper)
{
    return ipcWrapper->endpoint;
}

int registerGlobalWrapper(IpcWrapperOld ipcWrapper)
{
    if (numWrappers == MAX_IPC_WRAPPERS) {
        return -1;
    }
    globalIpcWrappers[numWrappers++] = ipcWrapper;
    return 0;
}

void deregisterGlobalWrapper(IpcWrapperOld ipcWrapper)
{
    int i;
    for (i = 0; i < numWrappers; ++i) {
        if (globalIpcWrappers[i] == ipcWrapper) {
            globalIpcWrappers[i] = globalIpcWrappers[numWrappers-1];
            globalIpcWrappers[numWrappers-1] = NULL;
            numWrappers--;
            break;
        }
    }
}
