/*************************************************************************
 * Copyright (c) 2015-2019, LWPU CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "lwmlwrap.h"

#ifndef LWML_DIRECT
#include <dlfcn.h>
#include "core.h"

static enum { lwmlUninitialized, lwmlInitializing, lwmlInitialized, lwmlError } lwmlState = lwmlUninitialized;

static lwmlReturn_t (*lwmlInternalInit)(void);
static lwmlReturn_t (*lwmlInternalShutdown)(void);
static lwmlReturn_t (*lwmlInternalDeviceGetHandleByPciBusId)(const char* pciBusId, lwmlDevice_t* device);
static lwmlReturn_t (*lwmlInternalDeviceGetIndex)(lwmlDevice_t device, unsigned* index);
static const char* (*lwmlInternalErrorString)(lwmlReturn_t r);
static lwmlReturn_t (*lwmlInternalDeviceGetLwLinkState)(lwmlDevice_t device, unsigned int link, lwmlEnableState_t *isActive);
static lwmlReturn_t (*lwmlInternalDeviceGetPciInfo)(lwmlDevice_t device, lwmlPciInfo_t* pci);
static lwmlReturn_t (*lwmlInternalDeviceGetLwLinkRemotePciInfo)(lwmlDevice_t device, unsigned int link, lwmlPciInfo_t *pci);
static lwmlReturn_t (*lwmlInternalDeviceGetLwLinkCapability)(lwmlDevice_t device, unsigned int link,
    lwmlLwLinkCapability_t capability, unsigned int *capResult);
static lwmlReturn_t (*lwmlInternalDeviceGetMinorNumber)(lwmlDevice_t device, unsigned int* minorNumber);


ncclResult_t wrapLwmlSymbols(void) {
  if (lwmlState == lwmlInitialized)
    return ncclSuccess;
  if (lwmlState == lwmlError)
    return ncclSystemError;

  if (__sync_bool_compare_and_swap(&lwmlState, lwmlUninitialized, lwmlInitializing) == false) {
    // Another thread raced in front of us. Wait for it to be done.
    while (lwmlState == lwmlInitializing) pthread_yield();
    return (lwmlState == lwmlInitialized) ? ncclSuccess : ncclSystemError;
  }

  static void* lwmlhandle = NULL;
  void* tmp;
  void** cast;

  lwmlhandle=dlopen("liblwidia-ml.so.1", RTLD_NOW);
  if (!lwmlhandle) {
    WARN("Failed to open liblwidia-ml.so.1");
    goto teardown;
  }

#define LOAD_SYM(handle, symbol, funcptr) do {         \
    cast = (void**)&funcptr;                             \
    tmp = dlsym(handle, symbol);                         \
    if (tmp == NULL) {                                   \
      WARN("dlsym failed on %s - %s", symbol, dlerror());\
      goto teardown;                                     \
    }                                                    \
    *cast = tmp;                                         \
  } while (0)

#define LOAD_SYM_OPTIONAL(handle, symbol, funcptr) do {\
    cast = (void**)&funcptr;                             \
    tmp = dlsym(handle, symbol);                         \
    if (tmp == NULL) {                                   \
      INFO(NCCL_INIT,"dlsym failed on %s, ignoring", symbol); \
    }                                                    \
    *cast = tmp;                                         \
  } while (0)

  LOAD_SYM(lwmlhandle, "lwmlInit", lwmlInternalInit);
  LOAD_SYM(lwmlhandle, "lwmlShutdown", lwmlInternalShutdown);
  LOAD_SYM(lwmlhandle, "lwmlDeviceGetHandleByPciBusId", lwmlInternalDeviceGetHandleByPciBusId);
  LOAD_SYM(lwmlhandle, "lwmlDeviceGetIndex", lwmlInternalDeviceGetIndex);
  LOAD_SYM(lwmlhandle, "lwmlErrorString", lwmlInternalErrorString);
  LOAD_SYM(lwmlhandle, "lwmlDeviceGetPciInfo", lwmlInternalDeviceGetPciInfo);
  LOAD_SYM(lwmlhandle, "lwmlDeviceGetMinorNumber", lwmlInternalDeviceGetMinorNumber);
  LOAD_SYM_OPTIONAL(lwmlhandle, "lwmlDeviceGetLwLinkState", lwmlInternalDeviceGetLwLinkState);
  LOAD_SYM_OPTIONAL(lwmlhandle, "lwmlDeviceGetLwLinkRemotePciInfo", lwmlInternalDeviceGetLwLinkRemotePciInfo);
  LOAD_SYM_OPTIONAL(lwmlhandle, "lwmlDeviceGetLwLinkCapability", lwmlInternalDeviceGetLwLinkCapability);

  lwmlState = lwmlInitialized;
  return ncclSuccess;

teardown:
  lwmlInternalInit = NULL;
  lwmlInternalShutdown = NULL;
  lwmlInternalDeviceGetHandleByPciBusId = NULL;
  lwmlInternalDeviceGetIndex = NULL;
  lwmlInternalDeviceGetPciInfo = NULL;
  lwmlInternalDeviceGetMinorNumber = NULL;
  lwmlInternalDeviceGetLwLinkState = NULL;
  lwmlInternalDeviceGetLwLinkRemotePciInfo = NULL;
  lwmlInternalDeviceGetLwLinkCapability = NULL;

  if (lwmlhandle != NULL) dlclose(lwmlhandle);
  lwmlState = lwmlError;
  return ncclSystemError;
}


ncclResult_t wrapLwmlInit(void) {
  if (lwmlInternalInit == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  lwmlReturn_t ret = lwmlInternalInit();
  if (ret != LWML_SUCCESS) {
    WARN("lwmlInit() failed: %s",
        lwmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapLwmlShutdown(void) {
  if (lwmlInternalShutdown == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  lwmlReturn_t ret = lwmlInternalShutdown();
  if (ret != LWML_SUCCESS) {
    WARN("lwmlShutdown() failed: %s ",
        lwmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapLwmlDeviceGetHandleByPciBusId(const char* pciBusId, lwmlDevice_t* device) {
  if (lwmlInternalDeviceGetHandleByPciBusId == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  lwmlReturn_t ret = lwmlInternalDeviceGetHandleByPciBusId(pciBusId, device);
  if (ret != LWML_SUCCESS) {
    WARN("lwmlDeviceGetHandleByPciBusId() failed: %s ",
        lwmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapLwmlDeviceGetIndex(lwmlDevice_t device, unsigned* index) {
  if (lwmlInternalDeviceGetIndex == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  lwmlReturn_t ret = lwmlInternalDeviceGetIndex(device, index);
  if (ret != LWML_SUCCESS) {
    WARN("lwmlDeviceGetIndex() failed: %s ",
        lwmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapLwmlDeviceGetPciInfo(lwmlDevice_t device, lwmlPciInfo_t* pci) {
  if (lwmlInternalDeviceGetPciInfo == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  lwmlReturn_t ret = lwmlInternalDeviceGetPciInfo(device, pci);
  if (ret != LWML_SUCCESS) {
    WARN("lwmlDeviceGetPciInfo() failed: %s ",
        lwmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapLwmlDeviceGetMinorNumber(lwmlDevice_t device, unsigned int* minorNumber) {
  if (lwmlInternalDeviceGetMinorNumber == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  lwmlReturn_t ret = lwmlInternalDeviceGetMinorNumber(device, minorNumber);
  if (ret != LWML_SUCCESS) {
    WARN("lwmlDeviceGetMinorNumber() failed: %s ",
        lwmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapLwmlDeviceGetLwLinkState(lwmlDevice_t device, unsigned int link, lwmlEnableState_t *isActive) {
  if (lwmlInternalDeviceGetLwLinkState == NULL) {
    /* Do not warn, this symbol is optional. */
    return ncclInternalError;
  }
  lwmlReturn_t ret = lwmlInternalDeviceGetLwLinkState(device, link, isActive);
  if (ret != LWML_SUCCESS) {
    if (ret != LWML_ERROR_NOT_SUPPORTED)
      INFO(NCCL_INIT,"lwmlDeviceGetLwLinkState() failed: %s ",
          lwmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapLwmlDeviceGetLwLinkRemotePciInfo(lwmlDevice_t device, unsigned int link, lwmlPciInfo_t *pci) {
  if (lwmlInternalDeviceGetLwLinkRemotePciInfo == NULL) {
    /* Do not warn, this symbol is optional. */
    return ncclInternalError;
  }
  lwmlReturn_t ret = lwmlInternalDeviceGetLwLinkRemotePciInfo(device, link, pci);
  if (ret != LWML_SUCCESS) {
    if (ret != LWML_ERROR_NOT_SUPPORTED)
      INFO(NCCL_INIT,"lwmlDeviceGetLwLinkRemotePciInfo() failed: %s ",
          lwmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapLwmlDeviceGetLwLinkCapability(lwmlDevice_t device, unsigned int link,
    lwmlLwLinkCapability_t capability, unsigned int *capResult) {
  if (lwmlInternalDeviceGetLwLinkCapability == NULL) {
    /* Do not warn, this symbol is optional. */
    return ncclInternalError;
  }
  lwmlReturn_t ret = lwmlInternalDeviceGetLwLinkCapability(device, link, capability, capResult);
  if (ret != LWML_SUCCESS) {
    if (ret != LWML_ERROR_NOT_SUPPORTED)
      INFO(NCCL_INIT,"lwmlDeviceGetLwLinkCapability() failed: %s ",
          lwmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}
#endif
