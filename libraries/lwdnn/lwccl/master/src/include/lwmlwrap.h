/*************************************************************************
 * Copyright (c) 2015-2019, LWPU CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_LWMLWRAP_H_
#define NCCL_LWMLWRAP_H_

#include "lwcl.h"

//#define LWML_DIRECT 1
#ifdef LWML_DIRECT
#include "lwml.h"

#define LWMLCHECK(cmd) do {                              \
    lwmlReturn_t e = cmd;                                \
    if( e != LWML_SUCCESS ) {                            \
      WARN("LWML failure '%s'", lwmlErrorString(e));     \
      return ncclSystemError;                            \
    }                                                    \
} while(false)

static ncclResult_t wrapLwmlSymbols(void) { return ncclSuccess; }
static ncclResult_t wrapLwmlInit(void) { LWMLCHECK(lwmlInit()); return ncclSuccess; }
static ncclResult_t wrapLwmlShutdown(void) { LWMLCHECK(lwmlShutdown()); return ncclSuccess; }
static ncclResult_t wrapLwmlDeviceGetHandleByPciBusId(const char* pciBusId, lwmlDevice_t* device) {
  LWMLCHECK(lwmlDeviceGetHandleByPciBusId(pciBusId, device));
  return ncclSuccess;
}
static ncclResult_t wrapLwmlDeviceGetIndex(lwmlDevice_t device, unsigned* index) {
  LWMLCHECK(lwmlDeviceGetIndex(device, index));
  return ncclSuccess;
}
static ncclResult_t wrapLwmlDeviceGetHandleByIndex(unsigned int index, lwmlDevice_t *device) {
  LWMLCHECK(lwmlDeviceGetHandleByIndex(index,device));
  return ncclSuccess;
}
static ncclResult_t wrapLwmlDeviceGetHandleByPciInfo(lwmlDevice_t device, lwmlPciInfo_t* pci) {
  LWMLCHECK(lwmlDeviceGetPciInfo(device, pci));
  return ncclSuccess;
}
static ncclResult_t wrapLwmlDeviceGetLwLinkState(lwmlDevice_t device, unsigned int link, lwmlEnableState_t *isActive) {
  LWMLCHECK(lwmlDeviceGetLwLinkState(device, link, isActive));
  return ncclSuccess;
}
static ncclResult_t wrapLwmlDeviceGetLwLinkRemotePciInfo(lwmlDevice_t device, unsigned int link, lwmlPciInfo_t *pci) {
  LWMLCHECK(lwmlDeviceGetLwLinkRemotePciInfo(device, link, pci));
  return ncclSuccess;
}
static ncclResult_t wrapLwmlDeviceGetLwLinkCapability(lwmlDevice_t device, unsigned int link,
                                                   lwmlLwLinkCapability_t capability, unsigned int *capResult) {
  LWMLCHECK(lwmlDeviceGetLwLinkCapability(device, link, capability, capResult));
  return ncclSuccess;
}
static ncclResult_t wrapLwmlDeviceGetMinorNumber(lwmlDevice_t device, unsigned int* minorNumber) {
  LWMLCHECK(lwmlDeviceGetMinorNumber(device, minorNumber));
  return ncclSuccess;
}
#else
// Dynamically handle dependencies on LWML

/* Extracted from lwml.h */
typedef struct lwmlDevice_st* lwmlDevice_t;
#define LWML_DEVICE_PCI_BUS_ID_BUFFER_SIZE   16

typedef enum lwmlEnableState_enum
{
    LWML_FEATURE_DISABLED    = 0,     //!< Feature disabled
    LWML_FEATURE_ENABLED     = 1      //!< Feature enabled
} lwmlEnableState_t;

typedef enum lwmlLwLinkCapability_enum
{
    LWML_LWLINK_CAP_P2P_SUPPORTED = 0,     // P2P over LWLink is supported
    LWML_LWLINK_CAP_SYSMEM_ACCESS = 1,     // Access to system memory is supported
    LWML_LWLINK_CAP_P2P_ATOMICS   = 2,     // P2P atomics are supported
    LWML_LWLINK_CAP_SYSMEM_ATOMICS= 3,     // System memory atomics are supported
    LWML_LWLINK_CAP_SLI_BRIDGE    = 4,     // SLI is supported over this link
    LWML_LWLINK_CAP_VALID         = 5,     // Link is supported on this device
    // should be last
    LWML_LWLINK_CAP_COUNT
} lwmlLwLinkCapability_t;

typedef enum lwmlReturn_enum
{
    LWML_SUCCESS = 0,                   //!< The operation was successful
    LWML_ERROR_UNINITIALIZED = 1,       //!< LWML was not first initialized with lwmlInit()
    LWML_ERROR_ILWALID_ARGUMENT = 2,    //!< A supplied argument is invalid
    LWML_ERROR_NOT_SUPPORTED = 3,       //!< The requested operation is not available on target device
    LWML_ERROR_NO_PERMISSION = 4,       //!< The current user does not have permission for operation
    LWML_ERROR_ALREADY_INITIALIZED = 5, //!< Deprecated: Multiple initializations are now allowed through ref counting
    LWML_ERROR_NOT_FOUND = 6,           //!< A query to find an object was unsuccessful
    LWML_ERROR_INSUFFICIENT_SIZE = 7,   //!< An input argument is not large enough
    LWML_ERROR_INSUFFICIENT_POWER = 8,  //!< A device's external power cables are not properly attached
    LWML_ERROR_DRIVER_NOT_LOADED = 9,   //!< LWPU driver is not loaded
    LWML_ERROR_TIMEOUT = 10,            //!< User provided timeout passed
    LWML_ERROR_IRQ_ISSUE = 11,          //!< LWPU Kernel detected an interrupt issue with a GPU
    LWML_ERROR_LIBRARY_NOT_FOUND = 12,  //!< LWML Shared Library couldn't be found or loaded
    LWML_ERROR_FUNCTION_NOT_FOUND = 13, //!< Local version of LWML doesn't implement this function
    LWML_ERROR_CORRUPTED_INFOROM = 14,  //!< infoROM is corrupted
    LWML_ERROR_GPU_IS_LOST = 15,        //!< The GPU has fallen off the bus or has otherwise become inaccessible
    LWML_ERROR_RESET_REQUIRED = 16,     //!< The GPU requires a reset before it can be used again
    LWML_ERROR_OPERATING_SYSTEM = 17,   //!< The GPU control device has been blocked by the operating system/cgroups
    LWML_ERROR_LIB_RM_VERSION_MISMATCH = 18,   //!< RM detects a driver/library version mismatch
    LWML_ERROR_IN_USE = 19,             //!< An operation cannot be performed because the GPU is lwrrently in use
    LWML_ERROR_UNKNOWN = 999            //!< An internal driver error oclwrred
} lwmlReturn_t;

typedef struct lwmlPciInfo_st
{
    char busId[LWML_DEVICE_PCI_BUS_ID_BUFFER_SIZE]; //!< The tuple domain:bus:device.function PCI identifier (&amp; NULL terminator)
    unsigned int domain;             //!< The PCI domain on which the device's bus resides, 0 to 0xffff
    unsigned int bus;                //!< The bus on which the device resides, 0 to 0xff
    unsigned int device;             //!< The device's id on the bus, 0 to 31
    unsigned int pciDeviceId;        //!< The combined 16-bit device id and 16-bit vendor id

    // Added in LWML 2.285 API
    unsigned int pciSubSystemId;     //!< The 32-bit Sub System Device ID

    // LWPU reserved for internal use only
    unsigned int reserved0;
    unsigned int reserved1;
    unsigned int reserved2;
    unsigned int reserved3;
} lwmlPciInfo_t;
/* End of lwml.h */

ncclResult_t wrapLwmlSymbols(void);

ncclResult_t wrapLwmlInit(void);
ncclResult_t wrapLwmlShutdown(void);
ncclResult_t wrapLwmlDeviceGetHandleByPciBusId(const char* pciBusId, lwmlDevice_t* device);
ncclResult_t wrapLwmlDeviceGetIndex(lwmlDevice_t device, unsigned* index);
ncclResult_t wrapLwmlDeviceGetHandleByIndex(unsigned int index, lwmlDevice_t *device);
ncclResult_t wrapLwmlDeviceGetPciInfo(lwmlDevice_t device, lwmlPciInfo_t* pci);
ncclResult_t wrapLwmlDeviceGetLwLinkState(lwmlDevice_t device, unsigned int link, lwmlEnableState_t *isActive);
ncclResult_t wrapLwmlDeviceGetLwLinkRemotePciInfo(lwmlDevice_t device, unsigned int link, lwmlPciInfo_t *pci);
ncclResult_t wrapLwmlDeviceGetLwLinkCapability(lwmlDevice_t device, unsigned int link,
                                                   lwmlLwLinkCapability_t capability, unsigned int *capResult);
ncclResult_t wrapLwmlDeviceGetMinorNumber(lwmlDevice_t device, unsigned int* minorNumber);

#endif // LWML_DIRECT

#endif // End include guard
