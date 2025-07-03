/*************************************************************************
 * Copyright (c) 2016-2019, LWPU CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_LWLINK_H_
#define NCCL_LWLINK_H_

#include <sys/stat.h>
#include <fcntl.h>
#include "lwmlwrap.h"
#include "topo.h"

#define CONNECT_LWLINK 0x10
#define CONNECT_LWSWITCH 0x100

enum ncclLwLinkDeviceType {
  ncclLwLinkDeviceGpu,
  ncclLwLinkDeviceSwitch,
  ncclLwLinkDeviceBridge, // IBM/Power LWLink bridge (Device 04ea)
};

static ncclResult_t ncclDeviceType(const char* busId, enum ncclLwLinkDeviceType* type) {
  char classPath[] =  "/sys/bus/pci/devices/0000:00:00.0/class";
  memcpy(classPath+sizeof("/sys/bus/pci/devices/")-1, busId, sizeof("0000:00:00.0")-1);
  char* rPath = realpath(classPath, NULL);
  int fd;
  if ((fd = open(rPath, O_RDONLY)) == -1) {
    // Could not find device. It might be because we're in a VM and
    // we don't see the whole machine. This is handled silently so
    // we don't want to print an INFO error.
    TRACE(NCCL_INIT, "Open of %s failed : %s\n", rPath, strerror(errno));
    return ncclSystemError;
  }
  free(rPath);
  char pciClass[9];
  strncpy(pciClass, "0x000000", 9);
  int len;
  SYSCHECKVAL(read(fd, pciClass, 8), "read", len);
  SYSCHECK(close(fd), "close");
  if (strcmp(pciClass, "0x068000") == 0) {
    // PCI device is of type "Bridge / Other Bridge Device" (LWswitch)
    *type = ncclLwLinkDeviceSwitch;
  } else if (strcmp(pciClass, "0x068001") == 0) {
    // PCI device is of type "Bridge: IBM Device 04ea"
    *type = ncclLwLinkDeviceBridge;
  } else if (strcmp(pciClass, "0x030200") == 0 // "3D Controller" (Tesla)
      || strcmp(pciClass, "0x030000") == 0) {  // "VGA Controller" (VdChip)
    *type = ncclLwLinkDeviceGpu;
  } else {
    // Ignore if we don't know what's on the other side.
    return ncclSystemError;
  }
  return ncclSuccess;
}

/* Get the maximum number of LWLinks based on the GPU generation */
static ncclResult_t getMaxLwlinks(int* maxLinks) {
  int lwdaDev;
  LWDACHECK(lwdaGetDevice(&lwdaDev));
  int ccMajor;
  LWDACHECK(lwdaDeviceGetAttribute(&ccMajor, lwdaDevAttrComputeCapabilityMajor, lwdaDev));
  // 6 for Volta, 4 for Pascal
  *maxLinks = (ccMajor > 6) ? 6 : 4;
  // INFO("Device %d detected %d LWLinks", lwdaDev, *maxLinks);
  return ncclSuccess;
}

static int getLwlinkGpu(const char* busId1, const char* busId2) {
  // Determine if that connection is through LWLink
  int links = 0;
  int lwswitch_links = 0;
  int maxLwLinks = ncclLwdaCompCap() > 6 ? 6 : 4;
  lwmlDevice_t lwmlDev;
  ncclResult_t res = wrapLwmlDeviceGetHandleByPciBusId(busId1, &lwmlDev);
  if (res != ncclSuccess) return 0;

  for(int l=0; l<maxLwLinks; ++l) {
    // Check whether we can use this LWLink for P2P
    unsigned canP2P;
    if ((wrapLwmlDeviceGetLwLinkCapability(lwmlDev, l, LWML_LWLINK_CAP_P2P_SUPPORTED, &canP2P) != ncclSuccess) || !canP2P) continue;

    // Make sure the Lwlink is up. The previous call should have trained the link.
    lwmlEnableState_t isActive;
    if ((wrapLwmlDeviceGetLwLinkState(lwmlDev, l, &isActive) != ncclSuccess) || (isActive != LWML_FEATURE_ENABLED)) continue;

    // Try to figure out what's on the other side of the LWLink
    lwmlPciInfo_t remoteProc;
    if (wrapLwmlDeviceGetLwLinkRemotePciInfo(lwmlDev, l, &remoteProc) != ncclSuccess) continue;

    // Old versions of LWML return a lowercase PCI ID
    char* p = remoteProc.busId;
    for (int c=0; c<LWML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
      if (p[c] == 0) break;
      p[c] = toupper(p[c]);
    }

    if (busId2 != NULL && strncmp(busId2, remoteProc.busId, LWML_DEVICE_PCI_BUS_ID_BUFFER_SIZE) == 0) {
      links++;
    } else {
      // Make a lower case copy of the bus ID for calling ncclDeviceType
      // PCI system path is in lower case
      char* p = remoteProc.busId;
      char lowerId[LWML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      for (int c=0; c<LWML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
        if (p[c] == 0) break;
        lowerId[c] = tolower(p[c]);
      }

      // Determine if the remote side is LWswitch or a GPU
      enum ncclLwLinkDeviceType type;
      ncclResult_t ret = ncclDeviceType(lowerId, &type);
      if (ret == ncclSuccess) {
        if (type == ncclLwLinkDeviceSwitch) {
          //TODO: we are making an assumption that all GPUs are connected to this switch
          //This assumption may change for future architectures
          lwswitch_links++;
        } else if (type == ncclLwLinkDeviceGpu && busId2 == NULL) {
          links++;
        }
      } else {
        // The LWLink is up but we couldn't find the PCI device on the other
        // side. Assume it's an LWswitch outside a VM.
        if (l==0) INFO(NCCL_INIT, "Assuming LWLink is connected to LWswitch");
        lwswitch_links++;
      }
    }
  }
  return lwswitch_links ? CONNECT_LWSWITCH*lwswitch_links : CONNECT_LWLINK*links;
}

#endif
