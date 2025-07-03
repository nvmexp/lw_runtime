/*
 * Copyright (c) 2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCI_TEST_GPU_PLATFORM_SPECIFIC_H
#define INCLUDED_LWSCI_TEST_GPU_PLATFORM_SPECIFIC_H

#include <cstdint>
#include <vector>
#include "lwRmShim/lwRmShim.h"
#include "lwscibuf.h"

typedef struct {
    LwU32 probed;
    LwU32 attached;
    LwU32 deviceId;
    LwU32 deviceInstance;
    LwU32 subdeviceInstance;
} DeviceInfo;

typedef struct GpuTestResourceRec
{
    // resman handles
    uint32_t hClient;
    uint32_t hDevice;
    uint32_t m_gpuId; // index of which gpu the hDevice corresponds to
    uint32_t hSubdevice;
    uint32_t hVASpace;
    uint32_t hGpuInstance;

    // all info for gpus available
    std::vector<DeviceInfo> deviceInfo;
    std::vector<LwSciRmGpuId> uuids;
    uint32_t m_gpuSize;

    // GPFIFO
    uint32_t gpfifoPhyMem_hMemory;
    void* gpfifo_phyAddress;
    uint64_t gpfifoPhyMem_length;
    uint32_t gpfifoVirtMem_hMemory;
    LwU64 gpfifo_virtAddress;

    // Channel info
    uint32_t hChannel;
    void* m_pChannelCtrl;
    uint32_t gpFifoClass;

    // USERD
    uint32_t hPhyMemoryUserd;
    void* phyAddrUserd;

    // push buffer
    uint32_t pusherPhyMem_hMemory;
    void* pusher_phyAddress;
    uint64_t pusherPhyMem_length;
    uint32_t pusherVirtMem_hMemory;
    LwU64 pusher_virtAddress;

    // source memory
    uint32_t srcPhyMem_hMemory;
    void* src_phyAddress;
    uint32_t srcVirtMem_hMemory;
    LwU64 src_virtAddress;

    // destination memory
    uint32_t destPhyMem_hMemory;
    void* dest_phyAddress;
    uint32_t destVirtMem_hMemory;
    LwU64 dest_virtAddress;

    // doorbell memory
    uint32_t hDoorBell_hMemory;
    LwU32* pDoorbell_phyAddress;

    // raw buffer size
    uint32_t memSize;

    // resman shim
    void* lib_h;
    LwRmShimSessionContext session;
} GpuTestResource;

#endif
