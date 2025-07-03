/*
 *  Copyright 2018-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

 #pragma once
 
#include <queue>
#include <unistd.h>
#include "FMCommonTypes.h"
#include "FMErrorCodesInternal.h"
#include "GlobalFabricManager.h"

class GlobalFmApiHandler
{
public:
    GlobalFmApiHandler(GlobalFabricManager *pGfm);
    ~GlobalFmApiHandler();
 
    /*****************************************************************************/
    // these methods will be called as part of FM Lib command handling
    FMIntReturn_t getSupportedFabricPartitions(fmFabricPartitionList_t &fmFabricPartition);
    FMIntReturn_t activateFabricPartition(fmFabricPartitionId_t partitionId);
    FMIntReturn_t activateFabricPartitionWithVfs(fmFabricPartitionId_t partitionId, fmPciDevice_t *vfList, unsigned int numVfs);
    FMIntReturn_t deactivateFabricPartition(fmFabricPartitionId_t partitionId);
    FMIntReturn_t setActivatedFabricPartitions(fmActivatedFabricPartitionList_t &fmFabricPartitions);
    FMIntReturn_t getLwlinkFailedDevices(fmLwlinkFailedDevices_t &devList);
    FMIntReturn_t getUnsupportedFabricPartitions(fmUnsupportedFabricPartitionList_t &fmFabricPartition);

    /*****************************************************************************/

    // these methods will be called as part of FM Internal command handling
    FMIntReturn_t prepareGpuForReset(char *gpuUuid);
    FMIntReturn_t shutdownGpuLwlink(char *gpuUuid);
    FMIntReturn_t resetGpuLwlink(char *gpuUuid);
    FMIntReturn_t completeGpuReset(char *gpuUuid);
 
private:
    GlobalFabricManager *mpGfm;

    // lock to serialize external and internal lib API commands
    LWOSCriticalSection mLock;

    FMIntReturn_t reconfigSwitchPortAfterGpuReset(GpuKeyType gpuKey);

    FMIntReturn_t isGpuResetSupported(char *gpuUuid);

    void updateLWLinkConnsRepo(uint32_t nodeId, GlobalFMLWLinkConnRepo &srcLinkConnRepo,
                               GlobalFMLWLinkConnRepo &destLinkConnRepo);

    uint32_t getGpuNumEnabledLinks(FMGpuInfo_t &gpuInfo);
    FMIntReturn_t prepareGpuForResetHelper(char *gpuUuid);
    FMIntReturn_t completeGpuResetHelper(char *gpuUuid);
};


