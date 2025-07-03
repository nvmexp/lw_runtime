#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include "json.h"

#include "FMDeviceProperty.h"
#include "fabricConfig.h"
#include "vulcan.h"

//
// This is a Vulcan board topology with 8 GPUs and 4 LWSwitches
// LWLink connection details and LWSwitch GPIO physical IDs are in
// //syseng/Projects/DGX/Boards/P5612_Vulcan_GPU_Base_Board/Docs/P5612_Lwlink_Mapping_sheet_A00.xlsx
//

vulcan::vulcan( fabricTopologyEnum topo ) : fabricConfig( topo )
{

    uint64_t targetId;

    mNumSwitches = 4;
    mNumGpus = 8;

    for (uint32_t targetId = 0; targetId < mNumGpus; targetId++) {

        gpuTargetId[targetId] = targetId;

        gpuFabricAddrBase[targetId]  = FMDeviceProperty::getGpaFromTargetId(LWSWITCH_ARCH_TYPE_LS10, targetId);
        gpuFabricAddrRange[targetId] = FMDeviceProperty::getAddressRangePerGpu(LWSWITCH_ARCH_TYPE_LS10);

        gpuGpaEgmAddrBase[targetId]  = FMDeviceProperty::getGpaEgmFromTargetId(LWSWITCH_ARCH_TYPE_LS10, targetId);
        gpuGpaEgmAddrRange[targetId] = FMDeviceProperty::getAddressRangePerGpu(LWSWITCH_ARCH_TYPE_LS10);

        gpuFlaAddrBase[targetId]  = FMDeviceProperty::getFlaFromTargetId(LWSWITCH_ARCH_TYPE_LS10, targetId);
        gpuFlaAddrRange[targetId] = FMDeviceProperty::getAddressRangePerGpu(LWSWITCH_ARCH_TYPE_LS10);
    }

    mEnableTrunkLookback  = false;
    mEnableSpray   = false;
    mMaxTargetId = 0;

    mSharedPartitionJsonFile = NULL;
    mSharedPartitiolwersion = 0;
    mNumSharedPartitions = VULCAN_NUM_SHARED_LWSWITCH_FABRIC_PARTITIONS;

    time_t lwrtime = time(NULL);;
    struct tm *loc_time = localtime(&lwrtime);
    mSharedPartitionTimeStamp = asctime(loc_time);

    memset( mSwitchPorts, 0, sizeof(SwitchPort_t)*mNumSwitches*MAX_PORTS_PER_LWSWITCH );
};

vulcan::~vulcan()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void
vulcan::setConfig( bool enableTrunkLookback, bool enableSpray, const char *sharedPartitionJsonFile )
{
    uint32_t swIndex, portIndex;
    const uint32_t numSwitchesPerBaseboard = 4;

    mEnableTrunkLookback = enableTrunkLookback;
    mEnableSpray   = enableSpray;

    mNumSwitches = 4;
    mNumGpus = 8;

    if ( sharedPartitionJsonFile != NULL )
    {
        mSharedPartitionJsonFile = sharedPartitionJsonFile;
    }

    memset( mSwitchPorts, 0, sizeof(SwitchPort_t)*mNumSwitches*MAX_PORTS_PER_LWSWITCH );

    // Access ports
    // Switch 0
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[0][40]  = {0,      0,            0,      0,        true,         0,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][41]  = {0,      0,            0,      1,        true,         0,                 3,           false,       0,            0,                0};
    mSwitchPorts[0][44]  = {0,      0,            0,      2,        true,         0,                12,           false,       0,            0,                0};
    mSwitchPorts[0][45]  = {0,      0,            0,      3,        true,         0,                13,           false,       0,            0,                0};
    mSwitchPorts[0][42]  = {0,      0,            0,      0,        true,         1,                15,           false,       0,            0,                0};
    mSwitchPorts[0][43]  = {0,      0,            0,      1,        true,         1,                14,           false,       0,            0,                0};
    mSwitchPorts[0][46]  = {0,      0,            0,      2,        true,         1,                 8,           false,       0,            0,                0};
    mSwitchPorts[0][47]  = {0,      0,            0,      3,        true,         1,                 9,           false,       0,            0,                0};
    mSwitchPorts[0][53]  = {0,      0,            0,      0,        true,         2,                 6,           false,       0,            0,                0};
    mSwitchPorts[0][52]  = {0,      0,            0,      1,        true,         2,                 7,           false,       0,            0,                0};
    mSwitchPorts[0][49]  = {0,      0,            0,      2,        true,         2,                12,           false,       0,            0,                0};
    mSwitchPorts[0][48]  = {0,      0,            0,      3,        true,         2,                13,           false,       0,            0,                0};
    mSwitchPorts[0][32]  = {0,      0,            0,      0,        true,         3,                 9,           false,       0,            0,                0};
    mSwitchPorts[0][33]  = {0,      0,            0,      1,        true,         3,                 8,           false,       0,            0,                0};
    mSwitchPorts[0][36]  = {0,      0,            0,      2,        true,         3,                13,           false,       0,            0,                0};
    mSwitchPorts[0][37]  = {0,      0,            0,      3,        true,         3,                12,           false,       0,            0,                0};
    mSwitchPorts[0][63]  = {0,      0,            0,      0,        true,         4,                13,           false,       0,            0,                0};
    mSwitchPorts[0][62]  = {0,      0,            0,      1,        true,         4,                12,           false,       0,            0,                0};
    mSwitchPorts[0][59]  = {0,      0,            0,      2,        true,         4,                 6,           false,       0,            0,                0};
    mSwitchPorts[0][58]  = {0,      0,            0,      3,        true,         4,                 7,           false,       0,            0,                0};
    mSwitchPorts[0][34]  = {0,      0,            0,      0,        true,         5,                 6,           false,       0,            0,                0};
    mSwitchPorts[0][35]  = {0,      0,            0,      1,        true,         5,                 7,           false,       0,            0,                0};
    mSwitchPorts[0][38]  = {0,      0,            0,      2,        true,         5,                15,           false,       0,            0,                0};
    mSwitchPorts[0][39]  = {0,      0,            0,      3,        true,         5,                14,           false,       0,            0,                0};
    mSwitchPorts[0][55]  = {0,      0,            0,      0,        true,         6,                12,           false,       0,            0,                0};
    mSwitchPorts[0][54]  = {0,      0,            0,      1,        true,         6,                13,           false,       0,            0,                0};
    mSwitchPorts[0][51]  = {0,      0,            0,      2,        true,         6,                16,           false,       0,            0,                0};
    mSwitchPorts[0][50]  = {0,      0,            0,      3,        true,         6,                17,           false,       0,            0,                0};
    mSwitchPorts[0][61]  = {0,      0,            0,      0,        true,         7,                16,           false,       0,            0,                0};
    mSwitchPorts[0][60]  = {0,      0,            0,      1,        true,         7,                17,           false,       0,            0,                0};
    mSwitchPorts[0][57]  = {0,      0,            0,      2,        true,         7,                13,           false,       0,            0,                0};
    mSwitchPorts[0][56]  = {0,      0,            0,      3,        true,         7,                12,           false,       0,            0,                0};

    // Switch 1
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[1][36]  = {0,      0,            0,      0,        true,         0,                 0,           false,       0,            0,                0};
    mSwitchPorts[1][37]  = {0,      0,            0,      1,        true,         0,                 1,           false,       0,            0,                0};
    mSwitchPorts[1][40]  = {0,      0,            0,      2,        true,         0,                11,           false,       0,            0,                0};
    mSwitchPorts[1][46]  = {0,      0,            0,      3,        true,         0,                16,           false,       0,            0,                0};
    mSwitchPorts[1][47]  = {0,      0,            0,      4,        true,         0,                17,           false,       0,            0,                0};
    mSwitchPorts[1][32]  = {0,      0,            0,      0,        true,         1,                11,           false,       0,            0,                0};
    mSwitchPorts[1][2]   = {0,      0,            0,      1,        true,         1,                 2,           false,       0,            0,                0};
    mSwitchPorts[1][3]   = {0,      0,            0,      2,        true,         1,                 3,           false,       0,            0,                0};
    mSwitchPorts[1][4]   = {0,      0,            0,      3,        true,         1,                 7,           false,       0,            0,                0};
    mSwitchPorts[1][5]   = {0,      0,            0,      4,        true,         1,                 6,           false,       0,            0,                0};
    mSwitchPorts[1][33]  = {0,      0,            0,      0,        true,         2,                10,           false,       0,            0,                0};
    mSwitchPorts[1][38]  = {0,      0,            0,      1,        true,         2,                 3,           false,       0,            0,                0};
    mSwitchPorts[1][39]  = {0,      0,            0,      2,        true,         2,                 2,           false,       0,            0,                0};
    mSwitchPorts[1][0]   = {0,      0,            0,      3,        true,         2,                17,           false,       0,            0,                0};
    mSwitchPorts[1][1]   = {0,      0,            0,      4,        true,         2,                16,           false,       0,            0,                0};
    mSwitchPorts[1][53]  = {0,      0,            0,      0,        true,         3,                10,           false,       0,            0,                0};
    mSwitchPorts[1][63]  = {0,      0,            0,      1,        true,         3,                15,           false,       0,            0,                0};
    mSwitchPorts[1][62]  = {0,      0,            0,      2,        true,         3,                14,           false,       0,            0,                0};
    mSwitchPorts[1][51]  = {0,      0,            0,      3,        true,         3,                 3,           false,       0,            0,                0};
    mSwitchPorts[1][50]  = {0,      0,            0,      4,        true,         3,                 2,           false,       0,            0,                0};
    mSwitchPorts[1][57]  = {0,      0,            0,      0,        true,         4,                 0,           false,       0,            0,                0};
    mSwitchPorts[1][56]  = {0,      0,            0,      1,        true,         4,                 1,           false,       0,            0,                0};
    mSwitchPorts[1][52]  = {0,      0,            0,      2,        true,         4,                11,           false,       0,            0,                0};
    mSwitchPorts[1][49]  = {0,      0,            0,      3,        true,         4,                16,           false,       0,            0,                0};
    mSwitchPorts[1][48]  = {0,      0,            0,      4,        true,         4,                17,           false,       0,            0,                0};
    mSwitchPorts[1][34]  = {0,      0,            0,      0,        true,         5,                17,           false,       0,            0,                0};
    mSwitchPorts[1][35]  = {0,      0,            0,      1,        true,         5,                16,           false,       0,            0,                0};
    mSwitchPorts[1][42]  = {0,      0,            0,      2,        true,         5,                11,           false,       0,            0,                0};
    mSwitchPorts[1][6]   = {0,      0,            0,      3,        true,         5,                 8,           false,       0,            0,                0};
    mSwitchPorts[1][7]   = {0,      0,            0,      4,        true,         5,                 9,           false,       0,            0,                0};
    mSwitchPorts[1][43]  = {0,      0,            0,      0,        true,         6,                10,           false,       0,            0,                0};
    mSwitchPorts[1][59]  = {0,      0,            0,      1,        true,         6,                 5,           false,       0,            0,                0};
    mSwitchPorts[1][58]  = {0,      0,            0,      2,        true,         6,                 4,           false,       0,            0,                0};
    mSwitchPorts[1][55]  = {0,      0,            0,      3,        true,         6,                 1,           false,       0,            0,                0};
    mSwitchPorts[1][54]  = {0,      0,            0,      4,        true,         6,                 0,           false,       0,            0,                0};
    mSwitchPorts[1][41]  = {0,      0,            0,      0,        true,         7,                10,           false,       0,            0,                0};
    mSwitchPorts[1][44]  = {0,      0,            0,      1,        true,         7,                 5,           false,       0,            0,                0};
    mSwitchPorts[1][45]  = {0,      0,            0,      2,        true,         7,                 4,           false,       0,            0,                0};
    mSwitchPorts[1][61]  = {0,      0,            0,      3,        true,         7,                 1,           false,       0,            0,                0};
    mSwitchPorts[1][60]  = {0,      0,            0,      4,        true,         7,                 0,           false,       0,            0,                0};

    // Switch 2
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[2][45]  = {0,      0,            0,      0,        true,         0,                10,           false,       0,            0,                0};
    mSwitchPorts[2][42]  = {0,      0,            0,      1,        true,         0,                15,           false,       0,            0,                0};
    mSwitchPorts[2][43]  = {0,      0,            0,      2,        true,         0,                14,           false,       0,            0,                0};
    mSwitchPorts[2][63]  = {0,      0,            0,      3,        true,         0,                 7,           false,       0,            0,                0};
    mSwitchPorts[2][62]  = {0,      0,            0,      4,        true,         0,                 6,           false,       0,            0,                0};
    mSwitchPorts[2][40]  = {0,      0,            0,      0,        true,         1,                 5,           false,       0,            0,                0};
    mSwitchPorts[2][41]  = {0,      0,            0,      1,        true,         1,                 4,           false,       0,            0,                0};
    mSwitchPorts[2][46]  = {0,      0,            0,      2,        true,         1,                 0,           false,       0,            0,                0};
    mSwitchPorts[2][47]  = {0,      0,            0,      3,        true,         1,                 1,           false,       0,            0,                0};
    mSwitchPorts[2][34]  = {0,      0,            0,      4,        true,         1,                10,           false,       0,            0,                0};
    mSwitchPorts[2][52]  = {0,      0,            0,      0,        true,         2,                11,           false,       0,            0,                0};
    mSwitchPorts[2][51]  = {0,      0,            0,      1,        true,         2,                 9,           false,       0,            0,                0};
    mSwitchPorts[2][50]  = {0,      0,            0,      2,        true,         2,                 8,           false,       0,            0,                0};
    mSwitchPorts[2][17]  = {0,      0,            0,      3,        true,         2,                15,           false,       0,            0,                0};
    mSwitchPorts[2][16]  = {0,      0,            0,      4,        true,         2,                14,           false,       0,            0,                0};
    mSwitchPorts[2][38]  = {0,      0,            0,      0,        true,         3,                16,           false,       0,            0,                0};
    mSwitchPorts[2][39]  = {0,      0,            0,      1,        true,         3,                17,           false,       0,            0,                0};
    mSwitchPorts[2][2]   = {0,      0,            0,      2,        true,         3,                 7,           false,       0,            0,                0};
    mSwitchPorts[2][3]   = {0,      0,            0,      3,        true,         3,                 6,           false,       0,            0,                0};
    mSwitchPorts[2][35]  = {0,      0,            0,      4,        true,         3,                11,           false,       0,            0,                0};
    mSwitchPorts[2][36]  = {0,      0,            0,      0,        true,         4,                15,           false,       0,            0,                0};
    mSwitchPorts[2][37]  = {0,      0,            0,      1,        true,         4,                14,           false,       0,            0,                0};
    mSwitchPorts[2][61]  = {0,      0,            0,      2,        true,         4,                 3,           false,       0,            0,                0};
    mSwitchPorts[2][60]  = {0,      0,            0,      3,        true,         4,                 2,           false,       0,            0,                0};
    mSwitchPorts[2][44]  = {0,      0,            0,      4,        true,         4,                10,           false,       0,            0,                0};
    mSwitchPorts[2][32]  = {0,      0,            0,      0,        true,         5,                 1,           false,       0,            0,                0};
    mSwitchPorts[2][33]  = {0,      0,            0,      1,        true,         5,                 0,           false,       0,            0,                0};
    mSwitchPorts[2][0]   = {0,      0,            0,      2,        true,         5,                 4,           false,       0,            0,                0};
    mSwitchPorts[2][1]   = {0,      0,            0,      3,        true,         5,                 5,           false,       0,            0,                0};
    mSwitchPorts[2][19]  = {0,      0,            0,      4,        true,         5,                10,           false,       0,            0,                0};
    mSwitchPorts[2][57]  = {0,      0,            0,      0,        true,         6,                 9,           false,       0,            0,                0};
    mSwitchPorts[2][56]  = {0,      0,            0,      1,        true,         6,                 8,           false,       0,            0,                0};
    mSwitchPorts[2][49]  = {0,      0,            0,      2,        true,         6,                14,           false,       0,            0,                0};
    mSwitchPorts[2][48]  = {0,      0,            0,      3,        true,         6,                15,           false,       0,            0,                0};
    mSwitchPorts[2][53]  = {0,      0,            0,      4,        true,         6,                11,           false,       0,            0,                0};
    mSwitchPorts[2][59]  = {0,      0,            0,      0,        true,         7,                 6,           false,       0,            0,                0};
    mSwitchPorts[2][58]  = {0,      0,            0,      1,        true,         7,                 7,           false,       0,            0,                0};
    mSwitchPorts[2][55]  = {0,      0,            0,      2,        true,         7,                15,           false,       0,            0,                0};
    mSwitchPorts[2][54]  = {0,      0,            0,      3,        true,         7,                14,           false,       0,            0,                0};
    mSwitchPorts[2][18]  = {0,      0,            0,      4,        true,         7,                11,           false,       0,            0,                0};

    // Switch 3
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[3][63]  = {0,      0,            0,      0,        true,         0,                 8,           false,       0,            0,                0};
    mSwitchPorts[3][62]  = {0,      0,            0,      1,        true,         0,                 9,           false,       0,            0,                0};
    mSwitchPorts[3][59]  = {0,      0,            0,      2,        true,         0,                 5,           false,       0,            0,                0};
    mSwitchPorts[3][58]  = {0,      0,            0,      3,        true,         0,                 4,           false,       0,            0,                0};
    mSwitchPorts[3][34]  = {0,      0,            0,      0,        true,         1,                12,           false,       0,            0,                0};
    mSwitchPorts[3][35]  = {0,      0,            0,      1,        true,         1,                13,           false,       0,            0,                0};
    mSwitchPorts[3][38]  = {0,      0,            0,      2,        true,         1,                16,           false,       0,            0,                0};
    mSwitchPorts[3][39]  = {0,      0,            0,      3,        true,         1,                17,           false,       0,            0,                0};
    mSwitchPorts[3][61]  = {0,      0,            0,      0,        true,         2,                 0,           false,       0,            0,                0};
    mSwitchPorts[3][60]  = {0,      0,            0,      1,        true,         2,                 1,           false,       0,            0,                0};
    mSwitchPorts[3][57]  = {0,      0,            0,      2,        true,         2,                 4,           false,       0,            0,                0};
    mSwitchPorts[3][56]  = {0,      0,            0,      3,        true,         2,                 5,           false,       0,            0,                0};
    mSwitchPorts[3][42]  = {0,      0,            0,      0,        true,         3,                 5,           false,       0,            0,                0};
    mSwitchPorts[3][43]  = {0,      0,            0,      1,        true,         3,                 4,           false,       0,            0,                0};
    mSwitchPorts[3][46]  = {0,      0,            0,      2,        true,         3,                 1,           false,       0,            0,                0};
    mSwitchPorts[3][47]  = {0,      0,            0,      3,        true,         3,                 0,           false,       0,            0,                0};
    mSwitchPorts[3][53]  = {0,      0,            0,      0,        true,         4,                 8,           false,       0,            0,                0};
    mSwitchPorts[3][52]  = {0,      0,            0,      1,        true,         4,                 9,           false,       0,            0,                0};
    mSwitchPorts[3][49]  = {0,      0,            0,      2,        true,         4,                 5,           false,       0,            0,                0};
    mSwitchPorts[3][48]  = {0,      0,            0,      3,        true,         4,                 4,           false,       0,            0,                0};
    mSwitchPorts[3][32]  = {0,      0,            0,      0,        true,         5,                13,           false,       0,            0,                0};
    mSwitchPorts[3][33]  = {0,      0,            0,      1,        true,         5,                12,           false,       0,            0,                0};
    mSwitchPorts[3][36]  = {0,      0,            0,      2,        true,         5,                 3,           false,       0,            0,                0};
    mSwitchPorts[3][37]  = {0,      0,            0,      3,        true,         5,                 2,           false,       0,            0,                0};
    mSwitchPorts[3][40]  = {0,      0,            0,      0,        true,         6,                 7,           false,       0,            0,                0};
    mSwitchPorts[3][41]  = {0,      0,            0,      1,        true,         6,                 6,           false,       0,            0,                0};
    mSwitchPorts[3][44]  = {0,      0,            0,      2,        true,         6,                 3,           false,       0,            0,                0};
    mSwitchPorts[3][45]  = {0,      0,            0,      3,        true,         6,                 2,           false,       0,            0,                0};
    mSwitchPorts[3][55]  = {0,      0,            0,      0,        true,         7,                 9,           false,       0,            0,                0};
    mSwitchPorts[3][54]  = {0,      0,            0,      1,        true,         7,                 8,           false,       0,            0,                0};
    mSwitchPorts[3][51]  = {0,      0,            0,      2,        true,         7,                 3,           false,       0,            0,                0};
    mSwitchPorts[3][50]  = {0,      0,            0,      3,        true,         7,                 2,           false,       0,            0,                0};

    // Trunk ports
    if ( enableTrunkLookback )
    {
        // TBD
    }

    // Switch ports setting
    for ( swIndex = 0; swIndex < mNumSwitches; swIndex++ )
    {
        for ( portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
        {
            mSwitchPorts[swIndex][portIndex].nodeId = 0;

            //
            // Switch GPIO:
            // according to http://lwbugs/3324387/10
            // Switch GPIO physical ID is the same as 0 based swIndex
            //
            mSwitchPorts[swIndex][portIndex].swPhysicalId = swIndex;

            // port index
            mSwitchPorts[swIndex][portIndex].swPort = portIndex;

        } // portIndex loop
    } // swIndex loop

    printf("index nodeId swPhysicalId swPort connectToGpu peerGpuPhysicalId peerGpuPort connectToSw peerSwNodeId peerSwPhysicalId peerSwPort\n");
    for ( int j = 0; j < 2; j++ )
    {
        printf("Switch %d\n", j);

        for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
        {
            printf("%5d %6d %12d %6d %12d %17d %11d %11d %12d %16d %10d\n",
                    i, mSwitchPorts[j][i].nodeId, mSwitchPorts[j][i].swPhysicalId, mSwitchPorts[j][i].swPort, mSwitchPorts[j][i].connectToGpu,
                    mSwitchPorts[j][i].peerGpuPhysicalId, mSwitchPorts[j][i].peerGpuPort, mSwitchPorts[j][i].connectToSw, mSwitchPorts[j][i].peerSwNodeId,
                    mSwitchPorts[j][i].peerSwPhysicalId, mSwitchPorts[j][i].peerSwPort);
        }
    }
}

void vulcan::makeRemapTable( int nodeIndex, int swIndex )
{
    int32_t gpuEndpointId, portIndex, index, i;
    int64_t mappedAddr, range;
    GPU *gpu;
    uint32_t targetId;

    if ( (nodeIndex == 0) && (swIndex < (int)mNumSwitches) )
    {
        // on all access ports
        for ( portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
        {
            if ( mSwitchPorts[swIndex][portIndex].connectToGpu == false )
                continue;

            for ( targetId = 0; targetId < mNumGpus; targetId++ )
            {
                gpu   = gpus[nodeIndex][targetId];

                // GPA map slot
                index = FMDeviceProperty::getGpaRemapIndexFromTargetId(LWSWITCH_ARCH_TYPE_LS10, targetId);
                range = gpu->gparange();

                makeOneRemapEntry( nodeIndex,           // nodeIndex
                                   swIndex,             // swIndex
                                   portIndex,           // portIndex
                                   index,               // rmap table index
                                   1,                   // entry valid
                                   0,                   // 64 bits remap fabric address (remap to 0 physical)
                                   0,                   // context match
                                   0,                   // context mask
                                   0,                   // context replacement
                                   0,                   // address offset
                                   0,                   // address base
                                   0,                   // address limit
                                   targetId,            // target ID
                                   1,                   // remapFlags = remap GPA address, AddrType[1:0] 2’b10 Map Slot
                                   0,                   // irl select
                                   0,                   // FAM swizzle
                                   0,                   // FAM mult2
                                   0,                   // FAM plane select
                                   EXTENDED_RANGE_B);   // remap table select

                // FLA map slot
                index = FMDeviceProperty::getFlaRemapIndexFromTargetId(LWSWITCH_ARCH_TYPE_LS10, targetId);
                range = gpu->flarange();

                makeOneRemapEntry( nodeIndex,           // nodeIndex
                                   swIndex,             // swIndex
                                   portIndex,           // portIndex
                                   index,               // rmap table index
                                   1,                   // entry valid
                                   gpu->flabase(),      // 64 bits remap FLA to itsef, due to bug 2498189
                                   0,                   // context match
                                   0,                   // context mask
                                   0,                   // context replacement
                                   0,                   // address offset
                                   0,                   // address base
                                   0,                   // address limit
                                   targetId,            // target ID
                                   1,                   // remapFlags = remap FLA address, AddrType[1:0] 2’b10 Map Slot
                                   0,                   // irl select
                                   0,                   // FAM swizzle
                                   0,                   // FAM mult2
                                   0,                   // FAM plane select
                                   NORMAL_RANGE);       // remap table select
            }
        }
    }
}

void vulcan::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    return;
}

void vulcan::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    return;
}

int
vulcan::getNumTrunkPorts( uint32_t nodeIndex, uint32_t swIndex )
{
    int count = 0;
    for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i ++ )
    {
        if ( ( mSwitchPorts[swIndex][i].nodeId == nodeIndex ) &&
             ( mSwitchPorts[swIndex][i].connectToSw == true ) )
        {
            count++;
        }
    }
    return count;
}

int
vulcan::getSwitchPortIndex( uint32_t nodeIndex, uint32_t swIndex, uint32_t swPort )
{
    for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i ++ )
    {
        if ( ( mSwitchPorts[swIndex][i].nodeId == nodeIndex ) &&
             ( mSwitchPorts[swIndex][i].swPort == swPort ) )
        {
            return i;
        }
    }

    return -1;
}

int
vulcan::getSwitchPortIndexWithRlanId( uint32_t nodeIndex, uint32_t swIndex, uint32_t rlanId)
{
    for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i ++ )
    {
        if ( ( mSwitchPorts[swIndex][i].nodeId == nodeIndex ) &&
             ( mSwitchPorts[swIndex][i].connectToGpu == true ) &&
             ( mSwitchPorts[swIndex][i].peerGpuPort == rlanId ) )
        {
            return i;
        }
    }

    return -1;
}

bool
vulcan::isGpuConnectedToSwitch( uint32_t swNodeId, uint32_t swIndex, uint32_t gpuNodeId, uint32_t gpuPhysicalId )
{
    for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i ++ )
    {
        if ( ( mSwitchPorts[swIndex][i].nodeId == swNodeId ) &&
             ( mSwitchPorts[swIndex][i].connectToGpu == true ) &&
             ( mSwitchPorts[swIndex][i].nodeId == gpuNodeId ) &&
             ( mSwitchPorts[swIndex][i].peerGpuPhysicalId == gpuPhysicalId ) )
        {
            return true;
        }
    }

    return false;
}

void vulcan::getEgressPortsToGpu( uint32_t nodeIndex, uint32_t swIndex, uint32_t ingressPortNum,
                                       uint32_t destGpuNodeId, uint32_t destGpuPhysicalId,
                                       uint32_t *egressPortIndex, uint32_t *numEgressPorts, bool *isLastHop )
{
    int32_t i, portCount = 0;
    bool useTrunkPorts;

    *numEgressPorts = 0;
    *isLastHop = false;

    if ( isGpuConnectedToSwitch( nodeIndex, swIndex, destGpuNodeId, destGpuPhysicalId ) == true )
    {
        // destination GPU is on the same switch
        // could use trunk port to loopback
        useTrunkPorts = mEnableTrunkLookback;
    }
    else
    {
        // destination GPU is not on the same switch, use trunk ports
        useTrunkPorts = true;
    }

    int ingressPortIndex = getSwitchPortIndex(nodeIndex, swIndex, ingressPortNum);
    if ( ( ingressPortIndex < 0 ) || ( ingressPortIndex > MAX_PORTS_PER_LWSWITCH ) )
    {
        // cannot find the ingress port
        printf("%s: Cannot find the ingress port swNodeId %d, swPhysicalId %d, ingressPortNum %d.\n",
                __FUNCTION__, nodeIndex, swIndex, ingressPortNum);
        return;
    }

    SwitchPort_t ingressPort = mSwitchPorts[swIndex][ingressPortIndex];

    if ( ingressPort.connectToGpu )
    {
        // ingress port is an access port

        // destGpu is connected to the same ingress port
        if ( ( ingressPort.nodeId == destGpuNodeId ) &&
             ( ingressPort.peerGpuPhysicalId == destGpuPhysicalId ) )
        {
            // Do not go back the same ingress access port
            *numEgressPorts = 0;
            *isLastHop = true;
            return;
        }

        // destGpu is connected to access ports that are different from the ingress port
        for ( i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
        {
            if ( useTrunkPorts )
            {
                // useTrunkPorts is true, go to trunk ports
                if ( mSwitchPorts[swIndex][i].connectToSw == true )
                {
                    // TODO trunk loopback
                    egressPortIndex[portCount] = i;
                    portCount++;
                }
            }
            else
            {
                // useTrunkPorts is false, go to access ports connected to destGpu
                if ( ( mSwitchPorts[swIndex][i].connectToGpu ) &&
                     ( mSwitchPorts[swIndex][i].nodeId == destGpuNodeId ) &&
                     ( mSwitchPorts[swIndex][i].peerGpuPhysicalId == destGpuPhysicalId ) )
                {
                    egressPortIndex[portCount] = i;
                    portCount++;
                }
            }
        }
        *numEgressPorts = portCount > INGRESS_RID_MAX_PORTS ? INGRESS_RID_MAX_PORTS : portCount;
        *isLastHop = (useTrunkPorts == true) ? false : true;
        return;
    }

    // ingress port is a trunk port
    if ( ingressPort.connectToSw )
    {
        // go to access ports connected to destGpu
        for ( i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
        {
            if ( ( mSwitchPorts[swIndex][i].connectToGpu ) &&
                 ( mSwitchPorts[swIndex][i].nodeId == destGpuNodeId ) &&
                 ( mSwitchPorts[swIndex][i].peerGpuPhysicalId == destGpuPhysicalId ) )
            {
                egressPortIndex[portCount] = i;
                portCount++;
            }

        }
        *numEgressPorts = portCount > INGRESS_RID_MAX_PORTS ? INGRESS_RID_MAX_PORTS : portCount;
        *isLastHop = true;
        return;
    }
}

void vulcan::makeRIDandRlanRouteTable( int nodeIndex, int swIndex )
{
    uint32_t i, egressPortCount;
    uint32_t egressPortIndex[INGRESS_RID_MAX_PORTS];
    int egressPort[INGRESS_RID_MAX_PORTS], vcMap[INGRESS_RID_MAX_PORTS];
    int groupSelect[INGRESS_RLAN_MAX_GROUPS], groupSize[INGRESS_RLAN_MAX_GROUPS];
    bool isLastHop;

    if ( (nodeIndex == 0) && (swIndex < (int)mNumSwitches) )
    {
        for ( uint32_t gpuPhysicalId = 0; gpuPhysicalId < mNumGpus; gpuPhysicalId++ )
        {
            for ( uint32_t portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
            {
                int ingressPortIndex = getSwitchPortIndex(nodeIndex, swIndex, portIndex);
                if ( ( ingressPortIndex < 0 ) || ( ingressPortIndex > MAX_PORTS_PER_LWSWITCH ) )
                {
                    // cannot find the ingress port
                    printf("%s: Cannot find the ingress port swNodeId %d, swPhysicalId %d, ingressPortNum %d.\n",
                           __FUNCTION__, nodeIndex, swIndex, portIndex);
                    continue;
                }

                if ( ( mSwitchPorts[swIndex][ingressPortIndex].connectToGpu == false ) &&
                     ( mSwitchPorts[swIndex][ingressPortIndex].connectToSw == false ) )

                {
                    // the port is not connected to anything
                    continue;
                }

                memset(egressPort, 0, sizeof(int)*INGRESS_RID_MAX_PORTS);
                memset(vcMap, 0, sizeof(int)*INGRESS_RID_MAX_PORTS);
                memset(groupSelect, 0, sizeof(int)*INGRESS_RLAN_MAX_GROUPS);
                memset(groupSize, 0, sizeof(int)*INGRESS_RLAN_MAX_GROUPS);

                getEgressPortsToGpu( nodeIndex, swIndex, portIndex,
                                     nodeIndex, gpuPhysicalId,
                                     egressPortIndex, &egressPortCount, &isLastHop);

                if ( egressPortCount == 0 )
                {
                    printf("%s: there is no egress port to gpuPhysicalId %d from nodeIndex %d swIndex %d, portIndex %d.\n",
                           __FUNCTION__, gpuPhysicalId, nodeIndex, swIndex, portIndex);
                    continue;
                }
                else
                {
                    printf("%s: On port %d egress ports to gpuPhysicalId %d: ",
                            __FUNCTION__, portIndex, gpuPhysicalId);
                    for ( i = 0; i < egressPortCount; i++ )
                    {
                        printf("%d ", egressPortIndex[i]);
                    }
                    printf("\n");
                }

                if ( mEnableSpray == false )
                {
                    // no spray
                    uint32_t selectedIndex = 0xFFFF;

                    // ingress port is an access port
                    if ( mSwitchPorts[swIndex][ingressPortIndex].connectToGpu == true )
                    {
                        // egress port is an access port
                       if ( mSwitchPorts[swIndex][egressPortIndex[0]].connectToGpu == true )
                       {
                           // egress port is an access port
                           // To select one egress access port from a set of 2 access ports
                           // select the one with the same rlanId
                           // select only one port with the same rlanId from the egress port list
                           uint32_t j;
                           for ( j = 0; j < egressPortCount; j++ )
                           {
                               if ( mSwitchPorts[swIndex][egressPortIndex[j]].rlanId == mSwitchPorts[swIndex][ingressPortIndex].rlanId )
                               {
                                   selectedIndex = j;
                                   break;
                               }
                           }

                           if ( j >= egressPortCount )
                           {
                               printf("Failed to find an egress port with rlandId %d\n",
                                       mSwitchPorts[swIndex][ingressPortIndex].rlanId);
                               continue;
                           }
                       }
                       else
                       {
                           // egress port is a trunk port
                           // To select one egress trunk port from a set of 8 (Switch 0 and 3) or 10 (Switch 1 and 2) trunk ports
                           uint32_t srcGpuPhysicalId = mSwitchPorts[swIndex][ingressPortIndex].peerGpuPhysicalId;

                           // each GPU is connected to 2 access ports, rlanId 0 and rlanId 1
                           // use the ingress port to select an egress port
                           selectedIndex = (srcGpuPhysicalId*2 + mSwitchPorts[swIndex][ingressPortIndex].rlanId) % egressPortCount;

                           if ( selectedIndex >= egressPortCount )
                           {
                               printf("selectedIndex %d is larger than egressPortCount %d.\n", selectedIndex, egressPortCount);
                               continue;
                           }
                       }
                    }
                    else
                    {
                        // ingress port is a trunk port
                        if ( mSwitchPorts[swIndex][egressPortIndex[0]].connectToGpu == true )
                        {
                            // egress port is an access port
                            // To select one egress access port from a set of 2 access ports
                            // select the one with the opposite rlanId
                            uint32_t j;
                            for ( j = 0; j < egressPortCount; j++ )
                            {
                                if ( mSwitchPorts[swIndex][egressPortIndex[j]].rlanId != mSwitchPorts[swIndex][ingressPortIndex].rlanId )
                                {
                                    selectedIndex = j;
                                    break;
                                }
                            }

                            if ( j >= egressPortCount )
                            {
                                printf("Failed to find an egress port with rlandId %d\n",
                                        mSwitchPorts[swIndex][ingressPortIndex].rlanId);
                                continue;
                            }
                        }
                        else
                        {
                            // egress port is a trunk port
                            // TODO for multinode
                        }
                    }

                    if ( selectedIndex > INGRESS_RID_MAX_PORTS )
                    {
                        printf("Failed to select an egress port to GPU gpuPhysicalId %d.\n", gpuPhysicalId);
                        continue;
                    }

                    vcMap[0] = 0;
                    egressPort[0] = mSwitchPorts[swIndex][egressPortIndex[selectedIndex]].swPort;
                    printf("select egress port selectedIndex %d, number %d\n",
                            selectedIndex, egressPort[0]);

                    // RID entry
                    makeOneRIDRouteEntry( nodeIndex,        // nodeIndex
                                          swIndex,          // swIndex
                                          portIndex,        // portIndex
                                          gpuTargetId[gpuPhysicalId],    // RID Route table index
                                          1,                // entry valid
                                          0,                // rmod (no special routing)
                                          1,                // number of ports
                                          vcMap,            // pointer to array of VC controls
                                          egressPort);      // pointer to array of ports

                    // RLAN entry
                    makeOneRLANRouteEntry( nodeIndex,        // nodeIndex
                                           swIndex,          // swIndex
                                           portIndex,        // portIndex
                                           gpuTargetId[gpuPhysicalId],    // RID Route table index
                                           1,                // entry valid
                                           0,                // group count
                                           NULL,             // group select array
                                           NULL );           // group size array
                }
                else
                {
                    // spray
                    for ( uint32_t j = 0; j < egressPortCount; j++ )
                    {
                        groupSelect[mSwitchPorts[swIndex][egressPortIndex[j]].rlanId] = j;
                        groupSize[mSwitchPorts[swIndex][egressPortIndex[j]].rlanId] = 1;
                        egressPort[j] = mSwitchPorts[swIndex][egressPortIndex[j]].swPort;
                    }

                    int rmod = isLastHop ? 0x40 : 0;

                    // RID entry
                    makeOneRIDRouteEntry( nodeIndex,        // nodeIndex
                                          swIndex,          // swIndex
                                          portIndex,        // portIndex
                                          gpuTargetId[gpuPhysicalId], // RID Route table index
                                          1,                // entry valid
                                          rmod,             // rmod (no special routing)
                                          egressPortCount,  // spray to all egress ports
                                          vcMap,            // pointer to array of VC controls
                                          egressPort);      // pointer to array of ports

                    // RLAN entry
                    makeOneRLANRouteEntry( nodeIndex,        // nodeIndex
                                           swIndex,          // swIndex
                                           portIndex,        // portIndex
                                           gpuTargetId[gpuPhysicalId], // RLAN Route table index
                                           1,                // entry valid
                                           INGRESS_RLAN_MAX_GROUPS, // group count
                                           groupSelect,      // group select array
                                           groupSize );      // group size array

                }

            } // for portIndex
        } // gpuPhysicalId
    }
}

void vulcan::makeGangedLinkTable( int nodeIndex, int swIndex )
{
    return;
}

void vulcan::makeAccessPorts( int nodeIndex, int swIndex )
{
    uint32_t farPeerID, farPeerTargetID;

    if ( (nodeIndex == 0) && (swIndex < (int)mNumSwitches) )
    {
        for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
        {
            if ( mSwitchPorts[swIndex][i].connectToGpu == false )
                continue;

            makeOneAccessPort( nodeIndex,
                               swIndex,
                               mSwitchPorts[swIndex][i].swPort,   // portIndex
                               0,                                 // farNodeID
                               mSwitchPorts[swIndex][i].peerGpuPhysicalId, // farPeerID
                               mSwitchPorts[swIndex][i].peerGpuPort,       // farPortNum
                               DC_COUPLED,                                 // phyMode
                               mSwitchPorts[swIndex][i].peerGpuPhysicalId, // farPeerTargetID
                               mSwitchPorts[swIndex][i].rlanId,   // rlanID, set to connected GPU port number
                               mMaxTargetId);                     // maxTargetID
        }
    }
    else
    {
        printf("%s: Invalid LimeRock nodeIndex %d swIndex %d.\n", __FUNCTION__, nodeIndex, swIndex);
    }
}

void vulcan::makeTrunkPorts( int nodeIndex, int swIndex )
{
    if ( (nodeIndex == 0) && (swIndex < (int)mNumSwitches) )
    {
        for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
        {
            if ( mSwitchPorts[swIndex][i].connectToSw == false )
                continue;

            makeOneTrunkPort( nodeIndex,
                              swIndex,
                              mSwitchPorts[swIndex][i].swPort,  // portIndex
                              0,                                // farNodeID
                              mSwitchPorts[swIndex][i].peerSwPhysicalId, // farSwitchID
                              mSwitchPorts[swIndex][i].peerSwPort,       // farPortNum
                              DC_COUPLED,                       // phyMode
                              mMaxTargetId);                    // maxTargetID
        }
    }
    else
    {
        printf("%s: Invalid LimeRock nodeIndex %d swIndex %d.\n", __FUNCTION__, nodeIndex, swIndex);
    }
}

void vulcan::makeOneLwswitch( int nodeIndex, int swIndex )
{
    std::stringstream ecid;

    ecid << "N" << nodeIndex << "_S" << swIndex;

    if ( (nodeIndex == 0) && (swIndex < (int)mNumSwitches) )
    {
        switches[nodeIndex][swIndex]->set_version( FABRIC_MANAGER_VERSION );
        switches[nodeIndex][swIndex]->set_ecid( ecid.str().c_str() );

        // Configure access ports
        makeAccessPorts( nodeIndex, swIndex );

        // Configure trunk ports
        makeTrunkPorts( nodeIndex, swIndex );

        // Configure ingress remap table
        makeRemapTable( nodeIndex, swIndex );

        // Configure ingress RID and Rlan Route table
        makeRIDandRlanRouteTable( nodeIndex, swIndex );

        //
        // Switch GPIO:
        // according to http://lwbugs/3324387/10
        // Switch GPIO physical ID is the same as 0 based swIndex
        switches[nodeIndex][swIndex]->set_physicalid( swIndex );
        switches[nodeIndex][swIndex]->set_arch( LWSWITCH_ARCH_TYPE_LS10 );
    }
    else
    {
        printf("%s: Invalid LimeRock nodeIndex %d swIndex %d.\n", __FUNCTION__, nodeIndex, swIndex);
    }
}

bool vulcan::getSwitchIndexByPhysicalId( uint32_t physicalId, uint32_t &index  )
{
    for (int i = 0; i < MAX_NUM_LWSWITCH_PER_NODE; i++ )
    {
        if ( mSwitchPorts[i][0].swPhysicalId == physicalId )
        {
            index = i;
            return true;
        }
    }
    return false;
}

void
vulcan::updateSharedPartInfoLinkMasks( SharedPartInfoTable_t &partInfo )
{
    uint32_t s, g, swIndex;
    SharedPartSwitchInfoTable_t *switchInfo;
    SharedPartGpuInfoTable_t *gpuInfo;

    // Add switch trunk ports
    if ( partInfo.lwLinkIntraTrunkConnCount > 0 )
    {
        for ( s = 0; s < partInfo.switchCount; s++ )
        {
            switchInfo = &partInfo.switchInfo[s];
            if ( getSwitchIndexByPhysicalId( switchInfo->physicalId, swIndex ) == false )
            {
                printf("%s: failed to get switch index for physicalId 0x%x.\n",
                       __FUNCTION__, switchInfo->physicalId);
                continue;
            }

            for ( int p = 0; p < MAX_PORTS_PER_LWSWITCH; p++ )
            {
                if ( mSwitchPorts[swIndex][p].connectToSw == true )
                {
                    switchInfo->enabledLinkMask |= ( 1LL << p );
                }
            }
        }
    }

    // Add switch access ports and GPU ports
    for ( s = 0; s < partInfo.switchCount; s++ )
    {
        switchInfo = &partInfo.switchInfo[s];
        if ( getSwitchIndexByPhysicalId( switchInfo->physicalId, swIndex ) == false )
        {
            printf("%s: failed to get switch index for physicalId 0x%x.\n",
                   __FUNCTION__, switchInfo->physicalId);
            continue;
        }

        for ( g = 0; g < partInfo.gpuCount; g++ )
        {
            gpuInfo = &partInfo.gpuInfo[g];

            for ( int p = 0; p < MAX_PORTS_PER_LWSWITCH; p++ )
            {
                if ( ( mSwitchPorts[swIndex][p].connectToGpu == true ) &&
                     ( mSwitchPorts[swIndex][p].peerGpuPhysicalId == gpuInfo->physicalId ) )
                {
                    // add the switch access port
                    switchInfo->enabledLinkMask |= ( 1LL << p );

                    // add the gpu port
                    gpuInfo->enabledLinkMask |= ( 1LL << mSwitchPorts[swIndex][p].peerGpuPort );
                }
            }
        }
    }
}

void vulcan::fillSharedPartInfoTable(int nodeIndex)
{
    memset(mSharedVMPartInfo, 0, sizeof(SharedPartInfoTable_t)*VULCAN_NUM_SHARED_LWSWITCH_FABRIC_PARTITIONS);
    uint32_t partitionId = 0;

    // fill in the LWSwitch and GPUs, the enabledLinkMask will be computed and filled in later
    mSharedVMPartInfo[partitionId] =
            // partitionId 0 - 8 GPUs, 4 LWSwitch, 0 intra-trunk conn, 0 inter-trunk conn
            { partitionId, 8, 4, 0, 0,
              // all the GPUs, id, numlink, mask
              // all 18 GPU ports are used
              {   {0,    18, 0},
                  {1,    18, 0},
                  {2,    18, 0},
                  {3,    18, 0},
                  {4,    18, 0},
                  {5,    18, 0},
                  {6,    18, 0},
                  {7,    18, 0}  },

              // all the Switches, id, numlink, mask
              // access ports are used
              // switch 0 and 3 has 32 access ports
              // switch 1 and 2 has 40 access ports
              {   {0x0, 32, 0},
                  {0x1, 40, 0},
                  {0x2, 40, 0},
                  {0x3, 32, 0}
              }
            };

    partitionId++;
    mSharedVMPartInfo[partitionId] =
            // partitionId 1 - 4 GPUs, 4 LWSwitch, 0 intra-trunk conn, 0 inter-trunk conn
            { partitionId, 4, 4, 0, 0,
              // GPU id, numlink, mask
              // all GPU ports are used
              {   {0,    18, 0},
                  {1,    18, 0},
                  {2,    18, 0},
                  {3,    18, 0}  },

              // all the Switches, id, numlink, mask
              // access ports are used
              // switch 0 and 3 uses 16 access ports ( 4 GPU * 4 links )
              // switch 1 and 2 uses 20 access ports ( 4 GPU * 5 links )
              {   {0x0, 16, 0},
                  {0x1, 20, 0},
                  {0x2, 20, 0},
                  {0x3, 16, 0}
              }
            };

    partitionId++;
    mSharedVMPartInfo[partitionId] =
            // partitionId 2 - 4 GPUs, 4 LWSwitch, 0 intra-trunk conn, 0 inter-trunk conn
            { partitionId, 4, 4, 0, 0,
              // GPU id, numlink, mask
              // all GPU ports are used
              {   {4,    18, 0},
                  {5,    18, 0},
                  {6,    18, 0},
                  {7,    18, 0}  },

              // all the Switches, id, numlink, mask
              // access ports are used
              // switch 0 and 3 uses 16 access ports ( 4 GPU * 4 links )
              // switch 1 and 2 uses 20 access ports ( 4 GPU * 5 links )
              {   {0x0, 16, 0},
                  {0x1, 20, 0},
                  {0x2, 20, 0},
                  {0x3, 16, 0}
              }
            };

    partitionId++;
    mSharedVMPartInfo[partitionId] =
            // partitionId 3 - 2 GPUs, 4 LWSwitch, 0 intra-trunk conn, 0 inter-trunk conn
            { partitionId, 2, 4, 0, 0,
              // GPU id, numlink, mask
              // all GPU ports are used
              {   {0,    18, 0},
                  {1,    18, 0}  },

              // all the Switches, id, numlink, mask
              // access ports are used
              // switch 0 and 3 uses 8 access ports  ( 2 GPU * 4 links )
              // switch 1 and 2 uses 10 access ports ( 2 GPU * 5 links )
              {   {0x0,  8, 0},
                  {0x1, 10, 0},
                  {0x2, 10, 0},
                  {0x3,  8, 0}
              }
            };

    partitionId++;
    mSharedVMPartInfo[partitionId] =
            // partitionId 4 - 2 GPUs, 4 LWSwitch, 0 intra-trunk conn, 0 inter-trunk conn
            { partitionId, 2, 4, 0, 0,
              // GPU id, numlink, mask
              // all GPU ports are used
              {   {2,    18, 0},
                  {3,    18, 0}  },

              // all the Switches, id, numlink, mask
              // access ports are used
              // switch 0 and 3 uses 8 access ports  ( 2 GPU * 4 links )
              // switch 1 and 2 uses 10 access ports ( 2 GPU * 5 links )
              {   {0x0,  8, 0},
                  {0x1, 10, 0},
                  {0x2, 10, 0},
                  {0x3,  8, 0}
              }
            };

    partitionId++;
    mSharedVMPartInfo[partitionId] =
            // partitionId 5 - 2 GPUs, 4 LWSwitch, 0 intra-trunk conn, 0 inter-trunk conn
            { partitionId, 2, 4, 0, 0,
              // GPU id, numlink, mask
              // all GPU ports are used
              {   {4,    18, 0},
                  {5,    18, 0}  },

              // all the Switches, id, numlink, mask
              // access ports are used
              // switch 0 and 3 uses 8 access ports  ( 2 GPU * 4 links )
              // switch 1 and 2 uses 10 access ports ( 2 GPU * 5 links )
              {   {0x0,  8, 0},
                  {0x1, 10, 0},
                  {0x2, 10, 0},
                  {0x3,  8, 0}
              }
            };

    partitionId++;
    mSharedVMPartInfo[partitionId] =
            // partitionId 6 - 2 GPUs, 4 LWSwitch, 0 intra-trunk conn, 0 inter-trunk conn
            { partitionId, 2, 4, 0, 0,
              // GPU id, numlink, mask
              // all GPU ports are used
              {   {6,    18, 0},
                  {7,    18, 0}  },

              // all the Switches, id, numlink, mask
              // access ports are used
              // switch 0 and 3 uses 8 access ports  ( 2 GPU * 4 links )
              // switch 1 and 2 uses 10 access ports ( 2 GPU * 5 links )
              {   {0x0,  8, 0},
                  {0x1, 10, 0},
                  {0x2, 10, 0},
                  {0x3,  8, 0}
              }
            };

    // 8 one GPU partitions,
    // 1 GPUs, 0 LWSwitches, 0 intra-trunk conn, 0 inter-trunk conn
    partitionId++;
    for ( uint32_t physicalId = 0; physicalId < 8;
          physicalId++, partitionId++)
    {
        mSharedVMPartInfo[partitionId].partitionId = partitionId;
        mSharedVMPartInfo[partitionId].gpuCount = 1;
        mSharedVMPartInfo[partitionId].switchCount = 0;
        mSharedVMPartInfo[partitionId].lwLinkIntraTrunkConnCount = 0;
        mSharedVMPartInfo[partitionId].lwLinkInterTrunkConnCount = 0;
        mSharedVMPartInfo[partitionId].gpuInfo[0] = {physicalId, 0 , 0};
    }

    // update all switch and GPU link masks
    for ( partitionId = 0; partitionId < VULCAN_NUM_SHARED_LWSWITCH_FABRIC_PARTITIONS; partitionId++ )
    {
        updateSharedPartInfoLinkMasks( mSharedVMPartInfo[partitionId] );
    }
}

void vulcan::fillSystemPartitionInfo(int nodeIndex)
{
    node *nodeInfo = nodes[nodeIndex];

    nodeSystemPartitionInfo *systemPartInfo = new nodeSystemPartitionInfo();

    // fill all the bare metal partition information

    // Vulcan bare metal partition information
    bareMetalPartitionInfo *bareMetalPartInfo1 = systemPartInfo->add_baremetalinfo();
    partitionMetaDataInfo *bareMetaData1 = new partitionMetaDataInfo();
    bareMetaData1->set_gpucount( nodeInfo->gpu_size() );
    bareMetaData1->set_switchcount( nodeInfo->lwswitch_size() );
    // no interanode trunk connections 0
    bareMetaData1->set_lwlinkintratrunkconncount( 0 );
    // no internode trunk connection
    bareMetaData1->set_lwlinkintertrunkconncount( 0 );
    bareMetalPartInfo1->set_allocated_metadata( bareMetaData1 );

    // fill all the Pass-through virtualization partition information
    //
    //  GPUs     Switches    Number of trunk connections
    //    8         4        0
    //
    ptVMPartitionInfo *ptPartition1 = systemPartInfo->add_ptvirtualinfo();
    partitionMetaDataInfo *ptMetadata1 = new partitionMetaDataInfo();
    ptMetadata1->set_gpucount( 8 );
    ptMetadata1->set_switchcount( 4 );
    ptMetadata1->set_lwlinkintratrunkconncount( 0 );
    ptMetadata1->set_lwlinkintertrunkconncount( 0 );
    ptPartition1->set_allocated_metadata( ptMetadata1 );

    // fill all the GPU Pass-through only (Shared LWSwitch) virtualization partition information
    if ( mSharedPartitionJsonFile )
    {
        // the lwstmer provided partition definitions
        parsePartitionJsonFile( 0 );

        // update all switch and GPU link masks
        for ( uint i = 0; i < mNumSharedPartitions; i++ )
        {
            updateSharedPartInfoLinkMasks( mSharedVMPartInfo[i] );
        }
    }
    else
    {
        // Lwpu provided partition definitions
        fillSharedPartInfoTable( nodeIndex );
    }

    for ( uint32 idx = 0; idx < mNumSharedPartitions; idx++ )
    {
        SharedPartInfoTable_t partEntry = mSharedVMPartInfo[idx];
        sharedLWSwitchPartitionInfo *sharedPartition = systemPartInfo->add_sharedlwswitchinfo();
        partitionMetaDataInfo *sharedMetaData = new partitionMetaDataInfo();
        sharedMetaData->set_gpucount( partEntry.gpuCount );
        sharedMetaData->set_switchcount( partEntry.switchCount );
        sharedMetaData->set_lwlinkintratrunkconncount( partEntry.lwLinkIntraTrunkConnCount );
        sharedMetaData->set_lwlinkintertrunkconncount( partEntry.lwLinkInterTrunkConnCount );

        sharedPartition->set_partitionid( partEntry.partitionId );
        sharedPartition->set_allocated_metadata( sharedMetaData );
        // populate all the GPU information
        for ( uint32 gpuIdx = 0; gpuIdx < partEntry.gpuCount; gpuIdx++ )
        {
            SharedPartGpuInfoTable_t gpuEntry = partEntry.gpuInfo[gpuIdx];
            sharedLWSwitchPartitionGpuInfo *partGpu = sharedPartition->add_gpuinfo();
            partGpu->set_physicalid( gpuEntry.physicalId );
            partGpu->set_numenabledlinks( gpuEntry.numEnabledLinks );
            partGpu->set_enabledlinkmask ( gpuEntry.enabledLinkMask );
        }

        // populate all the Switch information
        for ( uint32 switchIdx = 0; switchIdx < partEntry.switchCount; switchIdx++ )
        {
            SharedPartSwitchInfoTable_t switchEntry = partEntry.switchInfo[switchIdx];
            sharedLWSwitchPartitionSwitchInfo *partSwitch = sharedPartition->add_switchinfo();
            partSwitch->set_physicalid( switchEntry.physicalId );
            partSwitch->set_numenabledlinks( switchEntry.numEnabledLinks );
            partSwitch->set_enabledlinkmask ( switchEntry.enabledLinkMask );
        }
    }

    nodes[nodeIndex]->set_allocated_partitioninfo( systemPartInfo );

    if ( mSharedPartitionJsonFile )
    {
        mSystemPartInfo = systemPartInfo;
        systemPartInfo->set_name(mSharedPartitionJsonFile);
    }
    else
    {
        // TODO replace with official name
        systemPartInfo->set_name("VULCAN");
    }
    systemPartInfo->set_version(FABRIC_PARTITION_VERSION);
    systemPartInfo->set_time(mSharedPartitionTimeStamp.c_str());
}

void vulcan::makeOneNode( int nodeIndex, int gpuNum, int lrNum )
{
    int i, j;

    // Add GPUs
    for ( i = 0; i < gpuNum; i++ )
    {
        gpus[nodeIndex][i] = nodes[nodeIndex]->add_gpu();
        makeOneGpu( nodeIndex, i, ((nodeIndex * 8) + i), 0, 0x3F, 0xFFFFFFFF,
                    gpuFabricAddrBase[i], gpuFabricAddrRange[i],
                    gpuFlaAddrBase[i], gpuFlaAddrRange[i],
                    i, gpuTargetId[i]);

    }

    // Add LimeRocks
    for ( i = 0; i < lrNum; i++ )
    {
        switches[nodeIndex][i] = nodes[nodeIndex]->add_lwswitch();
        switches[nodeIndex][i]->set_version( FABRIC_MANAGER_VERSION );
        makeOneLwswitch( nodeIndex, i);
    }

    fillSystemPartitionInfo( nodeIndex );
}

void vulcan::makeNodes()
{
    int i, nodeIndex = 0;
    std::stringstream nodeip;

    mNumSwitches = 4;
    mNumGpus = 8;

    for (nodeIndex = 0; nodeIndex < 1; nodeIndex++)
    {
        nodes[nodeIndex] = mFabric->add_fabricnode();
        nodes[nodeIndex]->set_version( FABRIC_MANAGER_VERSION );

        switch ( nodeIndex ) {
        case 0:
            // node 0 has 8 GPUs, 4 LimeRock
            makeOneNode( nodeIndex, mNumGpus, mNumSwitches);
            break;

        default:
            printf("%s: Invalid nodeIndex %d.\n", __FUNCTION__, nodeIndex);
            break;
        }
    }
}

// implement willow virtual functions
void vulcan::makeOneWillow( int nodeIndex, int willowIndex )
{
    return;
}

void vulcan::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    return;
}

void vulcan::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    return;
}
