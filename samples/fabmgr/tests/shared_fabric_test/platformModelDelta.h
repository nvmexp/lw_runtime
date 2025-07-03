#pragma once

#define MAX_PARTITION_ACTIVATION_COUNT 50
#define MAX_DELTA_PARTITIONS 35

#define PARTITION_LIST_DUMP_FILE_NAME "partition_list.json"

class platformModelDelta
{
public:
    static void doPartitionList(fmHandle_t pFmHandle);
    static void doPartitionActivationStreeTest(fmHandle_t pFmHandle);
    static void doPartitionActivation(fmHandle_t pFmHandle, int partitionId);
    static void doPartitionDeactivation(fmHandle_t pFmHandle, int partitionId);
    static void setActivatedPartitionList(fmHandle_t pFmHandle, SharedFabricCmdParser_t *pCmdParser);

private:
    static void resetPartitionGpus(int partitionId);
    static void unBindPartitionGpus(int partitionId);
    static void resetAndBindPartitionGpus(int partitionId);
    static void unbindAllPartitionGpus(void);
    static void doSharedSwitchPartitionActivation(fmHandle_t pFmHandle, int partitionId);
    static void doVgpuPartitionActivation(fmHandle_t pFmHandle, int partitionId);
};
