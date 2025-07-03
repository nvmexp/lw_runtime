
/*****************************************************************************
 * Module Internal APIs - Fabric Manager
 *****************************************************************************/

/*****************************************************************************/
// entry point of fabric partition related internal APIs

DCGM_INT_ENTRY_POINT(dcgmGetSupportedFabricPartitions, tsapiGetSupportedFabricPartitions,
                 (dcgmHandle_t pDcgmHandle, dcgmFabricPartitionList_t *pDcgmFabricPartition),
                 "(%p %p)",
                 pDcgmHandle, pDcgmFabricPartition);

DCGM_INT_ENTRY_POINT(dcgmActivateFabricPartition, tsapiActivateFabricPartition,
                 (dcgmHandle_t pDcgmHandle, unsigned int partitionId),
                 "(%p %d)",
                 pDcgmHandle, partitionId);

DCGM_INT_ENTRY_POINT(dcgmDeactivateFabricPartition, tsapiDeactivateFabricPartition,
                 (dcgmHandle_t pDcgmHandle, unsigned int partitionId),
                 "(%p %d)",
                 pDcgmHandle, partitionId);

DCGM_INT_ENTRY_POINT(dcgmSetActivatedFabricPartitions, tsapiSetActivatedFabricPartitions,
                 (dcgmHandle_t pDcgmHandle, dcgmActivatedFabricPartitionList_t *pDcgmActivatedPartitionList),
                 "(%p %p)",
                 pDcgmHandle, pDcgmActivatedPartitionList);
/*****************************************************************************/
