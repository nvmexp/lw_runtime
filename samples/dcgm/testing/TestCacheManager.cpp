
#include "TestCacheManager.h"
#include "dcgm_fields.h"
#include "LwcmCacheManager.h"
#include <string>
#include <iostream>
#include <stdexcept>
#include <bitset>
#include "dcgm_structs.h"


/* No. of iterations corresponding to different sample set of vgpuIds */
#define NUM_VGPU_LISTS  5
#define TEST_MAX_NUM_VGPUS_PER_GPU  16
/*****************************************************************************/
TestCacheManager::TestCacheManager()
{

}

/*****************************************************************************/
TestCacheManager::~TestCacheManager()
{

}

/*************************************************************************/
std::string TestCacheManager::GetTag()
{
    return std::string("cachemanager");
}

/*****************************************************************************/
int TestCacheManager::Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus)
{
    m_gpus = gpus;
    return 0;
}

/*****************************************************************************/
int TestCacheManager::Cleanup()
{
    return 0;
}

template<typename T>
void deleteNotNull(T *&obj)
{
    if (NULL != obj) {
        delete obj;
        obj = NULL;
    }
}

static
DcgmCacheManager* createCacheManager(int pollInLockStep)
{
    int st;
    DcgmCacheManager *cacheManager = new DcgmCacheManager();

    if(!cacheManager)
    {
        fprintf(stderr, "Unable to allocate cacheManager\n");
        return cacheManager;
    }

    st = cacheManager->Init(pollInLockStep, 86400.0);
    if(st)
    {
        fprintf(stderr, "cacheManager->Init returned %d\n", st);
        deleteNotNull(cacheManager);
        return cacheManager;
    }

    st = cacheManager->Start();
    if(st)
    {
        fprintf(stderr, "cacheManager->Start() returned %d\n", st);
        deleteNotNull(cacheManager);
        return cacheManager;
    }

    return cacheManager;
}

int TestCacheManager::AddPowerUsageWatchAllGpusHelper(DcgmCacheManager *cacheManager)
{
    int st;
    test_lwcm_gpu_t *gpu;
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);

    for(int i = 0; i < (int)m_gpus.size(); i++)
    {
        gpu = &m_gpus[i];

        st = cacheManager->AddFieldWatch(DCGM_FE_GPU, gpu->gpuId, DCGM_FI_DEV_POWER_USAGE, 
                                         1000000, 86400.0, 0, watcher, false);
        if(st)
        {
            fprintf(stderr, "AddFieldWatch returned %d for lwml gpu %d\n",
                    st, (int)gpu->lwmlIndex);
            return -1;
        }
    }
    return 0;
}

/*****************************************************************************/
int TestCacheManager::TestRecording()
{
    int st = 0;
    int i, Msamples;
    test_lwcm_gpu_p gpu = 0;
    dcgmcm_sample_t sample;
    DcgmCacheManager *cacheManager = createCacheManager(1);
    if (NULL == cacheManager)
        return -1;

    /* Add a watch on our field for all GPUs */
    st = AddPowerUsageWatchAllGpusHelper(cacheManager);
    if (st != 0)
    {
        deleteNotNull(cacheManager);
        return st;
    }

    /* Now make sure all values are read */
    st = cacheManager->UpdateAllFields(1);
    if(st)
    {
        fprintf(stderr, "UpdateAllFields returned %d\n", st);
        deleteNotNull(cacheManager);
        return -1;
    }

    /* Verify all field values were saved */
    for(i = 0; i < (int)m_gpus.size(); i++)
    {
        gpu = &m_gpus[i];

        st = cacheManager->GetLatestSample(DCGM_FE_GPU, gpu->gpuId, DCGM_FI_DEV_POWER_USAGE, &sample, 0);
        if(st != DCGM_ST_OK)
        {
            fprintf(stderr, "Got st %d from GetLatestSample() for lwml gpu %u\n",
                    st, gpu->lwmlIndex);
            deleteNotNull(cacheManager);
            return 1;
            /* Non-fatal */
        }

        Msamples = 1; /* Only fetch one */
        st = cacheManager->GetSamples(DCGM_FE_GPU, gpu->gpuId, DCGM_FI_DEV_POWER_USAGE, &sample,
                                      &Msamples, 0, 0, DCGM_ORDER_ASCENDING);
        if(st != DCGM_ST_OK)
        {
            fprintf(stderr, "Got st %d from GetSamples() for lwml gpu %u\n",
                    st, gpu->lwmlIndex);
            deleteNotNull(cacheManager);
            return 1;
            /* Non-fatal */
        }
    }

    deleteNotNull(cacheManager);
    return 0;
}

int TestCacheManager::TestGpuFieldBytesUsed()
{
    int st = 0;
    int i, Msamples;
    test_lwcm_gpu_p gpu = 0;
    dcgmcm_sample_t sample;
    DcgmCacheManager *cacheManager = createCacheManager(1);
    if (NULL == cacheManager)
        return -1;

    /* Add a watch on a field for all GPUs */
    st = AddPowerUsageWatchAllGpusHelper(cacheManager);
    if (st != 0)
    {
        deleteNotNull(cacheManager);
        return st;
    }

    /* Now make sure all values are read */
    st = cacheManager->UpdateAllFields(1);
    if(st)
    {
        fprintf(stderr, "UpdateAllFields returned %d\n", st);
        deleteNotNull(cacheManager);
        return -1;
    }

    /* Now test the function we want to test */
    for(i = 0; i < (int)m_gpus.size(); i++)
    {
        long long bytesUsed = 0;
        gpu = &m_gpus[i];

        // try to retrieve bytes used a couple times since it sometimes takes a couple ms for the first
        // metadata gathering run to be complete
        for (int attempts = 3; attempts > 0; --attempts)
        {
            st = cacheManager->GetGpuFieldBytesUsed(gpu->gpuId, DCGM_FI_DEV_POWER_USAGE, &bytesUsed);
            if (st == DCGM_ST_OK)
                break;
            usleep(10000);   // 10 ms
        }

        if (st != DCGM_ST_OK
            || bytesUsed < 1024             // 1 KB
            || bytesUsed > 1024*1024*20)    // 20 MB
        {
            fprintf(stderr, "return value from GetGpuFieldBytesUsed was outside of acceptable range.  Value: %lld",
                    bytesUsed);
            deleteNotNull(cacheManager);
            return 1;
        }
    }

    deleteNotNull(cacheManager);
    return 0;
}

/*****************************************************************************/
int TestCacheManager::TestRecordingGlobal()
{
    int st, retSt = 0;
    int i, Msamples;
    DcgmCacheManager *cacheManager = new DcgmCacheManager();
    test_lwcm_gpu_p gpu = 0;
    dcgmcm_sample_t sample;
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);

    if(!cacheManager)
    {
        fprintf(stderr, "Unable to allocate cacheManager\n");
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->Init(1, 86400.0);
    if(st)
    {
        fprintf(stderr, "cacheManager->Init returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->Start();
    if(st)
    {
        fprintf(stderr, "cacheManager->Start() returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    
    st = cacheManager->AddFieldWatch(DCGM_FE_NONE, 0, DCGM_FI_DRIVER_VERSION, 1000000, 86400.0, 0, watcher, false);
    if(st)
    {
        fprintf(stderr, "AddGlobalFieldWatch returned %d \n", st);
        retSt = -1;
        goto CLEANUP;
    }

    /* Now make sure all values are read */
    st = cacheManager->UpdateAllFields(1);
    if(st)
    {
        fprintf(stderr, "UpdateAllFields returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->GetLatestSample(DCGM_FE_NONE, 0, DCGM_FI_DRIVER_VERSION, &sample, 0);
    if (st != DCGM_ST_OK) {
        fprintf(stderr, "Got st %d from GetLatestSample()\n", st);
        retSt = 1;
        /* Non-fatal */
    }
    free(sample.val.str);    // sample for this field is string type which has been strdup'ed

    Msamples = 1; /* Only fetch one */
    st = cacheManager->GetSamples(DCGM_FE_NONE, 0, DCGM_FI_DRIVER_VERSION, &sample,
            &Msamples, 0, 0, DCGM_ORDER_ASCENDING);
    if (st != DCGM_ST_OK) {
        fprintf(stderr, "Got st %d from GetSamples()\n", st);
        retSt = 1;
        /* Non-fatal */
    }
    free(sample.val.str);    // sample for this field is string type which has been strdup'd


    CLEANUP:
    deleteNotNull(cacheManager);

    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestEmpty()
{
    int st, retSt = 0;
    DcgmCacheManager *cacheManager = new DcgmCacheManager();
    dcgm_field_meta_p fieldMeta = 0;
    unsigned int gpuId = 0;
    dcgmcm_sample_t sample;

    memset(&sample, 0, sizeof(sample));

    if(!cacheManager)
    {
        fprintf(stderr, "Unable to allocate cacheManager\n");
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->Init(1, 86400.0);
    if(st)
    {
        fprintf(stderr, "cacheManager->Init returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->Start();
    if(st)
    {
        fprintf(stderr, "cacheManager->Start() returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_GPU_TEMP);
    if(!fieldMeta)
    {
        fprintf(stderr, "Unable to get fieldMeta for field DCGM_FI_DEV_GPU_TEMP\n");
        retSt = 1;
        goto CLEANUP;
    }

    /* Verify there are no samples yet */
    st = cacheManager->GetLatestSample(DCGM_FE_GPU, gpuId, fieldMeta->fieldId,
                                       &sample, 0);
    if(st != DCGM_ST_NOT_WATCHED)
    {
        fprintf(stderr, "GetLatestSample returned unexpected st %d for field %d. "
                "Expected not watched (-15)\n", st, fieldMeta->fieldId);
        retSt = 100;
        goto CLEANUP;
    }

    /* Inject a fake value */
    st = InjectSampleHelper(cacheManager, fieldMeta, DCGM_FE_GPU, gpuId, 0);
    if(st)
    {
        fprintf(stderr, "InjectSampleHelper returned unexpected st %d for field %d. "
                "Expected 0\n", st, fieldMeta->fieldId);
        retSt = 200;
        goto CLEANUP;
    }

    /* now the sample should be there */
    st = cacheManager->GetLatestSample(DCGM_FE_GPU, gpuId, fieldMeta->fieldId,
                                       &sample, 0);
    if(st != DCGM_ST_OK)
    {
        fprintf(stderr, "GetLatestSample returned unexpected st %d for field %d. "
                "Expected 0\n", st, fieldMeta->fieldId);
        retSt = 300;
        goto CLEANUP;
    }

    /* Clear the data structure */
    st = cacheManager->EmptyCache();
    if(st)
    {
        fprintf(stderr, "EmptyCache returned unexpected st %d for field %d. "
                        "Expected 0\n", st, fieldMeta->fieldId);
        retSt = 400;
        goto CLEANUP;
    }

    /* Verify there are no samples after clearing */
    st = cacheManager->GetLatestSample(DCGM_FE_GPU, gpuId, fieldMeta->fieldId,
                                       &sample, 0);
    if(st != DCGM_ST_NOT_WATCHED)
    {
        fprintf(stderr, "GetLatestSample returned unexpected st %d for field %d. "
                "Expected not watched (-15)\n", st, fieldMeta->fieldId);
        retSt = 500;
        goto CLEANUP;
    }


    CLEANUP:
    if(cacheManager)
        delete(cacheManager);
    cacheManager = 0;

    return retSt;
}

/*****************************************************************************/
/*
 * Helper function to generate a unique vgpuId list based on a gpuId and a number of vgpus
 *
 * This function must be deterministic
 *
 */
static
std::vector<lwmlVgpuInstance_t> gpuIdToVgpuList(unsigned int gpuId, int numVgpus)
{
    int i;
    std::vector<lwmlVgpuInstance_t>retList;

    /* When calling ManageVgpuList(), the first element contains the number of elements */
    retList.push_back((unsigned int)numVgpus);

    for(i = 0; i < numVgpus; i++)
    {
        retList.push_back((lwmlVgpuInstance_t)(gpuId * DCGM_MAX_VGPU_INSTANCES_PER_PGPU)+i);
    }
    return retList;
}

/*****************************************************************************/
int TestCacheManager::TestFieldValueColwersion()
{
    long long LwmlFieldValueToInt64(lwmlFieldValue_t *v);
    int       retSt = 0;

    lwmlFieldValue_t v;

    v.valueType = LWML_VALUE_TYPE_DOUBLE;
    v.value.dVal = 4.0;

    long long i64 = LwmlFieldValueToInt64(&v);

    if (i64 != 4)
    {
        fprintf(stderr, "Expected 4.0 to be colwerted as 4, but got %lld.\n", i64);
        retSt = 100;
    }

    memset(&v, 0, sizeof(v));
    v.valueType = LWML_VALUE_TYPE_UNSIGNED_LONG_LONG;
    v.value.ullVal = 5;
    i64 = LwmlFieldValueToInt64(&v);

    if (i64 != 5)
    {
        fprintf(stderr, "Expected 5 to be colwerted as 5, but got %lld.\n", i64);
        retSt = 100;
    }

    memset(&v, 0, sizeof(v));
    v.valueType = LWML_VALUE_TYPE_UNSIGNED_INT;
    v.value.uiVal = 10;
    i64 = LwmlFieldValueToInt64(&v);
    
    if (i64 != 10)
    {
        fprintf(stderr, "Expected 10 to be colwerted as 10, but got %lld.\n", i64);
        retSt = 100;
    }
   
    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestColwertVectorToBitmask()
{
    uint64_t bitmask;
    std::vector<unsigned int> gpuIds;
    DcgmCacheManager cacheManager;
    int retSt = 0;

    cacheManager.ColwertVectorToBitmask(gpuIds, bitmask, gpuIds.size());
    if (bitmask != 0)
    {
        fprintf(stderr, "An empty vector should produce an empty bitmask.\n");
        retSt = 100;
    }

    gpuIds.push_back(0);
    gpuIds.push_back(2);
    
    cacheManager.ColwertVectorToBitmask(gpuIds, bitmask, gpuIds.size());
    if (bitmask != (uint64_t)0x5)
    {
        fprintf(stderr, "Expected 0x05 but got %llx.\n", static_cast<long long>(bitmask));
        retSt = 100;
    }

    gpuIds.push_back(7);
    gpuIds.push_back(8);
    gpuIds.push_back(9);
    cacheManager.ColwertVectorToBitmask(gpuIds, bitmask, gpuIds.size());
    if (bitmask != (uint64_t)0x385)
    {
        fprintf(stderr, "Expected 0x385 but got %llx.\n", static_cast<long long>(bitmask));
        retSt = 100;
    }
    
    cacheManager.ColwertVectorToBitmask(gpuIds, bitmask, 2);
    if (bitmask != (uint64_t)0x05)
    {
        fprintf(stderr, "Expected 0x05 but got %llx.\n", static_cast<long long>(bitmask));
        retSt = 100;
    }

    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestAffinityBitmasksMatch()
{
    DcgmCacheManager cacheManager;
    dcgmAffinity_t   affinity = { 0 };
    int retSt = 0;

    affinity.affinityMasks[0].bitmask[0] = 0x1;
    affinity.affinityMasks[0].bitmask[1] = 0x10;
    affinity.affinityMasks[1].bitmask[0] = 0x1;
    if (cacheManager.AffinityBitmasksMatch(affinity, 0, 1) == true)
    {
        fprintf(stderr, "Should have failed since bitmask 1 is different for the two gpus.\n");
        retSt = 100;
    }

    affinity.affinityMasks[1].bitmask[1] = 0x10;
    if (cacheManager.AffinityBitmasksMatch(affinity, 0, 1) == false)
    {
        fprintf(stderr, "Shouldn't have failed since bitmask 1 is the same for the two gpus.\n");
        retSt = 100;
    }

    affinity.affinityMasks[2].bitmask[0] = 0x10;
    affinity.affinityMasks[3].bitmask[0] = 0x10;
    affinity.affinityMasks[4].bitmask[0] = 0x100;
    affinity.affinityMasks[5].bitmask[0] = 0x100;

    if ((cacheManager.AffinityBitmasksMatch(affinity, 0, 2) == true) ||
        (cacheManager.AffinityBitmasksMatch(affinity, 1, 2) == true) ||
        (cacheManager.AffinityBitmasksMatch(affinity, 1, 3) == true) ||
        (cacheManager.AffinityBitmasksMatch(affinity, 1, 4) == true) ||
        (cacheManager.AffinityBitmasksMatch(affinity, 2, 4) == true) ||
        (cacheManager.AffinityBitmasksMatch(affinity, 2, 5) == true))
    {
        fprintf(stderr, "Shouldn't have matched different bitmasks.\n");
        retSt = 100;
    }

    if ((cacheManager.AffinityBitmasksMatch(affinity, 2, 3) == false) ||
        (cacheManager.AffinityBitmasksMatch(affinity, 4, 5) == false))
    {
        fprintf(stderr, "Should have matched identical bitmasks.\n");
        retSt = 100;
    }

    for (int i = 0; i < DCGM_AFFINITY_BITMASK_ARRAY_SIZE; i++)
    {
        affinity.affinityMasks[2].bitmask[i] = 0x10;
        affinity.affinityMasks[3].bitmask[i] = 0x10;
        affinity.affinityMasks[4].bitmask[i] = 0x100;
        affinity.affinityMasks[5].bitmask[i] = 0x100;
    }
    if ((cacheManager.AffinityBitmasksMatch(affinity, 2, 3) == false) ||
        (cacheManager.AffinityBitmasksMatch(affinity, 4, 5) == false))
    {
        fprintf(stderr, "Should have matched identical bitmasks.\n");
        retSt = 100;
    }

    affinity.affinityMasks[2].bitmask[0] = 0;
    if (cacheManager.AffinityBitmasksMatch(affinity, 2, 3) == true)
    {
        fprintf(stderr, "Shouldn't have matched different bitmasks.\n");
        retSt = 100;
    }

    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestCreateGroupsFromCpuAffinities()
{
    DcgmCacheManager dcm;
    dcgmAffinity_t   affinity = { 0 };
    std::vector< std::vector<unsigned int> >affinityGroups;
    std::vector<unsigned int> gpuIds;
    int retSt = 0;

    affinity.numGpus = 8;

    for (int i = 0; i < 4; i++)
    {
        affinity.affinityMasks[i].bitmask[0] = 0x1;
        affinity.affinityMasks[i].dcgmGpuId = i;
        affinity.affinityMasks[i+4].bitmask[0] = 0x10;
        affinity.affinityMasks[i+4].dcgmGpuId = i + 4;
        gpuIds.push_back(i);
    }


    dcm.CreateGroupsFromCpuAffinities(affinity, affinityGroups, gpuIds);
    
    if (affinityGroups.size() != 1)
    {
        fprintf(stderr, "There should be 1 group because only some gpus were eligible, but found %d.\n",
                static_cast<int>(affinityGroups.size()));
        retSt = 100;
    }
    
    for (unsigned int i = 0; i < 4; i++)
        gpuIds.push_back(i+4);

    affinityGroups.clear();
    
    dcm.CreateGroupsFromCpuAffinities(affinity, affinityGroups, gpuIds);

    if (affinityGroups.size() != 2)
    {
        fprintf(stderr, "Expected two groups to be created but found %d.\n", static_cast<int>(affinityGroups.size()));
        retSt = 100;
    }
    else
    {
        if ((affinityGroups[0].size() != 4) ||
            (affinityGroups[1].size() != 4))
        {
            fprintf(stderr, "Expected both groups to have a size of 4.\n");
            retSt = 100;
        }
        else
        {
            for (unsigned int i = 0; i < 4; i++)
            {
                if (affinityGroups[0][i] != i)
                {
                    fprintf(stderr, "Expected group 1's gpu %u to be %u, but found %u.\n",
                            i, i, affinityGroups[0][i]);
                    retSt = 100;
                    break;
                }

                if (affinityGroups[1][i] != i + 4)
                {
                    fprintf(stderr, "Expected group 2's gpu %u to be %u, but found %u.\n",
                            i, i + 4, affinityGroups[0][i]);
                    retSt = 100;
                    break;
                }
            }
        }
    }

    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestPopulatePotentialCpuMatches()
{
    uint32_t numGpus = 2;
    std::vector<std::vector<unsigned int> > affinityGroups;
    std::vector<size_t> potentialCpuMatches;
    int retSt = 0;
    DcgmCacheManager dcm;

    for (unsigned int i = 0; i < 4; i++)
    {
        std::vector<unsigned int> group;

        for (unsigned int j = 0; j < i + 1; j++)
            group.push_back(j);

        affinityGroups.push_back(group);
    }

    dcm.PopulatePotentialCpuMatches(affinityGroups, potentialCpuMatches, numGpus);
    
    if (potentialCpuMatches.size() != 3)
    {
        fprintf(stderr, "Expected 3 potential matches but got %d.\n", static_cast<int>(potentialCpuMatches.size()));
        retSt = 100;
        return retSt;
    }
    
    std::vector<unsigned int> &group1 = affinityGroups[potentialCpuMatches[0]];
    std::vector<unsigned int> &group2 = affinityGroups[potentialCpuMatches[1]];
    std::vector<unsigned int> &group3 = affinityGroups[potentialCpuMatches[2]];

    if ((group1.size() != 2) ||
        (group2.size() != 3) ||
        (group3.size() != 4))
    {
        fprintf(stderr, "Potential group sizes don't match expectations.\n");
        retSt = 100;
        return retSt;
    }
    
    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestCombineAffinityGroups()
{
    std::vector<std::vector<unsigned int> > affinityGroups;
    std::vector<unsigned int> combinedGpuList;
    unsigned int gpuIndex = 0;
    int retSt = 0;
    DcgmCacheManager dcm;

    for (unsigned int i = 0; i < 5; i++)
    {
        std::vector<unsigned int> group;

        for (unsigned int j = 0; j < i + 1; j++)
        {
            group.push_back(gpuIndex);
            gpuIndex++;
        }

        affinityGroups.push_back(group);
    }

    /*
     * Looks like this:
     * group 0 : gpu  0
     * group 1 : gpus 1,  2
     * group 2 : gpus 3,  4,  5
     * group 3 : gpus 6,  7,  8,  9
     * group 4 : gpus 10, 11, 12, 13, 14
     */
    dcm.CombineAffinityGroups(affinityGroups, combinedGpuList, 8);
    for (unsigned int i = 0; i < 5; i++)
    {
        if (combinedGpuList[i] != i + 10)
        {
            fprintf(stderr, "Expected to find %u, but found %u.\n", i + 10, combinedGpuList[i]);
            return 100;
        }
    }

    for (unsigned int i = 0; i < 3; i++)
    {
        if (combinedGpuList[i + 5] != i + 3)
        {
            fprintf(stderr, "Expected to find %u, but found %u.\n", i + 6, combinedGpuList[i + 5]);
            return 100;
        }
    }

    combinedGpuList.clear();
    dcm.CombineAffinityGroups(affinityGroups, combinedGpuList, 12);
    for (unsigned int i = 0; i < 5; i++)
    {
        if (combinedGpuList[i] != i + 10)
        {
            fprintf(stderr, "Expected to find %u, but found %u.\n", i + 10, combinedGpuList[i]);
            return 100;
        }
    }

    for (unsigned int i = 0; i < 4; i++)
    {
        if (combinedGpuList[i + 5] != i + 6)
        {
            fprintf(stderr, "Expected to find %u, but found %u.\n", i + 6, combinedGpuList[i + 5]);
            return 100;
        }
    }
    
    for (unsigned int i = 0; i < 3; i++)
    {
        if (combinedGpuList[i + 9] != i + 3)
        {
            fprintf(stderr, "Expected to find %u, but found %u.\n", i + 3, combinedGpuList[i + 9]);
            return 100;
        }
    }

    return retSt;
}

void setup_topology(dcgmTopology_t &top)
{
    // Make up some topology data
    unsigned int numElements = 0;
    top.element[numElements].dcgmGpuA = 0;
    top.element[numElements].dcgmGpuB = 1;
    top.element[numElements].path = DCGM_TOPOLOGY_LWLINK1;
    numElements++;
    
    top.element[numElements].dcgmGpuA = 0;
    top.element[numElements].dcgmGpuB = 2;
    top.element[numElements].path = DCGM_TOPOLOGY_LWLINK2;
    numElements++;
    
    top.element[numElements].dcgmGpuA = 0;
    top.element[numElements].dcgmGpuB = 3;
    top.element[numElements].path = DCGM_TOPOLOGY_LWLINK2;
    numElements++;
    
    top.element[numElements].dcgmGpuA = 1;
    top.element[numElements].dcgmGpuB = 2;
    top.element[numElements].path = DCGM_TOPOLOGY_LWLINK2;
    numElements++;
    
    top.element[numElements].dcgmGpuA = 1;
    top.element[numElements].dcgmGpuB = 3;
    top.element[numElements].path = DCGM_TOPOLOGY_LWLINK2;
    numElements++;
    
    top.element[numElements].dcgmGpuA = 2;
    top.element[numElements].dcgmGpuB = 3;
    top.element[numElements].path = DCGM_TOPOLOGY_LWLINK1;
    numElements++;

    top.numElements = numElements;
}

bool containsConnectionPair(const std::vector<DcgmGpuConnectionPair> &level, const DcgmGpuConnectionPair &cp)
{
    bool found = false;
    for (size_t i = 0; i < level.size(); i++)
    {
        if (level[i] == cp)
        {
            found = true;
            break;
        }
    }
    
    return found;
}

/*****************************************************************************/
int TestCacheManager::TestSetIOConnectionLevels()
{
    int retSt = 0;
    dcgmTopology_t top = { 0 };
    std::map<unsigned int, std::vector<DcgmGpuConnectionPair> > connectionLevel;
    DcgmCacheManager dcm;

    std::vector<unsigned int> affinityGroup;
    for (unsigned int i = 0; i < 4; i++)
        affinityGroup.push_back(i);

    setup_topology(top);

    dcm.SetIOConnectionLevels(affinityGroup, &top, connectionLevel);

    if (connectionLevel.size() != 2)
    {
        fprintf(stderr, "Expected 2 populated connection levels, but found %d.\n",
                static_cast<int>(connectionLevel.size()));
        retSt = 100;
    }

    if (connectionLevel[1].size() != 2)
    {
        fprintf(stderr, "Level 0 should have 2 connections, but it has %d.\n",
                static_cast<int>(connectionLevel[0].size()));
        return 100;
    }

    if (connectionLevel[2].size() != 4)
    {
        fprintf(stderr, "Level 1 should have 4 connections, but it has %d.\n",
                static_cast<int>(connectionLevel[1].size()));
        return 100;
    }

    DcgmGpuConnectionPair level1[2] = { DcgmGpuConnectionPair(0,1), DcgmGpuConnectionPair(2,3) };
    DcgmGpuConnectionPair level2[4] = { DcgmGpuConnectionPair(0,2), DcgmGpuConnectionPair(0,3),
                                 DcgmGpuConnectionPair(1,2), DcgmGpuConnectionPair(1,3) };

    for (unsigned int i = 0; i < 2; i++)
    {
        if (containsConnectionPair(connectionLevel[1], level1[i]) == false)
        {
            fprintf(stderr, "Expected to find %u,%u in level 0, but didn't.\n", level1[i].gpu1, level1[i].gpu2);
            return 100;
        }
    }

    for (unsigned int i = 0; i < 4; i++)
    {
        if (containsConnectionPair(connectionLevel[2], level2[i]) == false)
        {
            fprintf(stderr, "Expected to find %u,%u in level 1, but didn't.\n", level2[i].gpu1, level2[i].gpu2);
            return 100;
        }
    }

    uint64_t outputGpus = 0;
    if (dcm.HasStrongConnection(connectionLevel[1], 4, outputGpus) == true)
    {
        fprintf(stderr, "Level 0 doesn't have a strong connection between 4 gpus, but reported to.\n");
        return 100;
    }
    if (outputGpus != 0)
    {
        fprintf(stderr, "Output gpus shouldn't have been set, but was.\n");
        return 100;
    }

    if (dcm.HasStrongConnection(connectionLevel[1], 2, outputGpus) == false)
    {
        fprintf(stderr, "Level 0 has a strong connection between 2 gpus, but said it didn't.\n");
        return 100;
    }

    if (outputGpus != 0x3)
    {
        fprintf(stderr, "Output gpus should've been set to 0x3, but is %llx.\n",
                static_cast<long long>(outputGpus));
        return 100;
    }

    outputGpus = 0;
    if (dcm.HasStrongConnection(connectionLevel[2], 4, outputGpus) == false)
    {
        fprintf(stderr, "Level 1 has a strong connection between 4 gpus, but said it didn't.\n");
        return 100;
    }

    if (outputGpus != 0xF)
    {
        fprintf(stderr, "Output gpus should've been set to 0xF, but is %llx.\n",
                static_cast<long long>(outputGpus));
        return 100;
    }

    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestMatchByIO()
{
    int retSt = 0;
    dcgmTopology_t top = { 0 };
    std::vector<std::vector<unsigned int> >affinityGroups;
    std::vector<size_t> potentialCpuMatches;
    uint64_t outputGpus;
    DcgmCacheManager dcm;

    std::vector<unsigned int> group;
    for (unsigned int i = 0; i < 4; i++)
        group.push_back(i);
    affinityGroups.push_back(group);

    setup_topology(top);

    potentialCpuMatches.push_back(0);

    dcm.MatchByIO(affinityGroups, &top, potentialCpuMatches, 2, outputGpus);

    if (outputGpus != 0x5)
    {
        fprintf(stderr, "Output gpus should've been set to 0x5, but is %llx.\n",
                static_cast<long long>(outputGpus));
        retSt = 100;
    }

    dcm.MatchByIO(affinityGroups, &top, potentialCpuMatches, 3, outputGpus);

    if (outputGpus != 0xd)
    {
        fprintf(stderr, "Output gpus should've been set to 0xd, but is %llx.\n",
                static_cast<long long>(outputGpus));
        retSt = 100;
    }

    dcm.MatchByIO(affinityGroups, &top, potentialCpuMatches, 4, outputGpus);

    if (outputGpus != 0xF)
    {
        fprintf(stderr, "Output gpus should've been set to 0xF, but is %llx.\n",
                static_cast<long long>(outputGpus));
        retSt = 100;
    }
    
    // Alter the topology
    top.element[5].path = DCGM_TOPOLOGY_LWLINK3;

    dcm.MatchByIO(affinityGroups, &top, potentialCpuMatches, 2, outputGpus);

    if (outputGpus != 0xC)
    {
        fprintf(stderr, "Output gpus should've been set to 0xC, but is %llx.\n",
                static_cast<long long>(outputGpus));
    }

    return retSt;
}


/*****************************************************************************/
int TestCacheManager::TestWatchesVisited()
{
    DcgmCacheManager *cacheManager = new DcgmCacheManager();
    int st, retSt = 0, i, j;
    std::vector<unsigned short>validFieldIds;
    std::bitset<DCGM_FI_MAX_FIELDS>fieldIdIsGlobal; /* 1/0 of if each entry in validFieldIds is global (1) or not (0) */
    timelib64_t watchFreq = 30000000; /* 30 seconds */
    double maxSampleAge = 3600;
    int maxKeepSamples = 0;
    dcgmcm_watch_info_t watchInfo;
    dcgm_field_meta_p fieldMeta;
    dcgm_field_entity_group_t fieldEntityGroup;
    std::vector<lwmlVgpuInstance_t>vgpuIds;
    std::vector<lwmlVgpuInstance_t>::iterator vgpuIt;
    int numVgpus = 3; /* Number of VGPUs to create per GPU */
    unsigned int fakeGpuId;
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);

    if(m_gpus.size() < 1)
    {
        fprintf(stderr, "Can't watch TestWatchesVisited() without live GPUs.\n");
        retSt = 100;
        goto CLEANUP;
    }


    if(!cacheManager)
    {
        fprintf(stderr, "Unable to allocate cacheManager\n");
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->Init(1, 86400.0);
    if(st)
    {
        fprintf(stderr, "cacheManager->Init returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->Start();
    if(st)
    {
        fprintf(stderr, "cacheManager->Start() returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    fakeGpuId = cacheManager->AddFakeGpu();
    if(fakeGpuId == DCGM_GPU_ID_BAD)
    {
        if(m_gpus.size() >= DCGM_MAX_NUM_DEVICES)
        {
            printf("Skipping TestWatchesVisited() due to having no space for a fake GPU.\n");
            retSt = 0;
            goto CLEANUP;
        }

        fprintf(stderr, "Unable to add fake GPU\n");
        retSt = -1;
        goto CLEANUP;
    }

    cacheManager->GetValidFieldIds(validFieldIds, false);

    /* Mark each field as global or not */
    for(i = 0; i < (int)validFieldIds.size(); i++)
    {
        fieldMeta = DcgmFieldGetById(validFieldIds[i]);
        if (!fieldMeta)
        {
            fprintf(stderr, "FieldId %u had no metadata.\n", validFieldIds[i]);
            retSt = -1;
            goto CLEANUP;
        }

        if(fieldMeta->scope == DCGM_FS_GLOBAL)
            fieldIdIsGlobal[fieldMeta->fieldId] = 1;
        else
            fieldIdIsGlobal[fieldMeta->fieldId] = 0;
    }


    /* Watch all valid fields on all GPUs. We're going to check to make sure that they
     * were visited by the watch code */
    for(i = 0; i < (int)m_gpus.size(); i++)
    {
        vgpuIds = gpuIdToVgpuList(m_gpus[i].gpuId, numVgpus);

        st = cacheManager->ManageVgpuList(m_gpus[i].gpuId, (unsigned int *)&vgpuIds[0]);
        if(st)
        {
            fprintf(stderr, "cacheManager->ManageVgpuList failed with %d", (int)st);
            retSt = -1;
            goto CLEANUP;
        }

        for(j = 0; j < (int)validFieldIds.size(); j++)
        {
            fieldEntityGroup = DCGM_FE_GPU;
            if(fieldIdIsGlobal[validFieldIds[j]])
                fieldEntityGroup = DCGM_FE_NONE;

            st = cacheManager->AddFieldWatch(fieldEntityGroup, m_gpus[i].gpuId, validFieldIds[j],
                                             watchFreq, maxSampleAge, maxKeepSamples, watcher, false);
            if(st == DCGM_ST_REQUIRES_ROOT && geteuid() != 0)
            {
                printf("Skipping fieldId %u that isn't supported for non-root\n",
                       validFieldIds[j]);
                validFieldIds.erase(validFieldIds.begin()+j);
                j--; /* Since we deleted one, we're going to be at the same index */
                continue;
            }
            if(st) /* Purposely leaving as if rather than else if in case both of above aren't true */
            {
                fprintf(stderr, "cacheManager->AddFieldWatch() returned %d\n", st);
                        retSt = -1;
                goto CLEANUP;
            }

            /* Don't do VGPU watches for global fields */
            if(fieldEntityGroup == DCGM_FE_NONE)
                continue;

            fieldEntityGroup = DCGM_FE_VGPU;


            /* Add a watch on every GPU field for every VGPU */
            for(vgpuIt = vgpuIds.begin()+1; vgpuIt != vgpuIds.end(); ++vgpuIt)
            {
                st = cacheManager->AddFieldWatch(fieldEntityGroup, *vgpuIt, validFieldIds[j],
                                                 watchFreq, maxSampleAge, maxKeepSamples, watcher, false);
                if(st)
                {
                    fprintf(stderr, "cacheManager->AddFieldWatch() returned %d\n", st);
                            retSt = -1;
                    goto CLEANUP;
                }
            }
        }
    }

    /* Force a field update */
    cacheManager->UpdateAllFields(1);

    /* Verify that all fields of all GPUs were visited */
    for(i = 0; i < (int)m_gpus.size(); i++)
    {
        for(j = 0; j < (int)validFieldIds.size(); j++)
        {
            fieldEntityGroup = DCGM_FE_GPU;
            if(fieldIdIsGlobal[validFieldIds[j]])
                fieldEntityGroup = DCGM_FE_NONE;

            st = cacheManager->GetEntityWatchInfoSnapshot(fieldEntityGroup, m_gpus[i].gpuId,
                                                          validFieldIds[j], &watchInfo);
            if(st)
            {
                fprintf(stderr, "cacheManager->GetEntityWatchInfoSnapshot() returned %d\n", st);
                        retSt = 200;
                goto CLEANUP;
            }

            if(!watchInfo.isWatched)
            {
                fprintf(stderr, "gpuId %u, fieldId %u was not watched.\n",
                        m_gpus[i].gpuId, validFieldIds[j]);
                retSt = 300;
                continue;
            }

            if(!watchInfo.lastQueriedUsec)
            {
                fprintf(stderr, "gpuId %u, fieldId %u has never updated.\n",
                        m_gpus[i].gpuId, validFieldIds[j]);
                retSt = 400;
                continue;
            }

            /* Don't check VGPU watches for global fields */
            if(fieldEntityGroup == DCGM_FE_NONE)
                continue;
            fieldEntityGroup = DCGM_FE_VGPU;
            vgpuIds = gpuIdToVgpuList(m_gpus[i].gpuId, numVgpus);

            /* Add a watch on every GPU field for every VGPU */
            for(vgpuIt = vgpuIds.begin()+1; vgpuIt != vgpuIds.end(); ++vgpuIt)
            {
                st = cacheManager->GetEntityWatchInfoSnapshot(fieldEntityGroup, *vgpuIt,
                                                              validFieldIds[j], &watchInfo);
                if(st)
                {
                    fprintf(stderr, "cacheManager->GetEntityWatchInfoSnapshot() returned %d\n", st);
                            retSt = 500;
                    goto CLEANUP;
                }

                if(!watchInfo.isWatched)
                {
                    fprintf(stderr, "gpuId %u, vgpu %u, fieldId %u was not watched.\n",
                            m_gpus[i].gpuId, *vgpuIt, validFieldIds[j]);
                    retSt = 600;
                    continue;
                }

                if(!watchInfo.lastQueriedUsec)
                {
                    fprintf(stderr, "gpuId %u, vgpu %u, fieldId %u has never updated.\n",
                            m_gpus[i].gpuId, *vgpuIt, validFieldIds[j]);
                    retSt = 700;
                    continue;
                }
            }
        }
    }


    CLEANUP:
    if(cacheManager)
        delete(cacheManager);
    cacheManager = 0;

    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestManageVgpuList()
{
    int st, retSt = 0;
    DcgmCacheManager *cacheManager = new DcgmCacheManager();
    dcgm_field_meta_p fieldMeta = 0;
    unsigned int gpuId = 0;
    dcgmcm_sample_t sample;

    memset(&sample, 0, sizeof(sample));

    /* First element represents vgpuCount */
    lwmlVgpuInstance_t vgpuIds[NUM_VGPU_LISTS][TEST_MAX_NUM_VGPUS_PER_GPU] = {
                                {11,41,52,61,32,45,91,21,43,29,19,93,0,0,0,0},
                                {8,32,45,91,21,43,29,19,93,0,0,0,0,0,0,0},
                                {7,41,52,32,45,91,21,43,0,0,0,0,0,0,0,0},
                                {4,41,32,91,43,0,0,0,0,0,0,0,0,0,0,0},
                                {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
                               };

    if(!cacheManager)
    {
        fprintf(stderr, "Unable to allocate cacheManager\n");
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->Init(1, 86400.0);
    if(st)
    {
        fprintf(stderr, "cacheManager->Init returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->Start();
    if(st)
    {
        fprintf(stderr, "cacheManager->Start() returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    for(int i = 0; i < NUM_VGPU_LISTS; i++)
    {
        /* First element of vgpuIds array must hold the vgpuCount */
        st = cacheManager->ManageVgpuList(gpuId, (lwmlVgpuInstance_t*)(vgpuIds + i));
        /* DCGM_ST_GENERIC_ERROR returned from cacheManager when Vgpu list for this GPU does not matches with given sample set of vgpuIds. */
        if(st != DCGM_ST_OK)
        {
            fprintf(stderr, "ManageVgpuList returned unexpected st %d "
                    "Expected %d\n", st, DCGM_ST_OK);
            retSt = st;
            goto CLEANUP;
        }

        fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_VGPU_VM_NAME);
        if(!fieldMeta)
        {
            fprintf(stderr, "Unable to get fieldMeta for field DCGM_FI_DEV_VGPU_VM_NAME\n");
            retSt = 1;
            goto CLEANUP;
        }

        for(int j = 0; j < (TEST_MAX_NUM_VGPUS_PER_GPU - 1); j++)
        {
            /* Since 0 is not a valid vgpuId, so existing the loop as subsequent elements will also be zero */
            if (vgpuIds[i][j+1] == 0)
                break;
            st = InjectSampleHelper(cacheManager, fieldMeta, DCGM_FE_VGPU, vgpuIds[i][j+1], 0);
            if(st)
            {
                fprintf(stderr, "InjectSampleHelper returned unexpected st %d  "
                        "Expected 0\n", st);
                retSt = st;
                goto CLEANUP;
            }

            /* To verify retrieved sample against whatever sample which was injected in the cache */
            memset(&sample, 0, sizeof(sample));
            st = cacheManager->GetLatestSample(DCGM_FE_VGPU, vgpuIds[i][j+1], fieldMeta->fieldId,
                                               &sample, 0);
            if(st != DCGM_ST_OK)
            {
                fprintf(stderr, "GetLatestSample returned unexpected st %d . "
                        "Expected %d\n", st, DCGM_ST_OK);
                retSt = st;
                goto CLEANUP;
            }

            if(std::string(sample.val.str) != std::string("lwpu"))
            {
                fprintf(stderr, "GetLatestSample returned unexpected sample. "
                    "Expected 'lwpu'\n");
                cacheManager->FreeSamples(&sample, 1, fieldMeta->fieldId);
                retSt = 100;
                goto CLEANUP;
            }

            st = cacheManager->FreeSamples(&sample, 1, fieldMeta->fieldId);
            if(st)
            {
                fprintf(stderr, "Got st %d from FreeSamples for field %d\n", st, (int)fieldMeta->fieldId);
                retSt = st;
                goto CLEANUP;
            }

        }


        /* Inject-retrieve routine for vGPU field 'DCGM_FI_DEV_VGPU_TYPE' for single vgpuId(41) which is of int type of value */
        if(i == 0)
        {
            fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_VGPU_TYPE);
            if(!fieldMeta)
            {
                fprintf(stderr, "Unable to get fieldMeta for field DCGM_FI_DEV_VGPU_TYPE\n");
                retSt = 1;
                goto CLEANUP;
            }
            st = InjectSampleHelper(cacheManager, fieldMeta, DCGM_FE_VGPU, 41, 0);
            if(st)
            {
                fprintf(stderr, "InjectSampleHelper returned unexpected st %d  "
                        "Expected 0\n", st);
                retSt = st;
                goto CLEANUP;
            }

            /* To verify retrieved sample against whatever sample which was injected in the cache */
            st = cacheManager->GetLatestSample(DCGM_FE_VGPU, 41, fieldMeta->fieldId,
                                               &sample, 0);
            if(st != DCGM_ST_OK)
            {
                fprintf(stderr, "GetLatestSample returned unexpected st %d . "
                        "Expected %d\n", st, DCGM_ST_OK);
                retSt = st;
                goto CLEANUP;
            }
            if(sample.val.i64 != 1)
            {
                fprintf(stderr, "GetLatestSample returned unexpected injected sample . "
                        "Expected 1\n");
                retSt = 100;
                goto CLEANUP;
            }
        }

        /* To verify that no samples retrieved for a vgpuId 41 which has been removed from the List. */
        if(i == (NUM_VGPU_LISTS - 1))
        {
            /* For vGPU field 'DCGM_FI_DEV_VGPU_VM_NAME' */
            st = cacheManager->GetLatestSample(DCGM_FE_VGPU, 41, fieldMeta->fieldId,
                                               &sample, 0);
            if(st != DCGM_ST_NOT_WATCHED)
            {
                fprintf(stderr, "GetLatestSample returned unexpected st %d . "
                        "Expected %d\n", st, DCGM_ST_NOT_WATCHED);
                retSt = st;
                goto CLEANUP;
            }

            /* For vGPU field 'DCGM_FI_DEV_VGPU_TYPE' */
            fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_VGPU_TYPE);
            if(!fieldMeta)
            {
                fprintf(stderr, "Unable to get fieldMeta for field DCGM_FI_DEV_VGPU_TYPE\n");
                retSt = 1;
                goto CLEANUP;
            }

            st = cacheManager->GetLatestSample(DCGM_FE_VGPU, 41, fieldMeta->fieldId,
                                               &sample, 0);
            if(st != DCGM_ST_NOT_WATCHED)
            {
                fprintf(stderr, "GetLatestSample returned unexpected st %d . "
                        "Expected %d\n", st, DCGM_ST_NOT_WATCHED);
                retSt = st;
                goto CLEANUP;
            }
        }
    }

    CLEANUP:
    if(cacheManager)
        delete(cacheManager);
    cacheManager = 0;

    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestInjection()
{
    int st, retSt = 0;
    int fieldIndex;
    DcgmCacheManager *cacheManager = new DcgmCacheManager();
    dcgm_field_meta_p fieldMeta = 0;
    unsigned int gpuId = 0;
    dcgmcm_sample_t sample;

    memset(&sample, 0, sizeof(sample));

    if(!cacheManager)
    {
        fprintf(stderr, "Unable to allocate cacheManager\n");
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->Init(1, 86400.0);
    if(st)
    {
        fprintf(stderr, "cacheManager->Init returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->Start();
    if(st)
    {
        fprintf(stderr, "cacheManager->Start() returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    /* test for each field */
    for(fieldIndex = 1; fieldIndex < DCGM_FI_MAX_FIELDS; fieldIndex++)
    {
        fieldMeta = DcgmFieldGetById(fieldIndex);
        if(!fieldMeta)
        {
            /* fieldIds are sparse */
            continue;
        }

        memset(&sample, 0, sizeof(sample));

        /* Verify there are no samples yet */
        st = cacheManager->GetLatestSample(DCGM_FE_GPU, gpuId, fieldMeta->fieldId,
                                           &sample, 0);
        if(st != DCGM_ST_NOT_WATCHED)
        {
            fprintf(stderr, "GetLatestSample returned unexpected st %d for field %d. "
                    "Expected not watched (-15)\n", st, fieldIndex);
            retSt = 100;
            goto CLEANUP;
        }

        /* Don't need to free sample here since we only get here with DCGM_ST_NOT_WATCHED */

        /* Inject a fake value */
        st = InjectSampleHelper(cacheManager, fieldMeta, DCGM_FE_GPU, gpuId, 0);
        if(st)
        {
            fprintf(stderr, "InjectSampleHelper returned unexpected st %d for field %d. "
                    "Expected 0\n", st, fieldIndex);
            retSt = 200;
            goto CLEANUP;
        }

        /* now the sample should be there */
        st = cacheManager->GetLatestSample(DCGM_FE_GPU, gpuId, fieldMeta->fieldId,
                                           &sample, 0);
        if(st != DCGM_ST_OK)
        {
            fprintf(stderr, "GetLatestSample returned unexpected st %d for field %d. "
                    "Expected 0\n", st, fieldIndex);
            retSt = 300;
            goto CLEANUP;
        }

        st = cacheManager->FreeSamples(&sample, 1, fieldMeta->fieldId);
        if(st)
        {
            fprintf(stderr, "Got st %d from FreeSamples for field %d\n", st, (int)fieldMeta->fieldId);
            retSt = 400;
            goto CLEANUP;
        }

    }


    CLEANUP:
    if(cacheManager)
        delete(cacheManager);
    cacheManager = 0;

    return retSt;
}

/*****************************************************************************/
bool AllowEntryCB(timeseries_entry_p entry, void *userData)
{
    if (entry->val2.i64 % 2 == 0) {
        return true;
    } else {
        return false;
    }
}


/*****************************************************************************/
int TestCacheManager::TestSummary()
{
    int st, retSt = 0;
    int fieldIndex;
    DcgmCacheManager *cacheManager = new DcgmCacheManager();
    dcgm_field_meta_p fieldMeta = 0;
    unsigned int gpuId = 0;
    dcgmcm_sample_t sample;
    DcgmcmSummaryType_t summaryTypes[2];

    memset(&sample, 0, sizeof(sample));

    if(!cacheManager)
    {
        fprintf(stderr, "Unable to allocate cacheManager\n");
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->Init(1, 86400.0);
    if(st)
    {
        fprintf(stderr, "cacheManager->Init returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->Start();
    if(st)
    {
        fprintf(stderr, "cacheManager->Start() returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }
    

    /* test for each field */
    for(fieldIndex = 1; fieldIndex < DCGM_FI_MAX_FIELDS; fieldIndex++)
    {
        fieldMeta = DcgmFieldGetById(fieldIndex);
        if(!fieldMeta)
        {
            /* fieldIds are sparse */
            continue;
        }
        
        /* Test only for Inte64 and device scoped fields */
        if (!((fieldMeta->fieldType == DCGM_FT_INT64) && (fieldMeta->scope == DCGM_FS_DEVICE))) {
            continue;
        }

        memset(&sample, 0, sizeof(sample));

        /* Verify there are no samples yet */
        st = cacheManager->GetLatestSample(DCGM_FE_GPU, gpuId, fieldMeta->fieldId,
                                           &sample, 0);
        if(st != DCGM_ST_NOT_WATCHED)
        {
            fprintf(stderr, "GetLatestSample returned unexpected st %d for field %d. "
                    "Expected not watched (-15)\n", st, fieldIndex);
            retSt = 100;
            goto CLEANUP;
        }

        /* Don't need to free sample here since we only get here with DCGM_ST_NOT_WATCHED */
        
        timelib64_t startTimeStamp = 0;
        timelib64_t endTimeStamp = 0;
        int numEntries = 100; /* Keep it a even value */
        int value = 1;
        for (int i = 1; i <= numEntries; i++) {
            memset(&sample, 0, sizeof(sample));
            endTimeStamp = startTimeStamp + i;
            sample.val.i64 = 1;
            sample.val2.i64 = i;
            sample.timestamp = endTimeStamp;            

            /* Inject a fake value */
            st = InjectUserProvidedSampleHelper(cacheManager, sample, fieldMeta, gpuId);
            if(st)
            {
                fprintf(stderr, "InjectUserProvidedSampleHelper returned unexpected st %d for field %d. "
                    "Expected 0\n", st, fieldIndex);
                retSt = 200;
                goto CLEANUP;
            }
        }
        
        long long i64Val = 0;
        summaryTypes[0] = DcgmcmSummaryTypeSum; 
        (void)cacheManager->GetInt64SummaryData(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, 1, &summaryTypes[0],
                                                &i64Val, startTimeStamp, endTimeStamp, AllowEntryCB, NULL);
        
        long long expectedValue = (numEntries * value)/2;
        
        if (i64Val != expectedValue) {
            fprintf(stderr, "For field %d Got summary as %lld Expected %lld\n", fieldMeta->fieldId, i64Val, expectedValue);
        }
        
        st = cacheManager->FreeSamples(&sample, 1, fieldMeta->fieldId);
        if(st)
        {
            fprintf(stderr, "Got st %d from FreeSamples for field %d\n", st, (int)fieldMeta->fieldId);
            retSt = 400;
            goto CLEANUP;
        }

    }


    CLEANUP:
    deleteNotNull(cacheManager);

    return retSt;
}




/*****************************************************************************/
int TestCacheManager::InjectSampleHelper(DcgmCacheManager *cacheManager,
                                         dcgm_field_meta_p fieldMeta,
                                         dcgm_field_entity_group_t entityGroupId,
                                         dcgm_field_eid_t entityId,
                                         timelib64_t timestamp)
{
    int st;
    dcgmcm_sample_t sample;

    memset(&sample, 0, sizeof(sample));

    if(!fieldMeta)
        return DCGM_ST_BADPARAM;

    sample.timestamp = timestamp;

    /* Do per-type assignment of value */
    switch(fieldMeta->fieldType)
    {
        case DCGM_FT_DOUBLE:
            sample.val.d = 1.0;
            break;

        case DCGM_FT_TIMESTAMP:
            sample.val.i64 = timelib_usecSince1970();
            break;

        case DCGM_FT_INT64:
            sample.val.i64 = 1;
            break;

        case DCGM_FT_STRING:
            sample.val.str = (char *)"lwpu"; /* Use static string so we don't have to alloc/free */
            sample.val2.ptrSize = strlen(sample.val.str) + 1;
            break;

        case DCGM_FT_BINARY:
            /* Just insert any blob of data */
            sample.val.blob = &sample;
            sample.val2.ptrSize = sizeof(sample);
            break;

        default:
            fprintf(stderr, "Can't inject unknown type %c\n", fieldMeta->fieldType);
            return -1;
    }

    /* Actually inject the value */
    st = cacheManager->InjectSamples(entityGroupId, entityId, fieldMeta->fieldId, &sample, 1);
    if(st)
    {
        fprintf(stderr, "InjectSamples returned %d for field %d\n",
                st, (int)fieldMeta->fieldId);
        return -1;
    }

    /* Don't free sample here since the string is a static value */

    return 0;
}


/*****************************************************************************/
int TestCacheManager::InjectUserProvidedSampleHelper(DcgmCacheManager *cacheManager,
                                                        dcgmcm_sample_t sample,
                                                        dcgm_field_meta_p fieldMeta, unsigned int gpuId)
{
    int st;

    if(!fieldMeta)
        return DCGM_ST_BADPARAM;

    /* Actually inject the value */
    st = cacheManager->InjectSamples(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, &sample, 1);
    if(st)
    {
        fprintf(stderr, "InjectSamples returned %d for field %d\n",
                st, (int)fieldMeta->fieldId);
        return -1;
    }

    /* Don't free sample here since the string is a static value */

    return 0;
}



/*****************************************************************************/
timelib64_t TestCacheManager::GetAverageSampleFrequency(dcgmcm_sample_t *samples, int Nsamples)
{
    int sampleIndex;
    timelib64_t averageDiff = 0;

    if(!samples || Nsamples < 1)
    {
        fprintf(stderr, "Bad parameter to GetAverageSampleFrequency\n");
        return 0;
    }

    for(sampleIndex = 1; sampleIndex < Nsamples; sampleIndex++)
    {
        averageDiff += (samples[sampleIndex].timestamp - samples[sampleIndex-1].timestamp);
    }

    averageDiff /= (Nsamples - 1);
    return averageDiff;
}

/*****************************************************************************/
int TestCacheManager::TestRecordTiming()
{
    int i, st, retSt = 0;
    int fieldIndex;
    DcgmCacheManager *cacheManager = new DcgmCacheManager();
    dcgm_field_meta_p fieldMeta = 0;
    unsigned int gpuId = 0;
    dcgmcm_sample_t samples[100];
    int Msamples = 100; /* Same size as samples[] */
    int Nsamples;
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);

    int Nfields = 3; /* same as size of arrays below */
    unsigned short fieldIds[3] = {DCGM_FI_DEV_NAME, DCGM_FI_DEV_BRAND, DCGM_FI_DEV_SERIAL};
    timelib64_t fieldFrequency[3] = {100000, 200000, 500000};

    memset(&samples, 0, sizeof(samples));

    if(!cacheManager)
    {
        fprintf(stderr, "Unable to allocate cacheManager\n");
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->Init(0, 86400.0);
    if(st)
    {
        fprintf(stderr, "cacheManager->Init returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    for(i = 0; i < Nfields; i++)
    {
        st = cacheManager->AddFieldWatch(DCGM_FE_GPU, gpuId, fieldIds[i], fieldFrequency[i], 86400.0, 0, watcher, false);
        if(st)
        {
            fprintf(stderr, "Error from AddFieldWatch index %d: %d\n", i, st);
            retSt = -1;
            goto CLEANUP;
        }
    }

    st = cacheManager->Start();
    if(st)
    {
        fprintf(stderr, "cacheManager->Start() returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    /* Wait for one update cycle to complete */
    cacheManager->UpdateAllFields(1);


    /* Sleep for 10x the last fieldFrequency (1000usec / 100) = 10x in msec */
    lwosSleep(fieldFrequency[Nfields-1]/100);


    /* Wait for one final update */
    cacheManager->UpdateAllFields(1);

    for(i = 0; i < Nfields; i++)
    {
        timelib64_t averageDiff, tenPercent;


        Nsamples = Msamples;
        st = cacheManager->GetSamples(DCGM_FE_GPU, gpuId, fieldIds[i], &samples[0], &Nsamples,
                                      0, 0, DCGM_ORDER_ASCENDING);
        if(st)
        {
            fprintf(stderr, "Got st %d from GetSamples for field %d\n", st, (int)fieldIds[i]);
            retSt = -1;
            goto CLEANUP;
        }

        averageDiff = GetAverageSampleFrequency(samples, Nsamples);
        tenPercent = fieldFrequency[i] / 10;

        if(averageDiff < fieldFrequency[i] - tenPercent)
        {
            fprintf(stderr, "Field frequency index %d of %lld < %lld\n",
                    i, (long long int)averageDiff, (long long int)(fieldFrequency[i] - tenPercent));
            retSt = -1;
            /* Keep going */
        }
        else if(averageDiff > fieldFrequency[i] + tenPercent)
        {
            fprintf(stderr, "Field frequency index %d of %lld > %lld\n",
                    i, (long long int)averageDiff, (long long int)(fieldFrequency[i] + tenPercent));
            retSt = -1;
            /* Keep going */
        }

        st = cacheManager->FreeSamples(&samples[0], Nsamples, fieldIds[i]);
        if(st)
        {
            fprintf(stderr, "Got st %d from FreeSamples for field %d\n", st, (int)fieldIds[i]);
            retSt = -1;
            goto CLEANUP;
        }
    }

    CLEANUP:
    if(cacheManager)
        delete(cacheManager);
    cacheManager = 0;

    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestTimeBasedQuota()
{
    int st, retSt = 0, i = 0;
    DcgmCacheManager *cacheManager = new DcgmCacheManager();
    test_lwcm_gpu_p gpu = 0;
    int Msamples = 100;
    dcgmcm_sample_t samples[100];
    unsigned int gpuId;
    dcgm_field_meta_p fieldMeta = 0;
    timelib64_t startTime, sampleTime, now;
    double maxKeepAge = 1.0;
    timelib64_t maxKeepAgeUsec = (timelib64_t)(1000000.0 * maxKeepAge);
    dcgmReturn_t lwcmSt;
    int Nsamples;
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);

    fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_GPU_TEMP);
    if(!fieldMeta)
    {
        fprintf(stderr, "Unable to get fieldMeta for field DCGM_FI_DEV_GPU_TEMP\n");
        retSt = 1;
        goto CLEANUP;
    }

    if(!cacheManager)
    {
        fprintf(stderr, "Unable to allocate cacheManager\n");
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->Init(1, 86400.0);
    if(st)
    {
        fprintf(stderr, "cacheManager->Init returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    gpuId = cacheManager->AddFakeGpu();
    if(gpuId == DCGM_GPU_ID_BAD)
    {
        if(m_gpus.size() >= DCGM_MAX_NUM_DEVICES)
        {
            printf("Skipping TestTimeBasedQuota() due to having no space for a fake GPU.\n");
            retSt = 0;
            goto CLEANUP;
        }

        fprintf(stderr, "Unable to add fake GPU\n");
        retSt = -1;
        goto CLEANUP;
    }

    /* Add a watch to populate metadata for the field (like maxKeepAge) */
    st = cacheManager->AddFieldWatch(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, 1000000, maxKeepAge, 0, watcher, false);
    if(st)
    {
        fprintf(stderr, "cacheManager->AddFieldWatch returned %d\n", st);
        retSt = 1;
        goto CLEANUP;
    }

    st = cacheManager->Start();
    if(st)
    {
        fprintf(stderr, "cacheManager->Start() returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    startTime = timelib_usecSince1970();
    now = startTime;

    /* Inject samples from 10 seconds ago to now so we have at least one but
     * most should be pruned by the time quota */
    for(sampleTime = startTime - 10000000; sampleTime < now; sampleTime += 2000000)
    {
        /* Readjust "now" since doing stuff takes time and we're splitting microseconds */
        now = timelib_usecSince1970();

        st = InjectSampleHelper(cacheManager, fieldMeta, DCGM_FE_GPU, gpuId, sampleTime);
        if(st)
        {
            fprintf(stderr, "InjectSampleHelper returned %d\n", st);
            retSt = -1;
            goto CLEANUP;
        }
    }

    /* This will return the oldest sample. This should be within maxKeepAgeUsec of "now" */
    Nsamples = Msamples;
    lwcmSt = cacheManager->GetSamples(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, &samples[0], &Nsamples,
                                      0, 0, DCGM_ORDER_ASCENDING);
    if(lwcmSt != DCGM_ST_OK)
    {
        fprintf(stderr, "GetSamples returned %d\n", st);
        retSt = 1;
        goto CLEANUP;
    }

    if(samples[0].timestamp < now - maxKeepAgeUsec)
    {
        fprintf(stderr, "Got sample that should have been pruned:\n\tage %lld"
                "\n\tnow: %lld\n\tsample ts: %lld\n\tmaxAge: %lld\n",
                (long long)(now - samples[0].timestamp), (long long)now,
                (long long)samples[0].timestamp, (long long)maxKeepAgeUsec);
        retSt = 1;
        goto CLEANUP;
    }

#if 0 //Debug printing
    for(i = 0; i < Nsamples; i++)
    {
        printf("ts: %lld. age: %lld\n", (long long)samples[i].timestamp, (long long)(now - samples[i].timestamp));
    }
#endif

    CLEANUP:
    if(cacheManager)
        delete(cacheManager);
    cacheManager = 0;
    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestCountBasedQuota()
{
    int st, retSt = 0, i = 0;
    DcgmCacheManager *cacheManager = new DcgmCacheManager();
    test_lwcm_gpu_p gpu = 0;
    int Msamples = 100;
    dcgmcm_sample_t samples[100];
    unsigned int gpuId;
    dcgm_field_meta_p fieldMeta = 0;
    timelib64_t startTime, sampleTime;
    int maxKeepSamples = 5;
    dcgmReturn_t lwcmSt;
    int Nsamples;
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);

    fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_GPU_TEMP);
    if(!fieldMeta)
    {
        fprintf(stderr, "Unable to get fieldMeta for field DCGM_FI_DEV_GPU_TEMP\n");
        retSt = 1;
        goto CLEANUP;
    }

    if(!cacheManager)
    {
        fprintf(stderr, "Unable to allocate cacheManager\n");
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->Init(1, 86400.0);
    if(st)
    {
        fprintf(stderr, "cacheManager->Init returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    gpuId = cacheManager->AddFakeGpu();
    if(gpuId == DCGM_GPU_ID_BAD)
    {
        if(m_gpus.size() >= DCGM_MAX_NUM_DEVICES)
        {
            printf("Skipping TestCountBasedQuota() due to having no space for a fake GPU.\n");
            retSt = 0;
            goto CLEANUP;
        }

        fprintf(stderr, "Unable to add fake GPU\n");
        retSt = -1;
        goto CLEANUP;
    }

    /* Add a watch to populate metadata for the field (like maxKeepAge) */
    st = cacheManager->AddFieldWatch(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, 1000000, 86400.0, maxKeepSamples, watcher, false);
    if(st)
    {
        fprintf(stderr, "cacheManager->AddFieldWatch returned %d\n", st);
        retSt = 1;
        goto CLEANUP;
    }

    st = cacheManager->Start();
    if(st)
    {
        fprintf(stderr, "cacheManager->Start() returned %d\n", st);
        retSt = -1;
        goto CLEANUP;
    }

    startTime = timelib_usecSince1970();

    /* Inject maxKeepSamples historical samples. Starttime is callwlated so that the
     * last sample will be right before now */
    sampleTime = startTime - (1000000 * maxKeepSamples * 2);
    for(Nsamples = 0; Nsamples < (maxKeepSamples * 2); Nsamples++)
    {
        st = InjectSampleHelper(cacheManager, fieldMeta, DCGM_FE_GPU, gpuId, sampleTime);
        if(st)
        {
            fprintf(stderr, "InjectSampleHelper returned %d\n", st);
            retSt = -1;
            goto CLEANUP;
        }

        sampleTime += 1000000;
    }

    /* This will return the oldest sample. This should be within maxKeepAgeUsec of "now" */
    Nsamples = Msamples;
    lwcmSt = cacheManager->GetSamples(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, &samples[0], &Nsamples,
                                      0, 0, DCGM_ORDER_ASCENDING);
    if(lwcmSt != DCGM_ST_OK)
    {
        fprintf(stderr, "GetSamples returned %d\n", st);
        retSt = 1;
        goto CLEANUP;
    }

    /* Allow maxKeepSamples-1 because all count quotas are colwerted to time quotas.
       Based off timing, we could have one less sample than we expect */
    if(Nsamples != maxKeepSamples && Nsamples != maxKeepSamples - 1)
    {
        fprintf(stderr, "Expected %d samples. Got %d samples\n", maxKeepSamples, Nsamples);
        retSt = 1;
        goto CLEANUP;
    }

#if 0 //Debug printing
    for(i = 0; i < Nsamples; i++)
    {
        printf("ts: %lld\n", (long long)samples[i].timestamp);
    }
#endif

    CLEANUP:
    if(cacheManager)
        delete(cacheManager);
    cacheManager = 0;
    return retSt;
}

class IterateFieldsFunctor
{
public:
    IterateFieldsFunctor()
        : gpuTempIterated(false), driverVersionIterated(false), itCount(0) {}
    dcgmReturn_t operator()(unsigned short fieldId)
    {
        itCount++;
        if (fieldId == DCGM_FI_DEV_GPU_TEMP)
            gpuTempIterated = true;
        if (fieldId == DCGM_FI_DRIVER_VERSION)
            driverVersionIterated = true;

        return DCGM_ST_OK;
    }

    bool gpuTempIterated;
    bool driverVersionIterated;
    unsigned int itCount;
};

class IterateGpuFieldsFunctor
{
public:
    bool gpuTempIterated;
    bool driverVersionIterated;
    unsigned int itCount;

    IterateGpuFieldsFunctor()
        : gpuTempIterated(false), driverVersionIterated(false), itCount(0) {}
    dcgmReturn_t operator()(unsigned short fieldId, unsigned int gpuId)
    {
        itCount++;
        if (fieldId == DCGM_FI_DEV_GPU_TEMP)
            gpuTempIterated = true;
        if (fieldId == DCGM_FI_DRIVER_VERSION)
            driverVersionIterated = true;

        return DCGM_ST_OK;
    }
};

int TestCacheManager::TestWatchedFieldsIteration()
{
    dcgmReturn_t st;
    int retSt = 0;
    unsigned int gpuId = m_gpus[0].gpuId;
    IterateFieldsFunctor functor;
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);
    DcgmCacheManager *cacheManager = createCacheManager(1);
    if (!cacheManager)
    {
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP, 1000000, 86400.0, 10, watcher, false);
    if(st)
    {
        fprintf(stderr, "add gpu watch returned %d\n", st);
        retSt = 1;
        goto CLEANUP;
    }

    st = cacheManager->AddFieldWatch(DCGM_FE_NONE, gpuId, DCGM_FI_DRIVER_VERSION, 1000000, 86400.0, 10, watcher, false);
    if(st)
    {
        fprintf(stderr, "add global watch returned %d\n", st);
        retSt = 1;
        goto CLEANUP;
    }

    st = cacheManager->ForEachWatchedField(&functor);
    if (st)
    {
        fprintf(stderr, "cacheManager->ForEachWatchedField returned %d\n", st);
        retSt = 3;
        goto CLEANUP;
    }

    if (!functor.driverVersionIterated)
    {
        fprintf(stderr, "did not iterate over watched global field\n");
        retSt = 4;
    }
    if (!functor.gpuTempIterated)
    {
        fprintf(stderr, "did not iterate over watched gpu field\n");
        retSt = 4;
    }
    if (!functor.itCount == 2)
    {
        fprintf(stderr, "iterated over %d fields but expected it to be 2\n", functor.itCount);
        retSt = 4;
    }

CLEANUP:
    if (cacheManager)
    {
        delete cacheManager;
        cacheManager = NULL;
    }
    return retSt;
}

int TestCacheManager::TestWatchedFieldsIterationGpu()
{
    dcgmReturn_t st;
    int retSt = 0;
    IterateGpuFieldsFunctor functor;
    unsigned int gpuId = m_gpus[0].gpuId;
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);
    DcgmCacheManager *cacheManager = createCacheManager(1);
    if (!cacheManager)
    {
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP, 1000000, 86400.0, 10, watcher, false);
    if(st)
    {
        fprintf(stderr, "add gpu watch returned %d\n", st);
        retSt = 1;
        goto CLEANUP;
    }

    st = cacheManager->AddFieldWatch(DCGM_FE_NONE, gpuId, DCGM_FI_DRIVER_VERSION, 1000000, 86400.0, 10, watcher, false);
    if(st)
    {
        fprintf(stderr, "add global watch returned %d\n", st);
        retSt = 1;
        goto CLEANUP;
    }


    st = cacheManager->ForEachWatchedGpuField(&functor);
    if (st)
    {
        fprintf(stderr, "cacheManager->ForEachWatchedGpuField returned %d\n", st);
        retSt = 3;
        goto CLEANUP;
    }

    if (functor.driverVersionIterated)
    {
        fprintf(stderr, "iterated over global field when trying to iterate over gpu fields\n");
        retSt = 4;
    }
    if (!functor.gpuTempIterated)
    {
        fprintf(stderr, "did not iterate over watched gpu field\n");
        retSt = 4;
    }
    if (!functor.itCount == 1)
    {
        fprintf(stderr, "iterated over %d fields but expected it to be 1\n", functor.itCount);
        retSt = 4;
    }

CLEANUP:
    if (cacheManager)
    {
        delete cacheManager;
        cacheManager = NULL;
    }
    return retSt;
}
int TestCacheManager::TestWatchedFieldsIterationGlobal()
{
    dcgmReturn_t st;
    int retSt = 0;
    unsigned int gpuId = m_gpus[0].gpuId;
    IterateFieldsFunctor functor;
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);
    DcgmCacheManager *cacheManager = createCacheManager(1);
    if (!cacheManager)
    {
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP, 1000000, 86400.0, 10, watcher, false);
    if(st)
    {
        fprintf(stderr, "add gpu watch returned %d\n", st);
        retSt = 1;
        goto CLEANUP;
    }

    st = cacheManager->AddFieldWatch(DCGM_FE_NONE, gpuId, DCGM_FI_DRIVER_VERSION, 1000000, 86400.0, 10, watcher, false);
    if(st)
    {
        fprintf(stderr, "add global watch returned %d\n", st);
        retSt = 1;
        goto CLEANUP;
    }

    st = cacheManager->ForEachWatchedGlobalField(&functor);
    if (st)
    {
        fprintf(stderr, "cacheManager->ForEachWatchedGlobalField returned %d\n", st);
        retSt = 3;
        goto CLEANUP;
    }

    if (!functor.driverVersionIterated)
    {
        fprintf(stderr, "did not iterate over watched global field\n");
        retSt = 4;
    }
    if (functor.gpuTempIterated)
    {
        fprintf(stderr, "iterated over a gpu field when trying to iterate over global fields\n");
        retSt = 4;
    }
    if (!functor.itCount == 1)
    {
        fprintf(stderr, "iterated over %d fields but expected it to be 1\n", functor.itCount);
        retSt = 4;
    }

CLEANUP:
    if (cacheManager)
    {
        delete cacheManager;
        cacheManager = NULL;
    }
    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestTimedModeAwakeTime()
{
    test_lwcm_gpu_p gpu = 0;
    dcgmcm_runtime_stats_t stats;

    if(m_gpus.size() < 1)
    {
        std::cout << "Skipping TestTimedModeAwakeTime() with 0 GPUs" << std::endl;
        return 0;
    }

    memset(&stats, 0, sizeof(stats));

    DcgmCacheManager *cacheManager = createCacheManager(0);
    if (NULL == cacheManager)
        return -1;

    std::cout << "TestTimedModeAwakeTime sleeping for 10 seconds." << std::endl;
    /* Sleep for 10 seconds to allow timed mode to wake up a few times */
    usleep(10 * 1000000);

    cacheManager->GetRuntimeStats(&stats);

    long long onePercentUsec = (stats.awakeTimeUsec + stats.sleepTimeUsec) / 100;

    if(stats.awakeTimeUsec > onePercentUsec)
    {
        fprintf(stderr, "Timed mode using more than 1%% CPU on idle. awakeUsec %lld. "
                "1%% usec = %lld. Total usec: %lld\n",
                onePercentUsec, stats.awakeTimeUsec, (stats.awakeTimeUsec + stats.sleepTimeUsec));

        /* Don't fail in debug mode or if logging is enabled */
        if(!loggingDebugLevel && !IsDebugBuild())
        {
            deleteNotNull(cacheManager);
            return 1;
        }
    }
    else
    {
        printf("Timed mode numSleepsDone %lld, awakeTimeUsec %lld, sleepTimeUsec %lld, "
               "updateCycleFinished %lld (IsDebug %d, loggingLevel %d)\n",
               stats.numSleepsDone, stats.awakeTimeUsec, stats.sleepTimeUsec, stats.updateCycleFinished,
               IsDebugBuild(), loggingDebugLevel);
    }

    deleteNotNull(cacheManager);
    return 0;
}

/*****************************************************************************/
int TestCacheManager::TestLockstepModeAwakeTime()
{
    test_lwcm_gpu_p gpu = 0;
    dcgmcm_runtime_stats_t stats;

    if(m_gpus.size() < 1)
    {
        std::cout << "Skipping TestTimedModeAwakeTime() with 0 GPUs" << std::endl;
        return 0;
    }

    memset(&stats, 0, sizeof(stats));

    DcgmCacheManager *cacheManager = createCacheManager(1);
    if (NULL == cacheManager)
        return -1;

    std::cout << "TestLockstepModeAwakeTime sleeping for 10 seconds." << std::endl;
    /* Sleep for 10 seconds to allow timed mode to wake up a few times */
    usleep(10 * 1000000);

    cacheManager->GetRuntimeStats(&stats);

    long long onePercentUsec = (stats.awakeTimeUsec + stats.sleepTimeUsec) / 100;

    if(stats.awakeTimeUsec > onePercentUsec)
    {
        fprintf(stderr, "Lockstep mode using more than 1%% CPU on idle. awakeUsec %lld. "
                "1%% usec = %lld. Total usec: %lld (IsDebug %d, loggingLevel %d)\n",
                onePercentUsec, stats.awakeTimeUsec, (stats.awakeTimeUsec + stats.sleepTimeUsec),
                IsDebugBuild(), loggingDebugLevel);

        /* Don't fail in debug mode or if logging is enabled */
        if(!loggingDebugLevel && !IsDebugBuild())
        {
            deleteNotNull(cacheManager);
            return 1;
        }
    }
    else
    {
        printf("Timed mode numSleepsDone %lld, awakeTimeUsec %lld, sleepTimeUsec %lld, "
               "updateCycleFinished %lld (IsDebug %d, loggingLevel %d)\n",
               stats.numSleepsDone, stats.awakeTimeUsec, stats.sleepTimeUsec, stats.updateCycleFinished,
               IsDebugBuild(), loggingDebugLevel);
    }

    deleteNotNull(cacheManager);
    return 0;
}

/*****************************************************************************/
int TestCacheManager::TestUpdatePerf()
{
    test_lwcm_gpu_p gpu = 0;
    dcgmcm_sample_t sample;
    dcgmcm_runtime_stats_t stats, statsBefore;
    int i;
    timelib64_t startTime, endTime;
    dcgmReturn_t dcgmReturn;
    int numLoops = 1000;

    if(m_gpus.size() < 1)
    {
        std::cout << "Skipping TestUpdatePerf() with 0 GPUs" << std::endl;
        return 0;
    }

    memset(&stats, 0, sizeof(stats));
    memset(&statsBefore, 0, sizeof(stats));

    DcgmCacheManager *cacheManager = createCacheManager(1);
    if (NULL == cacheManager)
        return -1;

    /* Get stats before our test so we can see how many locks oclwrred during our test */
    cacheManager->GetRuntimeStats(&statsBefore);

    startTime = timelib_usecSince1970();

    /* Measure the time each wake-up takes in the cache manager */
    for(i = 0; i < numLoops; i++)
    {
        dcgmReturn = cacheManager->UpdateAllFields(1);
        if(dcgmReturn != DCGM_ST_OK)
        {
            fprintf(stderr, "Got dcgmReturn %d from UpdateAllFields()\n", (int)dcgmReturn);
            deleteNotNull(cacheManager);
            return 100;
        }
    }

    endTime = timelib_usecSince1970();

    printf("TestUpdatePerf completed %d UpdateAllFields(1) in %lld usec\n",
           numLoops, (long long)(endTime-startTime));

    cacheManager->GetRuntimeStats(&stats);

    if(stats.updateCycleFinished < numLoops)
    {
        fprintf(stderr, "stats.updateCycleFinished %d < numLoops %d\n",
                (int)stats.updateCycleFinished, numLoops);
        deleteNotNull(cacheManager);
        return 200;
    }

    long long totalUpdateUsec = stats.awakeTimeUsec + stats.sleepTimeUsec;
    long long awakeTimePerLoopPerGpu = (stats.awakeTimeUsec / stats.updateCycleFinished) / (long long)m_gpus.size();
    long long totalLockCount = stats.lockCount - statsBefore.lockCount;

    printf("TestUpdatePerf Awake usec per gpu: %lld (IsDebug %d, loggingLevel %d)\n",
           awakeTimePerLoopPerGpu, IsDebugBuild(), loggingDebugLevel);
    printf("TestUpdatePerf locksPerUpdate %lld, totalLockCount %lld\n", 
           totalLockCount / numLoops, totalLockCount);
    deleteNotNull(cacheManager);
    return 0;
}

/*****************************************************************************/
int TestCacheManager::TestSimulatedAttachDetach()
{
    DcgmCacheManager   dcm;
    dcgmcm_gpu_info_t  detected[DCGM_MAX_NUM_DEVICES];

    memset(detected, 0, sizeof(detected));

    for (int i = 0; i < 8; i++)
    {
        detected[i].status = DcgmcmGpuStatusOk;
        detected[i].gpuId = i;
        detected[i].lwmlIndex = i;
        snprintf(detected[i].uuid, sizeof(detected[i].uuid), "nighteyes%d", i);
    }

    // Add the 8 GPUs to our cache manager
    dcm.MergeNewlyDetectedGpuList(detected, 8);

    // Make sure they are all found
    for (int i = 0; i < 8; i++)
    {
        if (dcm.GetGpuStatus(i) != DcgmcmGpuStatusOk)
        {
            fprintf(stderr, "TestSimulatedAttachDetach() GPU %d in a bad state after attaching.\n", i);
            return 100;
        }
        
        int lwmlIndex = dcm.GpuIdToLwmlIndex(i);
        if (lwmlIndex != i)
        {
            fprintf(stderr, "TestSimulatedAttachDetach() GPU %d expected lwmlIndex %d, but found %d\n",
                    i, lwmlIndex, i);
            return 100;
        }
    }

    dcm.DetachGpus();
    
    // Make sure they are no longer found
    for (int i = 0; i < 8; i++)
    {
        if (dcm.GetGpuStatus(i) != DcgmcmGpuStatusDetached)
        {
            fprintf(stderr, "TestSimulatedAttachDetach() GPU %d not in detached state after detaching.\n", i);
            return 100;
        }
    }

    dcm.MergeNewlyDetectedGpuList(detected, 4);
    // Make sure the first 4 are found and the last 4 are not
    for (int i = 0; i < 4; i++)
    {
        if (dcm.GetGpuStatus(i) != DcgmcmGpuStatusOk)
        {
            fprintf(stderr, "TestSimulatedAttachDetach() GPU %d in a bad state after attaching.\n", i);
            return 100;
        }
        
        int lwmlIndex = dcm.GpuIdToLwmlIndex(i);
        if (lwmlIndex != i)
        {
            fprintf(stderr, "TestSimulatedAttachDetach() GPU %d expected lwmlIndex %d, but found %d\n",
                    i, lwmlIndex, i);
            return 100;
        }

        if (dcm.GetGpuStatus(i + 4) != DcgmcmGpuStatusDetached)
        {
            fprintf(stderr, "TestSimulatedAttachDetach() GPU %d not in detached state after detaching.\n", i);
            return 100;
        }
    }

    // Add the 6th and 8th GPUs back and then check detection again
    detected[5].lwmlIndex = 4;
    detected[7].lwmlIndex = 5;
    dcm.MergeNewlyDetectedGpuList(detected + 5, 1);
    dcm.MergeNewlyDetectedGpuList(detected + 7, 1);
    for (int i = 0; i < 4; i ++)
    {
        if (dcm.GetGpuStatus(i) != DcgmcmGpuStatusOk)
        {
            fprintf(stderr, "TestSimulatedAttachDetach() GPU %d in a bad state after attaching.\n", i);
            return 100;
        }
    }

    int index = 4;
    for (int i = 4; i < 8; i+=2)
    {
        if (dcm.GetGpuStatus(i+1) != DcgmcmGpuStatusOk)
        {
            fprintf(stderr, "TestSimulatedAttachDetach() GPU %d in a bad state after attaching.\n", i+1);
            return 100;
        }

        int lwmlIndex = dcm.GpuIdToLwmlIndex(i+1);
        if (lwmlIndex != index)
        {
            fprintf(stderr, "TestSimulatedAttachDetach() GPU %d expected lwmlIndex %d, but found %d\n",
                    i+1, index, lwmlIndex);
            return 100;
        }
        index++;

        if (dcm.GetGpuStatus(i) != DcgmcmGpuStatusDetached)
        {
            fprintf(stderr, "TestSimulatedAttachDetach() GPU %d not in detached state after detaching.\n", i);
            return 100;
        }
        
    }

    return 0;
}

int AttachDetach(DcgmCacheManager &dcm, std::string &error)
{
    dcgmReturn_t ret;
    int old_count = 0;
    int count;
    char buf[128];
    timelib64_t start;
    timelib64_t attach_diff = 0;
    timelib64_t detach_diff = 0;
    unsigned int lwmlDeviceCount = 0;

    for (int i = 0; i < 50; i++)
    {
        start = timelib_usecSince1970();
        ret = dcm.AttachGpus();
        if (ret != DCGM_ST_OK)
        {
            snprintf(buf, sizeof(buf), "Error attaching to GPUs: %s.", errorString(ret));
            error = buf;
            return 100;
        }
        attach_diff += timelib_usecSince1970() - start;

        count = dcm.GetGpuCount(1);
        if (old_count == 0)
            old_count = count;
        else if (old_count != count)
        {
            snprintf(buf, sizeof(buf), "Old GPU count was %d, but after attaching again new count is %d.",
                     old_count, count);
            error = buf;
            return 100;
        }

        start = timelib_usecSince1970();
        ret = dcm.DetachGpus();
        if (ret != DCGM_ST_OK)
        {
            snprintf(buf, sizeof(buf), "Error detaching from GPUs: %s.", errorString(ret));
            error = buf;
            return 100;
        }
        detach_diff += timelib_usecSince1970() - start;

        count = dcm.GetGpuCount(1);
        if (count != 0)
        {
            dcm.AttachGpus(); /* Don't leave LWML uninitialized. This will break later tests */
            snprintf(buf, sizeof(buf), "After detaching, found %d GPUs instead of 0.", count);
            error = buf;
            return 100;
        }
    
        lwmlReturn_t lwmlSt = lwmlDeviceGetCount(&lwmlDeviceCount);
        if (lwmlSt != LWML_ERROR_UNINITIALIZED)
        {
            snprintf(buf, sizeof(buf), "Expected LWML_ERROR_UNINITIALIZED after lwmlShutdown(), but found %s",
                     lwmlErrorString(lwmlSt));
            return 100;
        }
    }

    dcm.AttachGpus(); /* Don't leave LWML uninitialized. This will break later tests */
    
    double attach_avg = attach_diff / 50.0;
    double detach_avg = detach_diff / 50.0;
    printf("Average duration in microseconds: AttachGpus() - %.2f DetachGpus() - %.2f\n", attach_avg, detach_avg);

    return 0;
}

/*****************************************************************************/
int TestCacheManager::TestAttachDetachNoWatches(void)
{
    DcgmCacheManager dcm;
    std::string      error;
    
    // Must detach first so that lwmlInit() is called
    dcm.DetachGpus();
    dcgmReturn_t ret = dcm.Init(1, 86400.0);
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "TestAttachDetachNoWatches(): Error attaching to GPUs: %s.", errorString(ret));
        return 100;
    }

    ret = dcm.DetachGpus();
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "TestAttachDetachNoWatches(): Error detaching from GPUs: %s.", errorString(ret));
        return 100;
    }

    int rc = AttachDetach(dcm, error);
    if (rc != 0)
        fprintf(stderr, "TestAttachDetachNoWatches(): %s\n", error.c_str());

    return rc;
}

/*****************************************************************************/
int TestCacheManager::TestAttachDetachWithWatches(void)
{
    DcgmCacheManager dcm;
    std::string      error;
    dcgmReturn_t     ret;
    unsigned int     lwmlDeviceCount = 0;

    // Add watches for each GPU
    unsigned short fieldIds[] =
    {
        DCGM_FI_DEV_XID_ERRORS,
        DCGM_FI_DEV_POWER_VIOLATION,
        DCGM_FI_DEV_THERMAL_VIOLATION,
        DCGM_FI_DEV_SYNC_BOOST_VIOLATION,
        DCGM_FI_DEV_BOARD_LIMIT_VIOLATION,
        DCGM_FI_DEV_LOW_UTIL_VIOLATION,
        DCGM_FI_DEV_RELIABILITY_VIOLATION,
        DCGM_FI_DEV_TOTAL_APP_CLOCKS_VIOLATION,
        DCGM_FI_DEV_TOTAL_BASE_CLOCKS_VIOLATION,
        DCGM_FI_DEV_FB_FREE,
        DCGM_FI_DEV_FB_USED,
        DCGM_FI_DEV_ECC_LWRRENT,
        DCGM_FI_DEV_ECC_PENDING,
        DCGM_FI_DEV_ECC_SBE_VOL_TOTAL,
        DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
        DCGM_FI_DEV_ECC_SBE_AGG_TOTAL,
        DCGM_FI_DEV_ECC_DBE_AGG_TOTAL,
        DCGM_FI_DEV_ECC_SBE_VOL_L1,
        DCGM_FI_DEV_ECC_DBE_VOL_L1,
        DCGM_FI_DEV_ECC_SBE_VOL_L2,
        DCGM_FI_DEV_ECC_DBE_VOL_L2,
        DCGM_FI_DEV_ECC_SBE_VOL_DEV,
        DCGM_FI_DEV_ECC_DBE_VOL_DEV,
        DCGM_FI_DEV_ECC_SBE_VOL_REG,
        DCGM_FI_DEV_ECC_DBE_VOL_REG,
        DCGM_FI_DEV_ECC_SBE_VOL_TEX,
        DCGM_FI_DEV_ECC_DBE_VOL_TEX,
        DCGM_FI_DEV_ECC_SBE_AGG_L1,
        DCGM_FI_DEV_ECC_DBE_AGG_L1,
        DCGM_FI_DEV_ECC_SBE_AGG_L2,
        DCGM_FI_DEV_ECC_DBE_AGG_L2,
        DCGM_FI_DEV_ECC_SBE_AGG_DEV,
        DCGM_FI_DEV_ECC_DBE_AGG_DEV,
        DCGM_FI_DEV_ECC_SBE_AGG_REG,
        DCGM_FI_DEV_ECC_DBE_AGG_REG,
        DCGM_FI_DEV_ECC_SBE_AGG_TEX,
        DCGM_FI_DEV_ECC_DBE_AGG_TEX,
        DCGM_FI_DEV_RETIRED_SBE,
        DCGM_FI_DEV_RETIRED_DBE,
        DCGM_FI_DEV_RETIRED_PENDING,
        DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
        DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_TOTAL,
        DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_TOTAL,
        DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_TOTAL,
        DCGM_FI_DEV_LWLINK_BANDWIDTH_TOTAL,
        DCGM_FI_DEV_GPU_LWLINK_ERRORS,
        DCGM_FI_DEV_ENC_STATS,
        DCGM_FI_DEV_FBC_STATS,
        0
    };

    // Must detach first so we don't think lwml has been initialized
    dcm.DetachGpus();
    ret = dcm.Init(1, 86400.0);
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "TestAttachDetachWithWatches(): Error attaching to GPUs: %s.", errorString(ret));
        return 100;
    }

    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);

    int count = dcm.GetGpuCount(1);
    for (int i = 0; i < count; i++)
    {
        int j = 0;
        while (fieldIds[j] != 0)
        {
            ret = dcm.AddFieldWatch(DCGM_FE_GPU, i, fieldIds[j], 1, 86400.0, 0, watcher, false);
            if (ret != DCGM_ST_OK)
                fprintf(stderr, "TestAttachDeteachWithWatches(): AddFieldWatch error: %s\n", errorString(ret));
            j++;
        }
    }

    ret = dcm.DetachGpus();
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "TestAttachDetachWithWatches(): Error detaching from GPUs: %s.", errorString(ret));
        return 100;
    }
        
    lwmlReturn_t lwmlSt = lwmlDeviceGetCount(&lwmlDeviceCount);
    if (lwmlSt != LWML_ERROR_UNINITIALIZED)
    {
        fprintf(stderr, "Expected LWML_ERROR_UNINITIALIZED after lwmlShutdown(), but found %s",
                 lwmlErrorString(lwmlSt));
        dcm.AttachGpus(); /* Don't leave LWML uninitialized. This will break later tests */
        return 100;
    }

    int rc = AttachDetach(dcm, error);
    if (rc != 0)
        fprintf(stderr, "TestAttachDetachWithWatches(): %s\n", error.c_str());

    return 0;
}

/*****************************************************************************/
int TestCacheManager::TestAddFakeSwitches(void)
{
    unsigned int switchId;

    int i;
    int NtoAdd = 5;
    int Nerrors = 0;

    DcgmCacheManager *cacheManager = createCacheManager(1);
    if (NULL == cacheManager)
        return -1;

    for(i = 0; i < NtoAdd; i++)
    {
        switchId = cacheManager->AddFakeLwSwitch();
        if(switchId == DCGM_ENTITY_ID_BAD)
        {
            fprintf(stderr, "Got error from AddFakeLwSwitch. i=%d\n", i);
            Nerrors += 1;
        }
    }

    deleteNotNull(cacheManager);

    if(Nerrors > 0)
        return 1;
    else
        return 0;
}

/*****************************************************************************/
int TestCacheManager::TestAreAllGpuIdsSameSku()
{
    DcgmCacheManager *cacheManager = createCacheManager(1);
    std::vector<unsigned int> gpuIds;
    int errorCount = 0;

    // Add 4 GPUs that are all the same part
    for (int i = 0; i < 4; i++)
    {
        gpuIds.push_back(cacheManager->AddFakeGpu(0x15FC10DE, 0x119510DE));
    }
            
    int rc = cacheManager->AreAllGpuIdsSameSku(gpuIds);
    if(rc == 0)
    {
        // Should've all been the same
        fprintf(stderr, "4 Fake GPUS should all be the same SKU, but somehow they're not\n");
        errorCount++;
    }
            
    // Add a GPU with a different subsystem
    unsigned int differentSubsystemId = cacheManager->AddFakeGpu(0x15FC10DE, 0x119A10DE);
    std::vector<unsigned int> secondGroup(gpuIds);
    secondGroup.push_back(differentSubsystemId);
    rc = cacheManager->AreAllGpuIdsSameSku(secondGroup);
    if (rc == 1)
    {       
        // Shouldn't have been considered the same
        fprintf(stderr, "Different subsystem failed to register as a different GPU!\n");
        errorCount++;
    }
            
    // Add a GPU with a different pci device id
    unsigned int differentPciId = cacheManager->AddFakeGpu(0x15F810DE, 0x119510DE);
    std::vector<unsigned int> thirdGroup(gpuIds);
    thirdGroup.push_back(differentPciId);
    rc = cacheManager->AreAllGpuIdsSameSku(thirdGroup);
    if (rc == 1)
    {
        // Shouldn't have been considered the same
        fprintf(stderr, "Different PCI device ID failed to register as a different GPU!\n");
        errorCount++;
    }
            
    deleteNotNull(cacheManager);
            
    if (errorCount > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

/*****************************************************************************/
void TestCacheManager::CompleteTest(std::string testName, int testReturn, int &Nfailed)
{
    if (testReturn)
    {
        Nfailed++;
        std::cerr << "TestCacheManager::" << testName << " FAILED with " << testReturn << std::endl;

        // fatal test failure
        if (testReturn < 0)
            throw std::runtime_error("fatal test failure");
    }
    else
    {
        std::cout << "TestCacheManager::" << testName << " PASSED" << std::endl;
    }
}

/*****************************************************************************/
int TestCacheManager::Run()
{
    int Nfailed = 0;

    try
    {
        CompleteTest("TestAddFakeSwitches", TestAddFakeSwitches(), Nfailed);
        CompleteTest("TestUpdatePerf", TestUpdatePerf(), Nfailed);
        CompleteTest("TestLockstepModeAwakeTime", TestLockstepModeAwakeTime(), Nfailed);
        CompleteTest("TestTimedModeAwakeTime", TestTimedModeAwakeTime(), Nfailed);
        CompleteTest("TestWatchesVisited", TestWatchesVisited(), Nfailed);
        CompleteTest("TestRecording", TestRecording(), Nfailed);
        CompleteTest("TestGpuFieldBytesUsed", TestGpuFieldBytesUsed(), Nfailed);
        CompleteTest("TestInjection", TestInjection(), Nfailed);
        CompleteTest("TestManageVgpuList", TestManageVgpuList(), Nfailed);
        CompleteTest("TestSummary", TestSummary(), Nfailed);
        CompleteTest("TestEmpty", TestEmpty(), Nfailed);
        CompleteTest("TestRecordTiming", TestRecordTiming(), Nfailed);
        CompleteTest("TestTimeBasedQuota", TestTimeBasedQuota(), Nfailed);
        CompleteTest("TestCountBasedQuota", TestCountBasedQuota(), Nfailed);
        CompleteTest("TestRecordingGlobal", TestRecordingGlobal(), Nfailed);
        CompleteTest("TestWatchedFieldsIteration", TestWatchedFieldsIteration(), Nfailed);
        CompleteTest("TestWatchedFieldsIterationGpu", TestWatchedFieldsIterationGpu(), Nfailed);
        CompleteTest("TestWatchedFieldsIterationGlobal", TestWatchedFieldsIterationGlobal(), Nfailed);
        CompleteTest("TestFieldValueColwersion", TestFieldValueColwersion(), Nfailed);
        CompleteTest("TestColwertVectorToBitmask", TestColwertVectorToBitmask(), Nfailed);
        CompleteTest("TestAffinityBitmasksMatch", TestAffinityBitmasksMatch(), Nfailed);
        CompleteTest("TestCreateGroupsFromCpuAffinities", TestCreateGroupsFromCpuAffinities(), Nfailed);
        CompleteTest("TestPopulatePotentialCpuMatches", TestPopulatePotentialCpuMatches(), Nfailed);
        CompleteTest("TestCombineAffinityGroups", TestCombineAffinityGroups(), Nfailed);
        CompleteTest("TestSetIOConnectionLevels", TestSetIOConnectionLevels(), Nfailed);
        CompleteTest("TestMatchByIO", TestMatchByIO(), Nfailed);
        CompleteTest("TestSimulatedAttachDetach", TestSimulatedAttachDetach(), Nfailed);
        CompleteTest("TestAttachDetachNoWatches", TestAttachDetachNoWatches(), Nfailed);
        CompleteTest("TestAttachDetachWithWatches", TestAttachDetachWithWatches(), Nfailed);
        CompleteTest("TestAreAllGpuIdsSameSku", TestAreAllGpuIdsSameSku(), Nfailed);
    }
    // fatal test return olwrred
    catch(const std::runtime_error &e)
    {
        return -1;
    }

    if(Nfailed > 0)
    {
        fprintf(stderr, "%d tests FAILED\n", Nfailed);
        return 1;
    }

    printf("All tests passed\n");

    return 0;
}

/*****************************************************************************/
