#include <stdexcept>

#include "Whitelist.h"
#include "PluginStrings.h"
#include "ParsingUtility.h"
#include <dcgm_fields.h>

/*****************************************************************************/
Whitelist::Whitelist()
{
    // initialize map of maps
    FillMap();
}

/*****************************************************************************/
Whitelist::~Whitelist()
{
    std::map<std::string, std::map<std::string, TestParameters *> >::iterator outerIt;
    std::map<std::string, TestParameters *>::iterator innerIter;

    for(outerIt = m_featureDb.begin(); outerIt != m_featureDb.end(); outerIt++)
    {
        for(innerIter = (*outerIt).second.begin(); innerIter != (*outerIt).second.end(); innerIter++)
        {
            TestParameters *tp;

            tp = innerIter->second;

            if(tp)
            {
                delete(tp);
                tp = 0;
            }
        }

        outerIt->second.clear();
    }

    m_featureDb.clear();
}

/*****************************************************************************/
bool Whitelist::isWhitelisted(std::string deviceId)
{
    std::map<std::string, std::map<std::string, TestParameters *> >::const_iterator outerIt = m_featureDb.find(deviceId);
    std::set<std::string>::const_iterator globalChangesIt = m_globalChanges.find(deviceId);

    if (outerIt == m_featureDb.end() && globalChangesIt == m_globalChanges.end())
    {
        PRINT_INFO("%s", "DeviceId %s is NOT whitelisted", deviceId.c_str());
        return false;
    }
    else
    {
        PRINT_INFO("%s", "DeviceId %s is whitelisted", deviceId.c_str());
        return true;
    }
}

/*****************************************************************************/
void Whitelist::getDefaultsByDeviceId(const std::string &testName, const std::string &deviceId, TestParameters * tp)
{
    int st;
    TestParameters *testDeviceTp = m_featureDb[deviceId][testName];
    bool requiresGlobalChanges = m_globalChanges.find(deviceId) != m_globalChanges.end();

    if (!testDeviceTp && !requiresGlobalChanges)
    {
        PRINT_INFO("%s %s", "No whitelist overrides found for deviceId %s test %s",
                   deviceId.c_str(), testName.c_str());
        return; /* No overrides configured for this subtest */
    }
    
    if (testDeviceTp)
    {
        st = tp->OverrideFrom(testDeviceTp);
        if (st)
        {
            throw std::runtime_error("Unable to override bootstrapped test values with device ones.");
        }
    }

    if (requiresGlobalChanges)
    {
        UpdateGlobalsForDeviceId(deviceId);
    }
}

/*****************************************************************************/
void Whitelist::postProcessWhitelist(std::vector<Gpu *> & gpus)
{
    std::vector<Gpu *>::iterator gpuIter;
    unsigned int minMemClockSeen = DCGM_INT32_BLANK;
    unsigned int memClock;

    static const char* TESLA_V100_ID = "1db1";
    const unsigned int TESLA_V100_BASE_SKU_MEM_CLOCK = 877;

    /* Record the minimum memory clock seen for Tesla V100 (1db1).
       It's possible that we have a mix of V100 recovery SKUs and healthy SKUs.
       We'll just use the threshold for the recovery SKU since the healthy SKUs
       will easily exceed that threshold
     */
    for (gpuIter = gpus.begin(); gpuIter != gpus.end(); ++gpuIter)
    {
        if ((*gpuIter)->getDevicePciDeviceId() != TESLA_V100_ID)
        {
            continue;
        }
        
        memClock = (*gpuIter)->getMaxMemoryClock();
        if (DCGM_INT32_IS_BLANK(memClock))
        {
            continue;
        }
        
        if (DCGM_INT32_IS_BLANK(minMemClockSeen) || memClock < minMemClockSeen)
        {
            minMemClockSeen = memClock;
        }
    }

    if (DCGM_INT32_IS_BLANK(minMemClockSeen))
    {
        return; /* No 1db1 SKUs. Nothing to do */
    }
    if (minMemClockSeen == TESLA_V100_BASE_SKU_MEM_CLOCK)
    {
        return; /* All SKUs are the base SKU that has a 877 memory clock */
    }
    
    double ratio = (double)minMemClockSeen / (double)TESLA_V100_BASE_SKU_MEM_CLOCK;
    
    double existingValue = m_featureDb[TESLA_V100_ID][MEMBW_PLUGIN_WL_NAME]->GetDouble(MEMBW_STR_MINIMUM_BANDWIDTH);
    double discountedValue = ratio * existingValue;
    m_featureDb[TESLA_V100_ID][MEMBW_PLUGIN_WL_NAME]->SetDouble(MEMBW_STR_MINIMUM_BANDWIDTH, discountedValue);
    PRINT_DEBUG("%f %f %u", "Updated whitelist for minimium_bandwidth from %f -> %f due to memory clock %u.",
                existingValue, discountedValue, minMemClockSeen);
}

/*****************************************************************************/
void Whitelist::FillMap()
{
    std::string id;
    TestParameters *tp;

	// Tesla K8
    id = "1194";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 75.0, 65.0, 115.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 92.0, 40.0, 100.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 330.0, 30.0, 500.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 330.0, 30.0, 500.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 92.0, 40.0, 100.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla K80 ("Stella Duo" Gemini)
    id = "102d";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 148.0, 70.0, 149.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 85.0, 40.0, 88.0);
    tp->AddString(CP_STR_USE_DGEMM, "False");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 950.0, 30.0, 1200.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 950.0, 30.0, 1200.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 85.0, 40.0, 88.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    // Add device id to m_globalChanges since this device requires a throttle mask
    m_globalChanges.insert(id);
    tp = 0;

    // GP100 SKU 201 - (based on http://teams.lwpu.com/sites/gpu/ae/Tesla/Lists/Pascal%20Board%20Lineup/Standard%20View1.aspx)
    id = "15fa";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 249.0, 70.0, 250.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 85.0, 40.0, 88.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 3000.0, 30.0, 8000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 3000.0, 30.0, 8000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 85.0, 40.0, 88.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // GP100 SKU 200 - (based on http://teams.lwpu.com/sites/gpu/ae/Tesla/Lists/Pascal%20Board%20Lineup/Standard%20View1.aspx)
    id = "15fb";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 249.0, 70.0, 250.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 85.0, 40.0, 88.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 3000.0, 30.0, 8000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 3000.0, 30.0, 8000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 85.0, 40.0, 88.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // GP100 SKU 201 - (based on http://teams.lwpu.com/sites/gpu/ae/Tesla/Lists/Pascal%20Board%20Lineup/Standard%20View1.aspx)
    // DGX Station GP100
    id = "15fc";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 700.0, 1.0, 2048.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 299.0, 150.0, 300.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 85.0, 40.0, 88.0);
    tp->AddDouble(CP_STR_OPS_PER_REQUEUE, 16.0, 1.0, 32.0);
    //tp->AddDouble(CP_STR_LWDA_STREAMS_PER_GPU, 8.0, 1.0, 48.0); //Can't achieve target without higher conlwrrency
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 3000.0, 30.0, 8000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 3000.0, 30.0, 8000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 85.0, 40.0, 88.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // COPIED from -- GP100 SKU 203 - DB2 HBM2
    // Tesla P100-PCIE-16GB
    // Serial Number : PH400-C01-P0203
    id = "15f8";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 400.0, 1.0, 2048.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 249.0, 70.0, 250.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 82.0, 40.0, 85.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 3000.0, 30.0, 8000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 3000.0, 30.0, 8000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 85.0, 40.0, 88.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla P100-PCIE-12GB
    // Serial Number : PH400 SKU 202
    id = "15f7";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 400.0, 1.0, 2048.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 249.0, 70.0, 250.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 82.0, 40.0, 85.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 3000.0, 30.0, 8000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 3000.0, 30.0, 8000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 85.0, 40.0, 88.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;


    // GP100 Bringup - eris-l64-test12*
    id = "15ff";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 249.0, 70.0, 250.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 85.0, 40.0, 88.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 3000.0, 30.0, 8000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 3000.0, 30.0, 8000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 85.0, 40.0, 88.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla P100-SXM2-16GB
    id = "15f9";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 299.0, 70.0, 300.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 85.0, 40.0, 88.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 3000.0, 30.0, 8000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 3000.0, 30.0, 8000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 85.0, 40.0, 88.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla Stella Solo - Internal only
    id = "102f";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 234.0, 150.0, 235.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 88.0, 40.0, 93.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 900.0, 200.0, 1700.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 1000.0, 100.0, 10000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 88.0, 40.0, 93.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla K40d ("Stella" DFF)
    id = "102e";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 234.0, 120.0, 235.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 200.0, 100.0, 210.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 200.0, 100.0, 210.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;


    // Tesla K40m
    id = "1023";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 200.0, 150.0, 235.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    tp->AddDouble(CP_STR_OPS_PER_REQUEUE, 4.0, 1.0, 10.0);
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 600.0, 1.0, 1024.0); //Change max if CP_MAX_DIMENSION changes in constantpower.h
    tp->AddDouble(CP_STR_MAX_GRAPHICS_CLOCK, 811.0, 666.0, 876.0);
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 1300.0, 200.0, 1500.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 200.0, 150.0, 235.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla K40c
    id = "1024";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 134.0, 100.0, 235.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 1300.0, 200.0, 1500.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 1300.0, 200.0, 1500.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;


    // Tesla K40t ("Atlas" TTP) - values copied from Tesla K40c
    id = "102a";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 234.0, 120.0, 235.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 900.0, 200.0, 1500.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 900.0, 200.0, 1500.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla K40s - values copied from Tesla K40c
    id = "1029";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 234.0, 150.0, 235.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 550.0, 200.0, 630.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 550.0, 200.0, 630.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla K20X - values copied from Tesla K20xm
    id = "101e";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 234.0, 150.0, 235.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 900.0, 200.0, 1400.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 900.0, 200.0, 1400.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla K20Xm
    id = "1021";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 234.0, 150.0, 235.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 900.0, 200.0, 1400.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 900.0, 200.0, 1400.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla K20c - values copied from Tesla K20xm (except power)
    id = "1022";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 224.0, 150.0, 225.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 900.0, 200.0, 1400.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 900.0, 200.0, 1400.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;


    // Tesla K20m - values copied from Tesla K20xm
    id = "1028";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 234.0, 150.0, 235.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 900.0, 200.0, 1400.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 900.0, 200.0, 1400.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;


   // Tesla K20X
    id = "1020";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 234.0, 150.0, 235.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 900.0, 200.0, 1400.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 900.0, 200.0, 1400.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 90.0, 40.0, 95.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;


    // Tesla K10
    id = "118f";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 116.0, 80.0, 117.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 88.0, 40.0, 97.0);
    tp->AddString(CP_STR_USE_DGEMM, "False");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 700.0, 100.0, 900.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 700.0, 100.0, 900.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 88.0, 40.0, 97.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;


    // Lwdqro M6000 //Not supported officially for LWVS
#if 1
    id = "17f0";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 249.0, 80.0, 250.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 92.0, 80.0, 97.0);
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
    tp->AddString(CP_STR_USE_DGEMM, "False");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 1000.0, 100.0, 900000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 4000.0, 100.0, 900000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 92.0, 80.0, 97.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMBW_STR_IS_ALLOWED, "True");
    tp->AddDouble(MEMBW_STR_MINIMUM_BANDWIDTH, 200000.0, 1.0, 1000000.0);
    m_featureDb[id][MEMBW_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = 0;
#endif

    // Lwdqro M5000
    id = "13f0";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 149.0, 10.0, 150.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 100.0, 80.0, 105.0);
    tp->AddString(CP_STR_USE_DGEMM, "False");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 1000.0, 100.0, 900000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 4000.0, 100.0, 900000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 100.0, 80.0, 105.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // VdChip GTX TITAN X
    id = "17c2";
    tp = new TestParameters();
    tp->AddDouble(BG_STR_MAX_MEMORY_CLOCK, 3301.0, 0.0, 3600.0);
    tp->AddDouble(BG_STR_MAX_GRAPHICS_CLOCK, 0.0, 0.0, 1500.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 249.0, 80.0, 250.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 92.0, 80.0, 97.0);
    tp->AddString(CP_STR_USE_DGEMM, "False");
    tp->AddDouble(CP_STR_LWDA_STREAMS_PER_GPU, 8.0, 1.0, 48.0); //Can't achieve target without higher conlwrrency
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 1000.0, 100.0, 900000.0);
    tp->AddDouble(CPERF_STR_MAX_MEMORY_CLOCK, 3301.0, 0.0, 3600.0);
    tp->AddDouble(CPERF_STR_MAX_GRAPHICS_CLOCK, 0.0, 0.0, 1500.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 2000.0, 100.0, 9000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 92.0, 80.0, 97.0);
    tp->AddDouble(SMPERF_STR_MAX_MEMORY_CLOCK, 3301.0, 0.0, 3600.0);
    tp->AddDouble(SMPERF_STR_MAX_GRAPHICS_CLOCK, 0.0, 0.0, 1500.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla M60
    id = "13f2";
    tp = new TestParameters();
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 161.0, 80.0, 162.0);
    tp->AddDouble(CP_STR_OPS_PER_REQUEUE, 4.0, 1.0, 32.0);
    tp->AddDouble(CP_STR_READJUST_INTERVAL, 3.0, 1.0, 10.0);
    tp->AddDouble(CP_STR_TARGET_MOVAVG_MIN_RATIO, 0.9, 0.5, 1.0);
    tp->AddDouble(CP_STR_TARGET_MOVAVG_MAX_RATIO, 1.1, 1.0, 2.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 88.0, 80.0, 91.0);
    tp->AddString(CP_STR_USE_DGEMM, "False");
    //tp->AddDouble(CP_STR_LWDA_STREAMS_PER_GPU, 8.0, 1.0, 48.0); //Can't achieve target without higher conlwrrency
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 2500.0, 100.0, 900000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 2500.0, 100.0, 9000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 88.0, 80.0, 91.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla M6 (Copied from Tesla M60)
    id = "13f3";
    tp = new TestParameters();
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 161.0, 80.0, 162.0);
    tp->AddDouble(CP_STR_MAX_GRAPHICS_CLOCK, 1130.0, 0.0, 1200.0); /* 1177 max so far */
    tp->AddDouble(CP_STR_OPS_PER_REQUEUE, 4.0, 1.0, 32.0);
    tp->AddDouble(CP_STR_READJUST_INTERVAL, 3.0, 1.0, 10.0);
    tp->AddDouble(CP_STR_TARGET_MOVAVG_MIN_RATIO, 0.9, 0.5, 1.0);
    tp->AddDouble(CP_STR_TARGET_MOVAVG_MAX_RATIO, 1.1, 1.0, 2.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 88.0, 80.0, 91.0);
    tp->AddString(CP_STR_USE_DGEMM, "False");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 2500.0, 100.0, 900000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 2500.0, 100.0, 9000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 88.0, 80.0, 91.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla M40
    id = "17fd";
    tp = new TestParameters();
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 249.0, 100.0, 250.0);
    tp->AddDouble(CP_STR_MAX_GRAPHICS_CLOCK, 1113.0, 0.0, 1200.0);
    tp->AddDouble(CP_STR_OPS_PER_REQUEUE, 4.0, 1.0, 32.0);
    tp->AddDouble(CP_STR_READJUST_INTERVAL, 3.0, 1.0, 10.0);
    tp->AddDouble(CP_STR_TARGET_MOVAVG_MIN_RATIO, 0.9, 0.5, 1.0);
    tp->AddDouble(CP_STR_TARGET_MOVAVG_MAX_RATIO, 1.1, 1.0, 2.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 89.0, 80.0, 92.0);
    tp->AddString(CP_STR_USE_DGEMM, "False");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 2500.0, 100.0, 900000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 2500.0, 100.0, 9000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 89.0, 80.0, 92.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla M4
    id = "1431";
    tp = new TestParameters();
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 74.0, 20.0, 75.0);
    tp->AddDouble(CP_STR_MAX_GRAPHICS_CLOCK, 1113.0, 0.0, 1200.0);
    tp->AddDouble(CP_STR_OPS_PER_REQUEUE, 4.0, 1.0, 32.0);
    tp->AddDouble(CP_STR_READJUST_INTERVAL, 3.0, 1.0, 10.0);
    tp->AddDouble(CP_STR_TARGET_MOVAVG_MIN_RATIO, 0.9, 0.5, 1.0);
    tp->AddDouble(CP_STR_TARGET_MOVAVG_MAX_RATIO, 1.1, 1.0, 2.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 93.0, 80.0, 96.0);
    tp->AddString(CP_STR_USE_DGEMM, "False");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 2500.0, 100.0, 900000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 2500.0, 100.0, 9000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 93.0, 80.0, 96.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla M10
    // Mostly copied from Tesla M40
    id = "13bd";
    tp = new TestParameters();
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_TARGET_POWER, 53.0, 26.5, 53.0);
    tp->AddDouble(CP_STR_MAX_GRAPHICS_CLOCK, 1032.0, 0.0, 1202.0);
    tp->AddDouble(CP_STR_OPS_PER_REQUEUE, 4.0, 1.0, 32.0);
    tp->AddDouble(CP_STR_READJUST_INTERVAL, 3.0, 1.0, 10.0);
    tp->AddDouble(CP_STR_TARGET_MOVAVG_MIN_RATIO, 0.9, 0.5, 1.0);
    tp->AddDouble(CP_STR_TARGET_MOVAVG_MAX_RATIO, 1.1, 1.0, 2.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 91.0, 80.0, 96.0);
    tp->AddString(CP_STR_USE_DGEMM, "False");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 2500.0, 100.0, 900000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 2500.0, 100.0, 9000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 91.0, 80.0, 96.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla GV100 DGX1-V
    // Serial Number: PG503-A00-P0447
    id = "1dbd";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_MAX_GRAPHICS_CLOCK, 1140.0, 0.0, 1455.0);
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 299.0, 50.0, 300.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 82.0, 40.0, 85.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 3000.0, 30.0, 10000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 3000.0, 30.0, 10000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 82.0, 40.0, 85.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMBW_STR_IS_ALLOWED, "True");
    tp->AddDouble(MEMBW_STR_MINIMUM_BANDWIDTH, 607500.0, 1.0, 1000000.0);
    m_featureDb[id][MEMBW_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    tp->AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

	// Tesla GV100 DGX Station - Part number 692-2G500-0201-300
    id = "1db2";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 299.0, 100.0, 300.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 90.0, 40.0, 93.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 3350.0, 30.0, 10000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 3000.0, 30.0, 20000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 90.0, 40.0, 93.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMBW_STR_IS_ALLOWED, "True");
    tp->AddDouble(MEMBW_STR_MINIMUM_BANDWIDTH, 607500.0, 1.0, 1000000.0);
    m_featureDb[id][MEMBW_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    tp->AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

	// Tesla GV100 SXM2-32GB SKU 895
    id = "1db1";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 299.0, 100.0, 300.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 102.0, 40.0, 105.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 3350.0, 30.0, 10000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 3500.0, 30.0, 20000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 102.0, 40.0, 105.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMBW_STR_IS_ALLOWED, "True");
    tp->AddDouble(MEMBW_STR_MINIMUM_BANDWIDTH, 607500.0, 1.0, 1000000.0);
    m_featureDb[id][MEMBW_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    tp->AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

	// Tesla GV100 SXM2-16GB SKU 890
	id = "1db0";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
	tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
	tp->AddDouble(CP_STR_TARGET_POWER, 299.0, 100.0, 300.0);
	tp->AddDouble(CP_STR_TEMPERATURE_MAX, 82.0, 40.0, 85.0);
	tp->AddString(CP_STR_USE_DGEMM, "True");
	m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
	tp->AddString(CPERF_STR_USE_DGEMM, "True");
	tp->AddDouble(CPERF_STR_TARGET_PERF, 3000.0, 30.0, 10000.0);
	m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
	tp->AddDouble(SMPERF_STR_TARGET_PERF, 3500.0, 30.0, 20000.0);
	tp->AddString(SMPERF_STR_USE_DGEMM, "True");
	tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 82.0, 40.0, 85.0);
	m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
	m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMBW_STR_IS_ALLOWED, "True");
    tp->AddDouble(MEMBW_STR_MINIMUM_BANDWIDTH, 607500.0, 1.0, 1000000.0);
    m_featureDb[id][MEMBW_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    tp->AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
	tp = 0;

	// Tesla GV100 PCIE-16GB SKU 893
	id = "1db4";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
	tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
	tp->AddDouble(CP_STR_TARGET_POWER, 120.0, 100.0, 250.0);
	tp->AddDouble(CP_STR_TEMPERATURE_MAX, 87.0, 40.0, 90.0);
	tp->AddString(CP_STR_USE_DGEMM, "True");
	m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
	tp->AddString(CPERF_STR_USE_DGEMM, "True");
	tp->AddDouble(CPERF_STR_TARGET_PERF, 3000.0, 30.0, 10000.0);
	m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
	tp->AddDouble(SMPERF_STR_TARGET_PERF, 3500.0, 30.0, 20000.0);
	tp->AddString(SMPERF_STR_USE_DGEMM, "True");
	tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 87.0, 40.0, 90.0);
	m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
	m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMBW_STR_IS_ALLOWED, "True");
    tp->AddDouble(MEMBW_STR_MINIMUM_BANDWIDTH, 607500.0, 1.0, 1000000.0);
    m_featureDb[id][MEMBW_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    tp->AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
	tp = 0;

    // Tesla V100-SXM2-32GB-LS - PG503 SKU 203
	id = "1db5";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
	tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
	tp->AddDouble(CP_STR_TARGET_POWER, 249.0, 100.0, 250.0);
	tp->AddDouble(CP_STR_TEMPERATURE_MAX, 87.0, 40.0, 90.0);
	tp->AddString(CP_STR_USE_DGEMM, "True");
	m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
	tp->AddString(CPERF_STR_USE_DGEMM, "True");
	tp->AddDouble(CPERF_STR_TARGET_PERF, 3000.0, 30.0, 10000.0);
	m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
	tp->AddDouble(SMPERF_STR_TARGET_PERF, 3500.0, 30.0, 20000.0);
	tp->AddString(SMPERF_STR_USE_DGEMM, "True");
	tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 87.0, 40.0, 90.0);
	m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
	m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMBW_STR_IS_ALLOWED, "True");
    tp->AddDouble(MEMBW_STR_MINIMUM_BANDWIDTH, 607500.0, 1.0, 1000000.0);
    m_featureDb[id][MEMBW_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    tp->AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
	tp = 0;

    // Tesla V100 PCI-E-32GB (Passive) - PG500 SKU 202
    id = "1db6";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 249.0, 100.0, 250.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 87.0, 40.0, 90.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 3000.0, 30.0, 10000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 3500.0, 30.0, 20000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 87.0, 40.0, 90.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMBW_STR_IS_ALLOWED, "True");
    tp->AddDouble(MEMBW_STR_MINIMUM_BANDWIDTH, 607500.0, 1.0, 1000000.0);
    m_featureDb[id][MEMBW_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    tp->AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

	// Tesla P40
	id = "1b38";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
	tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
	tp->AddDouble(CP_STR_TARGET_POWER, 249.0, 125.0, 250.0);
	tp->AddDouble(CP_STR_TEMPERATURE_MAX, 92.0, 40.0, 95.0);
	tp->AddString(CP_STR_USE_DGEMM, "False");
	m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
	tp->AddString(CPERF_STR_USE_DGEMM, "False");
	tp->AddDouble(CPERF_STR_TARGET_PERF, 3500.0, 30.0, 10000.0);
	m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
	tp->AddDouble(SMPERF_STR_TARGET_PERF, 6000.0, 30.0, 20000.0);
	tp->AddString(SMPERF_STR_USE_DGEMM, "False");
	tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 92.0, 40.0, 95.0);
	m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla P10
    // "Same as Tesla P40 (PG610 SKU200 GP102-895) but with a lower power target (150W vs 250W))"
    // https://confluence.lwpu.com/display/CSWPM/Pascal+Board+Schedule
	id = "1b39";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
	tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
	tp->AddDouble(CP_STR_TARGET_POWER, 149.0, 75.0, 150.0);
	tp->AddDouble(CP_STR_TEMPERATURE_MAX, 92.0, 40.0, 95.0);
	tp->AddString(CP_STR_USE_DGEMM, "False");
	m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
	tp->AddString(CPERF_STR_USE_DGEMM, "False");
	tp->AddDouble(CPERF_STR_TARGET_PERF, 2100.0, 30.0, 10000.0); // P40 perf 3500 adjusted for 250->150 TDP
	m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
	tp->AddDouble(SMPERF_STR_TARGET_PERF, 3600.0, 30.0, 20000.0); // P40 perf 6000 adjusted for 250->150 TDP
	tp->AddString(SMPERF_STR_USE_DGEMM, "False");
	tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 92.0, 40.0, 95.0);
	m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
	tp = 0;

    // Tesla P6
	id = "1bb4";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
	tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
	tp->AddDouble(CP_STR_TARGET_POWER, 89.0, 70.0, 90.0);
	tp->AddDouble(CP_STR_TEMPERATURE_MAX, 91.0, 40.0, 94.0);
	tp->AddString(CP_STR_USE_DGEMM, "False");
	m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
	tp->AddString(CPERF_STR_USE_DGEMM, "False");
	tp->AddDouble(CPERF_STR_TARGET_PERF, 3250.0, 30.0, 10000.0);
	m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
	tp->AddDouble(SMPERF_STR_TARGET_PERF, 3750.0, 30.0, 20000.0);
	tp->AddString(SMPERF_STR_USE_DGEMM, "False");
	tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 91.0, 40.0, 94.0);
	m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMBW_STR_IS_ALLOWED, "True");
    tp->AddDouble(MEMBW_STR_MINIMUM_BANDWIDTH, 114263.0, 1.0, 1000000.0); /* Got 152.35 consistently in testing. taking 25% off this */
    m_featureDb[id][MEMBW_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
	tp = 0;

	// Tesla P4
	id = "1bb3";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
	tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
	tp->AddDouble(CP_STR_TARGET_POWER, 50.0, 45.0, 50.0);
	tp->AddDouble(CP_STR_TEMPERATURE_MAX, 91.0, 40.0, 94.0);
	tp->AddString(CP_STR_USE_DGEMM, "False");
	m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
	tp->AddString(CPERF_STR_USE_DGEMM, "False");
	tp->AddDouble(CPERF_STR_TARGET_PERF, 3250.0, 30.0, 10000.0);
	m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
	tp->AddDouble(SMPERF_STR_TARGET_PERF, 3750.0, 30.0, 20000.0);
	tp->AddString(SMPERF_STR_USE_DGEMM, "False");
	tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 91.0, 40.0, 94.0);
	m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMBW_STR_IS_ALLOWED, "True");
    // 1273750 = .75 * 165000. Was higher previously, see http://lwbugs/200480825 for details.
    // Lowered to 123750 as part of standardizing at 75%
    tp->AddDouble(MEMBW_STR_MINIMUM_BANDWIDTH, 123750.0, 1.0, 1000000.0);
    m_featureDb[id][MEMBW_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
	tp = 0;

	// Tesla V100-HS
	id = "1db3";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
	tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
	tp->AddDouble(CP_STR_TARGET_POWER, 149.0, 75.0, 150.0);
	tp->AddDouble(CP_STR_TEMPERATURE_MAX, 87.0, 40.0, 90.0);
	tp->AddString(CP_STR_USE_DGEMM, "True");
	m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
	tp->AddString(CPERF_STR_USE_DGEMM, "True");
	tp->AddDouble(CPERF_STR_TARGET_PERF, 2250.0, 30.0, 10000.0);
	m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    // V100 has a DGEMM performance of 7000 and max power level of 250, so
    // we'll set this to 7000 * .75 (to back off) * 3/5 (relative power ratio)
	tp->AddDouble(SMPERF_STR_TARGET_PERF, 3150.0, 30.0, 20000.0);
	tp->AddString(SMPERF_STR_USE_DGEMM, "True");
	tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 87.0, 40.0, 90.0);
	m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    tp->AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
	tp = 0;

	// Tesla V100 32GB
	id = "1db5";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
	tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
	tp->AddDouble(CP_STR_TARGET_POWER, 299.0, 100.0, 300.0);
	tp->AddDouble(CP_STR_TEMPERATURE_MAX, 87.0, 40.0, 90.0);
	tp->AddString(CP_STR_USE_DGEMM, "True");
	m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
	tp->AddString(CPERF_STR_USE_DGEMM, "True");
	tp->AddDouble(CPERF_STR_TARGET_PERF, 4250.0, 30.0, 10000.0);
	m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    // V100 has a DGEMM performance of 7000 and max power level of 250, close enough
    // to 300 given Max Q, so we'll set this to 5250 = 7000 * .75 to backs off to 75%.
	tp->AddDouble(SMPERF_STR_TARGET_PERF, 5250.0, 30.0, 20000.0);
	tp->AddString(SMPERF_STR_USE_DGEMM, "True");
	tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 87.0, 40.0, 90.0);
	m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    tp->AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
	tp = 0;

	// Tesla V100 DGX-Station 32GB
	id = "1db7";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
	tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
	tp->AddDouble(CP_STR_TARGET_POWER, 299.0, 100.0, 300.0);
	tp->AddDouble(CP_STR_TEMPERATURE_MAX, 87.0, 40.0, 90.0);
	tp->AddString(CP_STR_USE_DGEMM, "True");
	m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
	tp->AddString(CPERF_STR_USE_DGEMM, "True");
	tp->AddDouble(CPERF_STR_TARGET_PERF, 4250.0, 30.0, 10000.0);
	m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    // V100 has a DGEMM performance of 7000 and max power level of 250, close enough
    // to 300 given Max Q, so we'll set this to 5250 = 7000 * .75 to backs off to 75%.
	tp->AddDouble(SMPERF_STR_TARGET_PERF, 5250.0, 30.0, 20000.0);
	tp->AddString(SMPERF_STR_USE_DGEMM, "True");
	tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 87.0, 40.0, 90.0);
	m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
	                     3.0, 0.0, 3.0);
	tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
	                     16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    tp->AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMBW_STR_IS_ALLOWED, "True");
    tp->AddDouble(MEMBW_STR_MINIMUM_BANDWIDTH, 607500.0, 1.0, 1000000.0);
    m_featureDb[id][MEMBW_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
	tp = 0;

    // Tesla V100 DGX-Station 32GB
    id = "1db8";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 349.0, 100.0, 350.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 87.0, 40.0, 90.0);
    tp->AddString(CP_STR_USE_DGEMM, "True");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 4250.0, 30.0, 10000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    // V100 has a DGEMM performance of 7000 and max power level of 250, close enough
    // to 350 given Max Q, so we'll set this to 5250 = 7000 * .75 to backs off to 75%.
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 5250.0, 30.0, 20000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 87.0, 40.0, 90.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
                         3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
                         16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    tp->AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = 0;

    // Tesla T4 - TU104_PG183_SKU200
    id = "1eb8";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 550.0, 1.0, 2048.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 65.0, 60.0, 70.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 102.0, 40.0, 105.0);
    tp->AddString(CP_STR_USE_DGEMM, "False");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 3150.0, 30.0, 10000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    // T4 has an SGEMM performance of 5500 * .75 backs it off to 4125
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 4125.0, 30.0, 20000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 102.0, 40.0, 105.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN, 3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH, 16.0, 1.0, 16.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN, 3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH, 16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMBW_STR_IS_ALLOWED, "True");
    // Spec sheet says 672000, but experimentally we get about 244000 MB/s so we'll go with .75 of that
    // Update value when bug 200480825 is fully root caused and resolved
    tp->AddDouble(MEMBW_STR_MINIMUM_BANDWIDTH, 183000.0, 1.0, 1000000.0);
    m_featureDb[id][MEMBW_PLUGIN_WL_NAME] = tp;
    // Add device id to m_globalChanges since this device requires a throttle mask
    m_globalChanges.insert(id);
    tp = 0;

    // Tesla T40 - TU102-895
    id = "1e38";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 1024.0, 1.0, 2048.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 249.0, 200.0, 250.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 93.0, 40.0, 96.0);
    tp->AddString(CP_STR_USE_DGEMM, "False");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 5250.0, 30.0, 10000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    // T40 has an SGEMM performance of 14746; multiply by .85 to get 12534
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 12534.0, 30.0, 20000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 93.0, 40.0, 96.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN, 3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH, 16.0, 1.0, 16.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN, 3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH, 16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMBW_STR_IS_ALLOWED, "True");
    // Spec sheet says 448000, multiply by .75 to 336000
    tp->AddDouble(MEMBW_STR_MINIMUM_BANDWIDTH, 336000.0, 1.0, 1000000.0);
    m_featureDb[id][MEMBW_PLUGIN_WL_NAME] = tp;
    // Add device id to m_globalChanges since this device requires a throttle mask
    m_globalChanges.insert(id);
    tp = 0;

    // V100S
    id = "1df6";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 1024.0, 1.0, 2048.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 249.0, 200.0, 250.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 87.0, 40.0, 90.0);
    tp->AddString(CP_STR_USE_DGEMM, "False");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 3250.0, 30.0, 10000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    // V100S has a DGEMM performance of 7000; multiply by .75 to get 5250 
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 5250.0, 30.0, 20000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 87.0, 40.0, 90.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN, 3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH, 16.0, 1.0, 16.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN, 3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH, 16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMBW_STR_IS_ALLOWED, "True");
    // Spec sheet says 900000, multiply by .75 to 675000
    tp->AddDouble(MEMBW_STR_MINIMUM_BANDWIDTH, 675000.0, 1.0, 1000000.0);
    m_featureDb[id][MEMBW_PLUGIN_WL_NAME] = tp;
    // Add device id to m_globalChanges since this device requires a throttle mask
    m_globalChanges.insert(id);
    tp = 0;

    // RTX 8000 and 6000 (Lwdqro)                                                                                
    // This whitelist covers both because they use the same PCI Device Id, so the settings are the               
    // lowest value between the two 
    id = "1e30";
    tp = new TestParameters();
    tp->AddString(CP_STR_IS_ALLOWED, "True");
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 1024.0, 1.0, 2048.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 259.0, 200.0, 260.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 89.0, 40.0, 94.0);
    tp->AddString(CP_STR_USE_DGEMM, "False");
    m_featureDb[id][CP_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(CPERF_STR_IS_ALLOWED, "True");
    tp->AddString(CPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(CPERF_STR_TARGET_PERF, 3250.0, 30.0, 10000.0);
    m_featureDb[id][CPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(SMPERF_STR_IS_ALLOWED, "True");
    // RTX 6000 and 8000 had a SGEMM performance of 13000 in dcgmproftester -t 1007; multiply by .75 to get 9750
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 9750.0, 30.0, 20000.0);
    tp->AddString(SMPERF_STR_USE_DGEMM, "False");
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 89.0, 40.0, 94.0);
    m_featureDb[id][SMPERF_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(BG_STR_IS_ALLOWED, "True");
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN, 3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH, 16.0, 1.0, 16.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN, 3.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH, 16.0, 1.0, 16.0);
    m_featureDb[id][BG_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMORY_STR_IS_ALLOWED, "True");
    m_featureDb[id][MEMORY_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "True");
    m_featureDb[id][GPUBURN_PLUGIN_WL_NAME] = tp;
    tp = new TestParameters();
    tp->AddString(MEMBW_STR_IS_ALLOWED, "True");
    // dcgmproftester -t 1005 for the 8000 says 553700, multiply by .75 to 415275 (6000 was 672100, but we take  
    // the lower value)                                                                                          
    tp->AddDouble(MEMBW_STR_MINIMUM_BANDWIDTH, 415275.0, 1.0, 1000000.0);
    m_featureDb[id][MEMBW_PLUGIN_WL_NAME] = tp;
    // Add device id to m_globalChanges since this device requires a throttle mask
    m_globalChanges.insert(id);
    tp = 0;
}

/*****************************************************************************/
void Whitelist::UpdateGlobalsForDeviceId(const std::string &deviceId)
{
    /* NOTE: As this function is updated we need to update testing/python/test_utils.py's is_throttling_masked()
     * in order to avoid false positive test failures */

    // Tesla K80 ("Stella Duo" Gemini) and Tesla T4 - TU104_PG183_SKU200
    if ("102d" == deviceId || "1eb8" == deviceId)
    {
        // K80 expects some throttling issues - see bug2454355/DCGM-865 for details
        if (lwvsCommon.throttleIgnoreMask == DCGM_INT64_BLANK)
        {
            // Override the throttle mask only if the user has not specified a mask
            lwvsCommon.throttleIgnoreMask = MAX_THROTTLE_IGNORE_MASK_VALUE;
        }
        return;
    }
    // V100S experiences SW throttling
    else if ("1df6" == deviceId)
    {
        if (lwvsCommon.throttleIgnoreMask == DCGM_INT64_BLANK)
        {
            lwvsCommon.throttleIgnoreMask = DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL;
        }
    }
    // RTX 6000 experiences HW throttling
    else if ("1e30" == deviceId)
    {
        if (lwvsCommon.throttleIgnoreMask == DCGM_INT64_BLANK)
        {
            lwvsCommon.throttleIgnoreMask = DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN |                            
                                            DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL |                             
                                            DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL;
        }
    }
}
