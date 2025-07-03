#include <algorithm>

#include "ParameterValidator.h"
#include "PluginStrings.h"

void TestInfo::SetName(const std::string &testname)
{
    m_info.testname = testname;
}

void TestInfo::AddParameter(const std::string &parameter)
{
    m_info.parameters.insert(parameter);
}

void TestInfo::Clear()
{
    m_info.testname.clear();
    m_info.parameters.clear();
    m_subtests.clear();
}

void TestInfo::AddSubtest(const std::string &subtest)
{
    m_subtests[subtest].testname = subtest;
}

void TestInfo::AddSubtestParameter(const std::string &subtest, const std::string &parameter)
{
    m_subtests[subtest].parameters.insert(parameter);
}

bool TestInfo::HasParameter(const std::string &parameter) const
{
    return m_info.parameters.find(parameter) != m_info.parameters.end();
}

bool TestInfo::HasSubtestParameter(const std::string &subtest, const std::string &parameter)
{
    return m_subtests[subtest].parameters.find(parameter) != m_subtests[subtest].parameters.end();
}

bool TestInfo::HasSubtest(const std::string &subtest) const
{
    return m_subtests.find(subtest) != m_subtests.end();
}
    
bool ParameterValidator::IsValidTestName(const std::string &testname) const
{
    std::string loweredTest(testname);
    std::transform(loweredTest.begin(), loweredTest.end(), loweredTest.begin(), ::tolower);
    return m_possiblePlugins.find(loweredTest) != m_possiblePlugins.end();
}

bool ParameterValidator::IsValidParameter(const std::string &testname, const std::string &parameter)
{
    std::string loweredTest(testname);
    std::string loweredParam(parameter);
    std::transform(loweredTest.begin(), loweredTest.end(), loweredTest.begin(), ::tolower);
    std::transform(loweredParam.begin(), loweredParam.end(), loweredParam.begin(), ::tolower);
    return m_possiblePlugins[loweredTest].HasParameter(loweredParam);
}

bool ParameterValidator::IsValidSubtest(const std::string &testname, const std::string &subtest)
{
    std::string loweredTest(testname);
    std::string loweredSub(subtest);
    std::transform(loweredTest.begin(), loweredTest.end(), loweredTest.begin(), ::tolower);
    std::transform(loweredSub.begin(), loweredSub.end(), loweredSub.begin(), ::tolower);

    return m_possiblePlugins[loweredTest].HasSubtest(loweredSub);
}

bool ParameterValidator::IsValidSubtestParameter(const std::string &testname, const std::string &subtest,
                                                 const std::string &parameter) 
{
    std::string loweredTest(testname);
    std::string loweredSub(subtest);
    std::string loweredParam(parameter);

    std::transform(loweredTest.begin(), loweredTest.end(), loweredTest.begin(), ::tolower);
    std::transform(loweredSub.begin(), loweredSub.end(), loweredSub.begin(), ::tolower);
    std::transform(loweredParam.begin(), loweredParam.end(), loweredParam.begin(), ::tolower);
    return m_possiblePlugins[loweredTest].HasSubtestParameter(subtest, parameter);
}

ParameterValidator::ParameterValidator()
{
    Init();
}

void ParameterValidator::Init()
{
    AddSoftware();
    AddBusGrind();
    AddTargetPower();
    AddTargetStress();
    AddMemory();
    AddSMStress();
    AddGpuBurn();
    AddContextCreate();
    AddMemoryBandwidth();
}

void ParameterValidator::AddMemoryBandwidth()
{
    TestInfo ti;
    ti.SetName(MEMBW_PLUGIN_WL_NAME);
    ti.AddParameter(MEMBW_STR_IS_ALLOWED);
    ti.AddParameter(MEMBW_STR_SBE_ERROR_THRESHOLD);
    m_possiblePlugins[MEMBW_PLUGIN_WL_NAME] = ti;
}

void ParameterValidator::AddContextCreate()
{
    TestInfo ti;
    ti.SetName(CTXCREATE_PLUGIN_WL_NAME);
    ti.AddParameter(CTXCREATE_IS_ALLOWED);
    ti.AddParameter(CTXCREATE_IGNORE_EXCLUSIVE);
    m_possiblePlugins[CTXCREATE_PLUGIN_WL_NAME] = ti;
}

void ParameterValidator::AddGpuBurn()
{
    TestInfo ti;
    ti.SetName(GPUBURN_PLUGIN_WL_NAME);
    ti.AddParameter(GPUBURN_STR_SBE_ERROR_THRESHOLD);
    ti.AddParameter(GPUBURN_STR_TEST_DURATION);
    ti.AddParameter(GPUBURN_STR_USE_DOUBLES);
    ti.AddParameter(GPUBURN_STR_TEMPERATURE_MAX);
    ti.AddParameter(GPUBURN_STR_IS_ALLOWED);
    m_possiblePlugins[GPUBURN_PLUGIN_WL_NAME] = ti;
}

void ParameterValidator::AddSMStress()
{
    TestInfo ti;
    ti.SetName(SMPERF_PLUGIN_WL_NAME);
    ti.AddParameter(SMPERF_STR_TEST_DURATION);
    ti.AddParameter(SMPERF_STR_TARGET_PERF);
    ti.AddParameter(SMPERF_STR_TARGET_PERF_MIN_RATIO);
    ti.AddParameter(SMPERF_STR_TEMPERATURE_MAX);
    ti.AddParameter(SMPERF_STR_SBE_ERROR_THRESHOLD);
    ti.AddParameter(SMPERF_STR_IS_ALLOWED);
    ti.AddParameter(SMPERF_STR_USE_DGEMM);
    ti.AddParameter(SMPERF_STR_MAX_MEMORY_CLOCK);
    ti.AddParameter(SMPERF_STR_MAX_GRAPHICS_CLOCK);
    m_possiblePlugins[SMPERF_PLUGIN_WL_NAME] = ti;
}

void ParameterValidator::AddMemory()
{
    TestInfo ti;
    ti.SetName(MEMORY_PLUGIN_WL_NAME);
    ti.AddParameter(MEMORY_STR_IS_ALLOWED);
    m_possiblePlugins[MEMORY_PLUGIN_WL_NAME] = ti;
}

void ParameterValidator::AddTargetStress()
{
    TestInfo ti;
    ti.SetName(CPERF_PLUGIN_WL_NAME);
    ti.AddParameter(CPERF_STR_TEST_DURATION);
    ti.AddParameter(CPERF_STR_TARGET_PERF);
    ti.AddParameter(CPERF_STR_TARGET_PERF_MIN_RATIO);
    ti.AddParameter(CPERF_STR_TEMPERATURE_MAX);
    ti.AddParameter(CPERF_STR_IS_ALLOWED);
    ti.AddParameter(CPERF_STR_USE_DGEMM);
    ti.AddParameter(CPERF_STR_LWDA_STREAMS_PER_GPU);
    ti.AddParameter(CPERF_STR_LWDA_OPS_PER_STREAM);
    ti.AddParameter(CPERF_STR_MAX_PCIE_REPLAYS);
    ti.AddParameter(CPERF_STR_MAX_MEMORY_CLOCK);
    ti.AddParameter(CPERF_STR_MAX_GRAPHICS_CLOCK);
    ti.AddParameter(CPERF_STR_SBE_ERROR_THRESHOLD);
    m_possiblePlugins[CPERF_PLUGIN_WL_NAME] = ti;
}

void ParameterValidator::AddTargetPower()
{
    TestInfo ti;
    ti.SetName(CP_PLUGIN_WL_NAME);
    ti.AddParameter(CP_STR_TEST_DURATION);
    ti.AddParameter(CP_STR_TARGET_POWER);
    ti.AddParameter(CP_STR_TEMPERATURE_MAX);
    ti.AddParameter(CP_STR_FAIL_ON_CLOCK_DROP);
    ti.AddParameter(CP_STR_USE_DGEMM);
    ti.AddParameter(CP_STR_LWDA_STREAMS_PER_GPU);
    ti.AddParameter(CP_STR_READJUST_INTERVAL);
    ti.AddParameter(CP_STR_TARGET_POWER_MIN_RATIO);
    ti.AddParameter(CP_STR_TARGET_POWER_MAX_RATIO);
    ti.AddParameter(CP_STR_ENFORCED_POWER_LIMIT);
    ti.AddParameter(CP_STR_MAX_MEMORY_CLOCK);
    ti.AddParameter(CP_STR_MAX_GRAPHICS_CLOCK);
    ti.AddParameter(CP_STR_OPS_PER_REQUEUE);
    ti.AddParameter(CP_STR_STARTING_MATRIX_DIM);
    ti.AddParameter(CP_STR_IS_ALLOWED);
    ti.AddParameter(CP_STR_SBE_ERROR_THRESHOLD);
    m_possiblePlugins[CP_PLUGIN_WL_NAME] = ti;
}

void ParameterValidator::AddSoftware()
{
    TestInfo ti;
    ti.SetName(SW_PLUGIN_LF_NAME);
    ti.AddParameter(SW_STR_DO_TEST);
    ti.AddParameter(SW_STR_REQUIRE_PERSISTENCE);
    m_possiblePlugins[SW_PLUGIN_LF_NAME] = ti;
}

void ParameterValidator::AddBusGrind()
{
    TestInfo ti;
    ti.SetName(BG_PLUGIN_WL_NAME);
    ti.AddParameter(BG_STR_TEST_PINNED);
    ti.AddParameter(BG_STR_TEST_UNPINNED);
    ti.AddParameter(BG_STR_TEST_P2P_ON);
    ti.AddParameter(BG_STR_TEST_P2P_OFF);
    ti.AddParameter(BG_STR_LWSWITCH_NON_FATAL_CHECK);
    ti.AddParameter(BG_STR_CRC_ERROR_THRESHOLD);
    ti.AddParameter(BG_STR_IS_ALLOWED);
    ti.AddSubtest(BG_SUBTEST_H2D_D2H_SINGLE_PINNED);
    ti.AddSubtest(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED);
    ti.AddSubtest(BG_SUBTEST_H2D_D2H_CONLWRRENT_PINNED);
    ti.AddSubtest(BG_SUBTEST_H2D_D2H_CONLWRRENT_UNPINNED);
    ti.AddSubtest(BG_SUBTEST_H2D_D2H_LATENCY_PINNED);
    ti.AddSubtest(BG_SUBTEST_H2D_D2H_LATENCY_UNPINNED);
    ti.AddSubtest(BG_SUBTEST_P2P_BW_P2P_ENABLED);
    ti.AddSubtest(BG_SUBTEST_P2P_BW_P2P_DISABLED);
    ti.AddSubtest(BG_SUBTEST_P2P_BW_CONLWRRENT_P2P_ENABLED);
    ti.AddSubtest(BG_SUBTEST_P2P_BW_CONLWRRENT_P2P_DISABLED);
    ti.AddSubtest(BG_SUBTEST_1D_EXCH_BW_P2P_ENABLED);
    ti.AddSubtest(BG_SUBTEST_1D_EXCH_BW_P2P_DISABLED);
    ti.AddSubtest(BG_SUBTEST_P2P_LATENCY_P2P_ENABLED);
    ti.AddSubtest(BG_SUBTEST_P2P_LATENCY_P2P_DISABLED);

    ti.AddSubtestParameter(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_INTS_PER_COPY);
    ti.AddSubtestParameter(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_BANDWIDTH);
    ti.AddSubtestParameter(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN);
    ti.AddSubtestParameter(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH);

    ti.AddSubtestParameter(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_INTS_PER_COPY);
    ti.AddSubtestParameter(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_BANDWIDTH);
    ti.AddSubtestParameter(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN);
    ti.AddSubtestParameter(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH);

    ti.AddSubtestParameter(BG_SUBTEST_H2D_D2H_CONLWRRENT_PINNED, BG_STR_INTS_PER_COPY);
    ti.AddSubtestParameter(BG_SUBTEST_H2D_D2H_CONLWRRENT_PINNED, BG_STR_MIN_BANDWIDTH);
    
    ti.AddSubtestParameter(BG_SUBTEST_H2D_D2H_CONLWRRENT_UNPINNED, BG_STR_INTS_PER_COPY);
    ti.AddSubtestParameter(BG_SUBTEST_H2D_D2H_CONLWRRENT_UNPINNED, BG_STR_MIN_BANDWIDTH);

    ti.AddSubtestParameter(BG_SUBTEST_P2P_BW_P2P_ENABLED, BG_STR_INTS_PER_COPY);

    ti.AddSubtestParameter(BG_SUBTEST_P2P_BW_P2P_DISABLED, BG_STR_INTS_PER_COPY);

    ti.AddSubtestParameter(BG_SUBTEST_P2P_BW_CONLWRRENT_P2P_ENABLED, BG_STR_INTS_PER_COPY);
    ti.AddSubtestParameter(BG_SUBTEST_P2P_BW_CONLWRRENT_P2P_ENABLED, BG_STR_ITERATIONS);

    ti.AddSubtestParameter(BG_SUBTEST_P2P_BW_CONLWRRENT_P2P_DISABLED, BG_STR_INTS_PER_COPY);
    ti.AddSubtestParameter(BG_SUBTEST_P2P_BW_CONLWRRENT_P2P_DISABLED, BG_STR_ITERATIONS);

    ti.AddSubtestParameter(BG_SUBTEST_1D_EXCH_BW_P2P_ENABLED, BG_STR_INTS_PER_COPY);

    ti.AddSubtestParameter(BG_SUBTEST_1D_EXCH_BW_P2P_DISABLED, BG_STR_INTS_PER_COPY);

    ti.AddSubtestParameter(BG_SUBTEST_H2D_D2H_LATENCY_PINNED, BG_STR_MAX_LATENCY);

    ti.AddSubtestParameter(BG_SUBTEST_H2D_D2H_LATENCY_UNPINNED, BG_STR_MAX_LATENCY);

    // All subtests can use the BG_STR_ITERATIONS parameter
    for (std::map<std::string, subtestInfo_t>::iterator it = ti.m_subtests.begin(); it != ti.m_subtests.end(); it++)
    {
        ti.AddSubtestParameter(it->first, BG_STR_ITERATIONS);
    }
    
    m_possiblePlugins[BG_PLUGIN_WL_NAME] = ti;
}
