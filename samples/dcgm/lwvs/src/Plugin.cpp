#include "Plugin.h"

/*************************************************************************/
Plugin::Plugin(): m_results(), m_warnings(), m_warningsPerGPU(), m_verboseInfo(), m_verboseInfoPerGPU(),
                  m_values(), m_dataMutex(0), m_mutex(0)
{}

/*************************************************************************/
Plugin::~Plugin()
{}

/*************************************************************************/
void Plugin::ResetResultsAndMessages()
{
    DcgmLockGuard lock(&m_dataMutex);
    m_results.clear();
    m_warnings.clear();
    m_verboseInfo.clear();
    m_warningsPerGPU.clear();
    m_verboseInfoPerGPU.clear();
}

/*************************************************************************/
void Plugin::InitializeForGpuList(const std::vector<unsigned int> &gpuList)
{
    ResetResultsAndMessages();
    DcgmLockGuard lock(&m_dataMutex);

    for (size_t i = 0; i < gpuList.size(); i++)
    {
        // Accessing the value at non-existent key default constructs a value for the key
        m_warningsPerGPU[gpuList[i]];
        m_verboseInfoPerGPU[gpuList[i]];
        m_results[gpuList[i]] = LWVS_RESULT_PASS; // default result should be pass
    }
    m_gpuList = gpuList;
}

/* Logging */
/*************************************************************************/
void Plugin::AddError(const DcgmError &error)
{
    DcgmLockGuard lock(&m_dataMutex);
    PRINT_WARNING("%s %s", "plugin %s: %s", m_infoStruct.name.c_str(), error.GetMessage().c_str());
    m_errors.push_back(error);
}

/*************************************************************************/
void Plugin::AddInfo(const std::string &info)
{
    DcgmLockGuard lock(&m_dataMutex);
    PRINT_INFO("%s %s", "plugin %s: %s", m_infoStruct.name.c_str(), info.c_str());
}

/*************************************************************************/
void Plugin::AddInfoVerbose(const std::string &info)
{
    DcgmLockGuard lock(&m_dataMutex);
    m_verboseInfo.push_back(info);
    PRINT_INFO("%s %s", "plugin %s: %s", m_infoStruct.name.c_str(), info.c_str());
}

/*************************************************************************/
void Plugin::AddErrorForGpu(unsigned int gpuId, const DcgmError &error)
{
    DcgmLockGuard lock(&m_dataMutex);
    PRINT_WARNING("%s %s %u", "plugin %s: %s (GPU %u)", m_infoStruct.name.c_str(), error.GetMessage().c_str(),
                  gpuId);
    m_errorsPerGPU[gpuId].push_back(error);
}

/*************************************************************************/
void Plugin::AddInfoVerboseForGpu(unsigned int gpuId, const std::string &info)
{
    DcgmLockGuard lock(&m_dataMutex);
    PRINT_INFO("%s %s %d", "plugin %s: %s (GPU %d)", m_infoStruct.name.c_str(), info.c_str(), gpuId);
    m_verboseInfoPerGPU[gpuId].push_back(info);
}

/* Manage results */
/*************************************************************************/
lwvsPluginResult_t Plugin::GetOverallResult(const lwvsPluginGpuResults_t& results)
{
    bool warning = false;
    size_t skipCount = 0;
    lwvsPluginGpuResults_t::const_iterator it;

    for (it = results.begin(); it != results.end(); ++it)
    {
        switch (it->second)
        {
            case LWVS_RESULT_PASS:
                continue;
            case LWVS_RESULT_FAIL:
                return LWVS_RESULT_FAIL;
            case LWVS_RESULT_WARN:
                warning = true;
                break; /* Exit switch case */
            case LWVS_RESULT_SKIP:
                skipCount += 1;
                break; /* Exit switch case */

            default:
                PRINT_ERROR("%d", "Got unknown result value: %d", it->second);
                break;
        }
    }

    if (warning)
    {
        return LWVS_RESULT_WARN;
    }

    if (skipCount == results.size())
    {
        return LWVS_RESULT_SKIP;
    }

    return LWVS_RESULT_PASS;
}

/*************************************************************************/
lwvsPluginResult_t Plugin::GetResult()
{
    DcgmLockGuard lock(&m_dataMutex);
    return GetOverallResult(m_results);
}

/*************************************************************************/
void Plugin::SetResult(lwvsPluginResult_t res)
{
    DcgmLockGuard lock(&m_dataMutex);
    lwvsPluginGpuResults_t::iterator it;
    for (it = m_results.begin(); it != m_results.end(); ++it)
    {
        it->second = res;
    }
}

/*************************************************************************/
void Plugin::SetResultForGpu(unsigned int gpuId, lwvsPluginResult_t res)
{
    DcgmLockGuard lock(&m_dataMutex);
    m_results[gpuId] = res;
}
