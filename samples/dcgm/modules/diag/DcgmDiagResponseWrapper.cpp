#include <string.h>

#include "DcgmDiagResponseWrapper.h"
#include "logging.h"
#include "LwvsJsonStrings.h"
#include "dcgm_errors.h"

const std::string blacklistName("Blacklist");
const std::string lwmlLibName("LWML Library");
const std::string lwdaMainLibName("LWCA Main Library");
const std::string lwdaTkLibName("LWCA Toolkit Libraries");
const std::string permissionsName("Permissions and OS-related Blocks");
const std::string persistenceName("Persistence Mode");
const std::string elwName("Elwironmental Variables");
const std::string pageRetirementName("Page Retirement");
const std::string graphicsName("Graphics Processes");
const std::string inforomName("Inforom");

const std::string swTestNames[] =
{
    blacklistName,
    lwmlLibName,
    lwdaMainLibName,
    lwdaTkLibName,
    permissionsName,
    persistenceName,
    elwName,
    pageRetirementName,
    graphicsName,
    inforomName
};


/*****************************************************************************/
DcgmDiagResponseWrapper::DcgmDiagResponseWrapper() : m_version(0)
{
    memset(&m_response, 0, sizeof(m_response));
}

/*****************************************************************************/
bool DcgmDiagResponseWrapper::StateIsValid() const
{
    return m_version != 0;
}


/*****************************************************************************/
void DcgmDiagResponseWrapper::InitializeResponseStruct(unsigned int numGpus)
{
    if (StateIsValid() == false)
    {
        PRINT_ERROR("", "ERROR: Must initialize DcgmDiagResponseWrapper before using.");
        return;
    }

    if (m_version == dcgmDiagResponse_version3)
    {
        m_response.v3ptr->version = dcgmDiagResponse_version3;

        // initialize everything as a skip
        m_response.v3ptr->blacklist = DCGM_DIAG_RESULT_NOT_RUN;
        m_response.v3ptr->lwmlLibrary = DCGM_DIAG_RESULT_NOT_RUN;
        m_response.v3ptr->lwdaMainLibrary = DCGM_DIAG_RESULT_NOT_RUN;
        m_response.v3ptr->lwdaRuntimeLibrary = DCGM_DIAG_RESULT_NOT_RUN;
        m_response.v3ptr->permissions = DCGM_DIAG_RESULT_NOT_RUN;
        m_response.v3ptr->persistenceMode = DCGM_DIAG_RESULT_NOT_RUN;
        m_response.v3ptr->environment = DCGM_DIAG_RESULT_NOT_RUN;
        m_response.v3ptr->pageRetirement = DCGM_DIAG_RESULT_NOT_RUN;
        m_response.v3ptr->graphicsProcesses = DCGM_DIAG_RESULT_NOT_RUN;
        m_response.v3ptr->inforom = DCGM_DIAG_RESULT_NOT_RUN;

        m_response.v3ptr->gpuCount = numGpus;

        for (unsigned int i = 0; i < numGpus; i++)
        {
            for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT; j++) {
                m_response.v3ptr->perGpuResponses[i].results[j].status = DCGM_DIAG_RESULT_NOT_RUN;
                m_response.v3ptr->perGpuResponses[i].results[j].info[0] = '\0';
                m_response.v3ptr->perGpuResponses[i].results[j].warning[0] = '\0';
            }
            
            // Set correct GPU ids for the valid portion of the response
            m_response.v3ptr->perGpuResponses[i].gpuId = i;
        }

        // Set the unused part of the response to have bogus GPU ids
        for (unsigned int i = numGpus; i < DCGM_MAX_NUM_DEVICES; i++)
        {
            m_response.v3ptr->perGpuResponses[i].gpuId = DCGM_MAX_NUM_DEVICES;
        }
    }
    else if (m_version == dcgmDiagResponse_version4)
    {
        m_response.v4ptr->version = dcgmDiagResponse_version4;
        m_response.v4ptr->levelOneTestCount = DCGM_SWTEST_COUNT;

        // initialize everything as a skip
        for (unsigned int i = 0; i < DCGM_SWTEST_COUNT; i++)
        {
            m_response.v4ptr->levelOneResults[i].status = DCGM_DIAG_RESULT_NOT_RUN;
        }
        
        m_response.v4ptr->gpuCount = numGpus;

        for (unsigned int i = 0; i < numGpus; i++)
        {
            for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT; j++)
            {
                m_response.v4ptr->perGpuResponses[i].results[j].status = DCGM_DIAG_RESULT_NOT_RUN;
                m_response.v4ptr->perGpuResponses[i].results[j].info[0] = '\0';
                m_response.v4ptr->perGpuResponses[i].results[j].warning[0] = '\0';
            }
            
            // Set correct GPU ids for the valid portion of the response
            m_response.v4ptr->perGpuResponses[i].gpuId = i;
        }

        // Set the unused part of the response to have bogus GPU ids
        for (unsigned int i = numGpus; i < DCGM_MAX_NUM_DEVICES; i++)
        {
            m_response.v4ptr->perGpuResponses[i].gpuId = DCGM_MAX_NUM_DEVICES;
        }
    }
    else
    {
        // Version 5
        m_response.v5ptr->version = dcgmDiagResponse_version;
        m_response.v5ptr->levelOneTestCount = DCGM_SWTEST_COUNT;
        
        // initialize everything as a skip
        for (unsigned int i = 0; i < DCGM_SWTEST_COUNT; i++)
        {
            m_response.v5ptr->levelOneResults[i].status = DCGM_DIAG_RESULT_NOT_RUN;
        }
        
        m_response.v5ptr->gpuCount = numGpus;

        for (unsigned int i = 0; i < numGpus; i++)
        {
            for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT; j++)
            {
                m_response.v5ptr->perGpuResponses[i].results[j].status = DCGM_DIAG_RESULT_NOT_RUN;
                m_response.v5ptr->perGpuResponses[i].results[j].info[0] = '\0';
                m_response.v5ptr->perGpuResponses[i].results[j].error.msg[0] = '\0';
                m_response.v5ptr->perGpuResponses[i].results[j].error.code = DCGM_FR_OK;
            }
            
            // Set correct GPU ids for the valid portion of the response
            m_response.v5ptr->perGpuResponses[i].gpuId = i;
        }

        // Set the unused part of the response to have bogus GPU ids
        for (unsigned int i = numGpus; i < DCGM_MAX_NUM_DEVICES; i++)
        {
            m_response.v5ptr->perGpuResponses[i].gpuId = DCGM_MAX_NUM_DEVICES;
        }
    }
}

bool DcgmDiagResponseWrapper::IsValidGpuIndex(unsigned int gpuIndex)
{
    unsigned int count;

    switch (m_version)
    {
        case dcgmDiagResponse_version3:
            count = m_response.v3ptr->gpuCount;
            break;
        case dcgmDiagResponse_version4:
            count = m_response.v4ptr->gpuCount;
            break;
        case dcgmDiagResponse_version5:
            count = m_response.v5ptr->gpuCount;
            break;
        default:
            PRINT_ERROR("%u", "ERROR: Internal version %u doesn't match any supported!", m_version);
            return false;
            // Unreached
            break;
    }

    if (gpuIndex >= count)
    {
        PRINT_ERROR("%u %u", "ERROR: gpuIndex %u is higher than gpu count %u", gpuIndex, count);
        return false;
    }

    return true;
}

/*****************************************************************************/
void DcgmDiagResponseWrapper::SetPerGpuResponseState(unsigned int testIndex, dcgmDiagResult_t result,
                                                     unsigned int gpuIndex, unsigned int rc)
{
    if (StateIsValid() == false)
    {
        PRINT_ERROR("", "ERROR: Must initialize DcgmDiagResponseWrapper before using.");
        return;
    }

    // Only set the results for tests run for each GPU
    if (testIndex >= DCGM_PER_GPU_TEST_COUNT)
    {
        return;
    }

    if (IsValidGpuIndex(gpuIndex) == false)
    {
        return;
    }

    if (m_version == dcgmDiagResponse_version3)
    {
        m_response.v3ptr->perGpuResponses[gpuIndex].results[testIndex].status = result;
        if (testIndex == DCGM_DIAGNOSTIC_INDEX)
        {
            m_response.v3ptr->perGpuResponses[gpuIndex].hwDiagnosticReturn = rc;
        }
    }
    else if (m_version == dcgmDiagResponse_version4)
    {
        m_response.v4ptr->perGpuResponses[gpuIndex].results[testIndex].status = result;
        if (testIndex == DCGM_DIAGNOSTIC_INDEX)
            m_response.v4ptr->perGpuResponses[gpuIndex].hwDiagnosticReturn = rc;
    }
    else
    {
        // Version 5
        m_response.v5ptr->perGpuResponses[gpuIndex].results[testIndex].status = result;
        if (testIndex == DCGM_DIAGNOSTIC_INDEX)
        {
            m_response.v5ptr->perGpuResponses[gpuIndex].hwDiagnosticReturn = rc;
        }
    }
}
    
/*****************************************************************************/
dcgmReturn_t DcgmDiagResponseWrapper::AddErrorDetail(unsigned int gpuIndex, unsigned int testIndex,
                                                     const std::string &testname, dcgmDiagErrorDetail_t &ed,
                                                     dcgmDiagResult_t result)
{
    if (StateIsValid() == false)
    {
        PRINT_ERROR("", "ERROR: Must initialize DcgmDiagResponseWrapper before using.");
        return DCGM_ST_UNINITIALIZED;
    }

    unsigned int l1Index = 0; 
    if (testIndex >= DCGM_PER_GPU_TEST_COUNT)
    {
        l1Index = GetBasicTestResultIndex(testname);
        if (l1Index >= DCGM_SWTEST_COUNT)
        {
            PRINT_ERROR("%u %s", "ERROR: Test index %u indicates a level one test, but testname '%s' is not found.",
                        testIndex, testname.c_str()); 
            return DCGM_ST_BADPARAM;
        }
    }

    if (m_version == dcgmDiagResponse_version3)
    {
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT)
        {
            switch (l1Index)
            {
                case DCGM_SWTEST_BLACKLIST:
                    m_response.v3ptr->blacklist = result;
                    break;
                case DCGM_SWTEST_LWML_LIBRARY:
                    m_response.v3ptr->lwmlLibrary = result;
                    break;
                case DCGM_SWTEST_LWDA_MAIN_LIBRARY:
                    m_response.v3ptr->lwdaMainLibrary = result;
                    break;
                case DCGM_SWTEST_LWDA_RUNTIME_LIBRARY:
                    m_response.v3ptr->lwdaRuntimeLibrary = result;
                    break;
                case DCGM_SWTEST_PERMISSIONS:
                    m_response.v3ptr->permissions = result;
                    break;
                case DCGM_SWTEST_PERSISTENCE_MODE:
                    m_response.v3ptr->persistenceMode = result;
                    break;
                case DCGM_SWTEST_ELWIRONMENT:
                    m_response.v3ptr->environment = result;
                    break;
                case DCGM_SWTEST_PAGE_RETIREMENT:
                    m_response.v3ptr->pageRetirement = result;
                    break;
                case DCGM_SWTEST_GRAPHICS_PROCESSES:
                    m_response.v3ptr->graphicsProcesses = result;
                    break;
                case DCGM_SWTEST_INFOROM:
                    m_response.v3ptr->inforom = result;
                    break;
            }
        }
        else
        {
            snprintf(m_response.v3ptr->perGpuResponses[gpuIndex].results[testIndex].warning,
                     sizeof(m_response.v3ptr->perGpuResponses[gpuIndex].results[testIndex].warning),
                     "%s", ed.msg);
            m_response.v3ptr->perGpuResponses[gpuIndex].results[testIndex].status = result;
        }
    }
    else if (m_version == dcgmDiagResponse_version4)
    {
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT)
        {
            // We are looking at the l1 tests
            snprintf(m_response.v4ptr->levelOneResults[l1Index].warning,
                     sizeof(m_response.v4ptr->levelOneResults[l1Index].warning), "%s", ed.msg);
        }
        else
        {
            snprintf(m_response.v4ptr->perGpuResponses[gpuIndex].results[testIndex].warning,
                     sizeof(m_response.v4ptr->perGpuResponses[gpuIndex].results[testIndex].warning),
                     "%s", ed.msg);
            m_response.v4ptr->perGpuResponses[gpuIndex].results[testIndex].status = result;
        }
    }
    else
    {
        // version 5
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT)
        {
            // We are looking at the l1 tests
            snprintf(m_response.v5ptr->levelOneResults[l1Index].error.msg,
                     sizeof(m_response.v5ptr->levelOneResults[l1Index].error.msg), "%s", ed.msg);
            m_response.v5ptr->levelOneResults[l1Index].error.code = ed.code;
            m_response.v5ptr->levelOneResults[l1Index].status = result;
        }
        else
        {
            snprintf(m_response.v5ptr->perGpuResponses[gpuIndex].results[testIndex].error.msg,
                     sizeof(m_response.v5ptr->perGpuResponses[gpuIndex].results[testIndex].error.msg),
                     "%s", ed.msg);
            m_response.v5ptr->perGpuResponses[gpuIndex].results[testIndex].error.code = ed.code;
            m_response.v5ptr->perGpuResponses[gpuIndex].results[testIndex].status = result;
        }
    }
    
    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmDiagResponseWrapper::AddPerGpuMessage(unsigned int testIndex, const std::string &msg,
                                               unsigned int gpuIndex, bool warning)
{ 
    if (StateIsValid() == false)
    {
        PRINT_ERROR("", "ERROR: Must initialize DcgmDiagResponseWrapper before using.");
        return;
    }

    if (IsValidGpuIndex(gpuIndex) == false)
    {
        return;
    }

    if (m_version == dcgmDiagResponse_version3)
    {
        if (warning == true)
        {
            snprintf(m_response.v3ptr->perGpuResponses[gpuIndex].results[testIndex].warning,
                     sizeof(m_response.v3ptr->perGpuResponses[gpuIndex].results[testIndex].warning),
                     "%s", msg.c_str());
        }
        else
        {
            snprintf(m_response.v3ptr->perGpuResponses[gpuIndex].results[testIndex].info,
                     sizeof(m_response.v3ptr->perGpuResponses[gpuIndex].results[testIndex].info),
                     "%s", msg.c_str());
        }
    }
    else if (m_version == dcgmDiagResponse_version4)
    {
        if (warning == true)
        {
            snprintf(m_response.v4ptr->perGpuResponses[gpuIndex].results[testIndex].warning,
                     sizeof(m_response.v4ptr->perGpuResponses[gpuIndex].results[testIndex].warning),
                     "%s", msg.c_str());
        }
        else
        {
            snprintf(m_response.v4ptr->perGpuResponses[gpuIndex].results[testIndex].info,
                     sizeof(m_response.v4ptr->perGpuResponses[gpuIndex].results[testIndex].info),
                     "%s", msg.c_str());
        }
    }
    else
    {
        // version 5
        if (warning == true)
        {
            snprintf(m_response.v5ptr->perGpuResponses[gpuIndex].results[testIndex].error.msg,
                     sizeof(m_response.v5ptr->perGpuResponses[gpuIndex].results[testIndex].error.msg),
                     "%s", msg.c_str());
        }
        else
        {
            snprintf(m_response.v5ptr->perGpuResponses[gpuIndex].results[testIndex].info,
                     sizeof(m_response.v5ptr->perGpuResponses[gpuIndex].results[testIndex].info),
                     "%s", msg.c_str());
        }
    }
}

/*****************************************************************************/
void DcgmDiagResponseWrapper::SetGpuIndex(unsigned int gpuIndex)
{
    if (StateIsValid() == false)
    {
        PRINT_ERROR("", "ERROR: Must initialize DcgmDiagResponseWrapper before using.");
        return;
    }

    if (m_version == dcgmDiagResponse_version3)
    {
        m_response.v3ptr->perGpuResponses[gpuIndex].gpuId = gpuIndex;
    }
    else if (m_version == dcgmDiagResponse_version4)
    {
        m_response.v4ptr->perGpuResponses[gpuIndex].gpuId = gpuIndex;
    }
    else
    {
        m_response.v5ptr->perGpuResponses[gpuIndex].gpuId = gpuIndex;
    }
}

/*****************************************************************************/
unsigned int DcgmDiagResponseWrapper::GetBasicTestResultIndex(const std::string &testname)
{
    for (unsigned int i = 0; i < DCGM_SWTEST_COUNT; i++)
    {
        if (testname == swTestNames[i])
        {
            return i;
        }
    }

    return DCGM_SWTEST_COUNT;
}

/*****************************************************************************/
void DcgmDiagResponseWrapper::RecordSystemError(const std::string &sysError)
{
    if (m_version == dcgmDiagResponse_version3)
    {
        snprintf(m_response.v3ptr->systemError, sizeof(m_response.v3ptr->systemError), "%s", sysError.c_str());
    }
    else if (m_version == dcgmDiagResponse_version4)
    {
        snprintf(m_response.v4ptr->systemError, sizeof(m_response.v4ptr->systemError), "%s", sysError.c_str());
    }
    else
    {
        snprintf(m_response.v5ptr->systemError.msg, sizeof(m_response.v5ptr->systemError.msg), "%s",
                 sysError.c_str());
        m_response.v5ptr->systemError.code = DCGM_FR_INTERNAL;
    }
}

/*****************************************************************************/
void DcgmDiagResponseWrapper::SetGpuCount(unsigned int gpuCount)
{
    if (m_version == dcgmDiagResponse_version3)
    {
        m_response.v3ptr->gpuCount = gpuCount;
    }
    else if (m_version == dcgmDiagResponse_version4)
    {
        m_response.v4ptr->gpuCount = gpuCount;
    }
    else
    {
        m_response.v5ptr->gpuCount = gpuCount;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagResponseWrapper::SetVersion3(dcgmDiagResponse_v3 *response)
{
    if (m_version != 0)
    {
        // We don't support setting the version twice
        return DCGM_ST_NOT_SUPPORTED;
    }

    m_version = dcgmDiagResponse_version3;
    m_response.v3ptr = response;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagResponseWrapper::SetVersion4(dcgmDiagResponse_v4 *response)
{
    if (m_version != 0)
    {
        // We don't support setting the version twice
        return DCGM_ST_NOT_SUPPORTED;
    }
    
    m_version = dcgmDiagResponse_version4;
    m_response.v4ptr = response;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagResponseWrapper::SetVersion5(dcgmDiagResponse_v5 *response)
{
    if (m_version != 0)
    {
        // We don't support setting the version twice
        return DCGM_ST_NOT_SUPPORTED;
    }
    
    m_version = dcgmDiagResponse_version5;
    m_response.v5ptr = response;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagResponseWrapper::RecordTrainingMessage(const std::string &trainingMsg)
{
    if (m_version < dcgmDiagResponse_version4)
    {
        // Only version 4 and greater can set the training message
        return DCGM_ST_VER_MISMATCH;
    }
    else if (m_version == dcgmDiagResponse_version4)
    {
        snprintf(m_response.v4ptr->trainingMsg, sizeof(m_response.v4ptr->trainingMsg), "%s", trainingMsg.c_str());
    }
    else
    {
        snprintf(m_response.v5ptr->trainingMsg, sizeof(m_response.v5ptr->trainingMsg), "%s", trainingMsg.c_str());
    }

    return DCGM_ST_OK;
}

