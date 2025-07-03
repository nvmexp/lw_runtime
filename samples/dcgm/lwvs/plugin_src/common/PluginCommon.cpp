#include "PluginCommon.h"
#include "lwda_runtime.h"
#include "DcgmError.h"

#include <sstream>

std::string AppendLwdaDriverError(const std::string &error, LWresult lwRes)
{
    const char *lwdaErrorStr = 0;
    lwGetErrorString(lwRes, &lwdaErrorStr);

    if (lwdaErrorStr != 0)
    {
        std::stringstream buf;
        buf << error << ": '" << lwdaErrorStr << "'.";
        return buf.str();
    }
    else
    {
        return error;
    }
}


#define caseForEnumToString(name)   \
    case name:                      \
        return #name
/* lwdaGetErrorString equivalent for lwblasStatus_t */
const char* lwblasGetErrorString(lwblasStatus_t status)
{
    switch (status)
    {
        caseForEnumToString(LWBLAS_STATUS_SUCCESS);
        caseForEnumToString(LWBLAS_STATUS_NOT_INITIALIZED);
        caseForEnumToString(LWBLAS_STATUS_ALLOC_FAILED);
        caseForEnumToString(LWBLAS_STATUS_ILWALID_VALUE);
        caseForEnumToString(LWBLAS_STATUS_ARCH_MISMATCH);
        caseForEnumToString(LWBLAS_STATUS_MAPPING_ERROR);
        caseForEnumToString(LWBLAS_STATUS_EXELWTION_FAILED);
        caseForEnumToString(LWBLAS_STATUS_INTERNAL_ERROR);
        caseForEnumToString(LWBLAS_STATUS_NOT_SUPPORTED);
        caseForEnumToString(LWBLAS_STATUS_LICENSE_ERROR);
        default:
            return "Unknown error";
    }
}

/*************************************************************************/
std::string AddAPIError(Plugin *p, const char* callName, const char* errorText, unsigned int gpuId, size_t bytes,
                        bool isGpuSpecific)
{
    DcgmError d;

    if (isGpuSpecific)
    {
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_API_FAIL, d, callName, errorText);
    }
    else
    {
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_API_FAIL_GPU, d, callName, gpuId, errorText);
    }

    if (bytes)
    {
        std::stringstream ss;
        ss << "(for " << bytes << " bytes)";
        d.AddDetail(ss.str());
    }

    if (isGpuSpecific)
    {
        p->AddErrorForGpu(gpuId, d);
    }
    else
    {
        p->AddError(d);
    }

    return d.GetMessage();
}

/*****************************************************************************/
std::string AddLwdaError(Plugin *p, const char* callName, lwdaError_t lwSt, unsigned int gpuId, size_t bytes,
                         bool isGpuSpecific)
{
    return AddAPIError(p, callName, lwdaGetErrorString(lwSt), gpuId, bytes, isGpuSpecific);
}

/*****************************************************************************/
std::string AddLwdaError(Plugin *p, const char* callName, LWresult lwSt, unsigned int gpuId, size_t bytes,
                         bool isGpuSpecific)
{
    const char *errorText = NULL;
    lwGetErrorString(lwSt, &errorText);
    if (!errorText)
    {
        errorText = "Unknown error";
    }
    return AddAPIError(p, callName, errorText, gpuId, bytes, isGpuSpecific);
}

/*****************************************************************************/
std::string AddLwblasError(Plugin *p, const char* callName, lwblasStatus_t lwbSt, unsigned int gpuId, size_t bytes,
                           bool isGpuSpecific)
{
    return AddAPIError(p, callName, lwblasGetErrorString(lwbSt), gpuId, bytes, isGpuSpecific);
}

/*****************************************************************************/
void CheckAndSetResult(Plugin* p, const std::vector<unsigned int> &gpuList, size_t i, bool passed,
                       const std::vector<DcgmError> &errorList, bool &allPassed, bool dcgmCommError)
{
    if (passed)
    {
        p->SetResultForGpu(gpuList[i], LWVS_RESULT_PASS);
    }
    else
    {
        allPassed = false;
        p->SetResultForGpu(gpuList[i], LWVS_RESULT_FAIL);
        for (size_t j = 0; j < errorList.size(); j++)
        {
            p->AddErrorForGpu(gpuList[i], errorList[j]);
        }
    }

    if (dcgmCommError)
    {
        for (size_t j = i; j < gpuList.size(); j++)
        {
            p->SetResultForGpu(gpuList[j], LWVS_RESULT_FAIL);
        }
    }
}
