#ifndef _LWVS_LWVS_Plugin_common_H_
#define _LWVS_LWVS_Plugin_common_H_

#include <lwca.h>
#include <lwblas_v2.h>
#include <string>

#include "Plugin.h"

/*****************************************************************************/
#ifndef MAX
#define MAX(a,b) ((a)>(b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b) ((a)<(b) ? (a) : (b))
#endif

/********************************************************************/
/*
 * Translates lwRes to a string and concatenates it. If we cannot get an
 * error from lwGetErrorString(), then just error is returned.
 */
std::string AppendLwdaDriverError(const std::string &error, LWresult lwRes);

/********************************************************************/
/*
 * lwdaGetErrorString equivalent for lwblasStatus_t.
 * Returns the lwblas status name as a string.
 */
const char* lwblasGetErrorString(lwblasStatus_t status);

/*************************************************************************/
/*
 * Adds a warning for the relevant API call error and returns the added string for logging to a log file.
 * Thread-safe method.
 */
std::string AddAPIError(Plugin *p, const char* callName, const char* errorText, unsigned int gpuId, size_t bytes=0,
                        bool isGpuSpecific=true);

/*************************************************************************/
/*
 * Adds a warning for the relevant lwca error and returns the added string for logging to a log file.
 *
 * For simplicity when adding an API error to warnings and logging it, use the LOG_LWDA_ERROR* macro instead.
 * Thread-safe method.
 */
std::string AddLwdaError(Plugin *p, const char* callName, lwdaError_t lwSt, unsigned int gpuId, size_t bytes=0,
                         bool isGpuSpecific=true);

/* Overloaded version for lwca driver errors (i.e. LWResult) */
std::string AddLwdaError(Plugin *p, const char* callName, LWresult lwSt, unsigned int gpuId, size_t bytes=0,
                         bool isGpuSpecific=true);

/*
 * The purpose of these macros is to ensure that the logged message includes accurate line numbers and
 * file name corresponding to the location where the macro is called. If the error message was logged inside the
 * AddLwdaError method, the line number and file name would have been obslwred.
 */
#define LOG_LWDA_ERROR_FOR_PLUGIN(plugin, callName, lwSt, gpuId, ...)                                   \
    {                                                                                                   \
        std::string pluginCommonLwdaError = AddLwdaError(plugin, callName, lwSt, gpuId, ##__VA_ARGS__); \
        PRINT_ERROR("%s", "%s", pluginCommonLwdaError.c_str());                                         \
    }                                                                                                   \
    (void)0

// Only for use by the Plugin subclasses
#define LOG_LWDA_ERROR(callName, lwSt, gpuId, ...)                                                      \
    {                                                                                                   \
        std::string pluginCommonLwdaError = AddLwdaError(this, callName, lwSt, gpuId, ##__VA_ARGS__);   \
        PRINT_ERROR("%s", "%s", pluginCommonLwdaError.c_str());                                         \
    }                                                                                                   \
    (void)0

/*************************************************************************/
/*
 * Adds a warning for the relevant lwblas error and returns the added string for logging to a log file.
 *
 * For simplicity when adding an API error to warnings and logging it, use the LOG_LWBLAS_ERROR* macro instead.
 * Thread-safe method.
 */
std::string AddLwblasError(Plugin *p, const char* callName, lwblasStatus_t lwbSt, unsigned int gpuId, size_t bytes=0,
                           bool isGpuSpecific=true);

/*
 * The purpose of these macros is to ensure that the logged message includes accurate line numbers and
 * file name corresponding to the location where the macro is called. If the error message was logged inside the
 * AddLwblasError method, the line number and file name would have been obslwred.
 */
#define LOG_LWBLAS_ERROR_FOR_PLUGIN(plugin, callName, lwbSt, gpuId, ...)                                    \
    {                                                                                                       \
        std::string pluginCommonLwblasError = AddLwblasError(plugin, callName, lwbSt, gpuId, ##__VA_ARGS__);\
        PRINT_ERROR("%s", "%s", pluginCommonLwblasError.c_str());                                           \
    }                                                                                                       \
    (void)0

// Only for use by the Plugin subclasses
#define LOG_LWBLAS_ERROR(callName, lwbSt, gpuId, ...)                                                       \
    {                                                                                                       \
        std::string pluginCommonLwblasError = AddLwblasError(this, callName, lwbSt, gpuId, ##__VA_ARGS__);  \
        PRINT_ERROR("%s", "%s", pluginCommonLwblasError.c_str());                                           \
    }                                                                                                       \
    (void)0

/********************************************************************/
/*
 * Sets the result for the GPU based on the value of 'passed'. Logs warnings from 'errorList' if the result is fail.
 * Sets 'allPassed' to false if the result is failed for this GPU.
 *
 * 'i' is the index of the GPU in 'gpuList'; the GPU ID is the value in the vector (i.e. gpuList[i] is GPU ID).
 * If 'dcgmCommError' is true, sets result for all GPUs starting at index 'i' to fail and adds a warning to
 * 'errorListAllGpus'.
 */
void CheckAndSetResult(Plugin* p, const std::vector<unsigned int> &gpuList, size_t i, bool passed,
                       const std::vector<DcgmError> &errorList, bool &allPassed, bool dcgmCommError);

#endif // _LWVS_LWVS_Plugin_common_H_
